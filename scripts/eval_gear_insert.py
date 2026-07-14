#!/usr/bin/env python3
"""Headless deterministic eval of the LEARNED gear-insertion policy (no RTX render).

Loads checkpoints/task1_gear_insert_franka.pt, drives the gear env with the
policy's DETERMINISTIC action (mean), and reports the success / breakage rate
over many parallel episodes (task1: FORGE GearMesh, medium gear onto the
gear-base middle shaft).

    export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 \
           PYTHONPATH=/workspace/FORGE-plus_task3
    /workspace/.venv/bin/python scripts/eval_gear_insert.py --obj 1 --episodes 200
"""
from __future__ import annotations

import argparse
from isaacsim import SimulationApp  # noqa: E402

_app = SimulationApp({"headless": True})

import os  # noqa: E402
import torch  # noqa: E402
from forge_plus.isaac_gear_env import FrankaGearInsertEnv, GearInsertEnvCfg  # noqa: E402
from forge_plus.skills.policy_network import ForceConditionedPolicy, PolicyConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/task1_gear_insert_franka.pt")
    p.add_argument("--obj", type=int, default=1, help="object class (0=abs_gear, 1=steel_gear)")
    p.add_argument("--num_envs", type=int, default=128)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--budget", default="ours",
                   choices=["ours", "fixed_global", "no_ceiling", "oracle"],
                   help="budget-setter baseline (issue #26): ours=identity-only LLM; "
                        "fixed_global=60 N for every object; no_ceiling=120 N (the "
                        "controller hard cap, i.e. no per-object budget); "
                        "oracle=F_break-5 N (evaluator-side cheat)")
    p.add_argument("--gripper", default="franka_panda",
                   help="franka_panda | robotiq_2f140")
    p.add_argument("--release", action="store_true",
                   help="forge_release_mode gate: success = the LEARNED release "
                        "(act[7]>0) with the gear resting seated, upright, "
                        "settled and the hand retracted clear (8-dim ckpt)")
    p.add_argument("--table_pick", action="store_true",
                   help="rq_table_pick staging: the gear spawns RESTING on the "
                        "table and the staged grasp closes on the supported "
                        "part (no seat-window pin); drive-side return carry")
    args = p.parse_args()

    cfg = GearInsertEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.forge_mode = True
    cfg.forge_obj_cls = args.obj
    cfg.gripper = args.gripper
    if args.gripper == "robotiq_2f140":
        # staged grasp: long setup window (arrival fast-forward ends it early),
        # RAW parse for the four-bar, synchronized resets (global staging).
        cfg.forge_setup_steps = 4000
        cfg.warmup_substeps = 100
        cfg.scene.replicate_physics = False
        cfg.forge_no_term = True
        cfg.episode_length_s = 45.0
        if args.max_steps < 1500:
            args.max_steps = 1500   # staging ~800-1100 env steps + policy 240
        if args.table_pick:
            cfg.rq_table_pick = True
            cfg.episode_length_s = 120.0   # truncation must not cut the longer
            # table staging (45 s = 2700 steps; staging alone is ~2100)
            if args.max_steps < 3600:
                args.max_steps = 3600   # table staging (servo windows + return
                # traverse + settle hover) is ~2100 env steps before the policy;
                # search + seat + settle + release + retract-clear needs ~700
                # more (the 2600 cap timed out 64/64 mid-search at healthy 10 N)
    if args.release:
        cfg.forge_release_mode = True
        cfg.forge_hybrid_retract = True
        if args.gripper == "robotiq_2f140" and args.max_steps < 1700:
            args.max_steps = 1700   # + settle hold + release + retract-clear
    if args.budget == "fixed_global":
        cfg.budget_mode, cfg.budget_fixed_n = "fixed", 60.0
    elif args.budget == "no_ceiling":
        cfg.budget_mode, cfg.budget_fixed_n = "fixed", 120.0
    elif args.budget == "oracle":
        cfg.budget_mode = "oracle"
    env = FrankaGearInsertEnv(cfg)

    ck = torch.load(args.ckpt, map_location=env.device, weights_only=False)
    pc = ck["policy_cfg"]
    pcfg = pc if isinstance(pc, PolicyConfig) else PolicyConfig(**pc)
    policy = ForceConditionedPolicy(pcfg).to(env.device)
    policy.load_state_dict(ck["policy_state_dict"])
    policy.eval()
    print(f"[eval] loaded {args.ckpt}  obs={pcfg.obs_dim} act={pcfg.act_dim}  "
          f"F_max={env.f_max_n:.1f}N (obj {args.obj})", flush=True)

    out = env.reset()
    obs = (out[0] if isinstance(out, tuple) else out)["policy"]
    n_succ = n_brk = n_end = 0
    step = 0
    peak_f = torch.zeros(env.num_envs, device=env.device)   # per-episode peak Fins
    ep_peaks: list[float] = []          # collected at episode end
    over_budget_eps = 0                 # episodes whose F_max > F_break (dangerous budget)
    margins: list[float] = []           # F_break - F_max per ended episode
    # robotiq: forge_no_term (global staging needs synchronized resets), so the
    # EVAL owns the episode boundary — latch first-success/break per env (the
    # franka's terminate-on-success semantics, enforced script-side), then a
    # full synchronized env.reset() once every env is decided or the cap hits.
    rq = args.gripper == "robotiq_2f140"
    ep_succ = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    ep_brk = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    ep_bad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    n_badrel = 0
    ep_step = 0
    while n_end < args.episodes and step < args.max_steps * 40:
        with torch.no_grad():
            mean, _ = policy(obs, env.f_cmd_norm())
            act = mean.clamp(-1.0, 1.0)
        res = env.step(act)
        obs = res[0]["policy"]
        if rq:
            # only undecided envs accumulate peak force (franka records peaks
            # up to termination; post-seat pressing must not pollute the stats)
            und = ~(ep_succ | ep_brk)
            peak_f = torch.where(und, torch.maximum(peak_f, env._cf_insert), peak_f)
        else:
            peak_f = torch.maximum(peak_f, env._cf_insert)
        if rq:
            ep_succ |= (env._succeeded & ~ep_brk)
            ep_brk |= (env._broke & ~ep_succ)
            if args.release:
                ep_bad |= (env._bad_release & ~ep_succ & ~ep_brk)
            ep_step += 1
            if bool((ep_succ | ep_brk | ep_bad).all()) or ep_step >= args.max_steps:
                n_succ += int(ep_succ.sum().item())
                n_brk += int(ep_brk.sum().item())
                n_badrel += int(ep_bad.sum().item())
                n_end += env.num_envs
                ep_peaks.extend(peak_f.tolist())
                m = (env._f_break - env._budget_env)
                margins.extend(m.tolist())
                over_budget_eps += int((m < 0).sum().item())
                peak_f[:] = 0.0
                ep_succ[:] = False
                ep_bad[:] = False
                ep_brk[:] = False
                ep_step = 0
                out = env.reset()
                obs = (out[0] if isinstance(out, tuple) else out)["policy"]
        else:
            ended = res[2] | res[3]
            if bool(ended.any()):
                idx = ended.nonzero(as_tuple=True)[0]
                ep_peaks.extend(peak_f[idx].tolist())
                m = (env._f_break[idx] - env._budget_env[idx])
                margins.extend(m.tolist())
                over_budget_eps += int((m < 0).sum().item())
                peak_f[idx] = 0.0
            n_succ += int(res[4].get("n_succ", 0.0))
            n_brk += int(res[4].get("n_brk", 0.0))
            n_end += int(ended.sum().item())
        step += 1
        if step % 50 == 0:
            print(f"[eval] step {step}  ended={n_end}  succ={n_succ}  brk={n_brk}", flush=True)

    rate = n_succ / max(n_end, 1)
    brate = n_brk / max(n_end, 1)
    print(f"\n=== FORGE policy eval (budget={args.budget}) ===")
    print(f"episodes ended : {n_end}")
    print(f"SUCCESS rate   : {rate:.3f}  ({n_succ}/{n_end})")
    print(f"BREAK rate     : {brate:.3f}  ({n_brk}/{n_end})")
    if args.release:
        print(f"BAD RELEASE    : {n_badrel}/{n_end} (released and lost the gear)")
    if ep_peaks:
        pk = torch.tensor(ep_peaks)
        mg = torch.tensor(margins)
        print(f"peak Fins      : mean {pk.mean():.2f} N  p95 {pk.quantile(0.95):.2f} N  max {pk.max():.2f} N")
        print(f"budget margin  : mean {mg.mean():.1f} N (F_break - F_max); "
              f"over-budget eps: {over_budget_eps}/{len(margins)}")
    print(f"=========================", flush=True)
    # skip _app.close(): it hangs this pod's headless Kit after the summary
    # (the zero-shot run had to be killed by PID); hard exit frees the GPU
    os._exit(0)


if __name__ == "__main__":
    main()
