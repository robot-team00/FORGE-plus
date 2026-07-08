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
    args = p.parse_args()

    cfg = GearInsertEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.forge_mode = True
    cfg.forge_obj_cls = args.obj
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
    while n_end < args.episodes and step < args.max_steps * 40:
        with torch.no_grad():
            mean, _ = policy(obs, env.f_cmd_norm())
            act = mean.clamp(-1.0, 1.0)
        res = env.step(act)
        obs = res[0]["policy"]
        peak_f = torch.maximum(peak_f, env._cf_insert)
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
    if ep_peaks:
        pk = torch.tensor(ep_peaks)
        mg = torch.tensor(margins)
        print(f"peak Fins      : mean {pk.mean():.2f} N  p95 {pk.quantile(0.95):.2f} N  max {pk.max():.2f} N")
        print(f"budget margin  : mean {mg.mean():.1f} N (F_break - F_max); "
              f"over-budget eps: {over_budget_eps}/{len(margins)}")
    print(f"=========================", flush=True)
    _app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
