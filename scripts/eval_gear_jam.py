#!/usr/bin/env python3
"""Jam-recovery eval — recovery-policy baselines under induced wedge jams.

Stages every episode with a fixed lateral hand-off offset BEYOND the funnel
capture radius, so the bore mouth wedges on the shaft tip (the force-signature
jam). The TRAINED policy runs throughout; on a detected jam the chosen
recovery baseline picks a maneuver (issue #26):

  ours        : force-signature -> RecoverySelector (reads the signature)
  heuristic   : hand rule — lateral bias + rising axial -> rotate_align,
                else wiggle_search (never reads the full signature)
  vision_llm  : random menu choice (proxy for pixels-only methods that cannot
                see the force signature)
  press_harder: always retract_and_reapproach AND raise F_max x1.25 per
                attempt (cap 120 N) — the budget-escalation anti-pattern
  none        : no recovery (jams run to timeout; shows the budget modes'
                force behavior under sustained resistance)

Single env (the recovery interface acts on env 0), sequential episodes.

    export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 \
           PYTHONPATH=/workspace/FORGE-plus_task3
    /workspace/.venv/bin/python scripts/eval_gear_jam.py \
        --recovery ours --obj 0 --offset_mm 8 --episodes 25
"""
from __future__ import annotations

import argparse
import random
from isaacsim import SimulationApp  # noqa: E402

_app = SimulationApp({"headless": True})

import os  # noqa: E402
import torch  # noqa: E402
from forge_plus.isaac_gear_env import FrankaGearInsertEnv, GearInsertEnvCfg  # noqa: E402
from forge_plus.skills.policy_network import ForceConditionedPolicy, PolicyConfig  # noqa: E402
from forge_plus.llm.recovery_selector import RecoverySelector, RECOVERY_MENU  # noqa: E402
from forge_plus.llm.client import HeuristicLLMClient  # noqa: E402


def pick_recovery(mode: str, env, selector, attempt: int, rng) -> tuple[str, dict]:
    sig = env.failure_signature()
    if mode == "ours":
        resp = selector.select(sig, env.f_max_n, attempt, "insert", env.gripper)
        return resp.action, dict(resp.params or {}), sig
    if mode == "heuristic":
        act = "rotate_align" if (sig.lateral_bias != "none" and sig.axial_rising) \
            else "wiggle_search"
        return act, {}, sig
    if mode == "vision_llm":
        return rng.choice([m for m in RECOVERY_MENU if m != "abort"]), {}, sig
    if mode == "press_harder":
        # escalate the budget x1.25 (the defining sin) and re-approach
        new_b = min(float(env._budget_env[0]) * 1.25, 120.0)
        env._budget_env[0] = new_b
        env._f_cmd[0] = new_b
        return "retract_and_reapproach", {}, sig
    raise ValueError(mode)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/task1_gear_insert_franka.pt")
    p.add_argument("--recovery", default="ours",
                   choices=["ours", "heuristic", "vision_llm", "press_harder", "none"])
    p.add_argument("--budget", default="ours", choices=["ours", "no_ceiling"])
    p.add_argument("--obj", type=int, default=0, help="0=abs_gear (fragile), 1=steel_gear")
    p.add_argument("--offset_mm", type=float, default=8.0)
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--max_attempts", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=560, help="env steps per episode")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)

    cfg = GearInsertEnvCfg()
    cfg.scene.num_envs = 1
    cfg.forge_mode = True
    cfg.forge_obj_cls = args.obj
    cfg.forge_start_fixed_x = args.offset_mm / 1000.0
    if args.budget == "no_ceiling":
        cfg.budget_mode, cfg.budget_fixed_n = "fixed", 120.0
    env = FrankaGearInsertEnv(cfg)

    ck = torch.load(args.ckpt, map_location=env.device, weights_only=False)
    pc = ck["policy_cfg"]
    pcfg = pc if isinstance(pc, PolicyConfig) else PolicyConfig(**pc)
    policy = ForceConditionedPolicy(pcfg).to(env.device)
    policy.load_state_dict(ck["policy_state_dict"])
    policy.eval()
    selector = RecoverySelector(client=HeuristicLLMClient())

    print(f"[jam] recovery={args.recovery} budget={args.budget} obj={args.obj} "
          f"offset={args.offset_mm}mm episodes={args.episodes}", flush=True)

    n_succ = n_brk = n_timeout = 0
    attempts_used: list[int] = []
    peak_forces: list[float] = []
    sig_kinds: dict[str, int] = {}
    ep = 0
    out = env.reset()
    obs = (out[0] if isinstance(out, tuple) else out)["policy"]
    while ep < args.episodes:
        steps = attempts = 0
        peak = 0.0
        done = False
        while not done and steps < args.max_steps:
            with torch.no_grad():
                mean, _ = policy(obs, env.f_cmd_norm())
                act = mean.clamp(-1.0, 1.0)
            res = env.step(act)
            obs = res[0]["policy"]
            peak = max(peak, float(env._cf_insert[0]))
            if bool((res[2] | res[3])[0]):
                succ = bool(res[4].get("succ_mask", torch.zeros(1))[0]) \
                    if "succ_mask" in res[4] else res[4].get("n_succ", 0) > 0
                brk = bool(res[4].get("brk_mask", torch.zeros(1))[0]) \
                    if "brk_mask" in res[4] else res[4].get("n_brk", 0) > 0
                n_succ += int(succ)
                n_brk += int(brk)
                done = True
                break
            if args.recovery != "none" and attempts < args.max_attempts and env.is_failure():
                act_name, params, sig = pick_recovery(args.recovery, env, selector,
                                                      attempts + 1, rng)
                kind = ("wedge" if sig.lateral_bias != "none" else "friction")
                sig_kinds[kind] = sig_kinds.get(kind, 0) + 1
                print(f"[jam] ep{ep} attempt{attempts+1}: {kind} sig "
                      f"(peak {sig.peak_axial_N}N lat {sig.lateral_bias} "
                      f"rising {sig.axial_rising}) -> {act_name}"
                      + (f" F_max->{env.f_max_n:.0f}N" if args.recovery == "press_harder" else ""),
                      flush=True)
                env.apply_recovery(act_name, params)
                attempts += 1
            steps += 1
        if not done:
            n_timeout += 1
        attempts_used.append(attempts)
        peak_forces.append(peak)
        ep += 1
        if ep % 5 == 0:
            print(f"[jam] {ep}/{args.episodes}: succ {n_succ} brk {n_brk} "
                  f"timeout {n_timeout}", flush=True)

    pk = torch.tensor(peak_forces)
    print(f"\n=== JAM-RECOVERY EVAL (recovery={args.recovery}, budget={args.budget}, "
          f"obj={args.obj}, offset={args.offset_mm}mm) ===")
    print(f"episodes    : {ep}")
    print(f"SUCCESS     : {n_succ/max(ep,1):.3f} ({n_succ}/{ep})")
    print(f"BREAK       : {n_brk/max(ep,1):.3f} ({n_brk}/{ep})")
    print(f"TIMEOUT     : {n_timeout/max(ep,1):.3f} ({n_timeout}/{ep})")
    print(f"attempts    : mean {sum(attempts_used)/max(ep,1):.2f}")
    print(f"peak Fins   : mean {pk.mean():.2f} N  max {pk.max():.2f} N")
    print(f"signatures  : {sig_kinds}")
    print("==============================", flush=True)
    _app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
