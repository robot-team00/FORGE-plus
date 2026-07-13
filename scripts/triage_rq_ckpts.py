#!/usr/bin/env python3
"""Checkpoint triage: one deterministic episode block per snapshot, one boot.

Loads each checkpoint into the same env (32 envs, steel by default) and runs a
single synchronized episode block with the DETERMINISTIC mean, reporting seats,
breaks, peak insertion force, and final xy error. Cheap contact/seat signal to
pick which snapshot deserves the full clean gate.

    export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 \
           PYTHONPATH=/workspace/FORGE-plus_task3
    /workspace/.venv/bin/python scripts/triage_rq_ckpts.py --obj 1 \
        --ckpts checkpoints/task1_gear_rq_long.pt.it400 ...
"""
from __future__ import annotations

import argparse
from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})

import os  # noqa: E402
import torch  # noqa: E402
from forge_plus.isaac_gear_env import FrankaGearInsertEnv, GearInsertEnvCfg  # noqa: E402
from forge_plus.skills.policy_network import ForceConditionedPolicy, PolicyConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--obj", type=int, default=1)
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=1500)
    args = p.parse_args()

    cfg = GearInsertEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.forge_mode = True
    cfg.forge_obj_cls = args.obj
    cfg.gripper = "robotiq_2f140"
    cfg.forge_setup_steps = 4000
    cfg.warmup_substeps = 100
    cfg.scene.replicate_physics = False
    cfg.forge_no_term = True
    cfg.episode_length_s = 45.0
    env = FrankaGearInsertEnv(cfg)

    policy = None
    results = []
    for ck_path in args.ckpts:
        ck = torch.load(ck_path, map_location=env.device, weights_only=False)
        if policy is None:
            pc = ck["policy_cfg"]
            pcfg = pc if isinstance(pc, PolicyConfig) else PolicyConfig(**pc)
            policy = ForceConditionedPolicy(pcfg).to(env.device)
        policy.load_state_dict(ck["policy_state_dict"])
        policy.eval()

        out = env.reset()
        obs = (out[0] if isinstance(out, tuple) else out)["policy"]
        ep_succ = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        ep_brk = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        peak_f = torch.zeros(env.num_envs, device=env.device)
        for k in range(args.steps):
            with torch.no_grad():
                mean, _ = policy(obs, env.f_cmd_norm())
            res = env.step(mean.clamp(-1.0, 1.0))
            obs = res[0]["policy"]
            und = ~(ep_succ | ep_brk)
            peak_f = torch.where(und, torch.maximum(peak_f, env._cf_insert), peak_f)
            ep_succ |= (env._succeeded & ~ep_brk)
            ep_brk |= (env._broke & ~ep_succ)
            if bool((ep_succ | ep_brk).all()):
                break
        # final gear-base offset from the seat goal, per env (env-local frame)
        base = env._obj.data.root_pose_w[:, :3] - env.scene.env_origins
        goal = env._forge_goal_w() - env.scene.env_origins
        off = base - goal
        dxy = off[:, :2].norm(dim=-1)
        line = (f"[triage] {os.path.basename(ck_path):32s} obj={args.obj} "
                f"seat={int(ep_succ.sum())}/{env.num_envs} "
                f"brk={int(ep_brk.sum())} steps={k + 1} "
                f"peakF mean={peak_f.mean():.1f} p95={peak_f.quantile(0.95):.1f} "
                f"contact>1N={int((peak_f > 1.0).sum())}/{env.num_envs}\n"
                f"         off_x mean={off[:, 0].mean() * 1000:+.1f}mm std={off[:, 0].std() * 1000:.1f} "
                f"off_y mean={off[:, 1].mean() * 1000:+.1f}mm std={off[:, 1].std() * 1000:.1f} "
                f"off_z mean={off[:, 2].mean() * 1000:+.1f}mm | "
                f"dxy mean={dxy.mean() * 1000:.1f}mm p5={dxy.quantile(0.05) * 1000:.1f} "
                f"p95={dxy.quantile(0.95) * 1000:.1f}")
        print(line, flush=True)
        results.append(line)

    print("\n=== TRIAGE SUMMARY ===")
    for line in results:
        print(line)
    print("TRIAGE_DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
