#!/usr/bin/env python3
"""PLANT-CAPABILITY DIAGNOSTIC (dev probe — never an eval or render).

Question: can the robotiq 2F-140 gear plant seat AT ALL when the EE is aimed
perfectly, or does the mechanism itself fail (in-grip slip past the 19.4 N
pinch, shaft-tip shed, friction pinning)? Every rq checkpoint's deterministic
mean lands at the same dxy~16-20 mm / peakF~20 N signature regardless of
weights, which smells mechanical.

Case A: P-control the EE xy onto the bore axis while descending gently.
Case B: zero action after hand-off (pure plant drift).
Logs gear-vs-goal offset, gear-vs-EE offset (IN-GRIP SLIP), insertion force.

    /workspace/.venv/bin/python scripts/probe_rq_insert_plant.py --obj 0
"""
from __future__ import annotations

import argparse
from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})

import os  # noqa: E402
import torch  # noqa: E402
from forge_plus.isaac_gear_env import FrankaGearInsertEnv, GearInsertEnvCfg  # noqa: E402


def run_case(env, name: str, steps: int, kp: float, descend: float,
             force_gate: float = 0.0, dither: float = 0.0) -> None:
    import math
    out = env.reset()
    _ = (out[0] if isinstance(out, tuple) else out)["policy"]
    act = torch.zeros(env.num_envs, 7, device=env.device)
    grip0 = None   # gear-vs-EE offset snapshot at hand-off (grip reference)
    for k in range(steps):
        gear = env._obj.data.root_pose_w[:, :3]
        goal = env._forge_goal_w()
        ee = env._robot.data.body_pos_w[:, env._ee_idx]
        live = env._setup_ctr == 0
        if bool(live.any()) and grip0 is None:
            grip0 = (gear - ee).clone()
        err = goal[:, :2] - gear[:, :2]
        act.zero_()
        if kp > 0.0:
            act[:, :2] = (kp * err / env.cfg.forge_act_range).clamp(-1.0, 1.0)
            if force_gate > 0.0:
                # PROPORTIONAL force regulation around a gentle reference (the
                # behavior a policy must learn). The earlier bang-bang gate
                # (descend / +0.10 retreat) bounced 15 mm per contact, walked
                # the gear laterally and shed it out of the pads (probe 7).
                cf = env._cf_insert
                # gentle sustained-press regulation: ride the bore down at
                # ~4 N. The old gains bounced 10 mm per force burst and the
                # gear never descended past the mouth (probe 14).
                f_ref = 4.0
                act[:, 2] = (-0.02 * (f_ref - cf)).clamp(-0.12, 0.08)
                if dither > 0.0:
                    # tiny search wiggle only ABOVE the tip (never grind the
                    # 0.25 mm-clearance bore walls once entered)
                    gz_l = (env._obj.data.root_pose_w[:, 2]
                            - env.scene.env_origins[:, 2])
                    on = ((cf > 0.5) & (cf < force_gate) & (gz_l > 0.418)).float()
                    act[:, 0] += 0.05 * on * math.sin(2 * math.pi * k / 60.0)
                    act[:, 1] += 0.05 * on * math.cos(2 * math.pi * k / 60.0)
            else:
                act[:, 2] = descend
        act[:, :3] *= live.unsqueeze(-1).float()
        env.step(act.clamp(-1, 1))
        if k % 40 == 0 or k == steps - 1:
            dxy = (gear[:, :2] - goal[:, :2]).norm(dim=-1)
            gz = gear[:, 2] - env.scene.env_origins[:, 2]
            slip = (gear - ee) - (grip0 if grip0 is not None else (gear - ee))
            print(f"[{name}] k={k:4d} live={int(live.sum())}/{env.num_envs} "
                  f"dxy mm min={dxy.min() * 1000:.1f} med={dxy.median() * 1000:.1f} max={dxy.max() * 1000:.1f} "
                  f"gz={gz.median():.4f} cf med={env._cf_insert.median():.1f} max={env._cf_insert.max():.1f} "
                  f"gripslip mm xy={slip[:, :2].norm(dim=-1).median() * 1000:.1f} z={slip[:, 2].median() * 1000:+.1f} "
                  f"seat={int(env._succeeded.sum())} brk={int(env._broke.sum())}",
                  flush=True)
        if bool(env._succeeded.all()):
            print(f"[{name}] ALL SEATED at k={k}", flush=True)
            break
    print(f"[{name}] FINAL seat={int(env._succeeded.sum())}/{env.num_envs} "
          f"brk={int(env._broke.sum())}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--obj", type=int, default=0)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--kp", type=float, default=0.6, help="P gain on xy error (m->m target delta)")
    p.add_argument("--descend", type=float, default=-0.06, help="constant z action while aiming")
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

    run_case(env, "A:aim+fgate", args.steps, args.kp, args.descend,
             force_gate=5.0, dither=0.15)
    run_case(env, "B:zero-action", args.steps, 0.0, 0.0)
    print("PROBE_DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
