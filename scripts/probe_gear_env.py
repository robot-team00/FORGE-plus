#!/usr/bin/env python3
"""Headless smoke probe for FrankaGearInsertEnv (task1 gear-on-shaft, no RTX).

Builds the scene (Franka + table + FORGE gear base + medium gear), runs the
scripted FORGE setup (drive to the entrance above the middle shaft) and then
N zero-action steps, printing the state every 20 steps:

  phase / setup counter, EE z, gear origin (env frame), distance to the seated
  goal, insertion force, whether the gear is still in the grip.

Pass criteria (printed at the end):
  1. scene builds, no PhysX errors, sensors bind
  2. after warmup the gear hangs in the grip (gear tracks the EE, does not fall)
  3. setup ends with the gear origin near (rack_x, rack_y) at ~0.43 z
     (bore mouth over the shaft tip) — the learned policy's starting state

    export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 \
           PYTHONPATH=/workspace/FORGE-plus_task3
    /workspace/.venv/bin/python scripts/probe_gear_env.py --num_envs 4 --steps 260
"""
from __future__ import annotations

import argparse
from isaacsim import SimulationApp  # noqa: E402

_app = SimulationApp({"headless": True})

import torch  # noqa: E402
from forge_plus.isaac_gear_env import FrankaGearInsertEnv, GearInsertEnvCfg  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=260)
    p.add_argument("--obj", type=int, default=1, help="object class (0=abs_gear, 1=steel_gear)")
    p.add_argument("--gripper", default="franka_panda",
                   help="franka_panda | robotiq_2f140")
    args = p.parse_args()

    cfg = GearInsertEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.forge_mode = True
    cfg.forge_obj_cls = args.obj
    cfg.gripper = args.gripper
    if args.gripper == "robotiq_2f140":
        # staged grasp needs the long setup window (the arrival fast-forward
        # ends it early on a clean approach); four-bar loop joints only
        # survive a RAW parse; staging state machine is global -> forge_no_term
        cfg.forge_setup_steps = 4000
        cfg.warmup_substeps = 100
        cfg.scene.replicate_physics = False
        cfg.forge_no_term = True
        cfg.episode_length_s = 45.0
    env = FrankaGearInsertEnv(cfg)
    print(f"[probe] env up: {env.num_envs} envs, F_max={env.f_max_n:.1f} N", flush=True)

    out = env.reset()
    obs = (out[0] if isinstance(out, tuple) else out)["policy"]
    zero = torch.zeros(env.num_envs, cfg.action_space, device=env.device)

    orig = env.scene.env_origins
    grip_lost_at = -1
    handoff = None  # state snapshot right after the scripted setup ends
    for step in range(args.steps):
        res = env.step(zero)
        obs = res[0]["policy"]
        ee = env._robot.data.body_pos_w[:, env._ee_idx] - orig
        gear = env._obj.data.root_pose_w[:, :3] - orig
        goal = env._forge_goal_w() - orig
        dist = (gear - goal).norm(dim=-1)
        # gear-in-grip check: gear origin should sit mug_grip_z (+pad offset) below the EE
        ee_gear = (ee - gear).norm(dim=-1)
        if grip_lost_at < 0 and step > 30 and float(gear[0, 2]) < 0.30:
            grip_lost_at = step
        if handoff is None and int(env._setup_ctr.max()) == 0 and step > 10:
            from isaaclab.utils.math import matrix_from_quat as _mfq
            gu = _mfq(env._obj.data.root_pose_w[:, 3:7])[:, 2, 2]
            handoff = {
                "step": step,
                "dxy": (gear[:, :2] - goal[:, :2]).norm(dim=-1).clone(),
                "z": gear[:, 2].clone(),
                "gearup": gu.clone(),
            }
        if step % 20 == 0 or step == args.steps - 1:
            from isaaclab.utils.math import matrix_from_quat
            Rh = matrix_from_quat(env._robot.data.body_quat_w[:, env._ee_idx])
            hand_down = -Rh[:, 2, 2]  # hand +z vs world -z: 1.0 = straight down
            Rg = matrix_from_quat(env._obj.data.root_pose_w[:, 3:7])
            gear_up = Rg[:, 2, 2]     # gear +z vs world +z: 1.0 = flat/level
            print(
                f"[{step:4d}] setup={int(env._setup_ctr[0])} warm={int(env._warmup[0])} "
                f"ee_z={ee[0, 2]:.3f} gear=({gear[0, 0]:.3f},{gear[0, 1]:.3f},{gear[0, 2]:.3f}) "
                f"dist={dist[0]:.4f} ee↔gear={ee_gear[0]:.3f} "
                f"handdn={hand_down[0]:.3f} gearup={gear_up[0]:.3f} "
                f"fing={float(env._robot.data.joint_pos[0,7]):.4f}/{float(env._robot.data.joint_pos[0,8]):.4f} "
                f"ftgt={float(env._robot.data.joint_pos_target[0,7]):.4f} "
                f"f_ins={float(env._cf_insert[0]):.2f}N fseat={env.extras.get('fseat', 0.0):.2f}",
                flush=True,
            )

    # Judge the HAND-OFF state (what the learned policy inherits), not the end
    # state — with zero actions the arm drifts after hand-off by design.
    print("\n[probe] FINAL:", flush=True)
    if handoff is None:
        print("  setup never completed — FAIL", flush=True)
        ok = False
    else:
        print(f"  hand-off at step {handoff['step']}")
        print(f"  gear z at hand-off (want 0.42-0.47): {[round(float(z),3) for z in handoff['z']]}")
        print(f"  gear xy offset from shaft (want <0.03 — the policy corrects the rest): "
              f"{[round(float(d),4) for d in handoff['dxy']]}")
        print(f"  gear level (want >0.98): {[round(float(u),3) for u in handoff['gearup']]}")
        print(f"  grip lost (gear fell below 0.30): "
              f"{'NO' if grip_lost_at < 0 else f'YES at step {grip_lost_at}'}")
        ok = (grip_lost_at < 0
              and bool((handoff['z'] > 0.42).all()) and bool((handoff['z'] < 0.47).all())
              and bool((handoff['dxy'] < 0.008).all())
              and bool((handoff['gearup'] > 0.98).all()))
    print(f"  PROBE {'PASS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
