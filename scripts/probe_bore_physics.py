#!/usr/bin/env python3
"""Bore-physics drop test: does the SDF bore admit the shaft AT ALL?

Bypasses control precision entirely: teleports the FREE gear (fingers held
open) directly above the middle shaft at small offsets and lets gravity do
the insertion. If even the 0-offset drop cannot seat (gear rests at tip
height), the collision representation forbids insertion and no amount of
training can succeed — fix SDF fidelity first.

    /workspace/.venv/bin/python scripts/probe_bore_physics.py
"""
from __future__ import annotations
from isaacsim import SimulationApp  # noqa: E402
_app = SimulationApp({"headless": True})

import os  # noqa: E402
import torch  # noqa: E402
from forge_plus.isaac_gear_env import FrankaGearInsertEnv, GearInsertEnvCfg  # noqa: E402


def main() -> None:
    cfg = GearInsertEnvCfg()
    cfg.scene.num_envs = 1
    cfg.forge_mode = True
    cfg.forge_obj_cls = 0
    env = FrankaGearInsertEnv(cfg)
    env.reset()
    zero = torch.zeros(1, cfg.action_space, device=env.device)
    orig = env.scene.env_origins[0]
    shaft = torch.tensor([cfg.rack_x, cfg.rack_y], device=env.device) + orig[:2]

    for off_mm in (0.0, 0.5, 1.0, 2.0, 3.0):
        # park the arm's fingers OPEN and drop the gear from just above the tip
        pose = env._obj.data.root_pose_w.clone()
        pose[0, 0] = shaft[0] + off_mm / 1000.0
        pose[0, 1] = shaft[1]
        pose[0, 2] = orig[2] + 0.435          # bottom face 15 mm above the tip
        pose[0, 3] = 1.0
        pose[0, 4:7] = 0.0
        env._obj.write_root_pose_to_sim(pose)
        vel = env._obj.data.root_vel_w.clone(); vel[0] = 0.0
        env._obj.write_root_velocity_to_sim(vel)
        zs = []
        for step in range(120):
            # hold fingers open so the gripper cannot re-grab
            jp = env._robot.data.joint_pos.clone(); jp[0, 7:9] = 0.04
            jv = env._robot.data.joint_vel.clone(); jv[0, 7:9] = 0.0
            env._robot.write_joint_state_to_sim(jp, jv)
            env.step(zero)
            g = env._obj.data.root_pose_w[0, :3] - orig
            zs.append(float(g[2]))
        g = env._obj.data.root_pose_w[0, :3] - orig
        dxy = float(torch.hypot(g[0] - (shaft[0] - orig[0]), g[1] - (shaft[1] - orig[1])))
        seated = (abs(float(g[2]) - 0.400) < 0.006) and dxy < 0.005
        print(f"[bore] drop off={off_mm:3.1f}mm -> final z={float(g[2]):.4f} "
              f"dxy={dxy*1000:5.1f}mm minz={min(zs):.4f} "
              f"{'SEATED' if seated else 'BLOCKED' if float(g[2]) > 0.410 else 'partial/other'}",
              flush=True)
    _app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
