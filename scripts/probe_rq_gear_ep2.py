#!/usr/bin/env python3
"""Dead-episode diagnosis: run 2.5 POLICY-driven episodes with RQ_TRACE.

Fine-tune run 1 showed every episode that ENDS SEATED+PRESSING kills the NEXT
episode's staging (gear parked at 1.169 all episode, arm limp, seat never
arms). Trace the ep-1 -> ep-2 transition to find which staging gate dies.

    HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 RQ_TRACE=1 \
    PYTHONPATH=/workspace/FORGE-plus_task3 /workspace/.venv/bin/python \
        scripts/probe_rq_gear_ep2.py
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
    p.add_argument("--ckpt", default="checkpoints/task1_gear_rq.pt.it300")
    p.add_argument("--steps", type=int, default=3000)
    args = p.parse_args()

    cfg = GearInsertEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.replicate_physics = False
    cfg.gripper = "robotiq_2f140"
    cfg.forge_mode = True
    cfg.forge_obj_cls = 1
    cfg.forge_no_term = True
    cfg.forge_setup_steps = 4000
    cfg.warmup_substeps = 100
    cfg.episode_length_s = 20.0
    env = FrankaGearInsertEnv(cfg)

    ck = torch.load(args.ckpt, map_location=env.device, weights_only=False)
    pc = ck["policy_cfg"]
    pcfg = pc if isinstance(pc, PolicyConfig) else PolicyConfig(**pc)
    policy = ForceConditionedPolicy(pcfg).to(env.device)
    policy.load_state_dict(ck["policy_state_dict"])
    policy.eval()

    out = env.reset()
    obs = (out[0] if isinstance(out, tuple) else out)["policy"]
    orig = env.scene.env_origins[0]
    r = env._robot
    for k in range(args.steps):
        with torch.no_grad():
            m, _ = policy(obs, env.f_cmd_norm())
        res = env.step(torch.clamp(m, -1, 1))
        obs = res[0]["policy"]
        tr = bool(res[3][0])
        if k % 50 == 0 or tr:
            g = (env._obj.data.root_pose_w[0, :3] - orig).tolist()
            ee = (r.data.body_pos_w[0, env._ee_idx] - orig).tolist()
            jp = r.data.joint_pos[0]
            print(f"[ep2 {k:4d}]{' TRUNC' if tr else ''} "
                  f"setup={int(env._setup_ctr[0])} pd={int(env._rq_predrive.max())} "
                  f"sc={int(env._rq_seatctr[0])} dm={int(env._rq_drive_mode)} "
                  f"gear=({g[0]:.3f},{g[1]:.3f},{g[2]:.3f}) "
                  f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
                  f"fj={float(jp[env._grip_ids[0]]):+.3f} "
                  f"succ={int(env._succeeded[0])} "
                  f"kp0={float(r.data.joint_stiffness[0, 0]):.1f} "
                  f"Fins={float(env._cf_insert[0]):.1f}", flush=True)
    print("EP2_PROBE_DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
