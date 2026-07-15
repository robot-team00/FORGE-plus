#!/usr/bin/env python3
"""Hold-trace diagnostic: zero action after hand-off, dump the control chain.

Answers ONE question: during the post-hand-off walk, does the COMMAND move
(anchor / realized OSC target — a clamp or freeze is rewriting it) or does the
ARM move while the command stays put (impedance/disturbance)? Prints env-0
vectors every 20 steps.
"""
from __future__ import annotations

import argparse
from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})

import os  # noqa: E402
import torch  # noqa: E402
from forge_plus.isaac_gear_env import FrankaGearInsertEnv, GearInsertEnvCfg  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--obj", type=int, default=0)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--act_z", type=float, default=0.0,
                   help="constant z action after hand-off (descend probe)")
    args = p.parse_args()

    cfg = GearInsertEnvCfg()
    cfg.scene.num_envs = 2
    cfg.forge_mode = True
    cfg.forge_obj_cls = args.obj
    cfg.gripper = "robotiq_2f140"
    cfg.forge_setup_steps = 4000
    cfg.warmup_substeps = 100
    cfg.scene.replicate_physics = False
    cfg.forge_no_term = True
    cfg.episode_length_s = 45.0
    env = FrankaGearInsertEnv(cfg)

    env.reset()
    act = torch.zeros(env.num_envs, 7, device=env.device)
    o = env.scene.env_origins[0]
    jp0 = None
    for k in range(args.steps):
        act[:, 2] = args.act_z * float((env._setup_ctr == 0)[0])
        env.step(act)
        live = int((env._setup_ctr[0] == 0).item())
        if live and jp0 is None:
            jp0 = env._robot.data.joint_pos[0, env._arm_ids].clone()
        if k % 20 == 0:
            ee = env._robot.data.body_pos_w[0, env._ee_idx] - o
            gear = env._obj.data.root_pose_w[0, :3] - o
            goal = env._forge_goal_w()[0] - o
            anch = (env._pol_anchor[0] - o) if bool(env._pol_anchor_set[0]) else None
            lct = getattr(env, "_last_cmd_target", None)
            tgt = (lct[0] - o) if lct is not None else None
            q = env._robot.data.body_quat_w[0, env._ee_idx]
            qd = env._ee_quat_des[0]
            qe = torch.tensor([  # q_err = qd * conj(q): axis of the command-vs-body error
                qd[0]*q[0] + (qd[1:]*q[1:]).sum(),
                -qd[0]*q[1] + q[0]*qd[1] - (qd[2]*q[3] - qd[3]*q[2]),
                -qd[0]*q[2] + q[0]*qd[2] - (qd[3]*q[1] - qd[1]*q[3]),
                -qd[0]*q[3] + q[0]*qd[3] - (qd[1]*q[2] - qd[2]*q[1])])
            oerr = float(2.0 * torch.acos((q * qd).sum().abs().clamp(max=1.0)))
            djp = ((env._robot.data.joint_pos[0, env._arm_ids] - jp0)
                   if jp0 is not None else torch.zeros(7))
            def f(v):
                return "None" if v is None else f"({v[0]:+.4f},{v[1]:+.4f},{v[2]:+.4f})"
            rimz = float(env._rq_rimz[0]) if hasattr(env, "_rq_rimz") else -1.0
            cfi = float(env._cf_insert[0])
            cff = float(env._cf_filt[0])
            print(f"[hold] k={k:4d} live={live} ee={f(ee)} tgt={f(tgt)} "
                  f"gear_dxy={float((gear[:2]-goal[:2]).norm())*1000:.1f}mm "
                  f"gz={float(gear[2]):.4f} oerr={oerr:.4f} "
                  f"ax=({qe[1]:+.3f},{qe[2]:+.3f},{qe[3]:+.3f}) "
                  f"ag={float(env._rq_ag[0]):+.3f} rimz={rimz:.4f} "
                  f"cfi={cfi:.1f} cff={cff:.1f} "
                  f"djp={[round(float(x), 4) for x in djp]}", flush=True)
    print("HOLD_TRACE_DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
