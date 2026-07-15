#!/usr/bin/env python3
"""Grasp ground-truth probe (no rendering) — is the hub REALLY between the pads?

The closeup renders show open fingers hovering above the gear while the physics
buffers put the pad centre at the design grip offset. That offset alone cannot
distinguish "pads clamped on the hub" from "hand hovering at its trained carry
offset above a gear it is not holding". This prints the physics-level facts:
finger joint positions (closed-on-hub ~= 0.0168-0.0178 each), the finger<->object
contact-sensor force, the object<->shaft insertion force, and the poses.

    HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 \
    PYTHONPATH=/workspace/FORGE-plus_task3 /workspace/.venv/bin/python \
        scripts/probe_grasp.py [--fixed_x 0.0] [--steps 470]
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
    p.add_argument("--ckpt", default="checkpoints/task1_gear_sliprand.pt.it300")
    p.add_argument("--fixed_x", type=float, default=-1.0,
                   help=">=0 forces the start offset (eval_gear_jam uses 0.0); "
                        "-1 samples like the clean eval and the snapshot script")
    p.add_argument("--steps", type=int, default=470)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dump", default="",
                   help="write joint/gear states at the snapshot ks to this .npz "
                        "(for the PyBullet CPU close-up — the pod's RTX view draws "
                        "a displaced ghost of the gear and cannot be trusted)")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    cfg = GearInsertEnvCfg()
    cfg.scene.num_envs = 1
    cfg.forge_mode = True
    cfg.forge_obj_cls = 0
    cfg.forge_no_term = True          # match the snapshot run exactly
    if args.fixed_x >= 0.0:
        cfg.forge_start_fixed_x = args.fixed_x
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
    bn = list(env._robot.data.body_names)
    lf_i, rf_i = bn.index("panda_leftfinger"), bn.index("panda_rightfinger")
    hand_i = bn.index("panda_hand")

    print(f"[probe] ckpt={args.ckpt} fixed_x={args.fixed_x} grip_ids={env._grip_ids}",
          flush=True)
    dump_ks, dump_joints, dump_gear, dump_orig = [], [], [], []

    def _dual_pose(tag):
        # the isaaclab buffer vs the RAW PhysX tensor view — if these disagree,
        # the buffer is stale and every measurement built on it is fiction
        buf = env._obj.data.root_pose_w[0].tolist()
        try:
            raw = env._obj.root_physx_view.get_transforms()[0].tolist()
        except Exception as ex:
            raw = f"unavailable: {ex!r}"
        print(f"[dual] {tag} buffer pos=({buf[0]:.4f},{buf[1]:.4f},{buf[2]:.4f}) "
              f"quat(wxyz)=({buf[3]:.3f},{buf[4]:.3f},{buf[5]:.3f},{buf[6]:.3f})",
              flush=True)
        print(f"[dual] {tag} physx  raw={raw}", flush=True)
    for k in range(args.steps):
        with torch.no_grad():
            m, _ = policy(obs, env.f_cmd_norm())
        res = env.step(torch.clamp(m, -1, 1))
        obs = res[0]["policy"]
        if k in (170, 205):
            _dual_pose(f"k={k}")
        if args.dump and k in (170, 205, 220):
            dump_ks.append(k)
            dump_joints.append(env._robot.data.joint_pos[0].tolist())
            dump_gear.append(env._obj.data.root_pose_w[0].tolist())
            dump_orig.append(orig.tolist())
        if k % 20 == 0 or k == args.steps - 1:
            g = (env._obj.data.root_pose_w[0, :3] - orig).tolist()
            jp = env._robot.data.joint_pos[0]
            fj = [float(jp[i]) for i in env._grip_ids]
            lf = env._robot.data.body_pos_w[0, lf_i]
            rf = env._robot.data.body_pos_w[0, rf_i]
            pad_gap = float(torch.norm(lf - rf))          # finger-body separation
            # the env's own calibration: pad centre = hand flange - 0.103 (panda TCP,
            # _grasp_tcp_d). pad_dz vs gear should be +0.032 (mug_grip_z) for a
            # hub-centred clamp; ~+0.050 would mean a top-edge pinch (hub top +0.045)
            hand_z = float(env._robot.data.body_pos_w[0, hand_i, 2])
            pad_dz = (hand_z - 0.103) - float(env._obj.data.root_pose_w[0, 2])
            # gear tilt: angle of the gear's +z axis vs world up (0 deg = upright;
            # a tilted in-grip gear cannot thread a 0.25 mm-clearance bore, and a
            # tilted wedge at the shaft mouth must NOT count as a seat)
            qw, qx, qy, qz = env._obj.data.root_pose_w[0, 3:7].tolist()
            up_z = 1.0 - 2.0 * (qx * qx + qy * qy)   # z-component of rotated +z
            import math
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))
            grip_f = 0.0
            cs = getattr(env, "_contact_sensor", None)
            if cs is not None and getattr(cs.data, "net_forces_w", None) is not None:
                grip_f = float(cs.data.net_forces_w[0].norm(dim=-1).sum())
            print(f"[probe] k={k:3d} gear=({g[0]:.3f},{g[1]:.3f},{g[2]:.3f}) "
                  f"fingers=({fj[0]:.4f},{fj[1]:.4f}) body_gap={pad_gap:.4f} "
                  f"pad_dz={pad_dz:+.4f} tilt={tilt:5.1f}deg "
                  f"grip_F={grip_f:5.2f}N Fins={float(env._cf_insert[0]):5.2f}N "
                  f"setup={int(env._setup_ctr[0])} warm={int(env._warmup[0])}",
                  flush=True)
    if args.dump and dump_ks:
        import numpy as np
        np.savez(args.dump, ks=np.array(dump_ks), joints=np.array(dump_joints),
                 gear=np.array(dump_gear), orig=np.array(dump_orig))
        print(f"dumped {len(dump_ks)} states -> {args.dump}", flush=True)
    print("PROBE_DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
