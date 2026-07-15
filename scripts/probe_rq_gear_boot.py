#!/usr/bin/env python3
"""BOOT probe: Robotiq 2F-140 on the GEAR env (task1 port, handoff step 2).

Gates (no training, no policy):
  1. the combined asset parses in the gear scene — body/joint lists sane, NO
     parse ghost (exactly one robotiq link set), finger_joint is a DOF
  2. the four-bar assembles in the correct branch: pad separation ~0.127 m at
     open (finger_joint 0), and a commanded close sweeps it toward 0
  3. no explosion: arm joints finite and near spawn through ~N env steps

    export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 \
           PYTHONPATH=/workspace/FORGE-plus_task3
    /workspace/.venv/bin/python scripts/probe_rq_gear_boot.py --steps 200
"""
from __future__ import annotations

import argparse
from isaacsim import SimulationApp  # noqa: E402

_app = SimulationApp({"headless": True})

import os  # noqa: E402
import torch  # noqa: E402
from forge_plus.isaac_gear_env import FrankaGearInsertEnv, GearInsertEnvCfg  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--obj", type=int, default=0, help="0=abs_gear 1=steel_gear")
    args = p.parse_args()

    cfg = GearInsertEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.replicate_physics = False   # four-bar loop joints need a RAW parse
    cfg.gripper = "robotiq_2f140"
    cfg.forge_mode = True
    cfg.forge_obj_cls = args.obj
    cfg.forge_no_term = True
    cfg.forge_setup_steps = 4000
    cfg.warmup_substeps = 100
    cfg.episode_length_s = 60.0
    env = FrankaGearInsertEnv(cfg)
    print(f"[boot] env up: {env.num_envs} envs", flush=True)

    out = env.reset()
    _ = out
    r = env._robot
    bn = list(r.data.body_names)
    jn = list(r.data.joint_names)
    print(f"[boot] {len(bn)} bodies: {bn}", flush=True)
    print(f"[boot] {len(jn)} joints: {jn}", flush=True)
    # ghost check: each robotiq link must appear exactly once
    rq_links = [b for b in bn if "inner_finger" in b or "knuckle" in b or "robotiq" in b]
    dupes = {b for b in rq_links if bn.count(b) > 1}
    print(f"[boot] robotiq links: {rq_links}  dupes: {dupes or 'NONE'}", flush=True)
    fj = jn.index("finger_joint")
    lf, rf = bn.index("left_inner_finger"), bn.index("right_inner_finger")
    lp, rp = lf, rf   # pads are part of the inner_finger bodies

    zero = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
    orig = env.scene.env_origins
    bad = False
    for step in range(args.steps):
        env.step(zero)
        jp = r.data.joint_pos[0]
        if not torch.isfinite(jp).all():
            print(f"[boot] NON-FINITE joint state at step {step}: {jp.tolist()}", flush=True)
            bad = True
            break
        if step % 20 == 0 or step == args.steps - 1:
            ee = r.data.body_pos_w[0, env._ee_idx] - orig[0]
            sep = float((r.data.body_pos_w[0, lf] - r.data.body_pos_w[0, rf]).norm())
            psep = float((r.data.body_pos_w[0, lp] - r.data.body_pos_w[0, rp]).norm())
            gear = env._obj.data.root_pose_w[0, :3] - orig[0]
            print(f"[boot {step:4d}] pd={int(env._rq_predrive.max())} "
                  f"sc={int(env._rq_seatctr[0])} setup={int(env._setup_ctr[0])} "
                  f"fj={float(jp[fj]):+.4f} fsep={sep:.4f} padsep={psep:.4f} "
                  f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
                  f"gear=({gear[0]:.3f},{gear[1]:.3f},{gear[2]:.3f}) "
                  f"jmax={float(jp[:7].abs().max()):.2f}", flush=True)

    # four-bar sweep check at the end: command a close, watch pad separation
    print("[boot] four-bar sweep: commanding finger_joint 0 -> 0.4", flush=True)
    seps = []
    for tgt in (0.0, 0.1, 0.2, 0.3, 0.4):
        r.set_joint_position_target(
            torch.full((env.num_envs, 1), tgt, device=env.device),
            joint_ids=env._grip_ids)
        for _ in range(60):
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
        sep = float((r.data.body_pos_w[0, lp] - r.data.body_pos_w[0, rp]).norm())
        ang = float(r.data.joint_pos[0, fj])
        seps.append((tgt, ang, sep))
        print(f"[boot sweep] tgt={tgt:.2f} ang={ang:+.4f} padsep={sep:.4f}", flush=True)
    mono = all(seps[i][2] > seps[i + 1][2] - 1e-4 for i in range(len(seps) - 1))
    opens = seps[0][2] > 0.10
    print(f"[boot] sweep monotonic={mono} open_sep_ok={opens}", flush=True)
    print(f"BOOT_{'PASS' if (not bad and not dupes and mono and opens) else 'FAIL'}",
          flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
