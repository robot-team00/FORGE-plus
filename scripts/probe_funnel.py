#!/usr/bin/env python3
"""Funnel-capture probe: seat rate + force signature vs forced start offset.

Runs the TRAINED policy (deterministic mean) with the hand-off forced to a
fixed lateral offset from the shaft, sweeping offsets. Reports, per offset:
seat rate, mean peak insertion force, mean lateral-force fraction, and mean
net descent — the data that (a) measures the true funnel capture radius and
(b) picks the wedge-jam vs friction-jam staging offsets for the recovery eval.

    export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 \
           PYTHONPATH=/workspace/FORGE-plus_task3
    /workspace/.venv/bin/python scripts/probe_funnel.py --offsets 0,1,2,3,4,6,8
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
    p.add_argument("--offsets", default="0,1,2,3,4,6,8", help="mm, comma-separated")
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=420, help="env steps per offset run")
    p.add_argument("--obj", type=int, default=1)
    args = p.parse_args()
    offsets_mm = [float(x) for x in args.offsets.split(",")]

    # One env per offset run would rebuild the scene each time (4 min startup);
    # instead build ONCE and change cfg.forge_start_fixed_x between runs — the
    # value is read at _reset_idx time, so a full reset re-stages every env.
    cfg = GearInsertEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.forge_mode = True
    cfg.forge_obj_cls = args.obj
    cfg.forge_start_fixed_x = 0.0
    env = FrankaGearInsertEnv(cfg)

    ck = torch.load(args.ckpt, map_location=env.device, weights_only=False)
    pc = ck["policy_cfg"]
    pcfg = pc if isinstance(pc, PolicyConfig) else PolicyConfig(**pc)
    policy = ForceConditionedPolicy(pcfg).to(env.device)
    policy.load_state_dict(ck["policy_state_dict"])
    policy.eval()

    print(f"[funnel] policy loaded; sweeping offsets {offsets_mm} mm "
          f"x {args.num_envs} envs x {args.steps} steps", flush=True)

    results = []
    for off_mm in offsets_mm:
        env.cfg.forge_start_fixed_x = off_mm / 1000.0
        env.cfg.forge_start_fixed_y = 0.0
        out = env.reset()
        obs = (out[0] if isinstance(out, tuple) else out)["policy"]
        n_succ = n_end = n_brk = 0
        peak_f = torch.zeros(env.num_envs, device=env.device)
        lat_frac_sum = torch.zeros(env.num_envs, device=env.device)
        lat_n = torch.zeros(env.num_envs, device=env.device)
        for _ in range(args.steps):
            with torch.no_grad():
                mean, _std = policy(obs, env.f_cmd_norm())
                act = mean.clamp(-1.0, 1.0)
            res = env.step(act)
            obs = res[0]["policy"]
            peak_f = torch.maximum(peak_f, env._cf_insert)
            # lateral fraction of the contact force while in contact (>1 N)
            cv = env._raw_contact_vec3()
            cn = cv.norm(dim=-1)
            in_c = cn > 1.0
            lf = torch.where(in_c, cv[:, :2].norm(dim=-1) / cn.clamp(min=1e-6),
                             torch.zeros_like(cn))
            lat_frac_sum += lf
            lat_n += in_c.float()
            ended = res[2] | res[3]
            n_succ += int(res[4].get("n_succ", 0.0))
            n_brk += int(res[4].get("n_brk", 0.0))
            n_end += int(ended.sum().item())
            if bool(ended.any()):
                peak_f[ended.nonzero(as_tuple=True)[0]] = 0.0
        gear = env._obj.data.root_pose_w[:, :3] - env.scene.env_origins
        seat_rate = n_succ / max(n_end, 1)
        lf_mean = float((lat_frac_sum / lat_n.clamp(min=1)).mean())
        row = dict(off=off_mm, ended=n_end, succ=n_succ, brk=n_brk,
                   rate=seat_rate, pk=float(peak_f.mean()),
                   latfrac=lf_mean, gz=float(gear[:, 2].mean()))
        results.append(row)
        print(f"[funnel] off={off_mm:4.1f}mm  ended={n_end:3d} seat={seat_rate:.2f} "
              f"brk={n_brk} peakF={row['pk']:5.2f}N latfrac={lf_mean:.2f} "
              f"gear_z={row['gz']:.3f}", flush=True)

    print("\n[funnel] SUMMARY (offset mm -> seat rate | lateral-force fraction):")
    for r in results:
        print(f"  {r['off']:4.1f} mm : seat {r['rate']:.2f}  latfrac {r['latfrac']:.2f} "
              f" peakF {r['pk']:.2f} N  brk {r['brk']}", flush=True)
    _app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
