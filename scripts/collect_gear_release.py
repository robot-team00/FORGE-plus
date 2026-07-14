#!/usr/bin/env python3
"""Collect release-labelled rollouts for the gear LEARNED-release head.

Runs the 7-dim uni policy MEAN (the insertion skill, untouched) with the env
in forge_release_mode, and a scripted EXPERT schedule on the 8th (release)
dim: -1 until the gear has been geometrically seated and settled for
--settle_steps consecutive steps, then +1 for the rest of the episode (the
env latches the release, the 2F-140 drive opens, hybrid retract clears the
hand). Scripted-expert supervision is TRAINING data only — the deployed
policy is the learned head; nothing scripted runs in evals/renders.

Saves (obs, fcmd, act8, setup) per step for episodes that reach the env's
release-mode success (released + resting on the shaft + upright + settled +
hand clear). The BC step then fits ONLY the release row of the mean head
(trunk + arm dims frozen), so the insertion behaviour stays bit-identical
to the uni checkpoint.

    export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 \
           PYTHONPATH=/workspace/FORGE-plus_task3
    /workspace/.venv/bin/python scripts/collect_gear_release.py \
        --ckpt checkpoints/task1_gear_rq_uni.pt \
        --out /workspace/logs/gear_release_data.npz --blocks 4
"""
from __future__ import annotations

import argparse
from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})

import os  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from forge_plus.isaac_gear_env import FrankaGearInsertEnv, GearInsertEnvCfg  # noqa: E402
from forge_plus.skills.policy_network import ForceConditionedPolicy, PolicyConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--obj", type=int, default=-1, help="-1 = mixed (both classes)")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4, help="synchronized episode blocks")
    p.add_argument("--steps", type=int, default=1200, help="step cap per block")
    p.add_argument("--settle_steps", type=int, default=20,
                   help="consecutive geometric-seat steps before the expert releases")
    p.add_argument("--post_keep", type=int, default=25,
                   help="steps kept after the env success latch (teaches staying open)")
    p.add_argument("--table_pick", action="store_true",
                   help="collect under rq_table_pick staging — the release head "
                        "reads raw obs (incl. the phase onehot), and the table "
                        "flow leaves those in a DIFFERENT state than the pinned "
                        "flow: a head trained on pinned data never fires in "
                        "table mode (smoke 3: 64/64 pressed at budget, 0 rel)")
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
    cfg.forge_release_mode = True
    cfg.forge_hybrid_retract = True
    if args.table_pick:
        cfg.rq_table_pick = True
        cfg.episode_length_s = 120.0
        args.steps = max(args.steps, 3400)   # table staging ~2100 + insert
    env = FrankaGearInsertEnv(cfg)

    ck = torch.load(args.ckpt, map_location=env.device, weights_only=False)
    pc = ck["policy_cfg"]
    pcfg = pc if isinstance(pc, PolicyConfig) else PolicyConfig(**pc)
    assert pcfg.act_dim == 7, "collector drives the 7-dim insertion policy"
    policy = ForceConditionedPolicy(pcfg).to(env.device)
    policy.load_state_dict(ck["policy_state_dict"])
    policy.eval()

    N = env.num_envs
    dev = env.device
    goal_xy = torch.tensor([cfg.rack_x, cfg.rack_y], device=dev)

    obs_buf, fcmd_buf, act_buf, setup_buf = [], [], [], []
    n_succ_total = n_ep_total = 0
    for blk in range(args.blocks):
        out = env.reset()
        obs = (out[0] if isinstance(out, tuple) else out)["policy"]
        ep_succ = torch.zeros(N, dtype=torch.bool, device=dev)
        ep_bad = torch.zeros(N, dtype=torch.bool, device=dev)
        released = torch.zeros(N, dtype=torch.bool, device=dev)
        settle = torch.zeros(N, dtype=torch.long, device=dev)
        succ_step = torch.full((N,), -1, dtype=torch.long, device=dev)
        o_steps, f_steps, a_steps, s_steps = [], [], [], []
        for k in range(args.steps):
            fcmd = env.f_cmd_norm()
            with torch.no_grad():
                mean, _ = policy(obs, fcmd)
                arm = mean.clamp(-1.0, 1.0)
            # scripted expert release schedule (TRAINING SUPERVISION ONLY):
            # geometric seat = gear origin on the shaft axis at seat depth
            base = env._obj.data.root_pose_w[:, :3] - env.scene.env_origins
            seated_geo = ((base[:, :2] - goal_xy).norm(dim=-1) < 0.003) & \
                ((base[:, 2] - cfg.cell_floor_z).abs() < cfg.insert_depth_tol) & \
                (env._setup_ctr == 0)
            settle = torch.where(seated_geo, settle + 1,
                                 torch.zeros_like(settle))
            released |= settle >= args.settle_steps
            rel = torch.where(released, torch.ones(N, device=dev),
                              -torch.ones(N, device=dev))
            act = torch.cat([arm, rel.unsqueeze(-1)], dim=-1)
            o_steps.append(obs.cpu())
            f_steps.append(fcmd.cpu())
            a_steps.append(act.cpu())
            s_steps.append((env._setup_ctr > 0).cpu())
            res = env.step(act)
            obs = res[0]["policy"]
            new_succ = env._succeeded & ~ep_succ & ~ep_bad
            succ_step = torch.where(new_succ, torch.full_like(succ_step, k), succ_step)
            ep_succ |= new_succ
            ep_bad |= (env._broke | env._bad_release) & ~ep_succ
            done_mask = ep_bad | (ep_succ & (k - succ_step >= args.post_keep))
            if k % 100 == 0:
                print(f"[collect] blk{blk} k{k}: rel {int(released.sum())} "
                      f"succ {int(ep_succ.sum())} bad {int(ep_bad.sum())}", flush=True)
            if bool(done_mask.all()):
                break
        O = torch.stack(o_steps)
        F = torch.stack(f_steps)
        A = torch.stack(a_steps)
        SU = torch.stack(s_steps)
        T = O.shape[0]
        for i in range(N):
            if not bool(ep_succ[i]):
                continue
            end = min(int(succ_step[i]) + args.post_keep, T - 1)
            obs_buf.append(O[: end + 1, i])
            fcmd_buf.append(F[: end + 1, i])
            act_buf.append(A[: end + 1, i])
            setup_buf.append(SU[: end + 1, i])
        n_succ_total += int(ep_succ.sum())
        n_ep_total += N
        print(f"[collect] block {blk}: succ {int(ep_succ.sum())}/{N} "
              f"bad {int(ep_bad.sum())} steps {k + 1} "
              f"(total {n_succ_total}/{n_ep_total})", flush=True)

    obs_all = torch.cat(obs_buf).numpy() if obs_buf else np.zeros((0, 34))
    fcmd_all = torch.cat(fcmd_buf).numpy() if fcmd_buf else np.zeros((0, 1))
    act_all = torch.cat(act_buf).numpy() if act_buf else np.zeros((0, 8))
    setup_all = torch.cat(setup_buf).numpy() if setup_buf else np.zeros((0,), bool)
    np.savez_compressed(args.out, obs=obs_all, fcmd=fcmd_all, act=act_all,
                        setup=setup_all, n_succ=n_succ_total, n_ep=n_ep_total)
    n_open = int((act_all[:, 7] > 0).sum()) if act_all.shape[0] else 0
    print(f"[collect] saved {obs_all.shape[0]} steps ({n_open} release-phase) "
          f"from {n_succ_total} successful episodes -> {args.out}")
    print("COLLECT_DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
