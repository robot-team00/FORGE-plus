#!/usr/bin/env python3
"""Collect SUCCESSFUL sampled rollouts for self-imitation (BC) on the gear task.

The rq lineage's sampled policy seats via noise-driven funnel search while its
deterministic mean presses ~25 mm off the bore (std-anneal drifted onto a
plateau — see logs/rq_gear_anneal.log). Fix: run the SAMPLED policy, keep only
episodes that reach the strict seat, and save (obs, f_cmd, executed action)
per step so the mean can be behavior-cloned onto the funnel-search behavior
(self-imitation — still the policy's own learned actions, nothing scripted).

    export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 \
           PYTHONPATH=/workspace/FORGE-plus_task3
    /workspace/.venv/bin/python scripts/collect_gear_bc.py \
        --ckpt checkpoints/task1_gear_rq_anneal.pt.it200 \
        --out /workspace/logs/gear_bc_data.npz --blocks 6
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
    p.add_argument("--blocks", type=int, default=6, help="synchronized episode blocks")
    p.add_argument("--steps", type=int, default=1500, help="step cap per block")
    p.add_argument("--settle_keep", type=int, default=30, help="post-seat steps to keep (teaches holding)")
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

    ck = torch.load(args.ckpt, map_location=env.device, weights_only=False)
    pc = ck["policy_cfg"]
    pcfg = pc if isinstance(pc, PolicyConfig) else PolicyConfig(**pc)
    policy = ForceConditionedPolicy(pcfg).to(env.device)
    policy.load_state_dict(ck["policy_state_dict"])
    policy.eval()
    std = policy.log_std.data.exp()
    print(f"[collect] {args.ckpt} std mean={std.mean().item():.3f}", flush=True)

    obs_buf, fcmd_buf, act_buf = [], [], []
    n_succ_total = n_ep_total = 0
    for blk in range(args.blocks):
        out = env.reset()
        obs = (out[0] if isinstance(out, tuple) else out)["policy"]
        N = env.num_envs
        ep_succ = torch.zeros(N, dtype=torch.bool, device=env.device)
        ep_brk = torch.zeros(N, dtype=torch.bool, device=env.device)
        seat_step = torch.full((N,), -1, dtype=torch.long, device=env.device)
        o_steps, f_steps, a_steps = [], [], []
        for k in range(args.steps):
            fcmd = env.f_cmd_norm()
            with torch.no_grad():
                mean, _ = policy(obs, fcmd)
                act = (mean + std * torch.randn_like(mean)).clamp(-1.0, 1.0)
            o_steps.append(obs.cpu())
            f_steps.append(fcmd.cpu())
            a_steps.append(act.cpu())
            res = env.step(act)
            obs = res[0]["policy"]
            new_succ = env._succeeded & ~ep_succ & ~ep_brk
            seat_step = torch.where(new_succ, torch.full_like(seat_step, k), seat_step)
            ep_succ |= new_succ
            ep_brk |= (env._broke & ~ep_succ)
            # keep rolling until everyone decided or all seats have settle_keep extra
            done_mask = ep_brk | (ep_succ & (k - seat_step >= args.settle_keep))
            if bool(done_mask.all()):
                break
        O = torch.stack(o_steps)   # (T, N, 34)
        F = torch.stack(f_steps)   # (T, N, 1)
        A = torch.stack(a_steps)   # (T, N, act)
        T = O.shape[0]
        for i in range(N):
            if not bool(ep_succ[i]):
                continue
            end = min(int(seat_step[i]) + args.settle_keep, T - 1)
            obs_buf.append(O[: end + 1, i])
            fcmd_buf.append(F[: end + 1, i])
            act_buf.append(A[: end + 1, i])
        n_succ_total += int(ep_succ.sum())
        n_ep_total += N
        print(f"[collect] block {blk}: succ {int(ep_succ.sum())}/{N} "
              f"brk {int(ep_brk.sum())} steps {k + 1} "
              f"(total {n_succ_total}/{n_ep_total})", flush=True)

    obs_all = torch.cat(obs_buf).numpy() if obs_buf else np.zeros((0, 34))
    fcmd_all = torch.cat(fcmd_buf).numpy() if fcmd_buf else np.zeros((0, 1))
    act_all = torch.cat(act_buf).numpy() if act_buf else np.zeros((0, 1))
    np.savez_compressed(args.out, obs=obs_all, fcmd=fcmd_all, act=act_all,
                        n_succ=n_succ_total, n_ep=n_ep_total)
    print(f"[collect] saved {obs_all.shape[0]} steps from {n_succ_total} "
          f"successful episodes -> {args.out}")
    print("COLLECT_DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
