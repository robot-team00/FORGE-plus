#!/usr/bin/env python3
"""train_pick_place.py — Vectorized PPO trainer for FORGE-plus Task 3 pick-and-place.

Trains a ForceConditionedPolicy (FiLM MLP) to pick fragile objects from a low
table and place them onto an overhead rack, subject to per-object force budgets
set by an offline LLM supervisor (FORGE framework).

Architecture
------------
* ForceConditionedPolicy: FiLM-modulated MLP (obs_dim=34, act_dim=7, hidden=256)
* ValueNetwork:           same conditioning, outputs scalar V(s, F_cmd)
* Algorithm:              PPO with GAE (clipped surrogate + value loss + entropy)

Usage
-----
    DISPLAY=:99 HOME=/workspace/persist/ovhome MPLBACKEND=Agg \\
    /workspace/.venv/bin/python scripts/train_pick_place.py \\
        --num-envs 512 --iterations 600 --gripper franka_panda

Logs to W&B project "forge-plus-task3" (key auto-read from /workspace/.jr_notes).
Checkpoint: checkpoints/task3_pick_place_franka.pt
"""

from __future__ import annotations
import argparse

# Isaac Sim must be initialised BEFORE any omniverse imports
from isaacsim import SimulationApp
_app = SimulationApp({"headless": True})  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, "/workspace/FORGE-plus_task3")  # make forge_plus discoverable under Isaac Sim Python  # noqa: E402

import os
import re
import time
import datetime as _dt

import torch
import torch.nn.functional as F
from torch.optim import Adam

from forge_plus.isaac_pick_place_env import (  # noqa: E402
    FrankaPickPlaceEnv,
    PickPlaceEnvCfg,
)
from forge_plus.skills.policy_network import (  # noqa: E402
    PolicyConfig,
    ForceConditionedPolicy,
    ValueNetwork,
)


def main() -> None:
    p = argparse.ArgumentParser(description="FORGE-plus Task 3: pick-and-place PPO")
    p.add_argument("--num_envs",   type=int,   default=512,
                   help="number of parallel Isaac envs")
    p.add_argument("--gripper",                default="franka_panda",
                   help="franka_panda | robotiq_2f140")
    p.add_argument("--iterations", type=int,   default=600,
                   help="total PPO update iterations")
    p.add_argument("--rollout",    type=int,   default=32,
                   help="env steps per rollout per iteration")
    p.add_argument("--epochs",     type=int,   default=5,
                   help="PPO mini-batch epochs per update")
    p.add_argument("--minibatch",  type=int,   default=8192,
                   help="mini-batch size (flat across envs×steps)")
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--lam",        type=float, default=0.95,
                   help="GAE lambda")
    p.add_argument("--clip",       type=float, default=0.2,
                   help="PPO clipping epsilon")
    p.add_argument("--device",                 default="cuda")
    p.add_argument("--ckpt",                   default="checkpoints/task3_pick_place_franka.pt",
                   help="path to save checkpoint")
    p.add_argument("--resume",                 default=None,
                   help="resume from checkpoint file")
    p.add_argument("--run",                    default=None,
                   help="W&B run name; defaults to today's date")
    p.add_argument("--no_upright", action="store_true",
                   help="extrinsic curriculum stage A: place on the shelf WITHOUT the upright requirement")
    p.add_argument("--reset_std", type=float, default=None,
                   help="after --resume, reset policy log_std to this (re-inflate exploration for stage B)")
    p.add_argument("--forge", action="store_true",
                   help="FORGE-style LEARNED insertion: policy drives the EE, no scripted waypoints/base-aim (obs=34)")
    p.add_argument("--forge_obj", type=int, default=None,
                   help="fix the forge training object class (0=glass bottle/fragile, 2=metal/robust)")
    args = p.parse_args()
    dev  = torch.device(args.device)

    # ── W&B logging (optional — graceful fallback) ────────────────────────
    USE_WANDB = False
    try:
        import wandb
        txt = open("/workspace/.jr_notes").read()
        m   = re.search(r"^WANDB_API_KEY=(\S+)", txt, re.M)
        key = m.group(1) if m else None
        if key:
            os.environ["WANDB_API_KEY"] = key
            run_name = args.run or (_dt.date.today().isoformat() + "_pick_place")
            wandb.init(
                project="forge-plus-task3",
                name=f"pick_place_{args.gripper}_{run_name}",
                config=vars(args),
            )
            USE_WANDB = True
            print("wandb logging ON", flush=True)
    except Exception as exc:
        print(f"wandb off: {str(exc)[:80]}", flush=True)

    # ── Checkpoint path ───────────────────────────────────────────────────
    ckpt = args.ckpt
    os.makedirs(os.path.dirname(os.path.abspath(ckpt)), exist_ok=True)
    print(f"[train] checkpoint: {ckpt}", flush=True)

    # ── Environment ───────────────────────────────────────────────────────
    cfg = PickPlaceEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.gripper        = args.gripper
    if args.no_upright:
        cfg.require_upright = False   # curriculum stage A
    obs_dim = 37
    if args.forge:
        cfg.forge_mode = True         # LEARNED insertion: policy drives the EE (no waypoints)
        # Use the NATURAL (forward) grasp orientation — a top-down wrist cannot reach
        # out to the cell in y (the arm saturates ~12 cm short). The natural grip
        # reaches the cell (as the scripted insertion proved); the base-aim SETUP
        # corrects the resulting lean so the base still starts centered.
        cfg.grasp_topdown = False
        if args.forge_obj is not None:
            cfg.forge_obj_cls = args.forge_obj   # 0=fragile glass bottle, 2=robust
        obs_dim = 34
    print(f"[train] forge_mode={cfg.forge_mode} place_strategy={cfg.place_strategy} obs={obs_dim}", flush=True)
    env = FrankaPickPlaceEnv(cfg)
    N   = env.num_envs
    print(f"[train] envs={N}  device={dev}  obs={obs_dim}  act=7", flush=True)

    # ── Networks ──────────────────────────────────────────────────────────
    pcfg   = PolicyConfig(obs_dim=obs_dim, act_dim=7)
    policy = ForceConditionedPolicy(pcfg).to(dev)
    value  = ValueNetwork(pcfg).to(dev)
    aopt   = Adam(policy.parameters(), lr=args.lr)
    copt   = Adam(value.parameters(), lr=args.lr)

    start_iter = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt_data = torch.load(args.resume, map_location=dev, weights_only=False)
        policy.load_state_dict(ckpt_data["policy_state_dict"])
        print(f"[train] resumed from {args.resume}", flush=True)
        if args.reset_std is not None and hasattr(policy, "log_std"):
            policy.log_std.data.fill_(float(args.reset_std))
            print(f"[train] reset log_std -> {args.reset_std} (re-inflate exploration)", flush=True)

    # ── Rollout + PPO loop ────────────────────────────────────────────────
    out = env.reset()
    obs = (out[0] if isinstance(out, tuple) else out)["policy"].to(dev)
    t0  = time.perf_counter()

    for it in range(start_iter, args.iterations):
        # Buffers for one rollout
        O, Fc, A, LP, R, D, V = [], [], [], [], [], [], []
        ep_succ = 0.0
        ep_brk  = 0.0
        ep_end  = 0.0

        for _ in range(args.rollout):
            fcmd = env.f_cmd_norm().to(dev)   # (N, 1) normalised force budget
            with torch.no_grad():
                mean, std = policy(obs, fcmd)
                dist = torch.distributions.Normal(mean, std)
                act  = dist.sample()
                lp   = dist.log_prob(act).sum(-1)
                val  = value(obs, fcmd).squeeze(-1)

            res = env.step(act)
            O.append(obs); Fc.append(fcmd); A.append(act); LP.append(lp)
            R.append(res[1]); D.append((res[2] | res[3]).float()); V.append(val)
            ep_succ += res[4].get("n_succ", 0.0)
            ep_brk  += res[4].get("n_brk",  0.0)
            ep_end  += float(res[2].sum().item())
            diag_fdist = res[4].get("fdist", -1.0)   # last-step diagnostics (forge)
            diag_fbz   = res[4].get("fbz", -1.0)
            diag_fseat = res[4].get("fseat", -1.0)
            diag_fmin  = res[4].get("fmin", -1.0)    # best dist achieved this episode
            diag_fdxy  = res[4].get("fdxy", -1.0)    # xy offset from cell center
            diag_fins  = res[4].get("fins", -1.0)    # insertion-only force (gates break)
            diag_fsurf = res[4].get("fsurf", -1.0)   # whole-body surf force (artifact-prone)
            obs = res[0]["policy"].to(dev)

        # ── Generalised Advantage Estimation (GAE) ────────────────────────
        with torch.no_grad():
            last_v = value(obs, env.f_cmd_norm().to(dev)).squeeze(-1)

        O  = torch.stack(O);  Fc = torch.stack(Fc)
        A  = torch.stack(A);  LP = torch.stack(LP)
        R  = torch.stack(R);  D  = torch.stack(D);  V = torch.stack(V)
        adv = torch.zeros_like(R)
        gae = torch.zeros(N, device=dev)
        nv  = last_v
        for t in reversed(range(args.rollout)):
            delta  = R[t] + args.gamma * nv * (1.0 - D[t]) - V[t]
            gae    = delta + args.gamma * args.lam * (1.0 - D[t]) * gae
            adv[t] = gae
            nv     = V[t]
        ret = adv + V

        # Flatten for mini-batch updates
        bO   = O.reshape(-1, pcfg.obs_dim)
        bFc  = Fc.reshape(-1, 1)
        bA   = A.reshape(-1, 7)
        bLP  = LP.reshape(-1)
        bAdv = adv.reshape(-1)
        bRet = ret.reshape(-1)
        bAdv = (bAdv - bAdv.mean()) / (bAdv.std() + 1e-8)
        B    = bO.shape[0]

        # ── PPO mini-batch updates ────────────────────────────────────────
        for _ in range(args.epochs):
            idx = torch.randperm(B, device=dev)
            for s in range(0, B, args.minibatch):
                j     = idx[s : s + args.minibatch]
                mean, std = policy(bO[j], bFc[j])
                dist  = torch.distributions.Normal(mean, std)
                nlp   = dist.log_prob(bA[j]).sum(-1)
                ratio = torch.exp(nlp - bLP[j])
                a1    = ratio * bAdv[j]
                a2    = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * bAdv[j]
                aloss = -torch.min(a1, a2).mean()
                vloss = F.mse_loss(value(bO[j], bFc[j]).squeeze(-1), bRet[j])
                ent   = dist.entropy().sum(-1).mean()
                loss  = aloss + 0.5 * vloss - 0.0005 * ent
                aopt.zero_grad(); copt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                aopt.step(); copt.step()

        # ── Periodic logging + checkpoint ─────────────────────────────────
        if it % 5 == 0:
            succ = ep_succ / max(ep_end, 1.0)
            brk  = ep_brk  / max(ep_end, 1.0)
            fps  = (it + 1) * args.rollout * N / (time.perf_counter() - t0)
            print(
                f"it {it:4d}  rew {R.mean().item():.3f}  ret {bRet.mean().item():.2f}  "
                f"succ {succ:.3f}  brk {brk:.3f}  "
                f"dist {diag_fdist:.3f} dxy {diag_fdxy:.3f} bz {diag_fbz:.3f} "
                f"min {diag_fmin:.3f} seat {diag_fseat:.2f}  "
                f"Fins {diag_fins:.1f} Fsurf {diag_fsurf:.1f}  "
                f"fps {fps:.0f}",
                flush=True,
            )
            if USE_WANDB:
                wandb.log(
                    {
                        "rew":  R.mean().item(),
                        "ret":  bRet.mean().item(),
                        "succ": succ,
                        "brk":  brk,
                        "fps":  fps,
                    },
                    step=it,
                )
            torch.save(
                {"policy_state_dict": policy.state_dict(), "policy_cfg": dict(vars(pcfg))},
                ckpt,
            )

    # Final save
    torch.save(
        {"policy_state_dict": policy.state_dict(), "policy_cfg": pcfg},
        ckpt,
    )
    print(f"TRAIN_DONE -> {ckpt}", flush=True)
    _app.close()


if __name__ == "__main__":
    main()
