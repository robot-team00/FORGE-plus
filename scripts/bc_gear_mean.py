#!/usr/bin/env python3
"""Behavior-clone the policy MEAN onto successful sampled rollouts (self-imitation).

Takes the npz from collect_gear_bc.py (strict-seat episodes only) and fits the
mean head with MSE on the executed actions, starting from the source checkpoint's
weights (value head untouched). log_std is then SET to --set_log_std so the
subsequent PPO polish explores gently around the cloned mean.

No Isaac needed — pure torch.

    /workspace/.venv/bin/python scripts/bc_gear_mean.py \
        --data /workspace/logs/gear_bc_data.npz \
        --resume checkpoints/task1_gear_rq_anneal.pt.it200 \
        --out checkpoints/task1_gear_rq_bc.pt --epochs 40
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from forge_plus.skills.policy_network import ForceConditionedPolicy, PolicyConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--resume", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--holdout", type=float, default=0.05)
    p.add_argument("--set_log_std", type=float, default=-1.9, help="log_std written to the output ckpt (exp(-1.9)=0.15)")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(args.data)
    obs = torch.tensor(d["obs"], dtype=torch.float32)
    fcmd = torch.tensor(d["fcmd"], dtype=torch.float32)
    act = torch.tensor(d["act"], dtype=torch.float32)
    n = obs.shape[0]
    print(f"[bc] {n} steps from {int(d['n_succ'])}/{int(d['n_ep'])} episodes")

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    n_hold = max(int(n * args.holdout), 1)
    hold, tr = perm[:n_hold], perm[n_hold:]

    ck = torch.load(args.resume, map_location=dev, weights_only=False)
    pc = ck["policy_cfg"]
    pcfg = pc if isinstance(pc, PolicyConfig) else PolicyConfig(**pc)
    policy = ForceConditionedPolicy(pcfg).to(dev)
    policy.load_state_dict(ck["policy_state_dict"])
    # only the mean pathway trains; log_std is overwritten at save time
    opt = torch.optim.Adam((q for name, q in policy.named_parameters()
                            if name != "log_std"), lr=args.lr)

    def eval_hold() -> float:
        policy.eval()
        with torch.no_grad():
            m, _ = policy(obs[hold].to(dev), fcmd[hold].to(dev))
            return torch.nn.functional.mse_loss(m.clamp(-1, 1), act[hold].to(dev)).item()

    print(f"[bc] holdout MSE before: {eval_hold():.4f}")
    for ep in range(args.epochs):
        policy.train()
        perm_ep = tr[torch.randperm(tr.numel(), generator=g)]
        tot = cnt = 0.0
        for i in range(0, perm_ep.numel(), args.batch):
            idx = perm_ep[i: i + args.batch]
            m, _ = policy(obs[idx].to(dev), fcmd[idx].to(dev))
            loss = torch.nn.functional.mse_loss(m, act[idx].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * idx.numel()
            cnt += idx.numel()
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[bc] epoch {ep:3d} train MSE {tot / cnt:.4f} holdout {eval_hold():.4f}",
                  flush=True)

    policy.log_std.data.fill_(args.set_log_std)
    ck["policy_state_dict"] = policy.state_dict()
    torch.save(ck, args.out)
    print(f"[bc] saved -> {args.out} (log_std set to {args.set_log_std})")
    print("BC_DONE", flush=True)


if __name__ == "__main__":
    main()
