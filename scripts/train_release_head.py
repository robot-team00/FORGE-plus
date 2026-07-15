#!/usr/bin/env python3
"""Fit the gear LEARNED-release head (release_obs_head input-skip Linear).

History (all three failures shaped this design):
1. Frozen-trunk linear probe, all steps: learned 'open iff already open'
   (post-open obs are giveaways) — never fired at eval, 64/64 abs breaks.
2. Frozen-trunk probe, pre-open steps only: 27% FPR / 52% FNR — the uni
   trunk's features cannot separate 'seated, open now' from 'descending'.
3. Trunk fine-tune with arm distillation: the two losses fight (FNR pinned
   at 50%, zero-FPR window coverage 0/256, arm max-dev 0.11).

The decision IS linearly separable in the RAW observations (logreg: FNR 0%,
255/256 episodes covered at a zero-FPR threshold; features: phase onehot,
ee_z, wrench) — so the policy net gets a release input-skip head
(PolicyConfig.release_obs_head): act[7] = Linear(obs, f_cmd). This trainer
fits ONLY that head — trunk + arm rows stay bit-identical to the uni
checkpoint. Labels: +1 on the `lead` closed-gripper steps before the
expert's open; post-open steps DROPPED (the env latches the first +1).
The bias is then shifted to the zero-FPR operating point + margin.

    /workspace/.venv/bin/python scripts/train_release_head.py \
        --data /workspace/logs/gear_release_data2.npz \
        --resume checkpoints/task1_gear_rq_uni.pt \
        --out checkpoints/task1_gear_rq_uni_rel.pt
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
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lead", type=int, default=5)
    p.add_argument("--margin", type=float, default=0.15,
                   help="bias set so every hold logit sits this far below 0")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(args.data)
    setup = d["setup"].astype(bool)
    lab_raw = (d["act"][:, 7] > 0).astype(np.float32)
    starts = np.flatnonzero(setup & ~np.roll(setup, 1))
    starts[0] = 0
    bounds = list(zip(list(starts), list(starts[1:]) + [len(setup)]))
    keep = ~setup
    for a, b in bounds:
        pos = np.flatnonzero(lab_raw[a:b] > 0)
        if pos.size:
            lo = max(0, pos[0] - args.lead)
            lab_raw[a + lo: a + pos[0]] = 1.0
            keep[a + pos[0]: b] = False      # drop post-open (latched anyway)
    ep_of = np.zeros(len(setup), dtype=np.int64)
    for i, (a, b) in enumerate(bounds):
        ep_of[a:b] = i

    obs = torch.tensor(d["obs"][keep], dtype=torch.float32, device=dev)
    fcmd = torch.tensor(d["fcmd"][keep], dtype=torch.float32, device=dev)
    lab = torch.tensor(lab_raw[keep], device=dev)
    ep_id = torch.tensor(ep_of[keep], device=dev)
    n = obs.shape[0]
    n_pos = int(lab.sum())
    print(f"[rel] {n} pre-open steps ({n_pos} release-window) "
          f"from {int(d['n_succ'])}/{int(d['n_ep'])} episodes", flush=True)

    ck = torch.load(args.resume, map_location=dev, weights_only=False)
    src_sd = ck["policy_state_dict"]
    src_pc = ck["policy_cfg"]
    src_cfg = src_pc if isinstance(src_pc, PolicyConfig) else PolicyConfig(**src_pc)
    assert src_cfg.act_dim == 7
    pcfg = PolicyConfig(obs_dim=src_cfg.obs_dim, act_dim=8, release_obs_head=True)
    policy = ForceConditionedPolicy(pcfg).to(dev)
    sd = policy.state_dict()
    for k, v in src_sd.items():
        if k in ("mean_head.weight", "mean_head.bias", "log_std"):
            t = sd[k].clone()
            t[: v.shape[0]] = v
            if k == "mean_head.weight":
                t[v.shape[0]:] = 0.0      # dead row — release comes from the skip
            if k == "mean_head.bias":
                t[v.shape[0]:] = 0.0
            if k == "log_std":
                t[v.shape[0]:] = -3.0
            sd[k] = t
        elif k in sd and sd[k].shape == v.shape:
            sd[k] = v
    policy.load_state_dict(sd)
    for prm in policy.parameters():
        prm.requires_grad_(False)
    policy.release_head.weight.requires_grad_(True)
    policy.release_head.bias.requires_grad_(True)
    torch.nn.init.zeros_(policy.release_head.weight)
    torch.nn.init.constant_(policy.release_head.bias, -1.0)
    frozen = {k: v.clone() for k, v in policy.state_dict().items()
              if not k.startswith("release_head")}

    # input normalisation baked into the Linear afterwards: train on
    # standardized features for conditioning, then fold (mu, sd) into (W, b)
    x_in = torch.cat([obs, fcmd], dim=-1)
    mu = x_in.mean(0)
    sdv = x_in.std(0) + 1e-6
    live = x_in.std(0) > 1e-4     # near-constant dims (unused phase-onehot slots)
    sdv = torch.where(live, sdv, torch.ones_like(sdv))   # excluded from the fit;
    mu = torch.where(live, mu, torch.zeros_like(mu))     # folding w/std for a
    xn = (x_in - mu) / sdv                               # ~1e-6 std explodes fp32
    w = torch.zeros(xn.shape[1], device=dev, requires_grad=True)
    b = torch.tensor(-1.0, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=args.lr)
    pos_w = torch.tensor((n - n_pos) / max(n_pos, 1), device=dev)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    g = torch.Generator().manual_seed(0)
    for ep in range(args.epochs):
        perm = torch.randperm(n, generator=g).to(dev)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i: i + args.batch]
            loss = bce(xn[idx] @ w + b, lab[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * idx.shape[0]
        if ep % 50 == 0 or ep == args.epochs - 1:
            with torch.no_grad():
                logit = xn @ w + b
            fp = int(((logit > 0) & (lab < 0.5)).sum())
            fn = int(((logit <= 0) & (lab > 0.5)).sum())
            print(f"[rel] ep{ep:3d} loss {tot / n:.4f} "
                  f"FPR {fp}/{n - n_pos} FNR {fn}/{n_pos}", flush=True)

    # fold standardization into the Linear, THEN set the zero-FPR operating
    # point in DEPLOYMENT space (the folded policy forward — fp32 fold error
    # scales with the trained logit magnitude, so thresholding the pre-fold
    # logits drifts; measuring hold_max through the actual head is exact)
    with torch.no_grad():
        policy.release_head.weight.data[0] = w / sdv
        policy.release_head.bias.data[0] = float(b - (w * mu / sdv).sum())
        mean_all = torch.cat([policy(obs[i: i + 65536], fcmd[i: i + 65536])[0]
                              for i in range(0, n, 65536)])
        lg0 = mean_all[:, 7]
        hold_max = float(lg0[lab < 0.5].max())
        tau = -(hold_max + args.margin)
        policy.release_head.bias.data[0] += tau
        lg = lg0 + tau
        fired = torch.zeros(len(bounds), dtype=torch.bool, device=dev)
        haswin = torch.zeros(len(bounds), dtype=torch.bool, device=dev)
        pm = lab > 0.5
        fired.index_put_((ep_id[pm],), (lg[pm] > 0), accumulate=True)
        haswin.index_put_((ep_id[pm],), torch.ones_like(pm[pm]), accumulate=True)
        cov = int((fired & haswin).sum())
        tot_w = int(haswin.sum())
        fp0 = int(((lg > 0) & (lab < 0.5)).sum())
        for k, v in policy.state_dict().items():
            if not k.startswith("release_head"):
                assert torch.equal(v, frozen[k]), f"{k} drifted"
    print(f"[rel] operating point: shift {tau:+.3f} -> hold FP {fp0}/{n - n_pos}, "
          f"window coverage {cov}/{tot_w} eps; trunk+arm verified UNCHANGED",
          flush=True)

    torch.save({"policy_state_dict": policy.state_dict(),
                "policy_cfg": pcfg}, args.out)
    print(f"[rel] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
