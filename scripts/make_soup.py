#!/usr/bin/env python3
"""Weight-space interpolation between BC lineage checkpoints (no Isaac needed).

soup = alpha * A + (1 - alpha) * B, saved per alpha. Only valid between
warm-start-related checkpoints (same loss basin): bc3<->bc7 (bc7 warm from
bc3) interpolates cleanly at EVERY alpha 0.10-0.75 — 32/32 both classes,
0 breaks, steel peak force monotone in alpha (bc3's 120 N press habit).
bc3<->bc4 (a cold-refit see-saw apart) fails at every alpha (0/32 both).

The unified deliverable task1_gear_rq_uni.pt = 0.15*bc3 + 0.85*bc7
(abs 256/256 0 brk p95 18.5 N; steel 256/256 0 brk p95 57.6 N):

    /workspace/.venv/bin/python scripts/make_soup.py \
        --a checkpoints/task1_gear_rq_bc3.pt \
        --b checkpoints/task1_gear_rq_bc7.pt \
        --alphas 0.15 --out_prefix checkpoints/soup_b3b7_a
"""
import argparse
import torch

p = argparse.ArgumentParser()
p.add_argument("--a", required=True, help="checkpoint A (weight = alpha)")
p.add_argument("--b", required=True, help="checkpoint B (weight = 1-alpha)")
p.add_argument("--alphas", type=float, nargs="+", required=True)
p.add_argument("--out_prefix", required=True, help="writes <prefix><alpha_pct>.pt")
args = p.parse_args()

ca = torch.load(args.a, map_location="cpu", weights_only=False)
cb = torch.load(args.b, map_location="cpu", weights_only=False)
sa, sb = ca["policy_state_dict"], cb["policy_state_dict"]
assert sa.keys() == sb.keys()
for alpha in args.alphas:
    sd = {k: alpha * sa[k] + (1.0 - alpha) * sb[k] for k in sa}
    out = f"{args.out_prefix}{int(round(alpha * 100)):02d}.pt"
    torch.save({"policy_state_dict": sd, "policy_cfg": ca["policy_cfg"]}, out)
    print(f"saved {out} (alpha={alpha} on {args.a})")
