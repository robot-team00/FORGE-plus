#!/usr/bin/env python3
"""Vectorized PPO trainer for the Task-3 Isaac place/stack env (FORGE FiLM policy).

Trains the force-conditioned skill over thousands of parallel Isaac envs.
Usage: python scripts/train_place.py --num-envs 1024 --gripper franka_panda
"""
from __future__ import annotations
import argparse

from isaacsim import SimulationApp
_app = SimulationApp({"headless": True})

import os, time  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.optim import Adam  # noqa: E402
from forge_plus.isaac_place_env import FrankaPlaceEnv, PlaceEnvCfg  # noqa: E402
from forge_plus.skills.policy_network import PolicyConfig, ForceConditionedPolicy, ValueNetwork  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--gripper", default="franka_panda")
    p.add_argument("--iterations", type=int, default=400)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ckpt", default=None)
    args = p.parse_args()
    dev = torch.device(args.device)
    # --- optional W&B logging (key from /workspace/.jr_notes, not printed) ---
    USE_WANDB = False
    try:
        import re, wandb
        txt = open("/workspace/.jr_notes").read()
        cands = re.findall(r"(?<![A-Za-z0-9_])[0-9a-f]{40}(?![A-Za-z0-9_])", txt)
        if not cands:
            cands = [t for t in re.findall(r"(?<![A-Za-z0-9_])[0-9A-Za-z]{40}(?![A-Za-z0-9_])", txt)
                     if not t.startswith(("ghp", "github"))]
        key = cands[0] if cands else None
        if key:
            os.environ["WANDB_API_KEY"] = key
            wandb.init(project="forge-plus-task3", name=f"place_{args.gripper}", config=vars(args))
            USE_WANDB = True
            print("wandb logging ON", flush=True)
    except Exception as e:
        print("wandb off:", str(e)[:80], flush=True)
    ckpt = args.ckpt or f"checkpoints/task3_{args.gripper}.pt"
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    cfg = PlaceEnvCfg(); cfg.scene.num_envs = args.num_envs
    env = FrankaPlaceEnv(cfg)
    N = env.num_envs
    pcfg = PolicyConfig(obs_dim=34, act_dim=7)
    policy = ForceConditionedPolicy(pcfg).to(dev)
    value = ValueNetwork(pcfg).to(dev)
    aopt = Adam(policy.parameters(), lr=args.lr)
    copt = Adam(value.parameters(), lr=args.lr)

    out = env.reset(); obs = (out[0] if isinstance(out, tuple) else out)["policy"]
    t0 = time.perf_counter()
    for it in range(args.iterations):
        O, Fc, A, LP, R, D, V = [], [], [], [], [], [], []
        for _ in range(args.rollout):
            fcmd = env.f_cmd_norm().to(dev)
            with torch.no_grad():
                mean, std = policy(obs, fcmd)
                dist = torch.distributions.Normal(mean, std)
                act = dist.sample(); lp = dist.log_prob(act).sum(-1)
                val = value(obs, fcmd).squeeze(-1)
            res = env.step(act)
            O.append(obs); Fc.append(fcmd); A.append(act); LP.append(lp)
            R.append(res[1]); D.append((res[2] | res[3]).float()); V.append(val)
            obs = res[0]["policy"]
        with torch.no_grad():
            last_v = value(obs, env.f_cmd_norm().to(dev)).squeeze(-1)
        O = torch.stack(O); Fc = torch.stack(Fc); A = torch.stack(A); LP = torch.stack(LP)
        R = torch.stack(R); D = torch.stack(D); V = torch.stack(V)
        adv = torch.zeros_like(R); gae = torch.zeros(N, device=dev); nv = last_v
        for t in reversed(range(args.rollout)):
            delta = R[t] + args.gamma * nv * (1 - D[t]) - V[t]
            gae = delta + args.gamma * args.lam * (1 - D[t]) * gae
            adv[t] = gae; nv = V[t]
        ret = adv + V
        bO = O.reshape(-1, 34); bFc = Fc.reshape(-1, 1); bA = A.reshape(-1, 7)
        bLP = LP.reshape(-1); bAdv = adv.reshape(-1); bRet = ret.reshape(-1)
        bAdv = (bAdv - bAdv.mean()) / (bAdv.std() + 1e-8)
        B = bO.shape[0]
        for _ in range(args.epochs):
            idx = torch.randperm(B, device=dev)
            for s in range(0, B, args.minibatch):
                j = idx[s:s + args.minibatch]
                mean, std = policy(bO[j], bFc[j]); dist = torch.distributions.Normal(mean, std)
                nlp = dist.log_prob(bA[j]).sum(-1); ratio = torch.exp(nlp - bLP[j])
                a1 = ratio * bAdv[j]; a2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * bAdv[j]
                aloss = -torch.min(a1, a2).mean()
                vloss = F.mse_loss(value(bO[j], bFc[j]).squeeze(-1), bRet[j])
                ent = dist.entropy().sum(-1).mean()
                loss = aloss + 0.5 * vloss - 0.01 * ent
                aopt.zero_grad(); copt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                aopt.step(); copt.step()
        if it % 5 == 0:
            succ = env._succeeded.float().mean().item(); brk = env._broke.float().mean().item()
            fps = (it + 1) * args.rollout * N / (time.perf_counter() - t0)
            print(f"it {it} rew {R.mean().item():.3f} ret {bRet.mean().item():.2f} "
                  f"succ {succ:.3f} brk {brk:.3f} fps {fps:.0f}", flush=True)
            if USE_WANDB:
                wandb.log({"rew": R.mean().item(), "ret": bRet.mean().item(), "succ": succ, "brk": brk}, step=it)
            torch.save({"policy_state_dict": policy.state_dict(), "policy_cfg": pcfg}, ckpt)
    torch.save({"policy_state_dict": policy.state_dict(), "policy_cfg": pcfg}, ckpt)
    print("TRAIN_DONE ->", ckpt, flush=True)
    _app.close()


if __name__ == "__main__":
    main()
