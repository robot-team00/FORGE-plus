#!/usr/bin/env python3
"""train_gear_insert.py — Vectorized PPO trainer for FORGE-plus Task 1 gear insertion.

Trains a ForceConditionedPolicy (FiLM MLP) to insert the FORGE medium spur gear
onto the gear-base middle shaft (0.25 mm bore clearance), subject to per-object
force budgets set by an offline LLM supervisor (FORGE framework). Ported from
task3's train_pick_place.py; PPO machinery unchanged.

Architecture
------------
* ForceConditionedPolicy: FiLM-modulated MLP (obs_dim=34, act_dim=7, hidden=256)
* ValueNetwork:           same conditioning, outputs scalar V(s, F_cmd)
* Algorithm:              PPO with GAE (clipped surrogate + value loss + entropy)

Usage
-----
    DISPLAY=:99 HOME=/workspace/persist/ovhome MPLBACKEND=Agg \\
    /workspace/.venv/bin/python scripts/train_gear_insert.py \\
        --forge --forge_obj 1 --num_envs 512 --iterations 600

Logs to W&B project "forge-plus-task1" (key auto-read from /workspace/.jr_notes).
Checkpoint: checkpoints/task1_gear_insert_franka.pt
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

from forge_plus.isaac_gear_env import (  # noqa: E402
    FrankaGearInsertEnv,
    GearInsertEnvCfg,
)
from forge_plus.skills.policy_network import (  # noqa: E402
    PolicyConfig,
    ForceConditionedPolicy,
    ValueNetwork,
)


def main() -> None:
    p = argparse.ArgumentParser(description="FORGE-plus Task 1: gear-insertion PPO")
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
    p.add_argument("--ckpt",                   default="checkpoints/task1_gear_insert_franka.pt",
                   help="path to save checkpoint")
    p.add_argument("--resume",                 default=None,
                   help="resume from checkpoint file")
    p.add_argument("--run",                    default=None,
                   help="W&B run name; defaults to today's date")
    p.add_argument("--no_upright", action="store_true",
                   help="extrinsic curriculum stage A: place on the shelf WITHOUT the upright requirement")
    p.add_argument("--reset_std", type=float, default=None,
                   help="after --resume, reset policy log_std to this (re-inflate exploration for stage B)")
    p.add_argument("--std_anneal_to", type=float, default=None,
                   help="anneal a log_std CEILING from the resume value down to this target "
                        "(deterministic-mean consolidation: the robotiq policy converged to "
                        "std~1.0 — sampled noise does the funnel search and the mean never "
                        "gets gradient pressure to be precise; shrinking the ceiling forces "
                        "PPO to move the competence into the mean)")
    p.add_argument("--std_anneal_its", type=int, default=None,
                   help="iterations (from resume) to reach --std_anneal_to; default = all remaining")
    p.add_argument("--forge", action="store_true",
                   help="FORGE-style LEARNED insertion: policy drives the EE, no scripted waypoints/base-aim (obs=34)")
    p.add_argument("--forge_obj", type=int, default=None,
                   help="fix the forge training object class (0=abs_gear/fragile, 1=steel_gear/robust)")
    p.add_argument("--forge_release", action="store_true",
                   help="LEARNED safe release: 8th action dim lets the policy let go; success needs a settled drop (obs=34, act=8)")
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
            run_name = args.run or (_dt.date.today().isoformat() + "_gear_insert")
            wandb.init(
                project="forge-plus-task1",
                name=f"gear_insert_{args.gripper}_{run_name}",
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
    cfg = GearInsertEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.gripper        = args.gripper
    if args.no_upright:
        cfg.require_upright = False   # curriculum stage A
    obs_dim = 37
    act_dim = 7
    if args.forge or args.forge_release:
        cfg.forge_mode = True         # LEARNED insertion: policy drives the EE (no waypoints)
        # GEAR PORT: keep grasp_topdown=True (the env default). Task3 forced the
        # natural forward grip here because the bottle scene's cell was reach-
        # limited at z 0.68; the gear scene hands off at z 0.565 and the probe
        # confirms the top-down wrist reaches the shaft — and a rigidly-held
        # gear MUST stay level to enter the 0.25 mm-clearance bore.
        if args.forge_obj is not None:
            cfg.forge_obj_cls = args.forge_obj   # 0=fragile glass bottle, 2=robust
        obs_dim = 34
    if args.forge_release:
        cfg.forge_release_mode = True    # LEARNED gripper release (8th action dim)
        cfg.forge_hybrid_retract = True  # env retracts the hand after release (scripted clearing) so
                                         # the hand_clear success can fire; policy learns insert + release
        act_dim = 8
    if args.gripper == "robotiq_2f140" and cfg.forge_mode:
        # The robotiq episode is scripted-staging-heavy (TELEPORT CONTRACT: no
        # joint-state reset writes, so every episode DRIVES from wherever the arm
        # is to the grasp, then carries to the hand-off). Without these the setup
        # window (default 80) expires mid-air and every env times out untouched.
        cfg.forge_setup_steps = 4000   # predrive 600 + seat 270 + hover-lift + ~2 cm/s carry
        cfg.warmup_substeps   = 100    # 2F-140 drive needs ~30 env steps to reach kiss angle
        # Four-bar loop joints only survive a RAW parse — the physics-replicated
        # clone path drops them and the fingers can never close (Fins == 0).
        cfg.scene.replicate_physics = False
        # The staging state machine is GLOBAL (python bools, servo on env 0), so
        # envs must stay synchronized: no per-env early resets. Truncation is the
        # only done — all envs reset together and the staging re-arms globally.
        cfg.forge_no_term = True
        # GEAR PORT staging v2 (seat at the entrance pose): staging is only
        # ~250 env steps (predrive 60 + seat 135 + hover/handoff), so 20 s
        # (1200 steps) leaves ~950 live policy steps. (The bottle's staged
        # carry needed 45 s.)
        cfg.episode_length_s = 20.0
    print(f"[train] forge_mode={cfg.forge_mode} release={cfg.forge_release_mode} place_strategy={cfg.place_strategy} obs={obs_dim} act={act_dim}", flush=True)
    env = FrankaGearInsertEnv(cfg)
    N   = env.num_envs
    print(f"[train] envs={N}  device={dev}  obs={obs_dim}  act={act_dim}", flush=True)

    # ── Networks ──────────────────────────────────────────────────────────
    pcfg   = PolicyConfig(obs_dim=obs_dim, act_dim=act_dim)
    policy = ForceConditionedPolicy(pcfg).to(dev)
    value  = ValueNetwork(pcfg).to(dev)
    aopt   = Adam(policy.parameters(), lr=args.lr)
    copt   = Adam(value.parameters(), lr=args.lr)

    start_iter = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt_data = torch.load(args.resume, map_location=dev, weights_only=False)
        ckpt_sd = ckpt_data["policy_state_dict"]
        ckpt_adim = ckpt_sd["mean_head.bias"].shape[0]
        if ckpt_adim == act_dim:
            policy.load_state_dict(ckpt_sd)
            print(f"[train] resumed from {args.resume} (act_dim={act_dim})", flush=True)
        else:
            # Warm-start an 8-dim release policy from a 7-dim insertion checkpoint:
            # copy the arm dims exactly (preserve the learned insertion skill); init the new
            # release dim mildly closed (bias<0) with extra exploration std so the policy
            # discovers WHEN to let go without losing the arm behaviour.
            sd = policy.state_dict()
            for k, v in ckpt_sd.items():
                if k in ("mean_head.weight", "mean_head.bias", "log_std"):
                    t = sd[k].clone(); t[:v.shape[0]] = v
                    if k == "mean_head.bias": t[v.shape[0]:] = -0.5   # start gripper closed-ish
                    if k == "log_std":        t[v.shape[0]:] = -0.5   # std~0.6 -> explore release
                    sd[k] = t
                elif k in sd and sd[k].shape == v.shape:
                    sd[k] = v
            policy.load_state_dict(sd)
            print(f"[train] warm-started act {ckpt_adim}->{act_dim} from {args.resume} "
                  f"(arm dims preserved, release dim initialised)", flush=True)
        if args.reset_std is not None and hasattr(policy, "log_std"):
            policy.log_std.data.fill_(float(args.reset_std))
            print(f"[train] reset log_std -> {args.reset_std} (re-inflate exploration)", flush=True)

    # ── log_std anneal schedule (per-dim ceiling from the resume value) ──
    anneal_start_ls = None
    if args.std_anneal_to is not None and hasattr(policy, "log_std"):
        anneal_start_ls = policy.log_std.data.clone()
        anneal_its = args.std_anneal_its or max(args.iterations - start_iter, 1)
        print(f"[train] std anneal: log_std ceiling {anneal_start_ls.max().item():.2f} -> "
              f"{args.std_anneal_to} over {anneal_its} its", flush=True)

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
            diag_farm  = res[4].get("farm", -1.0)    # arm EE reaction force (FORGE-style, artifact-free)
            diag_nrel  = res[4].get("n_rel", -1.0)   # forge_release: # envs that have let go
            diag_nbad  = res[4].get("n_badrel", -1.0)# forge_release: # that let go and lost the bottle
            diag_relupz = res[4].get("relupz", -1.0) # forge_release: mean uprightness (cos) of released
            diag_releod = res[4].get("releod", -1.0) # forge_release: mean EE->bottle dist after release (retract)
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
        bA   = A.reshape(-1, act_dim)
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

        # ── std-anneal ceiling clamp (after the PPO update each iteration) ─
        if anneal_start_ls is not None:
            frac = min(1.0, (it - start_iter + 1) / anneal_its)
            tgt = torch.full_like(anneal_start_ls, float(args.std_anneal_to))
            ceil = anneal_start_ls + frac * (tgt - anneal_start_ls)
            # ceiling only shrinks: dims already below stay free to move
            policy.log_std.data = torch.minimum(policy.log_std.data, ceil)

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
                f"Fins {diag_fins:.1f} Fsurf {diag_fsurf:.1f} Farm {diag_farm:.1f}  "
                f"rel {diag_nrel:.0f} badrel {diag_nbad:.0f} relupz {diag_relupz:.2f} releod {diag_releod:.2f}  "
                f"std {policy.log_std.data.exp().mean().item():.3f}  "
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
        # PPO drift can collapse a converged policy (mixed round-2 lesson);
        # keep periodic snapshots so a good checkpoint survives later drift.
        if it > 0 and it % 100 == 0:
            torch.save(
                {"policy_state_dict": policy.state_dict(), "policy_cfg": dict(vars(pcfg))},
                f"{ckpt}.it{it}",
            )

    # Final save
    torch.save(
        {"policy_state_dict": policy.state_dict(), "policy_cfg": pcfg},
        ckpt,
    )
    print(f"TRAIN_DONE -> {ckpt}", flush=True)
    # _app.close() hangs this pod's headless Kit after the summary (the ftlong
    # trainer sat R-state for hours post-TRAIN_DONE); hard exit frees the GPU
    os._exit(0)


if __name__ == "__main__":
    main()
