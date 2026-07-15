#!/usr/bin/env python3
"""Collect DEMONSTRATION rollouts for BC from the probe-19 controller.

The scripted aim + force-gated press + spiral-search controller is the only
actor that seats on the 2F-140 plant (probe 19: funnel capture verified; v4-v7
PPO runs never sample it from the transferred mean's landing). Record its
(obs, f_cmd, action) stream on episodes that reach the STRICT seat and hold,
then behavior-clone the policy mean (bc_gear_mean.py) and PPO-polish. Training
data only — evals/renders stay 100%% learned-policy (never-script rule).

Note the controller is representable by the policy net: the aim term is linear
in base_to_goal (in obs), the force gate reads the wrench (in obs); only the
time dither isn't observable — the BC mean averages it out and the polish
noise substitutes.

    /workspace/.venv/bin/python scripts/collect_gear_demos.py \
        --out /workspace/logs/gear_demo_data.npz --blocks 4
"""
from __future__ import annotations

import argparse
import math
from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})

import os  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from forge_plus.isaac_gear_env import FrankaGearInsertEnv, GearInsertEnvCfg  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--obj", type=int, default=-1)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--settle_keep", type=int, default=40)
    p.add_argument("--noise", type=float, default=0.0,
                   help="DART: execute expert + N(0,noise), record the CLEAN "
                        "expert action. DEAD END at 0.12: 1/320 seats, ~40 "
                        "breaks/block — the insertion tolerance is far below "
                        "that noise scale.")
    p.add_argument("--driver", default=None,
                   help="DAgger: checkpoint whose DETERMINISTIC mean drives "
                        "the env while the expert relabels every visited "
                        "state. Keeps ALL episodes (the driver's drift IS "
                        "the distribution to correct).")
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
    N = env.num_envs

    driver = None
    if args.driver:
        from forge_plus.skills.policy_network import ForceConditionedPolicy, PolicyConfig
        ck = torch.load(args.driver, map_location=env.device, weights_only=False)
        pc = ck["policy_cfg"]
        pcfg = pc if isinstance(pc, PolicyConfig) else PolicyConfig(**pc)
        driver = ForceConditionedPolicy(pcfg).to(env.device)
        driver.load_state_dict(ck["policy_state_dict"])
        driver.eval()
        print(f"[demo] DAgger driver: {args.driver}", flush=True)

    obs_buf, fcmd_buf, act_buf = [], [], []
    n_succ_total = n_ep_total = 0
    for blk in range(args.blocks):
        out = env.reset()
        obs = (out[0] if isinstance(out, tuple) else out)["policy"]
        ep_succ = torch.zeros(N, dtype=torch.bool, device=env.device)
        ep_brk = torch.zeros(N, dtype=torch.bool, device=env.device)
        seat_step = torch.full((N,), -1, dtype=torch.long, device=env.device)
        brk_step = torch.full((N,), -1, dtype=torch.long, device=env.device)
        o_steps, f_steps, a_steps = [], [], []
        for k in range(args.steps):
            gear = env._obj.data.root_pose_w[:, :3]
            goal = env._forge_goal_w()
            gz_l = gear[:, 2] - env.scene.env_origins[:, 2]
            live = (env._setup_ctr == 0).float()
            cf = env._cf_insert
            act = torch.zeros(N, 7, device=env.device)
            err = goal[:, :2] - gear[:, :2]
            act[:, :2] = (0.6 * err / env.cfg.forge_act_range).clamp(-1.0, 1.0)
            act[:, 2] = (-0.02 * (4.0 - cf)).clamp(-0.12, 0.08)
            on = ((cf > 0.5) & (cf < 5.0) & (gz_l > 0.418)).float()
            act[:, 0] += 0.05 * on * math.sin(2 * math.pi * k / 60.0)
            act[:, 1] += 0.05 * on * math.cos(2 * math.pi * k / 60.0)
            # HOLD once seated: freeze so the settle counter latches and the
            # demo teaches "stop moving when seated"
            act[:, :3] *= (live * (~env._succeeded).float()).unsqueeze(-1)
            act = act.clamp(-1.0, 1.0)
            fcmd = env.f_cmd_norm()
            o_steps.append(obs.cpu())
            f_steps.append(fcmd.cpu())
            a_steps.append(act.cpu())   # label = CLEAN expert action
            exec_act = act
            if driver is not None:
                with torch.no_grad():
                    dmean, _ = driver(obs, fcmd)
                exec_act = dmean.clamp(-1.0, 1.0)
            elif args.noise > 0.0:
                exec_act = (act + args.noise * torch.randn_like(act)).clamp(-1.0, 1.0)
            res = env.step(exec_act)
            obs = res[0]["policy"]
            new_succ = env._succeeded & ~ep_succ & ~ep_brk
            seat_step = torch.where(new_succ, torch.full_like(seat_step, k), seat_step)
            ep_succ |= new_succ
            new_brk = env._broke & ~ep_succ & ~ep_brk
            brk_step = torch.where(new_brk, torch.full_like(brk_step, k), brk_step)
            ep_brk |= new_brk
            done_mask = ep_brk | (ep_succ & (k - seat_step >= args.settle_keep))
            if bool(done_mask.all()):
                break
        O = torch.stack(o_steps)
        F = torch.stack(f_steps)
        A = torch.stack(a_steps)
        T = O.shape[0]
        for i in range(N):
            if driver is not None:
                # DAgger: keep EVERY trajectory (expert labels correct the
                # driver's drift), truncated at break / seat+settle
                if bool(ep_brk[i]):
                    end = max(int(brk_step[i]), 0)
                elif bool(ep_succ[i]):
                    end = min(int(seat_step[i]) + args.settle_keep, T - 1)
                else:
                    end = T - 1
            else:
                if not bool(ep_succ[i]):
                    continue
                end = min(int(seat_step[i]) + args.settle_keep, T - 1)
            obs_buf.append(O[: end + 1, i])
            fcmd_buf.append(F[: end + 1, i])
            act_buf.append(A[: end + 1, i])
        n_succ_total += int(ep_succ.sum())
        n_ep_total += N
        print(f"[demo] block {blk}: succ {int(ep_succ.sum())}/{N} "
              f"brk {int(ep_brk.sum())} steps {k + 1} "
              f"(total {n_succ_total}/{n_ep_total})", flush=True)

    obs_all = torch.cat(obs_buf).numpy() if obs_buf else np.zeros((0, 34))
    fcmd_all = torch.cat(fcmd_buf).numpy() if fcmd_buf else np.zeros((0, 1))
    act_all = torch.cat(act_buf).numpy() if act_buf else np.zeros((0, 7))
    np.savez_compressed(args.out, obs=obs_all, fcmd=fcmd_all, act=act_all,
                        n_succ=n_succ_total, n_ep=n_ep_total)
    print(f"[demo] saved {obs_all.shape[0]} steps from {n_succ_total} "
          f"successful episodes -> {args.out}")
    print("COLLECT_DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
