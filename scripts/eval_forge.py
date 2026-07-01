#!/usr/bin/env python3
"""Headless deterministic eval of the LEARNED forge insertion policy (no RTX render).

Loads checkpoints/task3_forge_insert.pt, drives the forge env with the policy's
DETERMINISTIC action (mean), and reports the success / breakage rate over many
parallel episodes. Proves the learned policy inserts the bottle even when the RTX
render path is unavailable.

    export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 \
           PYTHONPATH=/workspace/FORGE-plus_task3
    /workspace/.venv/bin/python scripts/eval_forge.py --obj 0 --episodes 200
"""
from __future__ import annotations

import argparse
from isaacsim import SimulationApp  # noqa: E402

_app = SimulationApp({"headless": True})

import os  # noqa: E402
import torch  # noqa: E402
from forge_plus.isaac_pick_place_env import FrankaPickPlaceEnv, PickPlaceEnvCfg  # noqa: E402
from forge_plus.skills.policy_network import ForceConditionedPolicy, PolicyConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/task3_forge_insert.pt")
    p.add_argument("--obj", type=int, default=0, help="object class (0=glass bottle budget)")
    p.add_argument("--num_envs", type=int, default=128)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=400)
    args = p.parse_args()

    cfg = PickPlaceEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.forge_mode = True
    cfg.grasp_topdown = False
    cfg.forge_obj_cls = args.obj
    env = FrankaPickPlaceEnv(cfg)

    ck = torch.load(args.ckpt, map_location=env.device, weights_only=False)
    pc = ck["policy_cfg"]
    pcfg = pc if isinstance(pc, PolicyConfig) else PolicyConfig(**pc)
    policy = ForceConditionedPolicy(pcfg).to(env.device)
    policy.load_state_dict(ck["policy_state_dict"])
    policy.eval()
    print(f"[eval] loaded {args.ckpt}  obs={pcfg.obs_dim} act={pcfg.act_dim}  "
          f"F_max={env.f_max_n:.1f}N (obj {args.obj})", flush=True)

    out = env.reset()
    obs = (out[0] if isinstance(out, tuple) else out)["policy"]
    n_succ = n_brk = n_end = 0
    step = 0
    while n_end < args.episodes and step < args.max_steps * 40:
        with torch.no_grad():
            mean, _ = policy(obs, env.f_cmd_norm())
            act = mean.clamp(-1.0, 1.0)
        res = env.step(act)
        obs = res[0]["policy"]
        n_succ += int(res[4].get("n_succ", 0.0))
        n_brk += int(res[4].get("n_brk", 0.0))
        n_end += int((res[2] | res[3]).sum().item())
        step += 1
        if step % 50 == 0:
            print(f"[eval] step {step}  ended={n_end}  succ={n_succ}  brk={n_brk}", flush=True)

    rate = n_succ / max(n_end, 1)
    brate = n_brk / max(n_end, 1)
    print(f"\n=== FORGE policy eval ===")
    print(f"episodes ended : {n_end}")
    print(f"SUCCESS rate   : {rate:.3f}  ({n_succ}/{n_end})")
    print(f"BREAK rate     : {brate:.3f}  ({n_brk}/{n_end})")
    print(f"=========================", flush=True)
    _app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
