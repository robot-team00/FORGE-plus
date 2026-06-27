#!/usr/bin/env python3
"""eval_rollout_task3.py - CPU rollout for Task 3 (fragile place & stack).

Uses FrankaFragilePlaceEnv: full pick-and-place with breakage risk at
BOTH grasp (grip force) and place (contact force) phases.

Outputs data/task3/rollout_002.npz with trajectory + episode metrics.
"""
import sys, argparse
import numpy as np
import torch
from pathlib import Path

REPO = "/workspace/FORGE-plus_task3"
sys.path.insert(0, REPO)

from forge_plus.envs.franka_fragile_place_env import FrankaFragilePlaceEnv, FragilePlaceEnvConfig
from forge_plus.envs.object_configs import sample_f_break
from forge_plus.envs.base_assembly_env import EpisodeConfig
from forge_plus.control.force_clamp import Wrench

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=f"{REPO}/checkpoints/task3_franka_panda.pt")
    p.add_argument("--out", default=f"{REPO}/data/task3/rollout_002.npz")
    p.add_argument("--object-key", default="glass_bowl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--no-llm", action="store_true",
                   help="Skip Ollama LLM call, use heuristic fallback")
    return p.parse_args()

def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[rollout] Loading policy checkpoint: {args.checkpoint}", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from forge_plus.skills.policy_network import ForceConditionedPolicy
    policy = ForceConditionedPolicy(ckpt["policy_cfg"])
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    obs_dim = ckpt["policy_cfg"].obs_dim
    act_dim = ckpt["policy_cfg"].act_dim
    print(f"[rollout] Policy ready: obs_dim={obs_dim} act_dim={act_dim}", flush=True)

    # Build episode config
    rng = np.random.default_rng(args.seed)
    f_break = sample_f_break(args.object_key)
    ep_cfg = EpisodeConfig(
        object_key=args.object_key,
        task_name="task3_fragile_place",
        gripper="franka_panda",
        f_break_n=f_break,
        max_steps=args.max_steps,
        disturbance_seed=int(rng.integers(0, 10000)),
    )
    print(f"[rollout] Episode: object={args.object_key}  f_break={f_break:.2f}N  seed={ep_cfg.disturbance_seed}", flush=True)

    # Build env
    use_llm = not args.no_llm
    env = FrankaFragilePlaceEnv(use_llm=use_llm)
    obs = env.reset(ep_cfg)
    print(f"[rollout] Env reset. LLM F_max={env.f_max_n:.2f}N  use_llm={use_llm}", flush=True)

    # Franka Panda home joints
    joints = np.array([0., -0.785, 0., -2.356, 0., 1.571, 0.785], dtype=np.float32)
    f_cmd = torch.full((1, 1), env.f_max_n / 120.0)   # normalised budget

    ee_log, joints_log, force_log, phase_log, outcome_log = [], [], [], [], []
    done_at = args.max_steps

    print(f"[rollout] Running up to {args.max_steps} steps ...", flush=True)
    for step in range(args.max_steps):
        # 34-dim obs (same layout as original eval_rollout.py)
        obs_np = np.zeros(obs_dim, dtype=np.float32)
        obs_np[:7]    = joints
        obs_np[7:14]  = 0.0
        obs_np[14:17] = obs.ee_pos
        obs_np[17:21] = obs.ee_quat
        obs_np[21:24] = [0.48, 0.00, 0.665]   # place target
        obs_np[24:28] = [0., 0., 0., 1.]
        obs_np[28:31] = 0.0
        obs_np[31]    = f_cmd.item()

        obs_t = torch.from_numpy(obs_np).unsqueeze(0)
        with torch.no_grad():
            action, _ = policy.get_action(obs_t, f_cmd, deterministic=True)
        a = action[0].numpy()
        joints = np.clip(joints + a * 0.05, -3.14, 3.14).astype(np.float32)

        # Map policy action to wrench command
        wrench = Wrench(
            fx=float(a[0]) * 5.0,
            fy=float(a[1]) * 5.0,
            fz=float(a[2]) * env.f_max_n,
            tx=0.0, ty=0.0, tz=0.0,
        )
        obs, outcome = env.step(wrench)

        ee_log.append(obs.ee_pos.copy())
        joints_log.append(joints.copy())
        force_log.append(env.get_contact_force_magnitude())
        phase_log.append(obs.phase.value)
        outcome_log.append(outcome.value)

        if (step + 1) % 30 == 0:
            m = env.get_episode_metrics()
            print(f"  step {step+1:3d}  phase={env.episode_phase.value:<20s}  "
                  f"force={env.get_contact_force_magnitude():.2f}N  "
                  f"outcome={outcome.value}", flush=True)

        if env.is_done():
            done_at = step + 1
            break

    metrics = env.get_episode_metrics()
    print(f"\n[rollout] Episode complete at step {done_at}", flush=True)
    print(f"  outcome        : {env._outcome.value}", flush=True)
    print(f"  success        : {metrics.success}", flush=True)
    print(f"  broken         : {metrics.broken}", flush=True)
    print(f"  broken_at_phase: {metrics.broken_at_phase}", flush=True)
    print(f"  failure_mode   : {metrics.failure_mode}", flush=True)
    print(f"  force_economy  : {metrics.force_economy:.3f}", flush=True)
    print(f"  peak_grip_N    : {metrics.peak_grip_force_n:.2f}", flush=True)
    print(f"  peak_place_N   : {metrics.peak_place_force_n:.2f}", flush=True)
    print(f"  f_max_N        : {metrics.f_max_n:.2f}", flush=True)

    np.savez(
        str(out_path),
        joints   = np.array(joints_log, dtype=np.float32),
        ee       = np.array(ee_log, dtype=np.float32),
        force    = np.array(force_log, dtype=np.float32),
        phase    = np.array(phase_log),
        outcome  = np.array(outcome_log),
        done_at  = np.array([done_at]),
        # metrics
        success          = np.array([metrics.success]),
        broken           = np.array([metrics.broken]),
        broken_at_phase  = np.array([metrics.broken_at_phase]),
        failure_mode     = np.array([metrics.failure_mode]),
        force_economy    = np.array([metrics.force_economy]),
        f_max_n          = np.array([metrics.f_max_n]),
        f_break_n        = np.array([f_break]),
    )
    print(f"[rollout] Saved -> {out_path}", flush=True)
    print(f"[rollout] done.", flush=True)

if __name__ == "__main__":
    main()
