#!/usr/bin/env python3
"""Train a FORGE-style force-conditioned RL skill using PPO.

Usage:
    python scripts/train_skill.py --task task1 --gripper franka_panda
    python scripts/train_skill.py --task task2 --gripper robotiq_2f140 --num-envs 2048

The skill is trained with F_cmd conditioned observations. After training,
the checkpoint is saved and can be loaded by the episode runner.

Isaac Lab is required for GPU-parallel training. For testing without Isaac Lab,
use --mock-env which runs a lightweight CPU simulation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FORGE+ RL skill")
    p.add_argument("--task", choices=["task1", "task2", "task3"], default="task1")
    p.add_argument("--gripper", choices=["franka_panda", "robotiq_2f140"], default="franka_panda")
    p.add_argument("--num-envs", type=int, default=1, help="Parallel Isaac Lab envs (1 = mock)")
    p.add_argument("--total-steps", type=int, default=5_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=10, help="PPO update epochs per rollout")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--rollout-len", type=int, default=256, help="Steps per rollout collection")
    p.add_argument("--f-max-range", nargs=2, type=float, default=[10.0, 100.0],
                   help="Range of F_cmd values to sample during training")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--mock-env", action="store_true", default=True,
                   help="Use mock environment (no Isaac Lab required)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=10_000)
    return p.parse_args()


def build_env(args: argparse.Namespace):
    if args.mock_env or args.num_envs == 1:
        from forge_plus.envs.mock_assembly_env import MockAssemblyEnv, MockEnvConfig
        cfg = MockEnvConfig(jam_probability=0.3)
        return MockAssemblyEnv(cfg)
    else:
        from forge_plus.envs.isaac_lab_env import IsaacLabAssemblyEnv, IsaacLabEnvConfig
        cfg = IsaacLabEnvConfig(num_envs=args.num_envs, device=args.device)
        return IsaacLabAssemblyEnv(cfg)


def sample_episode_config(args, rng: np.random.Generator):
    """Sample a random episode config for training diversity."""
    from forge_plus.envs.object_configs import OBJECT_REGISTRY
    from forge_plus.envs.base_assembly_env import EpisodeConfig

    task_objects = {
        "task1": ["abs_round_connector", "steel_peg"],
        "task2": ["resin_planet_gear", "metal_planet_gear"],
        "task3": ["glass_bowl", "ceramic_plate", "metal_plate", "sturdy_mug"],
    }
    obj_key = rng.choice(task_objects[args.task])
    obj_cfg = OBJECT_REGISTRY[obj_key]
    f_break = obj_cfg.sample_f_break()

    return EpisodeConfig(
        object_key=obj_key,
        task_name=f"{args.task}_single_insertion",
        gripper=args.gripper,
        f_break_n=f_break,
        disturbance_seed=int(rng.integers(0, 100000)),
    )


def compute_reward(obs, outcome, contact_n: float, f_max_n: float, f_break_n: float) -> float:
    """FORGE-style reward: task completion + force penalty + ceiling violation penalty."""
    from forge_plus.envs.base_assembly_env import TaskOutcome

    if outcome == TaskOutcome.SUCCESS:
        return 10.0
    if outcome == TaskOutcome.BROKEN:
        return -5.0
    # Progress reward: insertion progress
    progress = obs.contact_step.insert_pos_mm * 0.01
    # Force penalty: penalize force use (encourage gentle contact)
    force_penalty = -0.001 * contact_n
    # Ceiling violation penalty (belt-and-suspenders signal)
    ceiling_penalty = -1.0 if contact_n > f_max_n else 0.0
    return progress + force_penalty + ceiling_penalty


def ppo_update(
    policy,
    value_net,
    actor_optimizer,
    critic_optimizer,
    rollout_obs,
    rollout_f_cmd,
    rollout_actions,
    rollout_log_probs,
    rollout_rewards,
    rollout_dones,
    rollout_values,
    args,
    device,
) -> dict:
    """One PPO update step (simplified; full implementation uses GAE advantages)."""
    obs_t = torch.tensor(rollout_obs, dtype=torch.float32, device=device)
    f_cmd_t = torch.tensor(rollout_f_cmd, dtype=torch.float32, device=device).unsqueeze(-1)
    acts_t = torch.tensor(rollout_actions, dtype=torch.float32, device=device)
    old_lp_t = torch.tensor(rollout_log_probs, dtype=torch.float32, device=device)
    rew_t = torch.tensor(rollout_rewards, dtype=torch.float32, device=device)
    done_t = torch.tensor(rollout_dones, dtype=torch.float32, device=device)
    val_t = torch.tensor(rollout_values, dtype=torch.float32, device=device)

    # Compute returns and advantages (GAE-lambda with lambda=0.95)
    returns = []
    gae = 0.0
    lam = 0.95
    next_val = 0.0
    for r, d, v in zip(reversed(rollout_rewards), reversed(rollout_dones), reversed(rollout_values)):
        delta = r + args.gamma * next_val * (1 - d) - v
        gae = delta + args.gamma * lam * (1 - d) * gae
        returns.insert(0, gae + v)
        next_val = v
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages = returns_t - val_t
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_loss = 0.0
    for _ in range(args.epochs):
        # Sample mini-batch
        idx = torch.randperm(len(rollout_obs))[:args.batch_size]
        mean, std = policy(obs_t[idx], f_cmd_t[idx])
        dist = torch.distributions.Normal(mean, std)
        new_lp = dist.log_prob(acts_t[idx]).sum(-1)
        ratio = torch.exp(new_lp - old_lp_t[idx])

        # Clipped surrogate objective
        clip = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
        actor_loss = -torch.min(ratio * advantages[idx], clip * advantages[idx]).mean()

        # Value loss
        pred_vals = value_net(obs_t[idx], f_cmd_t[idx]).squeeze(-1)
        critic_loss = F.mse_loss(pred_vals, returns_t[idx])

        # Entropy bonus
        entropy = dist.entropy().sum(-1).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        total_loss += loss.item()

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        actor_optimizer.step()
        critic_optimizer.step()

    return {"loss": total_loss / args.epochs}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    from forge_plus.skills.policy_network import PolicyConfig, ForceConditionedPolicy, ValueNetwork
    from forge_plus.skills.forge_skill import FORGESkill, SkillConfig

    policy_cfg = PolicyConfig(obs_dim=33, act_dim=7)
    policy = ForceConditionedPolicy(policy_cfg).to(device)
    value_net = ValueNetwork(policy_cfg).to(device)
    actor_opt = Adam(policy.parameters(), lr=args.lr)
    critic_opt = Adam(value_net.parameters(), lr=args.lr)

    env = build_env(args)
    skill = FORGESkill(SkillConfig(policy_cfg=policy_cfg, device=args.device, deterministic=False))
    skill.policy = policy

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = f"{args.checkpoint_dir}/{args.task}_{args.gripper}.pt"

    # Rollout buffers
    buf_obs, buf_fcmd, buf_acts, buf_lp, buf_rew, buf_done, buf_val = [], [], [], [], [], [], []

    total_steps = 0
    episode_cfg = sample_episode_config(args, rng)
    obs = env.reset(episode_cfg)
    f_cmd = float(rng.uniform(*args.f_max_range))

    t0 = time.perf_counter()
    log.info(f"Training {args.task} on {args.gripper} | device={args.device} | "
             f"total_steps={args.total_steps:,}")

    while total_steps < args.total_steps:
        obs_vec = skill._encode_obs(obs)
        skill.normalizer.update(obs_vec)
        obs_norm = skill.normalizer.normalize(obs_vec)

        obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
        fcmd_t = torch.tensor([[f_cmd / 120.0]], dtype=torch.float32, device=device)

        with torch.no_grad():
            action, log_prob = policy.get_action(obs_t, fcmd_t)
            value = value_net(obs_t, fcmd_t).squeeze()

        action_np = action.squeeze(0).cpu().numpy()
        wrench = skill._action_to_wrench(action_np, f_cmd)

        from forge_plus.control.force_clamp import ForceClamp
        clamp = ForceClamp(f_max_n=f_cmd)
        clamped_w, _ = clamp.clamp(wrench)

        obs, outcome = env.step(clamped_w)
        contact_n = env.get_contact_force_magnitude()
        reward = compute_reward(obs, outcome, contact_n, f_cmd, episode_cfg.f_break_n)
        done = env.is_done()

        buf_obs.append(obs_norm)
        buf_fcmd.append(f_cmd / 120.0)
        buf_acts.append(action_np)
        buf_lp.append(log_prob.item())
        buf_rew.append(reward)
        buf_done.append(float(done))
        buf_val.append(value.item())

        total_steps += 1

        if done or total_steps % args.rollout_len == 0:
            if len(buf_obs) >= args.batch_size:
                stats = ppo_update(
                    policy, value_net, actor_opt, critic_opt,
                    buf_obs, buf_fcmd, buf_acts, buf_lp, buf_rew, buf_done, buf_val,
                    args, device,
                )
                buf_obs.clear(); buf_fcmd.clear(); buf_acts.clear()
                buf_lp.clear(); buf_rew.clear(); buf_done.clear(); buf_val.clear()

            if done:
                episode_cfg = sample_episode_config(args, rng)
                obs = env.reset(episode_cfg)
                f_cmd = float(rng.uniform(*args.f_max_range))

        if total_steps % args.log_interval == 0:
            elapsed = time.perf_counter() - t0
            log.info(f"step={total_steps:,} elapsed={elapsed:.0f}s fps={total_steps/elapsed:.0f}")
            skill.save(ckpt_path)

    skill.save(ckpt_path)
    log.info(f"Training complete. Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
