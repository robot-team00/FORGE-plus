"""FORGE-style force-conditioned RL skill.

Wraps the policy network to handle:
  - Observation preprocessing and normalization
  - F_cmd normalization (force ceiling → [0, 1] for the network)
  - Action post-processing (delta EE pose → Wrench)
  - Checkpoint loading / saving

The skill requests forces ≤ F_max; the fast-loop clamp enforces the ceiling
regardless of what the skill outputs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from forge_plus.control.force_clamp import Wrench
from forge_plus.envs.base_assembly_env import EnvObservation, TaskPhase
from forge_plus.skills.policy_network import ForceConditionedPolicy, PolicyConfig


@dataclass
class SkillConfig:
    policy_cfg: PolicyConfig = None
    checkpoint_path: str | None = None
    f_max_normalization: float = 120.0   # global hard cap used for [0,1] normalization
    device: str = "cpu"
    deterministic: bool = False          # True for evaluation, False for training

    def __post_init__(self) -> None:
        if self.policy_cfg is None:
            self.policy_cfg = PolicyConfig()


class ObsNormalizer:
    """Online running mean/std normalizer (Welford's algorithm)."""

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = 0
        self.eps = eps

    def update(self, x: np.ndarray) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += (delta * (x - self.mean) - self.var) / self.count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + self.eps)

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.array(d["mean"])
        self.var = np.array(d["var"])
        self.count = d["count"]


class FORGESkill:
    """Force-conditioned FORGE-style skill.

    Inputs at each step:
      obs:   EnvObservation from the environment
      f_cmd: float (N) — the budget ceiling; network receives normalized version

    Output:
      Wrench — commanded EE wrench (to be passed through the force clamp)
    """

    def __init__(self, cfg: SkillConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.policy = ForceConditionedPolicy(cfg.policy_cfg).to(self.device)
        self.normalizer = ObsNormalizer(cfg.policy_cfg.obs_dim)

        if cfg.checkpoint_path and Path(cfg.checkpoint_path).exists():
            self.load(cfg.checkpoint_path)

        self.policy.eval() if cfg.deterministic else self.policy.train()

    def act(self, obs: EnvObservation, f_cmd: float) -> Wrench:
        """Compute a wrench command from the observation and force budget."""
        obs_vec = self._encode_obs(obs)
        obs_norm = self.normalizer.normalize(obs_vec)

        obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        f_cmd_norm = torch.tensor(
            [[f_cmd / self.cfg.f_max_normalization]], dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            action, _ = self.policy.get_action(
                obs_t, f_cmd_norm, deterministic=self.cfg.deterministic
            )

        a = action.squeeze(0).cpu().numpy()
        return self._action_to_wrench(a, f_cmd)

    def _encode_obs(self, obs: EnvObservation) -> np.ndarray:
        """Flatten observation into a fixed-size vector."""
        # Deterministic phase index. NOTE: do NOT use hash(phase.value) — str
        # hashing is salted per process (PYTHONHASHSEED), which would make the
        # encoding non-reproducible and inconsistent between training and eval,
        # and a modulo into too few slots collides distinct phases.
        phases = list(TaskPhase)
        phase_onehot = np.zeros(len(phases))
        phase_onehot[phases.index(obs.phase)] = 1.0

        ft = obs.ft_wrench
        ft_vec = np.array([ft.fx, ft.fy, ft.fz, ft.tx, ft.ty, ft.tz])

        obs_vec = np.concatenate([
            obs.joint_pos,           # 7
            obs.joint_vel,           # 7
            obs.ee_pos,              # 3
            obs.ee_quat,             # 4
            ft_vec,                  # 6
            phase_onehot,            # len(TaskPhase) == 7
        ])
        return obs_vec.astype(np.float32)

    def _action_to_wrench(self, action: np.ndarray, f_cmd: float) -> Wrench:
        """Convert policy action (delta EE pose) to a wrench command.

        The action is a delta pose; we map the insertion-axis delta to a
        force command proportional to f_cmd. Lateral deltas become lateral
        forces at reduced gain.
        """
        insertion_force = np.clip(action[2], -1.0, 1.0) * f_cmd
        lateral_x = np.clip(action[0], -1.0, 1.0) * f_cmd * 0.5
        lateral_y = np.clip(action[1], -1.0, 1.0) * f_cmd * 0.5
        torque_z = np.clip(action[6] if len(action) > 6 else 0.0, -1.0, 1.0) * 2.0
        return Wrench(
            fx=float(lateral_x),
            fy=float(lateral_y),
            fz=float(insertion_force),
            tx=0.0,
            ty=0.0,
            tz=float(torque_z),
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "normalizer": self.normalizer.state_dict(),
                "policy_cfg": self.cfg.policy_cfg,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        if "normalizer" in ckpt:
            self.normalizer.load_state_dict(ckpt["normalizer"])
