"""Force-conditioned policy network for FORGE-style skills.

Architecture: MLP with explicit F_cmd conditioning via FiLM (Feature-wise
Linear Modulation). The force budget is a first-class input, not appended
as a raw scalar — FiLM allows it to modulate the entire feature map.

Observation layout (dim = obs_dim):
  [joint_pos(7), joint_vel(7), ee_pos(3), ee_quat(4), ft_wrench(6), phase_onehot(7)]
  Total: 34 dims + task-specific additions (phase_onehot == len(TaskPhase))

Action layout (dim = act_dim):
  [delta_ee_pos(3), delta_ee_quat(4)] — delta EE pose command, 7 dims default
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PolicyConfig:
    obs_dim: int = 34
    act_dim: int = 7
    hidden_dim: int = 256
    num_layers: int = 4
    f_cmd_embed_dim: int = 64      # FiLM modulation vector size
    dropout: float = 0.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: scale + shift a feature map by a condition."""

    def __init__(self, feature_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(condition_dim, feature_dim)
        self.beta = nn.Linear(condition_dim, feature_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.gamma(cond) * x + self.beta(cond)


class ForceConditionedPolicy(nn.Module):
    """Actor network conditioned on F_cmd via FiLM.

    The force budget F_cmd is embedded into a conditioning vector that
    modulates every hidden layer. This ensures the entire feature map
    is budget-aware, not just the final layer.
    """

    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # F_cmd embedding: scalar → embedding vector
        self.f_cmd_embed = nn.Sequential(
            nn.Linear(1, cfg.f_cmd_embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.f_cmd_embed_dim, cfg.f_cmd_embed_dim),
        )

        # Trunk MLP with FiLM conditioning
        self.input_proj = nn.Linear(cfg.obs_dim, cfg.hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            for _ in range(cfg.num_layers - 1)
        ])
        self.film_layers = nn.ModuleList([
            FiLMLayer(cfg.hidden_dim, cfg.f_cmd_embed_dim)
            for _ in range(cfg.num_layers - 1)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(cfg.hidden_dim)
            for _ in range(cfg.num_layers - 1)
        ])

        # Output heads
        self.mean_head = nn.Linear(cfg.hidden_dim, cfg.act_dim)
        self.log_std = nn.Parameter(torch.full((cfg.act_dim,), -1.5))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        # Small init for output layer
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)

    def forward(
        self, obs: torch.Tensor, f_cmd: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        obs:   (batch, obs_dim) observation tensor
        f_cmd: (batch, 1) commanded max force, normalized to [0, 1]

        Returns
        -------
        mean, std: action distribution parameters
        """
        cond = self.f_cmd_embed(f_cmd)
        x = F.silu(self.input_proj(obs))

        for linear, film, norm in zip(self.layers, self.film_layers, self.norms):
            residual = x
            x = F.silu(linear(x))
            x = film(x, cond)
            x = norm(x + residual)

        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std, self.cfg.log_std_min, self.cfg.log_std_max)
        std = log_std.exp().expand_as(mean)
        return mean, std

    def get_action(
        self, obs: torch.Tensor, f_cmd: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and compute its log-probability."""
        mean, std = self.forward(obs, f_cmd)
        if deterministic:
            return mean, torch.zeros_like(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


class ValueNetwork(nn.Module):
    """Critic network for PPO — also conditioned on F_cmd."""

    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.f_cmd_embed = nn.Sequential(
            nn.Linear(1, cfg.f_cmd_embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.f_cmd_embed_dim, cfg.f_cmd_embed_dim),
        )

        layers = [nn.Linear(cfg.obs_dim, cfg.hidden_dim), nn.SiLU()]
        for _ in range(cfg.num_layers - 1):
            layers += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU()]
        layers += [nn.Linear(cfg.hidden_dim, 1)]
        self.trunk = nn.Sequential(*layers)

        self.film = FiLMLayer(cfg.hidden_dim, cfg.f_cmd_embed_dim)

    def forward(self, obs: torch.Tensor, f_cmd: torch.Tensor) -> torch.Tensor:
        cond = self.f_cmd_embed(f_cmd)
        x = obs
        for i, layer in enumerate(self.trunk):
            x = layer(x)
            if i == 1:  # after first SiLU
                x = self.film(x, cond)
        return x
