"""Lightweight mock assembly environment for unit testing and CI.

Simulates the interface of BaseAssemblyEnv without requiring Isaac Lab or GPU.
Physics is minimal: linear insertion with configurable friction and jam probability.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from forge_plus.control.force_clamp import Wrench
from forge_plus.encoding.signature_encoder import ContactStep
from forge_plus.envs.base_assembly_env import (
    BaseAssemblyEnv,
    EnvObservation,
    EpisodeConfig,
    TaskOutcome,
    TaskPhase,
)


@dataclass
class MockEnvConfig:
    """Tunable parameters for the mock simulation."""

    jam_probability: float = 0.3       # probability of a jam forming on each attempt
    jam_recovery_actions: list[str] = field(
        default_factory=lambda: ["retract_and_reapproach", "wiggle_search", "rotate_align"]
    )
    friction_coefficient: float = 0.3
    dt_ms: float = 8.0                  # ~120 Hz
    success_insertion_mm: float = 10.0  # displacement to count as success
    near_contact_velocity_m_s: float = 0.01


class MockAssemblyEnv(BaseAssemblyEnv):
    """Mock environment — deterministic physics, no GPU required."""

    def __init__(self, mock_cfg: MockEnvConfig | None = None) -> None:
        self._mock_cfg = mock_cfg or MockEnvConfig()
        self._cfg: EpisodeConfig | None = None
        self._rng = random.Random()
        self._reset_state()

    def _reset_state(self) -> None:
        self._step_count = 0
        self._insert_pos_mm = 0.0
        self._jammed = False
        self._jam_axis = 0.0
        self._contact_active = False
        self._phase = TaskPhase.APPROACH
        self._done = False
        self._outcome = TaskOutcome.IN_PROGRESS
        self._last_contact_n = 0.0
        self._peak_contact_n = 0.0
        self._joint_pos = np.zeros(7)
        self._joint_vel = np.zeros(7)
        self._ee_pos = np.array([0.5, 0.0, 0.3])
        self._ee_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def reset(self, cfg: EpisodeConfig) -> EnvObservation:
        self._cfg = cfg
        self._rng = random.Random(cfg.disturbance_seed)
        self._reset_state()
        self._jammed = self._rng.random() < self._mock_cfg.jam_probability
        self._jam_axis = self._rng.uniform(-5.0, 5.0)
        return self._make_obs()

    def step(self, wrench_cmd: Wrench) -> tuple[EnvObservation, TaskOutcome]:
        if self._done:
            return self._make_obs(), self._outcome

        self._step_count += 1
        axial_f = wrench_cmd.fz   # insertion along z
        lateral_f = math.sqrt(wrench_cmd.fx**2 + wrench_cmd.fy**2)

        # Compute contact force (simplified: axial + jam resistance)
        jam_resist = abs(self._jam_axis) * 3.0 if self._jammed else 0.0
        contact_n = abs(axial_f) + jam_resist
        self._last_contact_n = contact_n
        self._peak_contact_n = max(self._peak_contact_n, contact_n)

        # Check hidden breakage (evaluator only)
        if self._cfg and contact_n > self._cfg.f_break_n:
            self._done = True
            self._outcome = TaskOutcome.BROKEN
            return self._make_obs(), self._outcome

        # Update insertion progress
        if not self._jammed:
            friction_force = self._mock_cfg.friction_coefficient * abs(axial_f)
            net = max(0.0, abs(axial_f) - friction_force)
            self._insert_pos_mm += net * 0.002  # mm per N·step (mock scale)

        if self._phase == TaskPhase.APPROACH and self._insert_pos_mm > 0.5:
            self._phase = TaskPhase.INSERT
            self._contact_active = True

        if self._insert_pos_mm >= self._mock_cfg.success_insertion_mm:
            self._done = True
            self._outcome = TaskOutcome.SUCCESS
            return self._make_obs(), self._outcome

        # Detect stuck (no progress in 200 steps while force applied)
        if self._step_count > 200 and self._insert_pos_mm < 0.1 and abs(axial_f) > 1.0:
            self._done = True
            self._outcome = TaskOutcome.FAILURE_STUCK
            return self._make_obs(), self._outcome

        if self._cfg and self._step_count >= self._cfg.max_steps:
            self._done = True
            self._outcome = TaskOutcome.FAILURE_TIMEOUT

        return self._make_obs(), self._outcome

    def apply_recovery(self, action: str, params: dict[str, Any]) -> None:
        """Apply recovery — some actions fix a jam, others don't."""
        if action in self._mock_cfg.jam_recovery_actions:
            # Probabilistic jam clearance
            if self._jammed and self._rng.random() < 0.6:
                self._jammed = False
        # Reset progress for retract actions
        if action == "retract_and_reapproach":
            self._insert_pos_mm = max(0.0, self._insert_pos_mm - 2.0)
            self._phase = TaskPhase.APPROACH
            self._contact_active = False
        self._done = False
        self._outcome = TaskOutcome.IN_PROGRESS
        self._step_count = 0

    def observe(self) -> EnvObservation:
        return self._make_obs()

    def get_contact_force_magnitude(self) -> float:
        return self._last_contact_n

    def is_done(self) -> bool:
        return self._done

    @property
    def current_phase(self) -> TaskPhase:
        return self._phase

    def _make_obs(self) -> EnvObservation:
        lat_bias = self._jam_axis if self._jammed else 0.0
        cs = ContactStep(
            axial_force_n=self._last_contact_n,
            lateral_force_x_n=lat_bias,
            lateral_force_y_n=0.0,
            torque_z_nm=0.0,
            insert_pos_mm=self._insert_pos_mm,
            dt_ms=self._mock_cfg.dt_ms,
        )
        return EnvObservation(
            joint_pos=self._joint_pos.copy(),
            joint_vel=self._joint_vel.copy(),
            ee_pos=self._ee_pos.copy(),
            ee_quat=self._ee_quat.copy(),
            ft_wrench=Wrench(lat_bias, 0.0, self._last_contact_n, 0.0, 0.0, 0.0),
            contact_step=cs,
            phase=self._phase,
            step_count=self._step_count,
        )
