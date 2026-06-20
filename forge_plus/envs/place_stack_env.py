"""Place / stack environment for Task 3 (fragile place & stack).

No hole to insert into: the object is lowered onto a surface/stack and must
*settle*. Dominant failures are contact-force failures, and "press harder" is
maximally destructive here.

  - over_press : axial contact force exceeds the (hidden) breaking force.
  - edge_load  : lands on an edge -> lateral bias; pressing concentrates load and
                 breaks fragile parts at a LOWER axial force than a flat press.
  - tip        : a tilted seating surface induces an EE torque -> object rotates.

Success is settling: the normal contact force comes to rest below a small stable
threshold for a sustained duration.

F_break is read from EpisodeConfig (evaluator-only) and never written into the
observation / ContactStep, so the signature encoder's non-circularity guard
keeps holding. The press the controller applies is bounded by the fast-loop
clamp at F_max, so the *budget* decides whether a fragile object survives.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

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
from forge_plus.envs.object_configs import OBJECT_REGISTRY


@dataclass
class PlaceStackEnvConfig:
    """Tunable parameters for the place / stack simulation."""

    dt_ms: float = 8.0
    approach_steps: int = 12
    settle_duration_ms: float = 200.0
    normal_force_stable_n: float = 2.0

    # Minimum sustained axial force to confirm a seat. Scales with object weight;
    # for glass it is intentionally close to the breaking force (narrow band).
    seat_confirm_base_n: float = 2.0
    seat_confirm_mass_coeff_n_per_g: float = 0.012

    # An edge landing breaks at a fraction of the flat F_break (smaller contact).
    edge_break_factor: float = 0.55

    edge_lateral_gain: float = 0.5
    tilt_torque_gain_nm_per_deg: float = 0.045
    tip_torque_threshold_nm: float = 0.8

    disturbance_patience_steps: int = 40
    press_ramp_n_per_step: float = 0.9

    recovery_clear_prob: dict = field(
        default_factory=lambda: {
            "rotate_align": 0.9,
            "regrasp": 0.85,
            "retract_and_reapproach": 0.45,
            "wiggle_search": 0.15,
        }
    )


class PlaceStackEnv(BaseAssemblyEnv):
    """Deterministic place / stack environment (no GPU required)."""

    def __init__(self, env_cfg: PlaceStackEnvConfig | None = None) -> None:
        self._cfg_env = env_cfg or PlaceStackEnvConfig()
        self._cfg: EpisodeConfig | None = None
        self._rng = random.Random()
        self._reset_state()

    def _reset_state(self) -> None:
        self._step_count = 0
        self._contact_steps = 0
        self._descent_mm = 0.0
        self._press_n = 0.0
        self._cmd_peak_n = 0.0
        self._settle_ms = 0.0
        self._phase = TaskPhase.APPROACH
        self._done = False
        self._outcome = TaskOutcome.IN_PROGRESS
        self._failure_mode = None
        self._last_contact_n = 0.0
        self._peak_contact_n = 0.0
        self._lateral_x = 0.0
        self._lateral_y = 0.0
        self._torque_z = 0.0
        self._edge_active = False
        self._edge_offset_frac = 0.0
        self._tilt_active = False
        self._tilt_deg = 0.0
        self._joint_pos = np.zeros(7)
        self._joint_vel = np.zeros(7)
        self._ee_pos = np.array([0.5, 0.0, 0.30])
        self._ee_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def _seat_confirm_n(self) -> float:
        mass_g = 200.0
        if self._cfg is not None:
            obj = OBJECT_REGISTRY.get(self._cfg.object_key)
            if obj is not None:
                mass_g = obj.identity.nominal_mass_g
        return self._cfg_env.seat_confirm_base_n + (
            self._cfg_env.seat_confirm_mass_coeff_n_per_g * mass_g
        )

    def _effective_break_n(self) -> float:
        assert self._cfg is not None
        base = self._cfg.f_break_n
        return base * self._cfg_env.edge_break_factor if self._edge_active else base

    def reset(self, cfg: EpisodeConfig) -> EnvObservation:
        self._cfg = cfg
        self._rng = random.Random(cfg.disturbance_seed)
        self._reset_state()
        self._maybe_inject_disturbance()
        return self._make_obs()

    def _maybe_inject_disturbance(self, p: float = 0.5) -> None:
        self._edge_active = False
        self._tilt_active = False
        self._edge_offset_frac = 0.0
        self._tilt_deg = 0.0
        if self._rng.random() < p:
            if self._rng.random() < 0.5:
                self._edge_active = True
                self._edge_offset_frac = abs(self._rng.uniform(-4.0, 4.0)) / 4.0
            else:
                self._tilt_active = True
                self._tilt_deg = self._rng.uniform(0.0, 3.0)

    def step(self, wrench_cmd: Wrench) -> tuple[EnvObservation, TaskOutcome]:
        if self._done:
            return self._make_obs(), self._outcome

        self._step_count += 1
        cfg_env = self._cfg_env

        if self._step_count <= cfg_env.approach_steps:
            self._phase = TaskPhase.APPROACH
            self._descent_mm += 1.0
            self._last_contact_n = 0.0
            self._lateral_x = self._lateral_y = self._torque_z = 0.0
            self._maybe_timeout()
            return self._make_obs(), self._outcome

        self._phase = TaskPhase.CONTACT if self._settle_ms == 0.0 else TaskPhase.SETTLE
        self._contact_steps += 1

        cmd_axial = abs(wrench_cmd.fz)
        self._cmd_peak_n = max(self._cmd_peak_n, cmd_axial)
        self._press_n = min(self._cmd_peak_n, self._press_n + cfg_env.press_ramp_n_per_step)

        self._last_contact_n = self._press_n
        self._peak_contact_n = max(self._peak_contact_n, self._press_n)
        if self._edge_active:
            self._lateral_x = cfg_env.edge_lateral_gain * self._press_n * self._edge_offset_frac
            self._lateral_y = 0.0
        else:
            self._lateral_x = self._lateral_y = 0.0
        if self._tilt_active:
            self._torque_z = cfg_env.tilt_torque_gain_nm_per_deg * self._tilt_deg * (
                self._press_n / max(self._seat_confirm_n(), 1e-6)
            )
        else:
            self._torque_z = 0.0

        if self._press_n > self._effective_break_n():
            self._failure_mode = (
                "edge_load" if self._edge_active
                else ("tip" if self._tilt_active else "over_press")
            )
            return self._terminate(TaskOutcome.BROKEN)

        if self._tilt_active and self._torque_z > cfg_env.tip_torque_threshold_nm:
            self._failure_mode = "tip"
            return self._terminate(TaskOutcome.FAILURE_STUCK)

        if self._edge_active or self._tilt_active:
            if self._contact_steps >= cfg_env.disturbance_patience_steps:
                self._failure_mode = "edge_load" if self._edge_active else "tip"
                return self._terminate(TaskOutcome.FAILURE_STUCK)
            self._maybe_timeout()
            return self._make_obs(), self._outcome

        if self._press_n >= self._seat_confirm_n():
            self._settle_ms += cfg_env.dt_ms
            self._last_contact_n = min(self._press_n, cfg_env.normal_force_stable_n)
            if self._settle_ms >= cfg_env.settle_duration_ms:
                return self._terminate(TaskOutcome.SUCCESS)
        else:
            self._settle_ms = 0.0

        self._maybe_timeout()
        return self._make_obs(), self._outcome

    def _maybe_timeout(self) -> None:
        if self._cfg and self._step_count >= self._cfg.max_steps:
            if self._failure_mode is None:
                self._failure_mode = "under_seat"
            self._terminate(TaskOutcome.FAILURE_TIMEOUT)

    def _terminate(self, outcome: TaskOutcome) -> tuple[EnvObservation, TaskOutcome]:
        self._done = True
        self._outcome = outcome
        return self._make_obs(), outcome

    def apply_recovery(self, action: str, params: dict) -> None:
        clear_p = self._cfg_env.recovery_clear_prob.get(action, 0.0)
        if (self._edge_active or self._tilt_active) and self._rng.random() < clear_p:
            self._edge_active = False
            self._tilt_active = False
            self._edge_offset_frac = 0.0
            self._tilt_deg = 0.0
        self._step_count = 0
        self._contact_steps = 0
        self._descent_mm = 0.0
        self._press_n = 0.0
        self._cmd_peak_n = 0.0
        self._settle_ms = 0.0
        self._phase = TaskPhase.APPROACH
        self._done = False
        self._outcome = TaskOutcome.IN_PROGRESS
        self._failure_mode = None

    def observe(self) -> EnvObservation:
        return self._make_obs()

    def get_contact_force_magnitude(self) -> float:
        return self._last_contact_n

    def is_done(self) -> bool:
        return self._done

    def current_failure_mode(self):
        return self._failure_mode

    @property
    def current_phase(self) -> TaskPhase:
        return self._phase

    def _make_obs(self) -> EnvObservation:
        cs = ContactStep(
            axial_force_n=self._last_contact_n,
            lateral_force_x_n=self._lateral_x,
            lateral_force_y_n=self._lateral_y,
            torque_z_nm=self._torque_z,
            insert_pos_mm=self._descent_mm,
            dt_ms=self._cfg_env.dt_ms,
        )
        return EnvObservation(
            joint_pos=self._joint_pos.copy(),
            joint_vel=self._joint_vel.copy(),
            ee_pos=self._ee_pos.copy(),
            ee_quat=self._ee_quat.copy(),
            ft_wrench=Wrench(
                self._lateral_x, self._lateral_y, self._last_contact_n,
                0.0, 0.0, self._torque_z,
            ),
            contact_step=cs,
            phase=self._phase,
            step_count=self._step_count,
        )
