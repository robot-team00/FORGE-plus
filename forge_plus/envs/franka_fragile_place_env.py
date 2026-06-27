"""franka_fragile_place_env.py — Task 3 full pick-and-place with fragile object handling.

Episode structure (three explicit phases):
    GRASP → TRANSPORT → PLACE

Breakage risk is active at BOTH the grasp phase (grip force crushes fragile objects)
AND the place phase (contact force during landing). The evaluator checks force against
the hidden F_break at every contact step in both phases and records which phase caused
the break in the episode metrics.

The LLM budget-setter (llama3.1:8b via Ollama) is called once at reset() and sets
F_max for the entire episode. The same ceiling applies to grip force at grasp and
contact force at place. Results are cached so identical objects don’t re-query.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from forge_plus.control.force_clamp import ForceClamp, Wrench
from forge_plus.encoding.signature_encoder import ContactStep
from forge_plus.envs.base_assembly_env import (
    BaseAssemblyEnv,
    EnvObservation,
    EpisodeConfig,
    TaskOutcome,
    TaskPhase,
)
from forge_plus.envs.object_configs import OBJECT_REGISTRY, get_object_identity


# ---------------------------------------------------------------------------
# LLM budget-setter singleton (cached per object)
# ---------------------------------------------------------------------------

_BUDGET_SETTER = None


def _get_budget_setter():
    """Return a BudgetSetter backed by llama3.1:8b (Ollama) if reachable,
    otherwise by HeuristicLLMClient (identity-based rules, no network needed).
    Result is cached globally for the lifetime of the process."""
    global _BUDGET_SETTER
    if _BUDGET_SETTER is None:
        from forge_plus.llm.budget_setter import BudgetSetter
        from forge_plus.llm.client import OpenAICompatibleClient, HeuristicLLMClient
        import urllib.request

        _ollama_url = "http://localhost:11434/v1"
        _use_ollama = False
        try:
            # Quick liveness check — if Ollama isn't running this raises immediately
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            _use_ollama = True
        except Exception:
            pass

        if _use_ollama:
            _client = OpenAICompatibleClient(
                base_url=_ollama_url,
                model="llama3.1:8b",
                max_tokens=512,
                cache=True,
                use_json_mode=False,
                max_retries=2,
                retry_delay=1.0,
            )
            print("[FrankaFragilePlaceEnv] LLM backend: llama3.1:8b via Ollama")
        else:
            _client = HeuristicLLMClient()
            print("[FrankaFragilePlaceEnv] LLM backend: HeuristicLLMClient (Ollama not reachable)")

        _BUDGET_SETTER = BudgetSetter(client=_client)
    return _BUDGET_SETTER


# ---------------------------------------------------------------------------
# Episode phase (internal — maps to TaskPhase for the base interface)
# ---------------------------------------------------------------------------

class EpisodePhase(Enum):
    GRASP_APPROACH = "grasp_approach"
    GRASP_CLOSE    = "grasp_close"
    GRASP_LIFT     = "grasp_lift"
    TRANSPORT      = "transport"
    PLACE_APPROACH = "place_approach"
    PLACE_CONTACT  = "place_contact"
    PLACE_SETTLE   = "place_settle"
    DONE           = "done"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FragilePlaceEnvConfig:
    """Tunable parameters for the full pick-and-place simulation."""

    dt_ms: float = 8.0

    # --- Grasp phase ---
    grasp_approach_steps: int = 15   # descend to object
    grasp_close_steps:    int = 8    # close gripper (grip force ramps up here)
    grasp_lift_steps:     int = 10   # lift object off surface

    # Grip force model: force ramps from 0 → grip_ramp_n_per_step * step
    # until clamped by F_max. Breakage check fires if grip_force > F_break.
    grip_ramp_n_per_step: float = 4.0   # N per close-step (high enough to crush glass in ~6 steps)
    grip_min_hold_n:      float = 1.5   # minimum force to confirm a grasp

    # Probability of a grasp-slip disturbance (unrelated to force breakage)
    grasp_slip_prob: float = 0.04

    # --- Transport phase ---
    transport_steps: int = 30

    # --- Place phase ---
    place_approach_steps:       int   = 12
    place_settle_duration_ms:   float = 200.0
    place_normal_force_stable_n: float = 2.0
    seat_confirm_base_n:        float = 2.0
    seat_confirm_mass_coeff:    float = 0.012   # N per gram of object mass
    press_ramp_n_per_step:      float = 0.9

    # Disturbances injected at reset (affect place phase only)
    edge_break_factor:          float = 0.55    # edge landing reduces effective F_break
    edge_lateral_gain:          float = 0.5
    tilt_torque_gain_nm_per_deg: float = 0.045
    tip_torque_threshold_nm:    float = 0.8
    disturbance_patience_steps: int   = 40

    # Recovery success probabilities (place phase)
    recovery_clear_prob: dict = field(default_factory=lambda: {
        "rotate_align": 0.90,
        "regrasp":      0.85,
        "abort":        1.00,
    })


# ---------------------------------------------------------------------------
# Per-episode metrics
# ---------------------------------------------------------------------------

@dataclass
class EpisodeMetrics:
    """Full per-episode result (call get_episode_metrics() after done)."""
    success:               bool  = False
    broken:                bool  = False
    broken_at_phase:       str   = "none"   # "grasp" | "place" | "none"
    failure_mode:          str   = "none"   # over_press | edge_load | tip | grasp_fail | none
    force_economy:         float = 0.0      # mean contact force (place) / F_max
    grasp_success:         bool  = False
    transport_success:     bool  = False
    peak_grip_force_n:     float = 0.0      # peak grip force during close
    peak_place_force_n:    float = 0.0      # peak axial contact force at place
    contact_duration_steps: int  = 0        # place-phase contact steps
    f_max_n:               float = 0.0      # budget set by LLM this episode


# ---------------------------------------------------------------------------
# FrankaFragilePlaceEnv
# ---------------------------------------------------------------------------

class FrankaFragilePlaceEnv(BaseAssemblyEnv):
    """Full pick-and-place environment for Task 3 (fragile place & stack).

    Phase structure
    ---------------
    1. GRASP_APPROACH : EE descends toward object (no contact).
    2. GRASP_CLOSE    : Gripper closes; grip force ramps up.
                        >>> BREAKAGE CHECK: grip_force > F_break → BROKEN (phase=grasp) <<<
    3. GRASP_LIFT     : Object lifted; no additional force risk.
    4. TRANSPORT      : EE moves horizontally to target.
    5. PLACE_APPROACH : EE descends toward surface (no contact yet).
    6. PLACE_CONTACT  : Object touches surface; contact force ramps up.
                        >>> BREAKAGE CHECK: contact_force > F_break → BROKEN (phase=place) <<<
    7. PLACE_SETTLE   : Force stabilises; success declared.

    The same LLM-set F_max ceiling is applied (via ForceClamp) to the grip
    force at step (2) and the contact force at step (6).
    """

    def __init__(
        self,
        env_cfg: FragilePlaceEnvConfig | None = None,
        use_llm: bool = True,
        llm_fallback_f_max: float = 50.0,
    ) -> None:
        self._env_cfg = env_cfg or FragilePlaceEnvConfig()
        self._use_llm = use_llm
        self._llm_fallback_f_max = llm_fallback_f_max
        self._cfg: EpisodeConfig | None = None
        self._rng = random.Random()
        self._f_max_n: float = llm_fallback_f_max
        self._clamp: ForceClamp | None = None
        self._reset_state()

    # ------------------------------------------------------------------
    # BaseAssemblyEnv interface
    # ------------------------------------------------------------------

    def reset(self, cfg: EpisodeConfig) -> EnvObservation:
        self._cfg = cfg
        self._rng = random.Random(cfg.disturbance_seed)
        self._reset_state()

        # LLM sets F_max BEFORE any control begins
        self._f_max_n = self._call_llm_budget()
        self._clamp = ForceClamp(f_max_n=self._f_max_n)

        # Inject disturbances for the place phase (not yet visible to agent)
        self._inject_place_disturbance()
        return self._make_obs()

    def step(self, wrench_cmd: Wrench) -> tuple[EnvObservation, TaskOutcome]:
        if self._done:
            return self._make_obs(), self._outcome

        self._step_count += 1
        self._phase_step += 1

        ep = self._episode_phase
        if ep == EpisodePhase.GRASP_APPROACH:
            return self._step_grasp_approach()
        elif ep == EpisodePhase.GRASP_CLOSE:
            return self._step_grasp_close(wrench_cmd)
        elif ep == EpisodePhase.GRASP_LIFT:
            return self._step_grasp_lift()
        elif ep == EpisodePhase.TRANSPORT:
            return self._step_transport()
        elif ep == EpisodePhase.PLACE_APPROACH:
            return self._step_place_approach()
        elif ep in (EpisodePhase.PLACE_CONTACT, EpisodePhase.PLACE_SETTLE):
            return self._step_place_contact(wrench_cmd)
        return self._make_obs(), self._outcome

    def apply_recovery(self, action: str, params: dict[str, Any]) -> None:
        """Recovery actions (place phase only). press_harder is forbidden."""
        if action == "press_harder":
            raise ValueError("press_harder is forbidden for fragile-place task")
        ep = self._episode_phase
        if ep not in (EpisodePhase.PLACE_CONTACT, EpisodePhase.PLACE_SETTLE,
                      EpisodePhase.DONE):
            return
        if action == "abort":
            self._done = True
            self._outcome = TaskOutcome.FAILURE_STUCK
            self._failure_mode = self._failure_mode or "aborted"
            self._episode_phase = EpisodePhase.DONE
            return
        clear_p = self._env_cfg.recovery_clear_prob.get(action, 0.0)
        if (self._edge_active or self._tilt_active) and self._rng.random() < clear_p:
            self._edge_active = self._tilt_active = False
            self._edge_offset_frac = self._tilt_deg = 0.0
            # Roll back to place approach
            self._episode_phase = EpisodePhase.PLACE_APPROACH
            self._phase_step = 0
            self._place_contact_steps = 0
            self._press_n = 0.0
            self._cmd_peak_n = 0.0
            self._settle_ms = 0.0
            self._done = False
            self._outcome = TaskOutcome.IN_PROGRESS
            self._failure_mode = None

    def observe(self) -> EnvObservation:
        return self._make_obs()

    def get_contact_force_magnitude(self) -> float:
        return self._last_contact_n

    def is_done(self) -> bool:
        return self._done

    def current_failure_mode(self) -> str | None:
        return self._failure_mode

    @property
    def current_phase(self) -> TaskPhase:
        ep = self._episode_phase
        if ep in (EpisodePhase.GRASP_APPROACH, EpisodePhase.GRASP_CLOSE,
                  EpisodePhase.GRASP_LIFT, EpisodePhase.TRANSPORT,
                  EpisodePhase.PLACE_APPROACH):
            return TaskPhase.APPROACH
        elif ep == EpisodePhase.PLACE_CONTACT:
            return TaskPhase.CONTACT
        elif ep == EpisodePhase.PLACE_SETTLE:
            return TaskPhase.SETTLE
        return TaskPhase.SEAT

    # ------------------------------------------------------------------
    # Public metrics / accessors
    # ------------------------------------------------------------------

    def get_episode_metrics(self) -> EpisodeMetrics:
        """Return per-episode metrics. Call after episode completes."""
        success = self._outcome == TaskOutcome.SUCCESS
        broken  = self._outcome == TaskOutcome.BROKEN
        forces  = self._place_force_log
        f_econ  = float(np.mean(forces) / self._f_max_n) if (forces and self._f_max_n > 0) else 0.0
        return EpisodeMetrics(
            success=success,
            broken=broken,
            broken_at_phase=self._broken_at_phase,
            failure_mode=self._failure_mode or "none",
            force_economy=f_econ,
            grasp_success=self._grasp_success,
            transport_success=self._transport_success,
            peak_grip_force_n=self._peak_grip_n,
            peak_place_force_n=self._peak_contact_n,
            contact_duration_steps=self._place_contact_steps,
            f_max_n=self._f_max_n,
        )

    @property
    def f_max_n(self) -> float:
        return self._f_max_n

    @property
    def episode_phase(self) -> EpisodePhase:
        return self._episode_phase

    # ------------------------------------------------------------------
    # Private — state reset
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._episode_phase   = EpisodePhase.GRASP_APPROACH
        self._step_count      = 0
        self._phase_step      = 0
        self._done            = False
        self._outcome         = TaskOutcome.IN_PROGRESS
        self._failure_mode: str | None = None
        self._broken_at_phase = "none"

        # Grasp state
        self._grip_n          = 0.0
        self._peak_grip_n     = 0.0
        self._grasp_success   = False
        self._transport_success = False

        # Place state
        self._place_contact_steps = 0
        self._press_n         = 0.0
        self._cmd_peak_n      = 0.0
        self._settle_ms       = 0.0
        self._last_contact_n  = 0.0
        self._peak_contact_n  = 0.0
        self._lateral_x       = 0.0
        self._lateral_y       = 0.0
        self._torque_z        = 0.0
        self._edge_active     = False
        self._edge_offset_frac = 0.0
        self._tilt_active     = False
        self._tilt_deg        = 0.0
        self._place_force_log: list[float] = []

        # Kinematics: EE starts above the object on a table
        self._ee_pos  = np.array([0.35, 0.10, 0.55], dtype=np.float64)
        self._ee_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._joint_pos = np.zeros(7, dtype=np.float64)
        self._joint_vel = np.zeros(7, dtype=np.float64)

    # ------------------------------------------------------------------
    # Private — helpers
    # ------------------------------------------------------------------

    def _seat_confirm_n(self) -> float:
        mass_g = 200.0
        if self._cfg is not None:
            obj = OBJECT_REGISTRY.get(self._cfg.object_key)
            if obj is not None:
                mass_g = obj.identity.nominal_mass_g
        return (self._env_cfg.seat_confirm_base_n
                + self._env_cfg.seat_confirm_mass_coeff * mass_g)

    def _effective_break_n(self) -> float:
        """Effective breaking force at place contact (lower on edge landing)."""
        assert self._cfg is not None
        base = self._cfg.f_break_n
        return base * self._env_cfg.edge_break_factor if self._edge_active else base

    def _inject_place_disturbance(self, p: float = 0.5) -> None:
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

    def _call_llm_budget(self) -> float:
        if not self._use_llm or self._cfg is None:
            return self._llm_fallback_f_max
        try:
            identity = get_object_identity(self._cfg.object_key)
            resp = _get_budget_setter().set_budget(identity, "task3_fragile_place")
            return resp.F_max_N
        except Exception as exc:
            print(f"[FrankaFragilePlaceEnv] LLM budget call failed ({exc}); "
                  f"using fallback {self._llm_fallback_f_max:.1f} N")
            return self._llm_fallback_f_max

    def _maybe_timeout(self) -> None:
        if self._cfg and self._step_count >= self._cfg.max_steps:
            self._failure_mode = self._failure_mode or "timeout"
            self._terminate(TaskOutcome.FAILURE_TIMEOUT)

    def _terminate(self, outcome: TaskOutcome) -> tuple[EnvObservation, TaskOutcome]:
        self._done = True
        self._outcome = outcome
        self._episode_phase = EpisodePhase.DONE
        return self._make_obs(), outcome

    # ------------------------------------------------------------------
    # Private — phase step handlers
    # ------------------------------------------------------------------

    # 1. GRASP_APPROACH — descend toward object, no force on object yet
    def _step_grasp_approach(self) -> tuple[EnvObservation, TaskOutcome]:
        cfg = self._env_cfg
        frac = self._phase_step / cfg.grasp_approach_steps
        self._ee_pos[2] = 0.55 - 0.15 * frac   # 0.55 → 0.40
        self._last_contact_n = 0.0
        self._lateral_x = self._lateral_y = self._torque_z = 0.0
        if self._phase_step >= cfg.grasp_approach_steps:
            self._episode_phase = EpisodePhase.GRASP_CLOSE
            self._phase_step = 0
        self._maybe_timeout()
        return self._make_obs(), self._outcome

    # 2. GRASP_CLOSE — gripper closes; BREAKAGE CHECK for grip force
    #
    # Grip force tracks the commanded axial force (fz = gripper squeeze intent)
    # after clamping by F_max, then ramps up at most grip_ramp_n_per_step per
    # step.  This means a controller that commands high fz during close with a
    # high LLM budget (= large F_max) can crush a fragile glass bowl.
    def _step_grasp_close(self, wrench_cmd: Wrench) -> tuple[EnvObservation, TaskOutcome]:
        cfg = self._env_cfg
        assert self._cfg is not None

        # Grip force: commanded fz (after clamp) ramps at most grip_ramp per step
        clamped_close = self._clamp.clamp(wrench_cmd)[0] if self._clamp else wrench_cmd
        target_grip = abs(clamped_close.fz)
        self._grip_n = min(
            self._grip_n + cfg.grip_ramp_n_per_step,   # can't jump instantly
            target_grip,                                 # commanded intent
            self._f_max_n,                              # hard ceiling
        )
        self._peak_grip_n = max(self._peak_grip_n, self._grip_n)

        # Report grip as lateral "contact" so the observation captures it
        self._last_contact_n = self._grip_n
        self._lateral_x = self._grip_n * 0.5   # simplified: symmetric grip
        self._lateral_y = self._grip_n * 0.5

        # >>> BREAKAGE CHECK (grasp phase) <<<
        if self._grip_n > self._cfg.f_break_n:
            self._failure_mode = "over_press"
            self._broken_at_phase = "grasp"
            return self._terminate(TaskOutcome.BROKEN)

        # Grasp slip (disturbance, unrelated to force)
        if (self._phase_step == cfg.grasp_close_steps // 2
                and self._rng.random() < cfg.grasp_slip_prob):
            self._failure_mode = "grasp_fail"
            return self._terminate(TaskOutcome.FAILURE_STUCK)

        if self._phase_step >= cfg.grasp_close_steps:
            # Confirm grasp if enough force built up
            if self._grip_n >= cfg.grip_min_hold_n:
                self._grasp_success = True
                self._episode_phase = EpisodePhase.GRASP_LIFT
                self._phase_step = 0
                self._last_contact_n = 0.0
                self._lateral_x = self._lateral_y = self._torque_z = 0.0
            else:
                self._failure_mode = "grasp_fail"
                return self._terminate(TaskOutcome.FAILURE_STUCK)
        self._maybe_timeout()
        return self._make_obs(), self._outcome

    # 3. GRASP_LIFT — lift object; no additional breakage risk
    def _step_grasp_lift(self) -> tuple[EnvObservation, TaskOutcome]:
        cfg = self._env_cfg
        frac = self._phase_step / cfg.grasp_lift_steps
        self._ee_pos[2] = 0.40 + 0.20 * frac   # 0.40 → 0.60
        self._last_contact_n = 0.0
        self._lateral_x = self._lateral_y = self._torque_z = 0.0
        if self._phase_step >= cfg.grasp_lift_steps:
            self._episode_phase = EpisodePhase.TRANSPORT
            self._phase_step = 0
        self._maybe_timeout()
        return self._make_obs(), self._outcome

    # 4. TRANSPORT — move to target; no breakage risk
    def _step_transport(self) -> tuple[EnvObservation, TaskOutcome]:
        cfg = self._env_cfg
        frac = self._phase_step / cfg.transport_steps
        start = np.array([0.35, 0.10, 0.60])
        end   = np.array([0.48, 0.00, 0.75])   # above target
        self._ee_pos = start + frac * (end - start)
        self._last_contact_n = 0.0
        self._lateral_x = self._lateral_y = self._torque_z = 0.0
        if self._phase_step >= cfg.transport_steps:
            self._transport_success = True
            self._episode_phase = EpisodePhase.PLACE_APPROACH
            self._phase_step = 0
        self._maybe_timeout()
        return self._make_obs(), self._outcome

    # 5. PLACE_APPROACH — descend toward surface; no contact yet
    def _step_place_approach(self) -> tuple[EnvObservation, TaskOutcome]:
        cfg = self._env_cfg
        frac = self._phase_step / cfg.place_approach_steps
        self._ee_pos[2] = 0.75 - 0.085 * frac   # 0.75 → 0.665
        self._last_contact_n = 0.0
        self._lateral_x = self._lateral_y = self._torque_z = 0.0
        if self._phase_step >= cfg.place_approach_steps:
            self._episode_phase = EpisodePhase.PLACE_CONTACT
            self._phase_step = 0
            self._place_contact_steps = 0
        self._maybe_timeout()
        return self._make_obs(), self._outcome

    # 6+7. PLACE_CONTACT / PLACE_SETTLE — merged (breakage checked every step)
    #
    # Settle is tracked via _settle_ms but the phase handler does NOT change.
    # Breakage runs BEFORE settle so 80N against a 22N glass bowl still breaks
    # even though press_n crosses seat_confirm on step ~5.
    def _step_place_contact(self, wrench_cmd: Wrench) -> tuple[EnvObservation, TaskOutcome]:
        cfg = self._env_cfg
        assert self._cfg is not None

        self._place_contact_steps += 1
        in_settle = self._settle_ms > 0.0

        # Force clamp then ramp
        clamped = self._clamp.clamp(wrench_cmd)[0] if self._clamp else wrench_cmd
        cmd_axial = abs(clamped.fz)
        self._cmd_peak_n = max(self._cmd_peak_n, cmd_axial)
        self._press_n = min(self._cmd_peak_n,
                            self._press_n + cfg.press_ramp_n_per_step)
        self._last_contact_n = self._press_n
        self._peak_contact_n = max(self._peak_contact_n, self._press_n)
        self._place_force_log.append(self._press_n)

        # Lateral / tilt effects
        if self._edge_active:
            self._lateral_x = cfg.edge_lateral_gain * self._press_n * self._edge_offset_frac
            self._lateral_y = 0.0
        else:
            self._lateral_x = self._lateral_y = 0.0
        self._torque_z = (
            cfg.tilt_torque_gain_nm_per_deg * self._tilt_deg
            * (self._press_n / max(self._seat_confirm_n(), 1e-6))
            if self._tilt_active else 0.0
        )

        # Update current_phase for external observers
        self._episode_phase = (EpisodePhase.PLACE_SETTLE if in_settle
                               else EpisodePhase.PLACE_CONTACT)

        # ┃━━ BREAKAGE CHECK (runs every contact step, before settle) ┃━━━━━━━━━━━
        if self._press_n > self._effective_break_n():
            self._failure_mode = (
                "edge_load" if self._edge_active
                else ("tip" if self._tilt_active else "over_press")
            )
            self._broken_at_phase = "place"
            return self._terminate(TaskOutcome.BROKEN)

        # Tip failure (sustained torque, object would tip rather than break)
        if self._tilt_active and self._torque_z > cfg.tip_torque_threshold_nm:
            self._failure_mode = "tip"
            return self._terminate(TaskOutcome.FAILURE_STUCK)

        # Disturbance patience: if disturbance can't be cleared, stuck
        if (self._edge_active or self._tilt_active):
            if self._place_contact_steps >= cfg.disturbance_patience_steps:
                self._failure_mode = "edge_load" if self._edge_active else "tip"
                return self._terminate(TaskOutcome.FAILURE_STUCK)

        # Settle check (AFTER breakage so high force still breaks)
        if self._press_n >= self._seat_confirm_n():
            self._settle_ms += cfg.dt_ms
            # Dampen displayed force toward stable value (cosmetic only)
            self._last_contact_n = min(self._press_n, cfg.place_normal_force_stable_n)
            if self._settle_ms >= cfg.place_settle_duration_ms:
                return self._terminate(TaskOutcome.SUCCESS)
        else:
            self._settle_ms = 0.0

        self._maybe_timeout()
        return self._make_obs(), self._outcome

    # ------------------------------------------------------------------
    # Private — observation builder
    # ------------------------------------------------------------------

    def _make_obs(self) -> EnvObservation:
        cs = ContactStep(
            axial_force_n=self._last_contact_n,
            lateral_force_x_n=self._lateral_x,
            lateral_force_y_n=self._lateral_y,
            torque_z_nm=self._torque_z,
            insert_pos_mm=float(self._ee_pos[2] * 1000.0),
            dt_ms=self._env_cfg.dt_ms,
        )
        return EnvObservation(
            joint_pos=self._joint_pos.copy(),
            joint_vel=self._joint_vel.copy(),
            ee_pos=self._ee_pos.copy(),
            ee_quat=self._ee_quat.copy(),
            ft_wrench=Wrench(
                self._lateral_x,
                self._lateral_y,
                self._last_contact_n,
                0.0,
                0.0,
                self._torque_z,
            ),
            contact_step=cs,
            phase=self.current_phase,
            step_count=self._step_count,
        )
