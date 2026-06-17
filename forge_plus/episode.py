"""Force-Budgeted Recovery — closed-loop episode runner.

Implements the algorithm from §7 of the proposal exactly:

    def run_episode(object, task, K_max):
        F_max = clamp(LLM_budget_setter(object, task), 0, GLOBAL_HARD_CAP)
        for attempt in range(K_max):
            F_cmd = F_max
            while not done(task):
                a = skill.act(obs, F_cmd)
                w = controller(a)
                w = force_clamp(w, F_max)           # SAFETY — fast loop
                obs, contact = sim.step(w)
                if contact.peak > F_break: BROKEN   # evaluator-only
                history.append(obs, contact)
            if success(task): return SUCCESS
            sig = encode_signature(history[-N:])
            assert "F_break" not in sig              # non-circularity guard
            r = LLM_recovery_selector(sig, F_max, attempt, MENU)
            apply_recovery(r)                        # F_max unchanged
        return FAIL_NO_ATTEMPTS_LEFT

The episode runner is the integration seam — it owns the slow/fast boundary,
calls the two LLM components exactly as described, and enforces all invariants.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from forge_plus.control.force_clamp import ForceClamp, Wrench
from forge_plus.encoding.signature_encoder import ContactStep, SignatureEncoder
from forge_plus.envs.base_assembly_env import BaseAssemblyEnv, EpisodeConfig, TaskOutcome
from forge_plus.envs.object_configs import OBJECT_REGISTRY
from forge_plus.llm.budget_setter import BudgetResponse, BudgetSetter, ObjectIdentity
from forge_plus.llm.recovery_selector import ForceSignature, RecoverySelector
from forge_plus.recovery.recovery_actions import RecoveryActionExecutor, RecoveryStatus
from forge_plus.skills.forge_skill import FORGESkill

log = logging.getLogger(__name__)

GLOBAL_HARD_CAP_N: float = 120.0
SIGNATURE_WINDOW_STEPS: int = 50    # last N steps used for signature


class EpisodeTermination(str, Enum):
    SUCCESS = "SUCCESS"
    BROKEN = "BROKEN"
    FAIL_NO_ATTEMPTS_LEFT = "FAIL_NO_ATTEMPTS_LEFT"
    FAIL_ABORTED = "FAIL_ABORTED"
    FAIL_TIMEOUT = "FAIL_TIMEOUT"


@dataclass
class AttemptRecord:
    attempt_idx: int
    steps: int
    outcome: TaskOutcome
    recovery_action: str | None
    peak_contact_n: float
    net_insert_mm: float
    recovery_latency_s: float = 0.0


@dataclass
class EpisodeResult:
    """Full record of one episode — used by the evaluation framework."""

    # Identification
    object_key: str
    task: str
    gripper: str
    episode_seed: int

    # Outcome
    termination: EpisodeTermination
    total_attempts: int
    total_steps: int

    # Force budget
    f_max_n: float
    f_break_n: float                      # EVALUATOR ONLY — hidden from agent
    budget_confidence: float
    budget_rationale: str
    safety_margin_n: float = 0.0          # f_break - f_max (positive = safe)

    # Force tracking
    peak_contact_n: float = 0.0
    mean_contact_n: float = 0.0
    clamp_overshoot_mean_n: float = 0.0
    clamp_overshoot_max_n: float = 0.0

    # Per-attempt records
    attempts: list[AttemptRecord] = field(default_factory=list)

    # Flags
    broke: bool = False
    succeeded: bool = False
    over_budget: bool = False             # F_max > F_break (wrong budget)

    # Timing
    wall_time_s: float = 0.0

    def compute_derived(self) -> None:
        self.broke = self.termination == EpisodeTermination.BROKEN
        self.succeeded = self.termination == EpisodeTermination.SUCCESS
        self.safety_margin_n = self.f_break_n - self.f_max_n
        self.over_budget = self.f_max_n > self.f_break_n


@dataclass
class EpisodeRunner:
    """Orchestrates the full force-budgeted recovery episode.

    Holds references to all components but keeps them cleanly separated:
      - budget_setter: slow LLM call #1 (once per episode)
      - recovery_selector: slow LLM call #2 (once per failure)
      - signature_encoder: no-image F/T → text
      - force_clamp: fast loop safety (not the LLM)
      - skill: FORGE-style force-conditioned policy
      - env: simulation environment
      - recovery_executor: dispatches recovery primitives
    """

    budget_setter: BudgetSetter
    recovery_selector: RecoverySelector
    signature_encoder: SignatureEncoder
    skill: FORGESkill
    env: BaseAssemblyEnv
    recovery_executor: RecoveryActionExecutor
    k_max: int = 5                          # max attempts before FAIL_NO_ATTEMPTS_LEFT
    signature_window: int = SIGNATURE_WINDOW_STEPS
    verbose: bool = False

    def run(self, episode_cfg: EpisodeConfig) -> EpisodeResult:
        t_start = time.perf_counter()

        obj_cfg = OBJECT_REGISTRY.get(episode_cfg.object_key)
        if obj_cfg is None:
            raise ValueError(f"Unknown object key: {episode_cfg.object_key}")
        obj_identity = obj_cfg.identity

        # ----------------------------------------------------------------
        # Slow call #1: set per-object force budget (identity only, no F_break)
        # ----------------------------------------------------------------
        budget: BudgetResponse = self.budget_setter.set_budget(
            obj_identity, episode_cfg.task_name
        )
        f_max_n = budget.F_max_N

        if self.verbose:
            log.info(
                f"[Episode] object={episode_cfg.object_key} task={episode_cfg.task_name} "
                f"gripper={episode_cfg.gripper} F_max={f_max_n:.1f}N "
                f"(confidence={budget.confidence:.2f})"
            )

        force_clamp = ForceClamp(
            f_max_n=f_max_n,
            per_axis_n=budget.per_axis_N or None,
            global_hard_cap_n=GLOBAL_HARD_CAP_N,
        )

        result = EpisodeResult(
            object_key=episode_cfg.object_key,
            task=episode_cfg.task_name,
            gripper=episode_cfg.gripper,
            episode_seed=episode_cfg.disturbance_seed,
            termination=EpisodeTermination.FAIL_NO_ATTEMPTS_LEFT,
            total_attempts=0,
            total_steps=0,
            f_max_n=f_max_n,
            f_break_n=episode_cfg.f_break_n,   # evaluator only
            budget_confidence=budget.confidence,
            budget_rationale=budget.rationale,
        )

        history: list[ContactStep] = []
        all_contact_forces: list[float] = []
        all_wrench_commands: list[Wrench] = []

        obs = self.env.reset(episode_cfg)

        # ----------------------------------------------------------------
        # Attempt loop
        # ----------------------------------------------------------------
        for attempt_idx in range(self.k_max):
            result.total_attempts += 1
            f_cmd = f_max_n              # skill may internally request ≤ F_max
            attempt_steps = 0
            attempt_peak = 0.0
            attempt_start_insert = obs.contact_step.insert_pos_mm

            # -- Fast control loop --
            while not self.env.is_done():
                raw_wrench = self.skill.act(obs, f_cmd)
                clamped_wrench, _ = force_clamp.clamp(raw_wrench)
                obs, outcome = self.env.step(clamped_wrench)

                contact_f = self.env.get_contact_force_magnitude()
                history.append(obs.contact_step)
                all_contact_forces.append(contact_f)
                all_wrench_commands.append(clamped_wrench)
                attempt_peak = max(attempt_peak, contact_f)
                attempt_steps += 1
                result.total_steps += 1

                # Evaluator-only breakage — the BROKEN outcome is set inside env.step()
                if outcome == TaskOutcome.BROKEN:
                    net_insert = obs.contact_step.insert_pos_mm - attempt_start_insert
                    result.attempts.append(AttemptRecord(
                        attempt_idx=attempt_idx,
                        steps=attempt_steps,
                        outcome=outcome,
                        recovery_action=None,
                        peak_contact_n=attempt_peak,
                        net_insert_mm=net_insert,
                    ))
                    result.termination = EpisodeTermination.BROKEN
                    result.peak_contact_n = float(max(all_contact_forces, default=0.0))
                    result.mean_contact_n = float(
                        sum(all_contact_forces) / max(len(all_contact_forces), 1)
                    )
                    fidelity = ForceClamp.measure_clamp_fidelity(
                        all_wrench_commands, all_contact_forces, f_max_n
                    )
                    result.clamp_overshoot_mean_n = fidelity["mean_overshoot_n"]
                    result.clamp_overshoot_max_n = fidelity["max_overshoot_n"]
                    result.wall_time_s = time.perf_counter() - t_start
                    result.compute_derived()
                    return result

            # Check success
            if outcome == TaskOutcome.SUCCESS:
                net_insert = obs.contact_step.insert_pos_mm - attempt_start_insert
                result.attempts.append(AttemptRecord(
                    attempt_idx=attempt_idx,
                    steps=attempt_steps,
                    outcome=outcome,
                    recovery_action=None,
                    peak_contact_n=attempt_peak,
                    net_insert_mm=net_insert,
                ))
                result.termination = EpisodeTermination.SUCCESS
                break

            if outcome == TaskOutcome.FAILURE_TIMEOUT:
                result.termination = EpisodeTermination.FAIL_TIMEOUT
                break

            # ----------------------------------------------------------------
            # Failure: encode force signature (no images, no F_break)
            # ----------------------------------------------------------------
            window = history[-self.signature_window:]
            if not window:
                window = history or []

            if window:
                sig: ForceSignature = self.signature_encoder.encode(
                    window, gripper=episode_cfg.gripper
                )
            else:
                # No contact history yet — use a zero signature
                sig = ForceSignature(
                    peak_axial_N=0.0,
                    net_insert_mm=0.0,
                    axial_rising=False,
                    lateral_bias="none",
                    contact_persist_ms=0.0,
                    slip_events=0,
                )

            # ----------------------------------------------------------------
            # Slow call #2: pick recovery from the fixed menu
            # ----------------------------------------------------------------
            t_rec = time.perf_counter()
            recovery = self.recovery_selector.select(
                signature=sig,
                f_max_n=f_max_n,
                attempt=attempt_idx,
                subphase=obs.phase.value,
                gripper=episode_cfg.gripper,
            )
            rec_latency = time.perf_counter() - t_rec

            # F_max is immutable — the selector's keep_F_max_N is already
            # overwritten to f_max_n inside RecoverySelector.select()
            assert recovery.keep_F_max_N == f_max_n, (
                f"Budget integrity violation: LLM returned {recovery.keep_F_max_N}, "
                f"expected {f_max_n}"
            )

            net_insert = obs.contact_step.insert_pos_mm - attempt_start_insert
            result.attempts.append(AttemptRecord(
                attempt_idx=attempt_idx,
                steps=attempt_steps,
                outcome=outcome,
                recovery_action=recovery.action,
                peak_contact_n=attempt_peak,
                net_insert_mm=net_insert,
                recovery_latency_s=rec_latency,
            ))

            if self.verbose:
                log.info(
                    f"[Recovery] attempt={attempt_idx} action={recovery.action} "
                    f"sig_peak={sig.peak_axial_N:.1f}N lat={sig.lateral_bias} "
                    f"F_max={f_max_n:.1f}N (unchanged)"
                )

            rec_result = self.recovery_executor.execute(
                env=self.env,
                action=recovery.action,
                params=recovery.params,
                f_max_n=f_max_n,   # passed through, never modified
            )

            if rec_result.status == RecoveryStatus.ABORTED:
                result.termination = EpisodeTermination.FAIL_ABORTED
                break

            # Reset environment state for next attempt
            obs = self.env.reset(episode_cfg) if attempt_idx < self.k_max - 1 else obs

        # ----------------------------------------------------------------
        # Populate final metrics
        # ----------------------------------------------------------------
        result.peak_contact_n = float(max(all_contact_forces, default=0.0))
        result.mean_contact_n = float(
            sum(all_contact_forces) / max(len(all_contact_forces), 1)
        )
        fidelity = ForceClamp.measure_clamp_fidelity(
            all_wrench_commands, all_contact_forces, f_max_n
        )
        result.clamp_overshoot_mean_n = fidelity["mean_overshoot_n"]
        result.clamp_overshoot_max_n = fidelity["max_overshoot_n"]
        result.wall_time_s = time.perf_counter() - t_start
        result.compute_derived()
        return result
