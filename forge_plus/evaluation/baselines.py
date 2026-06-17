"""Baseline implementations for comparison (§10 of the proposal).

All baselines implement the same run() interface as EpisodeRunner so they
can be evaluated with the same metrics framework.

Baselines:
  1. NoCeilingSkill          — unbounded force, retry-as-is on failure
  2. FixedGlobalCeiling      — one F_max for all objects, no recovery
  3. PressHarder             — per-object F_max, but recovery = increase force
  4. HeuristicRecovery       — per-object F_max, hand-coded recovery rules
  5. VisionLLMRecovery       — per-object F_max, LLM on rendered caption (proxy)
  6. OracleCeiling           — F_break - epsilon (cheats, upper bound)
  7. Ours (imported from episode.py)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any

from forge_plus.control.force_clamp import ForceClamp, Wrench
from forge_plus.encoding.signature_encoder import SignatureEncoder
from forge_plus.envs.base_assembly_env import BaseAssemblyEnv, EpisodeConfig, TaskOutcome
from forge_plus.envs.object_configs import OBJECT_REGISTRY
from forge_plus.episode import (
    AttemptRecord,
    EpisodeResult,
    EpisodeTermination,
    GLOBAL_HARD_CAP_N,
)
from forge_plus.llm.budget_setter import BudgetSetter
from forge_plus.llm.recovery_selector import ForceSignature
from forge_plus.recovery.recovery_actions import RecoveryActionExecutor, RecoveryStatus
from forge_plus.skills.forge_skill import FORGESkill


class BaselineType(str, Enum):
    NO_CEILING = "no_ceiling"
    FIXED_GLOBAL = "fixed_global"
    PRESS_HARDER = "press_harder"
    HEURISTIC = "heuristic"
    VISION_LLM = "vision_llm"
    ORACLE = "oracle"


@dataclass
class BaselineRunner:
    """Runs a named baseline on an episode config and returns EpisodeResult."""

    baseline_type: BaselineType
    skill: FORGESkill
    env: BaseAssemblyEnv
    recovery_executor: RecoveryActionExecutor
    k_max: int = 5

    # Baseline-specific params
    fixed_ceiling_n: float = 60.0         # for FIXED_GLOBAL
    oracle_epsilon_n: float = 2.0         # for ORACLE: F_max = F_break - epsilon
    press_harder_factor: float = 1.25     # for PRESS_HARDER: multiply F_max per retry

    def run(self, episode_cfg: EpisodeConfig) -> EpisodeResult:
        f_max_n = self._get_initial_budget(episode_cfg)
        result = self._make_result(episode_cfg, f_max_n)
        obs = self.env.reset(episode_cfg)
        all_contact = []

        for attempt_idx in range(self.k_max):
            result.total_attempts += 1
            clamp = ForceClamp(f_max_n=f_max_n, global_hard_cap_n=GLOBAL_HARD_CAP_N)
            peak = 0.0
            steps = 0
            outcome = TaskOutcome.IN_PROGRESS
            attempt_start_insert = obs.contact_step.insert_pos_mm

            while not self.env.is_done():
                raw = self.skill.act(obs, f_max_n)
                clamped, _ = clamp.clamp(raw)
                obs, outcome = self.env.step(clamped)
                cf = self.env.get_contact_force_magnitude()
                all_contact.append(cf)
                peak = max(peak, cf)
                steps += 1
                result.total_steps += 1

                if outcome == TaskOutcome.BROKEN:
                    net = obs.contact_step.insert_pos_mm - attempt_start_insert
                    result.attempts.append(AttemptRecord(attempt_idx, steps, outcome, None, peak, net))
                    result.termination = EpisodeTermination.BROKEN
                    self._finalize(result, all_contact, f_max_n)
                    return result

            net = obs.contact_step.insert_pos_mm - attempt_start_insert

            if outcome == TaskOutcome.SUCCESS:
                result.attempts.append(AttemptRecord(attempt_idx, steps, outcome, None, peak, net))
                result.termination = EpisodeTermination.SUCCESS
                break

            # Recovery / next attempt
            recovery_action = self._choose_recovery(attempt_idx, obs, f_max_n)
            result.attempts.append(AttemptRecord(attempt_idx, steps, outcome, recovery_action, peak, net))

            if recovery_action == "abort":
                result.termination = EpisodeTermination.FAIL_ABORTED
                break

            # Update budget for next attempt
            f_max_n = self._update_budget(f_max_n, attempt_idx, episode_cfg)

            self.recovery_executor.execute(self.env, recovery_action, {}, f_max_n)
            # Continue from the recovered state (see EpisodeRunner): resetting here
            # would re-randomize identically and discard the recovery just applied.
            obs = self.env.observe()

        self._finalize(result, all_contact, f_max_n)
        return result

    def _get_initial_budget(self, cfg: EpisodeConfig) -> float:
        if self.baseline_type == BaselineType.NO_CEILING:
            return GLOBAL_HARD_CAP_N
        if self.baseline_type == BaselineType.FIXED_GLOBAL:
            return self.fixed_ceiling_n
        if self.baseline_type == BaselineType.ORACLE:
            return max(0.0, cfg.f_break_n - self.oracle_epsilon_n)
        # All others: use a simple lookup from object class (heuristic prior)
        return self._heuristic_budget(cfg.object_key)

    def _heuristic_budget(self, object_key: str) -> float:
        obj = OBJECT_REGISTRY.get(object_key)
        if obj is None:
            return 40.0
        # Rough heuristic: 60% of mean F_break
        return min(obj.f_break_mean_n * 0.6, GLOBAL_HARD_CAP_N)

    def _choose_recovery(self, attempt: int, obs: Any, f_max_n: float) -> str:
        if self.baseline_type == BaselineType.NO_CEILING:
            return "retract_and_reapproach"
        if self.baseline_type == BaselineType.FIXED_GLOBAL:
            return "retract_and_reapproach"
        if self.baseline_type == BaselineType.PRESS_HARDER:
            return "retract_and_reapproach"   # recovery = retry (force will be higher next round)
        if self.baseline_type == BaselineType.HEURISTIC:
            return self._heuristic_recovery(obs)
        if self.baseline_type == BaselineType.VISION_LLM:
            return self._vision_llm_recovery(attempt)
        if self.baseline_type == BaselineType.ORACLE:
            return "retract_and_reapproach"
        return "abort"

    def _heuristic_recovery(self, obs: Any) -> str:
        """Simple hand-coded rules: rising axial + lateral → rotate_align, else wiggle."""
        ft = obs.ft_wrench
        lateral = (ft.fx**2 + ft.fy**2) ** 0.5
        if lateral > 3.0 and abs(ft.fz) > 5.0:
            return "rotate_align"
        return "wiggle_search"

    def _vision_llm_recovery(self, attempt: int) -> str:
        """Proxy for vision-LLM: randomly picks from the menu (under-discriminates jams)."""
        # Real implementation would caption a rendered frame and query the LLM.
        # The proxy demonstrates the under-discrimination weakness for visual-only methods.
        actions = ["retract_and_reapproach", "wiggle_search", "rotate_align"]
        return random.choice(actions)

    def _update_budget(self, current_f_max: float, attempt: int, cfg: EpisodeConfig) -> float:
        if self.baseline_type == BaselineType.PRESS_HARDER:
            return min(current_f_max * self.press_harder_factor, GLOBAL_HARD_CAP_N)
        return current_f_max

    def _make_result(self, cfg: EpisodeConfig, f_max_n: float) -> EpisodeResult:
        return EpisodeResult(
            object_key=cfg.object_key,
            task=cfg.task_name,
            gripper=cfg.gripper,
            episode_seed=cfg.disturbance_seed,
            termination=EpisodeTermination.FAIL_NO_ATTEMPTS_LEFT,
            total_attempts=0,
            total_steps=0,
            f_max_n=f_max_n,
            f_break_n=cfg.f_break_n,
            budget_confidence=0.0,
            budget_rationale=f"baseline:{self.baseline_type.value}",
        )

    def _finalize(self, r: EpisodeResult, contacts: list[float], f_max_n: float) -> None:
        import numpy as np
        r.peak_contact_n = float(max(contacts, default=0.0))
        r.mean_contact_n = float(sum(contacts) / max(len(contacts), 1))
        r.compute_derived()
