"""Task-agnostic, force-signature LLM recovery loop.

This is the closed-loop recovery orchestrator from proposal §07, factored so it
drives ANY environment that implements the small ``RecoveryEnv`` protocol —
each task has its own Isaac env, but they all share this one loop.

The loop is deliberately thin and contains no task- or sim-specific logic:

    set budget (once)  ->  for each attempt:
        run the nominal skill until success / failure / timeout
        if success: done
        else: encode a TEXT force signature (no vision, no F_break)
              ask the LLM RecoverySelector for a recovery action
              apply it (F_max unchanged) and retry

It reuses the existing semantic layer wholesale: ``SignatureEncoder`` is not
needed here because each env reports its own ``ForceSignature`` (computed from
its observable contact history), and ``RecoverySelector`` makes the call. The
budget is set by the caller (``BudgetSetter``) and passed in as ``f_max_n`` on
the env; this loop never relaxes it (the menu has no "raise ceiling" option).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from forge_plus.llm.recovery_selector import ForceSignature, RecoverySelector

log = logging.getLogger(__name__)


class RecoveryOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    FAIL_NO_ATTEMPTS_LEFT = "FAIL_NO_ATTEMPTS_LEFT"
    ABORTED = "ABORTED"


@runtime_checkable
class RecoveryEnv(Protocol):
    """Contract a task env implements to be driven by ``RecoveryLoop``.

    Implemented by the Isaac task envs (over a single instance) — see
    ``FrankaPickPlaceEnv`` recovery hooks. Kept minimal so a CPU fake can stand
    in for unit tests.
    """

    f_max_n: float                  # the per-object force ceiling (set by BudgetSetter)
    max_steps_per_attempt: int

    def reset_episode(self) -> None: ...
    def step_skill(self) -> None:
        """Advance ONE control step under the nominal force-conditioned skill
        (e.g. zero-action OSC + the phase state machine)."""
        ...

    def is_success(self) -> bool: ...
    def is_failure(self) -> bool:
        """True if the attempt has failed (jam / over-force / no-progress)."""
        ...

    def failure_signature(self) -> ForceSignature:
        """Observable force/contact signature for the recovery call — NO vision,
        NO F_break."""
        ...

    def apply_recovery(self, action: str, params: dict[str, Any]) -> None:
        """Execute a recovery primitive (retract / wiggle / rotate_align /
        regrasp). F_max is never changed here."""
        ...

    def subphase(self) -> str: ...
    def gripper(self) -> str: ...


@dataclass
class AttemptLog:
    attempt: int
    steps: int
    result: str                       # "success" | "failure" | "timeout"
    signature: dict[str, Any] | None = None
    recovery_action: str | None = None
    recovery_params: dict[str, Any] | None = None
    recovery_rationale: str | None = None


@dataclass
class RecoveryEpisodeResult:
    outcome: RecoveryOutcome
    attempts: int
    f_max_n: float
    log: list[AttemptLog] = field(default_factory=list)


@dataclass
class RecoveryLoop:
    """Drives a ``RecoveryEnv`` through up to ``k_max`` force-guided attempts."""

    selector: RecoverySelector
    k_max: int = 4

    def run(self, env: RecoveryEnv, on_step=None) -> RecoveryEpisodeResult:
        """Run the closed loop. ``on_step(env, attempt, step)`` (optional) is
        called after every control step — used by the renderer to capture a frame."""
        result = RecoveryEpisodeResult(
            outcome=RecoveryOutcome.FAIL_NO_ATTEMPTS_LEFT,
            attempts=0,
            f_max_n=float(env.f_max_n),
        )
        env.reset_episode()

        for attempt in range(self.k_max):
            result.attempts = attempt + 1
            steps = 0
            outcome = "timeout"
            while steps < env.max_steps_per_attempt:
                env.step_skill()
                steps += 1
                if on_step is not None:
                    on_step(env, attempt, steps)
                if env.is_success():
                    outcome = "success"
                    break
                if env.is_failure():
                    outcome = "failure"
                    break

            if outcome == "success":
                result.log.append(AttemptLog(attempt, steps, "success"))
                result.outcome = RecoveryOutcome.SUCCESS
                log.info("[recovery] attempt %d: SUCCESS in %d steps", attempt, steps)
                return result

            # --- failure (or timeout treated as a failure to recover from) ---
            sig = env.failure_signature()
            decision = self.selector.select(
                signature=sig,
                f_max_n=float(env.f_max_n),
                attempt=attempt,
                subphase=env.subphase(),
                gripper=env.gripper(),
            )
            entry = AttemptLog(
                attempt=attempt,
                steps=steps,
                result=outcome,
                signature=sig.as_dict(),
                recovery_action=decision.action,
                recovery_params=decision.params,
                recovery_rationale=decision.rationale,
            )
            result.log.append(entry)
            log.info(
                "[recovery] attempt %d: %s -> signature(peak_axial=%.1fN, "
                "net_insert=%.1fmm, rising=%s, lateral=%s) -> action=%s %s",
                attempt, outcome, sig.peak_axial_N, sig.net_insert_mm,
                sig.axial_rising, sig.lateral_bias, decision.action, decision.params,
            )

            if decision.action == "abort":
                result.outcome = RecoveryOutcome.ABORTED
                return result

            env.apply_recovery(decision.action, decision.params)

        log.info("[recovery] no attempts left after %d", self.k_max)
        return result
