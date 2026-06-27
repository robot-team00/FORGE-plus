"""Tests for the task-agnostic force-signature recovery loop.

Uses a CPU fake env (no Isaac) + the real RecoverySelector with the heuristic
backend, so the full closed loop is exercised without a GPU or an API key:
    jam -> force signature -> LLM picks a recovery -> apply -> retry -> success.
"""

from __future__ import annotations

from forge_plus.llm.client import HeuristicLLMClient
from forge_plus.llm.recovery_selector import ForceSignature, RecoverySelector
from forge_plus.recovery.recovery_loop import RecoveryLoop, RecoveryOutcome


class FakeJamEnv:
    """A contact-rich insertion that JAMS with a lateral bias until the EE is
    re-aligned. The first attempt wedges; once a ``rotate_align`` recovery is
    applied, the next attempt seats. Stands in for an Isaac task env."""

    def __init__(self, f_max_n: float = 16.0, jam_step: int = 8):
        self.f_max_n = f_max_n
        self.max_steps_per_attempt = 40
        self.jam_step = jam_step
        self._aligned = False
        self._step = 0
        self.recoveries: list[str] = []

    def reset_episode(self) -> None:
        self._aligned = False
        self._step = 0

    def step_skill(self) -> None:
        self._step += 1

    def is_success(self) -> bool:
        # Once aligned, the bottle seats after a few clean steps.
        return self._aligned and self._step >= 4

    def is_failure(self) -> bool:
        # Misaligned -> wedges (jam) after jam_step steps of rising contact.
        return (not self._aligned) and self._step >= self.jam_step

    def failure_signature(self) -> ForceSignature:
        # Jam signature: at the ceiling, no displacement, steady lateral bias.
        return ForceSignature(
            peak_axial_N=self.f_max_n,
            net_insert_mm=0.3,
            axial_rising=True,
            lateral_bias="+x steady",
            contact_persist_ms=600.0,
            slip_events=0,
            peak_lateral_N=4.0,
        )

    def apply_recovery(self, action: str, params: dict) -> None:
        self.recoveries.append(action)
        self._step = 0
        if action in ("rotate_align", "regrasp"):
            self._aligned = True   # the corrective move clears the misalignment

    def subphase(self) -> str:
        return "insertion"

    def gripper(self) -> str:
        return "franka_panda"


def _loop() -> RecoveryLoop:
    selector = RecoverySelector(client=HeuristicLLMClient())
    return RecoveryLoop(selector=selector, k_max=4)


def test_jam_then_recover_succeeds():
    env = FakeJamEnv()
    result = _loop().run(env)
    assert result.outcome == RecoveryOutcome.SUCCESS
    assert result.attempts == 2                       # jam, recover, succeed
    # The force signature had a lateral bias -> heuristic must pick rotate_align.
    assert env.recoveries == ["rotate_align"]
    # First attempt logged the signature + chosen action.
    first = result.log[0]
    assert first.result == "failure"
    assert first.recovery_action == "rotate_align"
    assert first.signature["lateral_bias"] == "+x steady"


def test_loop_never_relaxes_fmax():
    env = FakeJamEnv()
    result = _loop().run(env)
    assert result.f_max_n == env.f_max_n   # budget unchanged through recovery


def test_unrecoverable_jam_aborts_eventually():
    # An env that never clears -> the selector escalates to abort by attempt 4.
    class NeverClears(FakeJamEnv):
        def apply_recovery(self, action: str, params: dict) -> None:
            self.recoveries.append(action)
            self._step = 0
            # never set _aligned -> stays jammed
    env = NeverClears()
    result = _loop().run(env)
    assert result.outcome in (RecoveryOutcome.ABORTED,
                              RecoveryOutcome.FAIL_NO_ATTEMPTS_LEFT)
    assert "abort" in env.recoveries or result.attempts == 4
