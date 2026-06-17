"""Recovery action executor.

Maps the LLM's menu choice → robot primitive.
All primitives operate within the current F_max — they reallocate motion,
not force. The force budget is passed in and never modified here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from forge_plus.envs.base_assembly_env import BaseAssemblyEnv


class RecoveryStatus(Enum):
    APPLIED = "applied"
    ABORTED = "aborted"
    UNKNOWN_ACTION = "unknown_action"


@dataclass
class RecoveryResult:
    status: RecoveryStatus
    action: str
    params: dict[str, Any]
    f_max_n: float              # budget is preserved through recovery


class RecoveryActionExecutor:
    """Dispatches recovery actions to the environment.

    Design: receives (action, params, f_max_n) — never modifies f_max_n.
    Each primitive is described below; the environment provides the
    actual robot motion implementation.
    """

    MENU = [
        "retract_and_reapproach",
        "wiggle_search",
        "rotate_align",
        "regrasp",
        "abort",
    ]

    def execute(
        self,
        env: BaseAssemblyEnv,
        action: str,
        params: dict[str, Any],
        f_max_n: float,
    ) -> RecoveryResult:
        if action not in self.MENU:
            return RecoveryResult(
                status=RecoveryStatus.UNKNOWN_ACTION,
                action=action,
                params=params,
                f_max_n=f_max_n,
            )

        if action == "abort":
            return RecoveryResult(
                status=RecoveryStatus.ABORTED,
                action=action,
                params=params,
                f_max_n=f_max_n,
            )

        # Each action delegates to the environment for actual robot motion
        dispatch = {
            "retract_and_reapproach": self._retract_and_reapproach,
            "wiggle_search":           self._wiggle_search,
            "rotate_align":            self._rotate_align,
            "regrasp":                 self._regrasp,
        }
        dispatch[action](env, params, f_max_n)
        return RecoveryResult(
            status=RecoveryStatus.APPLIED,
            action=action,
            params=params,
            f_max_n=f_max_n,
        )

    # ------------------------------------------------------------------ #
    # Primitives — each operates within f_max_n                           #
    # ------------------------------------------------------------------ #

    def _retract_and_reapproach(
        self, env: BaseAssemblyEnv, params: dict, f_max_n: float
    ) -> None:
        """Pull back from contact, then re-approach along the insertion axis.

        Useful for: wedge jams, misalignment — clears contact and re-centers.
        """
        retract_mm = float(params.get("retract_mm", 5.0))
        env.apply_recovery("retract_and_reapproach", {"retract_mm": retract_mm})

    def _wiggle_search(
        self, env: BaseAssemblyEnv, params: dict, f_max_n: float
    ) -> None:
        """Sinusoidal lateral search pattern while maintaining reduced axial load.

        Useful for: friction jams, slight misalignment.
        The lateral amplitude is bounded to respect f_max_n.
        """
        amplitude_mm = min(float(params.get("amplitude_mm", 1.5)), 3.0)
        frequency_hz = float(params.get("frequency_hz", 2.0))
        duration_s = float(params.get("duration_s", 1.0))
        env.apply_recovery(
            "wiggle_search",
            {"amplitude_mm": amplitude_mm, "frequency_hz": frequency_hz, "duration_s": duration_s},
        )

    def _rotate_align(
        self, env: BaseAssemblyEnv, params: dict, f_max_n: float
    ) -> None:
        """Rotate the end-effector by a small angle to correct angular misalignment.

        Useful for: off-axis lateral bias, threaded connector cross-threading.
        Follows up with wiggle_search if requested in params["then"].
        """
        yaw_deg = float(params.get("yaw_deg", 3.0))
        env.apply_recovery("rotate_align", {"yaw_deg": yaw_deg})
        if params.get("then") == "wiggle_search":
            self._wiggle_search(env, params, f_max_n)

    def _regrasp(
        self, env: BaseAssemblyEnv, params: dict, f_max_n: float
    ) -> None:
        """Open the gripper, release the part, and re-grasp.

        Useful for: slip-induced grasp displacement, poor initial grasp.
        """
        env.apply_recovery("regrasp", params)
