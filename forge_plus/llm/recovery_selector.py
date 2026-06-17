"""Recovery-Selector: maps force/contact signature → recovery action from a fixed menu.

Called once per failure event. The menu deliberately omits any "increase the ceiling"
option — recovery reallocates motion within the budget, never buys more force.
The F_max echoed back in the response must equal the one sent in; if it differs,
we reject it and keep the original.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, field_validator

from forge_plus.llm.client import LLMClient

log = logging.getLogger(__name__)

RECOVERY_MENU = [
    "retract_and_reapproach",
    "wiggle_search",
    "rotate_align",
    "regrasp",
    "abort",
]


class ForceSignature(BaseModel):
    """Observable force/contact features — NEVER includes F_break."""

    peak_axial_N: float
    net_insert_mm: float
    axial_rising: bool
    lateral_bias: str          # e.g. "+x steady", "none", "oscillating"
    contact_persist_ms: float
    slip_events: int
    peak_lateral_N: float = 0.0
    mean_axial_N: float = 0.0
    torque_z_Nm: float = 0.0    # for rotational tasks

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump()


class RecoveryResponse(BaseModel):
    """Validated output from the LLM recovery-selector call."""

    action: str
    params: dict[str, Any] = Field(default_factory=dict)
    # Defaulted because RecoverySelector overwrites it with the true budget;
    # a default also keeps an LLM that omits the field from crashing the episode.
    keep_F_max_N: float = 0.0
    rationale: str = ""

    @field_validator("action")
    @classmethod
    def action_in_menu(cls, v: str) -> str:
        if v not in RECOVERY_MENU:
            # Gracefully fall back to first menu item rather than crashing, but
            # surface it: a silent swap hides LLM failures from post-hoc analysis.
            log.warning(
                "RecoverySelector: LLM returned out-of-menu action %r; "
                "falling back to %r",
                v, RECOVERY_MENU[0],
            )
            return RECOVERY_MENU[0]
        return v


@dataclass
class RecoverySelector:
    """Slow-layer component that picks a recovery action from a force signature.

    Design invariant: F_max is read-only. The response's keep_F_max_N is
    checked against the input and overwritten if it differs.
    """

    client: LLMClient
    menu: list[str] = None

    def __post_init__(self) -> None:
        if self.menu is None:
            self.menu = RECOVERY_MENU

    def select(
        self,
        signature: ForceSignature,
        f_max_n: float,
        attempt: int,
        subphase: str,
        gripper: str,
    ) -> RecoveryResponse:
        payload = self._build_payload(signature, f_max_n, attempt, subphase, gripper)
        raw = self.client.call(payload)
        response = self._validate_response(raw, f_max_n)
        return response

    def _build_payload(
        self,
        sig: ForceSignature,
        f_max_n: float,
        attempt: int,
        subphase: str,
        gripper: str,
    ) -> dict[str, Any]:
        return {
            "call": "select_recovery",
            "subphase": subphase,
            "gripper": gripper,
            "attempt": attempt,
            "F_max_N": f_max_n,
            "signature": sig.as_dict(),
            "menu": self.menu,
            "instructions": (
                "Return a JSON object with keys: action (string, must be one of the menu options), "
                "params (dict of action parameters, may be empty), "
                "keep_F_max_N (float, must equal the F_max_N provided — the budget is fixed), "
                "rationale (string). "
                "The menu has no 'increase ceiling' option by design. "
                "Choose the action that addresses the contact failure mode indicated by the signature "
                "while staying within the force budget."
            ),
        }

    def _validate_response(self, raw: dict[str, Any], f_max_n: float) -> RecoveryResponse:
        try:
            resp = RecoveryResponse.model_validate(raw)
        except Exception as exc:
            raise ValueError(f"RecoverySelector: invalid LLM response {raw!r}: {exc}") from exc

        # The budget is immutable — overwrite whatever the LLM said
        resp.keep_F_max_N = f_max_n
        return resp
