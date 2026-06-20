"""Budget-Setter: maps object identity → numeric force ceiling F_max.

The LLM is called once per episode before any control begins.
It never receives F_break, which is hidden in the simulator.
The output is range-validated against a fixed global hard cap before
it reaches the controller — safety is never delegated to the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, field_validator

from forge_plus.control.force_clamp import GLOBAL_HARD_CAP_N
from forge_plus.llm.client import LLMClient


class ObjectIdentity(BaseModel):
    """Descriptor the LLM can read — contains NO ground-truth force data."""

    name: str
    material: str
    object_class: str = Field(alias="class")
    nominal_mass_g: float
    geometry_tags: list[str] = Field(default_factory=list)
    additional_notes: str = ""

    model_config = {"populate_by_name": True}


class BudgetResponse(BaseModel):
    """Validated output from the LLM budget-setter call.

    Note: F_max_N has no upper bound constraint in pydantic — the hard cap is
    enforced by pre-clamping the raw dict before validation (in _validate_response).
    This lets us receive an arbitrarily high value from the LLM and clamp it
    rather than rejecting it outright.
    """

    F_max_N: float = Field(ge=0.0)
    per_axis_N: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    rationale: str = ""


@dataclass
class BudgetSetter:
    """Slow-layer component that sets a per-object force ceiling from identity.

    Caches outputs per (object_name, material, task) so identical objects
    don't re-query the LLM — budget-setting is near-deterministic for a
    given identity, so caching is safe.
    """

    client: LLMClient
    global_hard_cap_n: float = GLOBAL_HARD_CAP_N
    _cache: dict[str, BudgetResponse] = field(default_factory=dict, init=False, repr=False)

    def set_budget(self, obj: ObjectIdentity, task: str) -> BudgetResponse:
        cache_key = f"{obj.name}|{obj.material}|{obj.object_class}|{task}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        payload = self._build_payload(obj, task)
        raw = self.client.call(payload)
        response = self._validate_response(raw)
        self._cache[cache_key] = response
        return response

    def _build_payload(self, obj: ObjectIdentity, task: str) -> dict[str, Any]:
        return {
            "call": "set_force_ceiling",
            "object": {
                "name": obj.name,
                "material": obj.material,
                "class": obj.object_class,
                "nominal_mass_g": obj.nominal_mass_g,
                "geometry_tags": obj.geometry_tags,
                **({"notes": obj.additional_notes} if obj.additional_notes else {}),
            },
            "task": task,
            "global_hard_cap_N": self.global_hard_cap_n,
            "instructions": (
                "Return a JSON object with keys: F_max_N (float, newtons), "
                "per_axis_N (dict with 'insertion' and 'lateral' keys, newtons), "
                "confidence (float 0-1), rationale (string). "
                f"F_max_N must be between 0 and {self.global_hard_cap_n}. "
                "Do not reference F_break — you have no access to it. "
                "Reason only from the object identity provided. As calibration guidance, typical SAFE sustained contact forces that avoid any damage, by material class: brittle glass or thin ceramic ~8-18 N; rigid plastics ~25-45 N; thick stoneware or wood ~40-80 N; metals ~80-115 N. Any object whose fragility tags include 'brittle' (e.g. glass, ceramic, resin) is fragile: use the 8-18 N band regardless of its size or wall thickness. Pick a conservative value inside the band that matches this object's material, wall thickness and fragility tags. For place or stack tasks excess normal force is the dominant failure, so prefer the lower end of the band."
            ),
        }

    def _validate_response(self, raw: dict[str, Any]) -> BudgetResponse:
        # Pre-clamp before pydantic sees the values — we clamp rather than reject.
        # The numeric coercion is inside the try so a non-numeric LLM value
        # (string/None/list) yields the same clear error instead of an
        # unwrapped float() crash.
        try:
            clamped = dict(raw)
            f_max = min(float(clamped.get("F_max_N", 0)), self.global_hard_cap_n)
            clamped["F_max_N"] = f_max
            if isinstance(clamped.get("per_axis_N"), dict):
                # Per-axis limits are subsidiary to the scalar ceiling: never
                # let an axis exceed F_max (nor the global hard cap).
                clamped["per_axis_N"] = {
                    k: min(float(v), f_max) for k, v in clamped["per_axis_N"].items()
                }
            resp = BudgetResponse.model_validate(clamped)
        except Exception as exc:
            raise ValueError(f"BudgetSetter: invalid LLM response {raw!r}: {exc}") from exc
        return resp
