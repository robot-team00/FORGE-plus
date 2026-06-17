"""Force Clamp: hard per-axis and scalar force ceiling enforcement.

This runs in the fast control loop (60-120 Hz) and is the authoritative
safety mechanism. The LLM only proposes a number; this clamp enforces it
regardless of what the LLM said. A fixed global hard cap provides a
second line of defense against hallucinated ceilings.

Clamping the *command* does not bound the *contact* force due to impedance
overshoot — that gap is tracked as a first-class metric (clamp_fidelity).
Overshoot mitigation is done at the controller level via low stiffness and
near-contact velocity limits.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import NamedTuple

GLOBAL_HARD_CAP_N: float = 120.0


class Wrench(NamedTuple):
    """6-DOF wrench: [fx, fy, fz, tx, ty, tz] in robot base frame (N, N·m)."""

    fx: float
    fy: float
    fz: float
    tx: float
    ty: float
    tz: float

    def as_array(self) -> np.ndarray:
        return np.array([self.fx, self.fy, self.fz, self.tx, self.ty, self.tz])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Wrench":
        return cls(*arr[:6])

    def force_magnitude(self) -> float:
        return float(np.linalg.norm([self.fx, self.fy, self.fz]))


@dataclass
class ForceClamp:
    """Saturate a commanded wrench to respect a force ceiling.

    Parameters
    ----------
    f_max_n:
        Per-episode force ceiling set by the LLM budget-setter (N).
    per_axis_n:
        Optional per-axis limits {"insertion": ..., "lateral": ...}.
        Applied in addition to the scalar ceiling.
    insertion_axis:
        Index into [fx, fy, fz] corresponding to the insertion direction.
    global_hard_cap_n:
        Absolute maximum — the controller-level backstop regardless of LLM.
    """

    f_max_n: float
    per_axis_n: dict[str, float] | None = None
    insertion_axis: int = 2          # z-axis by default
    global_hard_cap_n: float = GLOBAL_HARD_CAP_N

    def __post_init__(self) -> None:
        # Enforce the global cap on the LLM-proposed ceiling at init time
        self.f_max_n = min(self.f_max_n, self.global_hard_cap_n)
        if self.per_axis_n:
            self.per_axis_n = {
                k: min(v, self.global_hard_cap_n) for k, v in self.per_axis_n.items()
            }

    def clamp(self, cmd: Wrench) -> tuple[Wrench, float]:
        """Apply the force ceiling. Returns (clamped_wrench, overshoot_fraction).

        overshoot_fraction is how much of the ceiling the unclamped command
        exceeded (0 if within budget, >0 if the clamp fired).
        """
        forces = np.array([cmd.fx, cmd.fy, cmd.fz])

        # Scalar force magnitude clamp
        mag = np.linalg.norm(forces)
        effective_max = self.f_max_n
        if mag > effective_max:
            forces = forces * (effective_max / mag)

        # Per-axis clamp
        if self.per_axis_n:
            ins = self.per_axis_n.get("insertion", self.f_max_n)
            lat = self.per_axis_n.get("lateral", self.f_max_n)
            for i, ax_max in enumerate([lat, lat, ins] if self.insertion_axis == 2
                                        else [ins if i == self.insertion_axis else lat for i in range(3)]):
                forces[i] = np.clip(forces[i], -ax_max, ax_max)

        # Global hard cap (absolute backstop)
        final_mag = np.linalg.norm(forces)
        if final_mag > self.global_hard_cap_n:
            forces = forces * (self.global_hard_cap_n / final_mag)

        clamped = Wrench(
            fx=float(forces[0]),
            fy=float(forces[1]),
            fz=float(forces[2]),
            tx=cmd.tx,
            ty=cmd.ty,
            tz=cmd.tz,
        )
        overshoot = max(0.0, (mag - self.f_max_n) / max(self.f_max_n, 1e-6))
        return clamped, overshoot

    def update_ceiling(self, f_max_n: float) -> None:
        """Update the ceiling (e.g. between subtasks). Global cap still applies."""
        self.f_max_n = min(f_max_n, self.global_hard_cap_n)

    @staticmethod
    def measure_clamp_fidelity(
        commanded: list[Wrench],
        actual_contact: list[float],
        f_max_n: float,
    ) -> dict[str, float]:
        """Compute clamp-vs-contact gap metrics over an episode.

        Returns mean/max overshoot of actual contact force above the ceiling.
        """
        gaps = [max(0.0, c - f_max_n) for c in actual_contact]
        return {
            "mean_overshoot_n": float(np.mean(gaps)),
            "max_overshoot_n": float(np.max(gaps)) if gaps else 0.0,
            "overshoot_rate": float(np.mean([g > 0 for g in gaps])),
        }
