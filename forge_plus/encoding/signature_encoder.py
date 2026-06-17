"""Signature Encoder: converts recent F/T + contact history into a compact text token.

This sits between the fast control layer and the slow LLM layer.
INVARIANT: F_break never appears in the output — it is a hidden evaluator-only value.
The assertion guard at the bottom of encode() enforces this at runtime.

Feature set (all observable from F/T + kinematics):
  - peak_axial_N       peak contact force along the insertion/approach axis
  - net_insert_mm      net displacement achieved along insertion axis
  - axial_rising       whether axial force trend is increasing (bool)
  - lateral_bias       direction and steadiness of lateral force component
  - contact_persist_ms duration of continuous contact
  - slip_events        number of abrupt lateral-force sign reversals (slip proxy)
  - peak_lateral_N     peak lateral contact force magnitude
  - mean_axial_N       mean axial force over the window
  - torque_z_Nm        mean torque about insertion axis (for threaded tasks)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from forge_plus.llm.recovery_selector import ForceSignature


class ContactStep(NamedTuple):
    """One time-step of observable sensor data."""

    axial_force_n: float        # force along insertion axis (positive = toward socket)
    lateral_force_x_n: float    # lateral x component
    lateral_force_y_n: float    # lateral y component
    torque_z_nm: float          # torque about insertion axis
    insert_pos_mm: float        # displacement along insertion axis from episode start
    dt_ms: float                # time delta for this step in milliseconds


@dataclass
class SignatureEncoder:
    """Computes a ForceSignature from a sliding window of ContactStep records.

    Parameters
    ----------
    lateral_bias_threshold_n:
        Minimum steady lateral force (N) to report a directional bias.
    slip_threshold_n:
        Minimum lateral sign-reversal magnitude to count as a slip event.
    axial_trend_window:
        Fraction of the window to use for trend (rising/falling) detection.
    gripper_normalization:
        Per-gripper scale factor so signature values are gripper-agnostic.
        E.g. {"franka_panda": 1.0, "robotiq_2f140": 0.92}.
    """

    lateral_bias_threshold_n: float = 2.0
    slip_threshold_n: float = 1.5
    axial_trend_window: float = 0.25
    gripper_normalization: dict[str, float] | None = None

    def encode(self, history: list[ContactStep], gripper: str = "franka_panda") -> ForceSignature:
        if not history:
            raise ValueError("Cannot encode an empty history window.")

        norm = self._norm_factor(gripper)
        axial = np.array([s.axial_force_n * norm for s in history])
        lat_x = np.array([s.lateral_force_x_n * norm for s in history])
        lat_y = np.array([s.lateral_force_y_n * norm for s in history])
        torque = np.array([s.torque_z_nm for s in history])
        pos = np.array([s.insert_pos_mm for s in history])
        dt = np.array([s.dt_ms for s in history])

        lateral_mag = np.sqrt(lat_x**2 + lat_y**2)

        sig = ForceSignature(
            peak_axial_N=float(np.max(axial)),
            net_insert_mm=float(pos[-1] - pos[0]),
            axial_rising=self._is_rising(axial),
            lateral_bias=self._lateral_bias(lat_x, lat_y, lateral_mag),
            contact_persist_ms=float(np.sum(dt[axial > 0.5])),
            slip_events=self._count_slip_events(lat_x, lat_y),
            peak_lateral_N=float(np.max(lateral_mag)),
            mean_axial_N=float(np.mean(axial)),
            torque_z_Nm=float(np.mean(torque)),
        )

        # Non-circularity guard — F_break must never appear in the signature
        sig_dict = sig.as_dict()
        assert "F_break" not in sig_dict, "BUG: F_break leaked into force signature"
        assert "f_break" not in str(sig_dict).lower(), "BUG: f_break variant leaked into signature"

        return sig

    def _norm_factor(self, gripper: str) -> float:
        if self.gripper_normalization and gripper in self.gripper_normalization:
            return self.gripper_normalization[gripper]
        return 1.0

    def _is_rising(self, axial: np.ndarray) -> bool:
        n = max(1, int(len(axial) * self.axial_trend_window))
        early = axial[:n].mean()
        late = axial[-n:].mean()
        return bool(late > early + 0.5)

    def _lateral_bias(
        self, lat_x: np.ndarray, lat_y: np.ndarray, mag: np.ndarray
    ) -> str:
        if mag.mean() < self.lateral_bias_threshold_n:
            return "none"

        # Determine dominant axis and steadiness
        mean_x, mean_y = lat_x.mean(), lat_y.mean()
        std_x, std_y = lat_x.std(), lat_y.std()
        dominant_axis = "x" if abs(mean_x) >= abs(mean_y) else "y"
        dominant_mean = mean_x if dominant_axis == "x" else mean_y
        dominant_std = std_x if dominant_axis == "x" else std_y
        sign = "+" if dominant_mean > 0 else "-"
        steadiness = "steady" if dominant_std < 0.5 * abs(dominant_mean) else "oscillating"
        return f"{sign}{dominant_axis} {steadiness}"

    def _count_slip_events(self, lat_x: np.ndarray, lat_y: np.ndarray) -> int:
        slips = 0
        for arr in (lat_x, lat_y):
            for i in range(1, len(arr)):
                delta = arr[i] - arr[i - 1]
                prev_sign = math.copysign(1, arr[i - 1]) if abs(arr[i - 1]) > 0.1 else 0
                curr_sign = math.copysign(1, arr[i]) if abs(arr[i]) > 0.1 else 0
                if prev_sign != 0 and curr_sign != 0 and prev_sign != curr_sign:
                    if abs(delta) >= self.slip_threshold_n:
                        slips += 1
        return slips
