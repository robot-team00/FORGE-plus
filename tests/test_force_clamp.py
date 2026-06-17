"""Tests for the force clamp — the primary safety mechanism."""

import pytest
from forge_plus.control.force_clamp import ForceClamp, Wrench, GLOBAL_HARD_CAP_N


def test_clamp_respects_f_max():
    clamp = ForceClamp(f_max_n=20.0)
    cmd = Wrench(0, 0, 30, 0, 0, 0)
    clamped, overshoot = clamp.clamp(cmd)
    assert abs(clamped.fz) <= 20.0 + 1e-6
    assert overshoot > 0


def test_clamp_passthrough_within_budget():
    clamp = ForceClamp(f_max_n=50.0)
    cmd = Wrench(3, 4, 0, 0, 0, 0)  # lateral magnitude = 5.0
    clamped, overshoot = clamp.clamp(cmd)
    assert abs(clamped.fx - 3.0) < 1e-6
    assert abs(clamped.fy - 4.0) < 1e-6
    assert overshoot == 0.0


def test_global_hard_cap_enforced():
    clamp = ForceClamp(f_max_n=200.0)   # LLM hallucinated 200 N
    cmd = Wrench(0, 0, 200, 0, 0, 0)
    clamped, _ = clamp.clamp(cmd)
    assert abs(clamped.fz) <= GLOBAL_HARD_CAP_N + 1e-6


def test_init_clamps_f_max_to_global_cap():
    clamp = ForceClamp(f_max_n=999.0)
    assert clamp.f_max_n <= GLOBAL_HARD_CAP_N


def test_per_axis_clamp():
    clamp = ForceClamp(f_max_n=50.0, per_axis_n={"insertion": 30.0, "lateral": 10.0})
    cmd = Wrench(15, 0, 40, 0, 0, 0)   # lateral 15 exceeds 10, insertion 40 exceeds 30
    clamped, _ = clamp.clamp(cmd)
    assert abs(clamped.fx) <= 10.0 + 1e-6
    assert abs(clamped.fz) <= 30.0 + 1e-6


def test_torques_unchanged():
    clamp = ForceClamp(f_max_n=10.0)
    cmd = Wrench(0, 0, 5, 1.5, 2.0, 0.5)
    clamped, _ = clamp.clamp(cmd)
    assert abs(clamped.tx - 1.5) < 1e-6
    assert abs(clamped.ty - 2.0) < 1e-6


def test_update_ceiling():
    clamp = ForceClamp(f_max_n=30.0)
    clamp.update_ceiling(60.0)
    assert clamp.f_max_n == 60.0
    clamp.update_ceiling(999.0)
    assert clamp.f_max_n <= GLOBAL_HARD_CAP_N


def test_clamp_fidelity_metric():
    commanded = [Wrench(0, 0, 20, 0, 0, 0)] * 5
    actual = [20.0, 22.0, 18.0, 21.5, 19.0]   # some overshoot
    fidelity = ForceClamp.measure_clamp_fidelity(commanded, actual, 20.0)
    assert fidelity["max_overshoot_n"] == pytest.approx(2.0)
    assert fidelity["overshoot_rate"] == pytest.approx(0.4)  # 2/5 steps exceeded
