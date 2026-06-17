"""Tests for the signature encoder — non-circularity and feature correctness."""

import pytest
from forge_plus.encoding.signature_encoder import SignatureEncoder, ContactStep


def _make_steps(n: int, axial=10.0, lat_x=0.0, lat_y=0.0, pos_start=0.0) -> list[ContactStep]:
    return [
        ContactStep(
            axial_force_n=axial,
            lateral_force_x_n=lat_x,
            lateral_force_y_n=lat_y,
            torque_z_nm=0.0,
            insert_pos_mm=pos_start + i * 0.1,
            dt_ms=8.0,
        )
        for i in range(n)
    ]


def test_non_circularity_no_f_break():
    enc = SignatureEncoder()
    steps = _make_steps(20)
    sig = enc.encode(steps)
    sig_dict = sig.as_dict()
    assert "F_break" not in sig_dict
    assert "f_break" not in str(sig_dict).lower()


def test_empty_history_raises():
    enc = SignatureEncoder()
    with pytest.raises(ValueError):
        enc.encode([])


def test_peak_axial():
    steps = _make_steps(10, axial=15.0)
    steps[5] = ContactStep(25.0, 0, 0, 0, 0.5, 8.0)
    enc = SignatureEncoder()
    sig = enc.encode(steps)
    assert sig.peak_axial_N == pytest.approx(25.0)


def test_net_insert_mm():
    steps = _make_steps(10, pos_start=5.0)
    enc = SignatureEncoder()
    sig = enc.encode(steps)
    assert sig.net_insert_mm == pytest.approx(0.9, abs=0.01)


def test_axial_rising():
    # Rising: later steps have more force
    steps = [
        ContactStep(float(i), 0, 0, 0, float(i) * 0.1, 8.0)
        for i in range(1, 21)
    ]
    enc = SignatureEncoder()
    sig = enc.encode(steps)
    assert sig.axial_rising is True


def test_axial_not_rising():
    steps = [
        ContactStep(float(20 - i), 0, 0, 0, float(i) * 0.1, 8.0)
        for i in range(20)
    ]
    enc = SignatureEncoder()
    sig = enc.encode(steps)
    assert sig.axial_rising is False


def test_lateral_bias_detected():
    steps = _make_steps(20, lat_x=5.0)
    enc = SignatureEncoder()
    sig = enc.encode(steps)
    assert "+x" in sig.lateral_bias


def test_no_lateral_bias():
    steps = _make_steps(20, lat_x=0.1)
    enc = SignatureEncoder()
    sig = enc.encode(steps)
    assert sig.lateral_bias == "none"


def test_gripper_normalization():
    steps = _make_steps(10, axial=10.0)
    enc = SignatureEncoder(gripper_normalization={"robotiq_2f140": 0.5})
    sig = enc.encode(steps, gripper="robotiq_2f140")
    assert sig.peak_axial_N == pytest.approx(5.0)


def test_contact_persist_ms():
    steps = _make_steps(10, axial=2.0)  # all above 0.5 threshold
    enc = SignatureEncoder()
    sig = enc.encode(steps)
    assert sig.contact_persist_ms == pytest.approx(10 * 8.0)
