"""Tests for the recovery selector — menu enforcement and budget immutability."""

import pytest
from forge_plus.llm.recovery_selector import RecoverySelector, ForceSignature, RECOVERY_MENU
from forge_plus.llm.client import MockLLMClient


def _make_sig(peak=15.0, net_mm=0.3, lateral="none"):
    return ForceSignature(
        peak_axial_N=peak,
        net_insert_mm=net_mm,
        axial_rising=True,
        lateral_bias=lateral,
        contact_persist_ms=500.0,
        slip_events=0,
    )


def test_action_in_menu():
    client = MockLLMClient(recovery_action="wiggle_search")
    selector = RecoverySelector(client=client)
    resp = selector.select(_make_sig(), 20.0, 0, "insert", "franka_panda")
    assert resp.action in RECOVERY_MENU


def test_f_max_unchanged():
    client = MockLLMClient(recovery_action="retract_and_reapproach")
    selector = RecoverySelector(client=client)
    f_max = 18.0
    resp = selector.select(_make_sig(), f_max, 0, "insert", "franka_panda")
    assert resp.keep_F_max_N == pytest.approx(f_max)


def test_invalid_action_falls_back():
    class BadClient:
        def call(self, payload):
            return {"action": "INCREASE_FORCE", "params": {}, "keep_F_max_N": 999.0, "rationale": ""}
        def name(self): return "bad"

    selector = RecoverySelector(client=BadClient())
    resp = selector.select(_make_sig(), 20.0, 0, "insert", "franka_panda")
    assert resp.action in RECOVERY_MENU


def test_budget_overwrite_even_if_llm_changes_it():
    class InflatingClient:
        def call(self, payload):
            return {
                "action": "retract_and_reapproach",
                "params": {},
                "keep_F_max_N": payload["F_max_N"] * 10,  # tries to inflate
                "rationale": "trying to cheat",
            }
        def name(self): return "inflating"

    selector = RecoverySelector(client=InflatingClient())
    f_max = 15.0
    resp = selector.select(_make_sig(), f_max, 1, "insert", "franka_panda")
    assert resp.keep_F_max_N == pytest.approx(f_max)  # overwritten


def test_all_menu_items_accepted():
    for action in RECOVERY_MENU:
        client = MockLLMClient(recovery_action=action)
        selector = RecoverySelector(client=client)
        resp = selector.select(_make_sig(), 25.0, 0, "insert", "franka_panda")
        assert resp.action == action
