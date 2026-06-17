"""Tests for the budget setter — range validation and caching."""

import pytest
from forge_plus.llm.budget_setter import BudgetSetter, ObjectIdentity, GLOBAL_HARD_CAP_N
from forge_plus.llm.client import MockLLMClient


def _make_obj(name="test connector", material="ABS", cls="round_connector"):
    return ObjectIdentity(
        name=name,
        material=material,
        **{"class": cls},
        nominal_mass_g=10.0,
        geometry_tags=["thin_wall"],
    )


def test_budget_within_global_cap():
    client = MockLLMClient(budget_n=30.0)
    setter = BudgetSetter(client=client)
    resp = setter.set_budget(_make_obj(), "task1")
    assert resp.F_max_N <= GLOBAL_HARD_CAP_N


def test_over_cap_clamped():
    # Mock returns 999 N — must be clamped
    client = MockLLMClient(budget_n=999.0)
    setter = BudgetSetter(client=client)
    resp = setter.set_budget(_make_obj(), "task1")
    assert resp.F_max_N <= GLOBAL_HARD_CAP_N


def test_caching():
    client = MockLLMClient(budget_n=25.0)
    setter = BudgetSetter(client=client)
    obj = _make_obj()
    r1 = setter.set_budget(obj, "task1")
    # Corrupt the client to return different value — should be ignored due to cache
    client._budget_n = 99.0
    r2 = setter.set_budget(obj, "task1")
    assert r1.F_max_N == r2.F_max_N


def test_per_axis_clamped():
    client = MockLLMClient(budget_n=999.0)
    setter = BudgetSetter(client=client)
    resp = setter.set_budget(_make_obj(), "task1")
    for v in resp.per_axis_N.values():
        assert v <= GLOBAL_HARD_CAP_N


def test_confidence_in_range():
    client = MockLLMClient()
    setter = BudgetSetter(client=client)
    resp = setter.set_budget(_make_obj(), "task1")
    assert 0.0 <= resp.confidence <= 1.0


def test_different_tasks_different_cache_keys():
    client = MockLLMClient(budget_n=30.0)
    setter = BudgetSetter(client=client)
    obj = _make_obj()
    r1 = setter.set_budget(obj, "task1")
    client._budget_n = 50.0   # change
    r2 = setter.set_budget(obj, "task2")  # different task — new call
    # Both results are valid — they may differ because they hit different cache keys
    assert r1.F_max_N <= GLOBAL_HARD_CAP_N
    assert r2.F_max_N <= GLOBAL_HARD_CAP_N
