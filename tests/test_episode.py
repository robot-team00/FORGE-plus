"""Integration tests for the full episode runner."""

import numpy as np
import pytest
import torch
from forge_plus.episode import EpisodeRunner, EpisodeTermination
from forge_plus.envs.base_assembly_env import EpisodeConfig
from forge_plus.envs.mock_assembly_env import MockAssemblyEnv, MockEnvConfig
from forge_plus.llm.client import MockLLMClient
from forge_plus.llm.budget_setter import BudgetSetter
from forge_plus.llm.recovery_selector import RecoverySelector
from forge_plus.encoding.signature_encoder import SignatureEncoder
from forge_plus.recovery.recovery_actions import RecoveryActionExecutor
from forge_plus.skills.forge_skill import FORGESkill, SkillConfig
from forge_plus.skills.policy_network import PolicyConfig


def _make_runner(budget_n=30.0, recovery_action="retract_and_reapproach", jam_prob=0.0):
    # Seed so the (untrained, stochastic) policy is reproducible across runs.
    torch.manual_seed(0)
    np.random.seed(0)
    client = MockLLMClient(budget_n=budget_n, recovery_action=recovery_action)
    env = MockAssemblyEnv(MockEnvConfig(jam_probability=jam_prob))
    skill = FORGESkill(SkillConfig(policy_cfg=PolicyConfig()))
    return EpisodeRunner(
        budget_setter=BudgetSetter(client=client),
        recovery_selector=RecoverySelector(client=client),
        signature_encoder=SignatureEncoder(),
        skill=skill,
        env=env,
        recovery_executor=RecoveryActionExecutor(),
        k_max=5,
    )


def _make_cfg(obj_key="steel_peg", f_break=200.0, seed=0):
    return EpisodeConfig(
        object_key=obj_key,
        task_name="task1",
        gripper="franka_panda",
        f_break_n=f_break,
        disturbance_seed=seed,
    )


def test_episode_completes_without_jam():
    runner = _make_runner(jam_prob=0.0, budget_n=30.0)
    result = runner.run(_make_cfg(f_break=200.0))
    # Budget (30 N) is far below F_break (200 N), so the part must never break.
    assert result.termination != EpisodeTermination.BROKEN
    assert result.broke is False
    assert result.total_steps > 0


def test_f_max_never_exceeds_global_cap():
    from forge_plus.episode import GLOBAL_HARD_CAP_N
    runner = _make_runner(budget_n=999.0)  # LLM wants 999 N
    result = runner.run(_make_cfg(f_break=200.0))
    assert result.f_max_n <= GLOBAL_HARD_CAP_N


def test_broken_when_f_break_exceeded():
    # Budget (100 N) far exceeds F_break (1 N), so the commanded force drives the
    # contact force past the breaking threshold — the env must report BROKEN.
    runner = _make_runner(budget_n=100.0)
    cfg = _make_cfg(f_break=1.0)  # very fragile — breaks almost immediately
    result = runner.run(cfg)
    assert result.termination == EpisodeTermination.BROKEN
    assert result.broke is True


def test_safety_margin_computed():
    runner = _make_runner(budget_n=30.0)
    result = runner.run(_make_cfg(f_break=50.0))
    result.compute_derived()
    assert result.safety_margin_n == pytest.approx(50.0 - result.f_max_n)


def test_over_budget_flag():
    runner = _make_runner(budget_n=80.0)
    result = runner.run(_make_cfg(f_break=20.0))  # F_max > F_break
    result.compute_derived()
    assert result.over_budget is True


def test_budget_integrity_in_recovery():
    """Recovery must not change F_max — enforced by assertion in episode.py."""
    runner = _make_runner(budget_n=25.0, jam_prob=0.5, recovery_action="wiggle_search")
    cfg = _make_cfg(f_break=200.0, seed=1)
    result = runner.run(cfg)
    assert result.f_max_n <= 25.0 + 1e-6  # budget unchanged


def test_episode_result_fields_populated():
    runner = _make_runner(budget_n=30.0)
    result = runner.run(_make_cfg())
    assert result.object_key == "steel_peg"
    assert result.gripper == "franka_panda"
    assert result.total_attempts >= 1
    assert result.wall_time_s >= 0
    assert isinstance(result.budget_rationale, str)


def test_k_max_limits_attempts():
    runner = _make_runner(budget_n=30.0, jam_prob=1.0)  # always jammed
    result = runner.run(_make_cfg(f_break=200.0))
    assert result.total_attempts <= runner.k_max
