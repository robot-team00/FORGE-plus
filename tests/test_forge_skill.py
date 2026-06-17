"""Tests for the FORGE skill's observation encoding."""

import numpy as np

from forge_plus.control.force_clamp import Wrench
from forge_plus.encoding.signature_encoder import ContactStep
from forge_plus.envs.base_assembly_env import EnvObservation, TaskPhase
from forge_plus.skills.forge_skill import FORGESkill, SkillConfig
from forge_plus.skills.policy_network import PolicyConfig


def _obs(phase: TaskPhase) -> EnvObservation:
    return EnvObservation(
        joint_pos=np.zeros(7),
        joint_vel=np.zeros(7),
        ee_pos=np.zeros(3),
        ee_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        ft_wrench=Wrench(0, 0, 0, 0, 0, 0),
        contact_step=ContactStep(0, 0, 0, 0, 0.0, 8.0),
        phase=phase,
        step_count=0,
    )


def _skill() -> FORGESkill:
    return FORGESkill(SkillConfig(policy_cfg=PolicyConfig()))


def test_phase_onehot_is_distinct_per_phase():
    # Every phase must map to its own one-hot slot — no collisions. The previous
    # hash(value) % 6 collapsed 7 phases into 6 bins.
    skill = _skill()
    indices = {}
    for phase in TaskPhase:
        vec = skill._encode_obs(_obs(phase))
        onehot = vec[-len(TaskPhase):]
        assert onehot.sum() == 1.0
        indices[phase] = int(np.argmax(onehot))
    assert len(set(indices.values())) == len(TaskPhase)


def test_phase_onehot_is_deterministic():
    # The encoding must be stable across separate skill instances (and processes);
    # str hashing was salted by PYTHONHASHSEED and broke train/eval consistency.
    a = _skill()._encode_obs(_obs(TaskPhase.INSERT))
    b = _skill()._encode_obs(_obs(TaskPhase.INSERT))
    assert np.array_equal(a, b)


def test_encoded_obs_matches_policy_obs_dim():
    vec = _skill()._encode_obs(_obs(TaskPhase.APPROACH))
    assert vec.shape[0] == PolicyConfig().obs_dim
