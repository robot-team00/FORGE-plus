"""Tests for the Task 3 fragile place/stack environment and report pieces.

These avoid the torch policy on purpose (they drive the env with explicit press
commands) so they run fast without a GPU.
"""

import random

import pytest

from forge_plus.control.force_clamp import ForceClamp, Wrench, GLOBAL_HARD_CAP_N
from forge_plus.encoding.signature_encoder import SignatureEncoder
from forge_plus.envs.base_assembly_env import EpisodeConfig, TaskOutcome
from forge_plus.envs.place_stack_env import PlaceStackEnv, PlaceStackEnvConfig
from forge_plus.envs.object_configs import OBJECT_REGISTRY
from forge_plus.llm.client import HeuristicLLMClient, build_client
from forge_plus.llm.budget_setter import BudgetSetter

TASK = "task3_fragile_place"


def _cfg(obj, f_break, seed=0):
    return EpisodeConfig(object_key=obj, task_name=TASK, gripper="franka_panda",
                         f_break_n=f_break, max_steps=1500, disturbance_seed=seed)


def _press(env, cfg, f_max, recovery="rotate_align", escalate=1.0, k_max=5):
    obs = env.reset(cfg)
    f = f_max
    outcome = TaskOutcome.IN_PROGRESS
    for _ in range(k_max):
        clamp = ForceClamp(f_max_n=f, global_hard_cap_n=GLOBAL_HARD_CAP_N)
        while not env.is_done():
            w, _ = clamp.clamp(Wrench(0, 0, f, 0, 0, 0))
            obs, outcome = env.step(w)
        if outcome in (TaskOutcome.SUCCESS, TaskOutcome.BROKEN):
            return outcome
        f = min(f * escalate, GLOBAL_HARD_CAP_N)
        env.apply_recovery(recovery, {})
        env.observe()
    return outcome


def test_robust_object_seats_without_breaking():
    seed = 0
    for s in range(20):
        e = PlaceStackEnv(PlaceStackEnvConfig())
        e.reset(_cfg("sturdy_mug", 160.0, s))
        if not (e._edge_active or e._tilt_active):
            seed = s
            break
    out = _press(PlaceStackEnv(PlaceStackEnvConfig()), _cfg("sturdy_mug", 160.0, seed), f_max=60.0)
    assert out == TaskOutcome.SUCCESS


def test_overpress_breaks_fragile():
    env = PlaceStackEnv(PlaceStackEnvConfig())
    out = _press(env, _cfg("glass_bowl", 22.0, seed=1), f_max=GLOBAL_HARD_CAP_N)
    assert out == TaskOutcome.BROKEN
    assert env.current_failure_mode() in {"over_press", "edge_load", "tip"}


def test_under_budget_does_not_seat():
    env = PlaceStackEnv(PlaceStackEnvConfig())
    out = _press(env, _cfg("ceramic_plate", 26.0, seed=2), f_max=1.0)
    assert out != TaskOutcome.SUCCESS


def test_breakage_uses_hidden_fbreak_only():
    lo = _press(PlaceStackEnv(PlaceStackEnvConfig()), _cfg("glass_bowl", 8.0, 3), f_max=20.0)
    hi = _press(PlaceStackEnv(PlaceStackEnvConfig()), _cfg("glass_bowl", 60.0, 3), f_max=20.0)
    assert lo == TaskOutcome.BROKEN
    assert hi != TaskOutcome.BROKEN


def test_failure_mode_labels():
    env = PlaceStackEnv(PlaceStackEnvConfig())
    seen = set()
    rng = random.Random(0)
    for _ in range(60):
        _press(env, _cfg("glass_bowl",
                         OBJECT_REGISTRY["glass_bowl"].sample_f_break(rng),
                         rng.randint(0, 99999)),
               f_max=18.0, recovery="wiggle_search")
        if env.current_failure_mode():
            seen.add(env.current_failure_mode())
    assert seen
    assert seen <= {"over_press", "edge_load", "tip", "under_seat"}


def test_signature_encodes_place_features_without_fbreak():
    env = PlaceStackEnv(PlaceStackEnvConfig())
    seed = None
    for s in range(50):
        env.reset(_cfg("glass_bowl", 22.0, s))
        if env._edge_active or env._tilt_active:
            seed = s
            break
    assert seed is not None
    cfg = _cfg("glass_bowl", 22.0, seed)
    clamp = ForceClamp(f_max_n=10.0, global_hard_cap_n=GLOBAL_HARD_CAP_N)
    obs = env.reset(cfg)
    hist = []
    while not env.is_done():
        w, _ = clamp.clamp(Wrench(0, 0, 10.0, 0, 0, 0))
        obs, _ = env.step(w)
        hist.append(obs.contact_step)
    sig = SignatureEncoder().encode(hist[-50:], gripper="franka_panda")
    assert (sig.lateral_bias != "none") or (abs(sig.torque_z_Nm) > 0.0)
    assert "break" not in str(sig.as_dict()).lower()


def test_recovery_clears_disturbance():
    env = PlaceStackEnv(PlaceStackEnvConfig())
    seed = None
    for s in range(50):
        env.reset(_cfg("glass_bowl", 22.0, s))
        if env._edge_active or env._tilt_active:
            seed = s
            break
    assert seed is not None
    env.reset(_cfg("glass_bowl", 22.0, seed))
    for _ in range(20):
        env.apply_recovery("rotate_align", {})
        if not (env._edge_active or env._tilt_active):
            break
    assert not (env._edge_active or env._tilt_active)


def test_heuristic_budget_glass_below_mug():
    setter = BudgetSetter(client=HeuristicLLMClient())
    glass = setter.set_budget(OBJECT_REGISTRY["glass_bowl"].identity, TASK)
    mug = setter.set_budget(OBJECT_REGISTRY["sturdy_mug"].identity, TASK)
    metal = setter.set_budget(OBJECT_REGISTRY["metal_plate"].identity, TASK)
    assert glass.F_max_N < mug.F_max_N
    assert glass.F_max_N < metal.F_max_N
    assert glass.F_max_N <= GLOBAL_HARD_CAP_N


def test_heuristic_budget_is_in_safe_band_for_glass():
    env = PlaceStackEnv(PlaceStackEnvConfig())
    env.reset(_cfg("glass_bowl", 22.0, 0))
    seat = env._seat_confirm_n()
    setter = BudgetSetter(client=HeuristicLLMClient())
    fmax = setter.set_budget(OBJECT_REGISTRY["glass_bowl"].identity, TASK).F_max_N
    assert seat < fmax < OBJECT_REGISTRY["glass_bowl"].f_break_mean_n


def test_heuristic_recovery_picks_realign_on_lateral_bias():
    client = HeuristicLLMClient()
    resp = client.call({
        "call": "select_recovery", "attempt": 0, "F_max_N": 10.0,
        "signature": {"lateral_bias": "+x steady", "torque_z_Nm": 0.0,
                      "slip_events": 0, "peak_lateral_N": 4.0},
    })
    assert resp["action"] == "rotate_align"
    assert resp["keep_F_max_N"] == 10.0


def test_build_client_heuristic():
    c = build_client({"backend": "heuristic"})
    assert c.name() == "heuristic"
