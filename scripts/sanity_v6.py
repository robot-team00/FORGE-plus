#!/workspace/.venv/bin/python
"""Sanity suite v6 - uses env._episode_phase.value for phase-aware commands.
current_phase returns BASE CLASS TaskPhase; _episode_phase is our EpisodePhase.
"""
import sys
sys.path.insert(0, '/workspace/FORGE-plus_task3')
from forge_plus.envs.franka_fragile_place_env import FrankaFragilePlaceEnv
from forge_plus.envs.base_assembly_env import EpisodeConfig, Wrench, TaskOutcome

GRASP_PHASES = {'grasp_approach','grasp_close','grasp_lift','transport','place_approach'}

def make_env(object_key, f_break_n, seed=0):
    cfg = EpisodeConfig(
        object_key=object_key, task_name='fragile_place', gripper='franka_panda',
        f_break_n=f_break_n, max_steps=500, disturbance_seed=seed,
    )
    env = FrankaFragilePlaceEnv()
    env.reset(cfg)
    return env, cfg

def run_episode(env, fz_grasp, fz_place):
    for _ in range(500):
        phase = env._episode_phase.value   # internal EpisodePhase, not base TaskPhase
        fz = fz_grasp if phase in GRASP_PHASES else fz_place
        _, out = env.step(Wrench(0, 0, fz, 0, 0, 0))
        if out != TaskOutcome.IN_PROGRESS:
            break
    return env.get_episode_metrics()

PASS_ALL = True
print("=== SANITY SUITE v6 ===")

# A: gentle both phases, f_break >> f_max -> success
env, _ = make_env('glass_bowl', f_break_n=22.0)
m = run_episode(env, fz_grasp=6.0, fz_place=6.0)
ok = (not m.broken) and m.success
print("A SAFE   ->", "PASS" if ok else "FAIL",
      " grip:", round(m.peak_grip_force_n,1), " place:", round(m.peak_place_force_n,1))
if not ok: PASS_ALL = False

# B: gentle grasp(5N) < f_break(7N) < f_max(~10N) capped place(~10N) > f_break -> place break
env, _ = make_env('glass_bowl', f_break_n=7.0)
m = run_episode(env, fz_grasp=5.0, fz_place=80.0)
ok = m.broken and m.broken_at_phase == 'place'
print("B PLACE-BREAK ->", "PASS" if ok else "FAIL",
      " broken:", m.broken, " phase:", m.broken_at_phase,
      " grip:", round(m.peak_grip_force_n,1), " place:", round(m.peak_place_force_n,1))
if not ok: PASS_ALL = False

# C: hard grasp: step2 grip=8N (ramp 4N x 2) > f_break=7N -> grasp break
env, _ = make_env('glass_bowl', f_break_n=7.0)
m = run_episode(env, fz_grasp=80.0, fz_place=80.0)
ok = m.broken and m.broken_at_phase == 'grasp'
print("C GRASP-BREAK ->", "PASS" if ok else "FAIL",
      " broken:", m.broken, " phase:", m.broken_at_phase, " grip:", round(m.peak_grip_force_n,1))
if not ok: PASS_ALL = False

# D: sturdy_mug, high f_break, safe -> success
env, _ = make_env('sturdy_mug', f_break_n=180.0)
m = run_episode(env, fz_grasp=80.0, fz_place=80.0)
ok = (not m.broken) and m.success
print("D STURDY  ->", "PASS" if ok else "FAIL",
      " broken:", m.broken, " phase:", m.broken_at_phase)
if not ok: PASS_ALL = False

print("OVERALL:", "ALL_PASS" if PASS_ALL else "SOME_FAIL")
sys.exit(0 if PASS_ALL else 1)
