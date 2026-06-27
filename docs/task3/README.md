# Task 3 — Fragile Object Placement (Franka + Isaac Lab)

> **Naming note:** this is a *placement* task — the robot does **not** pick the object off a
> surface. The code keeps `pick_place` names (`FrankaPickPlaceEnv`, `*_pick_place.py`,
> gym id `FORGE-PickPlace-v0`) because the env *implements* the full pick→place pipeline, but we
> run it in `place_only=True` mode. See the "Pick vs place" box below.

> Read this first. This folder documents **everything** about FORGE-plus Task 3: the
> algorithm, the grasp/place physics, the headless RTX rendering pipeline, and how the
> realistic LIBERO objects are imported. It is written for engineers *and* for other AI
> coding agents picking up this work — every section records the *why*, not just the *what*,
> and calls out the hard-won gotchas that each cost hours to find.

## What this task is

A Franka Panda arm performs a **fragile PLACE**: it holds a realistic, breakable kitchen object
(a LIBERO wine bottle), carries it to a shelf, and **sets it down gently** — keeping the contact
force under a per-object "break force" budget the whole time. The policy is **force-conditioned**
(it sees a safe-force budget as an observation) and is trained with PPO. Everything runs with
**real physics** — the object is a dynamic rigid body held by a genuine friction grip, never
teleported or kinematically attached during the carry.

> ### ⚠️ Pick vs place — read this so you don't overclaim
> We run the env in **`place_only=True`** mode: the episode **starts already holding the object**.
> The object is seated into the closed gripper during a brief warmup (a short teleport) and then
> held by **real friction** for the rest of the episode. The robot does **not** reach down and
> close on an object from a surface — **there is no learned/executed pick maneuver.** When these
> docs say "grasp," they mean the **friction hold** (genuine physics, no kinematic attach), not a
> pick. The full pick pipeline (`PRE_GRASP → DESCEND → GRASP → LIFT`) exists in the code
> (`place_only=False`) but is **not trained or rendered**. The "Pick & Place" name is historical
> (`FrankaPickPlaceEnv`, `*_pick_place.py`); the demonstrated/trained task is the fragile **place**.
> The render HUD still reads "Fragile Pick & Place" — relabel to "Fragile Place" when convenient.

## Where it runs

Isaac Sim / Isaac Lab on a remote RunPod GPU pod. See the repo-root `CLAUDE.md` for pod
orientation (clones, shared venv, asset dirs, push/auth). Run all Isaac code with
`/workspace/.venv/bin/python`.

## Document map

| Doc | Covers |
|-----|--------|
| [`01-algorithm.md`](01-algorithm.md) | FORGE algorithm, OSC controller, the 7-phase state machine, force budgets (LLM cache), reward shaping, PPO training, and the reward-hovering fix. |
| [`02-grasp-and-placing.md`](02-grasp-and-placing.md) | Grasp physics (why flat faces work and round ones fail), the friction seat, geometric place detection, the gripper-release bug, the retract, and the standing-placement (topple) problem + object-selection rules. |
| [`03-rendering.md`](03-rendering.md) | Headless RTX live-physics rendering, the "app.update() steps physics" gotcha, the render↔physics **sync bug** that froze the object, the proven render loop, camera, and HUD. |
| [`04-libero-objects.md`](04-libero-objects.md) | LIBERO reference, the OBJ→USD→rigid-wrap import pipeline, asset layout, and which object shapes to pick. |

## Key files

| Path | Role |
|------|------|
| `forge_plus/isaac_pick_place_env.py` | The env: `PickPlaceEnvCfg`, `FrankaPickPlaceEnv`, `PickPlacePhase`. Spawn, OSC, phases, grip, reward, dones, reset. |
| `forge_plus/skills/policy_network.py` | `ForceConditionedPolicy`, `PolicyConfig`, `ValueNetwork`. |
| `scripts/train_pick_place.py` | PPO training entry point. |
| `scripts/render_pick_place.py` | Headless RTX renderer → demo mp4 with HUD. |
| `llm/budget_cache.json` | Cached per-object force budgets (avoids re-querying the LLM each run). |
| `checkpoints/task3_wine_bottle.pt` | Trained policy (gitignored — not in the repo; regenerate via training). |
| `/workspace/assets/libero/wine_bottle/wine_bottle_rigid.usd` | The graspable object (outside the repo; `assets/` is gitignored). |
| `docs/videos/task3/pick_place_eval_001.mp4` | Latest demo render. |

## Quickstart

```bash
# After every pod (re)start, once:
cd /workspace/FORGE-plus_task3
gcc -shared -fPIC -o /usr/local/lib/libGLU.so.1 scripts/libglu_stub.c && ldconfig
bash scripts/setup_runtime.sh
export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 PYTHONPATH=/workspace/FORGE-plus_task3

# Train (warm-start optional). ~256 envs fits the bottle's heavier collision mesh.
/workspace/.venv/bin/python scripts/train_pick_place.py \
    --num_envs 256 --iterations 600 --gripper franka_panda \
    --ckpt checkpoints/task3_wine_bottle.pt

# Render a demo (drives the trained policy):
/workspace/.venv/bin/python scripts/render_pick_place.py
# -> docs/videos/task3/pick_place_eval_001.mp4
```

## Current status (2026-06-27)

- ✅ Real friction grasp of a realistic LIBERO wine bottle (gripped by the neck).
- ✅ Carry → gentle place (contact ≤ ~1 N, budget 8.8 N, break ~22–27 N).
- ✅ Policy trained to **succ 1.0 / break 0.0**.
- ✅ Headless RTX render with the object correctly following the gripper on screen.
- ✅ Gripper genuinely **releases** the bottle and the arm **retracts** away.
- ✅ **Bottle places STANDING upright** (final tilt ~0.8°, base on the shelf) via **contact-then-
  verticalize** — descend soft, then ramp OSC orientation stiffness on shelf-contact to right the
  bottle about the contact pivot. Real physics, no retrain. See
  [`02-grasp-and-placing.md`](02-grasp-and-placing.md#9-standing-placement-solved-via-contact-then-verticalize).

## The one-paragraph mental model

The episode runs a fixed **phase state machine** (LIFT→TRANSPORT→PLACE_DESCEND→RELEASE in
`place_only` mode). Each phase has a **waypoint**; an **Operational-Space Controller (OSC)**
drives the end-effector toward `waypoint + policy_delta` with bounded per-step motion. The
gripper open/close is **scheduled by phase** (not the policy). The object is a **dynamic rigid
body** seated into the gripper during a short warmup and then held by **real friction**. The
policy's job is to modulate the approach/descent so the **contact force stays under the budget**
(fragility) while still completing the place. Reward = sparse success cliff + small shaping; the
critical lesson was that *any farmable per-step bonus* makes the policy hover forever instead of
finishing (see [`01-algorithm.md`](01-algorithm.md#reward-shaping-and-the-hovering-trap)).
