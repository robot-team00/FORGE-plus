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

A Franka Panda arm performs a **fragile PLACE** of a realistic, breakable kitchen object (a
LIBERO wine bottle). The current demonstrated variant is a **learned wine-cellar peg-in-hole
insertion + safe release**: a **FORGE PPO policy** descends the bottle into a rack cell under
force control (contact stays gentle, ~0–4 N), keeps it vertical, and **decides when to let go**,
leaving it standing upright while the arm retracts.

> ### ▶️ Start here for the real demo: **[`07-learned-place-release.md`](07-learned-place-release.md)**
> That is the **learned** policy (algorithm, reward shaping, the gripper-open bug, the render
> pipeline, and the learned-vs-scripted HUD). The insertion in
> [`05-wine-cellar-insertion.md`](05-wine-cellar-insertion.md) was an earlier **scripted
> scaffold** — now superseded.
> Video: **[`docs/videos/task3/forge_release.mp4`](../videos/task3/forge_release.mp4)**.

<div align="center">
  <img src="../videos/task3/forge_release.png" width="560" alt="Learned FORGE insertion: the PPO policy descends the wine bottle into the rack cell under gentle force; HUD shows the green LEARNED force-guided-insertion phase and the contact-force gauge">
  <br><em>The learned PPO policy descending the bottle into the cell (HUD: green “LEARNED — force-guided insertion”; force gauge well under the break limit).</em>
</div>


The earlier variant set the bottle on an **open shelf** and righted it about the contact pivot
(force-conditioned policy + "contact-then-verticalize"). Both run with **real physics** — the
object is a dynamic rigid body held by a genuine friction grip, never teleported or kinematically
attached during the carry. The placement strategy is selected by `cfg.place_strategy`:

| `place_strategy` | What it does | Status |
|---|---|---|
| `"insert"` (current) | Wine-cellar peg-in-hole — insert the bottle into a rack cell | ✅ bottle ends vertical, doc 05 |
| `"throw_upright"` | Open-shelf place + contact-then-verticalize ramp | ✅ stands (~0.8°), doc 02 §9 |
| `"extrinsic"` | Learned gentle roll-up (extrinsic dexterity) | ⚠️ never converged (abandoned) |

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
| [`04-libero-objects.md`](04-libero-objects.md) | LIBERO reference, the OBJ→USD→rigid-wrap import pipeline, the **procedural wine-rack USD build**, asset layout, and which object shapes to pick. |
| [`05-wine-cellar-insertion.md`](05-wine-cellar-insertion.md) | Wine-cellar peg-in-hole scene, asset pipeline, photorealism. **⚠️ The insertion shown there was a SCRIPTED scaffold (zero policy action + base-aim), not a learned policy — being replaced by a learned FORGE PPO policy.** |
| [`06-recovery.md`](06-recovery.md) | **Force-signature LLM recovery, closed loop in Isaac.** The task-agnostic `RecoveryLoop`, the env hooks (jam detection, force signature, recovery primitives), the soft force ceiling (stay under budget), the induced-jam scenario, and the verified jam→recover→seat result. |
| [`07-learned-place-release.md`](07-learned-place-release.md) | **✅ The current, LEARNED policy — start here.** FORGE PPO that descends the bottle into the cell (force-guided insertion), learns *when to release* (8-dim action), and safely places it upright before the arm retracts. The algorithm, reward shaping, the finger-open bug, the numpy-1.26 render-killer, the render harness, and the learned-vs-scripted HUD + force gauge. |

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
| `/workspace/assets/libero/wine_rack/wine_rack.usd` | The procedural 3×3 wine-cellar rack (outside the repo; `assets/` is gitignored). See doc 04 §7. |
| `docs/videos/task3/pick_place_eval_001.mp4` | Latest demo render (wine-cellar insertion). |

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

# Render the wine-cellar insertion SCAFFOLD (scripted; being replaced by a learned policy):
/workspace/.venv/bin/python scripts/render_pick_place.py
# -> docs/videos/task3/pick_place_eval_001.mp4
```

> ⚠️ The current insertion render is a **scripted scaffold**: it drives **zero policy action** and
> the env's hand-coded **base-aim** + phase waypoints do the work — **no learned policy**. This is
> being replaced by a **learned FORGE-style PPO policy** (`cfg.forge_mode`, in progress) that
> outputs the EE motion and learns the alignment/insertion from force, with no scripted waypoints.

## Current status (2026-07-01)

- ✅ **Learned FORGE insertion + safe release** (`cfg.forge_mode` + `cfg.forge_release_mode`) —
  **the real goal, done.** A PPO policy descends the bottle **into the cell** under force control
  (gentle ~0–4 N), keeps it vertical, and **learns when to release** (8th action dim); it then
  places it upright and the arm retracts. ~68% success, **break 0**, no scripted waypoints /
  base-aim for the insertion. Full write-up: **[`07-learned-place-release.md`](07-learned-place-release.md)**;
  video [`docs/videos/task3/forge_release.mp4`](../videos/task3/forge_release.mp4). The HUD labels
  every phase **LEARNED (green)** vs **SCRIPTED (orange)** in real time.
- 🗄️ **Wine-cellar peg-in-hole insertion (scripted scaffold)** — the earlier stopgap (zero policy
  action + base-aim + phase waypoints) that built the scene/asset. **Superseded** by the learned
  policy above. See [`05-wine-cellar-insertion.md`](05-wine-cellar-insertion.md).
- ✅ Real friction grasp of a realistic LIBERO wine bottle (gripped by the neck); RTX render
  (kitchen, wood counter + rack); gripper genuinely **releases** and the arm **retracts**.
- ✅ (Prior variant) Open-shelf place with a **learned** force-conditioned policy (obs=34) trained
  to **succ 1.0 / break 0.0** — this one *was* policy-driven. See
  [`02-grasp-and-placing.md`](02-grasp-and-placing.md#9-standing-placement-solved-via-contact-then-verticalize).
- ✅ **Force-signature LLM recovery — closed loop, driving the LEARNED policy** (proposal §07): the
  task-agnostic `RecoveryLoop` now runs the **trained FORGE policy** as its skill (`step_skill` →
  `env._skill_policy`). On a wedged insertion it reads a text force signature (no vision, no
  `F_break`), the LLM picks a recovery, applies it within the same `F_max`, and the learned policy
  re-inserts. Verified: induced jam caught at **~17 N** (≪ 180 N break) → `retract_and_reapproach`
  → the learned policy seats it. **SUCCESS in 3 attempts.** The loop also lifts the imperfect
  learned insertion (~68 %/attempt) to reliable success on the clean case. One loop for all task
  envs. **Rendered on the FRAGILE glass object** (break ~23 N): jam caught at **16.1 N**, LLM picks
  `rotate_align`, the learned policy re-inserts and seats it (still held), then **releases (learned)
  and retracts clear** — 9/9 headless seats, breaks 0. Video:
  [`forge_recovery.mp4`](../videos/task3/forge_recovery.mp4). See [`06-recovery.md`](06-recovery.md).
- ⚠️ Learned gentle "extrinsic-dexterity" roll-up was attempted and **never converged**
  (hard-exploration RL) — superseded by the cell-geometry insertion.

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
