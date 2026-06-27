# 01 — Algorithm: FORGE, OSC, phases, force budgets, reward, RL

This is the control + learning stack for Task 3. File: `forge_plus/isaac_pick_place_env.py`
(env), `forge_plus/skills/policy_network.py` (nets), `scripts/train_pick_place.py` (PPO).

## 1. FORGE in one paragraph

FORGE ("Force-conditioned grasping") is a contact-rich manipulation approach where the policy
is **conditioned on a force budget** and learns to keep interaction forces under it. We give
the policy a normalized **safe-force command** `f_cmd` as an extra observation, randomize it
per episode, and penalize force overshoot. Inference can then trade off speed vs. gentleness
by changing `f_cmd`. Our reward design follows FORGE's shipped code and the reward-shaping
literature (see §6): **no positive per-step bonus**, force as an observation + an overshoot
penalty, and a one-time success cliff.

## 2. The OSC controller (task-space impedance)

The arm is driven by Isaac Lab's `OperationalSpaceController` (OSC) — inertia-decoupled
task-space impedance + gravity compensation + null-space posture control. This replaced a
hand-rolled Jacobian-transpose controller.

- The proximal-joint actuator PD is **disabled** so the OSC's effort targets actually move the
  arm: `panda_shoulder`/`panda_forearm` get `stiffness=0`, but keep `damping=80` (without the
  velocity damping the torque-controlled joints buzz at high frequency).
- **Important:** the finger actuator (`panda_hand`) is **left at stiffness=2e3** (a stiff
  position drive). This matters for the gripper-release bug — see
  [`02-grasp-and-placing.md`](02-grasp-and-placing.md#the-gripper-release-bug).
- The OSC holds a fixed end-effector **orientation** (`ee_quat_des`, captured once at warmup).
  This is why the carried object keeps a constant tilt — the root cause of the topple problem.

Per-step target (in `_apply_action`):

```
target_w = ee_pos_w + clamp( (waypoint + action[:3]*act_range) - ee_pos_w, -lam, lam )
```

- `action[:3]` is an **EE-position delta** (the policy's main output), scaled by `act_range=0.05`.
- `lam=0.025` clips motion to ≤2.5 cm/step — FORGE's bounded per-step motion. The arm therefore
  moves *slowly and smoothly*; episodes are long (`episode_length_s=30` → ~1800 sim steps).
- `act_dim=7`, `obs_dim=34`.

## 3. The 7-phase state machine

`PickPlacePhase`: `PRE_GRASP=0, DESCEND=1, GRASP=2, LIFT=3, TRANSPORT=4, PLACE_DESCEND=5,
RELEASE=6`.

Each phase has a **waypoint** (`_phase_waypoint_world()`). Phase advances when the EE
"reaches" the waypoint (`_get_dones`), with phase-specific tolerances:

- Sweep phases (`LIFT`, `TRANSPORT`, both at `transport_z`) require reaching the waypoint **xy**
  (within 0.06 m) — otherwise the arm skips them instantly and never visibly moves across.
- Contact phases (`DESCEND`, `GRASP`) complete on **contact** (`cf > contact_eps`) because the
  compliant surface stops the fingers above the sub-surface waypoint z.
- `PLACE_DESCEND` completes **geometrically**: the object's base reaching the shelf
  (`|cup_bot - shelf_top| < place_settle_tol`), *not* by force (see §4 / place detection in doc 02).

**`place_only` mode (default `True`):** the episode starts at `LIFT` already holding the object
(seated during warmup) and runs `LIFT → TRANSPORT → PLACE_DESCEND → RELEASE`. This is the
fragile-**place** task. `start_phase = LIFT if place_only else 0` in `_reset_idx`.

Key cfg geometry (defaults; verify in code as they get tuned):
`transport_z=0.72`, `shelf_top_z=0.50`, `rack_x=0.35`, `rack_y=0.30` (non-zero y → lateral reach
demo), `mug_grip_z=0.12` (grip height up the object from its base origin),
`reach_tol=0.08`, `place_reach_tol=0.03`, `place_settle_tol=0.02`, `settle_steps=15`,
`warmup_substeps=10`.

## 4. Force budgets (the "LLM" cache)

Each fragile object class has a sampled **break force** `f_break` (mean±std) and a derived safe
**command** force `f_cmd`. These live in `llm/budget_cache.json` (e.g. `glass_bowl ≈ 8.8`,
`ceramic_plate`, etc.) and are loaded by `_load_or_query_budgets()`. The cache exists so runs
don't re-query the LLM (Ollama) every boot — **if the cache is missing the import can hang on
the Ollama call.** Keep `llm/budget_cache.json` in the repo.

- `f_cmd` is exposed to the policy normalized: `f_cmd_norm() = f_cmd / 120`.
- Break check is active only at force-monitored phases (`GRASP`, `PLACE_DESCEND`):
  `broke = (cf > f_break) & grace & force_active`.
- The HUD shows the budget class; the *rendered mesh* (wine bottle) is labelled separately —
  `glass_bowl`'s ~22 N break is a coherent stand-in for glass. The label was renamed in the
  renderer (`OBJ_KEY = "wine_bottle"`) for honesty.

## 5. Contact force measurement

The contact sensor is on `panda_(hand|leftfinger|rightfinger)`, **filtered to the object and
the rack**, and we read `net_forces_w` of the force applied to the object/rack. Reading the bare
hand sensor gave ~0 N (the hand never touches the surfaces) and finger-on-finger self-contact
polluted it. **Consequence:** the sensor *cannot* see the object resting on the shelf through a
gentle grip (the force nets ~0), which is why place detection is **geometric**, not force-based.

## 6. Reward shaping and the hovering trap

This was the single most important learning. **Symptom:** training success collapsed to 0 while
return kept rising to ~120 — the policy learned to **hover at `PLACE_DESCEND` forever**.

**Root cause:** *farmable per-step bonuses* whose horizon-integral beats the one-time success
reward. The old reward paid `+0.1 * phase_index` every step (a "living bonus" for being in a
later phase) and `+1.0 * force_in_window` every step. Hovering in a late phase, in-window,
collected those forever — worth more than ending the episode with a single success.

**Fix (what worked), grounded in FORGE's actual code + Ng 1999 potential-based shaping + the
reward-hacking literature:**

- **Delete** both farmable per-step bonuses.
- Keep a one-time `+2` phase-**advance** bonus (paid only on the step a phase is entered).
- Success **cliff**: `+20 * succeeded`.
- Breakage penalty: `-6 * broke`.
- Small **time penalty** `-0.02` per step (makes finishing strictly better than stalling).
- Plus smoothness and place-error shaping (`-2.5 * place * h_err`, `-0.5 * excess`).

**Result:** success → 1.0 and *stays*; return is **bounded** (~2–11, not 120); the deterministic
policy mean descends and completes the gentle place. The general rule for any agent touching this
reward: **never add a positive per-step term that can be farmed by not finishing.** Prefer
potential-based shaping (differences of a potential) and put the real value in the terminal cliff.

Other FORGE lessons noted but not all applied: potential-based shaping for the descent term, a
start-height (SBC) curriculum, and partial-episode bootstrapping + stall early-termination.

## 7. Success detection (and a render-only gotcha)

`_get_dones`: success requires reaching `RELEASE` and staying **gentle** (`cf < f_cmd`) for
`settle_steps=15` steps (a settle counter that increments while gentle, decrements otherwise).
On success the env terminates → auto-resets.

Gotcha for renderers/evaluators: by the time you read `env._succeeded` after `env.step`, the
auto-reset has already cleared it, so a single-env display often shows `succ=False` even though
the place succeeded. Training sees it correctly because `extras["n_succ"]` is written **before**
the reset.

## 8. PPO training

`scripts/train_pick_place.py` — vanilla PPO with GAE.

- Defaults: `--num_envs 512` (use **256** for the wine bottle — its convex-decomposition
  collision is ~3× heavier than a primitive), `--iterations 600`, `--rollout 32`,
  `--epochs 5`, `--minibatch 8192`, `--lr 3e-4`, `--gamma 0.99`, `--lam 0.95`, `--clip 0.2`.
- `--resume <ckpt>` warm-starts the policy weights (same obs/act dims) — converges much faster.
- Logs every 5 iters: `it rew ret succ brk fps`; checkpoints every 5 iters to `--ckpt`.
- The wine-bottle policy (`checkpoints/task3_wine_bottle.pt`) converged to **succ 1.0 / brk 0.0**
  by ~it 400, return bounded ~6–11, ~5600 fps at 256 envs.
- After `TRAIN_DONE`, the Isaac process often **hangs in RTX shutdown** — the checkpoint is
  already saved, so force-killing the PID is safe.
- Run **one Isaac process at a time**: concurrent boots/`SIGKILL`s wedge the shared GPU/caches
  and make subsequent boots hang. A bare `SimulationApp` boots in ~6 min alone.
