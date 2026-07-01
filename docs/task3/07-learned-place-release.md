# 07 — Learned FORGE Place + Safe Release (the real policy)

> This is the document for the **current, learned** demonstration: a FORGE-style PPO policy
> that **inserts** a wine bottle into a rack cell and then **safely releases** it, leaving it
> standing upright while the arm retracts. It supersedes the scripted scaffold of
> [`05-wine-cellar-insertion.md`](05-wine-cellar-insertion.md). Everything here records the
> *why* and the gotchas — several of which cost hours (or a whole GPU day) to find.
>
> **Deliverable:** [`docs/videos/task3/forge_release.mp4`](../videos/task3/forge_release.mp4)
> **Checkpoint:** `checkpoints/task3_forge_entrance.pt` (gitignored — retrain, see §8)
> **Render:** `RELEASE=1 RENDER_MINIMAL=0 scripts/render_forge_min.py`

---

## 0. TL;DR — what is learned vs scripted

The HUD in the video labels this honestly, per phase, in real time (green = LEARNED, orange =
SCRIPTED):

| Phase | HUD label | Learned? | What happens |
|---|---|---|---|
| Approach | `SCRIPTED — approach positioning (up-over-down)` | ✗ | The env's scripted "up-over-down" path drives the EE to a hand-off pose **at the cell entrance** (bottle base ≈ 0.51, cell top 0.50). **No contact.** |
| Insertion | `LEARNED — force-guided insertion + when-to-release (PPO policy)` | ✅ | The **PPO policy** descends the bottle **into the cell** under force control (contact stays ≈ 0–4 N, far below break), keeps it vertical, and **decides when to let go** (8th action dim). |
| Release + retract | `SCRIPTED — gripper-open + retract-to-clear (release was LEARNED)` | mixed | The **release decision** was learned; the gripper-open **actuation** (a direct joint write, see §4) and the moderate retract-to-clear are scripted post-task clearing. |

The **learned** contribution is the force-guided descent/insertion and the release timing — the
core FORGE skill. The scripted parts are the pre-approach positioning and the post-task
gripper-open + retract. This split is deliberate and is the honest scope of the demo.

---

## 1. Where this came from

The original Task-3 "insertion" was a **scripted scaffold** (zero policy action + a hand-coded
`base-aim` + phase waypoints — see doc 05). We replaced it with a genuinely **learned** policy
in three layers, each of which is a section below:

1. **Learned force-guided insertion** — a force-conditioned PPO policy (FORGE) outputs EE deltas
   through the OSC and inserts the bottle. (Existing `forge_mode`; extended here.)
2. **Learned safe release** — an 8th action dimension so the policy decides *when to let go*
   (`forge_release_mode`), trained with a safe-release reward.
3. **Learned full descent** — raise the scripted hand-off to the cell *entrance* and retrain so
   the policy owns the whole descent into the cell, not just the final seat.

---

## 2. The env: `forge_mode` + `forge_release_mode`

All in `forge_plus/isaac_pick_place_env.py`.

- **Base:** FORGE ([arXiv 2408.04587](https://arxiv.org/abs/2408.04587)). Force-conditioned PPO;
  the force budget `F_cmd` is a first-class observation, contact above it is penalized, and the
  policy learns to seat *under* the ceiling. No vision — proprioception + pose + F/T only.
- **Controller:** Isaac Lab `OperationalSpaceController`, `impedance_mode="variable_kp"`. Each
  step the command is `pose(7) + stiffness(6)`; the policy's xyz delta rides on top of the
  target, motion is clamped per step (`forge_lam`) so contact stays gentle.
- **Object:** a **dynamic rigid** LIBERO wine bottle, gripped by the **neck** by real friction
  (no kinematic attach). Seated into the closed gripper during a short warmup, then held by
  physics.

### 2.1 Action space (7 → 8)

| dim | meaning |
|---|---|
| 0–2 | EE position delta (xyz), scaled by `forge_act_range`, clamped by `forge_lam` |
| 3–6 | (reserved / orientation handling via `ee_quat_des` + `ori_k`) |
| **7** | **learned gripper release** — `action[7] > 0` latches the release (one-way commit) |

`cfg.forge_release_mode = True` bumps `action_space` to 8 (set in `FrankaPickPlaceEnv.__init__`
before `DirectRLEnv` reads it). Observation stays **34-dim** — the release dim needs no extra
obs (the policy conditions release on `base_to_goal`, `obj_up`, and F/T already in the obs).

### 2.2 The hand-off geometry (the "who does the insertion" fix)

The scripted setup drives the EE **up → over the cell xy → down** to `forge_approach_z`. The
critical number is where it *stops*:

- Cell floor `cell_floor_z = 0.40`, cell **top ≈ 0.50**, gripper grips `mug_grip_z = 0.12` up
  the bottle, and the neck-grip **lean drops the base ≈ 0.17 below the EE**.
- `forge_approach_z = 0.68` → bottle base ≈ **0.51 = right at the cell entrance**, **no contact**.
- `forge_setup_steps = 80` substeps (decimation 2 ⇒ ~32 env steps after the seat) — a **short**
  scripted intro.

> **Why this matters (the honest-scope fix).** With the old `approach_z = 0.56` the setup drove
> the base to ~0.44 — **6 cm inside the cell** — *ramming* it in at **42 N** (near breaking) while
> the HUD still said SCRIPTED. The learned window was ~3 frames. Raising the hand-off to the
> entrance means the **learned policy does the full descent** (0.51 → 0.40) under gentle force.

---

## 3. Reward shaping for insert + safe release

`_forge_get_rewards` / `_forge_get_dones`. Insertion terms are the FORGE ones (doc 01); the
release terms are new. The balance is **delicate** — see the failure modes in §3.2.

**Insertion (always):**
- **PBRS progress** `keypoint_k · (prev_dist − dist)` toward the cell floor — *pure* progress, no
  discount (a discounted living reward is farmable by hovering; see doc 01).
- **Force-overshoot penalty** `−β · max(0, (cf − F_cmd)/F_cmd)` — FORGE gentle-contact.
- small smoothness/time penalties; **success cliff +50**; **break −6**.

**Safe release (`forge_release_mode`):**
- **success now requires a real drop:** `seated = in_cell & at_floor & upright & settled &
  released` (and `& hand_clear` when the retract is on). Held `release_hold` steps.
- **release-LOW penalty** `−12 · newly_released · height_above_floor` — *don't let go while high*.
  Forces a genuine descent before release (without it the policy releases at the entrance and
  lets the bottle drop).
- **release-when-descended bonus** `+10 · newly_released · (base < floor+0.06)` — actively
  rewards *committing* to the drop once low (without it the policy descends and **holds**).
- **anti-lean** `−1.5 · released · (1 − up_z)`; **post-release push** `−0.25 · released ·
  max(0, cf−3)`; **retract-progress** `+ retract_prog`; **bad-release** `−bad_release_pen`.
- **wrist stiffness** `ori_k = 110` during the learned descent (was 40) so the bottle stays
  **vertical** through the longer descent (the compliant 40 let it lean ~40°). See §3.2.

### 3.1 Success / `PLACED` definition

`_succeeded` (and the HUD `PLACED`) requires, held for `release_hold` steps:
`in_cell` (xy within 5 cm of the cell) **&** `at_floor` (base within `release_floor_tol` of the
floor) **&** `upright` (`up_z > release_upright_cos ≈ 0.90`, ~26°) **&** `settled` (low velocity)
**&** `released` (gripper commanded open) **&** `hand_clear` (EE ≥ `release_clear_dist` from the
bottle, only when the retract is enabled).

### 3.2 The two failure modes we fought (write these down)

The descend↔release balance oscillates between two local optima; the fix was structural (wrist)
plus a balanced reward, not any single knob:

| Symptom | Cause | Fix |
|---|---|---|
| **Descend-and-hold** (`rel → 0`, deterministic policy never lets go) | release penalties too strong; the +50 is sparse | add the **release-when-descended +10**; keep the release-low penalty moderate (12, not 40) |
| **Release-eager-and-shallow** (releases at the entrance, bottle drops/tips) | at-floor tolerance too loose; low-reward threshold too high | tighten the low-reward to `base < floor+0.06`; the release-low penalty scales with height |
| **Bottle leans ~40°** during the descent | compliant wrist (`ori_k=40`) lets it tilt over the longer descent | **stiffen the wrist to 110** during the learned phase |

Final policy (450 iters, warm-start): **succ ≈ 0.68**, gentle **~4 N** insertion, upright,
**break 0**. The deterministic release is still *shy* → the renderer **samples** the release dim
(§5.4).

---

## 4. The gripper-release bug (fingers never opened)

**The single most important bug in this whole effort**, and the one the reviewer caught with a
logic check ("if it released, how can the bottle move with the robot?").

- The grip is a PD **effort** target: `gforce = grip_pos_ks·(target − fpos) − grip_pos_kd·fvel`,
  written via `set_joint_effort_target`.
- The Franka finger actuator has a **near-rigid position drive (~4 kN/m)**. The effort-based
  "open" command is **overpowered** — the fingers stay clamped on the neck. So the "released"
  flag was set, but the gripper **kept holding**, and any retract just **carried the bottle** out
  (or the sim eventually flung it — `obj_z → −135`).
- **Fix:** when `self._released`, **write the finger joints open directly** with
  `write_joint_state_to_sim` (ramped over ~5 steps). This *actuates* the learned release; it is
  not a scripted skill. Diagnostic after the fix: `fingers gap 0.077` (open), `obj_z` constant at
  the floor, `eo_dist` grows as the hand backs away → the bottle genuinely stays put.

---

## 5. Rendering

### 5.1 numpy **must** be 1.26.0 — the render-killer

isaacsim 5.1 pins `numpy==1.26.0`. numpy 2.x has an incompatible C-ABI, so the compiled
OmniGraph/replicator `.so`s read arrays as **size-0**, `rgb.attach` raises
`TypeError: Unable to write from unknown dtype, kind=f, size=0`, and **every render product comes
out 0×0** (all frames EMPTY). Installing ultralytics/opencv silently upgrades numpy and breaks
**all** RTX rendering. Fix: `pip install "numpy==1.26.0"`. (Also, after a full pod restart the
base image can revert `/usr/bin/python3` to 3.10 — repoint the venv:
`ln -sf /usr/bin/python3.11 /workspace/.venv/bin/python3`.) See repo-root `CLAUDE.md` facts 6–8.

### 5.2 `scripts/render_forge_min.py` — the working harness

The full-scene `render_pick_place.py` **does not render** on this pod: its kitchen USD + textured
PBR materials leave the render product 0×0, and it enabled **DLSS/DLAA** which need **NGX**
(broken here → also 0×0). `render_forge_min.py` is built on the proven `render_task3.py` harness:

- single immediate `rgb.attach` (no retry loop), **GI stack off**, **no DLSS** (`aa/op = 1`
  TAA), overscan patch, `enable_async=false`, clean studio scene (dome + sun + fill lights, matte
  ground — no kitchen).
- **Restore `DirectRLEnv.render`** before constructing the env — `FrankaPickPlaceEnv` no-ops
  `render()`/`close()` for training speed, which otherwise leaves the RTX context uninitialized
  (render product 0×0).
- **`capture_on_play` off** — the replicator orchestrator otherwise fires an `on_update` every
  frame and crashes with an empty capture schedule (same `size-0` family).

### 5.3 `RENDER_MINIMAL=0` — the train/eval match (subtle, important)

`render_minimal=1` skips the filtered insert-sensor + compliant material. That **changes the
force observation the policy sees**, so in the render the arm behaved differently than in
training (it *held high* instead of descending). Render with **`RENDER_MINIMAL=0`** (the full
training sensor set) so the render matches the trained descent. This was why an early render of a
good policy showed it not descending — the policy was fine; the render was lying to it.

### 5.4 Sampled release + render-until-success loop

The deterministic (mean) release is shy, so the render **samples `action[7]`** from the policy's
own Normal(mean, std) while keeping the arm on the mean (smooth). `scripts/render_until_success.sh`
renders repeatedly and keeps the first rollout that reaches a **sustained `PLACED`** (prints
`RESULT PLACED`); each clip **ends ~0.7 s after PLACED** (a `TAIL` cutoff) so the static hold
doesn't pad the video.

---

## 6. The HUD

Drawn per frame in `render_forge_min.py` (PIL, over the RTX RGB):

- **Top banner:** title + `step k/N` + `state:` (`inserting` → `RELEASED` → `PLACED`).
- **CONTROL line (color-coded):** `LEARNED` in **green**, `SCRIPTED` in **orange**, with the
  live phase description (see §0). A `LEARNED`/`SCRIPTED` **legend** sits top-right.
- **Force gauge (bottom-left):** a horizontal bar of the contact force with a yellow **`F_cmd`**
  (budget) marker and a red **`F_brk`** (break) marker; the bar is green under budget, amber over
  budget, red past break. In the final video it stays **green ~0–4 N**, far from `F_brk` — the
  FORGE gentle-contact story, made visible.

---

## 7. Retract (scripted post-task clearing)

After the learned release, `forge_hybrid_retract` drives the empty hand to a **moderate clear
pose** (up ~22 cm + back ~15 cm) and **holds** — not all the way to a ceiling/home pose. A brief
`settle` first lets the freed bottle drop onto the cell floor. This is post-task clearing, not a
manipulation skill; the *decision to let go* remains the learned `action[7]`. (With
`HYBRID_RETRACT=0` the hand just lets go and holds in place.)

---

## 8. Reproduce

```bash
# once per pod restart
cd /workspace/FORGE-plus_task3
gcc -shared -fPIC -o /usr/local/lib/libGLU.so.1 scripts/libglu_stub.c && ldconfig
bash scripts/setup_runtime.sh
export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 PYTHONPATH=/workspace/FORGE-plus_task3
/workspace/.venv/bin/pip install "numpy==1.26.0"   # if RTX renders come out EMPTY

# train the learned place+release (8-dim action; warm-start from the insertion policy)
/workspace/.venv/bin/python scripts/train_pick_place.py \
    --forge_release --forge_obj 2 \
    --resume checkpoints/task3_forge_insert.pt --reset_std -0.8 \
    --ckpt checkpoints/task3_forge_entrance.pt \
    --num_envs 512 --iterations 450 --rollout 32
# metrics to watch: dist↓  bz→0.42  Fins~4N  relupz→0.9  rel>0  succ~0.6  brk 0

# render a clean take (samples the release; keeps the first sustained PLACED)
cp checkpoints/task3_forge_entrance.pt checkpoints/task3_forge_release.pt
RELEASE=1 RENDER_MINIMAL=0 MAXATT=15 bash scripts/render_until_success.sh
# -> docs/videos/task3/forge_release.mp4
```

**Key files:** `forge_plus/isaac_pick_place_env.py` (release mode, hand-off, wrist, reward,
finger-open, retract), `scripts/train_pick_place.py` (`--forge_release` + 7→8 warm-start),
`scripts/render_forge_min.py` (harness + HUD + gauge + short-clip), `scripts/render_until_success.sh`.

## 9. Honest limitations

- **~68% success**, and the deterministic release is shy (render samples it) — the render-loop
  takes several attempts to land a clean take. The full peg-in-hole + release from above is the
  hard, not-fully-solved core (matches the FORGE-notes caveat).
- Trained on the **robust** object class (`forge_obj 2`) for a stable demo; the rendered mesh is
  the LIBERO wine bottle regardless (the class only sets the hidden force budget).
- The **approach positioning** and the **gripper-open actuation + retract** are scripted (§0) —
  by design, and labeled as such in the HUD.
