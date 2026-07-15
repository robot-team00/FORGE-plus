# Task 1 on the Robotiq 2F-140 — the complete write-up

**Full-cycle, fully learned-manipulation gear insertion on a second gripper:
table pick → force-budgeted insertion → learned release, with force-signature
LLM recovery (including a physical place-on-table regrasp) when the grip is
disturbed.** All manipulation during evaluation is either the learned policy
or labeled scripted *staging* (transport between poses); nothing about the
object is ever teleported, pinned, or scripted during contact-rich phases in
the final table-pick flow.

Date: 2026-07-14. Branch `task3`, commits `ed4c113 … 588e75b`.
Companion docs: [`task1_jam_recovery.md`](task1_jam_recovery.md) (recovery
baseline tables, Franka + 2F-140), [`task1_baselines.md`](task1_baselines.md)
(budget baselines), [`task1_robotiq_handoff.md`](task1_robotiq_handoff.md) /
[`task1_rq_followon_handoff.md`](task1_rq_followon_handoff.md) (the session
handoffs this arc executed), [`../checkpoints/README.md`](../checkpoints/README.md)
(checkpoint manifest), [`task3/08-robotiq-2f140.md`](task3/08-robotiq-2f140.md)
(the original 2F-140 asset build on the bottle task).

Videos: **[`videos/task3/gear_clean_robotiq.mp4`](videos/task3/gear_clean_robotiq.mp4)**
(28 s clean episode) and
**[`videos/task3/gear_recovery_robotiq.mp4`](videos/task3/gear_recovery_robotiq.mp4)**
(83 s recovery episode with two place-on-table regrasps).

---

## 1. Headline results

Task: insert the FORGE GearMesh medium gear (Ø35.5 mm hub, bore fit band
r 5.15 mm — **0.4 mm diametral clearance**) onto the middle shaft of the
3-shaft base plate. Two object classes: `abs_gear` (fragile, F_break 38±5 N,
identity-only LLM budget ≈ 10 N insertion) and `steel_gear` (robust, 100 N
budget). Gripper: Franka arm + Robotiq 2F-140 four-bar adaptive gripper.
Success = strict TRUE seat (gear origin at seated z within 0.6 mm, upright,
on the correct shaft). Every gate below is the deterministic policy mean,
256 episodes per class, and every gate reports breaks against the hidden
per-episode F_break draw.

| milestone | checkpoint | result |
|---|---|---|
| Clean insertion, single ckpt, both classes | `task1_gear_rq_uni.pt` | abs **256/256, 0 breaks**, peak 15.9 N mean / 18.5 p95; steel **256/256, 0 breaks**, 35.6 / 57.6; 0 over-budget episodes |
| + learned release (pinned staging) | `task1_gear_rq_uni_rel.pt` | abs **256/256, 0 breaks, 0 bad releases**, peak 15.7 / 18.5 / 20.1 max; steel **256/256, 0, 0**, 35.7 / 59.3 / 69.6 |
| Full table-pick flow + learned release | `task1_gear_rq_uni_rel_tp.pt` | abs 64/64 smoke, 0 breaks, 0 bad releases, peak **5.4 N mean / 5.8 max** — the best force numbers of the entire project |
| Jam recovery sweep (5 mm in-grip slip, abs, 25 eps/cell) | `task1_gear_rq_uni.pt` | **ours 40%** / vision_llm 28% / heuristic 0% / press_harder 0% (futile) / none 0% with 20% breaks — full table + caveats in [`task1_jam_recovery.md`](task1_jam_recovery.md) |
| Recovery with physical table regrasp | `task1_gear_rq_uni_rel_tp.pt` | 5-episode smoke: 3/5 seated+released (2 breaks, 0 timeouts) |

Release success is stricter than insertion success: **released + gear
standing seated + hand retracted clear**, judged after the policy's own
act[7] crosses the open threshold — the environment never opens the gripper
for it.

The interesting inversion: the *unpinned* table-pick flow produces far
*gentler* insertions (5.4 N mean peak) than the pinned staging (15.9 N).
The seat-window pin used by the legacy staging leaves a small systematic
grip offset that the policy pays for during the funnel search; a real
friction pick centers the hub in the pads better than the pin ever did.

---

## 2. What is learned vs what is scripted

Per the project's standing rule (no scripted manipulation in evals/renders),
the split is explicit and HUD-labeled in every video:

**Learned (the policy, `ForceConditionedPolicy`, obs 34):**
- The force-guided insertion: funnel search, alignment, descent, seating
  press — everything from the entrance hand-off until the seat. This is the
  contact-rich part where force decides success vs breakage.
- The release decision (act[7], `release_obs_head`): *when* to open after
  seating. The environment executes the open only when the learned head
  crosses its threshold.

**Scripted staging (labeled orange in the HUD):**
- The table pick close/lift sequence and the drive-side carry to the
  entrance (transport between known poses — upstream of the studied
  problem, exactly like the benchmark's episode-start setup).
- Recovery *primitives* (retract, place-on-table, re-pick trajectories) —
  the menu the LLM selects from. Selection is signature-driven; execution
  is a fixed maneuver, as in the proposal.
- The post-release hand retract.

The pick itself, though scripted in *timing*, is physically real: the
gripper closes on the resting gear and everything after is genuine friction
— no kinematic attach, no pose pinning, at any point in the table-pick flow.

---

## 3. The plant: making the 2F-140 controllable at 0.4 mm clearance

The port inherited the verified Franka+2F-140 USD from the bottle task
(S3 mirror recipe, outer-finger lock baked in — see
[`task3/08-robotiq-2f140.md`](task3/08-robotiq-2f140.md)). The gear task
then exposed three plant-level defects that made *every* policy fail before
learning was even in question (root-caused in `ed4c113`, comments in
`forge_plus/isaac_gear_env.py`):

1. **Symmetric knuckle drives.** Driving `finger_joint` alone let the
   four-bar close asymmetrically under contact; the paired knuckle must be
   driven symmetrically or the pinch walks the part.
2. **Anchored, command-continuous OSC target.** The wrist target must be
   anchored at hand-off and moved only by increments; re-deriving it from
   live FK injects the arm's standing error as a step input (standing
   errors are load-bearing — command continuity at hand-off is what the
   trained policy expects).
3. **Slide-search contact authority.** The funnel search needs enough
   lateral authority *while in contact*; the bottle-tuned gains stalled the
   search on the plate.

Grasp geometry: TCP 0.214 m along hand +z; pads grip the Ø35.5 hub with pad
center ~gear origin +0.026…0.032 (mid-hub-band, *below* the top — the
"pads-above-hub is physically impossible" realism rule is asserted in every
render via the `pad_dz` readout).

---

## 4. Why PPO-only fails here, and the method that works

The funnel tolerance (~1.5 mm capture radius, 0.25 mm bore clearance at
μ 1.0) sits far below workable exploration noise: action std 0.12 already
breaks 40/64 gears and seats 1/320. Nine PPO-on-robotiq runs (all kept in
the manifest as evidence) span the full hyperparameter escalation and none
ever learned to seat; tiny-std PPO polish (std 0.05) *destroys* a working
BC policy within 25 iterations — the 1/std² gradient amplification moves
the mean too coarsely for this clearance.

**The working method: scripted-expert demos → DAgger → BC.**
1. `scripts/collect_gear_demos.py --driver` collects expert demonstrations
   (scripted experts are allowed for *training* data only).
2. DAgger rounds relabel the policy's own visited states with the expert.
3. `scripts/bc_gear_mean.py` fits the deterministic mean.

Lineage `bc → bc2 → bc3 (abs 256/256) → bc4 (steel 256/256) → bc7` (best
single: steel 32/32, abs 84%). The two classes tug the policy in opposite
directions (abs sits on its own break floor: F_break min ~20 N vs 15–25 N
contact transients), and balanced-refit attempts see-sawed.

**Unification = weight soup.** `task1_gear_rq_uni.pt = 0.15·bc3 + 0.85·bc7`
(`scripts/make_soup.py`). Valid because bc7 was warm-started from bc3 (same
basin); every α in 0.10–0.75 passes triage, steel peak force is monotone in
α, and α = 0.15 minimizes the joint force tails. One checkpoint, both clean
gates 256/256, 0 breaks.

---

## 5. The learned release (`release_obs_head`)

Goal: the policy decides when to let go — act[7], as in the task3 bottle
work — without disturbing the proven arm behavior. Four designs were tried;
the failure modes are general lessons for adding a discrete decision head
to a trained continuous policy:

1. **Frozen-trunk head on all timesteps** — learns "open iff already open"
   (post-open observations are giveaways) and never fires at deployment.
   *Post-open samples are poison* — the env latches the first +1, so
   everything after carries no decision information. Drop them always.
2. **Pre-open-only labels on the frozen trunk** — 27% FPR / 52% FNR: the
   arm-trained trunk provably *discards* the seat-state signal (it never
   needed it).
3. **Trunk fine-tune + arm-distillation loss** — the two losses fight; FNR
   pins at 50%, coverage 0.
4. **Winner: input-skip head.** `PolicyConfig.release_obs_head` adds
   act[7] = Linear(raw obs ⊕ f_cmd). A raw-observation logistic-regression
   diagnostic first proved the signal is *linearly separable* in the raw
   obs (phase one-hot, ee_z, wrench): FNR 0, zero-FPR window coverage
   255/256. Train only that Linear (arm dims + trunk stay bit-identical to
   `uni`), fold the feature standardization into the weights, and set the
   zero-FPR threshold **in deployment space after folding** — fp32 fold
   error scales with logit magnitude, so thresholding before the fold
   silently shifts the operating point.

Two more traps with teeth:
- **Collection settle matters**: labeling the release 20 steps after the
  geometric seat let the policy's seating press continue — 36% of fragile
  gears broke *before the release could fire*. `settle_steps=5` → 256/256.
- **The head reads the phase one-hot, so it is staging-specific.** A head
  trained on pinned-staging data never fires in the table-pick flow (64/64
  press-at-budget-forever). Recollect per deployment staging:
  `collect_gear_release.py --table_pick` → `task1_gear_rq_uni_rel_tp.pt`.
- The release latch is one-way; in recovery flows, `rel_cmd` must be gated
  off during recovery/place phases (the head reads "gear lowered to table"
  as "seated → open") and the latch cleared when a regrasp re-arms.

Trainer: `scripts/train_release_head.py`; data:
`scripts/collect_gear_release.py` (uni mean drives the arm, scripted expert
supplies release timing — training data only); gate:
`scripts/eval_gear_insert.py --release`.

---

## 6. Table-pick staging (`cfg.rq_table_pick`)

The last non-physical assist in the pipeline was the seat-window pin that
placed the gear into the closed gripper at episode start. `rq_table_pick=1`
replaces it end-to-end: the gear spawns *resting on the table* at a pick
spot, the gripper descends open, closes a real friction grip on the hub
(grip stall angle 0.573 vs 0.784 closed-on-air — the in-hand
discriminator), lifts, carries to the entrance, and hands off to the
policy. Thirteen probe iterations; the durable lessons:

- **Pick spot by rotation, not new pose engineering**: the pick xy is the
  entrance xy rotated `rq_pick_dth = −0.33` rad about the robot base, so
  the pick arm pose is the *proven* entrance pose with j1 += dth.
- **THE OSC LATERAL TRAVERSE IS CURSED.** A wrist-twist disturbance walks
  the end-effector +y 15–30 mm during any OSC settle/hover/traverse. It
  killed the carry, the settle hover, and (later) the recovery place
  traverse — three independent appearances. Every precise transport in
  this env runs on the **stiff joint drives** (interpolate `_rq_arm_cmd`
  toward the target pose), with OSC engaged only for the policy's
  contact-rich work at the entrance. This is the load-bearing design
  choice of the staging.
- Grip band tolerance is tight: ±6 mm z slack put the pads 6 mm low on the
  hub → 3.5° carry lean → the policy *refuses* the hand-off (correctly —
  that tilt cannot thread the bore). `_ztol` 6 mm, xy 20 mm, and the
  arrival gate demands upright (cos 3°), still (<0.03), z < 0.448 and
  xy < 4.5 mm before the OSC hand-off.
- Never hand off wide: pressing from >15 mm off-axis back-drives the gear
  out of the pads.
- Servo details that bit: the gear-aim visual servo must be gated on
  actually-holding (else a closed-on-air gripper chases a perpetual
  carrot); joint clamps and the pitch formula inherited from the bottle
  scene were silently wrong for a top-down gear grip.

---

## 7. Recovery on the 2F-140, including the physical table regrasp

The disturbance: a 5 mm in-grip slip. As on the Franka, position snaps back
but orientation doesn't — the gear ends **tilted in the pinch**, which
cannot thread a 0.4 mm-clearance bore, so the policy correctly refuses and
hovers at zero force. The contactless-hover branch of `is_failure()` makes
that refusal visible; the force-signature → frozen-LLM chain routes a
*recurring* hover to `regrasp` — the only maneuver that fixes an in-grip
tilt. F_max is never raised; `keep_F_max_N` is overwritten server-side.

Sweep table (25 eps/cell, abs class) and its honest caveats live in
[`task1_jam_recovery.md`](task1_jam_recovery.md). Highlights: only the
signature chain ever restores an insertable grip (ours 40% vs 0% for the
heuristic that never reads the hover signature); the force signature is
worth 40% vs 28% against a random-menu vision proxy; `press_harder` is
*futile* here rather than destructive (the tilted bore never takes load),
while `none` is destructive (5/25 breaks at up to 50.5 N — the policy's own
descent eventually presses the tilted bore into the shaft tip). On steel
the same chain recovers 25/25 with 0 breaks.

**In the table-pick flow, `regrasp` becomes physically real**: place the
tilted gear back on the table, open, re-pick it upright, carry back, resume
— no seat-window pin anywhere. Ten smoke rounds of root-causing produced
these rules (each deterministic-reproducible; details in
`isaac_gear_env.py`):

- Drives may engage only in **confirmed free space** (contact force < 0.5 N)
  — the timeout-to-drives path once crushed a still-wedged gear at 213 N.
- **Droop-compensate every drive engage** (the 2×desired−reached rule):
  commanding the live joint readings sags the arm ~3.5 cm under load and
  grinds the held gear on the shaft tip.
- **j1-arc first at lifted altitude**; descend only over clear table —
  descending any pose over the shaft is a tip crush.
- Gate the place descent on the **realized gear z**, not the arm pose: a
  held gear rides ~30 mm lower than a resting one, so driving fully to the
  pick pose presses it into the table (319 N observed).
- Detector bookkeeping: arrival cooldowns must be repick-only (muzzling the
  detector after ordinary re-approaches let a tilted press ramp to 61 N);
  the near-cell ceiling and re-approach setup windows must account for
  table-flow altitudes and slews; concurrent recovery machinery
  (recontact re-descent) must be gated out while the place machine owns
  the flow.
- Eval `episode_length_s` must exceed the step cap (a 45 s truncation
  mid-recovery resets the sim and masquerades as teleportation).

Smoke with everything wired: 3/5 seated + learned-released (2 breaks —
wedge presses at 20–27 N catching low F_break draws during re-approach
descents, the known refinement frontier; 0 timeouts).

---

## 8. Videos and rendering

`scripts/render_gear_recovery.py` renders both takes with the pausecap RTX
pipeline (capture without stepping physics: `playSimulations` toggle +
usdrt flush of the gear and all 17 robot links; proxy-gear pattern for the
pod's ghost-draw quirk; see [`RENDERING.md`](RENDERING.md) and
[`task3/03-rendering.md`](task3/03-rendering.md)). HUD: green LEARNED vs
orange SCRIPTED phase labels, live force gauge vs F_max and F_break,
recovery attempt counter. Renderer honesty rules: takes end at the
release+hand-clear success latch; no jam detector runs in no-slip takes
(matching the clean eval protocol); `pad_dz` is verified mid-hub-band in
every take before delivery.

- `gear_clean_robotiq.mp4` (28 s): table pick → carry → learned insertion
  (seat at k2291) → learned release → hand clear.
- `gear_recovery_robotiq.mp4` (83 s): same start, 5 mm slip injected → 5
  recovery attempts including two full place-on-table regrasp cycles →
  seat at k6840 → learned release → hand clear.

---

## 9. Reproduce

```bash
export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 \
       PYTHONPATH=/workspace/FORGE-plus_task3
PY=/workspace/.venv/bin/python

# Clean gates (pinned staging), both classes:
$PY scripts/eval_gear_insert.py --gripper robotiq_2f140 --obj 0 \
    --episodes 256 --ckpt checkpoints/task1_gear_rq_uni.pt          # abs
$PY scripts/eval_gear_insert.py --gripper robotiq_2f140 --obj 1 \
    --episodes 256 --ckpt checkpoints/task1_gear_rq_uni.pt          # steel

# Release gates (learned release, pinned staging):
$PY scripts/eval_gear_insert.py --gripper robotiq_2f140 --obj 0 \
    --episodes 256 --release --ckpt checkpoints/task1_gear_rq_uni_rel.pt

# Full table-pick flow + learned release:
$PY scripts/eval_gear_insert.py --gripper robotiq_2f140 --obj 0 \
    --episodes 64 --release --table_pick \
    --ckpt checkpoints/task1_gear_rq_uni_rel_tp.pt

# Jam recovery (single cell / full sweep):
$PY scripts/eval_gear_jam.py --gripper robotiq_2f140 --recovery ours \
    --obj 0 --slip_mm 5 --episodes 25 --ckpt checkpoints/task1_gear_rq_uni.pt
bash scripts/sweep_jam_recovery.sh checkpoints/task1_gear_rq_uni.pt 25 robotiq_2f140

# Recovery with the physical table regrasp:
$PY scripts/eval_gear_jam.py --gripper robotiq_2f140 --recovery ours \
    --obj 0 --slip_mm 5 --episodes 5 --release --table_pick \
    --ckpt checkpoints/task1_gear_rq_uni_rel_tp.pt

# Videos (RTX, pausecap):
TABLE=1 RELEASE=1 SLIP_MM=0 bash scratch/run_render_clean.sh   # clean
TABLE=1 RELEASE=1 SLIP_MM=5 bash scratch/run_render_final.sh   # recovery
# (or drive scripts/render_gear_recovery.py directly with those env vars)
```

Training from scratch: demos via `scripts/collect_gear_demos.py --driver`,
DAgger + BC via `scripts/bc_gear_mean.py`, soup via `scripts/make_soup.py`,
release data via `scripts/collect_gear_release.py [--table_pick]`, release
head via `scripts/train_release_head.py`. Checkpoint provenance for every
file: [`../checkpoints/README.md`](../checkpoints/README.md).

---

## 10. Open items

1. Proper 256-episode/class release gates for `uni_rel_tp` (currently a
   64-episode clean smoke and a 5-episode jam smoke).
2. A 25-episode table-mode `ours` recovery cell for the sweep table —
   table-flow cycles are ~5× longer, so timeout economics differ from the
   pinned-flow row.
3. Steel-class table-pick smoke.
4. Gentler post-retract re-approach descent: the residual abs breaks are
   20–27 N wedge presses catching low F_break draws.
