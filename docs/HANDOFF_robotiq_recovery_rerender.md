# HANDOFF — Robotiq recovery video re-render (issue #28 follow-up)

**Date:** 2026-07-06 ~03:30 UTC, updated ~06:30 UTC. **Branch:** `task3`,
repo `/workspace/FORGE-plus_task3`.
**Everything described here is UNCOMMITTED** (env + 3 scripts).
Never `git add -A` (untracked mp4s); stage explicit paths.

## UPDATE 2026-07-06 ~06:30 UTC — RETRAIN DONE, post-recovery seat SOLVED in smoke

The re-train directive is complete and the smoke seats post-recovery. Render
loop 16 in flight. Root causes found & fixed since the section below was written
(all uncommitted, in this working tree):

1. **Trainer never ported the robotiq staging cfg** (setup 4000/warmup 100/
   raw-parse) → first 256-env run had every episode time out untouched
   (frozen `dist 0.769  Fins 0.0` stats). `replicate_physics=True` also DROPS
   the four-bar loop joints. Both fixed in `scripts/train_pick_place.py`
   (robotiq block, incl. `forge_no_term=True` + `episode_length_s=45`).
2. **Synchronized-episode training design**: staging state is global python
   bools, so training uses truncation-only dones (all envs reset in lockstep)
   + a full-reset re-arm in `_reset_idx` (predrive/drive-mode/aim state).
3. **Ep-2+ reset bugs** (found via 2-episode 4-env smokes — ALWAYS smoke 2 full
   episodes): (a) the seat→OSC handoff zeroes arm drive stiffness — snapshot
   (`_rq_drive_kp/_kd`) + restore at reset; (b) stale
   `set_joint_effort_target` from the last OSC step keeps applying through the
   next episode's staging (finger overdriven to 2.8 rad) — zero efforts at
   reset; (c) live-phase violence pushes joints past hard limits and drives
   can't pull them back → ARM-ONLY teleport at full reset
   (`write_joint_state_to_sim(default, 0, joint_ids=_arm_ids)`) — the teleport
   contract's four-bar breakage is about GRIPPER-DOF writes; arm-only writes
   are safe (verified: ep-2 grip_ang 0.702) and make every episode's staging
   bit-identical.
4. **forge_no_term reward hacks**: retract_prog paid +2/step forever for a
   FLUNG bottle (policy learned to throw) → gated on `in_cell`; dist shaping
   clamped to 2 m; obs clamped (ee_p/base_to_goal ±2 m, ft ±100 N); per-step
   reward clamped (-25, 60). Physics explosions in no-term episodes are
   CONTAINED (bounded pipes), not prevented — judge runs by rew/succ trends.
5. **Mid-demo truncation ghost**: demo `episode_length_s=120` truncates at
   7200 steps but robotiq attempts cost up to 2375 steps EACH → k_max≥4
   truncated MID-DEMO (bottle teleported back to the shelf; the old
   "attempt N: 0.0 N timeout" ghosts). Fixed: 360 s in
   `run_recovery_insertion.py` + `render_recovery.py`.
6. **DEAD-GATE REGRESSION (the actual loops 3–15 killer)**: the uncommitted
   live-aim refactor left the centered re-approach ARMING under
   `if self._rq_centered:` — unreachable. The whole post-recovery centered
   machinery (aim override, ±0.04 fence, 3 N rim gate, 500-substep re-descent)
   never engaged in ANY loop 3–15 take. Fixed: arming moved back to the
   recovery falling edge; `[rq-recontact]` in RQ_TRACE=1 proves it fires.

**Checkpoint:** `checkpoints/task3_forge_robotiq.pt` (256 env × 300 it, run 5,
log `/workspace/logs/rq_train_full5.log`, final ep: 89% clean releases, 0 brk).
**Smokes:** nominal (jam 0) SUCCESS in 1 attempt; jam 0.08 SUCCESS in 2 attempts
TWICE (smoke85/86: wedge ~14 N < F_brk → rotate_align → [rq-recontact] →
attempt-1 seat in ~92 steps). Memory file: `task3-robotiq-retrain`.
**In flight:** render loop 16 (`/workspace/logs/render_rq_loop16.log`,
CKPT=task3_forge_robotiq.pt). After success: verify frames → trim → deliver →
commit (add `scripts/train_pick_place.py` to the commit list below).

## The task

User caught a physics inconsistency in the delivered robotiq recovery video
(`docs/videos/task3/forge_recovery_robotiq.mp4`): **the HUD contact force exceeded
F_brk**, which must never happen for the fragile-glass episode. Task: fix the physics
so forces stay under break, re-render the video honestly, re-trim it to open with the
robot holding the bottle, deliver, and commit. Old (approved-but-over-break) take is
backed up: `/workspace/render_takes/forge_recovery_robotiq_take_001_OLD_overbreak.mp4`.

Constraints: franka env stays intact; only learned-policy manipulation in renders
(scripted approach positioning is OK if HUD-labeled); gripper axis perpendicular to
bottle axis, neck between fingers; video starts with the bottle already held.

## Where it stands

### SOLVED — the over-break force (the actual complaint)

Root cause (proved with per-substep traces, `RQ_TRACE2=1` env var): the RTX render
pipeline steps physics *between* control substeps (capture intervals), so the rim
press dug deeper each capture while the force-authority freeze re-baselined at the
new depth — a ratchet 8.8 → 17–31 N that per-substep guards can't see. Fixes, all in
`forge_plus/isaac_pick_place_env.py` (each has a long comment at the site):

1. **12-substep policy handoff** after the rim gate latches (was 60). The whole
   runaway lived in that scripted countdown; the learned policy unloads over-force in
   ~10 substeps (seen live: 27.5 → 6.5 N).
2. **Hard fragile ceiling**: `cf > 1.5×budget` ⇒ z target =
   `torch.maximum(existing_target, ee_z + 0.008)`. MAXIMUM, never overwrite — an
   `ee+0.002` overwrite CUT the normal 4.8 N up-pull to 1.2 N and made 29 N runaways.
3. **Contact slow zone**: asymmetric z-delta clamp — downward steps ≤ 0.003 in the
   last ~2 cm above wedge contact (`_zref + 0.075`), *uncentered descents only*.
   Slowing the symmetric `lam` instead throttles the restoring force and the arm
   sinks (26.5 N, smoke 60). Velocity is the only knob that reaches inside a capture
   interval.
4. **`scripts/render_recovery.py` gauge gate**: `ok` now also requires
   `peak < sampled F_break` — `env._broke` is disarmed during setup, which is exactly
   how the original over-break take passed. Bad takes auto-reroll in the loop.

Result: attempt-0 wedge lands 13.5–21.6 N in renders, bottle KEPT (was 17.7–31.8 N
with slip-out/fling every take). Smoke reference: a0 = 13.53 N, jam fires, always.

### UNSOLVED — post-recovery re-approach never seats in RENDER

~70 rendered post-recovery attempts across loops 3–15: **zero seats** (smokes seat
routinely). The bottle re-lands 6–12 cm off (usually +y), presses 0.3–3 N — under the
rim gates (1.5/3.0 N) and the jam threshold (6 N) — and the attempt times out; abort
at attempt 4. In the current config failing takes KEEP the bottle (no fling/drop), so
render-loop rerolling is safe, just so-far fruitless.

Root cause (current best theory, strongly supported): the **learned policy + arm
posture are out-of-distribution for the robotiq**. The insertion policy
(`checkpoints/task3_forge_entrance.pt`) was trained with the franka panda hand; the
robotiq port reuses it with geometry compensation, but the 2F-140's long four-bar
fingers, perpendicular-axis grip and ~40° hang change the in-contact dynamics. Render
physics-stepping perturbations push it further OOD. Also the 7-DOF arm settles into a
different elbow branch under render-speed swings (posture drift).

**USER DIRECTIVE (2026-07-06): re-train the policy for the robotiq gripper.**

### Retraining — just started

- Trainer: `scripts/train_pick_place.py --gripper robotiq_2f140 --forge_release
  --forge_obj 0` (same script that produced the franka forge checkpoints;
  `--forge_release` = 8-dim action incl. learned release, hybrid retract).
- `checkpoints/task3_robotiq_2f140.pt` exists but is 06-21 vintage (pre-FORGE, from
  `train_skill.py`) — NOT usable.
- **Blocker to watch**: the robotiq episode pipeline is single-env by construction —
  `_rq_lift_ok`, `_rq_drive_mode`, `_rq_centered` are PYTHON BOOLS (global) and ~12
  gates index `[0]`. Deterministic staging keeps parallel envs near-synchronized, so
  env-0 gates are *approximately* right — desyncs become PPO sample noise. Whether
  that's tolerable is being tested RIGHT NOW:
  - 4-env, 2-iteration training smoke running:
    log `/workspace/logs/rq_train_smoke1.log`, ckpt `checkpoints/task3_rq_trainsmoke.pt`.
  - If it steps and shows sane diagnostics (`fdist`, `n_rel`, `n_badrel` in the log
    lines), scale to a real run, e.g. `--num_envs 256 --iterations 300` overnight
    (W&B project `forge-plus-task3`; key auto-read from `/workspace/.jr_notes`).
  - If multi-env breaks the pipeline, the vectorization audit starts with those three
    python bools and the `[0]` gates (grep `_rq_lift_ok|_rq_drive_mode|_rq_centered`
    and `\[0\]` in the robotiq sections).
- After training: point the recovery scripts at the new checkpoint
  (`run_recovery_insertion.py --policy ...`; `render_recovery.py` reads env `CKPT`),
  smoke (`--jam 0.08 --obj 0 --k_max 6`), then render loop.

## Key numbers / calibrations (do not regress)

- Jam detector: `thresh = max(6.0, 0.18×F_max)`, jam_window 40, progress < 2 mm.
- F_max (glass budget) 8.8 N; F_break sampled ~19–25 N per episode (HUD shows it).
- Rim gates: 1.5 N instantaneous (wedge), 3.0 N (centered) + sustained-press counter
  (60 substeps > 1.5 N) + near-funnel guard (bottle within 0.12 of cell center).
- Grip: close target 0.785, neck stall ~0.705 (0.785 free-close = bottle LOST; seat
  validation gap must be ~0.70). NEVER overdrive past 0.785 (four-bar squirt).
- Bottle: 0.30 kg ⇒ resting weight reads cf ≈ 2.85–2.94 N. Constant 2.85 in a trace
  = bottle dropped, resting on the rack. Constant 0.00 = hovering, no contact.
- Recovery maneuver carrot: `c.lam` (0.025). 0.012–0.015 breaks the maneuver's
  correction (attempt-1 seat lost).
- Nullspace posture stiffness: raised 15 → **80** at first recovery (robotiq only,
  runtime-mutated `self._osc._nullspace_p_gain/_d_gain` in `apply_recovery`).
  NON-monotonic: 80 best (landing 3.4–8.5 N), 200 worse (0.74 N), 15 worst (0.36 N).
- Post-recovery base-aim: LIVE per-substep `ee + (rack − bottle)` via
  `_rq_freeze_aim()` (helper has a lost-bottle guard: rack-dist > 0.45, |b−ee| > 0.45,
  z < 0.25 → nominal aim; held |b−ee| ≈ 0.25 and GRAZES 0.30 in swings — don't
  tighten). Fence stays NOMINAL-centered, ±0.04 chimney when centered.

## Dead ends (all reverted, each documented by an in-code comment)

- Finger overdrive past 0.785 (stepped or ramped) → four-bar pads tilt, bottle squirts.
- Symmetric slow lam anywhere → restoring-force starvation, arm sinks through contact.
- Hard-ceiling backoff by target OVERWRITE → weakens up-pull, worse runaway.
- Staged up-over-down re-approach after recovery → full-speed swing flings the bottle
  (render loop 10).
- Aim-following fence, falling-edge frozen aim, alpha=0.02 low-pass aim — every aim
  filter either parked the descent sub-gate or hardened the landing; the P-push
  drifts the descent AFTER any snapshot.
- Grip-joint (>0.75) as held/lost test — the drive joint flutters past 0.75 while
  holding; use bottle-to-EE distance.

## Process lessons (cost a day to learn)

- **GPU PhysX run-to-run variance dominates attempt-level outcomes.** Attempt-0 stats
  reproduce bit-identically; attempt-1+ outcomes flip between runs of identical code.
  Never conclude from single-run smoke A/Bs on this flow. (Smokes 61–63's direct
  attempt-1 seats could not be reproduced later by reconstructing their configs.)
- Render-vs-smoke divergence: capture intervals both SPEED tracking (less sag) and
  blind the controller during the interval. Anything marginal in smoke fails in
  render; anything that depends on sag-damping (e.g. live-aim pendulum feedback) can
  go unstable in render (loop 12 fling).
- Instrument before theorizing: `RQ_TRACE2=1` gives per-substep `[rq-c]` lines
  (ee/bottle/grip/rimz/setup); `RQ_FORCE_REC=<action>` pins the recovery selector
  (debug smokes only, never renders).
- Kill Isaac procs with self-excluding patterns (`render_recover[y]`) — plain pkill
  matches your own shell. One Isaac proc at a time on the pod. If Isaac wedges in
  startup with 0 GPU compute apps, it was a transient from a force-kill: relaunch.
  There is NO Xvfb on this pod and renders don't need it (ignore setup_runtime's
  claim).

## Ops quick reference

- Smoke: `HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99
  PYTHONPATH=/workspace/FORGE-plus_task3 /workspace/.venv/bin/python
  scripts/run_recovery_insertion.py --gripper robotiq_2f140 --jam 0.08 --obj 0 --k_max 6`
  (~12 min; logs `/workspace/logs/rq_env_smoke*.log`, last = smoke81).
- Render loop: `bash scripts/render_recovery_until_success.sh <takes> robotiq`
  (~30 min/take; loop logs `/workspace/logs/render_rq_loop*.log`, last = loop15,
  killed for training; takes in `/workspace/logs/render_recovery_robotiq_takeN.log`;
  frames `/workspace/frames_recovery/`). On success auto-copies to
  `docs/videos/task3/forge_recovery_robotiq.mp4`.
- On a good take: verify frames (gauge under F_brk the whole way, "under break"
  label, JAM card → recovery → seat → release), re-locate the grasp moment, trim
  (`ffmpeg -ss <t> -i take -c:v libx264 -pix_fmt yuv420p -crf 20` via imageio_ffmpeg),
  SendUserFile, then commit EXPLICIT paths: `forge_plus/isaac_pick_place_env.py`,
  `scripts/render_recovery.py`, `scripts/run_recovery_insertion.py`,
  `docs/videos/task3/forge_recovery_robotiq.mp4` (+ this doc), with trailers
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com> and the Claude-Session URL.
- Memory files: `task3-robotiq-force-fix` (this effort), `task3-robotiq-port`,
  `task3-recovery-wired`, `never-script-manipulation`.

## UPDATE 2026-07-06 ~16:45 UTC — DELIVERED (take 91) + post-recovery render stack fixed

**Delivered:** `docs/videos/task3/forge_recovery_robotiq.mp4` = take 91 trimmed to the
held-bottle opening (`-ss 3.0`). RESULT SUCCESS by every script gate: peak 16.1 N <
F_brk, break=False, seat VALID (held, in-cell), placed=True (learned release + retract,
bottle vertical). Episode: seeded wedge jam -> retract_and_reapproach -> attempt-1
timeout (divider grind, honest 0-force hover ~80 s of the video) -> retract -> attempt-2
LEARNED seat + release. First-ever post-recovery seat in a render (loops 3-15 all
failed; root causes below).

**The post-recovery render failure was a STACK of five mechanisms** (each verified by
instrumented takes 97->89; details in memory `task3-robotiq-retrain`):
1. Settle gate (takes 97/96): rec_end now latches `_rq_need_settle`, holds the EE at a
   SNAPSHOTTED pose until bottle |v_xy|<0.08 for 30 substeps (600-substep livelock cap).
2. rec_end re-anchors ALL persistent command state (the 3rd instance of the regime-
   handoff rule): `_joint_centers` (null-space), `_rq_q_des` AND `_rq_quat_tgt`
   (hold-what-you-reached, both halves — re-anchoring only q_des woke the slerp and it
   outran the wrist's physical slew cap 12 Nm / 80 damping).
3. ori_k 400 -> 110 for the POST-RECOVERY (centered) setup only: 400 (and 300) are
   closed-loop UNSTABLE in the orientation channel under the render pipeline's
   held-torque stepping (oerr ratchets, saturated wrist wrench escapes through the
   shoulders = the outward spiral of loops 3-15 / takes 95-97).
4. Adaptive-gravity integrator: DOWNWARD-ONLY while centered (`_rq_ag_cap` = rec_end
   snapshot). Free adaptation railed +1.2 (take 92 ceiling-pin); a full freeze locked
   the smoke's staging-inflated +0.40 in as over-lift (smokes 90/91 ABORT, 0-force
   float). Shed-only fixes both.
5. Aim stays FROZEN at rec_end (take 90: an alpha=0.01 low-pass chased the wrist lean
   and seated the bottle tilted -> toppled on release; frozen aim seats vertical).

**Smoke 93 = final code passes** (wedge fail -> rotate_align -> attempt-1 seat, same as
golden smokes 85-89). NOTE: take 91 was rendered one iteration BEFORE the freeze->cap
change (its code differed ONLY in ag freeze vs cap); takes 90/89 with the exact final
code failed on seat-quality lotteries (tilt-topple / pinch slip) — the post-recovery
seat in renders remains a per-take lottery. The video is honest end-to-end (HUD labels
scripted vs learned, timeout attempt left uncut).

**Known residual (documented, out of scope):** the wrist sits saturated at its 12 Nm
effort clamp against the mis-modeled gravity moment of the long loaded 2F-140 (PhysX
gravity comp is short for the loop-jointed articulation; `ag` patches z-force, not the
wrist moment). The principled fix is real payload compensation (tool-mass wrench at the
EE COM through J^T, like a real Franka tool config) — would de-saturate the wrist and
likely remove the whole instability class.

**Ops addenda:** `pgrep -f "render_recover[y]"` still matches a LAUNCHER shell whose
command string contains the plain script name — check the python pid with `kill -0`.
FAIL takes do NOT encode an mp4 (frames persist in /workspace/frames_recovery until the
next run). F_break is sampled per-take (28 N take 91, 19 N take 90). `[rq-t]` trace
line (RQ_TRACE2=1) prints the ACTUAL commanded carrot: tgt/raw/aim/desc/st1/stl/oerr.

---

## UPDATE 2026-07-07 — ROOT CAUSE FOUND AND FIXED: the render-vs-smoke divergence itself

The "known residual" above is now RESOLVED — and the diagnosis was wrong in an
instructive way. The wrist saturation was NOT static payload gravity:

- **Payload measurement** (RQ_TRACE3 wrench fit, `[rq-pc]`): the 2F-140 subtree is
  0.70 kg and the bottle 0.30 kg — the un-modeled static wrench is ~3 N / ~0.7 Nm,
  nowhere near the 12 Nm wrist clamp. In the SMOKE no arm joint ever saturates and
  oerr stays <= 0.15 the whole episode. The saturation/lean/grind pathology existed
  ONLY in renders, i.e. it was the held-torque capture stepping all along.
- J^T payload compensation IS now implemented (cfg.rq_pc_grip / rq_pc_obj scales,
  PhysX-measured masses, force+moment rows about the jacobian's own reference,
  `RQ_TRACE3` diagnostic). It ships OFF (scales 0.0) — correct physics, but not the
  fix; kept as instrumentation and for future gripper ports.

**THE FIX — pause-capture (render_recovery.py `_grab`)**: capture frames WITHOUT
stepping physics, making the render regime IDENTICAL to the smoke:

1. `/app/player/playSimulations=False` around the capture `app.update()`s (the carb
   setting Isaac's replay tools use). NOT `timeline.pause()` — play() is processed on
   a later update, so the next `env.step()` livelocked at 109% CPU (take 98, killed).
2. **Rigid objects stop render-syncing while paused** (articulations keep syncing via
   fabric during `env.step`'s sim.step; rigid-object transforms sync only in the
   update-loop physics pass that pausing skips). Take 99 passed every physics gate
   while the RENDERED bottle floated frozen at its stale pose.
   `physx update_transformations(updateToUsd/FastCache)` does NOT fix it (renderer
   reads Fabric; probes v2-v4). The fix is a direct **usdrt Fabric write** of the
   bottle's world pose from `env._obj.data.root_pose_w` before each capture
   (`Rt.Xformable` world attrs; probe v4/v5: pixel centroid matches the unpaused
   calibration sub-pixel; scale correctly inited from USD by SetWorldXformFromUsd).
3. Beware the ghost: with pause-cap on, ANY un-flushed rigid prim renders at a stale
   pose. Take 99's frame-0 "parked bottle" was itself the spawn-pose ghost; in a
   correct take the parked bottle is OCCLUDED behind the rack early on — do not
   misread that as a broken visual (take 100 was killed on exactly that misread; its
   surviving frames f_0150/f_0275 show the bottle correctly in-grip and seated).

**Results with the fixed pipeline** (same checkpoint task3_forge_robotiq.pt, same
committed gains): takes 99/100/101 all reproduce the smoke pattern — wedge jam ->
rotate_align -> attempt-2 seat, max oerr 0.099 rad (vs 0.26-0.37 take 91), peak force
~14 N, NO timeout attempts, ~13.5 s of video instead of 96 s with an 80 s dead middle.
Seat gap 0.7064/0.7063 across takes — near-deterministic. The old detuned gains
(ori_k 110 post-recovery, settle gate, ag cap) are KEPT: the smoke validates them and
the render now IS the smoke; no retrain needed (the policy is no longer OOD at render
time). The "five mechanisms" above remain documented history — they were compensations
for the now-removed divergence.

Diagnostic tooling: `scripts/probe_pausecap.py` (boots the RTX pipeline, free-falls
the bottle, phases through flush candidates against an unpaused pixel-centroid
calibration). Probe cost ~10 min vs ~40 min per render take.
