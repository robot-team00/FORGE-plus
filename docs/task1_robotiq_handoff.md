# Gear insertion — Robotiq 2F-140 port: handoff for the next session

Written 2026-07-11 at the end of the session that solved jam recovery
(`037bff1`), produced the recovery-baseline table (`41a70a8`), and delivered
the recovery video (`09d9c94`). Branch `task3` in `/workspace/FORGE-plus_task3`
(ALL task1 work lives here; the `task1` branch/clone belongs to a parallel
session — do not touch it).

## Mission

Issue #26 DoD, remaining items in recommended order:

1. **Robotiq 2F-140 on the gear task** (this doc) — env runs both grippers;
   insertion + jam recovery demonstrated on the 2F-140.
2. Steel gear ≥95% clean (current mixed ckpt: abs 100% / steel 73%; careful
   short low-lr rounds, STOP at good snapshots — r2-style continuations
   collapse).
3. Budget-baselines table re-run (ours / oracle / fixed) with the current
   checkpoint on both objects.

Key invariant everywhere: **F_break never reaches the policy, encoder,
budget-setter, or recovery-selector.** Recovery never raises F_max.

## State of the world (what already works)

- **Franka gear insertion**: `checkpoints/task1_gear_sliprand.pt.it300` —
  clean gate 200/200 strict TRUE seats, 0 breaks (Fins 13.8 N mean / 15.9
  p95 / 17.6 max). Snapshots `.it100/.it200` + `task1_gear_mixed_strict.pt`
  (abs 100 / steel 73) also in `checkpoints/`.
- **Jam recovery SOLVED on Franka**: slip leaves the gear TILTED 7–10° in
  the grip (position snaps back, orientation doesn't) → contactless-hover
  branch in `is_failure()` → recurring hover routes to `regrasp` (the
  warmup seat's identity-quat write fixes the tilt) → arrival-gated
  re-approach setup → policy seats. 6/6 smoke; sweep: ours 64% / heuristic
  0% / vision_llm 32% / press_harder 4% (96% breaks) / none 0%. Full story
  + caveats: `docs/task1_jam_recovery.md`.
- **Render**: `scripts/render_gear_recovery.py` (RTX video with recovery
  HUD) and `scripts/snap_gear_pybullet.py` + `scripts/probe_grasp.py
  --dump` (CPU ground-truth stills). **The pod RTX view draws physics-
  managed rigid objects DISPLACED** (ghost) — the canonical workaround is
  in render_gear_recovery.py: hide the object's visual
  (`UsdGeom.Imageable.MakeInvisible`) and pose a plain-USD proxy from
  `root_pose_w` each capture. End takes at seat+8 frames (`forge_no_term`
  keeps pressing and will break the gear on camera).

## What the Robotiq port actually is

`forge_plus/isaac_gear_env.py` was cloned from the task3 bottle env **with
all the 2F-140 machinery intact** — asset spawn, four-bar drive config, seat
sequence, recovery branches. It is all calibrated for the BOTTLE scene
(16 mm neck at scale 0.5, rack cell, wedge staging). The port =
re-calibrate those branches for the gear scene, then train/fine-tune and
re-run the gates. Inventory of what exists (grep `robotiq` in the env):

- Asset spawn (~line 1008): `robot_cfg.spawn.usd_path =
  "/workspace/assets/isaac51/Robots/..."` — the asset built + verified in
  the task3 port (S3 mirror recipe: `scripts/build_franka_robotiq_2f140.py`,
  outer-finger lock baked in, NOTE 3).
- Actuators (~1031-1053): drive `finger_joint` only (effort 30);
  `.*_inner_finger_joint` mimic at effort 0.01; pads/outer passive — the
  four-bar owns the kinematics.
- Grasp constants (~865-935): `_grasp_tcp_d = 0.214` (grip point along hand
  +z from robotiq_base_link, probe v50), `_tcp_dz = -0.045` (bottle-specific
  hand-off lowering — RECALIBRATE for the gear's in-contact-free hand-off),
  `_rq_dest_dx` (bottle wedge staging — the gear uses the slip inducer
  instead, probably 0), kiss 0.218 / squeeze 0.785 finger_joint targets
  (calibrated on the 16 mm neck — the Ø35.5 mm hub stalls at a different
  angle: RECALIBRATE via a probe).
- Seat sequence `_rq_seatctr` / `_RQ_SEAT_HI` / pin-then-kiss-then-squeeze
  (~2823-2851): bottle-neck geometry (`_rq_grip_h`) — recalibrate for
  gripping the hub 30 mm band (pad centre at gear origin +0.032, the
  franka's `mug_grip_z`).
- Recovery branches: `_rq_aim` frozen base-aim (rec_end snapshot), nullspace
  pin at 80 (NON-MONOTONIC — 15 too weak, 200 distorts; 80 calibrated),
  `_rq_centered` deep post-recovery descent (-0.10 carrot + 3 N gate),
  12-substep handoff force ceiling, ag wind-up/freeze machinery. These were
  tuned through ~14 overnight render loops on the bottle — expect the gear
  to need the same class of tuning but NOT the same values.
- The 2026-07-11 franka recovery additions are gripper-split: the
  contactless-hover branch + noise-gated slip_events are SHARED;
  `regrasp` for robotiq re-runs the RQ seat window (`_rq_seatctr =
  _RQ_SEAT_HI`) — verify that seat also rewrites ORIENTATION for the gear
  (the tilt fix is the whole point; the franka path snaps identity quat).

## Suggested port sequence (gates between every step)

1. **Read first**: this doc; `docs/task1_jam_recovery.md`;
   `docs/REPLICATION_PLAYBOOK.md` (pod ops + robotiq lessons); memory files
   `task3-robotiq-port`, `task3-robotiq-force-fix`, `task3-robotiq-retrain`,
   `task1-recovery-arc`, `task1-grasp-realism`.
2. **Boot probe** (no training): `cfg.gripper = "robotiq_2f140"` on the gear
   env, 1 env, run the scripted setup only. Gate: asset parses (watch for
   the PARSE GHOST — a second half-parsed articulation from a bad reference;
   see task3 notes), the four-bar assembles in the correct branch, no
   explosion. `probe_grasp.py` works for any gripper — extend its finger
   readout to `finger_joint` angle for the 2F-140.
3. **Grasp calibration probe**: close on the hub; find the kiss/stall
   angles for Ø35.5 mm; verify pad-on-hub contact force and `pad_dz`-style
   geometry (grip point at gear origin +0.032). The TELEPORT CONTRACT
   applies: the 2F-140 is driven by finger_joint position targets — any
   teleport of a held object must respect the drive state or PhysX kicks
   (see env comment ~line 3178).
4. **Hand-off staging**: adapt the setup so the GEAR traverses/hands off at
   the same altitudes as with the panda hand (that is what `_tcp_dz` and
   `_rq_dest_dx` did for the bottle). Gate: scripted setup delivers the
   gear centered over the shaft at the entrance, grip holding, upright.
5. **Policy**: start from `task1_gear_sliprand.pt.it300` and fine-tune on
   the robotiq embodiment (obs include joint state — expect it NOT to
   transfer zero-shot; short runs, snapshot every 50-100 its, stop at good
   checkpoints). Gate: clean eval ≥95% strict seats on abs, 0 breaks,
   honest force profile (use `scripts/eval_gear_insert.py`).
6. **Jam recovery**: enable the slip inducer; verify the tilt mechanism on
   the 2F-140 grip (probe first — the four-bar pinch may hold orientation
   differently than the panda pads; if the slip does NOT tilt the gear,
   the failure mode may be different and the recovery routing needs
   re-diagnosis, not copy-paste). Then `eval_gear_jam.py --recovery ours`
   smoke → sweep row for the 2F-140.
7. **Render**: `render_gear_recovery.py` already takes the gripper from the
   env cfg; keep the PROXY gear pattern; re-frame the camera for the longer
   hand.

## Pod ops (the expensive lessons — do not relearn)

- ONE Isaac process at a time (`nvidia-smi --query-compute-apps`). Kit
  processes hang after their final print — `os._exit(0)` in new scripts
  (never `app.close()` alone), and kill leftovers BY EXPLICIT PID
  (`pkill -f` self-matches your own watcher command line).
- Long jobs: `nohup <script> & disown` + watch the LOG file (session
  restarts kill background tasks). Multi-command nohup strings get
  FLATTENED — write a script file and nohup that.
- numpy MUST stay 1.26.0; venv `/workspace/.venv`; env vars
  `HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99
  PYTHONPATH=/workspace/FORGE-plus_task3` before isaacsim imports.
- Eval protocol is the DETERMINISTIC mean. `--stochastic` is diagnostic
  only (sampling noise alone is 0/6 at this clearance). Never script
  manipulation in evals/renders (staging/setup positioning is fine and is
  HUD-labelled).
- Forge truncation is 1800 steps, not 600 — any shorter eval cap must
  force `env.reset()` on timeout (already in `eval_gear_jam.py`).
- Push with plain `git push origin task3` (token in remote); NEVER
  `git add -A` (untracked mp4s); stage explicit paths.

## Geometry crib sheet

| thing | value |
|---|---|
| gear (medium) | Ø41.9 teeth (z +5→+15 mm from origin), Ø35.5 hub (+15→+45), bore fit band r 5.15 mm (0.4 mm clearance), origin 5 mm below bottom face, mass 0.030 |
| shaft / seat | middle shaft at (0.449, 0.120), plate top z 0.405, shaft top 0.425, seated origin z 0.400, `insert_depth_tol` 0.006 |
| franka grip | pad centre at gear origin +0.032 (`mug_grip_z`), fingers 0.0178 each (35.7 mm pinch), ~74 N squeeze, TCP 0.103 |
| robotiq | TCP 0.214 along hand +z; kiss/squeeze angles need recalibration for the hub |
| forces (abs) | F_break 38±5, budget ~27.9 identity-only, clean Fins 13.8/15.9/17.6, recovery Fins 23.2/31.5 |
| jam detector | force branch ≥ max(6, 0.18·F_max) & <2 mm descent/40 steps; hover branch <1 N & above seat & not rising >10 mm |
