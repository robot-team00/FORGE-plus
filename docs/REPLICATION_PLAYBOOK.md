# FORGE-plus Bottle Placement — Replication Playbook

> **Purpose:** everything a new session (or a new task) needs to replicate the bottle-placement
> work end-to-end: pod operations, environment architecture, policy training, headless
> evaluation, RTX rendering (including the pause-capture pipeline), gripper porting,
> and the debugging protocols that actually worked. Written 2026-07-07 after the
> Robotiq 2F-140 recovery-render campaign (issue #28) completed.
>
> Companion docs — read these too, they are not duplicated here:
> - `CLAUDE.md` — orientation + the hard-won rendering facts (numpy ABI, libGLU, NGX…)
> - `docs/RENDERING.md` — fresh-pod RTX setup, step by step
> - `docs/HANDOFF_robotiq_recovery_rerender.md` — full history of the Robotiq
>   campaign, including every dead end (the "five mechanisms", the disproven
>   wrist-gravity theory, the pause-capture root cause)

---

## 0. Non-negotiable rules (learned the hard way)

1. **Only learned-policy manipulation in renders and evals.** No scripted waypoints,
   no scripted retract, no scripted release during the demonstrated skill. Scripted
   *approach positioning* before the skill starts is allowed but must be labeled
   SCRIPTED on the HUD (render_recovery.py does this). Violating this invalidates
   the demo.
2. **Never `git add -A`.** It sweeps untracked render mp4s (including protected
   ones). Stage explicit paths only.
3. **ONE Isaac process at a time on the pod.** Two SimulationApps fight over the GPU
   and both misbehave. Before every launch:
   `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`
   and `kill -9` any leftovers. `app.close()` frequently HANGS holding ~4 GB GPU —
   zombies are the norm, not the exception. `pgrep` is not sufficient (it matches
   launcher shells and misses reparented children); trust nvidia-smi.
4. **HUD force gate:** demo videos must show peak contact force under F_break.
   `render_recovery.py` enforces this (a take that breaks the object is a FAIL and
   doesn't encode). Keep that gate.
5. **Robotiq gripper:** never command the drive joint past 0.785 rad, and never
   write gripper-DOF joint states directly (the TELEPORT CONTRACT, §6).
6. **Encode ffmpeg BEFORE `app.close()`** — SimulationApp.close() hard-exits the
   process (and often hangs anyway; see rule 3).
7. **Commit trailers** (required on every commit):
   `Co-Authored-By:` and `Claude-Session:` lines — see repo git log for the format.

---

## 1. Pod operations

Everything runs on a RunPod GPU pod (RTX-class). The repo alone is useless without it.

- Clones: `/workspace/FORGE-plus_main` (main), `_task1` (task1), `_task3` (task3).
- Shared venv: **`/workspace/.venv`** — always `/workspace/.venv/bin/python`.
- Shared assets: `/workspace/assets/` (franka USD, robotiq USD mirror).
- Env vars for EVERY Isaac process (train, eval, probe, render):

  ```bash
  export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 \
         PYTHONPATH=/workspace/FORGE-plus_task3
  cd /workspace/FORGE-plus_task3
  ```

- After every pod restart: rebuild the libGLU stub + `bash scripts/setup_runtime.sh`
  (see `docs/RENDERING.md` quick-start). Check the venv python3 symlink (CLAUDE.md
  fact 7).
- **Git push works from the pod**: personal credentials live in
  `/workspace/.jr_notes` (KEY=value lines: `GITHUB_TOKEN`, `WANDB_API_KEY`); the
  `origin` remote already embeds the token, so plain `git push origin task3` works.
  Never print or commit the token itself.
- W&B logging: trainers auto-read `WANDB_API_KEY` from `/workspace/.jr_notes`;
  project `forge-plus-task3`.

Long-running jobs: launch with `nohup ... > logfile 2>&1 &`, poll the logfile.
Renders take ~40 min/take; probes ~10 min; training minutes-to-hours depending on
num_envs × iterations.

---

## 2. Architecture map

```
forge_plus/
  isaac_pick_place_env.py   # THE env (3.4k lines): FrankaPickPlaceEnv + PickPlaceEnvCfg
                            # phase machine (grasp→lift→transport→descend→place/insert),
                            # FORGE mode (learned insertion), release mode, jam injection,
                            # robotiq port, gravity comp (_rq_ag), payload-comp instrumentation
  skills/policy_network.py  # ForceConditionedPolicy: FiLM-modulated MLP, obs 34, hidden 256
  skills/forge_skill.py     # skill wrapper used by the episode runner
  recovery/recovery_loop.py # force-signature jam detection + recovery attempt loop
  recovery/recovery_actions.py  # rotate_align, retract_and_reapproach, wiggle_search...
  llm/recovery_selector.py  # RecoverySelector: picks the recovery primitive from the
                            # force signature (HeuristicLLMClient = offline heuristic stand-in)
  llm/budget_setter.py      # per-object F_max budgets (glass 8.8 N ... metal 72 N)
scripts/
  train_pick_place.py       # PPO trainer (the one actually used for all task3 skills)
  train_skill.py            # generic/legacy trainer entry
  eval_forge.py             # headless deterministic eval (success/breakage %, no RTX)
  render_recovery.py        # THE demo renderer: recovery episode, HUD, pause-capture
  render_forge_min.py       # learned insertion(+release) render  [pre-pause-cap pipeline]
  render_task3.py           # replay renderer from recorded .npz states
  probe_*.py                # one-off diagnostic probes (keep! see §8)
  build_franka_robotiq_2f140.py  # text-USD recipe that builds the Franka+2F-140 asset
```

Key `PickPlaceEnvCfg` groups (all in `isaac_pick_place_env.py`, heavily commented):

- **Geometry:** `table_top_z, rack_x/y/z, cell_floor_z, shelf_top_z` …
- **Control:** `decimation=2`, OSC-style EE control with `lam` per-step motion caps,
  orientation stiffness schedule (`ori_k_carry/insert/descend/vertical/extrinsic`).
- **FORGE mode** (`forge_mode=True`): scripted intro drives the EE to an entrance
  pose (`forge_setup_steps` substeps), then the LEARNED policy takes over with small
  action caps (`forge_lam=0.004 m/step`), compliant stiffness (`forge_pos_k=120`),
  PBRS keypoint reward (`keypoint_k=60`) + force-overshoot penalty
  (`force_pen_beta=2.5`), compliant contact (`contact_stiffness=900, damping=60`).
- **Release mode** (`forge_release_mode=True`): adds the 8th action dim (finger
  open); success = upright (`release_upright_cos`) + settled (`release_vel_tol`).
- **Jam/recovery:** `jam_dx/dy` inject a base-aim error to force a rim wedge;
  detection thresholds `jam_force_n/frac`, `jam_progress_mm`, `jam_window`.
- **Robotiq:** `gripper="robotiq_2f140"` + `forge_setup_steps=4000`,
  `warmup_substeps=100`, `scene.replicate_physics=False` (required!), and the
  gravity-comp/force-shaping machinery (`_rq_ag`, max-ceiling, down-clamp).
- **Payload comp (instrumentation, ships OFF):** `rq_pc_grip=0.0, rq_pc_obj=0.0` —
  J^T feedforward from PhysX-measured masses. Measurement proved the un-modeled
  static wrench is only ~3 N / 0.7 Nm, so it is NOT the fix for anything; enable
  only for future heavy grippers.

The env **no-ops `render()`** for training speed. Render scripts must restore it:
`FrankaPickPlaceEnv.render = DirectRLEnv.render` BEFORE constructing the env.

---

## 3. Policy training (PPO)

Trainer: `scripts/train_pick_place.py`. ForceConditionedPolicy (FiLM MLP, obs 34,
act 7 — 8 with release), PPO+GAE, W&B logging.

```bash
export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99
/workspace/.venv/bin/python scripts/train_pick_place.py \
    --num_envs 512 --iterations 600 --gripper franka_panda \
    --forge --forge_obj 0 --forge_release \
    --ckpt checkpoints/my_new_skill.pt [--resume checkpoints/prev.pt]
```

- `--forge` = train the learned insertion (FORGE mode); `--forge_release` adds the
  learned release head; `--forge_obj 0` = glass bottle (fragile budget).
- 512 envs × 600 iterations was the standard recipe; `--resume` for curriculum
  stages (e.g. insertion first, then release fine-tune on top).
- Training defaults come from the cfg; renders/evals must use the SAME cfg flags
  the checkpoint was trained with, or the policy is out-of-distribution.

**Curriculum that worked** (replicate this shape for new skills):
1. Train the base skill in the easiest regime (e.g. place-only, no upright
   requirement → ~97% success) — `checkpoints/task3_place_solved_97succ.pt`.
2. Add the hard constraint as a stage-B fine-tune (`require_upright=True`,
   entrance hand-off, release head) — resume from stage A.
3. Gripper ports fine-tune from the working Franka checkpoint, never from scratch.

**Checkpoint inventory (the ones that matter):**

| checkpoint | what it is |
|---|---|
| `task3_forge_entrance.pt` | learned FORGE insertion, Franka panda, entrance hand-off |
| `task3_forge_release.pt` | insertion + learned release (8-dim action), Franka |
| `task3_forge_robotiq.pt` | Robotiq 2F-140 port of insertion+release (the shipped demo) |
| `task3_place_solved_97succ.pt` | stage-A gentle place (curriculum base) |
| others (`*_smoke*`, `*_dbg*`, `_bak_buggyenv/`) | scratch/debug — do not ship |

**Reward design lessons:** PBRS keypoint shaping must DOMINATE motion penalties
(`keypoint_k=60` vs small penalties) or the policy learns to freeze; the force
penalty (`force_pen_beta`) is what makes it gentle — verify gentleness in eval
(peak force vs budget), don't assume it.

---

## 4. Headless evaluation (ALWAYS before rendering)

The iron rule of this project: **smoke-test in headless physics first, render
second.** A render take costs ~40 min; a headless eval costs ~2 min. Every render
pathology we ever chased for hours was diagnosable (or absent!) in the smoke.

```bash
/workspace/.venv/bin/python scripts/eval_forge.py \
    --ckpt checkpoints/task3_forge_robotiq.pt --obj 0 --episodes 200
```

Reports success / breakage over parallel episodes with the DETERMINISTIC action
(mean). Success gates to demand before rendering: high insert/place rate, peak
force under the object budget, no timeout episodes.

For the recovery episode there is an equivalent smoke path inside
`render_recovery.py`'s loop (the `[rq-c]` telemetry lines print physics state
per control step) — a FAIL take is diagnosed from the log, not from frames.

---

## 5. RTX rendering — the pipeline that works

Fresh-pod setup and the non-obvious environment facts are in `docs/RENDERING.md`
and CLAUDE.md ("Hard-won facts" — numpy==1.26.0 ABI, libGLU stub, NGX/DLSS dead,
missing-shader errors are benign, no `rep.orchestrator.step()` in live loops).
This section documents the live-physics demo pipeline as of the final architecture.

### 5.1 The renderers

- `scripts/render_recovery.py` — the flagship: live env + learned policy +
  RecoveryLoop, HUD (state machine, LLM decision, force gauge with F_max/F_brk
  markers, LEARNED/SCRIPTED control tags), versioned takes. Env knobs:
  `GRIPPER`, `TAKE`, `CKPT`, `JAM`, `OBJ`, `K_MAX`, `PAUSE_CAP`, `OUT`.

  ```bash
  GRIPPER=robotiq_2f140 TAKE=103 CKPT=checkpoints/task3_forge_robotiq.pt \
  nohup /workspace/.venv/bin/python scripts/render_recovery.py \
      > /workspace/render_takes/take103.log 2>&1 &
  ```

  Frames land in `/workspace/frames_recovery/f_NNNN.png` (wiped at next launch —
  inspect mid-run!); SUCCESS takes encode to
  `/workspace/render_takes/forge_recovery_<gripper>_take_NNN.mp4`; FAIL takes
  don't encode. Approved take is trimmed and installed under
  `docs/videos/task3/`.
- `scripts/render_forge_min.py` (`RELEASE=1` for insert+release) and
  `scripts/render_task3.py` (replay from .npz). **WARNING: these still use the
  OLD physics-stepping capture** — before re-rendering anything with them, port
  the pause-capture fix (§5.2) or the physics on camera will not match the smoke.
- `scripts/render_pick_place.py` does NOT render on this pod (kitchen USD/PBR
  materials → 0×0 render product). Don't use it.

### 5.2 Pause-capture (MANDATORY for smoke-faithful demos)

**The single most important fact in this playbook.** The naive live-render loop
captures frames via `app.update()` between control steps — but `app.update()`
STEPS PHYSICS while the actuator torques are HELD. That creates a low-effective-
rate control regime that exists ONLY in renders: on the Robotiq campaign it caused
wrist "saturation", orientation-error ratchets, lean/grind, timeout hovers, and an
entire stack of compensating gain hacks — all of which vanished when the capture
stopped stepping physics. If your render behaves worse than your smoke, THIS is
the first suspect.

The fix (implemented in `render_recovery.py::_grab`, copy it verbatim):

1. Freeze simulation during capture with
   `carb.settings.get_settings().set_bool("/app/player/playSimulations", False)`
   … capture `app.update()`s … restore `True`.
   - NOT `timeline.pause()/play()` — play() only takes effect on a later update;
     the next `env.step()` livelocks at 109% CPU (take 98).
   - NOT `timeline.stop()` — resets sim state.
2. **usdrt Fabric flush of EVERY dynamic prim in shot, robot links included.**
   While paused NOTHING render-syncs (render transforms sync in the update-loop
   physics pass that pausing skips). Un-flushed prims render frozen at a stale
   pose: bottle-only flush shipped a video of a frozen-statue robot with a
   self-flying bottle (take 101). Per capture, write world poses via
   `usdrt.Rt.Xformable`:
   - rigid object: from `env._obj.data.root_pose_w`
   - robot links: paths from `env._robot.root_physx_view.link_paths[0]`,
     name-matched to `env._robot.data.body_names`, poses from
     `data.body_link_pos_w` / `body_link_quat_w` (w,x,y,z actor frame)
   - `SetWorldXformFromUsd()` on first use (inits scale from USD).
   - PhysX `update_transformations()` does NOT work — the renderer reads Fabric,
     not USD, not the fast cache.
3. Self-checks (keep them): print `pause-cap check: dq5=… FROZEN` (a DOF must not
   move across a capture) and `pause-cap flush set: bottle + N robot links
   (0 unmatched)` on first flush.

**Ghost trap:** with pause-capture on, any prim you forgot to flush renders at a
stale pose — which can look like "the object teleported/vanished". Before
declaring a visual broken, verify against PHYSICS coordinates in the log (the
`[rq-c] b=…` lines). A working take (100) was once killed on exactly that misread
— the "missing" bottle was legitimately occluded behind the rack.

### 5.3 Verify → trim → deliver → commit

1. Mid-run: pixel-diff early frames to confirm everything renders live
   (`f_0000` vs `f_0018`: tens of thousands of changed px = arm moving; ~0 = frozen).
2. On SUCCESS: check the log line `RESULT SUCCESS … peak=…N break=False …
   seat_valid=True placed=True`, then sample ~6 frames across the mp4 and LOOK at
   them (this step was skipped once; the user caught a frozen-arm video).
3. Trim to start with the object already held (scripted approach mostly cut):
   `ffmpeg -ss 2.125 -i take.mp4 -c:v libx264 -pix_fmt yuv420p -crf 20 out.mp4`,
   verify frame 0 of the trim.
4. Install to `docs/videos/task3/`, commit EXPLICIT paths, push.

---

## 6. Gripper porting (Robotiq 2F-140 lessons — generalize these)

Full history: `docs/HANDOFF_robotiq_recovery_rerender.md`. The transferable rules:

- **Build the asset as text-USD** (`scripts/build_franka_robotiq_2f140.py`) from
  NVIDIA's own pattern; assets mirror on S3. Verify with `probe_built_asset.py` /
  `probe_robotiq_asset.py` before any env work.
- **TELEPORT CONTRACT:** never write joint states for the gripper DOFs of a
  closed-loop (mimic) mechanism — PhysX desyncs the loop. Position the ARM by
  joint-state writes if needed; drive the GRIPPER only through its actuator.
- **PARSE GHOST:** a stale articulation parse can survive prim edits; rebuild the
  stage/env rather than fighting it.
- `scene.replicate_physics=False` is required for this asset; expect a long
  scripted setup (`forge_setup_steps=4000`, `warmup_substeps=100`).
- Closed-loop grippers under-report gravity to the arm: the env adds a learned-
  free gravity assist (`_rq_ag`) with a max-ceiling + down-clamp, and a
  12-substep finger handoff to avoid over-break forces at grasp. Reroll gates
  (`gauge reroll`) keep F_break draws sane per take.
- **Do not chase render-only pathology with training or gain changes.** The
  entire "retrain + five mechanisms" arc happened because the render pipeline
  itself was the bug (§5.2). Sequence for any new gripper:
  smoke-eval → if smoke is good but render is bad, fix the RENDER pipeline;
  only retrain if the SMOKE is bad.

---

## 7. Recovery loop (force-signature jam recovery)

- Jam detection is force+progress based, no vision: sustained contact force
  (`jam_force_n` / `jam_force_frac·F_max`) with net descent < `jam_progress_mm`
  over `jam_window` steps.
- `RecoverySelector` (offline heuristic in `HeuristicLLMClient`) maps the force
  signature (peak, net insert, lateral bias, rising flag) to a primitive:
  `rotate_align`, `retract_and_reapproach`, `wiggle_search`.
- Recovery primitives inject TARGETS into the same controller the learned policy
  drives (`_forge_targets`) — the policy stays in control; the recovery is a bias,
  not a script (rule 0.1).
- Demo recipe: inject a wedge with `JAM=0.08` (base-aim error), let attempt 1 jam,
  recovery fires, attempt 2 seats. `K_MAX` bounds attempts.

---

## 8. Debugging protocol (what actually worked)

1. **Probe, don't re-render.** A targeted `probe_*.py` (boot pipeline → isolate
   one mechanism → print numbers) costs ~10 min vs ~40 min/take. Examples worth
   copying: `probe_pausecap.py` (flush candidates vs unpaused pixel-centroid
   control, with a phys→pixel calibration phase), `probe_rq_isolate.py`,
   `probe_policy_obs.py`.
2. **Instrument the env, gated and cheap:** the `RQ_TRACE*` prints (every N
   substeps, env 0 only) — saturation flags, oerr, gravity-assist state, wrench
   fits. Leave them in the code behind their gates.
3. **Trust physics logs over eyes.** Frames lie (ghosts, occlusion); `[rq-c]`
   coordinates don't. Conversely a SUCCESS log does not prove the VIDEO is right —
   check both (frozen-arm take 101 passed every physics gate).
4. **Wrench-fit trick:** at quasi-static, contact-free equilibrium,
   `lstsq(J^T, applied_torque − gravity_torque)` recovers the un-modeled wrench.
   Contaminated by contact and null-space leakage — use only clean phases, and
   prefer saturation flags + mass prints for the headline conclusion.
5. **When two takes disagree, suspect the pipeline, not the policy.** Rerolls
   reshuffle randomness; only code changes move systematic behavior. With
   pause-capture the render became near-deterministic (seat gap 0.7058–0.7064
   across 4 takes) — large take-to-take variance is itself a bug signal.

---

## 9. Replication checklist for a NEW task/skill

1. Pod up → `setup_runtime.sh` + libGLU stub → verify `nvidia-smi` is clean.
2. Extend `PickPlaceEnvCfg`/env with the new phase or object; keep changes flag-
   gated so franka/base behavior is untouched (the franka env must stay intact).
3. Headless smoke of the scripted skeleton (`check_env.py` / a small probe).
4. Train stage-A (easy regime) with `train_pick_place.py`; verify in W&B +
   `eval_forge.py`-style deterministic eval; then stage-B fine-tune with
   `--resume`.
5. Only after the smoke gates pass: render with a `render_recovery.py`-derived
   script — copy the pause-capture `_grab` (§5.2) INCLUDING the all-links flush,
   the HUD control-tagging, the versioned-take output, and the F_brk fail gate.
6. Verify frames visually AND against the log; trim; install under
   `docs/videos/task3/`; commit explicit paths with the required trailers; push.
7. Update the handoff doc + this playbook if you learned a new hard-won fact.
