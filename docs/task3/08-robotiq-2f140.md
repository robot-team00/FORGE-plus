# 08 — Robotiq 2F-140 gripper port (in progress)

> Goal (issue #28): reproduce the fragile recovery evaluation video
> ([doc 06 §7](06-recovery.md)) with the **Robotiq 2F-140** instead of the Franka panda
> hand. Distinct file names throughout so the two grippers' code paths and videos never
> mix: `forge_recovery_franka.*` vs `forge_recovery_robotiq.*` (renames land with the
> env port).

## Status

- ✅ **Asset built and mechanically verified**: a Franka Panda + Robotiq 2F-140 combined
  robot USD (NVIDIA ships that combo only for the 2F-**85**). Recipe + all findings in
  `scripts/build_franka_robotiq_2f140.py`.
- ✅ **Env port implemented** (`cfg.gripper == "robotiq_2f140"` branches in
  `isaac_pick_place_env.py`): combo-asset spawn + parse ghost, target-only gripper
  control, no-teleport resets, runtime arm-gain hand-off, pad contact sensor, TCP-shifted
  FORGE heights. Runner: `run_recovery_insertion.py --gripper robotiq_2f140`.
- ⚠️ **Blocked on grasp reliability** (see §5): the stock four-bar is too fragile for the
  warmup teleport-seat + contact cycles — pivoting to a mimic-tree v2 with gearings
  MEASURED from the healthy four-bar (`probe_rq_scene.py RELATIONS=1`).
- ⬜ Mimic-tree v2, seat calibration, zero-shot policy check, render with distinct names.

## 1. The asset

No Nucleus on the pod (`ISAAC_NUCLEUS_DIR = None`), so the 5.1 asset trees are mirrored
from NVIDIA's **public S3 bucket** into `/workspace/assets/isaac51/Robots/` (37 MB
FrankaPanda + 9 MB Robotiq/2F-140; `probe_robotiq_asset.py` shows the bucket listing
trick). `scripts/build_franka_robotiq_2f140.py` then authors two text-USD layers:

- `configuration/franka_Gripper_Robotiq_2F_140.usd` — replicates NVIDIA's 2F-85
  attachment recipe: payload the gripper under `/panda`, delete its
  `PhysicsArticulationRootAPI` (merging it into the panda articulation), pose it at the
  panda_hand flange (`(0.088, 0, 0.926)`, `Rz45·Rx180`), fixed-joint
  `panda_hand → robotiq_base_link` with identity local frames, deactivate the panda
  fingers + hand geometry. **`panda_hand` (the rigid body) remains** → the env's EE
  frame and OSC are unchanged across grippers.
- `franka_robotiq_2f140.usd` — standalone root referencing `franka.usd</panda>`
  (Gripper variant "None") + the config as a sublayer. The mirrored NVIDIA files are
  never modified.

Composition gotcha: a payload maps the target's defaultPrim ONTO the holder prim — the
gripper's children land directly under the holder, one level shallower than NVIDIA's
2F-85 file (their payload target has an extra nesting level). Overs at the wrong level
fail silently (`probe_built_asset.py` verifies the three load-bearing edits).

## 2. The two PhysX findings (each cost a day-equivalent of probing)

**Teleport fragility.** The 2F-140 closes each finger's four-bar with maximal-coordinate
loop joints (`*_inner_knuckle_joint`, `excludeFromArticulation=1`). On this PhysX build
those constraints do not survive `write_joint_state_to_sim` — ANY articulation teleport
(even arm-only with the gripper's shape unchanged) leaves the linkage flipped and the
pads collapsed (`probe_rq_isolate.py` tests A–D). A mimic-tree replacement (deactivate
loops, mimic the couplers) was prototyped and abandoned — the pad kinematics fight the
authored joint frames. **Contract**: spawn at the authored default pose, never write
joint states, drive the gripper only by `finger_joint` position targets, and let the
FORGE setup (OSC) drive the arm from the default pose to the hand-off. The recovery
demo needs no robot teleports at all under this contract.

**The parse ghost.** The merged gripper's excluded loop joints are only materialized by
the physics parser when a **standalone articulation instance of the same gripper USD
also exists in the scene**. Without it they silently never exist and the four-bar
collapses at spawn (5/5 solo-free runs broken vs 3/3 healthy with the ghost; the
`probe_ghost` run flipped only this variable). Workaround: spawn one gravity-free
2F-140 parked far outside the workspace — invisible, untouched, ~10 bodies of sim cost.

**Verified mechanism under the contract + ghost** (default pose, targets only):
`finger_joint` 0 → 0.7 rad sweeps the pad-body separation 0.040 → 0.127 m,
monotonic, stable — note the convention is **0 = closed, 0.785 = open** (inverted vs
the ROS URDF).

## 3. Actuators (from the isaaclab UR10e + 2F-140 template)

`gripper_drive` finger_joint (stiffness 11.25, damping 0.1, effort 10) ·
`gripper_finger` `.*_inner_finger_joint` (0.2 / 0.001 — the pad spring) ·
`gripper_passive` pads/outer/right-knuckle (0 / 0 — mimic- and loop-owned). The two
`inner_knuckle` loop joints are not articulation DOFs (runtime joint count: 7 arm + 8
gripper = 15; arm joints keep ids 0–6, `finger_joint` id 7).

## 4. Env-side findings (second debugging round)

- **Free-fall tears the loops too**: the env zeroes the arm actuator stiffness for the
  OSC; a limp arm sags at initialization and the fall tears the four-bar
  (`probe_rq_scene.py ZEROARM=1` reproduces it; healthy with holding gains). Fix in the
  env: the robotiq branch spawns with holding gains and hands the arm to the OSC at
  first reset via `write_joint_stiffness_to_sim` — a parameter write, no body motion.
- **The parse ghost works inside a full InteractiveScene** (mini-scene probe with cloner
  + table + bottle + rack + ACS: healthy) — ghost placement/registration/order,
  `replicate_physics`, sensors, and rack compliance were all exonerated one by one
  (`RQ_BISECT` switch in `_setup_scene`).
- **Grip-cycle wear (the open blocker)**: repeated seat-teleport + close cycles degrade
  the four-bar monotonically (pad-body separation 0.045 → 0.009 over four hold-test
  cycles) even in an otherwise healthy scene — and the bottle slips out (the pads catch
  the fat body, stall at ~35 mm, and squeeze it out). The stock loop-joint mechanism is
  not robust enough for the env's warmup seat + contact-rich episode on this PhysX
  build.
- **Contact-sensor regex**: prim-path expressions cannot span `/` — the robotiq sensor
  watches the four finger bodies (one path depth) instead of hand+fingers.

## 5. Remaining plan (next session)

1. **Mimic-tree v2**: re-author the surgery in `build_franka_robotiq_2f140.py` with
   the MEASURED joint relations below (`probe_rq_scene.py RELATIONS=1`, healthy
   four-bar; the first attempt guessed ±1 gearings and got it wrong — the linkage is
   asymmetric and mildly nonlinear in these joint frames). Linear mimics with
   gearing+offset fit the grip working range θ ∈ [0.05, 0.45]:

   | joint (vs `finger_joint` θ) | fit over working range |
   |---|---|
   | `right_outer_knuckle_joint` | ≈ +1.00·θ (stock mimic, keep) |
   | `left_outer_finger_joint` | ≈ 0.87·θ + 0.085 |
   | `right_outer_finger_joint` | ≈ 0.42·θ − 0.03 (nonlinear near closed) |
   | `left_inner_finger_joint` | ≈ 1.25·θ + 0.22 (saturates ~0.72 past θ≈0.45) |
   | `right_inner_finger_joint` | ≈ const −0.07 → stiff drive-lock at −0.07 |
   | `left_inner_finger_pad_joint` | ≈ −1.24·θ − 0.21 |
   | `right_inner_finger_pad_joint` | ≈ const +0.08 → stiff drive-lock at +0.08 |

   Raw sweep data: `/workspace/logs/probe_relations.log`. A pure mimic tree is
   teleport-, sag-, and contact-proof by construction — and removes the parse-ghost
   and holding-gains workarounds.
2. Seat calibration on the robust mechanism: hold-test grid (`probe_rq_scene.py`
   HOLD section — the bottle's neck sits around d ≈ 0.26–0.30 below the hand judging
   by the stall-angle trend), then set `_grasp_tcp_d` + `_rq_seat/_rq_close`.
3. Zero-shot `task3_forge_entrance.pt` (obs is arm+EE+force only — no finger joints);
   fine-tune only if the grasp-geometry shift breaks the descent.
4. Recovery runner + render with `--gripper robotiq_2f140`, videos to
   `forge_recovery_robotiq.mp4` (and the existing video renamed to
   `forge_recovery_franka.mp4`).
