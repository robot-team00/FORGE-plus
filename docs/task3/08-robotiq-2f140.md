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
- ⬜ Env port (`cfg.gripper == "robotiq_2f140"` branches), zero-shot policy check,
  render with distinct names.

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

## 4. Remaining plan (next session)

1. Env branches for `cfg.gripper == "robotiq_2f140"`: spawn the new USD + ghost;
   default-pose init (no arm overrides); `_reset_idx` skips all joint-state writes;
   grip close/seat/open via `finger_joint` targets (close ≈ 0.05, seat ≈ 0.10,
   open ≈ 0.5 — calibrate against the bottle); release = open target (no state write);
   contact sensor on `panda_hand|.*_inner_finger`; `_grasp_tcp_d` from a fingertip-prim
   measurement; `forge_approach_z`/retract heights shifted by the TCP delta.
2. Zero-shot `task3_forge_entrance.pt` (obs is arm+EE+force only — no finger joints);
   fine-tune only if the grasp-geometry shift breaks the descent.
3. Recovery runner + render with `--gripper robotiq_2f140`, videos to
   `forge_recovery_robotiq.mp4` (and the existing video renamed to
   `forge_recovery_franka.mp4`).
