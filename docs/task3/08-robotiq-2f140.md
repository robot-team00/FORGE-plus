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
- ✅ **Four-bar ROOT CAUSE solved** (§10, supersedes §5/§7/§9 grasp workarounds): the loop
  joints were alive all along; the mechanism assembled in the WRONG four-bar branch
  because NVIDIA's free `*_outer_finger_joint` (0–180°) lets the fingers fold. Fix =
  clamp that joint to (0, 0.01°) — baked into the asset. Pads now stay PARALLEL and the
  gripper closes fully; perpendicular neck pinch verified held in the probe (v50).
- ⬜ Env smoke with the ported v50 recipe, zero-shot policy check, render
  `forge_recovery_robotiq.mp4` + rename the franka video.

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

## 5. Mimic-tree v2 attempt log (2026-07-03)

Authored from the measured table: both couplers verified tracking their fits in-sim
(left 0.87·θ+0.085, right 0.42·θ−0.03 with limit clamping — the PhysX mimic convention
`follower = −(gearing·ref + offset)` and degree-unit offsets are confirmed working).
The PADS defeated every scheme tried: the mimic on `left_inner_finger_joint` silently
never attaches (the joint dangles); drive-locks at the closed pose form a V that
ejects the bottle (watermelon-seed); inward spring-loading crosses the tips and blocks
closing; limit-locking (lower=upper) at the grip-proper angle still leaves a
depth-independent stall. Rubber-pad friction material (μ=2.0, bound to the pad
collision prims — machinery in `probe_rq_scene.py`) did not change the outcome.

**Key visual finding** (dual-camera snapshots): in EVERY composed configuration —
including the restored stock four-bar — only ONE finger forms a proper articulated
chain; the second never assembles into a pincer. All earlier "healthy" verdicts were
based on pad-body-origin separation sweeps, which the working knuckle mimic can
produce even with a collapsed distal chain. The composition (attachment recipe) is
now the prime suspect.

## 6. RECIPE v3 — the breakthrough (NVIDIA's own 2F-140 pattern)

The bucket also holds `ur10e/configuration/ur10e_Gripper_2F_140.usd` — NVIDIA's OWN
2F-140 attachment (dumped to `/workspace/logs/usd_dumps/ur_cfg140.usda`). It differs
from the 2F-85 recipe in three decisive ways, now replicated for the Franka in
`build_franka_robotiq_2f140.py` (v3):
1. the gripper payload composes ONTO the arm's own EE-link prim (here `panda_hand`),
   no separate holder Xform;
2. NO added fixed joint — the arm's `panda_hand_joint` is RETARGETED
   (`physics:body1 = panda_hand/robotiq_base_link`, anchors unchanged) so the gripper
   base becomes the arm's distal link in a clean tree;
3. BOTH articulation APIs are deleted there (the stray `PhysxArticulationAPI` left on
   the old holder was the parser poison), and `panda_hand` stops being a rigid body.

**Result: a fully healthy two-finger gripper, stock four-bar intact, NO parse ghost
needed.** And the session's scariest bug dissolved: the "one-finger collapse" seen in
every earlier snapshot was an OCCLUSION artifact — the side cameras looked straight
down the finger-spread axis; the head-on view shows two perfect fingers
(`/workspace/logs/rqs_open_b.png`).

**ENV IMPACT (not yet applied)**: with v3, `panda_hand` is no longer a BODY — the env's
robotiq branch must use `robotiq_base_link` as the EE body (same frame as the old
panda_hand via the retargeted joint) and the contact-sensor path becomes
`Robot/panda_hand/(left|right)_(inner|outer)_finger`. The ghost + holding-gains
workarounds can likely be dropped.

**Grasp calibration status** (`probe_rq_scene.py`, single-cycle hold tests with pinned
snapshots — note: snapshots advance physics, so in-teleport frames need the PIN
mechanism in `snap()`): the bottle USD origin sits near its NECK (extent −0.16/+0.06);
seating the origin at `base_link − 0.20` puts the neck between the pad faces and the
squeeze stalls at neck thickness (0.158) — but the bottle still cone-slides out of the
40 mm pads (they straddle the conical shoulder; squeezing while pinned instead builds
penetration and ejects violently — kiss-then-squeeze avoids that but the slide
remains). Open options: pad-tip pinch on the thin neck section, a robotiq-specific
bottle scale (~0.6 → fatter, longer cylindrical neck), or letting RL fine-tune grip
force as the franka pipeline does.

## 7. Grasp calibration SOLVED in the probe; env integration remains (2026-07-03)

**The grasp works** (`probe_rq_scene.py` LIPGRIP, `held=True`): the missing piece was a
**2.4 cm lateral offset** — the pad faces are centered at the pad-body midpoint xy, NOT
on the `robotiq_base_link` axis. Held recipe (scale-0.62 bottle — a ~24% larger,
closer-to-real instance whose fatter cylindrical neck suits the 40 mm pads):
- seat the bottle origin (its neck) at the PAD-FACE midpoint xy, 0.22 below
  `robotiq_base_link`;
- pads open wide first, then close to the KISS angle (0.19) onto the pinned bottle
  FROM OPEN (slight preload), then unpin and squeeze to 0.08 **on a static arm**;
- rubber (μ=2.0) bound to pads AND bottle; full-close ejects the cone
  (watermelon-seed), squeeze-while-pinned builds penetration and ejects violently.

**Env wiring landed** (seat branch, two-phase warm, static-arm window, gentle robotiq
traverse 0.015, warm-paused setup counter, scale-0.62 spawn, `_apply_rubber_pads`,
`_tcp_dz` 0.25) — but the env's substep cadence + live OSC still diverges from the
probe's step cadence: the phase-B close slams the pinned bottle (finger wrenched past
its limit) or the squeeze races a falling bottle. One fake SUCCESS observed (ejected
bottle landed in the cell — the loop's geometric `is_success` cannot tell luck from
skill; the render's held-seat validation does).

## 8. Remaining plan (next session)

1. Refactor the robotiq warmup as an explicit scripted seat state-machine at
   ENV-STEP granularity mirroring the probe exactly: [open-wide, park] →
   [kiss-close onto pinned bottle, rate-limited] → [unpin] → [squeeze, static arm]
   → [hand to the traverse]. Verify with the runner's a0 trace (bottle carried
   through the traverse), then zero-shot policy, then render
   `forge_recovery_robotiq.mp4` (+ rename the franka video).
## 9. (older notes)  Remaining plan (next session)

0. ~~Baseline~~ superseded by §6. **Next**: env v3 branch fixes (EE body, sensor path, drop ghost), then grasp options above, zero-shot, render.

0b. (old) **Baseline first**: mirror NVIDIA's UR10e tree from S3 and spawn their STOCK
   `UR10e_ROBOTIQ_GRIPPER_CFG` (their own combo, none of our composition) — snapshot
   it. Two clean fingers → our attachment recipe is at fault (diff against their
   variant layer); broken too → the sim stack can't do this gripper and the honest
   fallback is the Franka-hand demo only (report as a limitation on issue #28).
## 7. Grasp calibration SOLVED in the probe; env integration remains (2026-07-03)

**The grasp works** (`probe_rq_scene.py` LIPGRIP, `held=True`): the missing piece was a
**2.4 cm lateral offset** — the pad faces are centered at the pad-body midpoint xy, NOT
on the `robotiq_base_link` axis. Held recipe (scale-0.62 bottle — a ~24% larger,
closer-to-real instance whose fatter cylindrical neck suits the 40 mm pads):
- seat the bottle origin (its neck) at the PAD-FACE midpoint xy, 0.22 below
  `robotiq_base_link`;
- pads open wide first, then close to the KISS angle (0.19) onto the pinned bottle
  FROM OPEN (slight preload), then unpin and squeeze to 0.08 **on a static arm**;
- rubber (μ=2.0) bound to pads AND bottle; full-close ejects the cone
  (watermelon-seed), squeeze-while-pinned builds penetration and ejects violently.

**Env wiring landed** (seat branch, two-phase warm, static-arm window, gentle robotiq
traverse 0.015, warm-paused setup counter, scale-0.62 spawn, `_apply_rubber_pads`,
`_tcp_dz` 0.25) — but the env's substep cadence + live OSC still diverges from the
probe's step cadence: the phase-B close slams the pinned bottle (finger wrenched past
its limit) or the squeeze races a falling bottle. One fake SUCCESS observed (ejected
bottle landed in the cell — the loop's geometric `is_success` cannot tell luck from
skill; the render's held-seat validation does).

## 8. Remaining plan (next session)

1. Refactor the robotiq warmup as an explicit scripted seat state-machine at
   ENV-STEP granularity mirroring the probe exactly: [open-wide, park] →
   [kiss-close onto pinned bottle, rate-limited] → [unpin] → [squeeze, static arm]
   → [hand to the traverse]. Verify with the runner's a0 trace (bottle carried
   through the traverse), then zero-shot policy, then render
   `forge_recovery_robotiq.mp4` (+ rename the franka video).
## 9. (older notes)  Remaining plan (next session)

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

## 10. FOUR-BAR ROOT CAUSE + FIX (2026-07-03, probe v42–v50) — supersedes §5/§7/§9

**The loop joints were never dead.** `JOINTS:` dump of the merged articulation shows the
parser keeps `*_inner_finger_pad_joint` (inner_finger↔inner_knuckle) as TREE edges and
demotes the two `base→inner_knuckle` joints to maximal-coordinate loop joints — which
REMAIN ACTIVE and exact (Isaac Sim "Rig Closed-Loop Structures" behavior).

**The real defect: wrong four-bar assembly branch.** NVIDIA models `*_outer_finger_joint`
as a FREE 0–180° revolute (the physical gripper's adaptive-grasp spring pivot; the ROS
URDF has it FIXED). At spawn the fingers fold onto that parasitic DOF (v48 free sweep:
outer_finger ≈ 0.24+0.5θ, inner_finger stuck +0.72, pads scissor, fingertip gap floors
~30 mm). At high closure the RIGHT finger snapped into the TRUE branch spontaneously:
`outer_finger = 0`, `inner_finger = −θ`, `pad_joint = +θ` — exact to 4 decimals, pads
parallel, loop-enforced.

**Fix (baked into `build_franka_robotiq_2f140.py`, NOTE 3):** clamp both
`*_outer_finger_joint` limits to (0, 0.01°). v49 relations sweep: both sides track
`inner_finger = −θ` / `pad = +θ` through the whole stroke from spawn. Convention is now
STANDARD: finger_joint 0 = OPEN (59 mm tip gap) → 0.785 = full close (gap 0 at ~0.35).

**Dead ends (do not repeat):**
- Mimic joints on the inner fingers with follower=+θ (v42–45) "worked" only by
  REINFORCING the wrong branch (tip-scissor pinch, 10.5 mm floor — user rejected).
- Mimic with the mechanically correct −θ FIGHTS the live loop → instant NaN (v47).
- Mimics on the demoted `inner_knuckle` loop joints → instant NaN (v46).
- Follower drives with real stiffness fight the loop; keep `.*_inner_finger_joint`
  actuators at stiffness 0 / effort ~0.01 (PADK0). Axis token rotX vs rotZ is
  irrelevant for 1-DOF revolutes (official rigging docs).

**Held grasp recipe (probe v50, `GRASP=1 LOCKOF=1 PADK0=1`, scale-0.5 bottle):**
perpendicular side grip, neck at the pad plane; grip point = EE(base_link) + 0.214 m
along hand local +z, lateral offset ZERO (the old 2.4 cm pad-mid offset was a
wrong-branch artifact); KISS 0.218 (24 mm), SQUEEZE = full-close command 0.785 → drive
stalls on the 16 mm neck at ang ≈ 0.589 (stall +0.196, effort-limit-bounded); squeeze
happens PINNED (kiss leaves 8 mm clearance) on a static arm, then unpin → held and
stable 400 substeps (`rqs_mg359_after*.png`). Env port: `_rq_open/seat/close =
0.0/0.218/0.785`, `_grasp_tcp_d = 0.214`, `_tcp_dz = 0`, `_rq_pin` = warmup+60,
`_rq_static` = warmup+110, bottle back to scale 0.5 (franka parity).

## 11. Env port complete — the recovery video (2026-07-04)

`docs/videos/task3/forge_recovery_robotiq.mp4` (take 1, RESULT SUCCESS: jam ->
recovery -> learned re-insert seats on attempt 2, learned sampled release, peak
35.4 N < break, seat validated held at the 0.71 neck stall).

The episode pipeline (all robotiq-gated; franka untouched, re-verified SUCCESS):
base at (-0.15, 0.12); open-loop predrive on holding gains to the probe pose;
seat (pin/kiss/full-close) on stiff drives; OSC handoff (gains zeroed, nullspace
+ orientation anchored at the REACHED state, kpos 600 throughout); anchored
hover-lift (adaptive gravity scale — PhysX gravity comp is ~50 N short for this
loop-jointed articulation); high carry; arrival-latched contact-terminated
descent aimed at the seeded 8 cm wedge; policy engages 60 substeps after wedge
contact. Grip at 0.14 on the neck CYLINDER with the string-ring interlock (0.12
is on the taper — cams open under any axial load); gripper_drive effort 30.
Policy free-space flee (trained in-contact only — offline probe
scripts/probe_policy_obs.py) contained by a workspace fence (+-10 cm -> +-4 cm
chimney once centered, rim+1 cm insertion ceiling); post-recovery re-contact
bursts re-engage the policy from contact with measured push-bias counters; the
final centered descent runs 6 cm deeper so the funnel ride reaches the seat.
Smoke history: /workspace/logs/rq_env_smoke4..46.log (46 = the green run).
