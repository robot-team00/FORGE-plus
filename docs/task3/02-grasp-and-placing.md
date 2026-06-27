# 02 — Grasp & Placing physics

How the object is grasped, carried, placed, and released — with the physical reasons each design
choice was forced on us. File: `forge_plus/isaac_pick_place_env.py`.

## 1. Real physics, no cheating

The object is a **dynamic** `RigidObject` (`kinematic_enabled=False`, gravity on). It is **never**
teleported or kinematically attached during the carry. It is held by a genuine **friction grip**.
The only teleport is a brief **warmup seat** (below) that places the object into the closed
gripper at episode start; after warmup, friction alone holds it. The user requirement is strict:
**real physics, no kinematic holding.**

## 2. The friction seat (warmup) — this is NOT a pick

In `place_only=True` mode (what we run/train), the object is placed into the gripper at episode
start by a brief teleport during warmup, then held by **real friction**. This is **setup**, not a
learned/executed **pick**: the robot never reaches down to a surface and closes on the object. The
demonstrated task is the fragile **place** (`LIFT → TRANSPORT → PLACE_DESCEND → RELEASE`). The
real pick (`PRE_GRASP → DESCEND → GRASP → LIFT`, `place_only=False`) exists but is untrained.

In `_apply_action`, while `_warmup > 0` (set to `warmup_substeps=10` on reset):

```
R_ee   = rotation of the hand
grasp_c = hand_pos + R_ee @ [0, 0, grasp_tcp_d]   # grasp_tcp_d≈0.067 → the TCP between the pads
pose    = object pose; pose[:3] = grasp_c; pose.z -= mug_grip_z   # grip up the body from the base
pose.quat = upright;  write_root_pose_to_sim(pose); zero its velocity
```

- `grasp_tcp_d=0.067` puts the object at the **finger-pad midpoint** (not the finger-body
  midpoint, which sits ~5 cm above the pads).
- The object's origin is at its **base**, so we drop it by `mug_grip_z=0.12` to grip the upper
  body / neck.
- After warmup, the gripper is closed by **effort** (see §5) and friction holds it.
- `_reset_idx` **re-places the object at its spawn pose every reset** (`default_root_state +
  env_origins`). DirectRLEnv does *not* auto-reset rigid-object poses; without this the object
  stays wherever the last episode (or the render's warmup `app.update()`s) left it.

## 3. Object shape: why flat faces, and what fails

This is the central grasp lesson. A **parallel-jaw** gripper held at an angle:

| Object | Result | Why |
|--------|--------|-----|
| **Round mug / cylinder** | ❌ Fails | Contacts the flat pads along a **line** → ~zero pitch resistance → tilts 25–40° and levers/slips. Force spikes 50–70 N → "breaks" the fragile object. |
| **Square tumbler** (0.04×0.04×0.10) | ✅ Works | Flat parallel faces = real pad contact; held stably. Short + low COM → **stays standing** even with the tilted grasp. Trained to ~97%. |
| **Wine bottle** (tall) | ⚠️ Grips, topples | Thin (~1.6 cm) neck is the only graspable part; flat-enough for a stable grip, but tall → **topples on release** (see §7). |

**Rule for picking an object:** it needs (a) **flat / near-flat parallel faces** narrower than
the ~8 cm max gripper opening (so the jaws get real contact, not a line), and (b) to be **short
with a low center of mass and a wide base** if it must stand after a tilted release. A box-shaped
grocery item (carton/box) satisfies both; a tall bottle or a round mug does not.

## 4. Pendulum intuition (and its limit)

Gripping **above** the center of mass (e.g. a bottle by the neck) makes the hanging object a
pendulum that self-rights toward vertical *while hanging freely*. We relied on this for the
bottle. **Limit:** once the grip is *firm* it holds the object at the **gripper's orientation**,
not vertical — the firm grip overrides the pendulum. With the OSC holding a ~30–40° tilted
`ee_quat_des`, the bottle is carried tilted. The pendulum only helps a loosely-held or
free-swinging object.

## 5. The grip: bounded-force effort control

Gripper closing must be **gentle** (fragile object). The fingers are driven by an **effort** PD
computed in `_apply_action`:

```
target = 0.002 (closed, inside the neck half-width) on close phases, else 0.040 (open)
target = grasp_seat_w (0.006) during warmup
gforce = grip_pos_ks*(target - fpos) - grip_pos_kd*fvel        # ks=1500, kd=40
set_joint_effort_target([arm_effort, gforce])
```

- Position-style PD on **effort** gives a **bounded squeeze** (`k·overlap`, ~15 N) that rests the
  pads at the surface. A constant-effort grip had no position feedback and over-penetrated to
  ~5 mm finger width. The closed target (0.002) sits *inside* the object half-width so the pads
  press the faces gently.
- Closing phases (in `close_mask`): `GRASP, LIFT, TRANSPORT, PLACE_DESCEND`. `RELEASE` is **not**
  in the mask → target switches to 0.040 (open).

## 6. Geometric place detection

Because the hand/finger contact sensor **cannot** see the object-on-shelf contact through a
gentle grip (it nets ~0 N), the place is detected **geometrically**:

```
cup_bot   = object_root_z (origin is at the base)
on_shelf  = |cup_bot - shelf_top_z| < place_settle_tol   # 0.02
place_ok  = (phase==PLACE_DESCEND & on_shelf) | (phase!=PLACE_DESCEND)
```

`PLACE_DESCEND → RELEASE` advances only when the object's base actually reaches the shelf.

**Spurious-success trap (fixed):** with a loose `reach_tol` (0.08) the object starts ~7 cm above
the place target, so a grip-settle force blip once "completed" the place 7 cm up. Fix: a tight
`place_reach_tol=0.03` *and* the geometric `on_shelf` gate.

## 7. The gripper-release bug (it never actually let go)

**Symptom:** the bottle was only ever *lowered to touch* the shelf while still gripped — the
robot never released it. Finger width stayed at 0.015 (= neck width) through `RELEASE`.

**Root cause:** the env "opens" via **effort only** (`set_joint_effort_target`, target 0.040),
but the Franka `panda_hand` actuator has a **stiff position drive** (`stiffness=2e3`; only the
shoulder/forearm were zeroed for the OSC). The open effort can't move the fingers off the neck.

**Fix (in the renderer):** after the place, **force the fingers open by writing the joints
directly**:

```python
env._robot.write_joint_state_to_sim(
    torch.full((n, 2), 0.04, device=dev),   # both finger joints fully open
    torch.zeros((n, 2), device=dev),
    joint_ids=[7, 8])                         # finger joint indices
```

This sets finger width to 0.08; the bottle is genuinely released and stays put while the hand
moves away. (A cleaner long-term fix is to command the fingers with a **position** target at
RELEASE, or zero the `panda_hand` stiffness — but that interacts with the gentle-grip force, so
it needs care + possibly a retrain.)

## 8. The retract (clearing the object)

After releasing, retracting **straight up** re-captured the just-placed object (the open gripper
lifts back around the neck). Instead, retract **laterally back toward the base**:

```python
RETRACT[:3] = (-1, -1, +1)   # EE-pos delta added to the RELEASE waypoint (rack_x,rack_y,transport_z)
                              # → target moves back/left and up; the open gripper clears sideways
```

Sequence in the renderer: trained policy → place → hold ~10 steps (let the gripper open + object
settle) → `RETRACT` for the tail. The trained policy itself never learned to retract (episodes
ended at success during training), so the retract is a scripted nominal-OSC motion.

## 9. Standing placement: SOLVED via contact-then-verticalize

**Status: solved (2026-06-27).** The bottle now places **standing upright** (final tilt ~0.8°,
base exactly on the shelf), with real physics and no retrain.

**The diagnosis (measured, not guessed):** the bottle is carried at only ~12.6° (the firm neck
grip + the forward-pointing hand → a pendulum equilibrium, *not* 30–40° as it looked). But it
topples because it must be released **within a few degrees of vertical** — a 12° release leans the
COM over the narrow base edge and it falls. A gradual release didn't help (tilt 12°→32°→92° as the
grip opened). There is a **coupling tension**: holding the bottle vertical needs high OSC
orientation stiffness, but that same stiffness **overpowers the descent** (the arm climbs instead
of lowering) — so a single fixed stiffness can't both verticalize *and* place:

| OSC orientation stiffness | bottle tilt | can it descend to the shelf? |
|---|---|---|
| 40 (descent default) | ~12.6° → topples | ✅ yes |
| 200 (verticalizes) | ~3° → would stand | ❌ no — EE rises even at max down-action |

**The fix — decouple them in time (FORGE-style force-guided settle).** Descend with **low**
orientation stiffness (place works), then once the base is **on the shelf** ramp the stiffness
**up** and command a top-down hand orientation: the OSC then rights the bottle **about the
base-contact pivot** (it can't lift it — the base is planted) → it ends vertical → release →
stands. Uses the contact as the manipulation aid, exactly FORGE's point.

Implementation (`isaac_pick_place_env.py`):

- OSC switched to **`impedance_mode="variable_kp"`** so orientation stiffness is set **per step**
  via the command (`command = [pose(7), stiffness(6)]`; `motion_stiffness_limits_task=(5,600)`).
- `_apply_action`: when the base is on the shelf during `PLACE_DESCEND`, increment `_vert_ctr` and
  ramp `ori_k` from `ori_k_descend` (40) → `ori_k_vertical` (200) over `vert_ramp_steps` (18),
  setting `ee_quat_des` top-down `(w,x,y,z)=(0,1,0,0)` while ramping.
- `_get_dones`: gate `PLACE_DESCEND → RELEASE` on `on_shelf & (_vert_ctr >= vert_ramp_steps)` so
  it doesn't release until uprighted.
- Renderer then does the gradual finger-open release (§7) + lateral retract (§8).

**No retrain needed:** the descent is unchanged (low stiffness), so the existing
`task3_wine_bottle.pt` policy still places; the righting is a pure env-side force-guided maneuver.
Result: tilt 16° → 1.9° → 0.8°, bottle locked at base z = 0.500, hand retracts away.

Note: the headless eval **cannot** reproduce the place (it stalls at `TRANSPORT`) — only the RTX
render reaches `RELEASE`, because `_grab()`'s extra `app.update()`s add per-frame physics settling.
**Instrument the renderer itself**, not a headless proxy. (See doc 03.)
