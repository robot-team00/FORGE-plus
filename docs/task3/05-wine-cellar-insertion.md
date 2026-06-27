# 05 — Wine-cellar peg-in-hole insertion

The current demonstrated task. A Franka Panda carries a realistic LIBERO **wine bottle** and
**inserts it into a cell of a wine-cellar rack** — a contact-rich **peg-in-hole** placement. The
bottle ends **standing perfectly vertical** (tilt ≈ 0°), seated on the cell floor, dead-centered
in the cell; the gripper then releases and the arm retracts, leaving the bottle in the rack.

> **Why this task.** The earlier fragile-place demo set the bottle on an open shelf; to make it
> *stand* the env had to right it about the shelf-contact pivot ("contact-then-verticalize",
> doc 02 §9), which looked like the robot *flicking* the bottle upright. A learned gentle
> "extrinsic-dexterity" roll-up was attempted and never converged (hard-exploration RL). The
> wine-cellar insertion is the clean alternative: the **cell geometry** keeps the bottle upright
> for free, so the task becomes an honest, gentle peg-in-hole — exactly the contact-rich
> manipulation FORGE is about, with no flick and no fragile RL.

File: `forge_plus/isaac_pick_place_env.py` (env, `place_strategy="insert"`),
`scripts/render_pick_place.py` (render). Rack asset:
`/workspace/assets/libero/wine_rack/wine_rack.usd` (procedural; `assets/` is gitignored).

## 1. The scene

- A **3×3 wood wine-rack** (egg-crate of cells) sits on the kitchen counter, spawned as a
  **kinematic** `RigidObject` at `(rack_x, rack_y, rack_z) = (0.45, 0.12, 0.38)`. The center cell
  top is at `shelf_top_z = 0.50`; the cell **floor** is at `cell_floor_z = 0.40`.
- The robot starts (in `place_only=True`) at `LIFT`, already holding the bottle by friction
  (warmup seat, doc 02 §2), and runs `LIFT → TRANSPORT → PLACE_DESCEND → RELEASE`.
- The target is the **center cell**. The bottle is the same LIBERO `wine_bottle_rigid.usd`
  (measured diam ≈ 0.043 m, length ≈ 0.150 m after the 0.5 scale).

## 2. The three things that make it seat upright (the whole trick)

A peg-in-hole with a tilted parallel-jaw grip is not obvious — the bottle is carried leaning
~12°, and a naive descent either misses the cell or jams against a wall and tips further. Three
decisions, all in `_apply_action` / the rack build, make it land vertical and centered:

### (a) Base-aim waypoint — aim the bottle BASE at the cell, not the gripper
The grip leans the bottle ~12°, so the **base** sits offset from the gripper TCP by a horizontal
vector `off_xy`. If we drove the *gripper* over the cell center, the base would land off-center
and clip a wall. So during the approach we **subtract the lean offset** from the xy waypoint:

```python
if self.cfg.place_strategy == "insert":
    off_xy   = self._obj.data.root_pose_w[:, :2] - ee_pos_w[:, :2]   # base→EE horizontal offset
    approach = (self._phase >= int(PickPlacePhase.TRANSPORT)).unsqueeze(-1)
    p_fixed  = torch.cat([p_fixed[:, :2] - approach.float() * off_xy, p_fixed[:, 2:3]], dim=-1)
```

Result: the **bottle base** tracks the cell center (`d_xy_cell` → ~0.006 m) regardless of the
lean. This was the fix for the carry stalling with the base off to the side.

### (b) Firm grip during insertion — let the cell do the verticalizing
We keep a **firm** OSC orientation stiffness while inserting (`ori_k_insert = 110`, vs
`ori_k_carry = 120` during carry — both firm). A firm grip holds the bottle steady on the way
down so it enters the cell cleanly; the **cell walls** then finish aligning it to vertical as it
seats. (Contrast the open-shelf place, which needed a *stiffness ramp* to right the bottle —
here the hole geometry does that work, so no ramp and no flick.) Stiffness is applied per step
via the OSC's `variable_kp` command:

```python
near  = (self._phase >= int(PickPlacePhase.PLACE_DESCEND)) & (base_z < cell_floor_z + 0.10)
ori_k = torch.where(near, full(ori_k_insert), full(ori_k_carry))
stiffness = torch.stack([k400, k400, k400, ori_k, ori_k, ori_k], dim=-1)   # (N,6) pos+ori
command   = torch.cat([tgt_pos_b, tgt_quat_b, stiffness], dim=-1)
```

### (c) Wide, shallow cell — no bind
A tight cell jammed the bottle (tilt 12°→32°, contact force ~9 N as it wedged). The rack is
built with generous clearance: `CLEAR = 0.06 m` over the bottle diameter → **cell = 0.103 m**,
wall height `0.08 m` (shallow — guides without trapping). With this the insertion is **bind-free**
(measured contact force ≈ 0 N all the way down). See doc 04 §7 for the rack build.

Near the cell we also **slow the motion** (`lam_eff = lam_place` instead of `lam`) so the
contact-rich seating stays gentle — FORGE's bounded, force-aware approach.

## 3. Success detection (geometric, in `_get_dones`)

Insertion is detected geometrically — the bottle base reaching the cell floor **inside** the
cell footprint:

```python
if c.place_strategy == "insert":
    base_xy  = self._obj.data.root_pose_w[:, :2] - self.scene.env_origins[:, :2]
    in_cell  = ((base_xy[:,0] - c.rack_x).abs() < 0.04) & ((base_xy[:,1] - c.rack_y).abs() < 0.04)
    on_shelf = ((cup_bot - c.cell_floor_z).abs() < c.insert_depth_tol) & in_cell   # tol 0.03
```

`PLACE_DESCEND → RELEASE` advances only when the base is **both** centered in the cell footprint
**and** down at the cell floor. (As with the open-shelf place, the contact sensor can't see the
gentle seat through the grip, so detection is geometric, not force-based — doc 01 §5.)

## 4. Measured result

From the render trace (zero-action OSC, no policy):

```
base=(0.458, 0.123, 0.400)   # x,y at cell center (0.45,0.12); z exactly at cell floor 0.40
tilt = 0.0°                  # perfectly vertical in the cell
cf   = 0 N                   # no jam / no binding force
```

…then the gripper force-opens (finger width → 0.08) and the arm retracts laterally; the bottle
stays locked at `(0.458, 0.123, 0.400)` while the hand moves away. The bottle is genuinely
released and stands in the rack.

## 5. Rendering the insertion

`scripts/render_pick_place.py` runs the insertion headless under RTX (same live-physics pipeline
as doc 03 — `app.update()` flushes PhysX→Fabric, no pose restore). Insert-specific points:

- **Zero-action OSC, no policy call.** The insertion mechanic is fully carried by the env's
  base-aim + firm-grip + cell geometry, so the render drives `act = zeros(7)` until the bottle is
  inserted, then a scripted lateral `RETRACT = [-1,-1,+1]` (doc 02 §8). The policy is **not**
  called — important because the obs is now 37-dim (an `obj_up` orientation triple was added,
  doc 01) and the old `task3_wine_bottle.pt` policy expects 34, so calling it would dim-mismatch.
- **Gradual release.** After insertion (`term_at`), finger width is lerped closed→open over
  `RAMP=28` steps for an impulse-free release, written directly to the finger joints (doc 02 §7).
- **Rack material.** The env's rack prim is a set of `UsdGeom.Mesh` boxes; the renderer binds a
  dark-wood PBR over it: `_bind("/World/envs/env_0/Rack", _pbr("/World/Mats/Rack",
  (0.30,0.17,0.09), rough=0.4, specular=0.4, clearcoat=0.3))`.
- **HUD title:** "FORGE+ Task 3 — Wine-Cellar Bottle Insertion".
- Output: `docs/videos/task3/pick_place_eval_001.mp4` (encode before `app.close()`).

## 6. Photorealism

The scene is dressed to look like a real kitchen / LIBERO scene (see also doc 03 §6, doc 04 §5):

- **Kitchen backdrop:** the LIBERO `kitchen_background` mesh is referenced under
  `/World/KitchenXform/Geo` (scale 0.01), giving cabinets, oven, and counters behind the robot.
- **Floor:** textured with `seamless_wood_planks_floor.png` (world-projected UVs).
- **Counter/table:** the env's primitive table cube is swapped for a textured box
  (`martin_novak_wood_table.png`) via `_replace_box` (the cube has no UVs, so we rebuild it as a
  UV'd Mesh).
- **Wine bottle:** keeps its **own baked LIBERO texture** (`label_wine.png`, `cork_texture.png`)
  — we deliberately do **not** bind a flat PBR over it.
- **Wine-rack:** dark-wood PBR with a slight clearcoat (above).

Net effect: a recognizable wine bottle going into a wood cellar rack on a wood counter in a
kitchen — not gray primitives.

## 7. Troubleshooting (each cost real time)

| Symptom | Cause | Fix |
|---|---|---|
| Carry stalls; base off to the side of the cell | Drove the **gripper** over the cell; the ~12° lean offsets the **base** | Base-aim waypoint — subtract `off_xy` during approach (§2a) |
| Bottle jams, tilt 12°→32°, contact ~9 N | Cell too tight / walls too tall | Widen `CLEAR` 0.03→0.06 (cell 0.103), shorten walls to 0.08 (§2c, doc 04 §7) |
| Success fires with the bottle still at the cell **top** | Used `shelf_top_z` (0.50) as the target | Use `cell_floor_z` (0.40) + `in_cell` footprint gate (§3) |
| y-reach limit (rack too far to the side) | Cell at `rack_y=0.30` out of comfortable reach | Move cell to `(0.45, 0.12)` (§1) |
| "could not find any bodies with contact reporter API" on the rack | Rack USD lacked a contact-report API | Apply `PhysxSchema.PhysxContactReportAPI` on the rack root in the build (doc 04 §7) |
| Render dim-mismatch (obs 34 vs 37) | Old policy expects 34-dim obs; env now emits 37 (`obj_up`) | Render with zero-action OSC, don't call the policy (§5) |
| Duplicate `rack_z` silently overrode the cell height | A legacy `rack_z=0.72` shadowed `0.38` | Removed the legacy field — there is one `rack_z` |

## 8. Optional next step — a learned FORGE insertion policy

The current demo is a **zero-action OSC** insertion (the env's geometry does the work, which is a
legitimate and robust result). To make it a *learned* contact-rich policy in the full FORGE
spirit: tighten the cell (so alignment is non-trivial), add the cell-relative pose + 3D contact
force to the observation, reward the keypoint distance to the seated pose with a force-overshoot
penalty (doc 01 §1), and train with PPO (`scripts/train_pick_place.py`, `obs_dim=37`). This is
deferred — the geometric insertion already satisfies the demo goal.
