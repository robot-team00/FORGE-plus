# 04 — LIBERO objects: import pipeline & selection

How we get realistic, recognizable objects (instead of gray primitives) into the env, and which
shapes to choose. The user requirement: **photorealistic, recognizable objects like LIBERO's** —
not gray cylinders.

## 1. What LIBERO is (reference)

[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) is a lifelong robot-learning
benchmark with a library of textured household object meshes (MIT-licensed). Asset groups in the
repo under `libero/libero/assets/`:

- `turbosquid_objects/` — e.g. mugs, **wine bottle** (what we use).
- `stable_scanned_objects/` — Google-Scanned-Object-style items (bowls, plate, …).
- `stable_hope_objects/` — HOPE grocery items (many **box/carton-shaped**: butter, milk,
  cream cheese, juice, soup, sauces).
- `articulated_objects/`, `scenes/`, `textures/`.

The **box/carton-shaped grocery items** (`stable_hope_objects`) are the most promising for *this*
gripper: flat faces (graspable) and short/wide (stand when placed). See §4.

## 2. Current object

`/workspace/assets/libero/wine_bottle/` (note: `assets/` is **gitignored** — outside the repo):

```
wine_bottle_red_main.obj    # mesh (from a turbosquid-style OBJ + MTL + textures)
wine_bottle_red_main.mtl
label_wine.png, cork_texture.png
wine_bottle.usd             # partial USD from the mesh converter (geometry + materials)
wine_bottle_rigid.usd       # ← the spawnable rigid object the env loads
```

The env spawns it as a `RigidObject` with `UsdFileCfg(usd_path=".../wine_bottle_rigid.usd",
scale=(0.5,0.5,0.5), mass=0.30, kinematic_enabled=False, ...)`.

## 3. Import pipeline: OBJ → USD → rigid wrap

The Isaac Lab `MeshConverter` expects a `/<name>/geometry` prim that the Isaac 5.1 asset
converter does **not** create (`MeshConverter: "Accessed invalid null prim"`). So we use a
two-step pipeline (helper scripts live in the session scratchpad; pattern reproduced here):

**Step 1 — OBJ → (partial) USD.** Use `omni.kit.asset_converter` (embed textures) or IsaacLab's
`scripts/tools/convert_mesh.py`. This yields a USD with the mesh + materials but **no** rigid-body
/ collision APIs.

**Step 2 — wrap into a spawnable rigid object.** Open the partial USD with `pxr` and apply:

```python
from pxr import Usd, UsdPhysics
stage = Usd.Stage.Open(SRC)
root  = stage.GetDefaultPrim()
UsdPhysics.RigidBodyAPI.Apply(root)
UsdPhysics.MassAPI.Apply(root).CreateMassAttr(0.30)
for mesh in [p for p in stage.Traverse() if p.GetTypeName() == "Mesh"]:
    UsdPhysics.CollisionAPI.Apply(mesh)
    UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr("convexDecomposition")
stage.Flatten().Export(OUT)   # -> *_rigid.usd
```

Pitfalls learned:

- Use **`convexDecomposition`**, not `convexHull`, for anything with a thin feature (e.g. a
  bottle neck) — a single hull seals the neck into a cone and it can't be gripped.
- Do **not** put `activate_contact_sensors` on the spawned object USD (it errors:
  "No rigid bodies under prim" due to sub-prim nesting). Contact sensors go on the robot.
- `UsdFileCfg` does **not** accept `physics_material` — set materials elsewhere.
- The wrapped USD needs `RigidBodyAPI` on the default prim or Isaac Lab errors "Failed to find a
  rigid body".
- `convexDecomposition` makes the collision **~3× heavier** than a primitive → use **256** envs
  for training, not 512.

## 4. Choosing an object (decision guide)

Combine the grasp rule (doc 02 §3) with stability:

| Want | Pick | Avoid |
|------|------|-------|
| Reliable parallel-jaw grip | **flat parallel faces**, width < ~8 cm | round mugs/cans (line contact → tilt/slip) |
| Stands after a (tilted) release | **short, wide base, low COM** | tall bottles (topple) |
| Realistic & recognizable | LIBERO textured mesh | gray primitives |

**Best fit for this gripper:** a **box/carton-shaped grocery item** from
`stable_hope_objects` (butter, cream cheese, milk/juice carton, pudding) — flat faces *and*
low/stable. A square tumbler also works but is less "realistic". The wine bottle is realistic and
grips fine, but topples on release (doc 02 §9).

## 5. Materials / textures in the render

The renderer keeps the object's **own baked texture** — do **not** bind a flat PBR material over
it (that hides the LIBERO texture). The bottle's `label_wine.png` / `cork_texture.png` then show.

## 6. Getting more LIBERO assets onto the pod

The pod has internet (`github.com`, `huggingface.co` reachable). To add a box object:

1. Fetch the mesh (OBJ/USD + textures) for a `stable_hope_objects` item from the LIBERO repo (or
   its asset release), into `/workspace/assets/libero/<name>/`.
2. Run the Step 1 + Step 2 pipeline above → `<name>_rigid.usd`.
3. Point the env's object `UsdFileCfg.usd_path` at it; set `scale`/`mass`; tune `mug_grip_z`
   (grip height) and the place geometry so the base reaches `shelf_top_z` (doc 02 §6).
4. Retrain (`scripts/train_pick_place.py`) — collision shape/mass changed.
