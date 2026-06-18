"""
render_preview.py — headless Isaac Sim render of the FORGE-plus parallel training scene.

Spawns N_ROWS × N_COLS Franka Panda arms in a grid, each with a table, peg, and socket,
and captures a single 1920×1080 frame using omni.replicator.

Requirements:
  - RunPod pod with NVIDIA_DRIVER_CAPABILITIES=all
  - Isaac Sim 4.x/5.x installed in the project venv
  - Xvfb running on :1 (or DISPLAY already set)

Usage:
  Xvfb :1 -screen 0 1920x1080x24 &
  DISPLAY=:1 timeout 600 .venv/bin/python scripts/render_preview.py

Output:
  docs/render_preview.png  — 1920×1080 RGB PNG
"""

import os
import subprocess
import time

# ── 1. Display setup ────────────────────────────────────────────────────────
try:
    subprocess.Popen(["Xvfb", ":1", "-screen", "0", "1920x1080x24"])
    time.sleep(2)
except Exception:
    pass  # already running

os.environ["DISPLAY"] = ":1"

# ── 2. Boot Isaac Sim ────────────────────────────────────────────────────────
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 1920, "height": 1080})
print("Isaac Sim booted.", flush=True)

import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402
import omni.replicator.core as rep  # noqa: E402

# ── 3. Scene parameters ──────────────────────────────────────────────────────
# Training uses 1024 parallel envs; we visualise a representative 5×5 grid.
N_ROWS, N_COLS = 5, 5
SPACING = 1.4  # metres between stations

# Franka USD — local copy downloaded from NVIDIA Isaac Lab asset pack.
# Falls back to procedural proxy geometry if the file is missing.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
FRANKA_USD = os.path.join(REPO_ROOT, "assets", "franka", "panda_instanceable.usd")

stage = omni.usd.get_context().get_stage()

# ── 4. Lighting ──────────────────────────────────────────────────────────────
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr(600)

key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
key.CreateIntensityAttr(2000)
key.CreateAngleAttr(0.53)
UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

# ── 5. Ground plane ───────────────────────────────────────────────────────────
gnd = UsdGeom.Mesh.Define(stage, "/World/Ground")
gnd.CreatePointsAttr([(-20, -20, 0), (20, -20, 0), (20, 20, 0), (-20, 20, 0)])
gnd.CreateFaceVertexCountsAttr([4])
gnd.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
gnd.CreateNormalsAttr([(0, 0, 1)] * 4)

# ── 6. Per-station assembly scene ─────────────────────────────────────────────
def make_station(idx: int, row: int, col: int) -> None:
    ox = col * SPACING - (N_COLS - 1) * SPACING / 2
    oy = row * SPACING - (N_ROWS - 1) * SPACING / 2
    base = f"/World/Station_{idx:02d}"

    # Table
    table = UsdGeom.Cube.Define(stage, f"{base}/Table")
    UsdGeom.Xformable(table).AddTranslateOp().Set(Gf.Vec3d(ox, oy, 0.375))
    UsdGeom.Xformable(table).AddScaleOp().Set(Gf.Vec3d(0.45, 0.45, 0.025))

    # Target socket (white cube)
    socket = UsdGeom.Cube.Define(stage, f"{base}/Socket")
    UsdGeom.Xformable(socket).AddTranslateOp().Set(Gf.Vec3d(ox - 0.07, oy, 0.425))
    UsdGeom.Xformable(socket).AddScaleOp().Set(Gf.Vec3d(0.03, 0.03, 0.025))

    # Insertion peg (blue cylinder)
    peg = UsdGeom.Cylinder.Define(stage, f"{base}/Peg")
    peg.CreateRadiusAttr(0.012)
    peg.CreateHeightAttr(0.10)
    UsdGeom.Xformable(peg).AddTranslateOp().Set(Gf.Vec3d(ox + 0.07, oy, 0.455))

    # Robot arm — use real USD mesh if available, else procedural links
    if os.path.exists(FRANKA_USD):
        robot = stage.DefinePrim(f"{base}/Franka", "Xform")
        robot.GetReferences().AddReference(FRANKA_USD)
        UsdGeom.Xformable(robot).AddTranslateOp().Set(Gf.Vec3d(ox, oy, 0.40))
        UsdGeom.Xformable(robot).AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 90))
        UsdGeom.Xformable(robot).AddScaleOp().Set(Gf.Vec3d(0.85, 0.85, 0.85))
    else:
        # Procedural 3-link arm with gripper fingers
        for i, (dz, angle) in enumerate([(0.12, 0), (0.20, 20), (0.28, -15)]):
            link = UsdGeom.Cube.Define(stage, f"{base}/Link{i}")
            UsdGeom.Xformable(link).AddTranslateOp().Set(Gf.Vec3d(ox, oy, 0.40 + dz))
            UsdGeom.Xformable(link).AddScaleOp().Set(Gf.Vec3d(0.018, 0.018, 0.055))
            UsdGeom.Xformable(link).AddRotateXYZOp().Set(Gf.Vec3f(0, angle, 0))
        for dy in (-0.018, 0.018):
            finger = UsdGeom.Cube.Define(stage, f"{base}/Finger{'L' if dy < 0 else 'R'}")
            UsdGeom.Xformable(finger).AddTranslateOp().Set(
                Gf.Vec3d(ox + 0.05, oy + dy, 0.68)
            )
            UsdGeom.Xformable(finger).AddScaleOp().Set(Gf.Vec3d(0.007, 0.007, 0.032))


for idx, (r, c) in enumerate(
    [(r, c) for r in range(N_ROWS) for c in range(N_COLS)]
):
    make_station(idx, r, c)

print(f"Scene built: {N_ROWS}×{N_COLS} = {N_ROWS*N_COLS} stations.", flush=True)

# ── 7. Camera ────────────────────────────────────────────────────────────────
camera = UsdGeom.Camera.Define(stage, "/World/Camera")
camera.CreateFocalLengthAttr(14.0)  # wide angle to fit the full grid
xf = UsdGeom.Xformable(camera)
xf.AddTranslateOp().Set(Gf.Vec3d(0, -6.5, 4.5))
xf.AddRotateXYZOp().Set(Gf.Vec3f(-32, 0, 0))

# ── 8. Warm up (let renderer initialise) ─────────────────────────────────────
print("Warming up...", flush=True)
for i in range(50):
    app.update()
    if i % 10 == 0:
        print(f"  warmup {i}/50", flush=True)

# ── 9. Render product + annotator ────────────────────────────────────────────
rp = rep.create.render_product("/World/Camera", (1920, 1080))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])

print("Running render frames...", flush=True)
for i in range(60):
    app.update()
    if i % 15 == 0:
        print(f"  frame {i}/60", flush=True)

# ── 10. Capture ───────────────────────────────────────────────────────────────
print("Capturing...", flush=True)
rep.orchestrator.step(rt_subframes=4)
for _ in range(20):
    app.update()
time.sleep(2)

data = rgb.get_data()
out = os.path.join(REPO_ROOT, "docs", "render_preview.png")
os.makedirs(os.path.dirname(out), exist_ok=True)

from PIL import Image  # noqa: E402

img = Image.fromarray(data[:, :, :3])
img.save(out)
sz = os.path.getsize(out)
print(f"Saved {sz:,} bytes → {out}", flush=True)

app.close()
print("Done.", flush=True)
