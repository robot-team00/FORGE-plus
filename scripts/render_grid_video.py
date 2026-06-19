"""
render_grid_video.py — headless Isaac Sim animation of the FORGE-plus parallel training scene.

Spawns N_ROWS × N_COLS Franka Panda arms in a grid, each with a table, peg, and socket,
animates the peg descending into the socket (smoothstep insertion motion), and saves
each frame as a PNG, then stitches them into an mp4 with ffmpeg.

Requirements:
  - RunPod pod with NVIDIA_DRIVER_CAPABILITIES=all (set at pod level, not shell)
  - Isaac Sim 4.x/5.x installed in the project venv
  - Xvfb running on :1 (or DISPLAY already set)
  - assets/franka/franka_visuals.usd present (falls back to procedural arm if missing)
  - ffmpeg available in PATH

Usage:
  Xvfb :1 -screen 0 1920x1080x24 &
  DISPLAY=:1 timeout 1200 .venv/bin/python scripts/render_grid_video.py
  # VIDFRAMES env var controls frame count (default 72 = ~2.4s at 30fps)
  VIDFRAMES=120 DISPLAY=:1 .venv/bin/python scripts/render_grid_video.py

Output:
  docs/insertion_success.mp4 — 1280×720, 30fps, libx264
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
N_ROWS, N_COLS = 5, 5
SPACING = 1.4  # metres between stations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
FRANKA_USD = os.path.join(REPO_ROOT, "assets", "franka", "franka_visuals.usd")
FRAMEDIR = os.path.join(REPO_ROOT, "frames")  # temp frame dir, gitignored

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

# ── 7. Camera (initial position — will be auto-framed in step 10) ─────────────
camera = UsdGeom.Camera.Define(stage, "/World/Camera")
camera.CreateFocalLengthAttr(14.0)
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

print("Running pre-render frames...", flush=True)
for i in range(60):
    app.update()
    if i % 15 == 0:
        print(f"  frame {i}/60", flush=True)

# ── 10. Auto-frame camera over stations + capture animated frames ─────────────
import time as _time  # noqa: E402
import numpy as np  # noqa: E402
from pxr import Usd  # noqa: E402
from PIL import Image  # noqa: E402

FRAMES = int(os.environ.get("VIDFRAMES", "72"))
os.makedirs(FRAMEDIR, exist_ok=True)

# Auto-frame: compute bounding box over all /World/Station_XX prims.
# Note: BBoxCache works for plain USD references (AddReference). It does NOT
# work for Isaac Lab instanced prims — if you switch to Isaac Lab envs, use
# the hardcoded camera position from render_preview.py instead.
bbc = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
)
rng = Gf.Range3d()
for _idx in range(256):
    _sp = stage.GetPrimAtPath("/World/Station_%02d" % _idx)
    if not _sp.IsValid():
        continue
    _b = bbc.ComputeWorldBound(_sp).ComputeAlignedRange()
    if not _b.IsEmpty():
        rng.UnionWith(_b)

if rng.IsEmpty():
    cx, cy, cz, ext = 0.0, 0.0, 0.4, 6.0
else:
    cmin, cmax = rng.GetMin(), rng.GetMax()
    cx = (cmin[0] + cmax[0]) / 2.0
    cy = (cmin[1] + cmax[1]) / 2.0
    cz = (cmin[2] + cmax[2]) / 2.0
    ext = max(cmax[0] - cmin[0], cmax[1] - cmin[1], cmax[2] - cmin[2], 1.0)
    print(
        "[VID] bbox min %s max %s"
        % (
            tuple(round(v, 2) for v in cmin),
            tuple(round(v, 2) for v in cmax),
        ),
        flush=True,
    )

dist = ext * 1.25
eye = Gf.Vec3d(cx, cy - dist, cz + dist * 0.85)
target = Gf.Vec3d(cx, cy, cz)
up = Gf.Vec3d(0, 0, 1)
fwd = target - eye
fwd = fwd / fwd.GetLength()
right = Gf.Cross(fwd, up)
right = right / right.GetLength()
tup = Gf.Cross(right, fwd)
M = Gf.Matrix4d(
    right[0], right[1], right[2], 0.0,
    tup[0], tup[1], tup[2], 0.0,
    -fwd[0], -fwd[1], -fwd[2], 0.0,
    eye[0], eye[1], eye[2], 1.0,
)
_camx = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Camera"))
_camx.ClearXformOpOrder()
_camx.AddTransformOp().Set(M)
print(
    "[VID] eye %s target %s ext %.2f"
    % (tuple(round(v, 2) for v in eye), tuple(round(v, 2) for v in target), ext),
    flush=True,
)

# Collect peg translate ops for animation
peg_ops = []
for _idx in range(256):
    _p = stage.GetPrimAtPath("/World/Station_%02d/Peg" % _idx)
    if not _p.IsValid():
        continue
    for _op in UsdGeom.Xformable(_p).GetOrderedXformOps():
        if _op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            peg_ops.append((_op, Gf.Vec3d(_op.Get())))
            break
print("[VID] animating %d pegs, FRAMES=%d" % (len(peg_ops), FRAMES), flush=True)

# Re-warm the render product after moving the camera
for _ in range(90):
    app.update()


def _grab():
    """Grab an RGB frame, retrying if the buffer is empty."""
    for _try in range(12):
        d = np.asarray(rgb.get_data())
        if d.ndim >= 3 and d.shape[0] > 1 and d.shape[1] > 1:
            return d
        rep.orchestrator.step(rt_subframes=2)
        for _ in range(8):
            app.update()
        _time.sleep(0.4)
    return None


# Animate: peg descends from LIFT height to socket (smoothstep)
LIFT = 0.18
DESCEND_FRAC = 0.7

for k in range(FRAMES):
    phase = min(1.0, k / (FRAMES * DESCEND_FRAC)) if FRAMES > 1 else 1.0
    s = phase * phase * (3 - 2 * phase)  # smoothstep
    z_off = LIFT * (1.0 - s)
    for _op, _base in peg_ops:
        _op.Set(Gf.Vec3d(_base[0], _base[1], _base[2] + z_off))
    rep.orchestrator.step(rt_subframes=4)
    for _ in range(10):
        app.update()
    _time.sleep(0.5)
    d = _grab()
    if d is None:
        print("[VID] frame %d EMPTY buffer" % k, flush=True)
        continue
    img = Image.fromarray(d[:, :, :3])
    img.save(os.path.join(FRAMEDIR, "f_%04d.png" % k))
    if k % 12 == 0:
        print(
            "[VID] frame %d/%d z=%.3f mean=%.1f std=%.1f max=%d"
            % (k, FRAMES, z_off, d.mean(), d.std(), d.max()),
            flush=True,
        )

app.close()
print("[VID] done — stitching with ffmpeg", flush=True)

# ── 11. Stitch frames into mp4 ────────────────────────────────────────────────
out_mp4 = os.path.join(REPO_ROOT, "docs", "insertion_success.mp4")
os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
subprocess.run(
    [
        "ffmpeg", "-y", "-r", "30",
        "-i", os.path.join(FRAMEDIR, "f_%04d.png"),
        "-vf", "scale=1280:720",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_mp4,
    ],
    check=True,
)
sz = os.path.getsize(out_mp4)
print(f"Saved {sz:,} bytes → {out_mp4}", flush=True)
print("Done.", flush=True)
