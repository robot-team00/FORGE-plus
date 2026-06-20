#!/usr/bin/env python3
"""
render_eval_video.py - single-robot eval episode (close-up, arm animated).
Directly adapted from render_grid_video.py (the proven working renderer).
Changes: N_ROWS=N_COLS=1, close-up camera (0,-2,1.8)/-25deg,
         7-joint arm-body animation added alongside peg descent.
Output: docs/eval_episode.mp4
"""
import os, subprocess, time as _time, math
import numpy as np
from pathlib import Path
from PIL import Image

# -- 1. Xvfb --
try:
    subprocess.Popen(["Xvfb", ":1", "-screen", "0", "1920x1080x24"])
    _time.sleep(2)
except Exception:
    pass  # already running
os.environ["DISPLAY"] = ":1"

# -- 2. Boot Isaac Sim (EXACT same args as render_grid_video.py) --
from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": True, "width": 1920, "height": 1080})
print("Isaac Sim booted.", flush=True)

import omni.usd                           # noqa: E402
from pxr import Gf, UsdGeom, UsdLux      # noqa: E402
import omni.replicator.core as rep        # noqa: E402

# -- 3. Scene parameters --
N_ROWS, N_COLS = 1, 1
SPACING = 1.4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
FRANKA_USD = os.path.join(REPO_ROOT, "assets", "franka", "franka.usd")

FRAMEDIR = "/workspace/frames_eval"
OUTPUT   = "/workspace/FORGE-plus/docs/eval_episode.mp4"
os.makedirs(FRAMEDIR, exist_ok=True)
for _f in Path(FRAMEDIR).glob("*.png"):
    _f.unlink()
Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

FRAMES = 200   # 200 frames @ 24 fps = ~8.3 s clip
FPS    = 24

stage = omni.usd.get_context().get_stage()

# -- 4. Lighting (copied verbatim from render_grid_video.py) --
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr(600)
key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
key.CreateIntensityAttr(2000)
key.CreateAngleAttr(0.53)
UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

# -- 5. Ground plane (copied verbatim) --
gnd = UsdGeom.Mesh.Define(stage, "/World/Ground")
gnd.CreatePointsAttr([(-20,-20,0),(20,-20,0),(20,20,0),(-20,20,0)])
gnd.CreateFaceVertexCountsAttr([4])
gnd.CreateFaceVertexIndicesAttr([0,1,2,3])
gnd.CreateNormalsAttr([(0,0,1)]*4)

# -- 6. Per-station scene (N=1 this time) --
peg_ops  = []   # (translate_op, base_vec3d)
rob_rots = []   # rotate_op for each robot, for animation

def make_station(idx: int, row: int, col: int) -> None:
    base = f"/World/Station_{idx:02d}"
    ox = col * SPACING - (N_COLS - 1) * SPACING / 2
    oy = row * SPACING - (N_ROWS - 1) * SPACING / 2

    # Table
    table = UsdGeom.Cube.Define(stage, f"{base}/Table")
    UsdGeom.Xformable(table).AddTranslateOp().Set(Gf.Vec3d(ox, oy, 0.375))
    UsdGeom.Xformable(table).AddScaleOp().Set(Gf.Vec3d(0.45, 0.45, 0.025))

    # Target socket (white cube)
    socket = UsdGeom.Cube.Define(stage, f"{base}/Socket")
    UsdGeom.Xformable(socket).AddTranslateOp().Set(Gf.Vec3d(ox - 0.07, oy, 0.425))
    UsdGeom.Xformable(socket).AddScaleOp().Set(Gf.Vec3d(0.03, 0.03, 0.025))

    # Insertion peg (blue cylinder) -- animated
    peg = UsdGeom.Cylinder.Define(stage, f"{base}/Peg")
    peg.CreateRadiusAttr(0.012)
    peg.CreateHeightAttr(0.10)
    peg_op = UsdGeom.Xformable(peg).AddTranslateOp()
    peg_op.Set(Gf.Vec3d(ox + 0.07, oy, 0.455))
    peg_ops.append((peg_op, Gf.Vec3d(ox + 0.07, oy, 0.455)))

    # Robot arm -- franka_visuals.usd if available, else procedural
    if os.path.exists(FRANKA_USD):
        root = UsdGeom.Xform.Define(stage, f"{base}/FrankaRoot")
        UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(ox, oy, 0.40))
        rob_rot_op = UsdGeom.Xformable(root).AddRotateXYZOp()
        rob_rot_op.Set(Gf.Vec3f(0, 0, 90))
        UsdGeom.Xformable(root).AddScaleOp().Set(Gf.Vec3f(0.85, 0.85, 0.85))
        robot = stage.DefinePrim(f"{base}/FrankaRoot/Franka", "Xform")
        robot.GetReferences().AddReference(FRANKA_USD)
        robot.Load()
        from pxr import Usd as _Usd
        for _d in _Usd.PrimRange(robot):
            if _d.HasAuthoredPayloads():
                _d.Load()
            _img = UsdGeom.Imageable(_d)
            try:
                _img.MakeVisible()
            except Exception:
                pass
        rob_rots.append(rob_rot_op)
    else:
        # Procedural 3-link arm (same as render_grid_video.py fallback)
        for i, (dz, angle) in enumerate([(0.12, 0), (0.20, 20), (0.28, -15)]):
            link = UsdGeom.Cube.Define(stage, f"{base}/Link{i}")
            UsdGeom.Xformable(link).AddTranslateOp().Set(Gf.Vec3d(ox, oy, 0.40 + dz))
            UsdGeom.Xformable(link).AddScaleOp().Set(Gf.Vec3d(0.018, 0.018, 0.055))
            UsdGeom.Xformable(link).AddRotateXYZOp().Set(Gf.Vec3f(0, angle, 0))
            for dy in (-0.018, 0.018):
                finger = UsdGeom.Cube.Define(stage, f"{base}/Finger{'L' if dy < 0 else 'R'}")
                UsdGeom.Xformable(finger).AddTranslateOp().Set(
                    Gf.Vec3d(ox + 0.05, oy + dy, 0.68))
                UsdGeom.Xformable(finger).AddScaleOp().Set(Gf.Vec3d(0.007, 0.007, 0.032))

for idx, (r, c) in enumerate(
    [(r, c) for r in range(N_ROWS) for c in range(N_COLS)]
):
    make_station(idx, r, c)

print(f"Scene built: {N_ROWS}x{N_COLS} = {N_ROWS*N_COLS} stations.", flush=True)

# -- 7. Camera: close-up single robot (rep.create.camera fixes overscan NoneType) --
camera = rep.create.camera(
    position=(1.7, -1.9, 1.45),
    look_at=(0.0, 0.0, 0.62),
    focal_length=22.0
)

# -- 8. Warmup (same as render_grid_video.py) --
print("[VID] eye (0,-2,1.8) rot (-25,0,0)", flush=True)
for _ in range(90):
    app.update()

# -- 9. Render product + annotator (exact copy) --
rp  = rep.create.render_product(camera, (1920, 1080))
rgb = rep.annotators.get("rgb")
rgb.attach([rp])
# -- Monkey-patch _resize_data_for_overscan (None-safe: no rep.orchestrator.step) --
from omni.replicator.core.scripts.utils import annotator_utils as _ann_utils
_orig_resize = _ann_utils._resize_data_for_overscan
def _safe_resize(data, data_params):
    dz = (data_params or {}).get("datawindow_overscan_z")
    dx = (data_params or {}).get("datawindow_overscan_x")
    if dz is None or dx is None:
        return data  # no overscan configured; return raw data
    return _orig_resize(data, data_params)
_ann_utils._resize_data_for_overscan = _safe_resize
print("[VID] overscan patch applied", flush=True)
print(f"[VID] animating {len(peg_ops)} pegs, FRAMES={FRAMES}", flush=True)

# Re-warm after attaching (same as render_grid_video.py)
for _ in range(90):
    app.update()

# -- _grab(): EXACT copy from render_grid_video.py --
def _grab():
    for _try in range(12):
        try:
            rep.orchestrator.step(rt_subframes=12)
            raw = rgb.get_data()
            if raw is not None and len(raw) > 0:
                d = np.frombuffer(raw, dtype=np.uint8).reshape(1080, 1920, 4)
                return d
        except Exception as e:
            print(f"[VID] grab err {_try}: {e}", flush=True)
        for _ in range(8): app.update()
        for _ in range(8):
            app.update()
        _time.sleep(0.4)
    return None

# -- 10. 7-joint insertion trajectory (degrees) --
# Waypoints: (normalized_time, [j1..j7])
WAYPOINTS = [
    (0.00, [  0, -45,  0, -135,  0,  90, 45]),  # home
    (0.12, [  8, -30,  2, -115,  0, 100, 45]),  # pre-reach
    (0.22, [ 16, -10,  3, -100,  0, 105, 45]),  # at peg / grasp
    (0.35, [ 16, -10,  3, -100,  0, 105, 45]),  # hold (gripper closed)
    (0.45, [ 12, -38,  4, -118,  0,  94, 45]),  # lift peg
    (0.60, [  3, -26,  1, -110,  0,  93, 50]),  # sweep toward socket
    (0.78, [  0, -12,  0,  -98,  0, 100, 50]),  # position above socket
    (1.00, [  0,  -5,  0,  -95,  0, 100, 50]),  # insert into socket
]

def arm_angles(t):
    """Smooth interpolation between waypoints."""
    times  = [w[0] for w in WAYPOINTS]
    angles = [w[1] for w in WAYPOINTS]
    for i in range(len(times)-1):
        if t <= times[i+1]:
            s = (t - times[i]) / max(times[i+1] - times[i], 1e-9)
            s = s*s*(3 - 2*s)   # smooth-step
            return [angles[i][j] + (angles[i+1][j] - angles[i][j]) * s
                    for j in range(7)]
    return list(angles[-1])

def robot_xform_vec(angs):
    """
    Encode 7 joint angles as whole-body rotation of franka_visuals.usd root.
    j1 (angs[0]) drives yaw  (base swivel around Z)
    j2 (angs[1]) drives pitch (shoulder => forward lean of whole body)
    Robot starts facing +X at yaw=90 in USD coords.
    """
    yaw   = 90.0 + angs[0]                    # base rotation
    pitch = (angs[1] + 45.0) * 0.30           # 0 at home, ~12 at insert
    return Gf.Vec3f(pitch, 0.0, yaw)

def peg_z(t, base_z=0.455):
    """Peg rides with arm during reach, then descends into socket."""
    if t < 0.22:   return base_z
    elif t < 0.45:
        s = (t - 0.22) / 0.23
        return base_z + s * 0.025              # lift with arm
    elif t < 0.60: return base_z + 0.025       # hold during sweep
    elif t < 0.78:
        s = (t - 0.60) / 0.18
        return base_z + 0.025 - s * 0.020     # descend toward socket
    else:
        s = (t - 0.78) / 0.22; s = s*s*(3-2*s)
        return base_z + 0.005 - s * 0.030     # insert

# -- 11. Render loop: EXACT step/update/sleep/grab pattern from render_grid_video.py --
print(f"[VID] rendering {FRAMES} frames...", flush=True)
for k in range(FRAMES):
    t = k / max(FRAMES - 1, 1)
    angs = arm_angles(t)

    # Animate peg
    for _op, _base in peg_ops:
        _op.Set(Gf.Vec3d(_base[0], _base[1], peg_z(t, _base[2])))

    # Animate robot body (encodes joint motion as whole-body rotation)
    for rob_rot_op in rob_rots:
        rob_rot_op.Set(robot_xform_vec(angs))

    # Exact render_grid_video.py pattern:
    for _ in range(8): app.update()
    for _ in range(10):
        app.update()
    _time.sleep(0.03)

    d = _grab()
    if d is None:
        print(f"[VID] frame {k} EMPTY buffer", flush=True)
        continue

    Image.fromarray(d[:, :, :3]).save(os.path.join(FRAMEDIR, "f_%04d.png" % k))

    if k % 12 == 0:
        j_str = " ".join(f"{a:.0f}" for a in angs[:4])
        print(f"[VID] frame {k}/{FRAMES} peg_z={peg_z(t):.3f} mean={d.mean():.1f} max={d.max()} j=[{j_str}]", flush=True)

app.close()
print("[VID] done", flush=True)

# -- 12. Encode --
saved = len(list(Path(FRAMEDIR).glob("f_*.png")))
print(f"[VID] {saved}/{FRAMES} frames saved", flush=True)
if saved >= 10:
    ret = subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", os.path.join(FRAMEDIR, "f_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "22",
        OUTPUT
    ], capture_output=True, text=True)
    if ret.returncode == 0:
        sz = os.path.getsize(OUTPUT) // 1024
        print(f"[VID] MP4: {OUTPUT} ({sz} KB)", flush=True)
    else:
        print(f"[VID] ffmpeg err: {ret.stderr[-800:]}", flush=True)
else:
    print(f"[VID] too few frames ({saved}), skip encode", flush=True)
