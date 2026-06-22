#!/usr/bin/env python3
"""
render_task3.py - RTX render of task3 block-stacking eval episode.
Based on FORGE-plus_main/scripts/render_eval_video.py (proven working renderer).
Key fix vs task3 broken script: extra_args subsurface disable at boot.
Loads /tmp/forge_traj_task3.npz -> docs/eval_episode_task3.mp4
"""
import os, subprocess, time as _time, math
import numpy as np
from pathlib import Path
from PIL import Image

# -- Persistent shader/compute cache (MUST be set before importing isaacsim) --
_PCACHE = "/workspace/persist"
os.environ["HOME"]                        = _PCACHE + "/ovhome"
os.environ["CUDA_CACHE_PATH"]             = _PCACHE + "/shadercache/cuda"
os.environ["__GL_SHADER_DISK_CACHE"]      = "1"
os.environ["__GL_SHADER_DISK_CACHE_PATH"] = _PCACHE + "/shadercache/gl"
for _d in (os.environ["HOME"], os.environ["CUDA_CACHE_PATH"],
           os.environ["__GL_SHADER_DISK_CACHE_PATH"]):
    os.makedirs(_d, exist_ok=True)

# -- 1. Xvfb (display :99 is task3; :1 belongs to task1 -- do not touch) --
try:
    subprocess.Popen(["Xvfb", ":99", "-screen", "0", "1920x1080x24"])
    _time.sleep(2)
except Exception:
    pass
os.environ["DISPLAY"] = ":99"

# -- 2. Boot Isaac Sim with extra_args -- disables subsurface at boot (cause #5) --
from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": True, "width": 1920, "height": 1080,
                     "extra_args": ["--/rtx/raytracing/subsurface/enabled=false"]})
print("[VID] Isaac Sim booted.", flush=True)

import omni.usd                               # noqa: E402
from pxr import Gf, UsdGeom, UsdLux           # noqa: E402
import omni.replicator.core as rep            # noqa: E402

# -- 3. Paths --
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
FRANKA_USD = "/workspace/assets/franka/franka_visuals.usd"
TRAJ       = "/tmp/forge_traj_task3.npz"
FRAMEDIR   = "/workspace/frames_task3"
OUTPUT     = os.path.join(REPO_ROOT, "docs", "eval_episode_task3.mp4")

os.makedirs(FRAMEDIR, exist_ok=True)
for _f in Path(FRAMEDIR).glob("*.png"):
    _f.unlink()
Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

FRAMES = 200
FPS    = 24

# -- Load trajectory from eval_rollout_task3.py --
print(f"[VID] loading trajectory: {TRAJ}", flush=True)
traj       = np.load(TRAJ)
joints_arr = traj["joints"]    # (N_STEPS, 7)
ee_arr     = traj["ee"]        # (N_STEPS, 3)
N_STEPS    = len(joints_arr)
print(f"[VID] trajectory: {N_STEPS} steps", flush=True)

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

# -- 6. Scene: table + rack + block + Franka --
ox, oy = 0.0, 0.0

table = UsdGeom.Cube.Define(stage, "/World/Station_00/Table")
table.CreateSizeAttr(1.0)
UsdGeom.Xformable(table).AddScaleOp().Set(Gf.Vec3f(0.8, 0.8, 0.375))
UsdGeom.Xformable(table).AddTranslateOp().Set(Gf.Vec3d(ox + 0.2, oy, 0.1875))

rack = UsdGeom.Cube.Define(stage, "/World/Station_00/Rack")
rack.CreateSizeAttr(1.0)
UsdGeom.Xformable(rack).AddScaleOp().Set(Gf.Vec3f(0.14, 0.14, 0.665))
UsdGeom.Xformable(rack).AddTranslateOp().Set(Gf.Vec3d(ox + 0.48, oy, 0.3325))

block_prim = UsdGeom.Cube.Define(stage, "/World/Station_00/Block")
block_prim.CreateSizeAttr(0.05)
block_op   = UsdGeom.Xformable(block_prim).AddTranslateOp()
block_op.Set(Gf.Vec3d(0.35, 0.10, 0.42))

root = stage.DefinePrim("/World/Station_00/Robot", "Xform")
root.GetReferences().AddReference(FRANKA_USD)
UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(ox, oy, 0.40))
rob_rot_op = UsdGeom.Xformable(root).AddRotateXYZOp()

# -- Helpers --
def sample_traj(k):
    idx = int(k / max(FRAMES - 1, 1) * max(N_STEPS - 1, 1))
    idx = min(idx, N_STEPS - 1)
    return joints_arr[idx], ee_arr[idx]

def robot_xform_vec(joints):
    return Gf.Vec3f(
        float(math.degrees(joints[1])) * 0.4,
        float(math.degrees(joints[0])) * 0.5,
        float(math.degrees(joints[3])) * 0.2,
    )

def block_pos_at(k, ee):
    t = k / max(FRAMES - 1, 1)
    ex, ey, ez = float(ee[0]), float(ee[1]), float(ee[2])
    bx, by, bz = 0.35, 0.10, 0.42
    rx, ry, rz = 0.48, 0.00, 0.665
    if t < 0.35:
        return Gf.Vec3d(bx, by, bz)
    elif t < 0.60:
        a = (t - 0.35) / 0.25
        return Gf.Vec3d(bx + a*(ex-bx), by + a*(ey-by), bz + a*(ez-bz))
    elif t < 0.85:
        return Gf.Vec3d(ex, ey, ez)
    else:
        a = (t - 0.85) / 0.15
        return Gf.Vec3d(ex + a*(rx-ex), ey + a*(ry-ey), ez + a*(rz-ez))

# -- 7. Camera (exact matrix from render_eval_video.py main) --
_CAM_PATH = "/World/EvalCamera"
_cam_prim = UsdGeom.Camera.Define(stage, _CAM_PATH)
_cam_prim.CreateFocalLengthAttr(22.0)
_eye    = Gf.Vec3d(1.7, -1.9, 1.45)
_target = Gf.Vec3d(0.0,  0.0, 0.62)
_up     = Gf.Vec3d(0, 0, 1)
_fwd    = (_target - _eye).GetNormalized()
_rgt    = Gf.Cross(_fwd, _up).GetNormalized()
_tup    = Gf.Cross(_rgt, _fwd).GetNormalized()
_mat    = Gf.Matrix4d(
    _rgt[0],_rgt[1],_rgt[2],0,
    _tup[0],_tup[1],_tup[2],0,
    -_fwd[0],-_fwd[1],-_fwd[2],0,
    _eye[0],_eye[1],_eye[2],1
)
UsdGeom.Xformable(_cam_prim).AddTransformOp().Set(_mat)

# -- 8. Warmup --
print("[VID] eye (1.7,-1.9,1.45) USD camera", flush=True)
for _ in range(90):
    app.update()

# -- 9. Render product + annotator (exact copy from render_grid_video.py) --
rp  = rep.create.render_product(_CAM_PATH, (1920, 1080))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])

# Monkey-patch _resize_data_for_overscan (None-safe)
from omni.replicator.core.scripts.utils import annotator_utils as _ann_utils
_orig_resize = _ann_utils._resize_data_for_overscan
def _safe_resize(data, data_params):
    dz = (data_params or {}).get("datawindow_overscan_z")
    dx = (data_params or {}).get("datawindow_overscan_x")
    if dz is None or dx is None:
        return data
    return _orig_resize(data, data_params)
_ann_utils._resize_data_for_overscan = _safe_resize
print("[VID] overscan patch applied", flush=True)

# Re-warm after attaching
for _ in range(90):
    app.update()

# -- _grab(): data-first, step on retry (matches render_grid_video.py) --
def _grab():
    for _try in range(12):
        d = np.asarray(rgb.get_data())
        if d.ndim >= 3 and d.shape[0] > 1 and d.shape[1] > 1:
            return d
        rep.orchestrator.step(rt_subframes=2)
        for _ in range(8):
            app.update()

# -- 11. Render loop --
print(f"[VID] rendering {FRAMES} frames...", flush=True)
saved = 0
for k in range(FRAMES):
    joints, ee = sample_traj(k)
    block_op.Set(block_pos_at(k, ee))
    rob_rot_op.Set(robot_xform_vec(joints))
    rep.orchestrator.step(rt_subframes=4)
    for _ in range(10):
        app.update()
    _time.sleep(0.5)
    d = _grab()
    if d is None:
        print(f"[VID] frame {k} EMPTY buffer", flush=True)
        continue
    Image.fromarray(d[:, :, :3]).save(os.path.join(FRAMEDIR, f"f_{k:04d}.png"))
    saved += 1
    if k % 20 == 0:
        print(f"[VID] frame {k}/{FRAMES}  saved={saved}", flush=True)

app.close()

# -- 12. ffmpeg encode --
if saved >= 10:
    print(f"[VID] encoding {saved} frames -> {OUTPUT}", flush=True)
    ret = subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-pattern_type", "glob",
        "-i", os.path.join(FRAMEDIR, "f_*.png"),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "22",
        OUTPUT,
    ], capture_output=True, text=True)
    if ret.returncode == 0:
        sz = os.path.getsize(OUTPUT) // 1024
        print(f"[VID] MP4: {OUTPUT} ({sz} KB)", flush=True)
    else:
        print(f"[VID] ffmpeg err: {ret.stderr[-800:]}", flush=True)
else:
    print(f"[VID] too few frames ({saved}), skip encode", flush=True)
