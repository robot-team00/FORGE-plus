#!/usr/bin/env python3
"""eval_render_isaac.py — Phase 2: Load /tmp/forge_traj.npz, render with Isaac Sim.
Uses the proven render_grid_video.py approach:
  - Procedural geometry (no franka_visuals.usd reference)
  - rep.orchestrator.step(rt_subframes=4) to trigger RTX frames
  - Single close-up station animated from policy EE trajectory
"""
import os, subprocess, time
import numpy as np

TRAJ    = "/tmp/forge_traj.npz"
if not os.path.exists(TRAJ):
    TRAJ = "/workspace/FORGE-plus/tmp_forge_traj.npz"
OUT     = "/workspace/FORGE-plus/docs/eval_episode.mp4"
FRAMEDIR = "/workspace/frames_eval"
FRAMES  = 200
FPS     = 24

# Load trajectory saved by eval_rollout.py
print(f"[render] Loading trajectory from {TRAJ} ...", flush=True)
data       = np.load(TRAJ)
ee_arr     = data["ee"]           # (200, 3) end-effector / peg positions
reward_arr = data["reward"]       # rewards per step
done_at    = int(data["done_at"][0])
print(f"[render] ee={ee_arr.shape}  done_at={done_at}", flush=True)
print(f"[render] EE range z: min={ee_arr[:,2].min():.3f}  max={ee_arr[:,2].max():.3f}", flush=True)

# Start Xvfb (ignore if already running)
try:
    subprocess.Popen(["Xvfb", ":1", "-screen", "0", "1920x1080x24"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
except Exception:
    pass
os.environ["DISPLAY"] = ":1"

# Boot Isaac Sim
from isaacsim import SimulationApp                    # noqa: E402
app = SimulationApp({"headless": True, "width": 1920, "height": 1080})
print("Isaac Sim booted.", flush=True)

import omni.usd                                       # noqa: E402
from pxr import Gf, UsdGeom, UsdLux                  # noqa: E402
import omni.replicator.core as rep                    # noqa: E402

stage = omni.usd.get_context().get_stage()

# --- Lighting ---
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr(600)
key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
key.CreateIntensityAttr(2000)
key.CreateAngleAttr(0.53)
UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

# --- Ground plane ---
gnd = UsdGeom.Mesh.Define(stage, "/World/Ground")
gnd.CreatePointsAttr([(-20, -20, 0), (20, -20, 0), (20, 20, 0), (-20, 20, 0)])
gnd.CreateFaceVertexCountsAttr([4])
gnd.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
gnd.CreateNormalsAttr([(0, 0, 1)] * 4)

# --- Single station ---
OX, OY = 0.0, 0.0

table = UsdGeom.Cube.Define(stage, "/World/Table")
UsdGeom.Xformable(table).AddTranslateOp().Set(Gf.Vec3d(OX, OY, 0.375))
UsdGeom.Xformable(table).AddScaleOp().Set(Gf.Vec3d(0.45, 0.45, 0.025))

socket = UsdGeom.Cube.Define(stage, "/World/Socket")
UsdGeom.Xformable(socket).AddTranslateOp().Set(Gf.Vec3d(OX - 0.07, OY, 0.425))
UsdGeom.Xformable(socket).AddScaleOp().Set(Gf.Vec3d(0.03, 0.03, 0.025))

# Peg — animated to EE position each frame
peg = UsdGeom.Cylinder.Define(stage, "/World/Peg")
peg.CreateRadiusAttr(0.012)
peg.CreateHeightAttr(0.10)
peg_op = UsdGeom.Xformable(peg).AddTranslateOp()
peg_op.Set(Gf.Vec3d(OX + 0.07, OY, 0.455))

# Procedural robot arm (3-link boxes + gripper fingers, identical to render_grid_video.py)
for i, (dz, angle) in enumerate([(0.12, 0), (0.20, 20), (0.28, -15)]):
    link = UsdGeom.Cube.Define(stage, f"/World/Link{i}")
    UsdGeom.Xformable(link).AddTranslateOp().Set(Gf.Vec3d(OX, OY, 0.40 + dz))
    UsdGeom.Xformable(link).AddScaleOp().Set(Gf.Vec3d(0.018, 0.018, 0.055))
    UsdGeom.Xformable(link).AddRotateXYZOp().Set(Gf.Vec3f(0, angle, 0))
for dy in [-0.018, 0.018]:
    name = "L" if dy < 0 else "R"
    finger = UsdGeom.Cube.Define(stage, f"/World/Finger{name}")
    UsdGeom.Xformable(finger).AddTranslateOp().Set(Gf.Vec3d(OX + 0.05, OY + dy, 0.68))
    UsdGeom.Xformable(finger).AddScaleOp().Set(Gf.Vec3d(0.007, 0.007, 0.032))

print("Scene built: single station.", flush=True)

# --- Camera (close-up, front-right view) ---
camera = rep.create.camera(
    position=(0.5, -1.0, 0.9),
    rotation=(-20, 0, 12),
    focal_length=18.0
)

# --- Warmup (same as render_grid_video.py) ---
print("Warming up...", flush=True)
for i in range(50):
    app.update()
    if i % 10 == 0:
        print(f"  warmup {i}/50", flush=True)

# --- Render product + RGB annotator ---
rp  = rep.create.render_product(camera, (1920, 1080))
rgb = rep.annotators.get("rgb")
rgb.attach([rp])

# Re-warmup after attaching render product
for _ in range(90):
    app.update()

def _grab():
    """Try up to 12 times to get a real frame via orchestrator.step."""
    for _ in range(12):
        d = np.asarray(rgb.get_data())
        if d.ndim >= 3 and d.shape[0] > 1 and d.shape[1] > 1:
            return d
        rep.orchestrator.step(rt_subframes=2)
        for _ in range(8):
            app.update()
        time.sleep(0.4)
    return None

os.makedirs(FRAMEDIR, exist_ok=True)
for f in os.listdir(FRAMEDIR):
    os.remove(os.path.join(FRAMEDIR, f))

print(f"[VID] animating {FRAMES} frames with policy EE trajectory...", flush=True)
from PIL import Image                                  # noqa: E402

for k in range(FRAMES):
    # Move peg to policy end-effector position
    ee = ee_arr[k]
    peg_op.Set(Gf.Vec3d(float(ee[0]), float(ee[1]), max(float(ee[2]), 0.40)))

    # Trigger RTX render (the key call missing from render_eval_video.py)
    rep.orchestrator.step(rt_subframes=4)
    for _ in range(10):
        app.update()
    time.sleep(0.5)

    d = _grab()
    if d is None:
        print(f"[VID] frame {k} EMPTY buffer", flush=True)
        continue

    arr = d[:, :, :3]
    img = Image.fromarray(arr)
    img.save(os.path.join(FRAMEDIR, f"f_{k:04d}.png"))

    if k % 12 == 0:
        rew_str = f"  R={reward_arr[k]:.3f}" if k < len(reward_arr) else ""
        phase   = "policy rollout" if k < done_at else "holding final pose"
        print(f"[VID] frame {k}/{FRAMES}  ee_z={ee[2]:.3f}  mean={arr.mean():.1f}  max={arr.max()}  [{phase}]{rew_str}", flush=True)

app.close()
print("[VID] Isaac Sim closed.", flush=True)

# Count saved frames
saved = sorted(f for f in os.listdir(FRAMEDIR) if f.endswith(".png"))
print(f"[VID] {len(saved)} frames saved to {FRAMEDIR}", flush=True)

if len(saved) == 0:
    print("[VID] ERROR: no frames — cannot encode MP4", flush=True)
else:
    print(f"[VID] Encoding -> {OUT} ...", flush=True)
    r = subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(FPS),
         "-i", os.path.join(FRAMEDIR, "f_%04d.png"),
         "-codec:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p", OUT],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        print(f"[VID] MP4: {OUT} ({os.path.getsize(OUT):,} bytes)", flush=True)
    else:
        print("[VID] ffmpeg error:", r.stderr[-600:], flush=True)

print("[VID] done.", flush=True)
