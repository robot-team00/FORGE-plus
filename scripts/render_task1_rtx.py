#!/usr/bin/env python3
"""render_task1_rtx.py
Isaac Lab RTX render of Task 1 peg insertion.
Based on render_pick_place.py proven patterns.

Run with:
    DISPLAY=:99 /workspace/.venv/bin/python scripts/render_task1_rtx.py

Output: docs/eval_episode_robot_rtx.mp4
"""
import os, sys, subprocess, time as _time, threading
from pathlib import Path
import numpy as np

# ── Persist / display env ────────────────────────────────────────────────────
_PC = "/workspace/persist"
os.environ.update({
    "HOME": _PC + "/ovhome",
    "DISPLAY": ":99",
    "MPLBACKEND": "Agg",
    "CUDA_CACHE_PATH": _PC + "/cuda_cache",
    "OMNI_USER_HOME": _PC + "/ovhome",
    "NUCLEUS_HOME": _PC + "/nucleus",
})
sys.path.insert(0, "/workspace/FORGE-plus_main")

# ── Isaac Sim boot (RTX-lite: reflections/AO off for speed) ─────────────────
_EXTRA = [
    "--/exts/isaacsim.core.throttling/enable_async=false",
    "--/rtx/raytracing/subsurface/enabled=false",
    "--/rtx/reflections/enabled=false",
    "--/rtx/translucency/enabled=false",
    "--/rtx/directLighting/sampledLighting/enabled=false",
    "--/rtx/indirectDiffuse/enabled=false",
    "--/rtx/ambientOcclusion/enabled=false",
    "--/rtx/raytracing/lightcache/spatialCache/enabled=false",
]

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 960, "height": 540,
                     "extra_args": _EXTRA})
print("booted", flush=True)

import torch, carb
import omni.usd
from pxr import Gf, UsdGeom, UsdShade, Sdf, UsdLux
import omni.replicator.core as rep
from forge_plus.isaac_insertion_env import FrankaInsertionEnv, InsertionEnvCfg
from forge_plus.skills.policy_network import ForceConditionedPolicy, PolicyConfig
from PIL import Image, ImageDraw
print("imports ok", flush=True)

# ── Env ──────────────────────────────────────────────────────────────────────
cfg = InsertionEnvCfg()
cfg.scene.num_envs = 1
cfg.decimation = 6  # 120Hz physics / 20Hz control = 6 physics steps per policy step
env = FrankaInsertionEnv(cfg)
print("env built", flush=True)

# ── Policy (Task 1 checkpoint) ────────────────────────────────────────────────
CKPT = "/workspace/FORGE-plus_main/checkpoints/task1_franka_panda.pt"
ckpt = torch.load(CKPT, map_location=env.device, weights_only=False)
pc = ckpt["policy_cfg"]
pcfg = pc if isinstance(pc, PolicyConfig) else PolicyConfig(**pc)
policy = ForceConditionedPolicy(pcfg).to(env.device)
policy.load_state_dict(ckpt["policy_state_dict"])
policy.eval()
print("policy loaded", flush=True)

out = env.reset()
obs = (out[0] if isinstance(out, tuple) else out)["policy"]
orig = env.scene.env_origins[0].cpu().numpy()

stage = omni.usd.get_context().get_stage()

# ── Force command (fixed budget for Task 1: 30N / 120N max = 0.25 norm) ─────
F_CMD_N = 30.0
F_MAX_NORM = 120.0
fcmd = torch.tensor([[F_CMD_N / F_MAX_NORM]], dtype=torch.float32, device=env.device)

# ── PBR material helper ──────────────────────────────────────────────────────
def _pbr(mat_path, rgb, rough=0.5, metal=0.0, opacity=1.0):
    mat = UsdShade.Material.Define(stage, mat_path)
    sh  = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor",  Sdf.ValueTypeNames.Color3f).Set(rgb)
    sh.CreateInput("roughness",     Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic",      Sdf.ValueTypeNames.Float).Set(metal)
    if opacity < 1.0:
        sh.CreateInput("opacity",   Sdf.ValueTypeNames.Float).Set(opacity)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat

def _bind(prim_path, mat):
    try:
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            UsdShade.MaterialBindingAPI(prim).Bind(mat)
            print("mat bound ->", prim_path, flush=True)
    except Exception as ex:
        print("bind skip %s: %s" % (prim_path, ex), flush=True)

# ── Lighting ─────────────────────────────────────────────────────────────────
dome = UsdLux.DomeLight.Define(stage, "/World/SkyDome")
dome.CreateIntensityAttr(800.0)
dome.CreateColorAttr((0.75, 0.82, 0.96))

sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(2200.0)
sun.CreateColorAttr((1.0, 0.95, 0.88))
sun.CreateAngleAttr(0.7)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 20.0, 0.0))

fill = UsdLux.SphereLight.Define(stage, "/World/Fill")
fill.CreateIntensityAttr(15000.0)
fill.CreateRadiusAttr(0.3)
fill.CreateColorAttr((0.9, 0.92, 1.0))
UsdGeom.Xformable(fill).AddTranslateOp().Set(
    Gf.Vec3d(float(orig[0]) - 0.6, float(orig[1]) - 1.0, float(orig[2]) + 1.5))
print("lights added", flush=True)

# ── Ground plane ──────────────────────────────────────────────────────────────
gp = UsdGeom.Mesh.Define(stage, "/World/Ground")
S = 6.0; gz = float(orig[2]) + 0.001
gp.CreatePointsAttr([(-S,-S,gz),(S,-S,gz),(S,S,gz),(-S,S,gz)])
gp.CreateFaceVertexCountsAttr([4])
gp.CreateFaceVertexIndicesAttr([0,1,2,3])
gp.CreateNormalsAttr([(0,0,1)]*4)
_bind("/World/Ground", _pbr("/World/Mats/Floor", (0.20,0.21,0.23), rough=0.6))

# ── PBR materials for scene objects ──────────────────────────────────────────
# Table: warm wood
_bind("/World/envs/env_0/Table",
      _pbr("/World/Mats/Table", (0.45, 0.28, 0.14), rough=0.5, metal=0.0))
# Socket: white ABS plastic
_bind("/World/envs/env_0/Socket",
      _pbr("/World/Mats/Socket", (0.88, 0.88, 0.92), rough=0.35, metal=0.0))
# Peg: vivid orange metallic
_bind("/World/envs/env_0/Peg",
      _pbr("/World/Mats/Peg", (0.95, 0.38, 0.08), rough=0.25, metal=0.3))
print("materials applied", flush=True)

# ── Camera: 3/4 view showing gripper + socket ─────────────────────────────────
cam = UsdGeom.Camera.Define(stage, "/World/EvalCam")
cam.CreateFocalLengthAttr(28.0)
eye = Gf.Vec3d(float(orig[0]) + 1.4, float(orig[1]) - 1.2, float(orig[2]) + 1.1)
tgt = Gf.Vec3d(float(orig[0]) + 0.3, float(orig[1]) + 0.0, float(orig[2]) + 0.50)
up  = Gf.Vec3d(0, 0, 1)
fwd = (tgt - eye).GetNormalized()
rgt = Gf.Cross(fwd, up).GetNormalized()
tup = Gf.Cross(rgt, fwd).GetNormalized()
M   = Gf.Matrix4d(rgt[0],rgt[1],rgt[2],0,
                  tup[0],tup[1],tup[2],0,
                  -fwd[0],-fwd[1],-fwd[2],0,
                  eye[0],eye[1],eye[2],1)
UsdGeom.Xformable(cam).AddTransformOp().Set(M)
print("camera defined", flush=True)

# ── Render product + annotator ────────────────────────────────────────────────
for _ in range(110): app.update()   # warmup 1: RTX pipeline init

rp  = rep.create.render_product("/World/EvalCam", (960, 540))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])

# Overscan patch — without this rgb.get_data() returns gray sky
try:
    from omni.replicator.core.scripts.utils import annotator_utils as _au
    _orig_fn = _au._resize_data_for_overscan
    def _safe(d, p):
        if not p or p.get("datawindow_overscan_z") is None: return d
        return _orig_fn(d, p)
    _au._resize_data_for_overscan = _safe
    print("overscan patch ok", flush=True)
except Exception as ex:
    print("overscan skip:", ex, flush=True)

for _ in range(110): app.update()   # warmup 2: annotator settle
print("render product ready", flush=True)

_sd = np.asarray(rgb.get_data())
print("sanity: shape=%s mean=%.1f" % (str(_sd.shape), float(np.mean(_sd))), flush=True)

# ── Output paths ──────────────────────────────────────────────────────────────
FRAMEDIR = "/workspace/frames_task1_rtx"
os.makedirs(FRAMEDIR, exist_ok=True)
for _f in Path(FRAMEDIR).glob("*.png"): _f.unlink()
OUTPUT = "/workspace/FORGE-plus_main/docs/eval_episode_robot_rtx.mp4"
Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

# ── Frame grabber (proven pattern: double app.update() before read) ───────────
def _grab():
    app.update()
    app.update()
    d = np.asarray(rgb.get_data())
    if d.ndim >= 3 and d.shape[0] > 1 and d.shape[1] > 1:
        return d
    _time.sleep(0.1)
    app.update()
    d = np.asarray(rgb.get_data())
    return d if (d.ndim >= 3 and d.shape[0] > 1) else None

# ── Re-reset right before rollout ─────────────────────────────────────────────
out = env.reset()
obs = (out[0] if isinstance(out, tuple) else out)["policy"]

# ── Episode rollout ───────────────────────────────────────────────────────────
N_MAX  = 300
PHASES = ["APPROACH", "DESCEND", "GRASP", "LIFT", "ALIGN", "INSERT", "DONE"]
W, H   = 960, 540
saved  = 0
t0     = _time.time()

for k in range(N_MAX):
    with torch.no_grad():
        act_m, _ = policy(obs, fcmd)
    act = act_m

    res  = env.step(act)
    obs  = res[0]["policy"]

    phase_idx = int(env._phase[0].item()) if hasattr(env, '_phase') else 0
    # Check truncation (env never terminates early; truncated = episode length exceeded)
    truncated = bool(res[3][0].item()) if len(res) > 3 else False

    data = _grab()
    if data is None:
        if k % 20 == 0: print("step %d EMPTY" % k, flush=True)
        continue

    img = Image.fromarray(data[:, :, :3]).convert("RGB")
    dr  = ImageDraw.Draw(img)
    ph_name = PHASES[min(phase_idx, len(PHASES)-1)]
    dr.text((20, 16), "FORGE+ Task 1 — Peg Insertion", fill=(255,255,255))
    dr.text((20, 38), "step %3d   phase [%d/7]: %s   F=%.0fN" % (k+1, phase_idx+1, ph_name, F_CMD_N),
            fill=(255,235,120))
    img.save(os.path.join(FRAMEDIR, "f_%04d.png" % saved))
    saved += 1

    if k % 20 == 0:
        print("step %d saved=%d phase=%d t=%.1fs" % (k, saved, phase_idx, _time.time()-t0),
              flush=True)

    if truncated:
        print("TRUNCATED at step %d" % k, flush=True)
        break

print("SAVED_FRAMES %d" % saved, flush=True)

# ── FFmpeg encode BEFORE app.close() ────────────────────────────────────────
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    ffmpeg_exe = "ffmpeg"

if saved >= 10:
    ret = subprocess.run(
        [ffmpeg_exe, "-y", "-framerate", "24",
         "-i", os.path.join(FRAMEDIR, "f_%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", OUTPUT],
        capture_output=True, text=True)
    if ret.returncode == 0:
        print("FFMPEG ok %dKB -> %s" % (os.path.getsize(OUTPUT)//1024, OUTPUT), flush=True)
    else:
        print("FFMPEG err:", ret.stderr[-400:], flush=True)
else:
    print("TOO FEW FRAMES (%d) — skipping encode" % saved, flush=True)

print("RENDER_ALL_DONE", flush=True)

# ── Watchdog: force-exit 30s after encode (RTX shutdown hangs) ───────────────
_wd = threading.Event()
def _watchdog():
    if not _wd.wait(30):
        print("WATCHDOGFORCE-exit", flush=True)
        os.kill(os.getpid(), 9)
threading.Thread(target=_watchdog, daemon=True).start()

env.close()
app.close()
_wd.set()
