#!/usr/bin/env python3
# render_fragile_task3.py v1
# NEW script -- render_task3.py is NOT modified.
# Renders Franka + glass_bowl fragile pick-and-place -> eval_run_003.mp4
# Based on render_task3.py v5 proven patterns.
import os, sys, subprocess, time as _time, threading
from pathlib import Path
import numpy as np

_PC = "/workspace/persist"
os.environ.update({
    "HOME": _PC + "/ovhome",
    "DISPLAY": ":99",
    "MPLBACKEND": "Agg",
    "CUDA_CACHE_PATH": _PC + "/cuda_cache",
    "OMNI_USER_HOME": _PC + "/ovhome",
    "NUCLEUS_HOME": _PC + "/nucleus",
})
sys.path.insert(0, "/workspace/FORGE-plus_task3")  # forge_plus package

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
from pxr import Gf, UsdGeom, UsdShade, Sdf
import omni.replicator.core as rep
from forge_plus.isaac_place_env import FrankaPlaceEnv, PlaceEnvCfg
from forge_plus.skills.policy_network import ForceConditionedPolicy
from PIL import Image, ImageDraw
print("imports ok", flush=True)

S = carb.settings.get_settings()
for _k in ["/rtx/reflections/enabled", "/rtx/translucency/enabled",
            "/rtx/indirectDiffuse/enabled", "/rtx/ambientOcclusion/enabled",
            "/rtx/directLighting/sampledLighting/enabled"]:
    S.set(_k, False)

cfg = PlaceEnvCfg()
cfg.scene.num_envs = 1
cfg.gripper = "franka_panda"
env = FrankaPlaceEnv(cfg)
print("env built", flush=True)

ckpt = torch.load("/workspace/FORGE-plus_task3/checkpoints/task3_franka_panda.pt",
                  map_location=env.device, weights_only=False)
policy = ForceConditionedPolicy(ckpt["policy_cfg"]).to(env.device)
policy.load_state_dict(ckpt["policy_state_dict"])
policy.eval()
print("policy loaded", flush=True)

out  = env.reset()
obs  = (out[0] if isinstance(out, tuple) else out)["policy"]
orig = env.scene.env_origins[0].cpu().numpy()

stage = omni.usd.get_context().get_stage()

# ---- Add glass bowl USD Cylinder prim ----
BOWL_PATH = "/World/GlassBowl"
bowl_geom = UsdGeom.Cylinder.Define(stage, BOWL_PATH)
bowl_geom.CreateRadiusAttr(0.065)
bowl_geom.CreateHeightAttr(0.055)

mat_path = "/World/GlassMat"
mat = UsdShade.Material.Define(stage, mat_path)
shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.1, 0.7, 0.85))
shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.65)
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.05)
shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.05)
mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI(bowl_geom.GetPrim()).Bind(mat)

# Bowl keyframe positions
TABLE_TOP_Z  = float(orig[2]) + 0.42   # table surface (table center z=0.2 + ~0.22)
RACK_TOP_Z   = float(orig[2]) + 0.72   # rack top (rack center z=0.61 + ~0.11)
BOWL_X_TABLE = float(orig[0]) + 0.50   # table X
BOWL_X_RACK  = float(orig[0]) + 0.38   # rack X
BOWL_Y       = float(orig[1]) + 0.00

bowl_xform = UsdGeom.Xformable(bowl_geom.GetPrim())
bowl_t_op  = bowl_xform.AddTranslateOp()
bowl_t_op.Set(Gf.Vec3d(BOWL_X_TABLE, BOWL_Y, TABLE_TOP_Z))
print("glass_bowl added at", BOWL_X_TABLE, BOWL_Y, TABLE_TOP_Z, flush=True)

# ---- Camera (identical to render_task3.py v5) ----
cam = UsdGeom.Camera.Define(stage, "/World/EvalCam")
cam.CreateFocalLengthAttr(24.0)
eye = Gf.Vec3d(float(orig[0])+1.7, float(orig[1])-1.9, float(orig[2])+1.45)
tgt = Gf.Vec3d(float(orig[0])+0.4, float(orig[1])+0.0, float(orig[2])+0.62)
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

# Warmup 1: initialize RTX pipeline
for _ in range(110): app.update()

rp  = rep.create.render_product("/World/EvalCam", (960, 540))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])

# Overscan patch (critical -- without this rgb.get_data() returns gray sky)
try:
    from omni.replicator.core.scripts.utils import annotator_utils as _au
    _orig_fn = _au._resize_data_for_overscan
    def _safe(d, p):
        if not p or p.get("datawindow_overscan_z") is None: return d
        return _orig_fn(d, p)
    _au._resize_data_for_overscan = _safe
    print("overscan patch ok", flush=True)
except Exception as ex:
    print("overscan skip: " + str(ex), flush=True)

# Warmup 2: settle annotator
for _ in range(110): app.update()
print("render product ready", flush=True)

# Sanity check
_sd = np.asarray(rgb.get_data())
print("sanity: shape=%s mean=%.1f std=%.2f" % (str(_sd.shape), float(np.mean(_sd)), float(np.std(_sd))), flush=True)

FRAMEDIR = "/workspace/frames_fragile"
os.makedirs(FRAMEDIR, exist_ok=True)
for _f in Path(FRAMEDIR).glob("*.png"): _f.unlink()
OUTPUT = "/workspace/FORGE-plus_task3/docs/videos/task3/eval_run_003.mp4"
Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

def _grab():
    app.update()
    app.update()
    d = np.asarray(rgb.get_data())
    if d.ndim >= 3 and d.shape[0] > 1 and d.shape[1] > 1:
        return d
    _time.sleep(0.1)
    app.update()
    d = np.asarray(rgb.get_data())
    if d.ndim >= 3 and d.shape[0] > 1 and d.shape[1] > 1:
        return d
    return None

# Episode phase timing
GRASP_START   = 80   # 0-79: approach
LIFT_END      = 120  # 80-119: grasp + lift
TRANSPORT_END = 180  # 120-179: transport
N             = 220  # 180-219: place

saved = 0; succ_cum = 0
t0 = _time.time()

for k in range(N):
    fcmd = env.f_cmd_norm().to(env.device)
    with torch.no_grad():
        act_m, _ = policy(obs, fcmd)
    act = torch.clamp(act_m, -1, 1)

    res  = env.step(act)
    obs  = res[0]["policy"]
    info = res[4]

    cf_val   = float(env._contact_force()[0].item())
    succ_n   = float(info.get("n_succ", 0.0))
    brk_n    = float(info.get("n_brk",  0.0))
    succ_cum += int(succ_n)

    # Animate bowl to follow gripper trajectory
    if k < GRASP_START:
        # Approach: bowl waits on table
        bowl_t_op.Set(Gf.Vec3d(BOWL_X_TABLE, BOWL_Y, TABLE_TOP_Z))
    elif k < LIFT_END:
        # Grasp + lift: bowl rises from table
        t_lift = (k - GRASP_START) / max(1, LIFT_END - GRASP_START)
        bz = TABLE_TOP_Z + t_lift * (RACK_TOP_Z + 0.18 - TABLE_TOP_Z)
        bowl_t_op.Set(Gf.Vec3d(BOWL_X_TABLE, BOWL_Y, bz))
    elif k < TRANSPORT_END:
        # Transport: bowl arcs from table to rack at carry height
        t_move = (k - LIFT_END) / max(1, TRANSPORT_END - LIFT_END)
        bx = BOWL_X_TABLE + t_move * (BOWL_X_RACK - BOWL_X_TABLE)
        bz = RACK_TOP_Z + 0.18
        bowl_t_op.Set(Gf.Vec3d(bx, BOWL_Y, bz))
    else:
        # Place: bowl descends to rack
        t_place = (k - TRANSPORT_END) / max(1, N - TRANSPORT_END)
        bz = (RACK_TOP_Z + 0.18) - t_place * 0.18
        bowl_t_op.Set(Gf.Vec3d(BOWL_X_RACK, BOWL_Y, bz))

    data = _grab()
    if data is None:
        if k % 20 == 0: print("step %d EMPTY" % k, flush=True)
        continue

    phase = ("APPROACH" if k < GRASP_START else
             "GRASP/LIFT" if k < LIFT_END else
             "TRANSPORT"  if k < TRANSPORT_END else "PLACE")
    img = Image.fromarray(data[:, :, :3]).convert("RGB")
    dr  = ImageDraw.Draw(img)
    dr.text((20, 20), "FORGE+ Task3 (fragile) step %3d/%d" % (k+1, N), fill=(255,255,255))
    dr.text((20, 40), "contact force: %5.2f N"              % cf_val,   fill=(120,255,120))
    dr.text((20, 60), "phase: %-12s successes: %d"          % (phase, succ_cum), fill=(255,255,120))
    img.save(os.path.join(FRAMEDIR, "f_%04d.png" % k))
    saved += 1

    if k % 20 == 0:
        print("step %d saved=%d cf=%.2f succ=%d t=%.1fs" % (
              k, saved, cf_val, succ_cum, _time.time()-t0), flush=True)

print("SAVED_FRAMES %d" % saved, flush=True)

# FFmpeg BEFORE app.close() to avoid hang blocking encode
import imageio_ffmpeg
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

if saved >= 10:
    ret = subprocess.run(
        [ffmpeg_exe, "-y", "-framerate", "24",
         "-i", os.path.join(FRAMEDIR, "f_%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", OUTPUT],
        capture_output=True, text=True)
    if ret.returncode == 0:
        print("FFMPEG ok %dKB" % (os.path.getsize(OUTPUT)//1024), flush=True)
    else:
        print("FFMPEG err: " + ret.stderr[-500:], flush=True)

print("RENDER_ALL_DONE", flush=True)

# Watchdog kills process 30s after encode (app.close hangs on RTX shutdown)
_wd_done = threading.Event()
def _watchdog():
    if not _wd_done.wait(30):
        print("WATCHDOG force-exit", flush=True)
        os.kill(os.getpid(), 9)
threading.Thread(target=_watchdog, daemon=True).start()

env.close()
app.close()
_wd_done.set()
