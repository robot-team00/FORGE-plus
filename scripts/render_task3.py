#!/usr/bin/env python3
# render_task3.py v5
# ROOT CAUSE FOUND (via v5-diag): rep.orchestrator.step() at k=0 schedules
# a deferred render callback. When k=1's env.step() -> app.update() fires,
# that callback tries to render again, but rendering needs app.update() to
# process -> circular deadlock. NEVER saw DIAG k=1 D (env.step DONE).
# FIX: Remove rep.orchestrator.step() entirely from the render loop.
# Annotators update on every app.update() (driven by env.step internally).
import os, sys, subprocess, time as _time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

_PC = "/workspace/persist"
os.environ.update({
    "HOME": _PC + "/ovhome",
    "CUDA_CACHE_PATH": _PC + "/shadercache/cuda",
    "__GL_SHADER_DISK_CACHE": "1",
    "__GL_SHADER_DISK_CACHE_PATH": _PC + "/shadercache/gl",
    "MPLBACKEND": "Agg",
    "DISPLAY": ":1",
})
for _d in (os.environ["HOME"], os.environ["CUDA_CACHE_PATH"],
           os.environ["__GL_SHADER_DISK_CACHE_PATH"]):
    os.makedirs(_d, exist_ok=True)

sys.path.insert(0, "/workspace/FORGE-plus_task3")

# enable_async=false: prevent isaacsim.core.throttling from toggling
# asyncRendering=True on timeline stop. With this flag, app.update() is
# synchronous so annotator data is committed before it returns.
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
from pxr import Gf, UsdGeom
import omni.replicator.core as rep
from forge_plus.isaac_place_env import FrankaPlaceEnv, PlaceEnvCfg
from forge_plus.skills.policy_network import ForceConditionedPolicy
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

# Warmup 2: settle render product (annotator will be populated after these)
for _ in range(110): app.update()
print("render product ready", flush=True)

FRAMEDIR = "/workspace/frames_task3"
os.makedirs(FRAMEDIR, exist_ok=True)
for _f in Path(FRAMEDIR).glob("*.png"): _f.unlink()
OUTPUT = "/workspace/FORGE-plus_task3/docs/eval_episode_task3.mp4"
Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

def _grab():
    # v5: no orchestrator.step() -- annotator updated by env.step's app.update()
    # With enable_async=false, render is sync so buffer is committed immediately.
    # Add 2 extra app.update() calls to ensure pipeline flush before reading.
    app.update()
    app.update()
    d = np.asarray(rgb.get_data())
    if d.ndim >= 3 and d.shape[0] > 1 and d.shape[1] > 1:
        return d
    # One more attempt after brief sleep
    _time.sleep(0.1)
    app.update()
    d = np.asarray(rgb.get_data())
    if d.ndim >= 3 and d.shape[0] > 1 and d.shape[1] > 1:
        return d
    return None

N = 220
saved = 0; succ_cum = 0
t0 = _time.time()

for k in range(N):
    fcmd = env.f_cmd_norm().to(env.device)
    with torch.no_grad():
        m, _ = policy(obs, fcmd)
    act = torch.clamp(m, -1, 1)

    # Physics step -- internally calls app.update() which drives render product
    # NO orchestrator.step() here (deadlocks via deferred callback)
    res = env.step(act)
    obs  = res[0]["policy"]
    info = res[4]

    cf_val  = float(env._contact_force()[0].item())
    succ_n  = float(info.get("n_succ", 0.0))
    brk_n   = float(info.get("n_brk",  0.0))
    succ_cum += int(succ_n)

    data = _grab()
    if data is None:
        if k % 20 == 0: print("step %d EMPTY frame" % k, flush=True)
        continue

    img = Image.fromarray(data[:, :, :3]).convert("RGB")
    dr  = ImageDraw.Draw(img)
    st  = "SUCCESS" if succ_n > 0 else ("BREAK" if brk_n > 0 else "contact")
    dr.text((20, 20), "FORGE+ Task3 step %3d/%d" % (k+1, N), fill=(255,255,255))
    dr.text((20, 40), "contact force: %5.2f N"   % cf_val,    fill=(120,255,120))
    dr.text((20, 60), "state: %s successes: %d"  % (st, succ_cum), fill=(255,255,120))
    img.save(os.path.join(FRAMEDIR, "f_%04d.png" % k))
    saved += 1

    if k % 20 == 0:
        print("step %d saved=%d cf=%.2f succ=%d t=%.1fs" % (
              k, saved, cf_val, succ_cum, _time.time()-t0), flush=True)

print("SAVED_FRAMES %d" % saved, flush=True)
env.close()
app.close()

if saved >= 10:
    ret = subprocess.run(["ffmpeg","-y","-framerate","24",
                          "-i", os.path.join(FRAMEDIR,"f_%04d.png"),
                          "-c:v","libx264","-pix_fmt","yuv420p","-crf","20",OUTPUT],
                         capture_output=True, text=True)
    if ret.returncode == 0:
        print("FFMPEG ok %dKB" % (os.path.getsize(OUTPUT)//1024), flush=True)
    else:
        print("FFMPEG err: " + ret.stderr[-500:], flush=True)
print("RENDER_ALL_DONE", flush=True)
