#!/usr/bin/env python3
# render_pick_place.py v1
# Live-physics RTX render of the FrankaPickPlaceEnv 7-phase fragile pick-and-place.
# Drives the trained ForceConditionedPolicy through one real episode and overlays a
# HUD with a force gauge (contact force vs LLM F_cmd budget and F_break), the active
# phase name, and table+rack visual geometry.
#
# Based on render_fragile_task3.py v1 proven RTX patterns:
#   * NEVER call rep.orchestrator.step() -- only env.step() + app.update() + annotator reads.
#   * overscan patch is required or rgb.get_data() returns a gray sky.
#   * ffmpeg encode BEFORE app.close(); watchdog force-exits (RTX shutdown hangs).
#
# Output: docs/videos/task3/pick_place_eval_001.mp4
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
    # Photorealistic quality: enable the full RT global-illumination stack.
    "--/rtx/reflections/enabled=true",
    "--/rtx/translucency/enabled=true",
    "--/rtx/indirectDiffuse/enabled=true",
    "--/rtx/ambientOcclusion/enabled=true",
    "--/rtx/directLighting/sampledLighting/enabled=true",
    "--/rtx/post/dlss/execMode=2",
]

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 960, "height": 540,
                     "extra_args": _EXTRA})
print("booted", flush=True)

import torch, carb
import omni.usd
from pxr import Gf, UsdGeom, UsdShade, Sdf, UsdLux, UsdPhysics
import omni.replicator.core as rep
from forge_plus.isaac_pick_place_env import (
    FrankaPickPlaceEnv, PickPlaceEnvCfg, PickPlacePhase, OBJ_KEYS, FRAGILE_OBJECTS,
)
from forge_plus.skills.policy_network import ForceConditionedPolicy, PolicyConfig
from PIL import Image, ImageDraw, ImageFont
print("imports ok", flush=True)

# ── HUD fonts (truetype if available, else PIL default) ──────────────────────
def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()
F_TITLE, F_BIG, F_MED, F_SM = _font(22), _font(20), _font(17), _font(14)

S = carb.settings.get_settings()
# Photorealistic RTX: path-traced global illumination with accumulation.
for _k in ["/rtx/reflections/enabled", "/rtx/translucency/enabled",
            "/rtx/indirectDiffuse/enabled", "/rtx/ambientOcclusion/enabled",
            "/rtx/directLighting/sampledLighting/enabled"]:
    S.set(_k, True)
try:
    # High-quality real-time RTX (reliable on a moving scene). Soft shadows,
    # multi-bounce indirect, denoised. (Path tracing is too slow per-frame for a
    # 360-frame physics sequence on this pod.)
    S.set("/rtx/directLighting/sampledLighting/samplesPerPixel", 4)
    S.set("/rtx/ambientOcclusion/enabled", True)
    S.set("/rtx/reflections/maxRoughness", 0.9)
    S.set("/rtx/indirectDiffuse/enabled", True)
    S.set("/rtx/indirectDiffuse/numBounces", 3)
    S.set("/rtx/sceneDb/ambientLightIntensity", 0.45)
    S.set("/rtx/post/aa/op", 3)                  # DLAA/TAA antialiasing
    S.set("/rtx/post/tonemap/op", 1)             # filmic tonemap
    print("high-quality RTX configured", flush=True)
except Exception as ex:
    print("rtx quality settings skipped: " + str(ex), flush=True)

# ── Environment (single env) ─────────────────────────────────────────────────
cfg = PickPlaceEnvCfg()
cfg.scene.num_envs = 1
cfg.gripper = "franka_panda"
env = FrankaPickPlaceEnv(cfg)
print("env built", flush=True)

# ── Policy (ckpt policy_cfg may be a plain dict after the weights_only re-save) ─
CKPT = "/workspace/FORGE-plus_task3/checkpoints/task3_pick_place_franka.pt"
ckpt = torch.load(CKPT, map_location=env.device, weights_only=False)
pc = ckpt["policy_cfg"]
pcfg = pc if isinstance(pc, PolicyConfig) else PolicyConfig(**pc)
policy = ForceConditionedPolicy(pcfg).to(env.device)
policy.load_state_dict(ckpt["policy_state_dict"])
policy.eval()
print("policy loaded", flush=True)

out  = env.reset()
obs  = (out[0] if isinstance(out, tuple) else out)["policy"]
orig = env.scene.env_origins[0].cpu().numpy()
c    = cfg

stage = omni.usd.get_context().get_stage()


def _pbr(mat_path, rgb, rough=0.5, metal=0.0, opacity=1.0, ior=1.5,
         clearcoat=0.0, specular=0.5, emissive=None):
    """Create a UsdPreviewSurface PBR material (renders physically under RTX)."""
    mat = UsdShade.Material.Define(stage, mat_path)
    sh  = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(rgb)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    sh.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(specular)
    sh.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(clearcoat)
    sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    if opacity < 1.0:
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)
    if emissive is not None:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(emissive)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _bind(prim_path, mat):
    try:
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            UsdShade.MaterialBindingAPI(prim).Bind(mat)
            print("material -> " + prim_path, flush=True)
    except Exception as ex:
        print("bind skip %s: %s" % (prim_path, ex), flush=True)


# ── Lighting: sky dome (soft ambient) + a key "sun" + a warm fill ───────────
def _light():
    dome = UsdLux.DomeLight.Define(stage, "/World/SkyDome")
    dome.CreateIntensityAttr(900.0)
    dome.CreateColorAttr((0.72, 0.80, 0.95))          # cool sky
    dome.CreateExposureAttr(0.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(2600.0)
    sun.CreateColorAttr((1.0, 0.96, 0.88))            # warm sun
    sun.CreateAngleAttr(0.6)                           # soft shadows
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-48.0, 18.0, 0.0))
    fill = UsdLux.SphereLight.Define(stage, "/World/Fill")
    fill.CreateIntensityAttr(18000.0)
    fill.CreateRadiusAttr(0.4)
    fill.CreateColorAttr((1.0, 0.92, 0.82))
    UsdGeom.Xformable(fill).AddTranslateOp().Set(
        Gf.Vec3d(float(orig[0]) - 0.8, float(orig[1]) - 1.2, float(orig[2]) + 1.6))
    print("lights added", flush=True)
_light()

# ── Ground plane (large, matte) so the robot sits in a real space ───────────
gp = UsdGeom.Mesh.Define(stage, "/World/Ground")
_S = 8.0
gz = float(orig[2]) + 0.001
gp.CreatePointsAttr([(-_S, -_S, gz), (_S, -_S, gz), (_S, _S, gz), (-_S, _S, gz)])
gp.CreateFaceVertexCountsAttr([4]); gp.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
gp.CreateNormalsAttr([(0, 0, 1)] * 4)
_bind("/World/Ground", _pbr("/World/Mats/Floor", (0.22, 0.23, 0.25), rough=0.55, specular=0.4))

# ── Table + rack PBR materials (existing env prims) ─────────────────────────
_bind("/World/envs/env_0/Table",
      _pbr("/World/Mats/Table", (0.42, 0.28, 0.16), rough=0.45, specular=0.4, clearcoat=0.3))  # varnished wood
_bind("/World/envs/env_0/Rack",
      _pbr("/World/Mats/Rack", (0.78, 0.80, 0.84), rough=0.18, metal=1.0))                      # brushed metal

# ── Fragile object visual prim (follows the gripper through the carry phases) ──
OBJ_KEY  = OBJ_KEYS[int(env._obj_cls[0].item())]
# Per-material physically-plausible look: translucent glass, glazed ceramic, metal.
_OBJ_MAT = {
    "glass_bowl":    dict(rgb=(0.45, 0.78, 0.85), rough=0.03, opacity=0.30, ior=1.5, specular=1.0),
    "ceramic_plate": dict(rgb=(0.92, 0.88, 0.78), rough=0.18, clearcoat=0.6, specular=0.7),
    "metal_plate":   dict(rgb=(0.80, 0.81, 0.84), rough=0.20, metal=1.0),
    "sturdy_mug":    dict(rgb=(0.62, 0.30, 0.22), rough=0.30, clearcoat=0.4),
}.get(OBJ_KEY, dict(rgb=(0.8, 0.5, 0.3), rough=0.3))
OBJ_PATH = "/World/FragileObj"
obj_geom = UsdGeom.Cylinder.Define(stage, OBJ_PATH)
obj_geom.CreateRadiusAttr(0.055)
obj_geom.CreateHeightAttr(0.05)
UsdShade.MaterialBindingAPI(obj_geom.GetPrim()).Bind(_pbr(OBJ_PATH + "/Mat", **_OBJ_MAT))
obj_xf = UsdGeom.Xformable(obj_geom.GetPrim())
obj_t  = obj_xf.AddTranslateOp()
OBJ_REST_W = (float(orig[0]) + c.table_x, float(orig[1]) + 0.0, float(orig[2]) + c.obj_rest_z)
obj_t.Set(Gf.Vec3d(*OBJ_REST_W))
print("fragile obj '%s' added" % OBJ_KEY, flush=True)

# ── Camera: frame the whole arm + table + rack (the place workspace) ──────────
cam = UsdGeom.Camera.Define(stage, "/World/EvalCam")
cam.CreateFocalLengthAttr(19.0)   # a bit wider so the full arm fits
eye = Gf.Vec3d(float(orig[0]) - 1.55, float(orig[1]) - 1.35, float(orig[2]) + 1.05)
tgt = Gf.Vec3d(float(orig[0]) + 0.22, float(orig[1]) + 0.14, float(orig[2]) + 0.50)
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

_sd = np.asarray(rgb.get_data())
print("sanity: shape=%s mean=%.1f std=%.2f" % (str(_sd.shape), float(np.mean(_sd)), float(np.std(_sd))), flush=True)

FRAMEDIR = "/workspace/frames_pick_place"
os.makedirs(FRAMEDIR, exist_ok=True)
for _f in Path(FRAMEDIR).glob("*.png"): _f.unlink()
OUTPUT = "/workspace/FORGE-plus_task3/docs/videos/task3/pick_place_eval_001.mp4"
Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

def _grab(settle=6):
    # Accumulate several RTX subframes per captured frame so the denoised
    # shadows / AO / reflections converge cleanly (the scene pose is fixed here).
    for _ in range(settle):
        app.update()
    d = np.asarray(rgb.get_data())
    if d.ndim >= 3 and d.shape[0] > 1 and d.shape[1] > 1:
        return d
    _time.sleep(0.1); app.update()
    d = np.asarray(rgb.get_data())
    if d.ndim >= 3 and d.shape[0] > 1 and d.shape[1] > 1:
        return d
    return None

PHASE_NAMES = [PickPlacePhase(i).name for i in range(7)]
F_CMD   = float(env._f_cmd[0].item())     # LLM-derived safe force budget (N)
F_BREAK = float(env._f_break[0].item())   # sampled break force for this episode (N)
GAUGE_MAX = max(F_BREAK * 1.15, F_CMD * 1.5, 30.0)

W, H = 960, 540
GX, GY, GW, GH = 24, H - 54, 360, 22   # force gauge geometry

def _hud(img, k, phase_idx, cf, broke, succ):
    dr = ImageDraw.Draw(img, "RGBA")
    # top banner
    dr.rectangle([0, 0, W, 86], fill=(0, 0, 0, 130))
    dr.text((20, 8),  "FORGE+ Task 3  -  Fragile Pick & Place", font=F_TITLE, fill=(255,255,255))
    dr.text((20, 38), "object: %-13s  F_break=%5.1f N   F_cmd(LLM)=%5.1f N"
            % (OBJ_KEY, F_BREAK, F_CMD), font=F_MED, fill=(200,220,255))
    dr.text((20, 60), "frame %3d   phase[%d/7]: %s" % (k+1, phase_idx+1, PHASE_NAMES[phase_idx]),
            font=F_MED, fill=(255,235,150))
    # status pill
    if broke:
        dr.text((W-150, 12), "BROKEN", font=F_BIG, fill=(255,80,80))
    elif succ:
        dr.text((W-150, 12), "SUCCESS", font=F_BIG, fill=(90,255,120))

    # force gauge panel
    dr.rectangle([GX-10, GY-26, GX+GW+150, GY+GH+12], fill=(0,0,0,140))
    dr.text((GX-4, GY-24), "contact force", font=F_SM, fill=(220,220,220))
    # track
    dr.rectangle([GX, GY, GX+GW, GY+GH], outline=(160,160,160), width=1, fill=(35,35,35,200))
    frac = max(0.0, min(1.0, cf / GAUGE_MAX))
    over = cf >= F_CMD
    danger = cf >= F_BREAK
    bar_col = (240,70,70) if danger else (245,180,60) if over else (90,220,120)
    dr.rectangle([GX, GY, GX + int(GW*frac), GY+GH], fill=bar_col)
    # F_cmd budget marker (yellow) and F_break marker (red)
    xc = GX + int(GW * min(1.0, F_CMD / GAUGE_MAX))
    xb = GX + int(GW * min(1.0, F_BREAK / GAUGE_MAX))
    dr.line([xc, GY-4, xc, GY+GH+4], fill=(255,230,90), width=2)
    dr.line([xb, GY-4, xb, GY+GH+4], fill=(255,90,90), width=2)
    dr.text((xc-10, GY-18), "F_cmd", font=F_SM, fill=(255,230,90))
    dr.text((xb-10, GY+GH+0), "F_brk", font=F_SM, fill=(255,120,120))
    dr.text((GX+GW+12, GY+1), "%5.1f N" % cf, font=F_BIG, fill=bar_col)
    return img

# Episode: run until terminal (+ short tail) or frame cap.
N_MAX   = 360
TAIL    = 20
ACT_BETA = 0.55   # action low-pass: smooths high-frequency control ripple so the
                  # rendered arm moves calmly (the policy still completes the place).
saved = 0
term_at = None
t0 = _time.time()
act_filt = torch.zeros(env.num_envs, 7, device=env.device)

for k in range(N_MAX):
    fcmd = env.f_cmd_norm().to(env.device)
    with torch.no_grad():
        act_m, _ = policy(obs, fcmd)
    act = torch.clamp(act_m, -1, 1)
    act_filt = ACT_BETA * act_filt + (1.0 - ACT_BETA) * act   # temporal smoothing

    res  = env.step(act_filt)
    obs  = res[0]["policy"]

    phase_idx = int(env._phase[0].item())
    cf        = float(env._contact_force()[0].item())
    broke     = bool(env._broke[0].item())
    succ      = bool(env._succeeded[0].item())

    # Animate fragile object: rest on table pre-grasp, follow gripper once grasped.
    ee_w = env._robot.data.body_pos_w[0, env._ee_idx].cpu().numpy()
    if phase_idx >= int(PickPlacePhase.GRASP) and not broke:
        obj_t.Set(Gf.Vec3d(float(ee_w[0]), float(ee_w[1]), float(ee_w[2]) - 0.10))
    else:
        obj_t.Set(Gf.Vec3d(*OBJ_REST_W))

    data = _grab()
    if data is None:
        if k % 20 == 0: print("step %d EMPTY" % k, flush=True)
        continue

    img = Image.fromarray(data[:, :, :3]).convert("RGB")
    img = _hud(img, k, phase_idx, cf, broke, succ)
    img.save(os.path.join(FRAMEDIR, "f_%04d.png" % saved))
    saved += 1

    if k % 20 == 0:
        print("step %d saved=%d phase=%d cf=%.2f succ=%s brk=%s t=%.1fs"
              % (k, saved, phase_idx, cf, succ, broke, _time.time()-t0), flush=True)

    if (succ or broke) and term_at is None:
        term_at = k
        print("TERMINAL at step %d (succ=%s brk=%s)" % (k, succ, broke), flush=True)
    if term_at is not None and k - term_at >= TAIL:
        break

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
        print("FFMPEG ok %dKB -> %s" % (os.path.getsize(OUTPUT)//1024, OUTPUT), flush=True)
    else:
        print("FFMPEG err: " + ret.stderr[-500:], flush=True)
else:
    print("TOO_FEW_FRAMES %d -- no encode" % saved, flush=True)

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
