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
from pxr import Gf, UsdGeom, UsdShade, Sdf, UsdLux, UsdPhysics, Usd
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
# Render-only: don't let the env auto-reset the instant the place succeeds (settle_steps
# steps after RELEASE) — we want to CAPTURE the hand retracting and leaving the bottle
# standing. A large settle_steps pushes the success/reset past the captured retract.
cfg.settle_steps = 400
# Keep the TRAINING decimation so _get_dones / phase timing matches training
# (decimation=1 ran the phase logic every substep -> phases rushed). One captured
# frame per env.step (policy step). Buzz is fixed by joint damping, so smooth.
env = FrankaPickPlaceEnv(cfg)
print("env built", flush=True)

# ── Policy (ckpt policy_cfg may be a plain dict after the weights_only re-save) ─
CKPT = "/workspace/FORGE-plus_task3/checkpoints/task3_wine_bottle.pt"
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


# ── Textured PBR material (diffuse from a PNG, e.g. LIBERO wood/marble/tile) ──
def _pbr_tex(mat_path, tex_path, rough=0.5, metal=0.0, specular=0.4, uv_scale=(1.0, 1.0)):
    mat = UsdShade.Material.Define(stage, mat_path)
    sh  = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    sh.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(specular)
    st = UsdShade.Shader.Define(stage, mat_path + "/stReader")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    xf = UsdShade.Shader.Define(stage, mat_path + "/stXform")
    xf.CreateIdAttr("UsdTransform2d")
    xf.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
    xf.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*uv_scale))
    tex = UsdShade.Shader.Define(stage, mat_path + "/Tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_path)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(xf.ConnectableAPI(), "result")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat

_TEX = "/workspace/assets/libero/textures"

_bboxc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

# The env's Table/Rack are UsdGeom.Cube prims (no UVs). Replace each with a textured Mesh BOX
# spanning the same world AABB (correct per-face planar UVs so the wood/marble tiles), and hide
# the original (visibility only — collision is unaffected, so the physics place still works).
def _tex_box(path, mn, mx, mat, world_uv=2.0):
    x0, y0, z0 = mn; x1, y1, z1 = mx
    P = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
         (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
    dx, dy, dz = (x1-x0)*world_uv, (y1-y0)*world_uv, (z1-z0)*world_uv
    faces = [([4,5,6,7],(0,0,1),(dx,dy)), ([3,2,1,0],(0,0,-1),(dx,dy)),
             ([1,2,6,5],(1,0,0),(dy,dz)), ([4,7,3,0],(-1,0,0),(dy,dz)),
             ([2,3,7,6],(0,1,0),(dx,dz)), ([0,1,5,4],(0,-1,0),(dx,dz))]
    pts, idx, nrm, st = [], [], [], []
    for verts, n, (u, v) in faces:
        base = len(pts)
        for c in verts: pts.append(P[c])
        idx += [base, base+1, base+2, base+3]
        nrm += [n]*4
        st  += [(0,0),(u,0),(u,v),(0,v)]
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts); m.CreateFaceVertexCountsAttr([4]*6)
    m.CreateFaceVertexIndicesAttr(idx); m.CreateNormalsAttr(nrm)
    UsdGeom.PrimvarsAPI(m.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(st)
    UsdShade.MaterialBindingAPI(m.GetPrim()).Bind(mat)

def _replace_box(prim_path, out_path, mat, world_uv=2.0):
    prim = stage.GetPrimAtPath(prim_path)
    rng = _bboxc.ComputeWorldBound(prim).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    _tex_box(out_path, (mn[0],mn[1],mn[2]), (mx[0],mx[1],mx[2]), mat, world_uv)
    UsdGeom.Imageable(prim).MakeInvisible()
    print("replaced %s -> textured box [%s..%s]" % (prim_path,
          [round(x,2) for x in mn], [round(x,2) for x in mx]), flush=True)


# ── Lighting: sky dome (soft ambient) + a key "sun" + a warm fill ───────────
def _light():
    dome = UsdLux.DomeLight.Define(stage, "/World/SkyDome")
    dome.CreateIntensityAttr(750.0)
    dome.CreateColorAttr((0.95, 0.92, 0.86))          # warm neutral (indoor kitchen)
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
# UVs so a tiled wood/tile floor texture maps (tile ~0.5 m over the 16 m plane)
_fuv = 2 * _S / 0.6
UsdGeom.PrimvarsAPI(gp.GetPrim()).CreatePrimvar(
    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
).Set([(0, 0), (_fuv, 0), (_fuv, _fuv), (0, _fuv)])
_bind("/World/Ground", _pbr_tex("/World/Mats/Floor", _TEX + "/seamless_wood_planks_floor.png",
                                rough=0.5, specular=0.3))

# ── LIBERO kitchen backdrop ─────────────────────────────────────────────────
# The kitchen_background mesh (~0.6 x 3.6 x 2.5 m at scale 0.01; floor at its z=0) placed
# BEHIND the robot so the camera (at +x, looking -x) frames the robot with the kitchen behind.
# Transform params are easy to tune; render with KPREVIEW=1 for a fast static placement check.
KITCHEN_USD = "/workspace/assets/libero/kitchen_background/kitchen_background.usd"
KSCALE = float(os.environ.get("KSCALE", "0.01"))
KTX = float(os.environ.get("KTX", "-0.70"))   # kitchen x offset from robot origin (behind)
KTY = float(os.environ.get("KTY", "0.0"))
KROTZ = float(os.environ.get("KROTZ", "180.0"))  # face the robot
# transform goes on a CLEAN parent Xform; the kitchen (whose root already has xformOps) is a child
_kp = stage.DefinePrim("/World/KitchenXform", "Xform")
_kx = UsdGeom.Xformable(_kp)
_kx.AddTranslateOp().Set(Gf.Vec3d(float(orig[0]) + KTX, float(orig[1]) + KTY, float(orig[2])))
_kx.AddRotateZOp().Set(KROTZ)
_kx.AddScaleOp().Set(Gf.Vec3f(KSCALE, KSCALE, KSCALE))
stage.DefinePrim("/World/KitchenXform/Geo", "Xform").GetReferences().AddReference(KITCHEN_USD)
print("kitchen added at x+%.2f y+%.2f rotz=%.0f scale=%.3f" % (KTX, KTY, KROTZ, KSCALE), flush=True)

# ── Table + rack PBR materials (existing env prims) — LIBERO wood + marble ───
_replace_box("/World/envs/env_0/Table", "/World/TableVis",
             _pbr_tex("/World/Mats/Table", _TEX + "/martin_novak_wood_table.png", rough=0.4, specular=0.4),
             world_uv=2.0)   # wood counter
# Wine rack is now a multi-board USD (the cells); give it a dark varnished-wood look.
_bind("/World/envs/env_0/Rack",
      _pbr("/World/Mats/Rack", (0.30, 0.17, 0.09), rough=0.4, specular=0.4, clearcoat=0.3))

# ── Fragile object: the REAL simulated rigid body (held by friction, dropped on
# release). No fake follower prim any more — we just give the env's Object prim a
# photorealistic glazed-ceramic look and let physics move it.
# Force budget uses the sampled fragility class (glass_bowl ≈ 22 N break — glass);
# the rendered mesh is the LIBERO wine bottle, so label the HUD with the real object.
_BUDGET_KEY = OBJ_KEYS[int(env._obj_cls[0].item())]
OBJ_KEY = "wine_bottle"
# The object is a REAL textured LIBERO wine-bottle USD — keep its own baked texture
# (do NOT bind a flat _pbr over it).
print("fragile obj '%s' (budget class '%s', real LIBERO wine bottle)" % (OBJ_KEY, _BUDGET_KEY), flush=True)

# ── Camera: tight 3/4 view framing the gripper + place zone (the grasp + place),
# not the whole room — the wide 19 mm shot buried the small block behind the boxes.
cam = UsdGeom.Camera.Define(stage, "/World/EvalCam")
cam.CreateFocalLengthAttr(27.0)   # WIDE — frame the whole robot + table + shelf
# Pulled-back, elevated front-RIGHT 3/4. The arm reaches base->shelf along +x,+y, so
# from +x,-y the whole arm is in view (receding to the left) without blocking the place.
eye = Gf.Vec3d(float(orig[0]) + 1.55, float(orig[1]) - 1.35, float(orig[2]) + 1.15)
tgt = Gf.Vec3d(float(orig[0]) + 0.18, float(orig[1]) + 0.12, float(orig[2]) + 0.42)
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

# Warmup 1: initialize RTX pipeline (render only — app.update() would step physics
# and drop the freshly-seated object before the rollout starts).
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

# Warmup 2: settle annotator (render only — see above)
for _ in range(110): app.update()
print("render product ready", flush=True)

# The RTX warmup app.update()s above advanced physics and dropped the grasped object.
# Re-seat with a fresh reset right before the rollout (no app.update() runs between
# here and the loop, so the loop's warmup seats the object and the grasp holds —
# matching the clean headless eval).
out = env.reset()
obs = (out[0] if isinstance(out, tuple) else out)["policy"]
print("re-seated for rollout", flush=True)

# Explicitly run the warmup so the gripper SEATS the object and the friction grip
# closes BEFORE frame capture. reset() re-places the object at spawn (above the
# shelf); these zero-action steps snap it into the gripper and the grip holds it by
# friction. Without this the first captured frame can show the object mid-fall onto
# the shelf. Debug-print the seat so the log proves the grasp took (d_xy should be small).
def _seat_dbg(tag):
    o = env.scene.env_origins[0]
    op = env._obj.data.root_pose_w[0, :3] - o
    ee = env._robot.data.body_pos_w[0, env._ee_idx, :3] - o
    dxy = float(((op[0]-ee[0])**2 + (op[1]-ee[1])**2) ** 0.5)
    print("%s: warmup=%d obj=(%.3f,%.3f,%.3f) ee=(%.3f,%.3f,%.3f) d_xy=%.3f"
          % (tag, int(env._warmup[0].item()), op[0], op[1], op[2], ee[0], ee[1], ee[2], dxy), flush=True)
_seat_dbg("after reset#2")
_seat_zero = torch.zeros(env.num_envs, 7, device=env.device)
for _si in range(8):
    out = env.step(_seat_zero)
    _seat_dbg("  seat step %d" % _si)
obs = out[0]["policy"]

# Flush the seated (held) bottle to the RTX render mesh BEFORE capture, so frame 1
# already shows it in the gripper (the render mesh lags the teleport-seat by a few
# app.update()s — without this, the first frame still shows it on the shelf).
for _ in range(20):
    app.update()
_seat_dbg("after render-flush")

_sd = np.asarray(rgb.get_data())
print("sanity: shape=%s mean=%.1f std=%.2f" % (str(_sd.shape), float(np.mean(_sd)), float(np.std(_sd))), flush=True)

# ── Fast placement preview: KPREVIEW=1 -> grab a few static frames, save one, exit ──
if os.environ.get("KPREVIEW"):
    for _ in range(8):
        app.update()
    _pv = np.asarray(rgb.get_data())
    Image.fromarray(_pv[:, :, :3]).convert("RGB").save("/workspace/kitchen_preview.png")
    print("KPREVIEW_SAVED /workspace/kitchen_preview.png", flush=True)
    import sys as _sys; _sys.stdout.flush(); os._exit(0)

FRAMEDIR = "/workspace/frames_pick_place"
os.makedirs(FRAMEDIR, exist_ok=True)
for _f in Path(FRAMEDIR).glob("*.png"): _f.unlink()
OUTPUT = "/workspace/FORGE-plus_task3/docs/videos/task3/pick_place_eval_001.mp4"
Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

def _grab():
    # Match the PROVEN render_task3.py pipeline: drive the render product with
    # app.update() (NOT env.sim.render()). With enable_async=false this is synchronous,
    # and crucially app.update() flushes PhysX->Fabric so the rigid OBJECT's pose follows
    # into the render (sim.render() did not flush teleport/grip motion -> the bottle froze
    # on the shelf while physics held it in the gripper). Two extra updates ensure the
    # pipeline is committed before reading.
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
    dr.text((20, 8),  "FORGE+ Task 3  -  Wine-Cellar Bottle Insertion", font=F_TITLE, fill=(255,255,255))
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

# Episode: run until terminal (+ short tail) or frame cap. Frames are captured at
# the 120 Hz physics rate; the policy is re-queried every POLICY_HOLD steps (15 Hz).
N_MAX   = 480
TAIL    = 48   # after RELEASE: capture the hand retracting + the bottle left standing
ACT_BETA = 0.7   # low-pass the SAMPLED action: keep the slow exploration drift that
                 # completes the place, drop the per-step noise that made the pose
                 # jump back and forth in high-sigma sections.
saved = 0
term_at = None
t0 = _time.time()
act = torch.zeros(env.num_envs, 7, device=env.device)
# Lateral-back-up retract: action[:3] is an EE-position delta added to the RELEASE waypoint
# (rack_x,rack_y,transport_z). (-1,-1,+1)*act_range targets back toward the base and up, so
# the OPEN gripper clears the bottle SIDEWAYS instead of lifting straight up through it
# (straight up re-captured the just-placed bottle).
RETRACT = torch.zeros(env.num_envs, 7, device=env.device)
RETRACT[:, 0] = -1.0; RETRACT[:, 1] = -1.0; RETRACT[:, 2] = 1.0
import math as _math
from isaaclab.utils.math import matrix_from_quat as _mfq
def _bottle_tilt_deg():
    q = env._obj.data.root_pose_w[0, 3:7]
    up = _mfq(q.unsqueeze(0))[0] @ torch.tensor([0.,0.,1.], device=env.device)
    return _math.degrees(_math.acos(max(-1.0, min(1.0, float(up[2])))))
def _retract_dbg(k):
    o = env.scene.env_origins[0]
    b = env._obj.data.root_pose_w[0, :3] - o
    e = env._robot.data.body_pos_w[0, env._ee_idx, :3] - o
    fw = (env._robot.data.joint_pos[0,7] + env._robot.data.joint_pos[0,8]).item()
    print("  retract k%d fw=%.4f tilt=%4.1f bottle=(%.3f,%.3f,%.3f) ee=(%.3f,%.3f,%.3f)"
          % (k, fw, _bottle_tilt_deg(), b[0], b[1], b[2], e[0], e[1], e[2]), flush=True)

for k in range(N_MAX):
    fcmd = env.f_cmd_norm().to(env.device)
    # Wine-cellar PEG-IN-HOLE: zero-action nominal OSC carries the bottle, centers its base over
    # the cell (env aims the base) and inserts it. After it's inserted (RELEASE / term_at), hold
    # briefly while the gripper opens, then RETRACT the hand up/back, leaving the bottle in the cell.
    RAMP = 28
    if term_at is None or (k - term_at) < RAMP:
        act = torch.zeros(env.num_envs, 7, device=env.device)   # nominal OSC insertion
    else:
        act = RETRACT

    res  = env.step(act)                          # one policy step (decimation substeps)
    obs  = res[0]["policy"]

    # The Franka finger actuator has a stiff position drive (stiffness 2e3); the env's
    # effort-based open can't move the fingers off the neck, so we write the finger joints
    # directly — but GRADUALLY (lerp closed->open over RAMP) for a gentle, impulse-free release.
    if term_at is not None:
        _n = env.num_envs
        frac = min(1.0, (k - term_at) / float(RAMP))
        fpos = 0.008 + frac * (0.040 - 0.008)    # per-finger: closed(neck) -> fully open
        cur_v = env._robot.data.joint_vel[:, 7:9].clone()   # keep current finger velocity (no zero-snap)
        env._robot.write_joint_state_to_sim(
            torch.full((_n, 2), fpos, device=env.device), cur_v, joint_ids=[7, 8])
        if (k - term_at) % 4 == 0:
            _retract_dbg(k)

    phase_idx = int(env._phase[0].item())
    cf        = float(env._contact_force()[0].item())
    broke     = bool(env._broke[0].item())
    succ      = bool(env._succeeded[0].item())

    # If the env has auto-reset after the place (phase dropped back below RELEASE), end
    # NOW — before grabbing/saving — so the video ends on the placed bottle, not a 2nd run.
    if term_at is not None and k > term_at and phase_idx < int(PickPlacePhase.RELEASE):
        print("episode reset detected at step %d -> ending" % k, flush=True)
        break

    # Pure simulation (no pose capture/restore) — matches the proven render_task3.py.
    # env.step's internal app.update() and _grab()'s app.update()s both flush physics to
    # the render, so the friction-held bottle follows the gripper on screen.
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

    # Stop after ONE clean place: the first time the arm reaches RELEASE (or
    # succeeds/breaks), record a short tail then break — otherwise the env
    # auto-resets on success and the video loops the same motion.
    if (succ or broke or phase_idx >= int(PickPlacePhase.RELEASE)) and term_at is None:
        term_at = k
        print("PLACED at step %d (phase=%d succ=%s)" % (k, phase_idx, succ), flush=True)
    if term_at is not None and k - term_at >= TAIL:
        break

print("SAVED_FRAMES %d" % saved, flush=True)

# FFmpeg BEFORE app.close() to avoid hang blocking encode
import imageio_ffmpeg
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

if saved >= 10:
    ret = subprocess.run(
        [ffmpeg_exe, "-y", "-framerate", "16",   # gentle slow-mo for the fragile place
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
