#!/usr/bin/env python3
# render_forge_min.py — render the LEARNED forge insertion policy using the PROVEN
# render_task3.py harness (which renders reliably), swapping only the env + policy + camera.
# No kitchen USD, no PBR material decoration — a clean RTX scene to isolate whether the
# FrankaPickPlaceEnv itself renders. GI/DLSS off (NGX is broken on this pod).
import os, sys, subprocess, time as _time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            try: return ImageFont.truetype(p, sz)
            except Exception: pass
    return ImageFont.load_default()
F_TITLE, F_BIG, F_MED, F_SM = _font(22), _font(20), _font(17), _font(14)

_PC = "/workspace/persist"
os.environ.update({
    "HOME": _PC + "/ovhome",
    "MPLBACKEND": "Agg",
    "DISPLAY": ":99",
})
os.makedirs(os.environ["HOME"], exist_ok=True)
sys.path.insert(0, "/workspace/FORGE-plus_task3")

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
from pxr import Gf, UsdGeom, UsdLux
import omni.replicator.core as rep
from forge_plus.isaac_pick_place_env import FrankaPickPlaceEnv, PickPlaceEnvCfg
from forge_plus.skills.policy_network import ForceConditionedPolicy, PolicyConfig
print("imports ok", flush=True)

S = carb.settings.get_settings()
for _k in ["/rtx/reflections/enabled", "/rtx/translucency/enabled",
            "/rtx/indirectDiffuse/enabled", "/rtx/ambientOcclusion/enabled",
            "/rtx/directLighting/sampledLighting/enabled"]:
    S.set(_k, False)

# RELEASE=1 renders the learned safe-release policy (8-dim action, lets go of the bottle);
# default renders the insertion-only policy (7-dim, force-holds without releasing).
RELEASE = bool(int(os.environ.get("RELEASE", "0")))
cfg = PickPlaceEnvCfg()
cfg.scene.num_envs = 1
cfg.gripper = "franka_panda"
cfg.settle_steps = 400
cfg.forge_mode = True
cfg.grasp_topdown = False
cfg.forge_no_term = True
# render_minimal skips the filtered insert-sensor + compliant material. That changes the force
# OBSERVATION the policy sees, so the arm behaves differently than in training. RENDER_MINIMAL=0
# uses the full training sensor set so the render matches the trained descent.
cfg.render_minimal = bool(int(os.environ.get("RENDER_MINIMAL", "0")))
cfg.forge_obj_cls = 2
if RELEASE:
    cfg.forge_release_mode = True   # 8th action dim: the policy commands the gripper release
    # After the learned release, a moderate scripted retract pulls the open hand up-and-back
    # clear of the placed bottle. (Set HYBRID_RETRACT=0 for no retract — hand just lets go & holds.)
    cfg.forge_hybrid_retract = bool(int(os.environ.get("HYBRID_RETRACT", "1")))

# Restore the real render() (the env no-ops it for training speed).
from isaaclab.envs import DirectRLEnv as _DRL
FrankaPickPlaceEnv.render = _DRL.render
env = FrankaPickPlaceEnv(cfg)
# render_minimal skips _apply_rack_compliance; apply it manually (the non-minimal env already
# applies it in setup, so only do this when render_minimal is on to avoid double-binding).
try:
    if cfg.render_minimal and cfg.contact_stiffness > 0 and hasattr(env, "_apply_rack_compliance"):
        env._apply_rack_compliance()
        print("rack compliance applied", flush=True)
except Exception as _e:
    print("rack compliance skip: " + str(_e), flush=True)
print("env built", flush=True)

CKPT = ("/workspace/FORGE-plus_task3/checkpoints/task3_forge_release.pt" if RELEASE
        else "/workspace/FORGE-plus_task3/checkpoints/task3_forge_insert.pt")
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

stage = omni.usd.get_context().get_stage()

# ── Lighting: sky dome (soft ambient) + key "sun" + warm fill (direct lights work with GI off).
dome = UsdLux.DomeLight.Define(stage, "/World/SkyDome")
dome.CreateIntensityAttr(750.0); dome.CreateColorAttr((0.95, 0.92, 0.86))
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(3200.0); sun.CreateColorAttr((1.0, 0.96, 0.88)); sun.CreateAngleAttr(0.6)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-48.0, 18.0, 0.0))
fill = UsdLux.SphereLight.Define(stage, "/World/Fill")
fill.CreateIntensityAttr(26000.0); fill.CreateRadiusAttr(0.4); fill.CreateColorAttr((1.0, 0.92, 0.82))
UsdGeom.Xformable(fill).AddTranslateOp().Set(
    Gf.Vec3d(float(orig[0]) - 0.8, float(orig[1]) - 1.2, float(orig[2]) + 1.6))
S.set("/rtx/sceneDb/ambientLightIntensity", 0.6)
print("lights added", flush=True)

# ── Ground plane (large, matte, untextured) so the robot sits in a real space + casts shadows.
gp = UsdGeom.Mesh.Define(stage, "/World/Ground")
_S = 8.0; gz = float(orig[2]) + 0.001
gp.CreatePointsAttr([(-_S, -_S, gz), (_S, -_S, gz), (_S, _S, gz), (-_S, _S, gz)])
gp.CreateFaceVertexCountsAttr([4]); gp.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
gp.CreateNormalsAttr([(0, 0, 1)] * 4)
gp.CreateDisplayColorAttr([(0.32, 0.30, 0.27)])
print("ground added", flush=True)

# ── Camera: pulled-back, elevated front-right 3/4 framing the whole robot + table + rack
# (render_pick_place's tuned values — the arm reaches base->shelf along +x,+y).
cam = UsdGeom.Camera.Define(stage, "/World/EvalCam")
cam.CreateFocalLengthAttr(27.0)
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

# Warmup 1
for _ in range(110): app.update()

rp  = rep.create.render_product("/World/EvalCam", (960, 540))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])
print("rgb attached", flush=True)

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

for _ in range(110): app.update()
_d = np.asarray(rgb.get_data())
print("render product ready (shape=%s)" % str(_d.shape), flush=True)

FRAMEDIR = "/workspace/frames_forge_min"
os.makedirs(FRAMEDIR, exist_ok=True)
for _f in Path(FRAMEDIR).glob("*.png"): _f.unlink()
OUTPUT = ("/workspace/FORGE-plus_task3/docs/videos/task3/forge_release.mp4" if RELEASE
          else "/workspace/FORGE-plus_task3/docs/videos/task3/forge_insert.mp4")
Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

# Force-gauge budget: F_cmd (the gentle FORGE force target) and F_break (the fragility limit).
F_CMD   = float(env._f_cmd[0].item())
F_BREAK = float(env._f_break[0].item())
GAUGE_MAX = max(F_BREAK * 1.15, F_CMD * 1.5, 30.0)
W, H = 960, 540
GX, GY, GW, GH = 24, H - 48, 360, 22   # force gauge geometry (bottom-left)

def _grab():
    app.update(); app.update()
    d = np.asarray(rgb.get_data())
    if d.ndim >= 3 and d.shape[0] > 1 and d.shape[1] > 1:
        return d
    _time.sleep(0.1); app.update()
    d = np.asarray(rgb.get_data())
    if d.ndim >= 3 and d.shape[0] > 1 and d.shape[1] > 1:
        return d
    return None

# The warmup app.update()s advanced physics and disturbed the grasp. Reset to the env's clean
# post-reset state, then run the LEARNED POLICY (not scripted zero actions) for a few uncaptured
# steps so the env's grip logic closes on the bottle before capture starts. Everything the arm
# and gripper do is driven by the env + policy — nothing here is scripted.
out = env.reset()
obs = (out[0] if isinstance(out, tuple) else out)["policy"]
for _si in range(8):
    with torch.no_grad():
        m, _ = policy(obs, env.f_cmd_norm().to(env.device))
    out = env.step(torch.clamp(m, -1, 1))
    obs = out[0]["policy"]
for _ in range(3): app.update()
print("grasp seated by env+policy", flush=True)

N = 240            # hard cap
TAIL = 16          # stop ~0.7 s after the bottle is PLACED (so the static hold doesn't drag on)
placed_at = None
saved = 0
t0 = _time.time()
for k in range(N):
    fcmd = env.f_cmd_norm().to(env.device)
    with torch.no_grad():
        m, s = policy(obs, fcmd)
    act = m.clone()
    # The arm follows the deterministic mean (smooth). The release dim (action[7]) is SAMPLED
    # from the policy's own distribution — the policy commits to the drop stochastically (as it
    # was trained), and its mean is height-conditioned so it releases once the bottle is seated.
    if RELEASE and act.shape[-1] >= 8:
        act[:, 7] = m[:, 7] + s[:, 7] * torch.randn_like(s[:, 7])
    act = torch.clamp(act, -1, 1)
    res = env.step(act)
    obs = res[0]["policy"]
    info = res[4]

    cf_val = float(env._contact_force()[0].item())
    succ   = bool(env._succeeded[0].item())   # read-only HUD diagnostics — not used to script
    broke  = bool(env._broke[0].item())
    rel    = bool(env._released[0].item()) if RELEASE else False
    setup_active = (int(env._setup_ctr[0].item()) > 0)
    if RELEASE and k % 10 == 0:
        from isaaclab.utils.math import matrix_from_quat as _mfq
        _base = env._obj.data.root_pose_w[0, :3] - env.scene.env_origins[0]
        _upz  = float(_mfq(env._obj.data.root_pose_w[:, 3:7])[0, 2, 2].item())
        _vel  = float(env._obj.data.root_vel_w[0, :3].norm().item())
        _incell = (abs(float(_base[0]) - env.cfg.rack_x) < 0.04) and (abs(float(_base[1]) - env.cfg.rack_y) < 0.04)
        _atfloor = abs(float(_base[2]) - env.cfg.cell_floor_z) < env.cfg.release_floor_tol
        print("  DIAG step %d rel=%d in_cell=%d at_floor=%d up_z=%.3f vel=%.3f base=(%.3f,%.3f,%.3f) succ=%d"
              % (k, int(rel), int(_incell), int(_atfloor), _upz, _vel,
                 float(_base[0]), float(_base[1]), float(_base[2]), int(succ)), flush=True)

    data = _grab()
    if data is None:
        if k % 20 == 0: print("step %d EMPTY" % k, flush=True)
        continue
    img = Image.fromarray(data[:, :, :3]).convert("RGB")
    dr  = ImageDraw.Draw(img, "RGBA")
    GREEN  = (120, 255, 120)   # LEARNED
    ORANGE = (255, 170,  60)   # SCRIPTED
    HYBRID = bool(int(os.environ.get("HYBRID_RETRACT", "1")))
    if RELEASE:
        st  = "PLACED" if succ else ("BREAK" if broke else ("RELEASED" if rel else "inserting"))
        title = "FORGE+  wine-bottle place + release"
        # What is driving the robot RIGHT NOW (honest learned-vs-scripted label):
        if setup_active:
            ctrl, cdesc, ccol = "SCRIPTED", "approach positioning (up-over-down)", ORANGE
        elif not rel:
            ctrl, cdesc, ccol = "LEARNED", "force-guided insertion + when-to-release (PPO policy)", GREEN
        elif HYBRID:
            ctrl, cdesc, ccol = "SCRIPTED", "gripper-open + retract-to-clear  (release was LEARNED)", ORANGE
        else:
            ctrl, cdesc, ccol = "LEARNED", "released — bottle placed  (gripper-open is actuated)", GREEN
    else:
        st  = "SEATED" if succ else ("BREAK" if broke else "inserting")
        title = "FORGE+ learned insertion"
        ctrl, cdesc, ccol = ("SCRIPTED", "approach positioning", ORANGE) if setup_active else \
                            ("LEARNED", "force-guided insertion (PPO policy)", GREEN)
    # ── top banner ──
    dr.rectangle([0, 0, W, 92], fill=(0, 0, 0, 130))
    dr.text((20, 8),  "%s   step %3d/%d" % (title, k+1, N), font=F_TITLE, fill=(255, 255, 255))
    dr.text((20, 38), "state: %s" % st, font=F_MED, fill=(255, 235, 150))
    dr.text((20, 62), "CONTROL:", font=F_MED, fill=(210, 210, 210))
    dr.text((112, 62), ctrl, font=F_MED, fill=ccol)
    _cx = 112 + (96 if ctrl.startswith("SCRIPTED") else 86)
    dr.text((_cx, 62), "— %s" % cdesc, font=F_MED, fill=(220, 220, 220))
    # legend pill
    dr.text((W-250, 10), "LEARNED", font=F_SM, fill=GREEN)
    dr.text((W-250, 28), "SCRIPTED", font=F_SM, fill=ORANGE)
    # ── force gauge ──
    dr.rectangle([GX-10, GY-26, GX+GW+150, GY+GH+12], fill=(0, 0, 0, 140))
    dr.text((GX-4, GY-24), "contact force", font=F_SM, fill=(220, 220, 220))
    dr.rectangle([GX, GY, GX+GW, GY+GH], outline=(160, 160, 160), width=1, fill=(35, 35, 35, 200))
    frac   = max(0.0, min(1.0, cf_val / GAUGE_MAX))
    over   = cf_val >= F_CMD
    danger = cf_val >= F_BREAK
    bar_col = (240, 70, 70) if danger else (245, 180, 60) if over else (90, 220, 120)
    dr.rectangle([GX, GY, GX + int(GW*frac), GY+GH], fill=bar_col)
    xc = GX + int(GW * min(1.0, F_CMD / GAUGE_MAX))
    xb = GX + int(GW * min(1.0, F_BREAK / GAUGE_MAX))
    dr.line([xc, GY-4, xc, GY+GH+4], fill=(255, 230, 90), width=2)
    dr.line([xb, GY-4, xb, GY+GH+4], fill=(255, 90, 90), width=2)
    dr.text((xc-10, GY-18), "F_cmd", font=F_SM, fill=(255, 230, 90))
    dr.text((xb-10, GY+GH+0), "F_brk", font=F_SM, fill=(255, 120, 120))
    dr.text((GX+GW+12, GY+1), "%5.1f N" % cf_val, font=F_BIG, fill=bar_col)
    img.save(os.path.join(FRAMEDIR, "f_%04d.png" % saved))
    saved += 1
    if k % 20 == 0:
        print("step %d saved=%d cf=%.2f st=%s t=%.1fs" % (k, saved, cf_val, st, _time.time()-t0), flush=True)
    # End the clip shortly after the bottle is PLACED so the static hold doesn't drag out.
    if RELEASE and succ and placed_at is None:
        placed_at = saved
        print("PLACED at frame %d" % placed_at, flush=True)
    if placed_at is not None and (saved - placed_at) >= TAIL:
        print("RESULT PLACED saved=%d (ended %d frames after placement)" % (saved, TAIL), flush=True)
        break
else:
    print("RESULT %s saved=%d" % ("PLACED" if placed_at is not None else "NOPLACE", saved), flush=True)

print("SAVED_FRAMES %d" % saved, flush=True)
env.close(); app.close()

if saved >= 10:
    try:
        import imageio_ffmpeg
        _ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        _ff = "ffmpeg"
    ret = subprocess.run([_ff,"-y","-framerate","24",
                          "-i", os.path.join(FRAMEDIR,"f_%04d.png"),
                          "-c:v","libx264","-pix_fmt","yuv420p","-crf","20",OUTPUT],
                         capture_output=True, text=True)
    if ret.returncode == 0:
        print("FFMPEG ok %dKB -> %s" % (os.path.getsize(OUTPUT)//1024, OUTPUT), flush=True)
    else:
        print("ffmpeg failed: " + ret.stderr[-500:], flush=True)
