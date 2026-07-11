#!/usr/bin/env python3
# render_gear_recovery.py — RTX video of the task1 gear insertion + JAM RECOVERY
# episode (issue #26), on the proven render_forge_min harness with
# render_recovery's pause-capture flush. Everything the arm does is the learned
# policy + the recovery loop's discrete maneuvers — nothing is scripted beyond
# the benchmark's own setup positioning (labelled honestly in the HUD).
#
# The episode arc this renders: carry -> descent -> 5 mm in-grip slip (leaves
# the gear tilted 7-10 deg in the pinch) -> contactless hover -> force-signature
# recovery (retract -> regrasp fixes the tilt -> re-approach) -> force-guided
# seat. Loops episodes until one seats (or MAX_EP), keeps frames of the last.
#
#   CKPT=checkpoints/task1_gear_sliprand.pt.it300 \
#   /workspace/.venv/bin/python scripts/render_gear_recovery.py
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
os.environ.update({"HOME": _PC + "/ovhome", "MPLBACKEND": "Agg", "DISPLAY": ":99"})
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

import math
import torch, carb
import omni.usd
from pxr import Gf, UsdGeom, UsdLux, Usd as PUsd
import omni.replicator.core as rep
from forge_plus.isaac_gear_env import FrankaGearInsertEnv, GearInsertEnvCfg
from forge_plus.skills.policy_network import ForceConditionedPolicy, PolicyConfig
from forge_plus.llm.recovery_selector import RecoverySelector
from forge_plus.llm.client import HeuristicLLMClient
print("imports ok", flush=True)

S = carb.settings.get_settings()
for _k in ["/rtx/reflections/enabled", "/rtx/translucency/enabled",
           "/rtx/indirectDiffuse/enabled", "/rtx/ambientOcclusion/enabled",
           "/rtx/directLighting/sampledLighting/enabled"]:
    S.set(_k, False)

cfg = GearInsertEnvCfg()
cfg.scene.num_envs = 1
cfg.forge_mode = True
cfg.forge_obj_cls = 0
cfg.forge_no_term = True          # we detect the seat ourselves + tail frames
cfg.forge_start_fixed_x = 0.0     # centered start (the jam eval staging)
cfg.slip_disturb_mm = float(os.environ.get("SLIP_MM", "5"))

from isaaclab.envs import DirectRLEnv as _DRL
FrankaGearInsertEnv.render = _DRL.render
env = FrankaGearInsertEnv(cfg)
print("env built", flush=True)

CKPT = os.environ.get(
    "CKPT", "/workspace/FORGE-plus_task3/checkpoints/task1_gear_sliprand.pt.it300")
ckpt = torch.load(CKPT, map_location=env.device, weights_only=False)
pc = ckpt["policy_cfg"]
pcfg = pc if isinstance(pc, PolicyConfig) else PolicyConfig(**pc)
policy = ForceConditionedPolicy(pcfg).to(env.device)
policy.load_state_dict(ckpt["policy_state_dict"])
policy.eval()
selector = RecoverySelector(client=HeuristicLLMClient())
print("policy loaded", flush=True)

out = env.reset()
obs = (out[0] if isinstance(out, tuple) else out)["policy"]
orig = env.scene.env_origins[0].cpu().numpy()
stage = omni.usd.get_context().get_stage()

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

gp = UsdGeom.Mesh.Define(stage, "/World/Ground")
_S = 8.0; gz = float(orig[2]) + 0.001
gp.CreatePointsAttr([(-_S, -_S, gz), (_S, -_S, gz), (_S, _S, gz), (-_S, _S, gz)])
gp.CreateFaceVertexCountsAttr([4]); gp.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
gp.CreateNormalsAttr([(0, 0, 1)] * 4)
gp.CreateDisplayColorAttr([(0.32, 0.30, 0.27)])

# tighter 3/4 framing than forge_min: arm + plate + gear readable
cam = UsdGeom.Camera.Define(stage, "/World/EvalCam")
cam.CreateFocalLengthAttr(30.0)
eye = Gf.Vec3d(float(orig[0]) + 1.05, float(orig[1]) - 0.68, float(orig[2]) + 0.85)
tgt = Gf.Vec3d(float(orig[0]) + 0.42, float(orig[1]) + 0.11, float(orig[2]) + 0.44)
up = Gf.Vec3d(0, 0, 1)
fwd = (tgt - eye).GetNormalized()
rgt = Gf.Cross(fwd, up).GetNormalized()
tup = Gf.Cross(rgt, fwd).GetNormalized()
M = Gf.Matrix4d(rgt[0], rgt[1], rgt[2], 0, tup[0], tup[1], tup[2], 0,
                -fwd[0], -fwd[1], -fwd[2], 0, eye[0], eye[1], eye[2], 1)
UsdGeom.Xformable(cam).AddTransformOp().Set(M)
print("scene dressed", flush=True)

for _ in range(110): app.update()
rp = rep.create.render_product("/World/EvalCam", (960, 540))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])
try:
    from omni.replicator.core.scripts.utils import annotator_utils as _au
    _orig_fn = _au._resize_data_for_overscan
    def _safe(d, p):
        if not p or p.get("datawindow_overscan_z") is None: return d
        return _orig_fn(d, p)
    _au._resize_data_for_overscan = _safe
except Exception as ex:
    print("overscan skip: " + str(ex), flush=True)
for _ in range(110): app.update()
print("render product ready shape=%s" % str(np.asarray(rgb.get_data()).shape), flush=True)

# ── pause-capture flush (render_recovery pattern): freeze physics for the
# capture, write gear + every robot link world pose into Fabric.
PAUSE_CAP = os.environ.get("PAUSE_CAP", "1") == "1"
_rtf = {"xfs": None}
def _rt_flush():
    from usdrt import Usd as RtUsd, Gf as RtGf, Rt
    if _rtf["xfs"] is None:
        _rts = RtUsd.Stage.Attach(omni.usd.get_context().get_stage_id())
        gear_path = env._obj.cfg.prim_path.replace("env_.*", "env_0")
        xfs = [(Rt.Xformable(_rts.GetPrimAtPath(gear_path)), None)]
        u_root = stage.GetPrimAtPath(gear_path)
        for child in PUsd.PrimRange(u_root):
            cp_ = str(child.GetPath())
            if cp_ != gear_path and child.GetTypeName() == "Mesh":
                pr = _rts.GetPrimAtPath(cp_)
                if pr.IsValid():
                    xfs.append((Rt.Xformable(pr), None))
        body_names = list(env._robot.data.body_names)
        for lp_ in env._robot.root_physx_view.link_paths[0]:
            prim = _rts.GetPrimAtPath(lp_)
            tail = lp_.rsplit("/", 1)[-1]
            if prim.IsValid() and tail in body_names:
                xfs.append((Rt.Xformable(prim), body_names.index(tail)))
        _rtf["xfs"] = xfs
        print("flush set: gear(+meshes) + %d links"
              % sum(1 for _, li in xfs if li is not None), flush=True)
    d = env._robot.data
    lp = getattr(d, "body_link_pos_w", None)
    lq = getattr(d, "body_link_quat_w", None)
    if lp is None:
        lp, lq = d.body_pos_w, d.body_quat_w
    bp = env._obj.data.root_pose_w[0].tolist()
    from usdrt import Gf as RtGf
    for xf, li in _rtf["xfs"]:
        p = bp if li is None else (lp[0, li].tolist() + lq[0, li].tolist())
        if not xf.HasWorldXform():
            xf.SetWorldXformFromUsd()
        xf.GetWorldPositionAttr().Set(RtGf.Vec3d(p[0], p[1], p[2]))
        xf.GetWorldOrientationAttr().Set(
            RtGf.Quatf(p[3], RtGf.Vec3f(p[4], p[5], p[6])))

def _grab():
    if PAUSE_CAP:
        S.set_bool("/app/player/playSimulations", False)
        try:
            _rt_flush()
        except Exception as fx:
            print("flush FAILED: %r" % (fx,), flush=True)
    try:
        app.update(); app.update()
        d = np.asarray(rgb.get_data())
    finally:
        if PAUSE_CAP:
            S.set_bool("/app/player/playSimulations", True)
    return d if (d.ndim >= 3 and d.shape[0] > 1 and d.shape[1] > 1) else None

FRAMEDIR = "/workspace/frames_gear_rec"
OUTPUT = os.environ.get(
    "OUT", "/workspace/render_takes/gear_recovery_take_%03d.mp4"
    % int(os.environ.get("TAKE", "1")))
Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
W, H = 960, 540
GX, GY, GW, GH = 24, H - 48, 360, 22
GREEN, ORANGE, CYAN = (120, 255, 120), (255, 170, 60), (110, 210, 255)

MAX_EP = int(os.environ.get("MAX_EP", "4"))
N = int(os.environ.get("MAX_STEPS", "780"))
TAIL = 20
CAP_EVERY = int(os.environ.get("CAP_EVERY", "2"))   # capture every Nth policy step

for ep in range(MAX_EP):
    os.makedirs(FRAMEDIR, exist_ok=True)
    for _f in Path(FRAMEDIR).glob("*.png"): _f.unlink()
    out = env.reset()
    obs = (out[0] if isinstance(out, tuple) else out)["policy"]
    F_CMD = float(env._f_cmd[0].item())
    F_BREAK = float(env._f_break[0].item())
    GAUGE_MAX = max(F_BREAK * 1.15, F_CMD * 1.5, 30.0)
    saved = 0
    seated_at = None
    attempts = 0
    last_rec = ""       # HUD: last recovery action
    slip_seen = False
    broke = False
    t0 = _time.time()
    for k in range(N):
        with torch.no_grad():
            m, _s = policy(obs, env.f_cmd_norm())
        res = env.step(torch.clamp(m, -1, 1))
        obs = res[0]["policy"]

        # recovery loop — identical wiring to eval_gear_jam
        if attempts < 5 and env.is_failure():
            sig = env.failure_signature()
            resp = selector.select(sig, env.f_max_n, attempts + 1, "insert", env.gripper)
            env.apply_recovery(resp.action, dict(resp.params or {}))
            attempts += 1
            kind = ("hover" if sig.peak_axial_N < 1.0
                    else "wedge" if sig.lateral_bias != "none" else "friction")
            last_rec = "%s -> %s" % (kind, resp.action)
            print("[rec] ep%d k%d attempt%d %s" % (ep, k, attempts, last_rec), flush=True)

        cf = float(env._cf_insert[0].item())
        g = env._obj.data.root_pose_w[0, :3] - env.scene.env_origins[0]
        qw, qx, qy, qz = env._obj.data.root_pose_w[0, 3:7].tolist()
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (qx*qx + qy*qy)))))
        if not slip_seen and hasattr(env, "_slip_done") and bool(env._slip_done[0]):
            slip_seen = True
        succ = bool(env._succeeded[0].item())
        broke = broke or bool(env._broke[0].item())
        setup_active = int(env._setup_ctr[0].item()) > 0
        rec_active = int(env._rec_steps[0].item()) > 0

        if k % CAP_EVERY == 0 or succ or broke:
            data = _grab()
            if data is not None:
                img = Image.fromarray(data[:, :, :3]).convert("RGB")
                dr = ImageDraw.Draw(img, "RGBA")
                st = ("SEATED" if succ else "BREAK" if broke
                      else "recovering" if (rec_active or (attempts and cf < 1.0))
                      else "inserting")
                dr.rectangle([0, 0, W, 92], fill=(0, 0, 0, 130))
                dr.text((20, 8), "FORGE+  task1 gear insertion + jam recovery   step %3d" % k,
                        font=F_TITLE, fill=(255, 255, 255))
                dr.text((20, 38), "state: %s%s" % (st, "   [5 mm in-grip SLIP injected]"
                        if slip_seen and not succ else ""), font=F_MED, fill=(255, 235, 150))
                if setup_active:
                    ctrl, cdesc, ccol = "SCRIPTED", "approach positioning", ORANGE
                elif rec_active:
                    ctrl, cdesc, ccol = "RECOVERY", last_rec, CYAN
                else:
                    ctrl, cdesc, ccol = "LEARNED", "force-guided insertion (PPO policy)", GREEN
                dr.text((20, 62), "CONTROL:", font=F_MED, fill=(210, 210, 210))
                dr.text((112, 62), ctrl, font=F_MED, fill=ccol)
                dr.text((112 + 92, 62), "— %s" % cdesc, font=F_MED, fill=(220, 220, 220))
                if last_rec:
                    dr.text((W - 330, 10), "recovery: %s (attempt %d)" % (last_rec, attempts),
                            font=F_SM, fill=CYAN)
                dr.text((W - 330, 28), "gear tilt: %4.1f deg" % tilt, font=F_SM,
                        fill=(255, 120, 120) if tilt > 5 else (200, 255, 200))
                dr.rectangle([GX-10, GY-26, GX+GW+150, GY+GH+12], fill=(0, 0, 0, 140))
                dr.text((GX-4, GY-24), "insertion contact force", font=F_SM, fill=(220, 220, 220))
                dr.rectangle([GX, GY, GX+GW, GY+GH], outline=(160, 160, 160), width=1,
                             fill=(35, 35, 35, 200))
                frac = max(0.0, min(1.0, cf / GAUGE_MAX))
                bar = ((240, 70, 70) if cf >= F_BREAK else
                       (245, 180, 60) if cf >= F_CMD else (90, 220, 120))
                dr.rectangle([GX, GY, GX + int(GW*frac), GY+GH], fill=bar)
                xc = GX + int(GW * min(1.0, F_CMD / GAUGE_MAX))
                xb = GX + int(GW * min(1.0, F_BREAK / GAUGE_MAX))
                dr.line([xc, GY-4, xc, GY+GH+4], fill=(255, 230, 90), width=2)
                dr.line([xb, GY-4, xb, GY+GH+4], fill=(255, 90, 90), width=2)
                dr.text((xc-10, GY-18), "F_cmd", font=F_SM, fill=(255, 230, 90))
                dr.text((xb-10, GY+GH), "F_brk", font=F_SM, fill=(255, 120, 120))
                dr.text((GX+GW+12, GY+1), "%5.1f N" % cf, font=F_BIG, fill=bar)
                img.save(os.path.join(FRAMEDIR, "f_%04d.png" % saved))
                saved += 1
        if k % 50 == 0:
            print("ep%d k%d saved=%d cf=%.1f tilt=%.1f gear_z=%.3f att=%d t=%.0fs"
                  % (ep, k, saved, cf, tilt, float(g[2]), attempts, _time.time()-t0),
                  flush=True)
        if succ and seated_at is None:
            seated_at = saved
            print("SEATED ep%d k%d frame %d" % (ep, k, seated_at), flush=True)
        if broke:
            print("BREAK ep%d k%d" % (ep, k), flush=True)
            break
        if seated_at is not None and (saved - seated_at) >= TAIL:
            break
    if seated_at is not None:
        print("TAKE GOOD: ep%d frames=%d" % (ep, saved), flush=True)
        break
    print("ep%d no seat (broke=%s) — reroll" % (ep, broke), flush=True)
else:
    print("NO GOOD TAKE in %d episodes" % MAX_EP, flush=True)

# encode BEFORE app.close (SimulationApp.close hard-exits)
if saved >= 10 and seated_at is not None:
    try:
        import imageio_ffmpeg
        _ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        _ff = "ffmpeg"
    ret = subprocess.run([_ff, "-y", "-framerate", "24",
                          "-i", os.path.join(FRAMEDIR, "f_%04d.png"),
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", OUTPUT],
                         capture_output=True, text=True)
    if ret.returncode == 0:
        print("FFMPEG ok %dKB -> %s" % (os.path.getsize(OUTPUT)//1024, OUTPUT), flush=True)
    else:
        print("ffmpeg failed: " + ret.stderr[-400:], flush=True)
print("RENDER_DONE", flush=True)
os._exit(0)
