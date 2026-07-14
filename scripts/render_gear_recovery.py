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
GRIPPER = os.environ.get("GRIPPER", "franka_panda")
cfg.gripper = GRIPPER
RELEASE = os.environ.get("RELEASE", "0") == "1"
TABLE = os.environ.get("TABLE", "0") == "1"
if GRIPPER == "robotiq_2f140":
    # same staging as eval_gear_jam: entrance grasp (seat-at-B), RAW parse
    # for the four-bar, generous setup window ended by arrival fast-forward
    cfg.forge_setup_steps = 4000
    cfg.warmup_substeps = 100
    cfg.scene.replicate_physics = False
    cfg.episode_length_s = 45.0
    if TABLE:
        cfg.rq_table_pick = True          # unpinned grasp from the table +
        cfg.episode_length_s = 120.0      # place-on-table recovery regrasp
if RELEASE:
    cfg.forge_release_mode = True         # LEARNED release (8-dim ckpt) ends
    cfg.forge_hybrid_retract = True       # the episode: open + retract clear

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
if GRIPPER == "robotiq_2f140":
    # the 2F-140 tower is ~2x the panda hand and its open fingers span
    # 140 mm — the panda framing crops it at the top. Wider lens, pulled
    # back and raised, aim lifted to mid-gripper. Table mode: aim midway
    # between the pick spot (0.46,-0.04) and the shaft (0.45,0.12) so the
    # table grab and the insertion both stay in frame.
    cam.CreateFocalLengthAttr(26.0)
    eye = Gf.Vec3d(float(orig[0]) + 1.24, float(orig[1]) - 0.80,
                   float(orig[2]) + 1.00)
    tgt = Gf.Vec3d(float(orig[0]) + 0.42,
                   float(orig[1]) + (0.04 if TABLE else 0.10),
                   float(orig[2]) + 0.54)
up = Gf.Vec3d(0, 0, 1)
fwd = (tgt - eye).GetNormalized()
rgt = Gf.Cross(fwd, up).GetNormalized()
tup = Gf.Cross(rgt, fwd).GetNormalized()
M = Gf.Matrix4d(rgt[0], rgt[1], rgt[2], 0, tup[0], tup[1], tup[2], 0,
                -fwd[0], -fwd[1], -fwd[2], 0, eye[0], eye[1], eye[2], 1)
UsdGeom.Xformable(cam).AddTransformOp().Set(M)

# ── RENDER-PROXY gear: the physics gear's visual draws DISPLACED on this pod
# (user-visible: shaft through the gear body, gear floating off the pinch —
# the RTX object-sync ghost; physics verified correct via raw-PhysX probes).
# Hide the broken visual and draw a plain USD proxy at the buffer pose every
# capture: non-physics prims render from USD reliably (plate/pegs always did).
PROXY = os.environ.get("PROXY_GEAR", "1") == "1"
_proxy_xf = None
if PROXY:
    _gear_path = env._obj.cfg.prim_path.replace("env_.*", "env_0")
    UsdGeom.Imageable(stage.GetPrimAtPath(_gear_path)).MakeInvisible()
    px = UsdGeom.Xform.Define(stage, "/World/GearProxy")
    _proxy_xf = UsdGeom.Xformable(px).AddTransformOp()
    # exact FORGE gear dims, origin 5 mm below the bottom face
    teeth = UsdGeom.Cylinder.Define(stage, "/World/GearProxy/teeth")
    teeth.CreateRadiusAttr(0.02095); teeth.CreateHeightAttr(0.010)
    teeth.CreateAxisAttr("Z")
    UsdGeom.Xformable(teeth).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.010))
    teeth.CreateDisplayColorAttr([(0.38, 0.40, 0.46)])
    hub = UsdGeom.Cylinder.Define(stage, "/World/GearProxy/hub")
    hub.CreateRadiusAttr(0.01775); hub.CreateHeightAttr(0.030)
    hub.CreateAxisAttr("Z")
    UsdGeom.Xformable(hub).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.030))
    hub.CreateDisplayColorAttr([(0.50, 0.52, 0.58)])
    bore = UsdGeom.Cylinder.Define(stage, "/World/GearProxy/bore")
    bore.CreateRadiusAttr(0.0053); bore.CreateHeightAttr(0.0406)
    bore.CreateAxisAttr("Z")
    UsdGeom.Xformable(bore).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.0253))
    bore.CreateDisplayColorAttr([(0.10, 0.10, 0.12)])
    print("gear proxy active (physics visual hidden)", flush=True)

def _proxy_pose():
    if _proxy_xf is None:
        return
    p = env._obj.data.root_pose_w[0].tolist()
    Mp = Gf.Matrix4d(1.0)
    Mp.SetRotateOnly(Gf.Rotation(Gf.Quatd(p[3], p[4], p[5], p[6])))
    Mp.SetTranslateOnly(Gf.Vec3d(p[0], p[1], p[2]))
    _proxy_xf.Set(Mp)
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
        _proxy_pose()
    except Exception as fx:
        print("proxy pose FAILED: %r" % (fx,), flush=True)
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
TAIL = int(os.environ.get("TAIL", "12" if RELEASE else "8"))
           # non-release: stop right after the seat — forge_no_term keeps
           # pressing and take-1 BROKE the gear 19 steps post-seat. Release
           # mode: success = released + hand clear, so the tail just holds
           # the finished scene.
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

        # recovery loop — identical wiring to eval_gear_jam. NOT run for
        # no-slip takes: the clean protocol (eval_gear_insert) has no
        # detector, and a detector false-positive on a slow clean descent
        # churned regrasp cycles through an otherwise-clean episode
        # (final-take log: att=2 in a SLIP_MM=0 episode).
        if cfg.slip_disturb_mm > 0 and attempts < 5 and env.is_failure():
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
        released = RELEASE and bool(env._released[0].item())
        placing = TABLE and getattr(env, "_rq_pl_phase", 0) > 0
        # capture at half rate through the long quiet staging legs (servo,
        # return traverse); full rate whenever the fingers move, recovery
        # runs, or the policy is live
        closing = int(getattr(env, "_rq_seatctr", torch.zeros(1))[0]) != 0
        _ce = CAP_EVERY * (2 if (setup_active and not closing
                                 and not rec_active and not placing) else 1)

        if k % _ce == 0 or succ or broke:
            data = _grab()
            if data is not None:
                img = Image.fromarray(data[:, :, :3]).convert("RGB")
                dr = ImageDraw.Draw(img, "RGBA")
                st = ("DONE — released, hand clear" if succ
                      else "BREAK" if broke
                      else "RELEASING (learned)" if released
                      else "recovering" if (rec_active or placing
                                            or (attempts and cf < 1.0))
                      else "inserting")
                dr.rectangle([0, 0, W, 92], fill=(0, 0, 0, 130))
                dr.text((20, 8), "FORGE+  task1 gear insertion + jam recovery   step %3d" % k,
                        font=F_TITLE, fill=(255, 255, 255))
                dr.text((20, 38), "state: %s%s" % (st, "   [5 mm in-grip SLIP injected]"
                        if slip_seen and not succ else ""), font=F_MED, fill=(255, 235, 150))
                if placing:
                    ctrl, cdesc, ccol = "RECOVERY", "place on table + re-pick", CYAN
                elif setup_active:
                    ctrl, cdesc, ccol = "SCRIPTED", ("table pick + carry" if TABLE
                                                     else "approach positioning"), ORANGE
                elif rec_active:
                    ctrl, cdesc, ccol = "RECOVERY", last_rec, CYAN
                elif released:
                    ctrl, cdesc, ccol = "LEARNED", "release commanded (act[7])", GREEN
                else:
                    ctrl, cdesc, ccol = "LEARNED", "force-guided insertion (policy)", GREEN
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
            # pad_dz: grasp-realism check (design +0.030 = grip_h; pads must sit
            # on the hub band, never above it) — verify in the log per take
            from isaaclab.utils.math import matrix_from_quat as _mfq
            _eep = env._robot.data.body_pos_w[0, env._ee_idx]
            _ax = _mfq(env._robot.data.body_quat_w[0, env._ee_idx].unsqueeze(0))[0, :, 2]
            _pad_dz = float(_eep[2] + float(env._grasp_tcp_d) * _ax[2]) \
                - float(env._obj.data.root_pose_w[0, 2])
            print("ep%d k%d saved=%d cf=%.1f tilt=%.1f gear_z=%.3f att=%d "
                  "pad_dz=%+.4f t=%.0fs"
                  % (ep, k, saved, cf, tilt, float(g[2]), attempts, _pad_dz,
                     _time.time()-t0), flush=True)
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
