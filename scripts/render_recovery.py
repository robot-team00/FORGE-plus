#!/usr/bin/env python3
# render_recovery.py — RTX render of the force-signature LLM RECOVERY loop driving the
# LEARNED forge policy on the FRAGILE object (glass, F_break ~22 N), with an induced jam.
#
# Built on the proven render_forge_min.py harness (numpy 1.26, GI/DLSS off, clean studio
# scene). The episode is driven end-to-end by RecoveryLoop.run(env, on_step=capture):
#     scripted approach -> LEARNED insertion -> the bottle WEDGES on the cell rim
#     -> jam caught from the FORCE SIGNATURE (well below break) -> LLM picks a recovery
#     -> lift + realign -> the LEARNED policy re-inserts -> seats.
# Nothing here scripts the manipulation: the HUD labels every phase LEARNED vs SCRIPTED,
# and the recovery maneuver is one of the loop's fixed primitives (an honest SCRIPTED label).
#
#   JAM=0.05 OBJ=0 K_MAX=5 /workspace/.venv/bin/python scripts/render_recovery.py
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
from forge_plus.llm.client import HeuristicLLMClient
from forge_plus.llm.recovery_selector import RecoverySelector
from forge_plus.recovery.recovery_loop import RecoveryLoop, RecoveryOutcome
print("imports ok", flush=True)

S = carb.settings.get_settings()
for _k in ["/rtx/reflections/enabled", "/rtx/translucency/enabled",
            "/rtx/indirectDiffuse/enabled", "/rtx/ambientOcclusion/enabled",
            "/rtx/directLighting/sampledLighting/enabled"]:
    S.set(_k, False)

JAM   = float(os.environ.get("JAM", "0.05"))
OBJ   = int(os.environ.get("OBJ", "0"))       # 0 = glass (fragile, break ~22 N)
K_MAX = int(os.environ.get("K_MAX", "5"))
CAP_EVERY = int(os.environ.get("CAP_EVERY", "1"))   # capture every Nth control step

cfg = PickPlaceEnvCfg()
cfg.scene.num_envs   = 1
cfg.gripper          = "franka_panda"
cfg.place_strategy   = "insert"
cfg.settle_steps     = 400
cfg.episode_length_s = 120.0   # the loop owns the timeline — no auto-reset mid-demo
cfg.jam_dx           = JAM
cfg.forge_mode          = True
cfg.forge_release_mode  = True    # the trained policy is 8-dim (arm + release)
cfg.grasp_topdown       = False
cfg.forge_obj_cls       = OBJ
cfg.render_minimal      = bool(int(os.environ.get("RENDER_MINIMAL", "0")))

from isaaclab.envs import DirectRLEnv as _DRL
FrankaPickPlaceEnv.render = _DRL.render
env = FrankaPickPlaceEnv(cfg)
print("env built", flush=True)

CKPT = os.environ.get("CKPT", "/workspace/FORGE-plus_task3/checkpoints/task3_forge_entrance.pt")
ckpt = torch.load(CKPT, map_location=env.device, weights_only=False)
pc = ckpt["policy_cfg"]
pcfg = pc if isinstance(pc, PolicyConfig) else PolicyConfig(**pc)
policy = ForceConditionedPolicy(pcfg).to(env.device)
policy.load_state_dict(ckpt["policy_state_dict"])
policy.eval()
env._skill_policy = policy    # step_skill runs the LEARNED policy
print("policy loaded -> env._skill_policy", flush=True)

out  = env.reset()
orig = env.scene.env_origins[0].cpu().numpy()

stage = omni.usd.get_context().get_stage()

# ── Lighting: sky dome + key sun + warm fill (direct lights work with GI off).
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

# ── Ground plane
gp = UsdGeom.Mesh.Define(stage, "/World/Ground")
_S = 8.0; gz = float(orig[2]) + 0.001
gp.CreatePointsAttr([(-_S, -_S, gz), (_S, -_S, gz), (_S, _S, gz), (-_S, _S, gz)])
gp.CreateFaceVertexCountsAttr([4]); gp.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
gp.CreateNormalsAttr([(0, 0, 1)] * 4)
gp.CreateDisplayColorAttr([(0.32, 0.30, 0.27)])

# ── Camera (same tuned 3/4 framing as render_forge_min)
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
print("scene ready", flush=True)

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

FRAMEDIR = "/workspace/frames_recovery"
os.makedirs(FRAMEDIR, exist_ok=True)
for _f in Path(FRAMEDIR).glob("*.png"): _f.unlink()
OUTPUT = os.environ.get("OUT", "/workspace/FORGE-plus_task3/docs/videos/task3/forge_recovery.mp4")
Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

W, H = 960, 540
GX, GY, GW, GH = 24, H - 48, 360, 22
GREEN  = (120, 255, 120)   # LEARNED
ORANGE = (255, 170,  60)   # SCRIPTED
RED    = (255,  90,  90)
CYAN   = (120, 220, 255)

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

# ── Recovery loop wiring. Wrap the selector + signature so the HUD can show the
# LLM decision and the force signature it was made from (display only — the loop
# behaves identically).
selector = RecoverySelector(client=HeuristicLLMClient())
hud = {"decision": None, "sig": None, "flash_until": -1, "saved": 0,
       "peak_n": 0.0, "attempt": 0, "seated_frame": None}

_orig_select = selector.select
def _select_hud(**kw):
    d = _orig_select(**kw)
    hud["decision"] = d
    # flash "JAM DETECTED" for ~2.3 s of video regardless of capture density
    hud["flash_until"] = hud["saved"] + max(28, 56 // CAP_EVERY)
    return d
selector.select = _select_hud

_orig_sig = env.failure_signature
def _sig_hud():
    s = _orig_sig()
    hud["sig"] = s
    return s
env.failure_signature = _sig_hud

loop = RecoveryLoop(selector=selector, k_max=K_MAX)

# Budget/break markers for the gauge. f_max_n/_f_break are valid after reset_episode
# (the loop resets first); read them lazily on the first captured frame.
gauge = {"f_max": None, "f_brk": None, "max": None}

def _draw_frame(attempt, step):
    # The INSERTION force (bottle<->rack) — the same channel that gates breakage and the
    # FORGE budget freeze, so the gauge, the F_brk marker, and the BREAK state are consistent.
    cf_val = float(env._insertion_force()[0].item())
    hud["peak_n"] = max(hud["peak_n"], cf_val)
    if gauge["f_max"] is None:
        gauge["f_max"] = float(env.f_max_n)
        gauge["f_brk"] = float(env._f_break[0].item())
        gauge["max"]   = max(gauge["f_brk"] * 1.15, gauge["f_max"] * 1.5, 30.0)
        print("gauge: F_max=%.1f F_break=%.1f" % (gauge["f_max"], gauge["f_brk"]), flush=True)

    data = _grab()
    if data is None:
        return False
    img = Image.fromarray(data[:, :, :3]).convert("RGB")
    dr  = ImageDraw.Draw(img, "RGBA")

    setup_active = int(env._setup_ctr[0].item()) > 0
    rec_active   = int(env._rec_steps[0].item()) > 0
    seated       = env.is_success()
    broke        = bool(env._broke[0].item())
    flashing     = hud["saved"] < hud["flash_until"]

    # What drives the robot RIGHT NOW (honest learned-vs-scripted label):
    if seated:
        st, ctrl, cdesc, ccol = "SEATED", "—", "recovery complete — bottle seated in the cell", GREEN
    elif rec_active and hud["decision"] is not None:
        st = "recovering"
        ctrl, ccol = "SCRIPTED", ORANGE
        cdesc = "RECOVERY primitive: %s  (picked by the LLM)" % hud["decision"].action
    elif setup_active:
        st, ctrl, cdesc, ccol = "approaching", "SCRIPTED", "approach positioning (up-over-down)", ORANGE
    else:
        st = "inserting"
        ctrl, cdesc, ccol = "LEARNED", "force-guided insertion (PPO policy)", GREEN
    if broke:
        st = "BREAK"

    # ── top banner ──
    dr.rectangle([0, 0, W, 116], fill=(0, 0, 0, 130))
    dr.text((20, 8), "FORGE+  fragile recovery — glass bottle  (F_break %.0f N)"
            % (gauge["f_brk"] or 22.0), font=F_TITLE, fill=(255, 255, 255))
    dr.text((20, 38), "attempt %d/%d   state: %s" % (attempt + 1, K_MAX, st),
            font=F_MED, fill=(255, 235, 150))
    dr.text((20, 62), "CONTROL:", font=F_MED, fill=(210, 210, 210))
    dr.text((112, 62), ctrl, font=F_MED, fill=ccol)
    _cx = 112 + (96 if ctrl.startswith("SCRIPTED") else 86)
    dr.text((_cx, 62), "— %s" % cdesc, font=F_MED, fill=(220, 220, 220))
    # induced-jam note (only while the misalignment is live)
    if bool(env._jam_on[0].item()) and not setup_active and not seated:
        dr.text((20, 88), "induced misalignment: base-aim +%.0f cm off-center (the demo's seeded fault)"
                % (JAM * 100), font=F_SM, fill=(200, 200, 200))
    # legend
    dr.text((W-250, 10), "LEARNED", font=F_SM, fill=GREEN)
    dr.text((W-250, 28), "SCRIPTED", font=F_SM, fill=ORANGE)

    # ── JAM DETECTED flash + force signature + LLM decision (top-center card) ──
    if flashing and hud["decision"] is not None and hud["sig"] is not None:
        s = hud["sig"]
        card_y = 128
        dr.rectangle([W//2 - 330, card_y, W//2 + 330, card_y + 96], fill=(0, 0, 0, 170))
        dr.text((W//2 - 316, card_y + 6), "JAM DETECTED — from the force signature alone (no vision)",
                font=F_MED, fill=RED)
        dr.text((W//2 - 316, card_y + 30),
                "peak %.1f N   net insert %.1f mm   rising=%s   lateral=%s"
                % (s.peak_axial_N, s.net_insert_mm, s.axial_rising, s.lateral_bias),
                font=F_SM, fill=(230, 230, 230))
        dr.text((W//2 - 316, card_y + 50), "LLM recovery: %s" % hud["decision"].action,
                font=F_MED, fill=CYAN)
        dr.text((W//2 - 316, card_y + 74), "\"%s\"  (same F_max — never press harder)"
                % hud["decision"].rationale, font=F_SM, fill=(200, 220, 235))

    # ── force gauge (F_max = budget marker, F_brk = break marker) ──
    fmx, fbk, gmax = gauge["f_max"], gauge["f_brk"], gauge["max"]
    dr.rectangle([GX-10, GY-26, GX+GW+150, GY+GH+12], fill=(0, 0, 0, 140))
    dr.text((GX-4, GY-24), "contact force   (peak %.1f N — under break)" % hud["peak_n"],
            font=F_SM, fill=(220, 220, 220))
    dr.rectangle([GX, GY, GX+GW, GY+GH], outline=(160, 160, 160), width=1, fill=(35, 35, 35, 200))
    frac   = max(0.0, min(1.0, cf_val / gmax))
    over   = cf_val >= fmx
    danger = cf_val >= fbk
    bar_col = (240, 70, 70) if danger else (245, 180, 60) if over else (90, 220, 120)
    dr.rectangle([GX, GY, GX + int(GW*frac), GY+GH], fill=bar_col)
    xc = GX + int(GW * min(1.0, fmx / gmax))
    xb = GX + int(GW * min(1.0, fbk / gmax))
    dr.line([xc, GY-4, xc, GY+GH+4], fill=(255, 230, 90), width=2)
    dr.line([xb, GY-4, xb, GY+GH+4], fill=(255, 90, 90), width=2)
    dr.text((xc-14, GY-18), "F_max", font=F_SM, fill=(255, 230, 90))
    dr.text((xb-10, GY+GH+0), "F_brk", font=F_SM, fill=(255, 120, 120))
    dr.text((GX+GW+12, GY+1), "%5.1f N" % cf_val, font=F_BIG, fill=bar_col)

    img.save(os.path.join(FRAMEDIR, "f_%04d.png" % hud["saved"]))
    hud["saved"] += 1
    return True

t0 = _time.time()
def _on_step(e, attempt, step):
    hud["attempt"] = attempt
    if step % CAP_EVERY != 0:
        return
    ok = _draw_frame(attempt, step)
    if step % 40 == 0:
        cf = float(e._insertion_force()[0].item())
        print("a%d s%3d saved=%d cf=%5.2f rec=%d %s t=%.0fs"
              % (attempt, step, hud["saved"], cf, int(e._rec_steps[0]),
                 "" if ok else "EMPTY", _time.time() - t0), flush=True)

print("running recovery loop (jam=%.3f obj=%d k_max=%d)..." % (JAM, OBJ, K_MAX), flush=True)
result = loop.run(env, on_step=_on_step)
print("loop done: %s in %d attempt(s)" % (result.outcome.value, result.attempts), flush=True)
for a in result.log:
    print("  attempt %d: %s -> %s" % (a.attempt, a.result, a.recovery_action or "-"), flush=True)

# Tail: hold on the seated bottle for ~1 s of video (render-only updates; physics settles).
if result.outcome == RecoveryOutcome.SUCCESS:
    for _ in range(24 // max(1, CAP_EVERY) * CAP_EVERY):
        _draw_frame(hud["attempt"], 0)

broke = bool(env._broke[0].item())
peak  = hud["peak_n"]
ok = (result.outcome == RecoveryOutcome.SUCCESS) and not broke
print("RESULT %s saved=%d peak=%.1fN break=%s attempts=%d"
      % ("SUCCESS" if ok else "FAIL", hud["saved"], peak, broke, result.attempts), flush=True)

# Encode BEFORE app.close() — SimulationApp.close() hard-exits the process.
if ok and hud["saved"] >= 60:
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
        print("ffmpeg failed: " + ret.stderr[-500:], flush=True)

env.close(); app.close()
