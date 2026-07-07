#!/usr/bin/env python3
# probe_pausecap.py — decisive probe for the pause-capture visual-sync bug
# (take 99: physics seated the bottle but the RENDERED bottle stayed frozen
# at its early-carry pose — the arm articulation synced, the rigid-object
# bottle did not, once /app/player/playSimulations=False during capture).
#
# Protocol: boot the exact render_recovery RTX pipeline, drop the bottle
# 30 cm, then for each env.step capture one frame with playSimulations
# paused. Phase A (steps 0-5): capture with NO flush (expect: bottle pixels
# frozen — reproduces the bug). Phase B (steps 6-11): call the PhysX
# update_transformations flush before rendering (expect: pixels track).
# The bottle is dark green — a green-dominant pixel-centroid tracks its
# rendered height; the physics z comes from env._obj.data.root_pose_w.
import os, sys
import numpy as np

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

import torch, carb
import omni.usd
from pxr import Gf, UsdGeom, UsdLux
import omni.replicator.core as rep
from forge_plus.isaac_pick_place_env import FrankaPickPlaceEnv, PickPlaceEnvCfg
print("imports ok", flush=True)

S = carb.settings.get_settings()

cfg = PickPlaceEnvCfg()
cfg.scene.num_envs = 1
cfg.place_strategy = "insert"
cfg.forge_mode = True
cfg.episode_length_s = 360.0
cfg.forge_no_term = True
# v5: mirror the take-100 ROBOTIQ cfg — probe v4 (franka defaults) validated
# the usdrt flush, but under this cfg the same flush VANISHED the bottle
# visual entirely (take 100). Suspects: replicate_physics=False fabric
# hierarchy (HasWorldXform already true -> stale local xform composed in?)
# or a lost world scale.
GRIPPER = os.environ.get("GRIPPER", "robotiq_2f140")
cfg.gripper = GRIPPER
cfg.forge_release_mode = True
cfg.grasp_topdown = False
cfg.forge_obj_cls = 0
if GRIPPER == "robotiq_2f140":
    cfg.forge_setup_steps = 4000
    cfg.warmup_substeps = 100
    cfg.scene.replicate_physics = False

from isaaclab.envs import DirectRLEnv as _DRL
FrankaPickPlaceEnv.render = _DRL.render
env = FrankaPickPlaceEnv(cfg)
print("env built", flush=True)
env.reset()
orig = env.scene.env_origins[0].cpu().numpy()

stage = omni.usd.get_context().get_stage()
dome = UsdLux.DomeLight.Define(stage, "/World/SkyDome")
dome.CreateIntensityAttr(750.0)
cam = UsdGeom.Camera.Define(stage, "/World/EvalCam")
cam.CreateFocalLengthAttr(27.0)
eye = Gf.Vec3d(float(orig[0]) + 1.55, float(orig[1]) - 1.35, float(orig[2]) + 1.15)
tgt = Gf.Vec3d(float(orig[0]) + 0.18, float(orig[1]) + 0.12, float(orig[2]) + 0.42)
up = Gf.Vec3d(0, 0, 1)
fwd = (tgt - eye).GetNormalized()
rgt = Gf.Cross(fwd, up).GetNormalized()
tup = Gf.Cross(rgt, fwd).GetNormalized()
M = Gf.Matrix4d(rgt[0], rgt[1], rgt[2], 0, tup[0], tup[1], tup[2], 0,
                -fwd[0], -fwd[1], -fwd[2], 0, eye[0], eye[1], eye[2], 1)
UsdGeom.Xformable(cam).AddTransformOp().Set(M)

for _ in range(110): app.update()
rp = rep.create.render_product("/World/EvalCam", (960, 540))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])
for _ in range(110): app.update()
print("render product ready", flush=True)

# ── PhysX flush candidates (API name/signature varies by version) ──────────
import omni.physx as _ophx
_pi = _ophx.get_physx_interface()
print("physx iface methods with 'transform'/'update':",
      [m for m in dir(_pi) if "ransf" in m or "pdate" in m], flush=True)

def _flush_ut():
    for args in ((True, True), (True, True, False)):
        try:
            _pi.update_transformations(*args)
            return "ut" + str(args)
        except Exception:
            continue
    return "UT_FAIL"

_rtxf = None
_diag = {"done": False}
def _flush_usdrt(fix_scale=False):
    # Direct Fabric write of the bottle's world pose (the Isaac "teleport
    # visual in fabric" pattern) — bypasses every physx sync path.
    global _rtxf
    try:
        from usdrt import Usd as RtUsd, Gf as RtGf, Rt
        _path = env._obj.cfg.prim_path.replace("env_.*", "env_0")
        if _rtxf is None:
            _rtstage = RtUsd.Stage.Attach(omni.usd.get_context().get_stage_id())
            _rtprim = _rtstage.GetPrimAtPath(_path)
            if not _rtprim.IsValid():
                return "USDRT_BADPATH:" + _path
            print("DIAG path=%s HasWorldXform(pre)=%s" %
                  (_path, Rt.Xformable(_rtprim).HasWorldXform()), flush=True)
            _rtxf = Rt.Xformable(_rtprim)
        p = env._obj.data.root_pose_w[0].tolist()
        if not _rtxf.HasWorldXform():
            _rtxf.SetWorldXformFromUsd()
        _rtxf.GetWorldPositionAttr().Set(RtGf.Vec3d(p[0], p[1], p[2]))
        _rtxf.GetWorldOrientationAttr().Set(
            RtGf.Quatf(p[3], RtGf.Vec3f(p[4], p[5], p[6])))
        if fix_scale:
            from pxr import UsdGeom as PUG, Gf as PGf, Usd as PUsd
            _uprim = stage.GetPrimAtPath(_path)
            _M = PUG.Xformable(_uprim).ComputeLocalToWorldTransform(
                PUsd.TimeCode.Default())
            _s = PGf.Transform(_M).GetScale()
            _rtxf.GetWorldScaleAttr().Set(RtGf.Vec3d(_s[0], _s[1], _s[2]))
        if not _diag["done"]:
            _diag["done"] = True
            print("DIAG write=%s readback pos=%s scale=%s" %
                  (["%.3f" % v for v in p[:3]],
                   _rtxf.GetWorldPositionAttr().Get(),
                   _rtxf.GetWorldScaleAttr().Get()), flush=True)
        return "usdrt" + ("+scale" if fix_scale else "")
    except Exception as ex:
        return "USDRT_FAIL:" + repr(ex)[:80]

# ── Drop the bottle 40 cm above its parked pose. v2: raw env.sim.step()
# only — env.step()'s phase machine re-parks the object during setup and
# swallowed the v1 toss (phys_z ROSE; no free fall; probe inconclusive).
rs = env._obj.data.root_state_w.clone()
rs[0, 2] += 0.40
rs[0, 7:] = 0.0
env._obj.write_root_pose_to_sim(rs[:, :7])
env._obj.write_root_velocity_to_sim(rs[:, 7:])
print("bottle raised 0.40 m; free-falling via sim.step", flush=True)

def _green_centroid(img):
    a = img[:, :, :3].astype(np.int16)
    m = (a[:, :, 1] > a[:, :, 0] + 18) & (a[:, :, 1] > a[:, :, 2] + 18) & (a[:, :, 1] > 50)
    ys, xs = np.nonzero(m)
    if len(ys) < 20:
        return None, 0
    return float(ys.mean()), len(ys)

def _toss():
    # v4: toss into the CAMERA-VISIBLE band (v3 tossed to z=1.1, out of
    # frame — flushed poses were invisible, indistinguishable from frozen).
    # P0 calibration: bz 0.74 -> cy 41, bz 0.485 -> cy 193, bz 0.402 -> cy 237.
    _rs = env._obj.data.root_state_w.clone()
    _rs[0, 2] = env.scene.env_origins[0, 2] + 0.74
    _rs[0, 7:] = 0.0
    env._obj.write_root_pose_to_sim(_rs[:, :7])
    env._obj.write_root_velocity_to_sim(_rs[:, 7:])

# Phases of 6 frames each, bottle re-tossed into view at each phase start:
#   P0 UNPAUSED capture         -> control: detection + phys->pixel calibration
#   P1 paused + ut(True,True)   -> fabric fast-cache flush candidate
#   P2 paused + usdrt write     -> direct fabric world-pose write
#   P3 paused + BOTH
#   P4 paused, no flush         -> control: frozen expected
for i in range(30):
    if i % 6 == 0:
        _toss()
    for _ in range(4):
        env.sim.step(render=False)
    env._obj.update(env.physics_dt)
    bz = float(env._obj.data.root_pose_w[0, 2] - env.scene.env_origins[0, 2])
    ph = i // 6
    if ph == 0:
        mode = "UNPAUSED"
        app.update(); app.update()
        d = np.asarray(rgb.get_data())
    else:
        S.set_bool("/app/player/playSimulations", False)
        mode = {1: _flush_usdrt,
                2: lambda: _flush_usdrt(fix_scale=True),
                3: _flush_ut,
                4: lambda: "noflush"}[ph]()
        app.update(); app.update()
        d = np.asarray(rgb.get_data())
        S.set_bool("/app/player/playSimulations", True)
    cy, npx = _green_centroid(d)
    print("PROBE i=%02d ph=%d phys_z=%.3f pix_cy=%s npx=%d mode=%s"
          % (i, ph, bz, "%.1f" % cy if cy is not None else "none", npx, mode),
          flush=True)

print("PROBE DONE", flush=True)
app.close()
