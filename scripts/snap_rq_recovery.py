#!/usr/bin/env python3
"""RTX snapshots of the 2F-140 jam-RECOVERY episode (slip -> jam -> regrasp).

Drives the env exactly like eval_gear_jam (uni policy deterministic mean,
obj 0, 5 mm in-grip slip, recovery=ours) and captures stills at the recovery
beats: post-slip tilt, retract lift, pend traverse, pinned-open seat window,
squeeze, post-seat grip, arrival hand-off, end state. Same proven harness as
snap_rq_gear.py (proxy-gear ghost workaround; pad_dz logged per the
grasp-realism rule — gear design pad_dz = grip_h = +0.030).

    /workspace/.venv/bin/python scripts/snap_rq_recovery.py
"""
import os, sys
os.environ.setdefault("HOME", "/workspace/persist/ovhome")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("DISPLAY", ":99")
sys.path.insert(0, "/workspace/FORGE-plus_task3")

_EXTRA = [
    "--/exts/isaacsim.core.throttling/enable_async=false",
    "--/rtx/raytracing/subsurface/enabled=false",
    "--/rtx/reflections/enabled=false",
    "--/rtx/translucency/enabled=false",
    "--/rtx/directLighting/sampledLighting/enabled=false",
    "--/rtx/indirectDiffuse/enabled=false",
    "--/rtx/ambientOcclusion/enabled=false",
    "--/rtx/post/aa/op=1",
]
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 960, "height": 720,
                     "extra_args": _EXTRA})
print("booted", flush=True)

import numpy as np
import torch
import carb
import omni.usd
from pxr import Gf, UsdGeom, UsdLux
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
cfg.scene.replicate_physics = False
cfg.gripper = "robotiq_2f140"
cfg.forge_mode = True
cfg.forge_obj_cls = 0
cfg.forge_no_term = True
cfg.forge_setup_steps = 4000
cfg.warmup_substeps = 100
cfg.episode_length_s = 60.0
cfg.slip_disturb_mm = 5.0
env = FrankaGearInsertEnv(cfg)
out = env.reset()
obs = (out[0] if isinstance(out, tuple) else out)["policy"]
print("env up", flush=True)

ck = torch.load("checkpoints/task1_gear_rq_uni.pt", map_location=env.device,
                weights_only=False)
pc = ck["policy_cfg"]
pcfg = pc if isinstance(pc, PolicyConfig) else PolicyConfig(**pc)
policy = ForceConditionedPolicy(pcfg).to(env.device)
policy.load_state_dict(ck["policy_state_dict"])
policy.eval()
selector = RecoverySelector(client=HeuristicLLMClient())

stage = omni.usd.get_context().get_stage()
orig = env.scene.env_origins[0].tolist()

lux = UsdLux.DistantLight.Define(stage, "/World/SnapSun")
lux.CreateIntensityAttr(3200.0)
UsdGeom.Xformable(lux).AddRotateXYZOp().Set(Gf.Vec3f(-32.0, 18.0, 0.0))
dome = UsdLux.DomeLight.Define(stage, "/World/SnapDome")
dome.CreateIntensityAttr(420.0)

UsdGeom.Imageable(stage.GetPrimAtPath("/World/envs/env_0/Object")).MakeInvisible()
px = UsdGeom.Xform.Define(stage, "/World/GearProxy")
_proxy_xf = UsdGeom.Xformable(px).AddTransformOp()
teeth = UsdGeom.Cylinder.Define(stage, "/World/GearProxy/teeth")
teeth.CreateRadiusAttr(0.02095); teeth.CreateHeightAttr(0.010); teeth.CreateAxisAttr("Z")
UsdGeom.Xformable(teeth).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.010))
teeth.CreateDisplayColorAttr([(0.38, 0.40, 0.46)])
hub = UsdGeom.Cylinder.Define(stage, "/World/GearProxy/hub")
hub.CreateRadiusAttr(0.01775); hub.CreateHeightAttr(0.030); hub.CreateAxisAttr("Z")
UsdGeom.Xformable(hub).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.030))
hub.CreateDisplayColorAttr([(0.50, 0.52, 0.58)])
bore = UsdGeom.Cylinder.Define(stage, "/World/GearProxy/bore")
bore.CreateRadiusAttr(0.0053); bore.CreateHeightAttr(0.0406); bore.CreateAxisAttr("Z")
UsdGeom.Xformable(bore).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.0253))
bore.CreateDisplayColorAttr([(0.10, 0.10, 0.12)])

def _proxy_pose():
    p = env._obj.data.root_pose_w[0].tolist()
    Mp = Gf.Matrix4d(1.0)
    Mp.SetRotateOnly(Gf.Rotation(Gf.Quatd(p[3], p[4], p[5], p[6])))
    Mp.SetTranslateOnly(Gf.Vec3d(p[0], p[1], p[2]))
    _proxy_xf.Set(Mp)

def _make_cam(path, eye, tgt):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(35.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
    up = Gf.Vec3d(0, 0, 1)
    fwd = (tgt - eye).GetNormalized()
    rgt = Gf.Cross(fwd, up).GetNormalized()
    tup = Gf.Cross(rgt, fwd).GetNormalized()
    M = Gf.Matrix4d(rgt[0], rgt[1], rgt[2], 0, tup[0], tup[1], tup[2], 0,
                    -fwd[0], -fwd[1], -fwd[2], 0, eye[0], eye[1], eye[2], 1)
    UsdGeom.Xformable(cam).AddTransformOp().Set(M)

_make_cam("/World/SnapCam",
          Gf.Vec3d(orig[0] + 1.05, orig[1] + 0.10, orig[2] + 0.62),
          Gf.Vec3d(orig[0] + 0.33, orig[1] + 0.06, orig[2] + 0.50))

for _ in range(110):
    app.update()
rp = rep.create.render_product("/World/SnapCam", (960, 720))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])
try:
    from omni.replicator.core.scripts.utils import annotator_utils as _au
    _orig_fn = _au._resize_data_for_overscan
    def _safe(d, p):
        if not p or p.get("datawindow_overscan_z") is None:
            return d
        return _orig_fn(d, p)
    _au._resize_data_for_overscan = _safe
except Exception as ex:
    print("overscan skip: " + str(ex), flush=True)
for _ in range(110):
    app.update()
print("render product ready shape=%s" % str(np.asarray(rgb.get_data()).shape),
      flush=True)

SNAPDIR = "/workspace/frames_rq_recovery_snap"
os.makedirs(SNAPDIR, exist_ok=True)
import imageio.v2 as imageio
import math


def gear_tilt_deg():
    qw, qx, qy, qz = env._obj.data.root_pose_w[0, 3:7].tolist()
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (qx * qx + qy * qy)))))


def snap(name, step):
    _proxy_pose()
    for _ in range(6):
        app.update()
    img = np.asarray(rgb.get_data())
    if img.size and img.ndim == 3:
        fn = f"{SNAPDIR}/s{step:04d}_{name}.png"
        imageio.imwrite(fn, img[..., :3])
        from isaaclab.utils.math import matrix_from_quat
        eep = env._robot.data.body_pos_w[0, env._ee_idx]
        eeq = env._robot.data.body_quat_w[0, env._ee_idx]
        ax = matrix_from_quat(eeq.unsqueeze(0))[0, :, 2]
        gz = float(env._obj.data.root_pose_w[0, 2])
        g = env._obj.data.root_pose_w[0, :3] - env.scene.env_origins[0]
        pad_dz = float(eep[2] + float(env._grasp_tcp_d) * ax[2]) - gz
        print(f"[snap] {name} s{step}: pad_dz={pad_dz:+.4f} (design +0.030) "
              f"gear=({float(g[0]):.3f},{float(g[1]):.3f},{float(g[2]):.3f}) "
              f"tilt={gear_tilt_deg():.1f} "
              f"Fins={float(env._cf_insert[0]):.2f} "
              f"grip={float(env._robot.data.joint_pos[0, env._grip_ids[0]]):.3f}",
              flush=True)
    else:
        print(f"[snap] {name}: EMPTY frame", flush=True)


seen = set()


def once(name, step):
    if name not in seen:
        seen.add(name)
        snap(name, step)


attempts = 0
prev_seat = 0
post_seat_ctr = -1
post_hand_ctr = -1
handoffs = 0
for step in range(1400):
    with torch.no_grad():
        mean, _ = policy(obs, env.f_cmd_norm())
        act = mean.clamp(-1.0, 1.0)
    res = env.step(act)
    obs = res[0]["policy"]
    tilt = gear_tilt_deg()
    setup = int(env._setup_ctr[0])
    seat = int(env._rq_seatctr[0])

    if setup == 0 and handoffs == 0 and step > 100:
        handoffs = 1
        once("clean_handoff", step)
    if tilt > 12.0 and "post_slip_tilt" not in seen:
        once("post_slip_tilt", step)
    if attempts < 5 and env.is_failure():
        resp = selector.select(env.failure_signature(), env.f_max_n,
                               attempts + 1, "insert", env.gripper)
        print(f"[rec] attempt{attempts + 1} -> {resp.action}", flush=True)
        env.apply_recovery(resp.action, dict(resp.params or {}))
        attempts += 1
        if attempts == 1:
            once("first_jam_maneuver", step)
    if bool(env._rq_regrasp_pend[0]) and int(env._rec_steps[0]) == 0:
        once("pend_traverse", step)
    if seat > prev_seat and prev_seat == 0:      # deferred seat fired
        once("seat_fired_pin", step)
    if 0 < seat <= 60:
        once("seat_squeeze_end", step)
    elif 170 < seat <= 240:
        once("seat_kiss", step)
    elif 260 < seat <= 330:
        once("seat_open_pinned", step)
    if prev_seat > 0 and seat == 0:
        post_seat_ctr = 3
    prev_seat = seat
    if post_seat_ctr > 0:
        post_seat_ctr -= 1
        if post_seat_ctr == 0:
            once("post_seat_grip", step)
            post_hand_ctr = 60
    if post_hand_ctr > 0:
        post_hand_ctr -= 1
        if post_hand_ctr == 0:
            once("post_seat_plus60", step)
    if bool(env._succeeded[0]):
        once("SEATED", step)
        break
    if bool(env._broke[0]):
        once("BROKE", step)
        break
snap("end_state", step)
print("SNAP_DONE", flush=True)
os._exit(0)
