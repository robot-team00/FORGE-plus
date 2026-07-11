#!/usr/bin/env python3
"""RTX snapshots of the Robotiq 2F-140 gear episode (staging + hand-off).

Runs the gear env with --gripper robotiq_2f140 (zero policy action) through
seat -> carry -> descent -> hand-off and saves stills at the key beats.
GEAR GHOST WORKAROUND (canonical, render_gear_recovery.py): the RTX view
draws the gripped physics gear DISPLACED — hide its visual and pose an
exact-dimension plain-USD proxy from root_pose_w before every capture.
Per the grasp-realism rule, every capture logs pad_dz (design +0.040).

    export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 \
           PYTHONPATH=/workspace/FORGE-plus_task3
    /workspace/.venv/bin/python scripts/snap_rq_gear.py
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
env = FrankaGearInsertEnv(cfg)
env.reset()
print("env up", flush=True)

stage = omni.usd.get_context().get_stage()
orig = env.scene.env_origins[0].tolist()

# light
lux = UsdLux.DistantLight.Define(stage, "/World/SnapSun")
lux.CreateIntensityAttr(3200.0)
UsdGeom.Xformable(lux).AddRotateXYZOp().Set(Gf.Vec3f(-32.0, 18.0, 0.0))
dome = UsdLux.DomeLight.Define(stage, "/World/SnapDome")
dome.CreateIntensityAttr(420.0)

# hide the physics gear visual; draw an exact-dim proxy from root_pose_w
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

# head-on +x view (the finger spread is along y — a side view looks straight
# down the spread axis and hides one finger, probe_rq_scene occlusion lesson)
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

SNAPDIR = "/workspace/frames_rq_gear_snap"
os.makedirs(SNAPDIR, exist_ok=True)
import imageio.v2 as imageio

def snap(name):
    _proxy_pose()
    for _ in range(6):
        app.update()
    img = np.asarray(rgb.get_data())
    if img.size and img.ndim == 3:
        imageio.imwrite(f"{SNAPDIR}/{name}.png", img[..., :3])
        # grasp-realism check: pad centre (grasp point) vs gear origin
        from isaaclab.utils.math import matrix_from_quat
        eep = env._robot.data.body_pos_w[0, env._ee_idx]
        eeq = env._robot.data.body_quat_w[0, env._ee_idx]
        ax = matrix_from_quat(eeq.unsqueeze(0))[0, :, 2]
        gz = float(env._obj.data.root_pose_w[0, 2])
        pad_dz = float(eep[2] + 0.214 * ax[2]) - gz
        print(f"[snap] {name}: saved  pad_dz={pad_dz:+.4f} (design +0.040) "
              f"gear_z={gz - orig[2]:.4f} grip_ang="
              f"{float(env._robot.data.joint_pos[0, env._grip_ids[0]]):.4f}",
              flush=True)
    else:
        print(f"[snap] {name}: EMPTY frame", flush=True)

zero = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
# staging v2 beats: predrive ~0-60, seat window ~65-200 (pin/kiss/squeeze),
# handoff ~200 (setup fast-forwards to 12 once the gear is over the shaft)
beats = {40: "predrive_hold", 90: "seat_kiss", 140: "seat_squeeze",
         180: "held"}
done_handoff_snap = False
for step in range(400):
    env.step(zero)
    if step in beats:
        snap(f"{step:04d}_{beats[step]}")
    if not done_handoff_snap and int(env._setup_ctr[0]) == 0 and step > 100:
        snap(f"{step:04d}_policy_handoff")
        done_handoff_snap = True
        break
print("SNAP_DONE", flush=True)
os._exit(0)
