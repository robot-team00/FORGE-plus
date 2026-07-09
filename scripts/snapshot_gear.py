#!/usr/bin/env python3
"""Single-scene RTX snapshots of the task1 gear-insertion env.

Clone of render_forge_min.py's proven harness (same RTX flags, lights,
camera math, annotator patches), pointed at FrankaGearInsertEnv. Runs the
CURRENT checkpoint policy and saves stills at hand-off / mid-descent /
final state -> /workspace/snapshots_gear/gear_snap_{1,2,3}.png
"""
import os, sys, time as _time
import numpy as np
from pathlib import Path
from PIL import Image

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
from forge_plus.isaac_gear_env import FrankaGearInsertEnv, GearInsertEnvCfg
from forge_plus.skills.policy_network import ForceConditionedPolicy, PolicyConfig

S = carb.settings.get_settings()
for _k in ["/rtx/reflections/enabled", "/rtx/translucency/enabled",
           "/rtx/indirectDiffuse/enabled", "/rtx/ambientOcclusion/enabled",
           "/rtx/directLighting/sampledLighting/enabled"]:
    S.set(_k, False)

cfg = GearInsertEnvCfg()
cfg.scene.num_envs = 1
cfg.forge_mode = True
cfg.forge_obj_cls = 0
cfg.forge_no_term = True
from isaaclab.envs import DirectRLEnv as _DRL
FrankaGearInsertEnv.render = _DRL.render
env = FrankaGearInsertEnv(cfg)
print("env built", flush=True)

ckpt = torch.load("/workspace/FORGE-plus_task3/checkpoints/task1_gear_insert_franka.pt",
                  map_location=env.device, weights_only=False)
pc = ckpt["policy_cfg"]
pcfg = pc if isinstance(pc, PolicyConfig) else PolicyConfig(**pc)
policy = ForceConditionedPolicy(pcfg).to(env.device)
policy.load_state_dict(ckpt["policy_state_dict"])
policy.eval()
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

# closer 3/4 framing on the tabletop scene (gear + base at ~(0.43, 0.12, 0.40))
cam = UsdGeom.Camera.Define(stage, "/World/EvalCam")
cam.CreateFocalLengthAttr(30.0)
eye = Gf.Vec3d(float(orig[0]) + 1.25, float(orig[1]) - 0.85, float(orig[2]) + 0.95)
tgt = Gf.Vec3d(float(orig[0]) + 0.40, float(orig[1]) + 0.10, float(orig[2]) + 0.42)
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
print("render ready shape=%s" % str(np.asarray(rgb.get_data()).shape), flush=True)

def grab(path):
    app.update(); app.update()
    d = np.asarray(rgb.get_data())
    if d.ndim >= 3 and d.shape[0] > 1:
        Image.fromarray(d[..., :3]).save(path)
        print("saved " + path, flush=True)

OUT = "/workspace/snapshots_gear"
os.makedirs(OUT, exist_ok=True)

# warmup app.updates disturbed the reset; re-reset like render_forge_min
out = env.reset()
obs = (out[0] if isinstance(out, tuple) else out)["policy"]
snaps = {170: OUT + "/gear_snap_1_handoff.png",
         300: OUT + "/gear_snap_2_descent.png",
         460: OUT + "/gear_snap_3_final.png"}
for k in range(470):
    with torch.no_grad():
        m, _ = policy(obs, env.f_cmd_norm().to(env.device))
    res = env.step(torch.clamp(m, -1, 1))
    obs = res[0]["policy"]
    if k in snaps:
        g = env._obj.data.root_pose_w[0, :3].cpu().numpy() - orig
        print(f"snap at k={k}: gear=({g[0]:.3f},{g[1]:.3f},{g[2]:.3f}) "
              f"Fins={float(env._cf_insert[0]):.2f}", flush=True)
        grab(snaps[k])
print("DONE", flush=True)
app.close()
os._exit(0)
