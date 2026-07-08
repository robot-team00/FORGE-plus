#!/usr/bin/env python3
# snap_rq_handoff.py — RTX close-up SNAPSHOTS of the robotiq env hand-off grasp.
# Runs the same episode as run_recovery_insertion (--gripper robotiq_2f140) but with the
# proven render harness, and saves stills through the seat window (pin/kiss/squeeze/settle)
# plus the first live-policy steps, then exits. Diagnostic only — nothing is scripted
# beyond what the env itself does.
import os, sys, time as _time
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
app = SimulationApp({"headless": True, "width": 960, "height": 720, "extra_args": _EXTRA})
print("booted", flush=True)

import torch, carb
import omni.usd
from pxr import Gf, UsdGeom, UsdLux
import omni.replicator.core as rep
from forge_plus.isaac_pick_place_env import FrankaPickPlaceEnv, PickPlaceEnvCfg
from forge_plus.skills.policy_network import ForceConditionedPolicy, PolicyConfig
from forge_plus.llm.client import HeuristicLLMClient
from forge_plus.llm.recovery_selector import RecoverySelector
from forge_plus.recovery.recovery_loop import RecoveryLoop
print("imports ok", flush=True)

S = carb.settings.get_settings()
for _k in ["/rtx/reflections/enabled", "/rtx/translucency/enabled",
           "/rtx/indirectDiffuse/enabled", "/rtx/ambientOcclusion/enabled",
           "/rtx/directLighting/sampledLighting/enabled"]:
    S.set(_k, False)

cfg = PickPlaceEnvCfg()
cfg.scene.num_envs   = 1
cfg.gripper          = "robotiq_2f140"
cfg.place_strategy   = "insert"
cfg.settle_steps     = 400
cfg.episode_length_s = 120.0
cfg.jam_dx           = float(os.environ.get("JAM", "0.05"))
cfg.forge_mode          = True
cfg.forge_release_mode  = True
cfg.forge_hybrid_retract = True
cfg.forge_no_term       = True
cfg.rec_dur_steps       = 80
cfg.grasp_topdown       = False
cfg.forge_obj_cls       = int(os.environ.get("OBJ", "2"))
cfg.forge_setup_steps   = int(os.environ.get("SETUP", "4000"))
cfg.warmup_substeps     = 100
cfg.scene.replicate_physics = False

from isaaclab.envs import DirectRLEnv as _DRL
FrankaPickPlaceEnv.render = _DRL.render
env = FrankaPickPlaceEnv(cfg)
print("env built", flush=True)

ckpt = torch.load("/workspace/FORGE-plus_task3/checkpoints/task3_forge_entrance.pt",
                  map_location=env.device, weights_only=False)
pc = ckpt["policy_cfg"]
pcfg = pc if isinstance(pc, PolicyConfig) else PolicyConfig(**pc)
policy = ForceConditionedPolicy(pcfg).to(env.device)
policy.load_state_dict(ckpt["policy_state_dict"])
policy.eval()
env._skill_policy = policy

out  = env.reset()
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

# close-up camera on the hand-off point above the rack cell
cam = UsdGeom.Camera.Define(stage, "/World/SnapCam")
cam.CreateFocalLengthAttr(30.0)
eye = Gf.Vec3d(float(orig[0]) + cfg.rack_x + 0.62,
               float(orig[1]) + cfg.rack_y - 0.80,
               float(orig[2]) + 0.98)
tgt = Gf.Vec3d(float(orig[0]) + cfg.rack_x - 0.05,
               float(orig[1]) + cfg.rack_y,
               float(orig[2]) + 0.52)
up  = Gf.Vec3d(0, 0, 1)
fwd = (tgt - eye).GetNormalized()
rgt = Gf.Cross(fwd, up).GetNormalized()
tup = Gf.Cross(rgt, fwd).GetNormalized()
M   = Gf.Matrix4d(rgt[0],rgt[1],rgt[2],0, tup[0],tup[1],tup[2],0,
                  -fwd[0],-fwd[1],-fwd[2],0, eye[0],eye[1],eye[2],1)
UsdGeom.Xformable(cam).AddTransformOp().Set(M)
print("scene ready", flush=True)

for _ in range(110): app.update()
rp  = rep.create.render_product("/World/SnapCam", (960, 720))
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
print("render product ready (shape=%s)" % str(np.asarray(rgb.get_data()).shape), flush=True)

SNAPDIR = "/workspace/frames_rq_snap"
os.makedirs(SNAPDIR, exist_ok=True)
from pathlib import Path
for _f in Path(SNAPDIR).glob("*.png"): _f.unlink()
from PIL import Image

def _save(tag):
    app.update(); app.update()
    d = np.asarray(rgb.get_data())
    if d.ndim >= 3 and d.shape[0] > 1:
        Image.fromarray(d[:, :, :3]).convert("RGB").save(
            os.path.join(SNAPDIR, "snap_%s.png" % tag))
        print("saved snap_%s.png" % tag, flush=True)

class _Done(Exception):
    pass

# seat-window counters to photograph (sc counts 270 -> 0)
WANT = {265: "pin_open", 224: "kiss_start", 176: "kiss_end", 150: "squeeze_mid",
        112: "squeeze_end", 60: "settle", 4: "window_done"}
state = {"post": 0, "carry": 0, "carry_n": 0}

def _on_step(e, attempt, step):
    sc = int(e._rq_seatctr[0].item())
    setup = int(e._setup_ctr[0].item())
    ang = float(e._robot.data.joint_pos[0, e._grip_ids[0]].item())
    sep = float((e._robot.data.body_pos_w[0, e._lf_idx]
                 - e._robot.data.body_pos_w[0, e._rf_idx]).norm().item())
    oz = float(e._obj.data.root_pose_w[0, 2].item() - e.scene.env_origins[0, 2].item())
    if sc in WANT:
        print("[snap] sc=%d ang=%.3f sep=%.4f objz=%.3f" % (sc, ang, sep, oz), flush=True)
        _save("%03d_%s" % (270 - sc, WANT[sc]))
    if sc == 0 and setup > 0:
        # carry / descent / entrance-hold: a frame every ~150 control steps
        state["carry"] += 1
        if state["carry"] % 150 == 1 and state["carry_n"] < 14:
            state["carry_n"] += 1
            ob = e._obj.data.root_pose_w[0, :3] - e.scene.env_origins[0]
            print("[snap] carry_%02d setup=%d obj=(%.3f,%.3f,%.3f) ang=%.3f sep=%.4f"
                  % (state["carry_n"], setup, float(ob[0]), float(ob[1]), float(ob[2]),
                     ang, sep), flush=True)
            _save("carry_%02d" % state["carry_n"])
    if sc == 0 and setup == 0:
        state["post"] += 1
        if state["post"] % 15 == 1 and state["post"] <= 121:
            _save("post_%03d" % state["post"])
        if state["post"] > 121:
            raise _Done()

selector = RecoverySelector(client=HeuristicLLMClient())
loop = RecoveryLoop(selector=selector, k_max=1)
print("running (snapshots through the seat window + first live steps)...", flush=True)
try:
    loop.run(env, on_step=_on_step)
except _Done:
    pass
print("SNAPSHOTS DONE", flush=True)
os._exit(0)
