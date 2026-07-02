#!/usr/bin/env python3
"""Isolate the 2F-140 four-bar collapse:
A) standalone 2F-140 articulation — does the linkage hold by itself?
B) composed Franka+140 at the DEFAULT arm pose (no init override) — is the
   collapse caused by the initial-pose snap?
"""
import os, sys
os.environ.update({"HOME": "/workspace/persist/ovhome", "MPLBACKEND": "Agg", "DISPLAY": ":99"})
sys.path.insert(0, "/workspace/FORGE-plus_task3")

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import numpy as np
import torch
from PIL import Image
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))

GRIP_ACTS = {
    "gripper_drive": ImplicitActuatorCfg(joint_names_expr=["finger_joint"],
        effort_limit_sim=10.0, velocity_limit_sim=1.0, stiffness=11.25, damping=0.1,
        friction=0.0, armature=0.0),
    "gripper_finger": ImplicitActuatorCfg(joint_names_expr=[".*_inner_finger_joint"],
        effort_limit_sim=1.0, velocity_limit_sim=1.0, stiffness=0.2, damping=0.001,
        friction=0.0, armature=0.0),
    "gripper_passive": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_pad_joint", ".*_outer_finger_joint",
                          "right_outer_knuckle_joint"],
        effort_limit_sim=1.0, velocity_limit_sim=1.0, stiffness=0.0, damping=0.0,
        friction=0.0, armature=0.0),
}

# A-ghost) standalone gripper articulation in the same scene — testing whether its
# presence changes how PhysX materializes the excluded loop joints for the FRANKA one
# (healthy-B runs all had it; broken runs all lacked it).
solo_cfg = ArticulationCfg(
    prim_path="/World/Solo",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/assets/isaac51/Robots/Robotiq/2F-140/Robotiq_2F_140_physics_edit.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 1.0, 1.0),
        joint_pos={"finger_joint": 0.0, ".*_inner_finger_joint": 0.0,
                   ".*_inner_finger_pad_joint": 0.0, ".*_outer_.*_joint": 0.0}),
    actuators=dict(GRIP_ACTS),
)
solo = Articulation(solo_cfg)

# B) composed Franka+140 at the DEFAULT arm pose (matches the authored flange placement)
fr_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Franka")
fr_cfg.spawn.usd_path = "/workspace/assets/isaac51/Robots/FrankaRobotics/FrankaPanda/franka_robotiq_2f140.usd"
jp = {k: v for k, v in fr_cfg.init_state.joint_pos.items() if not k.startswith("panda_finger")}
jp.update({"finger_joint": 0.0, ".*_inner_finger_joint": 0.0,
           ".*_inner_finger_pad_joint": 0.0, ".*_outer_.*_joint": 0.0})
fr_cfg.init_state.joint_pos = jp
acts = dict(fr_cfg.actuators)
acts.pop("panda_hand", None)
acts.update(GRIP_ACTS)
fr_cfg.actuators = acts
fr = Articulation(fr_cfg)

sim.reset()

def settle(n):
    for _ in range(n):
        fr.write_data_to_sim()
        sim.step()
        fr.update(sim.get_physics_dt())

def sep(robot):
    bn = list(robot.data.body_names)
    lp = robot.data.body_pos_w[0, bn.index("left_inner_finger")]
    rp = robot.data.body_pos_w[0, bn.index("right_inner_finger")]
    return float((lp - rp).norm())

settle(240)
jn_f = list(fr.data.joint_names)
fid_f = jn_f.index("finger_joint")
print(f"B open : fr   ang={float(fr.data.joint_pos[0,fid_f]):+.3f} sep={sep(fr):.4f}", flush=True)

for tgt in [0.2, 0.4, 0.6, 0.7]:
    t = torch.zeros(1, 1, device=fr.device) + tgt
    fr.set_joint_position_target(t, joint_ids=[fid_f])
    settle(200)
    print(f"B tgt={tgt}: ang={float(fr.data.joint_pos[0,fid_f]):+.3f} sep={sep(fr):.4f}", flush=True)

# pad-mid TCP offset below panda_hand at the grip angle (for _grasp_tcp_d)
from isaaclab.utils.math import quat_apply_inverse
bnf = list(fr.data.body_names)
hand_i = bnf.index("panda_hand")
hp = fr.data.body_pos_w[0, hand_i]; hq = fr.data.body_quat_w[0, hand_i]
mid = 0.5 * (fr.data.body_pos_w[0, bnf.index("left_inner_finger")]
             + fr.data.body_pos_w[0, bnf.index("right_inner_finger")])
mh = quat_apply_inverse(hq.unsqueeze(0), (mid - hp).unsqueeze(0))[0]
print(f"C pad_mid_in_hand=({float(mh[0]):+.4f},{float(mh[1]):+.4f},{float(mh[2]):+.4f})", flush=True)

# snapshot both
import omni.usd
import omni.replicator.core as rep
from pxr import Gf, UsdGeom, UsdLux
stage = omni.usd.get_context().get_stage()
UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(1200.0)
cam = UsdGeom.Camera.Define(stage, "/World/Cam")
cam.CreateFocalLengthAttr(30.0)
eye, tgt_p, up = Gf.Vec3d(1.3, -0.9, 1.3), Gf.Vec3d(0.2, 0.5, 0.8), Gf.Vec3d(0, 0, 1)
fwd = (tgt_p - eye).GetNormalized(); rgt = Gf.Cross(fwd, up).GetNormalized(); tup = Gf.Cross(rgt, fwd).GetNormalized()
UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d(
    rgt[0],rgt[1],rgt[2],0, tup[0],tup[1],tup[2],0, -fwd[0],-fwd[1],-fwd[2],0, eye[0],eye[1],eye[2],1))
rp = rep.create.render_product("/World/Cam", (960, 540))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])
try:
    from omni.replicator.core.scripts.utils import annotator_utils as _au
    _of = _au._resize_data_for_overscan
    _au._resize_data_for_overscan = lambda d, p: d if not p or p.get("datawindow_overscan_z") is None else _of(d, p)
except Exception:
    pass
for _ in range(160): app.update()
d = np.asarray(rgb.get_data())
if d.ndim >= 3 and d.shape[0] > 1:
    Image.fromarray(d[:, :, :3]).save("/workspace/logs/rq_isolate.png")
    print("snap saved", d.shape, flush=True)
os._exit(0)   # skip app.close() entirely — it hangs headless and pins the GPU
