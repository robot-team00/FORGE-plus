#!/usr/bin/env python3
"""Runtime probe of the composed Franka + Robotiq 2F-140 articulation.

Reports what the env port needs:
  - joint ordering (are panda_joint1-7 still ids 0-6?) and body names
  - the pad-midpoint TCP distance below the panda_hand frame (replaces _grasp_tcp_d=0.067)
  - finger_joint angle -> pad separation calibration (for close/seat/hold thresholds)
"""
import os, sys
os.environ.update({"HOME": "/workspace/persist/ovhome", "MPLBACKEND": "Agg", "DISPLAY": ":99"})
sys.path.insert(0, "/workspace/FORGE-plus_task3")

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))

cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Robot")
cfg.spawn.usd_path = "/workspace/assets/isaac51/Robots/FrankaRobotics/FrankaPanda/franka_robotiq_2f140.usd"
cfg.spawn.activate_contact_sensors = bool(int(os.environ.get("ACS", "1")))
print("activate_contact_sensors =", cfg.spawn.activate_contact_sensors, flush=True)
# TELEPORT CONTRACT (build_franka_robotiq_2f140.py NOTE 2): spawn at the DEFAULT
# arm pose (parse-consistent with the authored flange placement); no arm-pose
# overrides, no joint-state writes — the gripper is driven by targets only.
jp = {k: v for k, v in cfg.init_state.joint_pos.items() if not k.startswith("panda_finger")}
jp.update({"finger_joint": 0.0, ".*_inner_finger_joint": 0.0,
           ".*_inner_finger_pad_joint": 0.0, ".*_outer_.*_joint": 0.0})
cfg.init_state.joint_pos = jp
# actuators: arm groups unchanged; replace the panda hand group with the UR10e
# 2F-140 template (the four-bar rods carry the pads; stock spring is right)
acts = dict(cfg.actuators)
acts.pop("panda_hand", None)
acts["gripper_drive"] = ImplicitActuatorCfg(joint_names_expr=["finger_joint"],
    effort_limit_sim=10.0, velocity_limit_sim=1.0, stiffness=11.25, damping=0.1,
    friction=0.0, armature=0.0)
acts["gripper_finger"] = ImplicitActuatorCfg(joint_names_expr=[".*_inner_finger_joint"],
    effort_limit_sim=1.0, velocity_limit_sim=1.0, stiffness=0.2, damping=0.001,
    friction=0.0, armature=0.0)
acts["gripper_passive"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_pad_joint", ".*_outer_finger_joint",
                      "right_outer_knuckle_joint"],
    effort_limit_sim=1.0, velocity_limit_sim=2.0, stiffness=0.0, damping=0.0,
    friction=0.0, armature=0.0)
cfg.actuators = acts

robot = Articulation(cfg)
sim.reset()
robot.update(sim.get_physics_dt())

jn = list(robot.data.joint_names)
bn = list(robot.data.body_names)
print("JOINTS:", list(enumerate(jn)), flush=True)
print("BODIES:", list(enumerate(bn)), flush=True)
fid = jn.index("finger_joint")
hand = bn.index("panda_hand")
lif, rif = bn.index("left_inner_finger"), bn.index("right_inner_finger")
lim = robot.data.joint_pos_limits[0, fid]
print(f"finger_joint id={fid} limits=({float(lim[0]):.3f}, {float(lim[1]):.3f}) rad", flush=True)

from isaaclab.utils.math import quat_apply_inverse

def report(tag):
    hp = robot.data.body_pos_w[0, hand]
    hq = robot.data.body_quat_w[0, hand]
    lp = robot.data.body_pos_w[0, lif]
    rp = robot.data.body_pos_w[0, rif]
    mid = 0.5 * (lp + rp)
    mid_h = quat_apply_inverse(hq.unsqueeze(0), (mid - hp).unsqueeze(0))[0]
    sep = float((lp - rp).norm())
    ang = float(robot.data.joint_pos[0, fid])
    print(f"  [{tag}] ang={ang:+.3f} pad_sep={sep:.4f} pad_mid_in_hand=({float(mid_h[0]):+.4f},"
          f"{float(mid_h[1]):+.4f},{float(mid_h[2]):+.4f})", flush=True)

def settle(n=120):
    for _ in range(n):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

settle(240)
report("open/init")

# ── one RGB frame so we can SEE the gripper state ──
import numpy as np
from PIL import Image
import omni.usd
import omni.replicator.core as rep
from pxr import Gf, UsdGeom, UsdLux
stage = omni.usd.get_context().get_stage()
dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
dome.CreateIntensityAttr(1200.0)
cam = UsdGeom.Camera.Define(stage, "/World/ProbeCam")
cam.CreateFocalLengthAttr(35.0)
hp = robot.data.body_pos_w[0, hand].cpu().numpy()
eye = Gf.Vec3d(float(hp[0]) + 0.7, float(hp[1]) - 0.7, float(hp[2]) + 0.25)
tgt_p = Gf.Vec3d(float(hp[0]), float(hp[1]), float(hp[2]) - 0.15)
up = Gf.Vec3d(0, 0, 1)
fwd = (tgt_p - eye).GetNormalized()
rgt = Gf.Cross(fwd, up).GetNormalized()
tup = Gf.Cross(rgt, fwd).GetNormalized()
M = Gf.Matrix4d(rgt[0], rgt[1], rgt[2], 0, tup[0], tup[1], tup[2], 0,
                -fwd[0], -fwd[1], -fwd[2], 0, eye[0], eye[1], eye[2], 1)
UsdGeom.Xformable(cam).AddTransformOp().Set(M)
rp = rep.create.render_product("/World/ProbeCam", (960, 540))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])
try:
    from omni.replicator.core.scripts.utils import annotator_utils as _au
    _of = _au._resize_data_for_overscan
    _au._resize_data_for_overscan = lambda d, p: d if not p or p.get("datawindow_overscan_z") is None else _of(d, p)
except Exception:
    pass
for _ in range(160):
    app.update()

def snap(name):
    app.update(); app.update()
    d = np.asarray(rgb.get_data())
    if d.ndim >= 3 and d.shape[0] > 1:
        Image.fromarray(d[:, :, :3]).save(f"/workspace/logs/rq_probe_{name}.png")
        print(f"snap {name} saved ({d.shape})", flush=True)
    else:
        print(f"snap {name} EMPTY", flush=True)

snap("open")

# sweep the drive target and record the calibration curve
for tgt in [0.2, 0.4, 0.6, 0.7]:
    t = torch.zeros(1, 1, device=robot.device) + tgt
    robot.set_joint_position_target(t, joint_ids=[fid])
    settle(180)
    report(f"tgt={tgt:.1f}")
    snap(f"t{int(tgt*10)}")

os._exit(0)   # skip app.close() entirely — it hangs headless and pins the GPU
