#!/usr/bin/env python3
"""Mini InteractiveScene probe: robot(env_.*) + ghost + cloner, NO other assets.
Bisects whether the env's four-bar collapse comes from the scene/cloner machinery
or from the other scene assets."""
import os, sys
os.environ.update({"HOME": "/workspace/persist/ovhome", "MPLBACKEND": "Agg", "DISPLAY": ":99"})
sys.path.insert(0, "/workspace/FORGE-plus_task3")

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

REPLICATE = bool(int(os.environ.get("REP", "0")))
sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, render_interval=4, device="cuda:0"))
scene = InteractiveScene(InteractiveSceneCfg(num_envs=1, env_spacing=4.0,
                                             replicate_physics=REPLICATE))
print("scene created; replicate_physics =", REPLICATE, flush=True)

# mimic-tree v2 roles: drive the finger_joint; LOCK the right pad at its constant
# (target = default_joint_pos); everything else is mimic-owned (passive).
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
        effort_limit_sim=1.0, velocity_limit_sim=2.0, stiffness=0.0, damping=0.0,
        friction=0.0, armature=0.0),
}

GHOST = bool(int(os.environ.get("GHOST", "0")))
ghost = None
if GHOST:
    ghost = ghost = Articulation(ArticulationCfg(
        prim_path="/World/GhostGripper",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/assets/isaac51/Robots/Robotiq/2F-140/Robotiq_2F_140_physics_edit.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0),
            joint_pos={"finger_joint": 0.0, ".*_inner_finger_joint": 0.0,
                       ".*_inner_finger_pad_joint": 0.0, ".*_outer_.*_joint": 0.0}),
        actuators={"all_passive": ImplicitActuatorCfg(joint_names_expr=[".*"],
            effort_limit_sim=1.0, velocity_limit_sim=2.0, stiffness=0.0, damping=0.01,
            friction=0.0, armature=0.0)},
    ))

fr_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
fr_cfg.spawn.usd_path = "/workspace/assets/isaac51/Robots/FrankaRobotics/FrankaPanda/franka_robotiq_2f140.usd"
fr_cfg.spawn.activate_contact_sensors = True
jp = {k: v for k, v in fr_cfg.init_state.joint_pos.items() if not k.startswith("panda_finger")}
jp.update({"finger_joint": 0.0, ".*_inner_finger_joint": 0.0,
           ".*_inner_finger_pad_joint": 0.0, ".*_outer_.*_joint": 0.0})
fr_cfg.init_state.joint_pos = jp
acts = dict(fr_cfg.actuators)
acts.pop("panda_hand", None)
acts.update(GRIP_ACTS)
fr_cfg.actuators = acts
# env parity: OSC arms run with ZERO joint stiffness (damping only)
if bool(int(os.environ.get("ZEROARM", "0"))):
    for _an in ("panda_shoulder", "panda_forearm"):
        fr_cfg.actuators[_an].stiffness = 0.0
        fr_cfg.actuators[_an].damping = 80.0
    print("arm stiffness zeroed (env parity)", flush=True)
fr = Articulation(fr_cfg)

# Optional env assets (the env's exact specs), selected via ASSETS=table,obj,rack
from isaaclab.assets import RigidObject, RigidObjectCfg
ASSETS = [a for a in os.environ.get("ASSETS", "").split(",") if a]
print("extra assets:", ASSETS, flush=True)
if "table" in ASSETS:
    table = RigidObject(RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(size=(0.6, 0.6, 0.40),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 0.20))))
    scene.rigid_objects["table"] = table
if "obj" in ASSETS:
    obj = RigidObject(RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/assets/libero/wine_bottle/wine_bottle_rigid.usd",
            scale=(0.5, 0.5, 0.5),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.30),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=False,
                max_depenetration_velocity=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.12, 0.72))))
    scene.rigid_objects["object"] = obj
if "rack" in ASSETS:
    rack = RigidObject(RigidObjectCfg(
        prim_path="/World/envs/env_.*/Rack",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/assets/libero/wine_rack/wine_rack.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.12, 0.38))))
    scene.rigid_objects["rack"] = rack

scene.articulations["robot"] = fr
if ghost is not None:
    scene.articulations["ghost"] = ghost
scene.clone_environments(copy_from_source=False)
print("cloned", flush=True)
import omni.usd as _ou
from pxr import UsdShade as _US, UsdPhysics as _UP
_st = _ou.get_context().get_stage()
_mat = _US.Material.Define(_st, "/World/RubberPadMat")
_UP.MaterialAPI.Apply(_mat.GetPrim())
_mapi = _UP.MaterialAPI(_mat.GetPrim())
_mapi.CreateStaticFrictionAttr(2.0); _mapi.CreateDynamicFrictionAttr(2.0)
_nb = 0
for _pr in _st.Traverse():
    _pth = _pr.GetPath().pathString
    if _pr.HasAPI(_UP.CollisionAPI) and (("Robot" in _pth and "inner_finger" in _pth)
                                          or "/Object" in _pth):
        _US.MaterialBindingAPI.Apply(_pr).Bind(_mat, materialPurpose="physics")
        _nb += 1
print(f"rubber pads bound to {_nb} collision prims", flush=True)
sim.reset()
print("sim reset ok", flush=True)

def sep():
    bn = list(fr.data.body_names)
    lp = fr.data.body_pos_w[0, bn.index("left_inner_finger")]
    rp = fr.data.body_pos_w[0, bn.index("right_inner_finger")]
    return float((lp - rp).norm())

def settle(n):
    for _ in range(n):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

jn = list(fr.data.joint_names)
fid = jn.index("finger_joint")
settle(240)
print(f"SCENE open : ang={float(fr.data.joint_pos[0,fid]):+.3f} sep={sep():.4f}", flush=True)

def fingertips(tag):
    """Fingertip_01 midpoint in the panda_hand frame (USD xforms; fabric may lag a
    step but the kinematic layout is what we need)."""
    import omni.usd
    from pxr import UsdGeom, Usd
    from isaaclab.utils.math import quat_apply_inverse
    stage2 = omni.usd.get_context().get_stage()
    tips = []
    for prim in stage2.Traverse():
        pth = prim.GetPath().pathString
        if "env_0/Robot" in pth and pth.endswith("Fingertip_01"):
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = m.ExtractTranslation()
            tips.append([t[0], t[1], t[2]])
    if len(tips) != 2:
        print(f"  [{tag}] fingertips found: {len(tips)} (expected 2)", flush=True)
        return
    bn2 = list(fr.data.body_names)
    hp = fr.data.body_pos_w[0, bn2.index("robotiq_base_link")]
    hq = fr.data.body_quat_w[0, bn2.index("robotiq_base_link")]
    mid = torch.tensor([[(tips[0][i] + tips[1][i]) / 2 for i in range(3)]], device=fr.device) - hp
    mh = quat_apply_inverse(hq.unsqueeze(0), mid)[0]
    tsep = sum((tips[0][i] - tips[1][i]) ** 2 for i in range(3)) ** 0.5
    print(f"  [{tag}] fingertip_mid_in_hand=({float(mh[0]):+.4f},{float(mh[1]):+.4f},{float(mh[2]):+.4f}) "
          f"tip_sep={tsep:.4f}", flush=True)

fingertips("open")

# ── JOINT-RELATION measurement (for the mimic-tree v2 gearings): in the HEALTHY
# four-bar, record every gripper joint angle across the drive sweep. The linear
# fits give the exact mimic gearing/offset each joint needs once the fragile loop
# constraints are replaced.
if bool(int(os.environ.get("RELATIONS", "0"))):
    gnames = [n for n in jn if not n.startswith("panda_")]
    gids = [jn.index(n) for n in gnames]
    print("RELATION joints:", gnames, flush=True)
    for tgt in [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]:
        fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + tgt, joint_ids=[fid])
        settle(180)
        vals = [round(float(fr.data.joint_pos[0, g]), 4) for g in gids]
        print(f"RELATION tgt={tgt:.2f}: {dict(zip(gnames, vals))}", flush=True)
    os._exit(0)


# ── HOLD TEST (ground truth for the env warmup seat): teleport the bottle so its
# grip height (base + mug_grip_z) sits at candidate points along the FINGER AXIS
# (knuckle-mid -> finger-mid, extended), close the drive, settle, and report which
# candidate the pads actually HOLD without tearing the four-bar.
# minimal camera for hold-test snapshots
import numpy as np
from PIL import Image
import omni.usd
import omni.replicator.core as rep
from pxr import Gf, UsdGeom, UsdLux
_stage = omni.usd.get_context().get_stage()
UsdLux.DomeLight.Define(_stage, "/World/Dome").CreateIntensityAttr(1200.0)
_cam = UsdGeom.Camera.Define(_stage, "/World/Cam")
_cam.CreateFocalLengthAttr(30.0)
_eye = Gf.Vec3d(0.95, -0.75, 1.05)
_tgt = Gf.Vec3d(0.21, 0.01, 0.82)   # the measured pad zone
_up = Gf.Vec3d(0, 0, 1)
_f = (_tgt - _eye).GetNormalized(); _r = Gf.Cross(_f, _up).GetNormalized(); _t = Gf.Cross(_r, _f).GetNormalized()
UsdGeom.Xformable(_cam).AddTransformOp().Set(Gf.Matrix4d(
    _r[0],_r[1],_r[2],0, _t[0],_t[1],_t[2],0, -_f[0],-_f[1],-_f[2],0, _eye[0],_eye[1],_eye[2],1))
_rp = rep.create.render_product("/World/Cam", (960, 540))
_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
_rgb.attach([_rp])
try:
    from omni.replicator.core.scripts.utils import annotator_utils as _au
    _of = _au._resize_data_for_overscan
    _au._resize_data_for_overscan = lambda d, pr: d if not pr or pr.get("datawindow_overscan_z") is None else _of(d, pr)
except Exception:
    pass
for _ in range(140): app.update()

PIN = {"pose": None, "obj": None}
def snap(name):
    for _ in range(2):
        if PIN["pose"] is not None:
            PIN["obj"].write_root_pose_to_sim(PIN["pose"])
            PIN["obj"].write_root_velocity_to_sim(torch.zeros_like(PIN["obj"].data.root_vel_w))
        app.update()
    d = np.asarray(_rgb.get_data())
    if d.ndim >= 3 and d.shape[0] > 1:
        Image.fromarray(d[:, :, :3]).save(f"/workspace/logs/rqs_{name}.png")
        print(f"snap {name} saved", flush=True)

# second camera: opposite azimuth on the robot gripper + one on the ghost
_cam2 = UsdGeom.Camera.Define(_stage, "/World/Cam2")
_cam2.CreateFocalLengthAttr(30.0)
_e2, _t2 = Gf.Vec3d(1.35, 0.02, 0.88), Gf.Vec3d(0.21, 0.0, 0.80)   # head-on (+x): both fingers side by side
_f2 = (_t2 - _e2).GetNormalized(); _r2 = Gf.Cross(_f2, Gf.Vec3d(0,0,1)).GetNormalized(); _u2 = Gf.Cross(_r2, _f2).GetNormalized()
UsdGeom.Xformable(_cam2).AddTransformOp().Set(Gf.Matrix4d(
    _r2[0],_r2[1],_r2[2],0, _u2[0],_u2[1],_u2[2],0, -_f2[0],-_f2[1],-_f2[2],0, _e2[0],_e2[1],_e2[2],1))
_rp2 = rep.create.render_product("/World/Cam2", (960, 540))
_rgb2 = rep.AnnotatorRegistry.get_annotator("rgb")
_rgb2.attach([_rp2])
_cam3 = UsdGeom.Camera.Define(_stage, "/World/Cam3")
_cam3.CreateFocalLengthAttr(30.0)
_e3, _t3 = Gf.Vec3d(0.6, -0.6, -4.6), Gf.Vec3d(0.0, 0.0, -5.05)
_f3 = (_t3 - _e3).GetNormalized(); _r3 = Gf.Cross(_f3, Gf.Vec3d(0,0,1)).GetNormalized(); _u3 = Gf.Cross(_r3, _f3).GetNormalized()
UsdGeom.Xformable(_cam3).AddTransformOp().Set(Gf.Matrix4d(
    _r3[0],_r3[1],_r3[2],0, _u3[0],_u3[1],_u3[2],0, -_f3[0],-_f3[1],-_f3[2],0, _e3[0],_e3[1],_e3[2],1))
_rp3 = rep.create.render_product("/World/Cam3", (960, 540))
_rgb3 = rep.AnnotatorRegistry.get_annotator("rgb")
_rgb3.attach([_rp3])

def snap2(name):
    for _ in range(2):
        if PIN["pose"] is not None:
            PIN["obj"].write_root_pose_to_sim(PIN["pose"])
            PIN["obj"].write_root_velocity_to_sim(torch.zeros_like(PIN["obj"].data.root_vel_w))
        app.update()
    for tag, ann in [("b", _rgb2), ("ghost", _rgb3)]:
        d = np.asarray(ann.get_data())
        if d.ndim >= 3 and d.shape[0] > 1:
            Image.fromarray(d[:, :, :3]).save(f"/workspace/logs/rqs_{name}_{tag}.png")
    print(f"snap2 {name} saved", flush=True)

snap("preclose")

if "obj" in ASSETS:
    bn3 = list(fr.data.body_names)
    MUG_GRIP_Z = 0.12
    hand_i = bn3.index("robotiq_base_link")
    from isaaclab.utils.math import quat_apply_inverse
    # CONTACT SCAN: sweep the bottle's grip height along the vertical line through the
    # PAD MIDPOINT; the close-stall angle maps where the pads are and what they touch:
    #   stall ~0.146 = pads touch each other (no bottle);  >0.15 = bottle contact;
    #   ~0.155-0.17 = neck (16-18 mm);  ~0.25+ = fat body.
    hp = fr.data.body_pos_w[0, hand_i]
    # ONE clean hold cycle on the STOCK four-bar (wear-free: single seat):
    # open wide, teleport the bottle so its grip height (base+0.12) is d below the
    # hand, squeeze overlapping the teleport hand-off, then physics owns it.
    d = 0.165
    gp = hp.clone(); gp[2] = hp[2] - d
    fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + 0.35, joint_ids=[fid])
    settle(40)
    snap("open")
    snap2("open")
    pose = obj.data.root_pose_w.clone()
    # measured: bottle origin ~ its NECK (extent -0.16/+0.06); pad faces ~0.17-0.20
    # below base_link -> origin at base_link - 0.20 puts the neck between the pads
    pose[0, 0] = gp[0]; pose[0, 1] = gp[1]; pose[0, 2] = hp[2] - 0.20
    pose[0, 3] = 1.0; pose[0, 4:7] = 0.0
    PIN["pose"] = pose; PIN["obj"] = obj
    for _i in range(40):
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_vel_w))
        settle(1)
        if _i == 20:
            _bn = list(fr.data.body_names)
            print(f"  [pin] authored_obj_z={float(pose[0,2]):.3f} actual_obj_z={float(obj.data.root_pose_w[0,2]):.3f} "
                  f"base_link_z={float(fr.data.body_pos_w[0,_bn.index('robotiq_base_link')][2]):.3f} "
                  f"l_innf_z={float(fr.data.body_pos_w[0,_bn.index('left_inner_finger')][2]):.3f} "
                  f"r_innf_z={float(fr.data.body_pos_w[0,_bn.index('right_inner_finger')][2]):.3f}", flush=True)
            snap("seated"); snap2("seated")   # mid-teleport: true seat pose
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_vel_w))
    # pads KISS the neck while pinned (no squeeze -> no penetration buildup)
    fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + 0.165, joint_ids=[fid])
    for _i in range(20):
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_vel_w))
        settle(1)
    PIN["pose"] = None
    # NOW squeeze — physics owns the bottle; the pads catch it within ~10 steps
    fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + 0.05, joint_ids=[fid])
    settle(30)
    snap("squeezed"); snap2("squeezed")
    settle(200)
    snap("held")
    bz1 = float(obj.data.root_pose_w[0, 2])
    print(f"STOCKHOLD d={d}: base z {float(pose[0,2]):.3f} -> {bz1:.3f} "
          f"held={(float(pose[0,2]) - bz1) < 0.03} ang={float(fr.data.joint_pos[0,fid]):+.3f} "
          f"sep={sep():.4f}", flush=True)
    fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + 0.40, joint_ids=[fid])
    settle(120)
    bz2 = float(obj.data.root_pose_w[0, 2])
    print(f"RELEASE : base z {bz1:.3f} -> {bz2:.3f} dropped={bz2 < bz1 - 0.05} "
          f"ang={float(fr.data.joint_pos[0,fid]):+.3f}", flush=True)
    snap("released")

os._exit(0)


# ── HOLD TEST (ground truth for the env warmup seat): teleport the bottle so its
# grip height (base + mug_grip_z) sits at candidate points along the FINGER AXIS
# (knuckle-mid -> finger-mid, extended), close the drive, settle, and report which
# candidate the pads actually HOLD without tearing the four-bar.
# minimal camera for hold-test snapshots
import numpy as np
from PIL import Image
import omni.usd
import omni.replicator.core as rep
from pxr import Gf, UsdGeom, UsdLux
_stage = omni.usd.get_context().get_stage()
UsdLux.DomeLight.Define(_stage, "/World/Dome").CreateIntensityAttr(1200.0)
_cam = UsdGeom.Camera.Define(_stage, "/World/Cam")
_cam.CreateFocalLengthAttr(30.0)
_eye = Gf.Vec3d(0.95, -0.75, 1.05)
_tgt = Gf.Vec3d(0.21, 0.01, 0.82)   # the measured pad zone
_up = Gf.Vec3d(0, 0, 1)
_f = (_tgt - _eye).GetNormalized(); _r = Gf.Cross(_f, _up).GetNormalized(); _t = Gf.Cross(_r, _f).GetNormalized()
UsdGeom.Xformable(_cam).AddTransformOp().Set(Gf.Matrix4d(
    _r[0],_r[1],_r[2],0, _t[0],_t[1],_t[2],0, -_f[0],-_f[1],-_f[2],0, _eye[0],_eye[1],_eye[2],1))
_rp = rep.create.render_product("/World/Cam", (960, 540))
_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
_rgb.attach([_rp])
try:
    from omni.replicator.core.scripts.utils import annotator_utils as _au
    _of = _au._resize_data_for_overscan
    _au._resize_data_for_overscan = lambda d, pr: d if not pr or pr.get("datawindow_overscan_z") is None else _of(d, pr)
except Exception:
    pass
for _ in range(140): app.update()

PIN = {"pose": None, "obj": None}
def snap(name):
    for _ in range(2):
        if PIN["pose"] is not None:
            PIN["obj"].write_root_pose_to_sim(PIN["pose"])
            PIN["obj"].write_root_velocity_to_sim(torch.zeros_like(PIN["obj"].data.root_vel_w))
        app.update()
    d = np.asarray(_rgb.get_data())
    if d.ndim >= 3 and d.shape[0] > 1:
        Image.fromarray(d[:, :, :3]).save(f"/workspace/logs/rqs_{name}.png")
        print(f"snap {name} saved", flush=True)

# second camera: opposite azimuth on the robot gripper + one on the ghost
_cam2 = UsdGeom.Camera.Define(_stage, "/World/Cam2")
_cam2.CreateFocalLengthAttr(30.0)
_e2, _t2 = Gf.Vec3d(1.35, 0.02, 0.88), Gf.Vec3d(0.21, 0.0, 0.80)   # head-on (+x): both fingers side by side
_f2 = (_t2 - _e2).GetNormalized(); _r2 = Gf.Cross(_f2, Gf.Vec3d(0,0,1)).GetNormalized(); _u2 = Gf.Cross(_r2, _f2).GetNormalized()
UsdGeom.Xformable(_cam2).AddTransformOp().Set(Gf.Matrix4d(
    _r2[0],_r2[1],_r2[2],0, _u2[0],_u2[1],_u2[2],0, -_f2[0],-_f2[1],-_f2[2],0, _e2[0],_e2[1],_e2[2],1))
_rp2 = rep.create.render_product("/World/Cam2", (960, 540))
_rgb2 = rep.AnnotatorRegistry.get_annotator("rgb")
_rgb2.attach([_rp2])
_cam3 = UsdGeom.Camera.Define(_stage, "/World/Cam3")
_cam3.CreateFocalLengthAttr(30.0)
_e3, _t3 = Gf.Vec3d(0.6, -0.6, -4.6), Gf.Vec3d(0.0, 0.0, -5.05)
_f3 = (_t3 - _e3).GetNormalized(); _r3 = Gf.Cross(_f3, Gf.Vec3d(0,0,1)).GetNormalized(); _u3 = Gf.Cross(_r3, _f3).GetNormalized()
UsdGeom.Xformable(_cam3).AddTransformOp().Set(Gf.Matrix4d(
    _r3[0],_r3[1],_r3[2],0, _u3[0],_u3[1],_u3[2],0, -_f3[0],-_f3[1],-_f3[2],0, _e3[0],_e3[1],_e3[2],1))
_rp3 = rep.create.render_product("/World/Cam3", (960, 540))
_rgb3 = rep.AnnotatorRegistry.get_annotator("rgb")
_rgb3.attach([_rp3])

def snap2(name):
    for _ in range(2):
        if PIN["pose"] is not None:
            PIN["obj"].write_root_pose_to_sim(PIN["pose"])
            PIN["obj"].write_root_velocity_to_sim(torch.zeros_like(PIN["obj"].data.root_vel_w))
        app.update()
    for tag, ann in [("b", _rgb2), ("ghost", _rgb3)]:
        d = np.asarray(ann.get_data())
        if d.ndim >= 3 and d.shape[0] > 1:
            Image.fromarray(d[:, :, :3]).save(f"/workspace/logs/rqs_{name}_{tag}.png")
    print(f"snap2 {name} saved", flush=True)

snap("preclose")

if "obj" in ASSETS:
    bn3 = list(fr.data.body_names)
    MUG_GRIP_Z = 0.12
    hand_i = bn3.index("robotiq_base_link")
    from isaaclab.utils.math import quat_apply_inverse
    # CONTACT SCAN: sweep the bottle's grip height along the vertical line through the
    # PAD MIDPOINT; the close-stall angle maps where the pads are and what they touch:
    #   stall ~0.146 = pads touch each other (no bottle);  >0.15 = bottle contact;
    #   ~0.155-0.17 = neck (16-18 mm);  ~0.25+ = fat body.
    pm = 0.5 * (fr.data.body_pos_w[0, bn3.index("left_inner_finger")]
                + fr.data.body_pos_w[0, bn3.index("right_inner_finger")])
    hp = fr.data.body_pos_w[0, hand_i]
    hq = fr.data.body_quat_w[0, hand_i]
    print(f"SCAN pad_mid_world=({float(pm[0]):.3f},{float(pm[1]):.3f},{float(pm[2]):.3f}) "
          f"hand=({float(hp[0]):.3f},{float(hp[1]):.3f},{float(hp[2]):.3f})", flush=True)
    results = []
    for dz in [-0.17, -0.20, -0.23, -0.26, -0.29]:
        zc = float(pm[2]) + dz                      # candidate grip height (world z)
        fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + 0.35, joint_ids=[fid])
        settle(30)
        pose = obj.data.root_pose_w.clone()
        pose[0, 0] = pm[0]; pose[0, 1] = pm[1]; pose[0, 2] = zc - MUG_GRIP_Z
        pose[0, 3] = 1.0; pose[0, 4:7] = 0.0
        # close to a firm target WHILE the teleport holds the bottle in place
        fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + 0.05, joint_ids=[fid])
        for _ in range(60):
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_vel_w))
            settle(1)
        ang = float(fr.data.joint_pos[0, fid])
        gp_h = quat_apply_inverse(hq.unsqueeze(0),
                                  (torch.tensor([float(pm[0]), float(pm[1]), zc],
                                                device=fr.device) - hp).unsqueeze(0))[0]
        print(f"SCAN dz={dz:+.2f}: zc={zc:.3f} stall={ang:+.3f} "
              f"grip_in_hand=({float(gp_h[0]):+.3f},{float(gp_h[1]):+.3f},{float(gp_h[2]):+.3f})",
              flush=True)
        snap(f"scan_{int(-dz*100)}")
        results.append((dz, ang))

    # HOLD test at the depth whose stall looks like the NECK (~0.15-0.19)
    _neck = [r for r in results if 0.14 < r[1] < 0.20]
    _pick = min(_neck, key=lambda r: abs(r[1] - 0.165))[0] if _neck else -0.23
    print(f"HOLD pick dz={_pick:+.2f}", flush=True)
    zc = float(pm[2]) + _pick
    fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + 0.20, joint_ids=[fid])
    settle(30)
    pose = obj.data.root_pose_w.clone()
    pose[0, 0] = pm[0]; pose[0, 1] = pm[1]; pose[0, 2] = zc - MUG_GRIP_Z
    pose[0, 3] = 1.0; pose[0, 4:7] = 0.0
    PIN["pose"] = pose; PIN["obj"] = obj
    for _i in range(40):
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_vel_w))
        settle(1)
        if _i == 20:
            _bn = list(fr.data.body_names)
            print(f"  [pin] authored_obj_z={float(pose[0,2]):.3f} actual_obj_z={float(obj.data.root_pose_w[0,2]):.3f} "
                  f"base_link_z={float(fr.data.body_pos_w[0,_bn.index('robotiq_base_link')][2]):.3f} "
                  f"l_innf_z={float(fr.data.body_pos_w[0,_bn.index('left_inner_finger')][2]):.3f} "
                  f"r_innf_z={float(fr.data.body_pos_w[0,_bn.index('right_inner_finger')][2]):.3f}", flush=True)
            snap("seated"); snap2("seated")   # mid-teleport: true seat pose
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_vel_w))
    fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + 0.08, joint_ids=[fid])
    for _ in range(15):
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_vel_w))
        settle(1)
    snap("squeezed")
    snap2("squeezed")
    settle(200)
    snap("released_teleport")
    bz1 = float(obj.data.root_pose_w[0, 2])
    print(f"HOLDNECK: base z {float(pose[0,2]):.3f} -> {bz1:.3f} "
          f"held={(float(pose[0,2]) - bz1) < 0.03} ang={float(fr.data.joint_pos[0,fid]):+.3f} "
          f"sep={sep():.4f}", flush=True)
    # and can it RELEASE? open target, bottle should drop free
    fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + 0.40, joint_ids=[fid])
    settle(120)
    bz2 = float(obj.data.root_pose_w[0, 2])
    print(f"RELEASE : base z {bz1:.3f} -> {bz2:.3f} dropped={bz2 < bz1 - 0.05} "
          f"ang={float(fr.data.joint_pos[0,fid]):+.3f}", flush=True)

os._exit(0)


# ── HOLD TEST (ground truth for the env warmup seat): teleport the bottle so its
# grip height (base + mug_grip_z) sits at candidate points along the FINGER AXIS
# (knuckle-mid -> finger-mid, extended), close the drive, settle, and report which
# candidate the pads actually HOLD without tearing the four-bar.
# minimal camera for hold-test snapshots
import numpy as np
from PIL import Image
import omni.usd
import omni.replicator.core as rep
from pxr import Gf, UsdGeom, UsdLux
_stage = omni.usd.get_context().get_stage()
UsdLux.DomeLight.Define(_stage, "/World/Dome").CreateIntensityAttr(1200.0)
_cam = UsdGeom.Camera.Define(_stage, "/World/Cam")
_cam.CreateFocalLengthAttr(30.0)
_eye = Gf.Vec3d(0.95, -0.75, 1.05)
_tgt = Gf.Vec3d(0.21, 0.01, 0.82)   # the measured pad zone
_up = Gf.Vec3d(0, 0, 1)
_f = (_tgt - _eye).GetNormalized(); _r = Gf.Cross(_f, _up).GetNormalized(); _t = Gf.Cross(_r, _f).GetNormalized()
UsdGeom.Xformable(_cam).AddTransformOp().Set(Gf.Matrix4d(
    _r[0],_r[1],_r[2],0, _t[0],_t[1],_t[2],0, -_f[0],-_f[1],-_f[2],0, _eye[0],_eye[1],_eye[2],1))
_rp = rep.create.render_product("/World/Cam", (960, 540))
_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
_rgb.attach([_rp])
try:
    from omni.replicator.core.scripts.utils import annotator_utils as _au
    _of = _au._resize_data_for_overscan
    _au._resize_data_for_overscan = lambda d, pr: d if not pr or pr.get("datawindow_overscan_z") is None else _of(d, pr)
except Exception:
    pass
for _ in range(140): app.update()

PIN = {"pose": None, "obj": None}
def snap(name):
    for _ in range(2):
        if PIN["pose"] is not None:
            PIN["obj"].write_root_pose_to_sim(PIN["pose"])
            PIN["obj"].write_root_velocity_to_sim(torch.zeros_like(PIN["obj"].data.root_vel_w))
        app.update()
    d = np.asarray(_rgb.get_data())
    if d.ndim >= 3 and d.shape[0] > 1:
        Image.fromarray(d[:, :, :3]).save(f"/workspace/logs/rqs_{name}.png")
        print(f"snap {name} saved", flush=True)

# second camera: opposite azimuth on the robot gripper + one on the ghost
_cam2 = UsdGeom.Camera.Define(_stage, "/World/Cam2")
_cam2.CreateFocalLengthAttr(30.0)
_e2, _t2 = Gf.Vec3d(1.35, 0.02, 0.88), Gf.Vec3d(0.21, 0.0, 0.80)   # head-on (+x): both fingers side by side
_f2 = (_t2 - _e2).GetNormalized(); _r2 = Gf.Cross(_f2, Gf.Vec3d(0,0,1)).GetNormalized(); _u2 = Gf.Cross(_r2, _f2).GetNormalized()
UsdGeom.Xformable(_cam2).AddTransformOp().Set(Gf.Matrix4d(
    _r2[0],_r2[1],_r2[2],0, _u2[0],_u2[1],_u2[2],0, -_f2[0],-_f2[1],-_f2[2],0, _e2[0],_e2[1],_e2[2],1))
_rp2 = rep.create.render_product("/World/Cam2", (960, 540))
_rgb2 = rep.AnnotatorRegistry.get_annotator("rgb")
_rgb2.attach([_rp2])
_cam3 = UsdGeom.Camera.Define(_stage, "/World/Cam3")
_cam3.CreateFocalLengthAttr(30.0)
_e3, _t3 = Gf.Vec3d(0.6, -0.6, -4.6), Gf.Vec3d(0.0, 0.0, -5.05)
_f3 = (_t3 - _e3).GetNormalized(); _r3 = Gf.Cross(_f3, Gf.Vec3d(0,0,1)).GetNormalized(); _u3 = Gf.Cross(_r3, _f3).GetNormalized()
UsdGeom.Xformable(_cam3).AddTransformOp().Set(Gf.Matrix4d(
    _r3[0],_r3[1],_r3[2],0, _u3[0],_u3[1],_u3[2],0, -_f3[0],-_f3[1],-_f3[2],0, _e3[0],_e3[1],_e3[2],1))
_rp3 = rep.create.render_product("/World/Cam3", (960, 540))
_rgb3 = rep.AnnotatorRegistry.get_annotator("rgb")
_rgb3.attach([_rp3])

def snap2(name):
    for _ in range(2):
        if PIN["pose"] is not None:
            PIN["obj"].write_root_pose_to_sim(PIN["pose"])
            PIN["obj"].write_root_velocity_to_sim(torch.zeros_like(PIN["obj"].data.root_vel_w))
        app.update()
    for tag, ann in [("b", _rgb2), ("ghost", _rgb3)]:
        d = np.asarray(ann.get_data())
        if d.ndim >= 3 and d.shape[0] > 1:
            Image.fromarray(d[:, :, :3]).save(f"/workspace/logs/rqs_{name}_{tag}.png")
    print(f"snap2 {name} saved", flush=True)

snap("preclose")

if "obj" in ASSETS:
    bn3 = list(fr.data.body_names)
    MUG_GRIP_Z = 0.12
    hand_i = bn3.index("robotiq_base_link")
    for d in [0.18, 0.21, 0.24]:
        hp = fr.data.body_pos_w[0, hand_i]
        gp = hp.clone(); gp[2] = hp[2] - d          # candidate grip point: d below the hand
        # seat with the pads JUST AT the neck width (~18 mm opening) so the grip is
        # already closing on the neck when the teleport lets go (franka-warmup style)
        fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + 0.165, joint_ids=[fid])
        pose = obj.data.root_pose_w.clone()
        # (stale duplicated block, not executed)
        pose[0, 0] = gp[0]; pose[0, 1] = gp[1]; pose[0, 2] = gp[2] - MUG_GRIP_Z
        pose[0, 3] = 1.0; pose[0, 4:7] = 0.0
        for _ in range(40):
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_vel_w))
            settle(1)
        # squeeze WHILE still holding the teleport a few steps, then let physics own it
        fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + 0.10, joint_ids=[fid])
        for _ in range(12):
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_vel_w))
            settle(1)
        settle(150)
        bz0 = float(pose[0, 2])
        bz1 = float(obj.data.root_pose_w[0, 2])
        held = (bz0 - bz1) < 0.03
        print(f"HOLD d={d:.2f}: gp=({float(gp[0]):.3f},{float(gp[1]):.3f},{float(gp[2]):.3f}) "
              f"base z {bz0:.3f}->{bz1:.3f} held={held} fourbar_sep={sep():.4f} "
              f"ang={float(fr.data.joint_pos[0,fid]):+.3f}", flush=True)
        snap(f"hold_d{int(d*100)}")

os._exit(0)   # skip app.close() — hangs headless
