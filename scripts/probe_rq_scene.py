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

ghost = Articulation(ArticulationCfg(
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
scene.articulations["ghost"] = ghost
scene.clone_environments(copy_from_source=False)
print("cloned", flush=True)
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
    hp = fr.data.body_pos_w[0, bn2.index("panda_hand")]
    hq = fr.data.body_quat_w[0, bn2.index("panda_hand")]
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
if "obj" in ASSETS:
    bn3 = list(fr.data.body_names)
    MUG_GRIP_Z = 0.12
    hand_i = bn3.index("panda_hand")
    for d in [0.12, 0.16, 0.20, 0.24]:
        hp = fr.data.body_pos_w[0, hand_i]
        gp = hp.clone(); gp[2] = hp[2] - d          # candidate grip point: d below the hand
        # seat WIDE OPEN (no overlap while teleporting), grip height at gp
        fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + 0.35, joint_ids=[fid])
        settle(40)
        pose = obj.data.root_pose_w.clone()
        pose[0, 0] = gp[0]; pose[0, 1] = gp[1]; pose[0, 2] = gp[2] - MUG_GRIP_Z
        pose[0, 3] = 1.0; pose[0, 4:7] = 0.0
        for _ in range(30):
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_vel_w))
            settle(1)
        # close and let go of the teleport
        fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + 0.05, joint_ids=[fid])
        settle(150)
        bz0 = float(pose[0, 2])
        bz1 = float(obj.data.root_pose_w[0, 2])
        held = (bz0 - bz1) < 0.03
        print(f"HOLD d={d:.2f}: gp=({float(gp[0]):.3f},{float(gp[1]):.3f},{float(gp[2]):.3f}) "
              f"base z {bz0:.3f}->{bz1:.3f} held={held} fourbar_sep={sep():.4f} "
              f"ang={float(fr.data.joint_pos[0,fid]):+.3f}", flush=True)

os._exit(0)   # skip app.close() — hangs headless
