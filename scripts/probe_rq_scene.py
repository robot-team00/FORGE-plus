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
        effort_limit_sim=20.0, velocity_limit_sim=2.0, stiffness=10.0, damping=0.05,
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
jp["panda_joint6"] = 1.45   # pitch the flange ~horizontal: gripper axis perpendicular to the bottle
jp.update({"finger_joint": 0.0, ".*_inner_finger_joint": 0.0,
           ".*_inner_finger_pad_joint": 0.0, ".*_outer_.*_joint": 0.0})
fr_cfg.init_state.joint_pos = jp
acts = dict(fr_cfg.actuators)
acts.pop("panda_hand", None)
acts.update(GRIP_ACTS)
if bool(int(os.environ.get("MIMIC", "0"))):
    acts["gripper_finger"].stiffness = 2.0
    acts["gripper_finger"].damping = 0.02
    acts["gripper_finger"].effort_limit_sim = 4.0
if bool(int(os.environ.get("PADK0", "0"))):   # free followers: observe what the residual
    acts["gripper_finger"].stiffness = 0.0    # maximal-coordinate loop joints enforce
    acts["gripper_finger"].damping = 0.001
    acts["gripper_finger"].effort_limit_sim = 0.01
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
        from pxr import UsdGeom as _UGx, Usd as _Ux
        _bc = _UGx.BBoxCache(_Ux.TimeCode.Default(), [_UGx.Tokens.default_])
        _r = _bc.ComputeLocalBound(_pr).ComputeAlignedRange()
        _apx = _pr.GetAttribute("physics:approximation")
        print(f"  COLLIDER {_pth} type={_pr.GetTypeName()} approx={_apx.Get() if _apx else '?'} "
              f"size=({_r.GetSize()[0]:.3f},{_r.GetSize()[1]:.3f},{_r.GetSize()[2]:.3f})", flush=True)
print(f"rubber pads bound to {_nb} collision prims", flush=True)
# v48 discovery: the four-bar LOOP joints are ALIVE (as maximal-coordinate joints);
# the mechanism just assembles in the WRONG four-bar branch at spawn because NVIDIA
# models outer_finger_joint as a free 0-180deg pivot (the adaptive-grasp spring DOF;
# the URDF has it FIXED). Right side proved the correct branch: outer_finger=0,
# inner_finger=-theta (pads PARALLEL), pad_joint=+theta — exact to 4 decimals.
# LOCKOF pins outer_finger at ~0 so only the correct branch can assemble.
if bool(int(os.environ.get("LOCKOF", "0"))):
    from pxr import UsdPhysics as _UP3
    for _pr in _st.Traverse():
        if _pr.GetName() in ("left_outer_finger_joint", "right_outer_finger_joint") \
                and "/Robot" in _pr.GetPath().pathString:
            _rj3 = _UP3.RevoluteJoint(_pr)
            _rj3.CreateLowerLimitAttr(0.0)
            _rj3.CreateUpperLimitAttr(0.01)
            print(f"  LOCKOF {_pr.GetPath()}", flush=True)
MIMIC = bool(int(os.environ.get("MIMIC", "0")))
if MIMIC:
    # four-bar loop joints are dead in the merged articulation; couple the follower
    # joints to the drive with PhysX mimic joints (follower = -(gearing*ref + offset)).
    # The 2F-140 is a parallelogram gripper: pads must STAY PARALLEL through the
    # stroke (gearing +1 scissored the tips inward — wrong sign). Inner knuckle bars
    # are coupled too so the parallelogram links track instead of dangling.
    from pxr import PhysxSchema as _PS
    # gearings derived from NVIDIA's physics_edit asset (dump_rq_joints.py): the
    # inner-finger springs target -45deg over the 0..45deg drive range (follower=-theta
    # -> gearing +1); knuckle joint axes are BOTH -x of base (not mirrored), so the
    # left bar follows +theta (gearing -1) and the right bar -theta (gearing +1).
    # v47 JOINTS discovery: the articulation parser drops the BASE->inner_knuckle
    # joints as the four-bar loop edges (they are NOT articulation joints; mimicking
    # them NaN'd v46) and keeps the *_inner_finger_pad_joints as tree edges — the
    # knuckle bars hang from the fingers. Only mimic real articulation joints.
    _gearing = {"left_inner_finger_joint": float(os.environ.get("GF_L", "1.0")),
                "right_inner_finger_joint": float(os.environ.get("GF_R", "1.0"))}
    _AXTOK = os.environ.get("AXTOK", "rotZ")   # NVIDIA authors its knuckle mimic on rotX
    _ref_path = None
    _joint_prims = {}
    for _pr in _st.Traverse():
        _nm = _pr.GetName()
        if _nm == "finger_joint" and "/Robot" in _pr.GetPath().pathString and _ref_path is None:
            _ref_path = _pr.GetPath()
        elif _nm in _gearing and "/Robot" in _pr.GetPath().pathString and _nm not in _joint_prims:
            _joint_prims[_nm] = _pr
    from pxr import UsdPhysics as _UP2
    for _nm, _pr in _joint_prims.items():
        _rj = _UP2.RevoluteJoint(_pr)
        _rj.CreateLowerLimitAttr(-50.0)
        _rj.CreateUpperLimitAttr(50.0)
        _mj = _PS.PhysxMimicJointAPI.Apply(_pr, "rotZ")
        _mj.GetReferenceJointRel().SetTargets([_ref_path])
        _mj.GetGearingAttr().Set(_gearing[_nm])
        _mj.GetOffsetAttr().Set(0.0)
        print(f"  MIMIC {_nm} <- {_ref_path} gearing={_gearing[_nm]}", flush=True)
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
print("JOINTS:", jn, flush=True)

# Capture each finger mesh's AABB in its BODY-LOCAL frame now, while USD (authored
# spawn pose) and physics agree — fabric never writes live transforms back to USD,
# so world AABBs must be reconstructed from live body poses + these local corners.
settle(2)
from isaaclab.utils.math import quat_apply_inverse as _qai, quat_apply as _qa2
def _fin_local():
    from pxr import UsdGeom as _UG, Usd as _U
    import omni.usd as _ou3
    _st3 = _ou3.get_context().get_stage()
    _cache = _UG.BBoxCache(_U.TimeCode.Default(), [_UG.Tokens.default_])
    _bn = list(fr.data.body_names)
    out = {}
    # USD-only: corners in the finger prim's own frame (physics pose never matches
    # USD after reset, so no cross-referencing between the two at capture time)
    from pxr import Usd as _U2, Gf as _Gf
    for side in ("left", "right"):
        prm = _st3.GetPrimAtPath(f"/World/envs/env_0/Robot/panda_hand/{side}_inner_finger")
        rng = _cache.ComputeWorldBound(prm).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        inv = _UG.Xformable(prm).ComputeLocalToWorldTransform(_U2.TimeCode.Default()).GetInverse()
        loc = [inv.Transform(_Gf.Vec3d(x, y, z)) for x in (mn[0], mx[0])
               for y in (mn[1], mx[1]) for z in (mn[2], mx[2])]
        out[side] = torch.tensor([[c[0], c[1], c[2]] for c in loc],
                                 device=fr.device, dtype=torch.float32)
        # collider sub-prims: where does PHYSICS think the pad geometry is, in the
        # same finger-body frame?
        for cp in ("Fingertip_01", "Finger4_01"):
            cprm = _st3.GetPrimAtPath(f"/World/envs/env_0/Robot/panda_hand/{side}_inner_finger/{cp}")
            if not cprm:
                print(f"  NO PRIM {side}/{cp}", flush=True)
                continue
            crng = _cache.ComputeWorldBound(cprm).ComputeAlignedRange()
            cc = 0.5 * (crng.GetMin() + crng.GetMax())
            lcc = inv.Transform(_Gf.Vec3d(cc[0], cc[1], cc[2]))
            print(f"  COLLIDER-LOCAL {side}/{cp}: center=({lcc[0]:+.3f},{lcc[1]:+.3f},{lcc[2]:+.3f}) "
                  f"(finger-frame; mesh spans {[f'{v:+.3f}' for v in (out[side].min(0).values.tolist())]}"
                  f"..{[f'{v:+.3f}' for v in (out[side].max(0).values.tolist())]})", flush=True)
    return out
LFIN = _fin_local()
def fin_world():
    _bn = list(fr.data.body_names)
    w = {}
    for side in ("left", "right"):
        bi = _bn.index(f"{side}_inner_finger")
        bp = fr.data.body_pos_w[0, bi]
        bq = fr.data.body_quat_w[0, bi]
        w[side] = bp + _qa2(bq.unsqueeze(0).expand(8, 4), LFIN[side])
    return w

# SERVO the wrist to the demo grip frame: approach axis HORIZONTAL along +x at a
# clear height. Three measured errors drive three joints (yaw->j1, pitch->j6,
# height->j4); the drives (holding gains) execute each iteration.
from isaaclab.utils.math import quat_apply as _qa
_arm = [0.0, -0.73, 0.0, -2.46, 0.0, 2.85, 0.72]   # env anchor; j7 from the converged roll-level servo
_bn0 = list(fr.data.body_names)
for _it in range(9):
    fr.set_joint_position_target(torch.tensor([_arm], device=fr.device), joint_ids=list(range(7)))
    settle(120)
    _bq = fr.data.body_quat_w[0, _bn0.index("robotiq_base_link")]
    _ap = _qa(_bq.unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]], device=fr.device))[0]
    _km = 0.5 * (fr.data.body_pos_w[0, _bn0.index("left_inner_finger")]
                 + fr.data.body_pos_w[0, _bn0.index("right_inner_finger")])
    _gz = float(_km[2] + 0.17 * _ap[2])
    import math as _m
    _yaw_err = _m.atan2(float(_ap[1]), float(_ap[0]))          # want approach along +x
    _pitch_err = _m.asin(max(-1, min(1, float(-_ap[2]))))      # want approach_z = 0
    _wf = fin_world()
    _d = _wf["left"].mean(0) - _wf["right"].mean(0)
    _roll_err = _m.atan2(float(_d[2]), float(_d[1]))           # want finger spread along pure y
    _z_err = _gz - 0.80
    print(f"  [servo {_it}] appr=({float(_ap[0]):+.2f},{float(_ap[1]):+.2f},{float(_ap[2]):+.2f}) "
          f"grip_z={_gz:.3f} yaw={_yaw_err:+.2f} pitch={_pitch_err:+.2f} roll={_roll_err:+.2f}", flush=True)
    if abs(_yaw_err) < 0.06 and abs(_pitch_err) < 0.06 and abs(_z_err) < 0.05:
        break
    _arm[0] -= 0.8 * _yaw_err
    _arm[5] += 0.8 * _pitch_err
    _arm[1] = max(-1.7, _arm[1] + 0.7 * _z_err)   # shoulder raises the wrist (j2 has headroom; j4 is limit-trapped)
settle(120)
print(f"SCENE open : ang={float(fr.data.joint_pos[0,fid]):+.3f} sep={sep():.4f}", flush=True)

# QUATONLY=1: dump the converged (user-approved side-grip) robotiq_base_link world
# orientation for the env's robotiq _ee_quat_des, then exit.
if bool(int(os.environ.get("QUATONLY", "0"))):
    from isaaclab.utils.math import quat_apply_inverse as _qai
    _bq = fr.data.body_quat_w[0, _bn0.index("robotiq_base_link")]
    _ap = _qa(_bq.unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]], device=fr.device))[0]
    _wf = fin_world()
    _sp = (_wf["left"].mean(0) - _wf["right"].mean(0))
    _sp = _sp / _sp.norm()
    _spl = _qai(_bq.unsqueeze(0), _sp.unsqueeze(0))[0]
    print(f"SERVO QUAT wxyz=({float(_bq[0]):+.6f},{float(_bq[1]):+.6f},"
          f"{float(_bq[2]):+.6f},{float(_bq[3]):+.6f})", flush=True)
    print(f"SERVO approach_w=({float(_ap[0]):+.4f},{float(_ap[1]):+.4f},{float(_ap[2]):+.4f}) "
          f"spread_w=({float(_sp[0]):+.4f},{float(_sp[1]):+.4f},{float(_sp[2]):+.4f}) "
          f"spread_local=({float(_spl[0]):+.4f},{float(_spl[1]):+.4f},{float(_spl[2]):+.4f})",
          flush=True)
    _ee = fr.data.body_pos_w[0, _bn0.index("robotiq_base_link")]
    print(f"SERVO ee_w=({float(_ee[0]):+.4f},{float(_ee[1]):+.4f},{float(_ee[2]):+.4f}) "
          f"arm_jpos={[round(float(v),4) for v in fr.data.joint_pos[0,:7]]}", flush=True)
    os._exit(0)

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
_eye = Gf.Vec3d(1.05, -0.85, 0.95)
_tgt = Gf.Vec3d(0.50, 0.0, 0.58)   # the servo'd side-grip zone
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
_e2, _t2 = Gf.Vec3d(1.45, 0.03, 0.70), Gf.Vec3d(0.50, 0.0, 0.58)   # down the +x approach at the pads
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
    print("BODIES:", bn3, flush=True)
    # EMPTY-CLOSE CURVE: sweep the drive target down and record ang -> tip gap.
    # The pads stop ~5 cm apart at ang 0.06, so the ang<->gap convention cannot be
    # assumed — measure it and derive kiss/squeeze angles + pocket from the curve.
    CURVE = []
    for _tgt in [0.785, 0.70, 0.60, 0.50, 0.45, 0.35, 0.25, 0.15, 0.05, 0.0]:
        fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + _tgt, joint_ids=[fid])
        settle(40)
        _wc = fin_world()
        _gap = float(_wc["left"][:, 1].min() - _wc["right"][:, 1].max())
        _tipx = float(max(_wc["left"][:, 0].max(), _wc["right"][:, 0].max()))
        _zmid = float(0.5 * (_wc["left"][:, 2].mean() + _wc["right"][:, 2].mean()))
        _ang = float(fr.data.joint_pos[0, fid])
        # pad parallelism: gap measured separately over the tip half (x-high corners)
        # and heel half (x-low) — a real 2F-140 keeps these equal through the stroke
        _xm = 0.5 * (_tipx + float(min(_wc["left"][:, 0].min(), _wc["right"][:, 0].min())))
        _lt2 = _wc["left"][_wc["left"][:, 0] > _xm]; _lh = _wc["left"][_wc["left"][:, 0] <= _xm]
        _rt2 = _wc["right"][_wc["right"][:, 0] > _xm]; _rh = _wc["right"][_wc["right"][:, 0] <= _xm]
        _gt = float(_lt2[:, 1].min() - _rt2[:, 1].max()) if len(_lt2) and len(_rt2) else float("nan")
        _gh = float(_lh[:, 1].min() - _rh[:, 1].max()) if len(_lh) and len(_rh) else float("nan")
        CURVE.append((_tgt, _ang, _gap, _tipx, _zmid))
        print(f"  CURVE tgt={_tgt:.2f} ang={_ang:+.3f} gap={_gap*1000:.1f}mm tip/heel={_gt*1000:.1f}/{_gh*1000:.1f}mm "
              f"tipx={_tipx:.3f} z={_zmid:.3f}", flush=True)
    _hi = bn3.index("robotiq_base_link")
    _hp0 = fr.data.body_pos_w[0, _hi]
    _dist = {b: float((fr.data.body_pos_w[0, bn3.index(b)] - _hp0).norm())
             for b in bn3 if b.startswith(("left_", "right_"))}
    for _b, _d in sorted(_dist.items(), key=lambda kv: -kv[1]):
        _p3 = fr.data.body_pos_w[0, bn3.index(_b)]
        print(f"  CLOSED {_b}: d={_d:.3f} pos=({float(_p3[0]):.3f},{float(_p3[1]):.3f},{float(_p3[2]):.3f})", flush=True)
    # body origins all collapse to the knuckle plane — measure the actual finger
    # MESHES instead: world AABBs of the inner fingers (pads are fixed geometry on
    # them) at the closed state give the true fingertip pocket. app.update() first
    # so fabric syncs body transforms back to USD.
    _lt, _rt = "left_inner_finger", "right_inner_finger"
    # derive grasp params from the curve: kiss = tightest state with gap > neck+4mm;
    # squeeze = state with the smallest gap; pocket at the squeeze state's tips
    _NECK = 0.016
    _kiss = min((c for c in CURVE if c[2] > _NECK + 0.004), key=lambda c: c[2], default=CURVE[0])
    _sq = min(CURVE, key=lambda c: c[2])
    KISS_ANG = _kiss[0]
    # squeeze: interpolate the curve for a 10 mm commanded gap (6 mm interference on
    # the 16 mm neck -> drive stalls on the neck with bounded grip force)
    SQ_ANG = _sq[0]
    _cs = sorted(CURVE, key=lambda c: c[2])
    for _lo, _hi in zip(_cs, _cs[1:]):
        if _lo[2] <= 0.010 <= _hi[2]:
            _f = (0.010 - _lo[2]) / max(1e-6, _hi[2] - _lo[2])
            SQ_ANG = _lo[0] + _f * (_hi[0] - _lo[0])
            break
    # the V-pocket is tightest at the very tip apex — pocket just inside it
    POCKET = torch.tensor([_sq[3], 0.0 * _sq[2], _sq[4]], device=fr.device)
    print(f"  KISS tgt={KISS_ANG:.2f} (gap {_kiss[2]*1000:.0f}mm)  SQUEEZE tgt={SQ_ANG:.2f} "
          f"(gap {_sq[2]*1000:.0f}mm)  POCKET=({float(POCKET[0]):.3f},{float(POCKET[1]):.3f},{float(POCKET[2]):.3f})", flush=True)
    if MIMIC or bool(int(os.environ.get("GRASP", "0"))):
        # drive-only grasp: kiss just above the neck, squeeze to ~12 mm commanded gap
        MUG_GRIP_Z = 0.12
        _cs2 = sorted(CURVE, key=lambda c: c[2])
        SQ_ANG = 0.785   # full-close command; the drive stalls on the neck at its
                         # effort limit -> maximum bounded pinch (mimics keep pads parallel)
        for _lo, _hi in zip(_cs2, _cs2[1:]):
            if _lo[2] <= 0.024 <= _hi[2]:
                KISS_ANG = _lo[0] + (0.024 - _lo[2]) / max(1e-6, _hi[2] - _lo[2]) * (_hi[0] - _lo[0])
                break
        print(f"  MIMIC grasp: kiss={KISS_ANG:.3f} squeeze={SQ_ANG:.3f} tip=({_sq[3]:.3f},0,{_sq[4]:.3f})", flush=True)
        # pad-assist drives must agree with the mimic: follower = -(gearing*ref)
        _PSGN = -float(os.environ.get("GF_L", "1.0"))
        for GX in (float(_sq[3]) - 0.030, float(_sq[3]) - 0.020, float(_sq[3]) - 0.010):
            fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + 0.30, joint_ids=[fid])
            settle(50)
            pose = obj.data.root_pose_w.clone()
            pose[0, 0] = GX; pose[0, 1] = 0.0; pose[0, 2] = float(_sq[4]) - MUG_GRIP_Z
            pose[0, 3] = 1.0; pose[0, 4:7] = 0.0
            PIN["pose"] = pose; PIN["obj"] = obj
            fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + KISS_ANG, joint_ids=[fid])
            fr.set_joint_position_target(torch.zeros(1, 2, device=fr.device) + _PSGN * KISS_ANG,
                                         joint_ids=[jn.index("left_inner_finger_joint"), jn.index("right_inner_finger_joint")])
            for _i in range(50):
                obj.write_root_pose_to_sim(pose)
                obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_vel_w))
                settle(1)
                if _i == 45:
                    snap(f"mg{int(GX*1000)}_pinned"); snap2(f"mg{int(GX*1000)}_pinned")
                    obj.write_root_pose_to_sim(pose)
                    obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_vel_w))
            fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + SQ_ANG, joint_ids=[fid])
            fr.set_joint_position_target(torch.zeros(1, 2, device=fr.device) + _PSGN * SQ_ANG,
                                         joint_ids=[jn.index("left_inner_finger_joint"), jn.index("right_inner_finger_joint")])
            for _i in range(60):
                obj.write_root_pose_to_sim(pose)
                obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_vel_w))
                settle(1)
            _angp = float(fr.data.joint_pos[0, fid])
            PIN["pose"] = None
            for _t, _n in ((10, 10), (50, 40), (100, 50), (200, 100), (400, 200)):
                settle(_n)
                print(f"    drop t={_t}: z={float(obj.data.root_pose_w[0, 2]):.3f} "
                      f"ang={float(fr.data.joint_pos[0, fid]):+.3f}", flush=True)
            bz1 = float(obj.data.root_pose_w[0, 2])
            held = (float(pose[0, 2]) - bz1) < 0.03
            print(f"MIMICGRIP x={GX:.3f}: stall={SQ_ANG - _angp:+.3f} obj z {float(pose[0,2]):.3f}->{bz1:.3f} held={held}", flush=True)
            snap(f"mg{int(GX*1000)}_after"); snap2(f"mg{int(GX*1000)}_after")
            if held:
                break
        os._exit(0)
    fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + 0.45, joint_ids=[fid])
    settle(80)
    MUG_GRIP_Z = 0.12
    hand_i = bn3.index("robotiq_base_link")
    from isaaclab.utils.math import quat_apply_inverse
    # CONTACT SCAN: sweep the bottle's grip height along the vertical line through the
    # PAD MIDPOINT; the close-stall angle maps where the pads are and what they touch:
    #   stall ~0.146 = pads touch each other (no bottle);  >0.15 = bottle contact;
    #   ~0.155-0.17 = neck (16-18 mm);  ~0.25+ = fat body.
    hp = fr.data.body_pos_w[0, hand_i]
    # SIDE GRIP (franka-parity): the horizontal gripper pinches the NECK from the
    # side — bottle axis PERPENDICULAR to the gripper axis. Pin the bottle upright
    # with its neck (origin+dz) at the PAD-FACE midpoint.
    # COLLIDER OCCUPANCY MAP: tips closed (ang 0.45), raster the pinned bottle
    # through the fingertip region and read its net contact force per cell — maps
    # where the physics colliders actually are (mesh corners say tips at x~0.37,
    # z~0.76, but nothing there ever touches).
    fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + 0.45, joint_ids=[fid])
    settle(60)
    _dtp = sim.get_physics_dt()
    # detector controls: palm pocket (known contact from v13) and inside the table
    for _cx, _cz, _tag in [(0.20, 0.751, "PALM-CTRL"), (0.45, 0.30, "TABLE-CTRL")]:
        pose = obj.data.root_pose_w.clone()
        pose[0, 0] = _cx; pose[0, 1] = 0.0; pose[0, 2] = _cz - MUG_GRIP_Z
        pose[0, 3] = 1.0; pose[0, 4:7] = 0.0
        for _i in range(8):
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_vel_w))
            settle(1)
        settle(2)
        _v = obj.data.root_vel_w[0]
        print(f"  {_tag}: shove={float((_v[0]**2+_v[1]**2)**0.5 + max(0.0, float(_v[2]))):.2f}", flush=True)
    # ROLL-LEVEL SERVO (empirical): raster the tip cross-section, split the
    # occupancy into +y / -y finger blobs, and roll j7 until their centroid z's
    # match. All from contact physics — no body-pose inference.
    def yz_map(step_y=0.015, step_z=0.03, at_x=0.375, z0=0.70, zspan=0.15):
        cells = []
        for _y in [round(-0.06 + step_y * i, 4) for i in range(int(0.12 / step_y) + 1)]:
            for _z in [round(z0 + step_z * j, 4) for j in range(int(zspan / step_z) + 1)]:
                pose = obj.data.root_pose_w.clone()
                pose[0, 0] = at_x; pose[0, 1] = _y; pose[0, 2] = _z - MUG_GRIP_Z
                pose[0, 3] = 1.0; pose[0, 4:7] = 0.0
                for _i in range(5):
                    obj.write_root_pose_to_sim(pose)
                    obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_vel_w))
                    settle(1)
                settle(2)
                _v = obj.data.root_vel_w[0]
                if float((_v[0] ** 2 + _v[1] ** 2) ** 0.5 + max(0.0, float(_v[2]))) > 0.05:
                    cells.append((_y, _z))
        return cells
    import math as _m2
    fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + 0.45, joint_ids=[fid])
    settle(60)
    for _rit in range(5):
        cells = yz_map()
        L = [(y, z) for (y, z) in cells if y > 0.004]
        R = [(y, z) for (y, z) in cells if y < -0.004]
        if not L or not R:
            print(f"  ROLL[{_rit}] blobs L={len(L)} R={len(R)} cells={cells} — one side missing", flush=True)
            _arm[6] += 0.35
        else:
            yL = sum(c[0] for c in L) / len(L); zL = sum(c[1] for c in L) / len(L)
            yR = sum(c[0] for c in R) / len(R); zR = sum(c[1] for c in R) / len(R)
            tilt = _m2.atan2(zL - zR, yL - yR)
            print(f"  ROLL[{_rit}] L=({yL:+.3f},{zL:.3f})x{len(L)} R=({yR:+.3f},{zR:.3f})x{len(R)} tilt={tilt:+.2f} j7={_arm[6]:+.2f}", flush=True)
            if abs(tilt) < 0.12:
                break
            _arm[6] -= 0.8 * tilt
        fr.set_joint_position_target(torch.tensor([_arm], device=fr.device), joint_ids=list(range(7)))
        settle(150)
        fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + 0.45, joint_ids=[fid])
        settle(60)
    snap("leveled"); snap2("leveled")
    # PAD-FOLLOW: the four-bar loop is dead (pads never counter-rotate; closed gap
    # floors at ~30 mm). Emulate the parallelogram by driving the pad joints with
    # the knuckle drive. Test both signs, measure the closed tip gap, grasp with
    # the better sign.
    _lpj = jn.index("left_inner_finger_joint"); _rpj = jn.index("right_inner_finger_joint")
    def close_with_pads(T, k, n=80):
        fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + T, joint_ids=[fid])
        fr.set_joint_position_target(torch.zeros(1, 2, device=fr.device) + k * T, joint_ids=[_lpj, _rpj])
        settle(n)
    def close_combo(Td, Tp, n=100):
        fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + Td, joint_ids=[fid])
        fr.set_joint_position_target(torch.zeros(1, 2, device=fr.device) + Tp, joint_ids=[_lpj, _rpj])
        settle(n)
    best = None
    for (Td, Tp) in [(0.45, 0.45), (0.45, 0.785), (0.60, 0.60), (0.60, 0.785), (0.785, 0.785)]:
        k = (Td, Tp)
        close_combo(0.10, 0.10, 60)
        close_combo(Td, Tp, 100)
        _lifa = float(fr.data.joint_pos[0, _lpj]); _rifa = float(fr.data.joint_pos[0, _rpj])
        cells = yz_map(step_y=0.004, step_z=0.02, at_x=0.355, z0=0.74, zspan=0.12)
        L = [(y, z) for (y, z) in cells if y > 0.002]
        R = [(y, z) for (y, z) in cells if y < -0.002]
        if L and R:
            yin_L = min(c[0] for c in L); yin_R = max(c[0] for c in R)
            zmid = 0.5 * (sum(c[1] for c in L) / len(L) + sum(c[1] for c in R) / len(R))
            w = (yin_L - yin_R) * 1000
            print(f"PADK d={k[0]:.2f} p={k[1]:.2f}: padang=({_lifa:+.2f},{_rifa:+.2f}) width={w:.0f}mm "
                  f"channel=({0.5*(yin_L+yin_R):+.3f},{zmid:.3f})", flush=True)
            if best is None or w < best[1]:
                best = (k, w, 0.5 * (yin_L + yin_R), zmid)
            if w < 14:
                break
        else:
            print(f"PADK d={k[0]:.2f} p={k[1]:.2f}: padang=({_lifa:+.2f},{_rifa:+.2f}) L={len(L)} R={len(R)} "
                  f"— blob missing (cells={len(cells)})", flush=True)
    if best is None:
        print("PADFOLLOW: no measurable channel", flush=True)
        os._exit(0)
    (KD, KP), W, CY, CZ = best
    print(f"PADFOLLOW chosen drive={KD:.2f} pads={KP:.2f} width={W:.0f}mm channel=({CY:+.3f},{CZ:.3f})", flush=True)
    snap("padclosed"); snap2("padclosed")
    # grasp with pad-follow, pinned DEEPER than the tip edge (the pad swing arcs
    # the contact region backward; at the tip edge the neck is only grazed)
    for GX in (0.330, 0.345):
      close_combo(KD, KP, 60)
      cells = yz_map(step_y=0.004, step_z=0.02, at_x=GX, z0=0.74, zspan=0.12)
      L = [(y, z) for (y, z) in cells if y > 0.002]
      R = [(y, z) for (y, z) in cells if y < -0.002]
      if L and R:
          CY = 0.5 * (min(c[0] for c in L) + max(c[0] for c in R))
          CZ = 0.5 * (sum(c[1] for c in L) / len(L) + sum(c[1] for c in R) / len(R))
          print(f"  x={GX:.3f} channel=({CY:+.3f},{CZ:.3f}) width={(min(c[0] for c in L)-max(c[0] for c in R))*1000:.0f}mm", flush=True)
      close_combo(0.10, 0.10, 40)
      pose = obj.data.root_pose_w.clone()
      pose[0, 0] = GX; pose[0, 1] = CY; pose[0, 2] = CZ - MUG_GRIP_Z
      pose[0, 3] = 1.0; pose[0, 4:7] = 0.0
      PIN["pose"] = pose; PIN["obj"] = obj
      fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + 0.40, joint_ids=[fid])
      fr.set_joint_position_target(torch.zeros(1, 2, device=fr.device) + 0.45, joint_ids=[_lpj, _rpj])
      for _i in range(40):
          obj.write_root_pose_to_sim(pose)
          obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_vel_w))
          settle(1)
          if _i == 35:
              snap(f"pf{int(GX*1000)}_pinned"); snap2(f"pf{int(GX*1000)}_pinned")
              obj.write_root_pose_to_sim(pose)
              obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_vel_w))
      # squeeze: pads to ~13 mm commanded gap (3 mm interference on the 16 mm neck) —
      # deeper targets back-drive the main joint over the bottle and eject it
      fr.set_joint_position_target(torch.zeros(1, 1, device=fr.device) + 0.45, joint_ids=[fid])
      fr.set_joint_position_target(torch.zeros(1, 2, device=fr.device) + 0.63, joint_ids=[_lpj, _rpj])
      for _i in range(25):
          obj.write_root_pose_to_sim(pose)
          obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_vel_w))
          settle(1)
      ang_pinned = float(fr.data.joint_pos[0, fid])
      PIN["pose"] = None
      settle(200)
      bz1 = float(obj.data.root_pose_w[0, 2])
      held = (float(pose[0, 2]) - bz1) < 0.03
      print(f"PFGRIP x={GX:.3f}: stall={KD - ang_pinned:+.3f} obj z {float(pose[0,2]):.3f}->{bz1:.3f} held={held}", flush=True)
      snap(f"pf{int(GX*1000)}_after"); snap2(f"pf{int(GX*1000)}_after")
      if held:
          break

os._exit(0)
