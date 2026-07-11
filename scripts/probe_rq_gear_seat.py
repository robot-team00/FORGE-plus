#!/usr/bin/env python3
"""Grasp-calibration probe: Robotiq 2F-140 TOP-DOWN grip on the FORGE gear hub.

Port handoff step 3. Standalone scene (no env staging) so every phase is
directly controlled. Phases:
  A. spawn at a candidate top-down arm pose (base 0,0,0 — franka-identical
     scene geometry), grid-scan (j2, j4) with j6 = pi - j2 + j4 (hand-down
     family, j1=j3=j5=0) for robotiq_base_link at TARGET (x~0.31, z=Z_EE)
  B. seat sequence at the winner: pin gear at grasp point (EE + 0.214 down,
     origin GRIP_H below), slow finger ramp -> kiss angle (first contact) and
     stall angle on the Ø35.5 hub, then full-close squeeze, unpin, settle
  C. hold check: pad_dz, tilt, grip force, 240-substep static hold
  D. carry check: lift the arm target, gear must follow rigidly
  E. SLIP-TILT probe: +5 mm in-grip y-kick — does the 2F-140 pinch keep the
     tilt (franka: 7-10 deg permanent) or re-level? Decides whether the
     franka recovery routing (hover->regrasp) transfers.

    export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 \
           PYTHONPATH=/workspace/FORGE-plus_task3
    /workspace/.venv/bin/python scripts/probe_rq_gear_seat.py
Env knobs: GRIP_H (default 0.040), Z_EE (default 0.70), SLIP_MM (default 5).
"""
import os, sys, math
os.environ.setdefault("HOME", "/workspace/persist/ovhome")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("DISPLAY", ":99")
sys.path.insert(0, "/workspace/FORGE-plus_task3")

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab.utils.math import matrix_from_quat

GRIP_H = float(os.environ.get("GRIP_H", "0.040"))
Z_EE   = float(os.environ.get("Z_EE", "0.70"))
SLIP_MM = float(os.environ.get("SLIP_MM", "5.0"))
TCP = 0.214

sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, render_interval=4, device="cuda:0"))
scene = InteractiveScene(InteractiveSceneCfg(num_envs=1, env_spacing=4.0,
                                             replicate_physics=False))

fr_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
fr_cfg.spawn.usd_path = ("/workspace/assets/isaac51/Robots/FrankaRobotics/"
                         "FrankaPanda/franka_robotiq_2f140.usd")
fr_cfg.spawn.activate_contact_sensors = True
fr_cfg.init_state.pos = (0.0, 0.0, 0.0)
# Spawn stays at the asset's parse pose (side grip) — the probe TELEPORTS the
# arm (arm-only joint write, the retrain-proven safe mechanism) to candidates.
READY = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
jp = {k: v for k, v in fr_cfg.init_state.joint_pos.items()
      if not k.startswith("panda_finger")}
jp.update({"finger_joint": 0.0, ".*_inner_finger_joint": 0.0,
           ".*_inner_finger_pad_joint": 0.0, ".*_outer_.*_joint": 0.0})
fr_cfg.init_state.joint_pos = jp
acts = dict(fr_cfg.actuators)
acts.pop("panda_hand", None)
# ENV-parity actuators (isaac_gear_env robotiq block)
acts["gripper_drive"] = ImplicitActuatorCfg(
    joint_names_expr=["finger_joint"], effort_limit_sim=30.0,
    velocity_limit_sim=1.0, stiffness=11.25, damping=0.1, friction=0.0, armature=0.0)
acts["gripper_finger"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_joint"], effort_limit_sim=0.01,
    velocity_limit_sim=1.0, stiffness=0.0, damping=0.001, friction=0.0, armature=0.0)
acts["gripper_passive"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_pad_joint", ".*_outer_finger_joint",
                      "right_outer_knuckle_joint"],
    effort_limit_sim=1.0, velocity_limit_sim=2.0, stiffness=0.0,
    damping=0.0, friction=0.0, armature=0.0)
fr_cfg.actuators = acts
fr = Articulation(fr_cfg)
scene.articulations["robot"] = fr

# NO table in this probe: several grid candidates sweep the fingers through
# table altitude and the contact explosion poisons the articulation (run 1:
# j5 wound to -155 rad, 34 kN phantom grip). The gear is pinned in air.
gear = RigidObject(RigidObjectCfg(
    prim_path="/World/envs/env_.*/Object",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/assets/factory/gear_medium_body.usda",
        scale=(1.0, 1.0, 1.0), activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.030),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False, disable_gravity=False,
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005, rest_offset=0.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=False)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.31, 0.0, 0.50))))
scene.rigid_objects["gear"] = gear

csens = ContactSensor(ContactSensorCfg(
    prim_path="/World/envs/env_.*/Robot/panda_hand/(left|right)_(inner|outer)_finger",
    update_period=0.0, history_length=1, track_air_time=False,
    filter_prim_paths_expr=["/World/envs/env_.*/Object"]))
scene.sensors["contact"] = csens

# rubber pads + gear (env parity)
import omni.usd
from pxr import UsdShade, UsdPhysics, PhysxSchema
stage = omni.usd.get_context().get_stage()
mat = UsdShade.Material.Define(stage, "/World/RubberPadMat")
UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
m = UsdPhysics.MaterialAPI(mat.GetPrim())
m.CreateStaticFrictionAttr(2.0); m.CreateDynamicFrictionAttr(2.0)
nrb = 0
for prim in stage.Traverse():
    pth = prim.GetPath().pathString
    if prim.HasAPI(UsdPhysics.CollisionAPI) and (
            ("Robot" in pth and "inner_finger" in pth) or "/Object" in pth):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose="physics")
        nrb += 1
obj_prim = stage.GetPrimAtPath("/World/envs/env_0/Object")
if obj_prim and obj_prim.IsValid():
    api = PhysxSchema.PhysxContactReportAPI.Apply(obj_prim)
    api.CreateThresholdAttr(0.0)
print(f"[cal] rubber on {nrb} prims", flush=True)

sim.reset()
print("[cal] sim up", flush=True)

bn = list(fr.data.body_names)
jn = list(fr.data.joint_names)
EE = bn.index("robotiq_base_link")
LF, RF = bn.index("left_inner_finger"), bn.index("right_inner_finger")
FJ = jn.index("finger_joint")
ARM = list(range(7))
N = 1
dev = fr.device

_arm_cmd = torch.tensor(READY, device=dev).unsqueeze(0)
_fj_cmd = torch.zeros(N, 1, device=dev)
PIN = {"pose": None}

def teleport_arm(cand):
    """Arm-only joint write (safe: gripper subtree moves rigidly, loop intact)
    + matching position targets so the holding gains keep it there."""
    t = torch.tensor(cand, device=dev).unsqueeze(0)
    fr.write_joint_state_to_sim(t, torch.zeros_like(t), joint_ids=ARM)
    _arm_cmd[0] = t[0]

def step(n=1):
    for _ in range(n):
        fr.set_joint_position_target(_arm_cmd, joint_ids=ARM)
        fr.set_joint_position_target(_fj_cmd, joint_ids=[FJ])
        if PIN["pose"] is not None:
            gear.write_root_pose_to_sim(PIN["pose"])
            gear.write_root_velocity_to_sim(torch.zeros_like(gear.data.root_vel_w))
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt=sim.get_physics_dt())

def ee_state():
    p = fr.data.body_pos_w[0, EE]
    q = fr.data.body_quat_w[0, EE]
    ax = matrix_from_quat(q.unsqueeze(0))[0, :, 2]   # hand local +z in world
    return p, q, ax

def settle(vmax=0.02, max_n=1500, chunk=30):
    """Step until the arm is quiescent. Pose-B-family configs carry ~0.27 rad
    of gravity deflection and take FAR longer than 90-120 substeps to reach
    drive equilibrium — run 7's Newton 'converged' on mid-transient readings
    and the arm kept drifting for hundreds of substeps after the pin."""
    n = 0
    while n < max_n:
        step(chunk)
        n += chunk
        if float(fr.data.joint_vel[0, :7].abs().max()) < vmax:
            break
    return n

def gear_tilt():
    q = gear.data.root_pose_w[0, 3:7]
    R = matrix_from_quat(q.unsqueeze(0))[0]
    upz = float(R[2, 2].clamp(-1, 1))
    return math.degrees(math.acos(upz))

def grip_force():
    f = csens.data.net_forces_w
    return 0.0 if f is None else float(f[0].norm(dim=-1).sum())

def report(tag):
    p, q, ax = ee_state()
    g = gear.data.root_pose_w[0]
    sep = float((fr.data.body_pos_w[0, LF] - fr.data.body_pos_w[0, RF]).norm())
    ang = float(fr.data.joint_pos[0, FJ])
    grasp_z = float(p[2] + TCP * ax[2])
    pad_dz = grasp_z - float(g[2])
    print(f"[cal:{tag}] ee=({float(p[0]):.3f},{float(p[1]):.3f},{float(p[2]):.3f}) "
          f"ax=({float(ax[0]):+.3f},{float(ax[1]):+.3f},{float(ax[2]):+.3f}) "
          f"gear=({float(g[0]):.3f},{float(g[1]):.3f},{float(g[2]):.4f}) "
          f"tilt={gear_tilt():.1f}deg ang={ang:+.4f} fsep={sep:.4f} "
          f"pad_dz={pad_dz:+.4f} gripF={grip_force():.2f}N", flush=True)

# park the gear far from the workspace while scanning (no table to rest on)
far = gear.data.root_pose_w.clone()
far[0, 0] = 1.5; far[0, 1] = 1.5; far[0, 2] = 0.5
far[0, 3] = 1.0; far[0, 4:7] = 0.0
PIN["pose"] = far
step(60)
report("spawn")

# ── Phase A: TELEPORT grid scan for the top-down pose at (x~0.31, z=Z_EE) ──
best = None
def try_pose(j2, j4, tag=""):
    global best
    j6 = math.pi - j2 + j4
    if not (-0.0174 < j6 < 3.75):
        return
    cand = [0.0, j2, 0.0, j4, 0.0, j6, 0.785]
    teleport_arm(cand)
    settle()
    jv = float(fr.data.joint_vel[0, :7].abs().max())
    jerr = float((fr.data.joint_pos[0, :7]
                  - torch.tensor(cand, device=dev)).abs().max())
    p, q, ax = ee_state()
    err = abs(float(p[2]) - Z_EE) + 2.0 * abs(float(ax[2]) + 1.0) \
        + 0.3 * max(0.0, abs(float(p[0]) - 0.31) - 0.06)
    print(f"[scanA{tag}] j2={j2:+.3f} j4={j4:+.3f} j6={j6:+.3f} "
          f"ee=({float(p[0]):.3f},{float(p[1]):.3f},{float(p[2]):.3f}) "
          f"axz={float(ax[2]):+.3f} jv={jv:.2f} jerr={jerr:.3f} err={err:.4f}",
          flush=True)
    if jv > 1.0 or jerr > 0.25:
        return   # unstable / sagging candidate — don't trust the measurement
    if best is None or err < best["err"]:
        best = {"err": err, "cand": cand, "p": p.clone(), "q": q.clone()}

_seed = os.environ.get("WINNER", "")
if _seed:
    v = [float(x) for x in _seed.split(",")]
    b = [0.0, v[0], 0.0, v[1], 0.0, v[2], 0.785]
else:
    for j2 in (-0.2, -0.35, -0.5, -0.65, -0.785):
        for j4 in (-1.7, -1.9, -2.1, -2.3, -2.5):
            try_pose(j2, j4)
    b = list(best["cand"])

# NEWTON SERVO on the COMMAND (droop included): iterate (j2, j4, j6) so the
# REACHED state hits x=0.31, z=Z_EE, pitch=0 (hand axis straight down). The
# grid winner sagged 20 deg off vertical (run 2) — sag breaks the analytic
# j2-j4+j6=pi identity, so measure the true Jacobian by finite differences.
def measure(cmd):
    teleport_arm(cmd)
    settle()
    p, _, ax = ee_state()
    pitch = math.atan2(float(ax[0]), -float(ax[2]))
    return float(p[0]) - 0.31, float(p[2]) - Z_EE, pitch

for it in range(8):
    ex, ez, ep = measure(b)
    print(f"[newton {it}] cmd j2={b[1]:+.4f} j4={b[3]:+.4f} j6={b[5]:+.4f} "
          f"ex={ex:+.4f} ez={ez:+.4f} pitch={math.degrees(ep):+.2f}deg", flush=True)
    if abs(ex) < 0.01 and abs(ez) < 0.008 and abs(ep) < 0.017:
        break
    J = []
    for ji in (1, 3, 5):
        bp = list(b)
        bp[ji] += 0.03
        exp_, ezp_, epp_ = measure(bp)
        J.append([(exp_ - ex) / 0.03, (ezp_ - ez) / 0.03, (epp_ - ep) / 0.03])
    Jm = torch.tensor(J, dtype=torch.float64).T   # rows: (x,z,pitch), cols: (j2,j4,j6)
    try:
        dq = torch.linalg.solve(Jm, -torch.tensor([ex, ez, ep], dtype=torch.float64))
    except Exception:
        dq = -torch.linalg.pinv(Jm) @ torch.tensor([ex, ez, ep], dtype=torch.float64)
    dq = dq.clamp(-0.15, 0.15)
    b[1] += float(dq[0]); b[3] += float(dq[1]); b[5] += float(dq[2])

teleport_arm(b)
settle()
p, q, ax = ee_state()
print(f"[cal] WINNER cmd={[round(v, 4) for v in b]} "
      f"ee=({float(p[0]):.4f},{float(p[1]):.4f},{float(p[2]):.4f}) "
      f"quat_wxyz=({float(q[0]):.5f},{float(q[1]):.5f},{float(q[2]):.5f},{float(q[3]):.5f}) "
      f"axz={float(ax[2]):+.4f} "
      f"jpos={[round(float(v), 4) for v in fr.data.joint_pos[0, :7]]}", flush=True)

# ── Phase A2: POSE B — grasp point over the SHAFT at entrance altitude ─────
# Solved BEFORE seating (gear still far-pinned) because the Newton teleports
# would drop a held gear. The staging redesign drives A -> B on stiff joint
# drives (the OSC free-space hold never converges — stage probes 1-6).
SHAFT = (0.449, 0.119)
# EE z for the PINNED gear at 0.438 (grasp 0.214 + grip_h 0.040 below the EE):
# 13 mm above the shaft top (0.425) — clear of tip interpenetration at unpin,
# inside the franka hand-off band (0.42-0.47); the policy descends the rest.
Z_B = 0.692
J1_B = math.atan2(SHAFT[1], SHAFT[0])
R_B = math.hypot(SHAFT[0], SHAFT[1])

def measure_b(cmd):
    teleport_arm(cmd)
    settle()
    p, _, ax = ee_state()
    r = math.hypot(float(p[0]), float(p[1]))
    pitch = math.atan2(math.hypot(float(ax[0]), float(ax[1])), -float(ax[2]))
    return r - R_B, float(p[2]) - Z_B, pitch

bB = list(b)
bB[0] = J1_B
for it in range(8):
    er, ez, ep = measure_b(bB)
    print(f"[newtonB {it}] j2={bB[1]:+.4f} j4={bB[3]:+.4f} j6={bB[5]:+.4f} "
          f"er={er:+.4f} ez={ez:+.4f} pitch={math.degrees(ep):+.2f}deg", flush=True)
    if abs(er) < 0.002 and abs(ez) < 0.004 and abs(ep) < 0.012:
        break
    J = []
    for ji in (1, 3, 5):
        bp = list(bB)
        bp[ji] += 0.03
        erp, ezp, epp = measure_b(bp)
        J.append([(erp - er) / 0.03, (ezp - ez) / 0.03, (epp - ep) / 0.03])
    Jm = torch.tensor(J, dtype=torch.float64).T
    try:
        dq = torch.linalg.solve(Jm, -torch.tensor([er, ez, ep], dtype=torch.float64))
    except Exception:
        dq = -torch.linalg.pinv(Jm) @ torch.tensor([er, ez, ep], dtype=torch.float64)
    dq = dq.clamp(-0.15, 0.15)
    bB[1] += float(dq[0]); bB[3] += float(dq[1]); bB[5] += float(dq[2])
teleport_arm(bB)
settle()
p, q, ax = ee_state()
print(f"[cal] POSE-B cmd={[round(v, 4) for v in bB]} "
      f"ee=({float(p[0]):.4f},{float(p[1]):.4f},{float(p[2]):.4f}) "
      f"axz={float(ax[2]):+.4f} "
      f"jpos={[round(float(v), 4) for v in fr.data.joint_pos[0, :7]]}", flush=True)

# STAGING v2: the seat happens AT pose B (over the shaft) — carrying the
# held gear through a joint-space hop tilts it in-grip 7-12 deg permanently
# (run 6: tilt 7.9 deg with the hand dead-vertical after the servo leveled
# it). Seat-position initialization is FORGE's own protocol; no transport.
# The arm is already at B from the Newton — stay and seat here.

# ── Phase B: GRIP_H SWEEP — map the clean seat zone at pose B ─────────────
# Three heights gave three modes (0.040 over-bite 61N/5deg; 0.036 taper-eject
# F->0; A-geometry clean 19N) with overlapping pad_dz readings — the pad gap
# varies along the finger, so map it empirically: full seat cycle per height.
settle()
p, q, ax = ee_state()
grasp = p + TCP * ax
results = []
for h in (0.030, 0.033, 0.036, 0.039, 0.042):
    # reset: fingers open, gear far
    _fj_cmd[0, 0] = 0.0
    far2 = gear.data.root_pose_w.clone()
    far2[0, 0] = 1.5; far2[0, 1] = 1.5; far2[0, 2] = 0.5
    far2[0, 3] = 1.0; far2[0, 4:7] = 0.0
    PIN["pose"] = far2
    step(180)
    ang0 = float(fr.data.joint_pos[0, FJ])
    if abs(ang0) > 0.05:
        print(f"[sweep h={h:.3f}] four-bar did not reopen (ang {ang0:+.3f}) — SKIP",
              flush=True)
        results.append((h, None))
        continue
    pin_pose = gear.data.root_pose_w.clone()
    pin_pose[0, 0] = grasp[0]; pin_pose[0, 1] = grasp[1]
    pin_pose[0, 2] = grasp[2] - h
    pin_pose[0, 3] = 1.0; pin_pose[0, 4:7] = 0.0
    PIN["pose"] = pin_pose
    step(60)
    kiss_h = None
    for i in range(150):
        tgt = min(0.785, 0.005 * (i + 1))
        _fj_cmd[0, 0] = tgt
        step(6)
        angi = float(fr.data.joint_pos[0, FJ])
        if kiss_h is None and (grip_force() > 0.5 or (tgt - angi) > 0.06):
            kiss_h = angi
    _fj_cmd[0, 0] = 0.785
    step(120)
    stall_h = float(fr.data.joint_pos[0, FJ])
    f_pin = grip_force()
    tilt_pin = gear_tilt()
    PIN["pose"] = None
    step(240)
    g = gear.data.root_pose_w[0]
    held_h = abs(float(g[2]) - float(pin_pose[0, 2])) < 0.03 and gear_tilt() < 3.0
    print(f"[sweep h={h:.3f}] kiss={kiss_h if kiss_h else -1:.4f} "
          f"stall={stall_h:.4f} Fpin={f_pin:.1f} tilt_pin={tilt_pin:.2f} "
          f"-> unpin: held={held_h} tilt={gear_tilt():.2f} F={grip_force():.1f} "
          f"gear_z={float(g[2]):.4f}", flush=True)
    results.append((h, (kiss_h, stall_h, f_pin, tilt_pin, held_h,
                        gear_tilt(), grip_force())))

# pick the best height: held, low tilt, sane force band
best_h, kiss_ang, stall_ang, held = None, None, 0.0, False
for h, r in results:
    if r is None:
        continue
    _, st, fp, tp, hd, tl, fh = r
    if hd and tl < 1.5 and 8.0 < fh < 40.0:
        best_h, kiss_ang, stall_ang, held = h, r[0], st, True
        break
print(f"[cal] SWEEP BEST GRIP_H={best_h}", flush=True)
if best_h is None:
    print("CAL_FAIL no clean grip height at pose B", flush=True)
    os._exit(0)
# re-seat at the winner so the slip phase runs on the final grip
_fj_cmd[0, 0] = 0.0
step(180)
pin_pose = gear.data.root_pose_w.clone()
pin_pose[0, 0] = grasp[0]; pin_pose[0, 1] = grasp[1]
pin_pose[0, 2] = grasp[2] - best_h
pin_pose[0, 3] = 1.0; pin_pose[0, 4:7] = 0.0
PIN["pose"] = pin_pose
step(60)
for i in range(150):
    _fj_cmd[0, 0] = min(0.785, 0.005 * (i + 1))
    step(6)
_fj_cmd[0, 0] = 0.785
step(120)
PIN["pose"] = None
step(240)
report("held")
gz_pin = float(pin_pose[0, 2])
held = abs(float(gear.data.root_pose_w[0, 2]) - gz_pin) < 0.03 and gear_tilt() < 5.0
if not held:
    print(f"CAL_FAIL re-seat at best height failed", flush=True)
    os._exit(0)

# ── Phase E FIRST (level hand): slip-tilt — the recovery-routing question ──
# (run 3 did the crude-carry first: the j2 trim pitched the hand 11 deg and
# contaminated the slip reading. The eval's kick fires with the hand level.)
spread = fr.data.body_pos_w[0, LF] - fr.data.body_pos_w[0, RF]
spread = spread / spread.norm()
print(f"[cal] finger spread dir (lf-rf, world): "
      f"({float(spread[0]):+.3f},{float(spread[1]):+.3f},{float(spread[2]):+.3f})",
      flush=True)
for axis, name in ((1, "y"), (0, "x")):
    tilt_pre = gear_tilt()
    g_pre = gear.data.root_pose_w[0, :3].clone()
    print(f"[cal] SLIP: +{SLIP_MM} mm {name} kick in-grip (tilt_pre={tilt_pre:.2f})",
          flush=True)
    sp = gear.data.root_pose_w.clone()
    sp[0, axis] += SLIP_MM / 1000.0
    gear.write_root_pose_to_sim(sp)
    for k in range(8):
        step(30)
        g = gear.data.root_pose_w[0]
        print(f"[slip-{name} {30*(k+1):3d}] gear=({float(g[0]):.4f},{float(g[1]):.4f},"
              f"{float(g[2]):.4f}) tilt={gear_tilt():.2f}deg F={grip_force():.2f}",
              flush=True)
    dp = gear.data.root_pose_w[0, :3] - g_pre
    print(f"[cal] slip-{name}: tilt {tilt_pre:.2f} -> {gear_tilt():.2f} deg, "
          f"residual d=({float(dp[0])*1000:+.1f},{float(dp[1])*1000:+.1f},"
          f"{float(dp[2])*1000:+.1f}) mm "
          f"(franka reference: 7-10 deg permanent)", flush=True)

# STAGING v2: no hop, no servo — the gear was seated AT pose B. Final gate:
g = gear.data.root_pose_w[0]
gdx = (float(g[0]) - SHAFT[0]) * 1000
gdy = (float(g[1]) - SHAFT[1]) * 1000
print(f"[cal] AT-B FINAL: gear=({float(g[0]):.4f},{float(g[1]):.4f},{float(g[2]):.4f}) "
      f"offset=({gdx:+.1f},{gdy:+.1f})mm tilt={gear_tilt():.2f}deg "
      f"F={grip_force():.2f}", flush=True)
report("staging_final")
ok_stage = held and abs(gdx) < 4.0 and abs(gdy) < 4.0 and gear_tilt() < 3.0
print(f"CAL_{'PASS' if (held and ok_stage) else 'FAIL'} kiss={kiss_ang} "
      f"stall={stall_ang:.4f} poseB_cmd={[round(v, 4) for v in bB]} "
      f"poseB_jpos={[round(float(v), 4) for v in fr.data.joint_pos[0, :7]]}",
      flush=True)
os._exit(0)
