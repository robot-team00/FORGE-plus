#!/usr/bin/env python3
"""Author the Franka Panda + Robotiq 2F-140 robot USD.

NVIDIA ships a Franka 'Gripper' variant only for the Robotiq 2F-85
(configuration/franka_Gripper_Robotiq_2F_85.usd). This script replicates that exact
attachment recipe for the 2F-140, from the locally mirrored 5.1 asset trees
(/workspace/assets/isaac51/Robots — see scripts/probe_robotiq_asset.py history):

  1. configuration/franka_Gripper_Robotiq_2F_140.usd — payloads the 2F-140
     physics_edit under /panda, deletes its ArticulationRootAPI (merges it into the
     panda articulation), poses it at the panda_hand flange, fixed-joints
     panda_hand -> robotiq_base_link, and deactivates the panda fingers/hand geometry.
  2. franka_robotiq_2f140.usd — a standalone root: references franka.usd</panda>
     (Gripper variant 'None') and sublayers the config. The mirrored franka.usd is
     NOT modified.

The panda_hand rigid body REMAINS (only its geometry/finger joints deactivate), so
the env's EE frame ("panda_hand") and OSC are unchanged across grippers.

Run:  /workspace/.venv/bin/python scripts/build_franka_robotiq_2f140.py
(The .usd files are written as usda TEXT — .usd is format-sniffed, text is legal.
Verify composition afterwards with scripts/probe_built_asset.py.)
"""
import os

FP_DIR = "/workspace/assets/isaac51/Robots/FrankaRobotics/FrankaPanda"
CFG_PATH = f"{FP_DIR}/configuration/franka_Gripper_Robotiq_2F_140.usd"
ROOT_PATH = f"{FP_DIR}/franka_robotiq_2f140.usd"

# ── 1. The configuration layer (attachment recipe, modeled on the 2F-85 one) ──
# The gripper root pose matches the panda_hand flange in the default joint pose:
# translate (0.088, 0, 0.926); quat (0, 0.92388, 0.38268, 0) = Rz(45 deg) * Rx(180 deg)
# (the panda hand flange is rotated 45 deg about Z and the gripper hangs downward).
# The fixed joint has localPos/localRot = identity on both sides, so PhysX binds the
# robotiq_base_link frame rigidly to the panda_hand frame.
# NOTE 1: a payload maps the target layer's defaultPrim ONTO the holder prim, so the
# 2F-140's children (robotiq_base_link, finger_joint, ...) land DIRECTLY under
# Robotiq_2F_140_edit — no inner "Robotiq_2F_140" level (NVIDIA's 2F-85 file has an
# extra nesting level; the 140 physics_edit does not). All overs are one level up,
# and the ArticulationRootAPI delete + flange pose sit on the holder prim itself.
#
# NOTE 2 — RECIPE v3 (NVIDIA's own 2F-140 attachment pattern, from
# ur10e/configuration/ur10e_Gripper_2F_140.usd): the 2F-85-style recipe (separate
# holder prim + AssemblerFixedJoint) produced a subtly broken mechanism (loop joints
# needing a "parse ghost", tearing under teleports/sag, one finger never assembling
# — see docs/task3/08 sections 2-5). NVIDIA attaches the 140 differently:
#   * payload the gripper ONTO the arm's own EE-link prim (here panda_hand), and
#     delete BOTH articulation APIs there (a stray PhysxArticulationAPI on the old
#     holder was likely the parser confusion all along);
#   * panda_hand also stops being a rigid body (APIs deleted) — it becomes the
#     container prim, exactly like NVIDIA's massless ur10e ee_link;
#   * NO added fixed joint: the arm's own panda_hand_joint is RETARGETED to
#     physics:body1 = panda_hand/robotiq_base_link with UNCHANGED local anchors
#     (the 2F-85 recipe proved base_link frame == panda_hand frame), so the gripper
#     base becomes the arm's distal link in a clean articulation tree;
#   * the stock four-bar (loop joints, pads, springs) is left fully intact.
# ENV NOTE: panda_hand is no longer a BODY — the robotiq EE body is
# "robotiq_base_link" (same frame). The parse ghost should no longer be needed.
CFG_USDA = """#usda 1.0
(
    defaultPrim = "panda"
)

def Xform "panda"
{
    over "panda_hand" (
        delete apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI", "PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", "PhysicsMassAPI"]
        prepend payload = @../../../Robotiq/2F-140/Robotiq_2F_140_physics_edit.usd@
    )
    {
        over "geometry" (
            active = false
        )
        {
        }
        over "panda_finger_joint1" (
            active = false
        )
        {
        }
        over "panda_finger_joint2" (
            active = false
        )
        {
        }
        over "finger_joint"
        {
            float state:angular:physics:position = 0
            float state:angular:physics:velocity = 0
        }
        over "right_outer_knuckle_joint"
        {
            float state:angular:physics:position = 0
            float state:angular:physics:velocity = 0
        }
        over "left_inner_knuckle_joint"
        {
            float state:angular:physics:position = 0
            float state:angular:physics:velocity = 0
        }
        over "right_inner_knuckle_joint"
        {
            float state:angular:physics:position = 0
            float state:angular:physics:velocity = 0
        }
        over "left_outer_finger_joint"
        {
            float state:angular:physics:position = 0
            float state:angular:physics:velocity = 0
        }
        over "right_outer_finger_joint"
        {
            float state:angular:physics:position = 0
            float state:angular:physics:velocity = 0
        }
        over "left_inner_finger_joint"
        {
            float state:angular:physics:position = 0
            float state:angular:physics:velocity = 0
        }
        over "right_inner_finger_joint"
        {
            float state:angular:physics:position = 0
            float state:angular:physics:velocity = 0
        }
        over "left_inner_finger_pad_joint"
        {
            float state:angular:physics:position = 0
            float state:angular:physics:velocity = 0
        }
        over "right_inner_finger_pad_joint"
        {
            float state:angular:physics:position = 0
            float state:angular:physics:velocity = 0
        }
    }

    over "panda_link7"
    {
        over "panda_hand_joint"
        {
            rel physics:body1 = </panda/panda_hand/robotiq_base_link>
        }
    }

    over "panda_leftfinger" (
        active = false
    )
    {
    }

    over "panda_rightfinger" (
        active = false
    )
    {
    }
}
"""

# ── 2. The standalone root USD (franka.usd untouched; Gripper variant 'None') ──
ROOT_USDA = """#usda 1.0
(
    defaultPrim = "panda"
    metersPerUnit = 1
    upAxis = "Z"
    subLayers = [
        @./configuration/franka_Gripper_Robotiq_2F_140.usd@
    ]
)

def Xform "panda" (
    prepend references = @./franka.usd@</panda>
    variants = {
        string Gripper = "None"
        string Mesh = "Performance"
    }
)
{
}
"""

for path, text in [(CFG_PATH, CFG_USDA), (ROOT_PATH, ROOT_USDA)]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    print("wrote %s (%d bytes)" % (path, os.path.getsize(path)), flush=True)
print("done — verify with scripts/probe_built_asset.py", flush=True)
