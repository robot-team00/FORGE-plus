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
# NOTE 2 — TELEPORT CONTRACT: the 2F-140 closes each finger's four-bar with
# maximal-coordinate loop joints (inner_knuckle_joints, excludeFromArticulation=1).
# On this PhysX build those loop constraints DO NOT survive articulation joint-state
# teleports (write_joint_state_to_sim) — verified in probe_rq_isolate tests A–D: any
# teleport (even arm-only, gripper shape unchanged) leaves the linkage in a flipped/
# degenerate branch and the pads collapse. The asset works when: (a) it spawns at its
# authored default pose (parse-consistent), (b) it is driven only by forces/position
# TARGETS afterwards, and (c) see NOTE 3. The env honors this for gripper
# "robotiq_2f140": no joint-state writes, grip/open via finger_joint drive targets,
# and the FORGE setup drive (OSC) takes the arm from the default pose to the hand-off.
#
# NOTE 3 — THE PARSE GHOST (PhysX quirk, empirically airtight): the merged gripper's
# excluded loop joints are only MATERIALIZED by the physics parser when a STANDALONE
# articulation instance of the same gripper USD also exists in the scene. Without it
# the loop constraints silently never exist and the four-bar collapses at spawn
# (5/5 broken solo-free runs vs 3/3 healthy runs with the ghost — probe_rq_isolate
# probe_ghost run flipped ONLY this variable). The env therefore spawns a "parse
# ghost": one gravity-free 2F-140 articulation parked far outside the workspace.
# It is invisible to the camera, never touched, and costs ~10 bodies of sim.
CFG_USDA = """#usda 1.0
(
    defaultPrim = "panda"
)

def Xform "panda"
{
    def Xform "Robotiq_2F_140_edit" (
        delete apiSchemas = ["PhysicsArticulationRootAPI"]
        prepend payload = @../../../Robotiq/2F-140/Robotiq_2F_140_physics_edit.usd@
    )
    {
        quatd xformOp:orient = (0, 0.9238795325112867, 0.3826834323650898, 0)
        double3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0.08799996972084045, 0, 0.9259999394416809)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]

        over "robotiq_base_link"
        {
            def PhysicsFixedJoint "AssemblerFixedJoint"
            {
                rel physics:body0 = </panda/panda_hand>
                rel physics:body1 = </panda/Robotiq_2F_140_edit/robotiq_base_link>
                point3f physics:localPos0 = (0, 0, 0)
                point3f physics:localPos1 = (0, 0, 0)
                quatf physics:localRot0 = (1, 0, 0, 0)
                quatf physics:localRot1 = (1, 0, 0, 0)
            }
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

    over "panda_hand"
    {
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
        over "geometry" (
            active = false
        )
        {
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
