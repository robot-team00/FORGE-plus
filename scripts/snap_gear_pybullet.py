#!/usr/bin/env python3
"""CPU close-up stills of the gear grasp — PyBullet TinyRenderer, no RTX.

The pod's RTX view draws a live-tracking GHOST of the gear (displaced ~2 cm,
tilted ~35 deg) while buffer == raw PhysX == upright (dual-source probe).
This renders the TRUE recorded physics state: real Panda URDF posed from the
recorded joints, gear built from exact-dimension primitives (Ø41.9x10 teeth
disc + Ø35.5x30 hub, origin 5 mm below the bottom face), plate + shafts.

    /workspace/.venv/bin/python scripts/snap_gear_pybullet.py \
        --states /workspace/gear_states.npz --outdir /workspace/snapshots_gear
"""
import argparse
import math
import os

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

W, H = 1280, 720

ap = argparse.ArgumentParser()
ap.add_argument("--states", default="/workspace/gear_states.npz")
ap.add_argument("--outdir", default="/workspace/snapshots_gear")
args = ap.parse_args()

d = np.load(args.states)
ks, joints, gear, orig = d["ks"], d["joints"], d["gear"], d["orig"]
os.makedirs(args.outdir, exist_ok=True)

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
arm_j = [i for i in range(p.getNumJoints(robot))
         if p.getJointInfo(robot, i)[2] != p.JOINT_FIXED]
print("movable joints:", [(i, p.getJointInfo(robot, i)[1].decode()) for i in arm_j])

# plate + shafts (base at rack (0.449,0.120); plate top 0.405, shaft top 0.425)
RX, RY = 0.449, 0.120
PLATE_TOP = 0.405
plate_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.085, 0.055, 0.0035],
                                rgbaColor=[0.55, 0.55, 0.57, 1])
p.createMultiBody(0, baseVisualShapeIndex=plate_vis,
                  basePosition=[RX - 0.019, RY, PLATE_TOP - 0.0035])
for sx, sr, sh in [(RX, 0.005, 0.020),           # middle shaft (the target)
                   (RX - 0.075, 0.007, 0.045),   # side pegs (context)
                   (RX + 0.055, 0.007, 0.045)]:
    v = p.createVisualShape(p.GEOM_CYLINDER, radius=sr, length=sh,
                            rgbaColor=[0.5, 0.5, 0.52, 1])
    p.createMultiBody(0, baseVisualShapeIndex=v,
                      basePosition=[sx, RY, PLATE_TOP + sh / 2])

# gear from exact dims: origin 5 mm below bottom face; teeth 0.005-0.015;
# hub 0.015-0.045 above origin
teeth_v = p.createVisualShape(p.GEOM_CYLINDER, radius=0.02095, length=0.010,
                              rgbaColor=[0.35, 0.37, 0.42, 1])
hub_v = p.createVisualShape(p.GEOM_CYLINDER, radius=0.01775, length=0.030,
                            rgbaColor=[0.45, 0.47, 0.52, 1])
bore_v = p.createVisualShape(p.GEOM_CYLINDER, radius=0.0052, length=0.0405,
                             rgbaColor=[0.12, 0.12, 0.14, 1])
teeth_b = p.createMultiBody(0, baseVisualShapeIndex=teeth_v)
hub_b = p.createMultiBody(0, baseVisualShapeIndex=hub_v)
bore_b = p.createMultiBody(0, baseVisualShapeIndex=bore_v)


def q_wxyz_to_pb(q):
    return [q[1], q[2], q[3], q[0]]


def set_gear(pose):
    pos, q = pose[:3], q_wxyz_to_pb(pose[3:7])
    for body, dz in ((teeth_b, 0.010), (hub_b, 0.030), (bore_b, 0.0205)):
        off, _ = p.multiplyTransforms(pos, q, [0, 0, dz], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(body, off, q)


def shoot(path, eye, tgt, fov=32):
    vm = p.computeViewMatrix(eye, tgt, [0, 0, 1])
    pm = p.computeProjectionMatrixFOV(fov, W / H, 0.01, 10)
    img = p.getCameraImage(W, H, vm, pm, renderer=p.ER_TINY_RENDERER,
                           lightDirection=[0.5, -0.8, 1.4])[2]
    Image.fromarray(np.reshape(img, (H, W, 4))[..., :3]).save(path)
    print("saved", path, flush=True)


names = {170: "carry", 205: "press", 220: "seatpress"}
for i, k in enumerate(ks):
    jp = joints[i]
    for idx, ji in enumerate(arm_j):
        p.resetJointState(robot, ji, float(jp[idx]))
    set_gear(gear[i] - np.concatenate([orig[i], np.zeros(4)]))
    tag = names.get(int(k), str(int(k)))
    gx, gy, gz = (gear[i] - np.concatenate([orig[i], np.zeros(4)]))[:3]
    # side close-up at hub height + a 3/4 view
    shoot(f"{args.outdir}/pb_{tag}_side.png",
          [gx + 0.02, gy - 0.30, gz + 0.05], [gx, gy, gz + 0.03])
    shoot(f"{args.outdir}/pb_{tag}_three4.png",
          [gx + 0.22, gy - 0.20, gz + 0.16], [gx, gy, gz + 0.02])
print("PB_DONE", flush=True)
