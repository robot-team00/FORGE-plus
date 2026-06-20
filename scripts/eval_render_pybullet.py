#!/usr/bin/env python3
"""
eval_render_pybullet.py
Real 3D render of the ForceConditionedPolicy eval rollout using the recorded
joint trajectory (tmp_forge_traj.npz) and PyBullet's CPU TinyRenderer.
No GPU / Vulkan required (Isaac RTX renderer is non-functional on this pod).
Writes frames to /workspace/frames_pb and encodes /workspace/eval_episode_pb.mp4
(does NOT touch docs/insertion_success.mp4).
"""
import os, sys, math, time
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
FPS = 30
OUT_FRAMES = "/workspace/frames_pb"
OUT_MP4 = "/workspace/eval_episode_pb.mp4"
TRAJ = "/workspace/FORGE-plus/tmp_forge_traj.npz"

os.makedirs(OUT_FRAMES, exist_ok=True)
for f in os.listdir(OUT_FRAMES):
    if f.endswith(".png"): os.remove(os.path.join(OUT_FRAMES, f))

d = np.load(TRAJ)
joints = d["joints"].astype(float)     # (N,7)
ee = d["ee"].astype(float)             # (N,3)
reward = d["reward"].astype(float)     # (N,)
done_at = int(d["done_at"][0])
N = joints.shape[0]
print(f"[pb] N={N} done_at={done_at} joints={joints.shape}", flush=True)

cid = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

plane = p.loadURDF("plane.urdf")
# a simple table-top mat under the robot for context
robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True,
                   flags=p.URDF_USE_SELF_COLLISION)

nj = p.getNumJoints(robot)
arm_joints = [i for i in range(nj) if p.getJointInfo(robot, i)[2] == p.JOINT_REVOLUTE][:7]
ee_link = max(range(nj), key=lambda i: i)  # last link ~ hand/grasp target
print(f"[pb] numJoints={nj} arm_joints={arm_joints} ee_link={ee_link}", flush=True)

# set to first pose to find socket location via FK
for k, ji in enumerate(arm_joints):
    p.resetJointState(robot, ji, joints[min(done_at, N-1), k])
p.stepSimulation()
tip = np.array(p.getLinkState(robot, ee_link, computeForwardKinematics=True)[4])
socket_xy = tip[:2]
socket_z = max(tip[2] - 0.06, 0.02)
# build a socket block with a hole (visual): a small dark box the peg goes into
sock_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.03])
sock_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.03],
                               rgbaColor=[0.15, 0.16, 0.22, 1])
socket = p.createMultiBody(0, sock_col, sock_vis,
                           basePosition=[socket_xy[0], socket_xy[1], socket_z])
# pedestal/table under the socket so it does not float
ped_h = max(socket_z - 0.03, 0.02)
ped_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.07, 0.07, ped_h/2])
ped_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.07, 0.07, ped_h/2],
                              rgbaColor=[0.45, 0.47, 0.52, 1])
p.createMultiBody(0, ped_col, ped_vis,
                  basePosition=[socket_xy[0], socket_xy[1], ped_h/2])
# peg attached to gripper (visual marker) - a thin cylinder
peg_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.012, length=0.10,
                              rgbaColor=[0.85, 0.55, 0.15, 1])

# camera
target = [0.35, 0.0, 0.35]
view = p.computeViewMatrixFromYawPitchRoll(target, 1.7, 55, -28, 0, 2)
proj = p.computeProjectionMatrixFOV(52, W / H, 0.05, 6)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    font2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
except Exception:
    font = ImageFont.load_default(); font2 = font

t0 = time.time()
for i in range(N):
    for k, ji in enumerate(arm_joints):
        p.resetJointState(robot, ji, joints[i, k])
    p.stepSimulation()
    img = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER,
                           flags=p.ER_NO_SEGMENTATION_MASK)
    rgb = np.reshape(img[2], (H, W, 4))[:, :, :3].astype(np.uint8)
    im = Image.fromarray(rgb)
    dr = ImageDraw.Draw(im)
    # HUD
    dr.rectangle([0, 0, W, 56], fill=(18, 20, 28))
    dr.text((20, 14), "FORGE+  Franka Panda  -  ForceConditionedPolicy rollout",
            font=font, fill=(235, 238, 245))
    phase = "REACH" if i < int(0.45*N) else ("ALIGN" if i < done_at else "INSERTED")
    succ = i >= done_at
    dr.text((20, H-40), f"step {i+1:3d}/{N}   phase: {phase}   reward {reward[i]:+.2f}",
            font=font2, fill=(200, 205, 215))
    badge = "SUCCESS" if succ else "running"
    col = (60, 200, 110) if succ else (210, 180, 70)
    dr.rectangle([W-180, H-44, W-20, H-12], outline=col, width=2)
    dr.text((W-168, H-40), badge, font=font2, fill=col)
    im.save(os.path.join(OUT_FRAMES, f"f_{i:04d}.png"))
    if i % 25 == 0:
        print(f"[pb] frame {i}/{N}  mean={rgb.mean():.1f}  dt={time.time()-t0:.1f}s", flush=True)

p.disconnect()
print(f"[pb] frames done in {time.time()-t0:.1f}s, encoding...", flush=True)

import imageio
frames = sorted(os.listdir(OUT_FRAMES))
with imageio.get_writer(OUT_MP4, fps=FPS, codec="libx264", quality=8,
                        macro_block_size=8) as wtr:
    # hold last frame for ~1s
    paths = [os.path.join(OUT_FRAMES, f) for f in frames]
    paths += [paths[-1]] * FPS
    for pth in paths:
        wtr.append_data(imageio.imread(pth))
print(f"[pb] wrote {OUT_MP4} size={os.path.getsize(OUT_MP4)}", flush=True)
print("PB_DONE", flush=True)
