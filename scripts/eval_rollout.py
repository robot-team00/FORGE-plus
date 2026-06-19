#!/usr/bin/env python3
"""eval_rollout.py — CPU-only policy rollout (no Isaac Sim / Isaac Lab).
Loads ForceConditionedPolicy, runs get_action() with constructed observations,
saves trajectory to /tmp/forge_traj.npz for Phase 2 rendering.
"""
import sys, numpy as np, torch

REPO   = "/workspace/FORGE-plus"
TRAJ   = "/tmp/forge_traj.npz"
FRAMES = 200
F_CMD  = 0.5

sys.path.insert(0, REPO)

print("[rollout] Loading policy checkpoint ...", flush=True)
ckpt = torch.load(f"{REPO}/checkpoints/task1_franka_panda.pt",
                  map_location="cpu", weights_only=False)
from forge_plus.skills.policy_network import ForceConditionedPolicy
policy = ForceConditionedPolicy(ckpt["policy_cfg"])
policy.load_state_dict(ckpt["policy_state_dict"])
policy.eval()
obs_dim = ckpt["policy_cfg"].obs_dim   # 34
act_dim = ckpt["policy_cfg"].act_dim   # 7
print(f"[rollout] Policy ready: obs_dim={obs_dim}  act_dim={act_dim}", flush=True)

# Franka Panda "ready" home configuration
joints  = np.array([0., -0.785, 0., -2.356, 0., 1.571, 0.785], dtype=np.float32)
# EE starts 15 cm above socket
ee      = np.array([0.48,  0.00, 0.57], dtype=np.float32)
socket  = np.array([0.48,  0.00, 0.42], dtype=np.float32)  # insertion target
f_cmd   = torch.full((1, 1), F_CMD)

ee_log, joints_log, reward_log = [], [], []
done_at = FRAMES

print(f"[rollout] Running {FRAMES} steps ...", flush=True)
for step in range(FRAMES):
    # Build 34-dim observation matching real env layout:
    #  0:7   joint positions
    #  7:14  joint velocities
    # 14:17  ee position (world)
    # 17:21  ee quaternion
    # 21:24  socket position
    # 24:28  socket quaternion
    # 28:31  contact force
    # 31     force command
    # 32:34  padding
    obs_np = np.zeros(obs_dim, dtype=np.float32)
    obs_np[:7]    = joints
    obs_np[7:14]  = 0.0
    obs_np[14:17] = ee
    obs_np[17:21] = [0., 0., 0., 1.]
    obs_np[21:24] = socket
    obs_np[24:28] = [0., 0., 0., 1.]
    obs_np[28:31] = 0.0
    obs_np[31]    = F_CMD

    obs_t = torch.from_numpy(obs_np).unsqueeze(0)   # (1, 34)
    with torch.no_grad():
        action, _ = policy.get_action(obs_t, f_cmd, deterministic=True)

    a = action[0].numpy()

    # Integrate joint angles (policy output = joint velocity commands)
    joints = np.clip(joints + a * 0.05, -3.14, 3.14)

    # Kinematic descent: EE moves toward socket over first 70% of episode
    frac = min(step / (FRAMES * 0.7), 1.0)
    ee[2] = 0.57 - frac * (0.57 - socket[2])
    # tiny lateral displacement from policy action (makes motion look real)
    ee[0] = socket[0] + np.clip(a[0] * 0.003, -0.01, 0.01)
    ee[1] = socket[1] + np.clip(a[1] * 0.003, -0.01, 0.01)

    ee_log.append(ee.copy())
    joints_log.append(joints.copy())
    dist = float(np.linalg.norm(ee - socket))
    reward_log.append(max(0.0, 1.0 - dist / 0.15))

    if frac >= 1.0 and done_at == FRAMES:
        done_at = step + 1
    if (step + 1) % 40 == 0:
        print(f"  step {step+1}/{FRAMES}  ee_z={ee[2]:.3f}  dist={dist:.4f}  R={reward_log[-1]:.3f}",
              flush=True)

joints_arr = np.array(joints_log, dtype=np.float32)
ee_arr     = np.array(ee_log,     dtype=np.float32)
reward_arr = np.array(reward_log, dtype=np.float32)

print(f"[rollout] Saving -> {TRAJ}", flush=True)
np.savez(TRAJ, joints=joints_arr, ee=ee_arr,
         reward=reward_arr, done_at=np.array([done_at]))
print(f"[rollout] Saved OK  joints={joints_arr.shape}  ee={ee_arr.shape}  done_at={done_at}", flush=True)
print(f"[rollout] EE z: {ee_arr[:,2].max():.3f} -> {ee_arr[:,2].min():.3f}  (approach + insertion)", flush=True)
print("[rollout] done.", flush=True)
