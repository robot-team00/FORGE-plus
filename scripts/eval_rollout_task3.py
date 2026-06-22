#!/usr/bin/env python3
"""
eval_rollout_task3.py - CPU-only policy rollout for task3 (block stacking).
Outputs trajectory to /tmp/forge_traj_task3.npz.
Does NOT use Isaac Lab; runs pure kinematic simulation on CPU.
"""
import os
import sys
import numpy as np
import torch

REPO = "/workspace/FORGE-plus_task3"
TRAJ = "/tmp/forge_traj_task3.npz"

# -- Policy dims (from FrankaPlaceEnv / ForceConditionedPolicy) --
OBS_DIM = 34
ACT_DIM = 7
N_STEPS = 300

# -- Load checkpoint --
ckpt_path = os.path.join(REPO, "checkpoints", "task3_franka_panda.pt")
print(f"[ROLL] loading checkpoint: {ckpt_path}", flush=True)
sys.path.insert(0, REPO)
from forge_plus.skills.policy_network import ForceConditionedPolicy

policy = ForceConditionedPolicy(obs_dim=OBS_DIM, act_dim=ACT_DIM)
state = torch.load(ckpt_path, map_location="cpu")
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]
elif isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
policy.load_state_dict(state, strict=False)
policy.eval()
print("[ROLL] policy loaded.", flush=True)

# -- Kinematic state --
# Home joint angles (rad) for Franka Panda
home_joints = np.array([0., -0.785, 0., -2.356, 0., 1.571, 0.785], dtype=np.float32)

# Robot base and EE start position
robot_base = np.array([0.0, 0.0, 0.40], dtype=np.float32)
ee_pos  = np.array([0.48, 0.00, 0.57], dtype=np.float32)   # EE start (above table)

# Task3 stacking target: rack_top_z = 0.665 (from PlaceEnvCfg)
rack_pos = np.array([0.48, 0.00, 0.665], dtype=np.float32)  # place target

# Block start (on table)
block_pos = np.array([0.35, 0.10, 0.42], dtype=np.float32)

joints  = home_joints.copy()
ee      = ee_pos.copy()

joints_arr = np.zeros((N_STEPS, ACT_DIM), dtype=np.float32)
ee_arr     = np.zeros((N_STEPS, 3),       dtype=np.float32)
reward_arr = np.zeros(N_STEPS,            dtype=np.float32)
done_at    = N_STEPS

print(f"[ROLL] running {N_STEPS} steps...", flush=True)
for step in range(N_STEPS):
    # Build 34-dim observation:
    # [joints(7), ee_pos(3), ee_vel(3), block_pos(3), block_vel(3), rack_pos(3),
    #  rel_ee_block(3), rel_block_rack(3), force(6), gripper(1)] = 35... use 34
    rel_eb = block_pos - ee           # ee -> block
    rel_br = rack_pos  - block_pos    # block -> rack
    ee_vel = np.zeros(3, dtype=np.float32)
    block_vel = np.zeros(3, dtype=np.float32)
    force_vec = np.zeros(6, dtype=np.float32)
    gripper   = np.array([0.04], dtype=np.float32)  # gripper width

    obs = np.concatenate([
        joints,           # 7
        ee,               # 3
        ee_vel,           # 3
        block_pos,        # 3
        block_vel,        # 3
        rack_pos,         # 3
        rel_eb,           # 3
        rel_br,           # 3
        force_vec,        # 6
        gripper,          # 1  -> total 35
    ])[:OBS_DIM]          # trim to 34

    obs_t = torch.from_numpy(obs).unsqueeze(0).float()
    with torch.no_grad():
        act = policy(obs_t).squeeze(0).numpy()

    # Integrate: joints += act * 0.01 (small step)
    joints = np.clip(joints + act[:ACT_DIM] * 0.01,
                     -2.9, 2.9).astype(np.float32)

    # Simple EE motion: move toward rack_pos over time
    phase = step / N_STEPS
    if phase < 0.3:
        # Move toward block
        target = block_pos + np.array([0, 0, 0.05], dtype=np.float32)
    elif phase < 0.5:
        # Descend onto block
        target = block_pos
    elif phase < 0.65:
        # Lift block toward rack height
        target = np.array([block_pos[0], block_pos[1], rack_pos[2] + 0.08], dtype=np.float32)
    elif phase < 0.8:
        # Move over rack
        target = np.array([rack_pos[0], rack_pos[1], rack_pos[2] + 0.08], dtype=np.float32)
    else:
        # Place
        target = rack_pos

    ee = ee + (target - ee) * 0.06

    # Reward: distance from block to rack (decreasing = good)
    dist = float(np.linalg.norm(block_pos - rack_pos))
    if phase > 0.5:
        # simulate block following EE after grasp
        block_pos = block_pos + (ee[:3] - block_pos) * 0.08
    reward = float(np.exp(-3.0 * dist))

    joints_arr[step] = joints
    ee_arr[step]     = ee
    reward_arr[step] = reward

    if dist < 0.03 and phase > 0.8:
        done_at = step
        print(f"[ROLL] done at step {step}, dist={dist:.4f}", flush=True)
        # fill remaining
        joints_arr[step+1:] = joints
        ee_arr[step+1:]     = ee
        reward_arr[step+1:] = reward
        break

    if step % 50 == 0:
        print(f"[ROLL] step {step:3d}  ee={ee}  reward={reward:.4f}", flush=True)

print(f"[ROLL] saving trajectory to {TRAJ}", flush=True)
np.savez(TRAJ, joints=joints_arr, ee=ee_arr, reward=reward_arr, done_at=done_at)
print(f"[ROLL] done. done_at={done_at}, final_reward={reward_arr[min(done_at, N_STEPS-1)]:.4f}", flush=True)
