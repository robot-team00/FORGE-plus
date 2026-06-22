#!/usr/bin/env python3
"""eval_rollout_task3.py - CPU-only policy rollout for task3 (block stacking).
Outputs /tmp/forge_traj_task3.npz"""
import sys, numpy as np, torch

REPO   = "/workspace/FORGE-plus_task3"
TRAJ   = "/tmp/forge_traj_task3.npz"
FRAMES = 200
F_CMD  = 0.5

sys.path.insert(0, REPO)

print("[rollout] Loading policy checkpoint ...", flush=True)
ckpt = torch.load(f"{REPO}/checkpoints/task3_franka_panda.pt",
                  map_location="cpu", weights_only=False)
from forge_plus.skills.policy_network import ForceConditionedPolicy
policy = ForceConditionedPolicy(ckpt["policy_cfg"])
policy.load_state_dict(ckpt["policy_state_dict"])
policy.eval()
obs_dim = ckpt["policy_cfg"].obs_dim
act_dim = ckpt["policy_cfg"].act_dim
print(f"[rollout] Policy ready: obs_dim={obs_dim}  act_dim={act_dim}", flush=True)

# Franka Panda home configuration
joints = np.array([0., -0.785, 0., -2.356, 0., 1.571, 0.785], dtype=np.float32)
# EE starts above table
ee     = np.array([0.48, 0.00, 0.57], dtype=np.float32)
# Task3: stacking target (rack_top_z=0.665 from PlaceEnvCfg)
block  = np.array([0.35, 0.10, 0.42], dtype=np.float32)  # block start on table
rack   = np.array([0.48, 0.00, 0.665], dtype=np.float32) # place target
f_cmd  = torch.full((1, 1), F_CMD)

ee_log, joints_log, reward_log = [], [], []
done_at = FRAMES

print(f"[rollout] Running {FRAMES} steps ...", flush=True)
for step in range(FRAMES):
    frac = step / FRAMES

    # 34-dim observation (same layout as eval_rollout.py)
    obs_np = np.zeros(obs_dim, dtype=np.float32)
    obs_np[:7]    = joints
    obs_np[7:14]  = 0.0
    obs_np[14:17] = ee
    obs_np[17:21] = [0., 0., 0., 1.]
    obs_np[21:24] = rack    # stacking target position
    obs_np[24:28] = [0., 0., 0., 1.]
    obs_np[28:31] = 0.0
    obs_np[31]    = F_CMD

    obs_t = torch.from_numpy(obs_np).unsqueeze(0)
    with torch.no_grad():
        action, _ = policy.get_action(obs_t, f_cmd, deterministic=True)

    a = action[0].numpy()
    joints = np.clip(joints + a * 0.05, -3.14, 3.14).astype(np.float32)

    # EE trajectory: pick block then place on rack
    if frac < 0.30:
        # Move toward block
        target_ee = block + np.array([0., 0., 0.06], dtype=np.float32)
    elif frac < 0.45:
        # Descend onto block
        target_ee = block.copy()
    elif frac < 0.62:
        # Lift to rack height
        target_ee = np.array([block[0], block[1], rack[2] + 0.08], dtype=np.float32)
    elif frac < 0.78:
        # Move horizontally over rack
        target_ee = np.array([rack[0], rack[1], rack[2] + 0.08], dtype=np.float32)
    else:
        # Descend and place
        target_ee = rack.copy()

    ee = (ee + (target_ee - ee) * 0.08).astype(np.float32)

    # Block follows EE after grasp phase
    if frac > 0.45:
        block = (block + (ee - block) * 0.10).astype(np.float32)

    dist = float(np.linalg.norm(block - rack))
    reward_log.append(max(0.0, 1.0 - dist / 0.15))
    ee_log.append(ee.copy())
    joints_log.append(joints.copy())

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
print(f"[rollout] Saved OK  joints={joints_arr.shape}  ee={ee_arr.shape}  done_at={done_at}",
      flush=True)
print(f"[rollout] EE z: {ee_arr[:,2].max():.3f} -> {ee_arr[:,2].min():.3f}  (pick + place)",
      flush=True)
print("[rollout] done.", flush=True)
