#!/usr/bin/env python3
"""
eval_video.py — Run FORGE policy in FrankaInsertionEnv and record eval_episode.mp4.

Pipeline:
  1. Start Xvfb (:1)
  2. Boot IsaacSim (headless, DISPLAY=:1)
  3. Import Isaac Lab (after SimulationApp is live)
  4. Create FrankaInsertionEnv (DirectRLEnv, real physics)
  5. Load ForceConditionedPolicy from checkpoint
  6. Run 200-step rollout, capture frames via omni.replicator
  7. Encode frames → docs/eval_episode.mp4 with ffmpeg
"""
import os, sys, subprocess, time, shutil

# ── 0. Paths ────────────────────────────────────────────────────────────────
REPO_ROOT   = "/workspace/FORGE-plus"
CHECKPOINT  = f"{REPO_ROOT}/checkpoints/task1_franka_panda.pt"
OUTPUT_MP4  = f"{REPO_ROOT}/docs/eval_episode.mp4"
FRAMEDIR    = "/workspace/frames_eval"
FRAMES      = 200
FPS         = 24
F_CMD       = 50.0          # force command fed to FiLM conditioning

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, "/workspace/IsaacLab/source/isaaclab")
sys.path.insert(0, "/workspace/IsaacLab/source/isaaclab_tasks")
sys.path.insert(0, "/workspace/IsaacLab/source/isaaclab_assets")

# ── 1. Xvfb ────────────────────────────────────────────────────────────────
print("[eval_video] Starting Xvfb :1 ...")
xvfb = subprocess.Popen(
    ["Xvfb", ":1", "-screen", "0", "1920x1080x24"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(2)
os.environ["DISPLAY"] = ":1"
print("[eval_video] DISPLAY=:1")

# ── 2. SimulationApp (must happen BEFORE any omni/isaac imports) ─────────────
print("[eval_video] Booting SimulationApp ...")
from isaacsim import SimulationApp
app = SimulationApp({
    "headless": True,
    "width":  1280,
    "height": 720,
    
})
print("[eval_video] SimulationApp ready.")

# ── 3. Isaac Lab / forge_plus imports (after app) ───────────────────────────
import torch
import numpy as np

# forge_plus
from forge_plus.skills.policy_network import ForceConditionedPolicy, PolicyConfig
from forge_plus.envs.base_assembly_env import EnvObservation, TaskPhase

# ── 4. Load checkpoint ───────────────────────────────────────────────────────
print(f"[eval_video] Loading checkpoint: {CHECKPOINT}")
ckpt = torch.load(CHECKPOINT, map_location="cuda:0", weights_only=False)
# Checkpoint format: {'policy_state_dict': ..., 'normalizer': ..., 'policy_cfg': ...}
if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
    state_dict = ckpt["policy_state_dict"]
    print("[eval_video] Loaded policy_state_dict from checkpoint.")
elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    state_dict = ckpt["model_state_dict"]
elif isinstance(ckpt, dict) and "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    state_dict = ckpt

policy_cfg = PolicyConfig()
policy = ForceConditionedPolicy(policy_cfg).cuda()
# Load - use non-strict since architecture attrs may differ slightly
missing, unexpected = policy.load_state_dict(state_dict, strict=False)
print(f"[eval_video] Checkpoint loaded. Missing: {missing}, Unexpected: {unexpected}")
policy.eval()

# ── 5. Create FrankaInsertionEnv ──────────────────────────────────────────────
print("[eval_video] Creating FrankaInsertionEnv ...")
from forge_plus.isaac_insertion_env import FrankaInsertionEnv, InsertionEnvCfg

env_cfg = InsertionEnvCfg()
env_cfg.scene.num_envs = 1
env_cfg.decimation = 6
env = FrankaInsertionEnv(cfg=env_cfg, render_mode=None)
print("[eval_video] Env created. Resetting ...")
obs_dict, _ = env.reset()
print("[eval_video] Reset done.")

# ── 6. Camera setup via omni.replicator ────────────────────────────────────
import omni.replicator.core as rep

# Create a render product for frame capture
camera = rep.create.camera(
    position=(1.2, -0.8, 1.4),
    look_at=(0.5, 0.0, 0.45),
)
rp = rep.create.render_product(camera, (1280, 720))
rgb_annot = rep.annotators.get("rgb")
rgb_annot.attach([rp])

os.makedirs(FRAMEDIR, exist_ok=True)
# Clear any old frames
for f in os.listdir(FRAMEDIR):
    os.remove(os.path.join(FRAMEDIR, f))

# ── 7. Rollout ────────────────────────────────────────────────────────────
def encode_obs_to_tensor(obs_dict: dict) -> torch.Tensor:
    """Extract 34-dim policy obs from env dict."""
    return obs_dict["policy"]  # (1, 34)

def obs_tensor_to_f_cmd(obs_t: torch.Tensor) -> float:
    """Use fixed force command for FiLM conditioning."""
    return F_CMD

print(f"[eval_video] Running {FRAMES}-step rollout ...")
for step in range(FRAMES):
    obs_t = encode_obs_to_tensor(obs_dict)  # (1, 34) on cuda

    with torch.no_grad():
        f_cmd_t = torch.tensor([[obs_tensor_to_f_cmd(obs_t) / 120.0]], device="cuda:0")
        action_t, _ = policy.get_action(obs_t, f_cmd_t, deterministic=True)  # (1,7)

    # Step env
    obs_dict, reward, terminated, truncated, info = env.step(action_t)

    # Capture frame
    rep.orchestrator.step(rt_subframes=1, pause_timeline=True)  # PATCH: trigger replicator render
    frame_data = rgb_annot.get_data()
    if frame_data is not None and len(frame_data) > 0:
        arr = np.frombuffer(frame_data, dtype=np.uint8).reshape(720, 1280, 4)
        rgb = arr[:, :, :3]  # drop alpha
        frame_path = os.path.join(FRAMEDIR, f"frame_{step:05d}.png")
        import cv2
        cv2.imwrite(frame_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    else:
        # Blank frame fallback
        arr = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        frame_path = os.path.join(FRAMEDIR, f"frame_{step:05d}.png")
        import cv2
        cv2.imwrite(frame_path, arr)

    if (step + 1) % 20 == 0:
        print(f"  step {step+1}/{FRAMES}")

    if terminated.any() or truncated.any():
        print(f"[eval_video] Episode done at step {step+1}.")
        # Fill remaining frames with last frame
        for s2 in range(step + 1, FRAMES):
            shutil.copy(frame_path, os.path.join(FRAMEDIR, f"frame_{s2:05d}.png"))
        break

env.close()
print("[eval_video] Rollout complete.")

# ── 8. Encode to MP4 ──────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_MP4), exist_ok=True)
print(f"[eval_video] Encoding frames → {OUTPUT_MP4} ...")
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-framerate", str(FPS),
    "-i", os.path.join(FRAMEDIR, "frame_%05d.png"),
    "-vcodec", "libx264",
    "-pix_fmt", "yuv420p",
    "-crf", "22",
    OUTPUT_MP4,
]
result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
if result.returncode != 0:
    print("ffmpeg stderr:", result.stderr[-2000:])
    sys.exit(1)

size_mb = os.path.getsize(OUTPUT_MP4) / 1e6
print(f"[eval_video] Done! {OUTPUT_MP4}  ({size_mb:.1f} MB)")

# Brightness check
frames_list = sorted(os.listdir(FRAMEDIR))
if frames_list:
    import cv2
    sample = cv2.imread(os.path.join(FRAMEDIR, frames_list[len(frames_list)//2]))
    mean_bright = sample.mean() if sample is not None else 0
    print(f"[eval_video] Brightness check (mid-frame mean): {mean_bright:.1f}")
    if mean_bright < 50 or mean_bright > 230:
        print("[eval_video] WARNING: brightness out of range 50-230!")
    else:
        print("[eval_video] Brightness OK.")

app.close()
xvfb.terminate()
print("[eval_video] All done.")
