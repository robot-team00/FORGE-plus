# Headless Isaac Sim RTX rendering on the FORGE-plus pod

**Status: WORKING** (fixed 2026-06-20, pod `kmvgljds0ebce8`, NVIDIA RTX 2000 Ada / sm_89, Isaac Sim 5.1.0).

This pod can now produce a photorealistic, RTX path-traced render of the Franka Panda
eval rollout: `docs/eval_episode.mp4`. This doc explains how, and the four
independent bugs that had to be fixed so nobody re-breaks them.

---

## TL;DR runbook

```bash
cd /workspace/FORGE-plus_main      # or your task clone
git pull                          # get latest scripts + this doc
bash scripts/setup_runtime.sh     # restore libEGL (Vulkan) + shader cache + Xvfb :1  (idempotent)
DISPLAY=:1 /workspace/.venv/bin/python scripts/render_eval_video.py   # -> docs/eval_episode.mp4
```

- First render after a pod restart is **slow (~8 min)**: Isaac compiles the full RTX
  pipeline shaders once. The cache persists under `.venv`, so later renders are fast.
- Verify Vulkan any time with: `DISPLAY=:1 vulkaninfo --summary | grep deviceName`
  (should print "NVIDIA RTX 2000 Ada Generation").

---

## What does NOT survive a pod restart (and how setup_runtime.sh fixes it)

| Thing | Where it lives | Survives restart? | Restored by |
|---|---|---|---|
| GLVND libs (libEGL.so.1 ...) | `/usr/lib` (apt) | NO (ephemeral) | `setup_runtime.sh` (apt install) |
| Isaac shader cache | `.venv/.../extscache` | YES (/workspace) | `fetch_shadercache.py` if ever missing |
| Script fixes + this doc | `/workspace/FORGE-plus` | YES (/workspace) | git |
| Xvfb display :1 | process | NO | `setup_runtime.sh` |

---

## The four root causes (all fixed; do not regress)

### 1. Vulkan was dead - missing GLVND `libEGL.so.1`
Symptom: at boot `[gpu.foundation.plugin] Cannot load shader file 'rtx/system/GenerateMipMap.comp.hlsl'`,
then every captured frame logs `[VID] frame N EMPTY buffer`. `vulkaninfo` reports
`vkCreateInstance ... ERROR_INCOMPATIBLE_DRIVER`. nvidia-smi and CUDA work fine.

Diagnosis: `vk_icdNegotiateLoaderICDInterfaceVersion` on `libGLX_nvidia.so.0` returns
-3 (VK_ERROR_INITIALIZATION_FAILED). `strace` shows the driver searching for `libEGL.so.1`
everywhere and not finding it - the container base image lacked GLVND. (The system Vulkan
loader version, libvulkan1 1.3.204, was a red herring; it is fine.)

Fix: `apt-get install -y libegl1 libglvnd0 libgl1 libglx0 libopengl0 libegl1-mesa`.
Then negotiate returns 0 and `vulkaninfo` enumerates the RTX 2000 Ada.

### 2. Incomplete shader cache (gpu_foundation stub)
Symptom: same "Cannot load shader file" even after Vulkan works; renderer logs
`Shader caches are missing from the application`.

Diagnosis: of the three shadercache extensions, only `omni.hydra.rtx.shadercache.vulkan`
(308 MB) was populated. `omni.gpu_foundation.shadercache.vulkan` was a STUB - its 19
cache files (8 compiled `.v` shaders + `cache/shadercache/common/version`) were listed
in the wheel RECORD but never written to disk (truncated install).
(`omni.rtx.shadercache.vulkan` being a stub is NORMAL - it is just a back-compat bundle.)

Fix: `scripts/fetch_shadercache.py` extracts just those files from the 3 GB
`isaacsim-extscache-kit` wheel on `pypi.nvidia.com` via HTTP range requests
(the server supports `accept-ranges: bytes`, so no full download). Idempotent.

### 3. Capture returned empty buffers
Symptom: Vulkan + shaders fine, GPU at 100%, but `rgb.get_data()` returns empty forever.

Diagnosis: `render_eval_video.py` originally avoided `rep.orchestrator.step()`
(a workaround from when RTX hung). With RTX working, that is exactly what prevents the
Replicator annotator from ever capturing - bare `app.update()` does not run the SDG graph.

Fix: call `rep.orchestrator.step(rt_subframes=12)` before each `rgb.get_data()`.

### 4. Camera at the floor + wrong Franka asset
Symptoms: (a) a perfectly rendered but EMPTY grey frame; (b) table/peg render but no arm.

Diagnosis: (a) `rep.create.camera(rotation=(-25,0,0))` looks ~straight down (Omniverse
cameras default to -Z); (b) `franka_visuals.usd` contains materials but NO mesh geometry.

Fix: (a) use `rep.create.camera(position=(1.7,-1.9,1.45), look_at=(0,0,0.62), focal_length=22)`;
(b) point `FRANKA_USD` at `assets/franka/franka.usd` (real meshes via `Props/`), and
reference it under a parent `Xform` (`/World/Station_XX/FrankaRoot/Franka`) so the animation
transforms go on the parent - referencing `franka.usd` directly clashes with its own root
xform ops (`xformOp:translate already exists`).

---

## Files

| File | Purpose |
|---|---|
| `scripts/render_eval_video.py` | The working RTX renderer -> `docs/eval_episode.mp4` |
| `scripts/setup_runtime.sh` | Run after every pod restart (GLVND + shader cache + Xvfb) |
| `scripts/fetch_shadercache.py` | Restore the gpu_foundation shader cache from the wheel |
| `scripts/eval_render_pybullet.py` | No-GPU fallback (PyBullet) -> `docs/eval_episode_pybullet.mp4` |
| `scripts/eval_rollout.py` | CPU policy rollout -> `tmp_forge_traj.npz` (the trajectory rendered) |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `vkCreateInstance ... ERROR_INCOMPATIBLE_DRIVER` | libEGL.so.1 missing | `bash scripts/setup_runtime.sh` (or apt install libegl1 ...) |
| `Cannot load shader file ...GenerateMipMap...` | Vulkan dead OR shader cache stub | setup_runtime.sh (covers both) |
| `Shader caches are missing from the application` | gpu_foundation cache stub | `/workspace/.venv/bin/python scripts/fetch_shadercache.py` |
| `[VID] frame N EMPTY buffer` repeating | no `rep.orchestrator.step()` OR Vulkan dead | check both fixes #1 and #3 |
| Empty grey frame (renders, no objects) | camera looking at floor | use `look_at=` not `rotation=` |
| Table/peg render but no arm | wrong asset (franka_visuals.usd) | use `franka.usd` under a parent Xform |
| `xformOp:translate already exists` | transforms on the referenced prim | put transforms on a parent Xform |
| First render hangs ~8 min at GPU 100% | one-time RTX shader compile | normal; wait. Cache persists for next time |
| `No module named 'pxr'/'imageio'` | used kernel python | use `/workspace/.venv/bin/python` |
