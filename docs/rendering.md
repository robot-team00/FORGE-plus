# Headless Rendering on RunPod

## Root Cause of Rendering Failures

Isaac Sim's RTX renderer requires the `graphics` NVIDIA driver capability. RunPod containers started without this capability will fail with:

```
IHydraTexture refResource had no GPU foundation
```

This occurs even on Ada Lovelace (sm_89) GPUs when the container lacks graphics capability.

### Diagnosis

Check your container's NVIDIA capabilities:
```bash
cat /proc/driver/nvidia/capabilities/*/clients 2>/dev/null || echo $NVIDIA_DRIVER_CAPABILITIES
```

If the output shows only `compute,utility` (no `graphics`), rendering will fail.

## Fix: Enable Graphics Capability

**Before starting your pod**, set the container environment variable:

```
NVIDIA_DRIVER_CAPABILITIES=all
```

In RunPod: Edit Pod → Environment Variables → add `NVIDIA_DRIVER_CAPABILITIES=all` → restart pod.

> ⚠️ This requires a pod restart. Do not attempt this while a training run is active.

## Supported GPU Architectures

| GPU Generation | Architecture | Rendering | Training | Notes |
|---|---|---|---|---|
| Ampere (sm_86) | A4000, A100, RTX 3090 | ✅ | ✅ | Recommended |
| Ada Lovelace (sm_89) | RTX 2000 Ada, RTX 4090 | ✅ | ✅ | Recommended |
| Blackwell (sm_120) | RTX PRO 4000 | ❌ | ✅ (CPU physics) | Vulkan ICD broken |

**Blackwell note:** Even with `NVIDIA_DRIVER_CAPABILITIES=all`, Blackwell containers currently ship with a broken Vulkan ICD (`ERROR_INCOMPATIBLE_DRIVER`). Use Ampere or Ada pods for rendering.

## Headless Rendering Setup (Ada/Ampere)

Once the pod has `NVIDIA_DRIVER_CAPABILITIES=all`, Isaac Sim runs headless via EGL — no Xvfb or display server needed.

### Required: libGLU stub

Isaac Sim's RTX NeuRay plugin attempts to load `libGLU.so.1`. If missing, compile the stub:

```bash
gcc -shared -fPIC -o /usr/local/lib/libGLU.so.1 scripts/libglu_stub.c
ldconfig
```

### Verify environment

```bash
python scripts/check_env.py
```

Expected output:
```
[OK] GPU: NVIDIA RTX 2000 Ada Generation Laptop GPU (sm_89)
[OK] NVIDIA_DRIVER_CAPABILITIES includes 'graphics'
[OK] EGL: libEGL_nvidia.so.0 found
[OK] Vulkan ICD: /etc/vulkan/icd.d/nvidia_icd.json
[OK] libGLU.so.1: /usr/local/lib/libGLU.so.1
```

### Capture a frame

```bash
source /workspace/FORGE-plus/.venv/bin/activate
unset DISPLAY  # EGL mode, no X11 needed
python scripts/capture_frame.py --output docs/render_preview.png
```

Isaac Sim takes ~2 minutes to start. Output will be saved to `docs/render_preview.png`.

## Training Without Rendering

If you only need training (no visualization), `NVIDIA_DRIVER_CAPABILITIES=compute,utility` is sufficient. The RL policy trains on GPU correctly; only the visual renderer is blocked.

```bash
# Training works fine without graphics capability
python scripts/train_skill.py --task task1 --num-envs 16
```
