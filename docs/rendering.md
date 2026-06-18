# Headless Rendering on RunPod — FORGE-plus

## Root Cause of Rendering Failures

Isaac Sim's RTX renderer requires two things that are easy to get wrong:

1. **`NVIDIA_DRIVER_CAPABILITIES` must include `graphics`** — pods started without this fail with `IHydraTexture refResource had no GPU foundation`.
2. **Xvfb must be running and `DISPLAY` must be set** — the NVIDIA Vulkan ICD on Ada/Ampere requires an X display handle to initialise. Running with `unset DISPLAY` (EGL mode) causes `vkCreateInstance → VK_ERROR_INCOMPATIBLE_DRIVER` even when all driver libs are present and version-matched.

> ⚠️ **Most common mistake across debugging sessions:** setting `unset DISPLAY` or trying EGL mode. This fails on this setup. Always use Xvfb + `DISPLAY=:1`.

---

## Setup Checklist

### 1. Pod-level: graphics capability

**Before starting your pod**, set in RunPod UI → Edit Pod → Environment Variables:

```
NVIDIA_DRIVER_CAPABILITIES=all
```

> Setting this in the shell (`export NVIDIA_DRIVER_CAPABILITIES=all`) after the container starts does **nothing** — driver capabilities are locked at container startup. It must be in the pod's environment variables so it takes effect when the container is created.
>
> Verify the actual runtime value (not just the shell variable):
> ```bash
> cat /proc/self/environ | tr '\0' '\n' | grep NVIDIA_DRIVER
> ```
> You should see `graphics` in the value.

### 2. Container-level: libGLU stub

Isaac Sim's RTX NeuRay plugin attempts to load `libGLU.so.1`. Build a stub if missing:

```bash
gcc -shared -fPIC -o /usr/local/lib/libGLU.so.1 /workspace/FORGE-plus/scripts/libglu_stub.c
ldconfig
```

### 3. Verify environment

```bash
cd /workspace/FORGE-plus
source .venv/bin/activate
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

> Note: `check_env.py` only checks that files exist, not that Vulkan can actually initialise. A passing `check_env` does **not** guarantee rendering will work — you must also have Xvfb running (see below).

---

## Running Headless Renders

**Always use Xvfb + DISPLAY=:1.** Do not use `unset DISPLAY` or EGL mode — this setup requires an X display handle for Vulkan.

```bash
# Start Xvfb if not already running
Xvfb :1 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
sleep 2

# Capture a still frame (render_preview.png)
cd /workspace/FORGE-plus
source .venv/bin/activate
DISPLAY=:1 timeout 600 python scripts/render_preview.py
```

`render_preview.py` boots Xvfb internally if not already running, sets `DISPLAY=:1`, and saves to `docs/render_preview.png`. Isaac Sim takes ~2 minutes to start. Expected output:
```
Isaac Sim booted.
Scene built: 5×5 = 25 stations.
Warming up...
  warmup 0/50
  ...
Running render frames...
Capturing...
Saved 194,803 bytes → docs/render_preview.png
Done.
```

---

## GPU Architecture Compatibility

| GPU Generation | Architecture | Rendering | Training | Notes |
|---|---|---|---|---|
| Ampere (sm_86) | A4000, A100, RTX 3090 | ✅ | ✅ | Recommended |
| Ada Lovelace (sm_89) | RTX 2000 Ada, RTX 4090 | ✅ | ✅ | Verified; use Xvfb |
| Blackwell (sm_120) | RTX PRO 4000 | ❌ | ✅ | Vulkan ICD broken even with Xvfb |

**Blackwell note:** Even with `NVIDIA_DRIVER_CAPABILITIES=all` and Xvfb, Blackwell containers ship with a broken Vulkan ICD (`VK_ERROR_INCOMPATIBLE_DRIVER`). Use Ampere or Ada pods for rendering.

---

## Training Without Rendering

If you only need training (no visualisation), `NVIDIA_DRIVER_CAPABILITIES=compute,utility` is sufficient and no Xvfb is needed:

```bash
python scripts/train_skill.py --task task1 --num-envs 1024
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `IHydraTexture refResource had no GPU foundation` | `graphics` not in NVIDIA capabilities | Set `NVIDIA_DRIVER_CAPABILITIES=all` in RunPod pod env vars and restart pod |
| `vkCreateInstance → VK_ERROR_INCOMPATIBLE_DRIVER` | No X display — Vulkan ICD needs one | Start Xvfb and set `DISPLAY=:1` before running any Isaac Sim script |
| `libGLU.so.1: No such file or directory` | Missing libGLU stub | Build stub: `gcc -shared -fPIC -o /usr/local/lib/libGLU.so.1 scripts/libglu_stub.c && ldconfig` |
| `check_env.py` passes but render still fails | `check_env` only checks file existence, not Vulkan init | Run the Xvfb + DISPLAY=:1 test above; check `vkCreateInstance` result directly |
| `ModuleNotFoundError: No module named 'carb'` | `SimulationApp` not imported first | See `scripts/train_skill.py` bootstrap pattern (PR #17) |
