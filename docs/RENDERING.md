# FORGE-plus Rendering on RunPod — Complete Guide

> **Canonical doc for fresh-pod RTX rendering.**  Supersedes `docs/ISAAC_RTX_RENDERING.md`.
> Last updated 2026-06-23.

---

## Quick-start (fresh pod)

```bash
cd /workspace/FORGE-plus_task3

# 1. Build the libGLU stub (required — MDL-SDK will fail without it)
gcc -shared -fPIC -o /usr/local/lib/libGLU.so.1 scripts/libglu_stub.c && ldconfig

# 2. Run the full runtime setup (GLVND, Xvfb, env vars)
bash scripts/setup_runtime.sh

# 3. Record a rollout
export HOME=/workspace/persist/ovhome
export MPLBACKEND=Agg
export DISPLAY=:99
/workspace/.venv/bin/python scripts/eval_rollout_task3.py \
    --checkpoint checkpoints/task3_latest.pt \
    --out /workspace/forge_task3_states.npz

# 4. Render the video
/workspace/.venv/bin/python scripts/render_task3.py \
    --states /workspace/forge_task3_states.npz \
    --out docs/videos/task3/eval_run_001.mp4
```

---

## Required environment variables (set before importing isaacsim)

| Variable | Value | Why |
|---|---|---|
| `HOME` | `/workspace/persist/ovhome` | Kit/OV + driver shader caches persist across restarts |
| `MPLBACKEND` | `Agg` | Prevents matplotlib_inline crash on headless pods |
| `DISPLAY` | `:99` | Xvfb virtual display (see note below) |
| `CUDA_CACHE_PATH` | `/workspace/persist/cudacache` | CUDA JIT kernel cache |
| `__GL_SHADER_DISK_CACHE_PATH` | `/workspace/persist/shadercache/gl` | GL shader cache |

**Display note:** Use `:99` (or any free display) for new renders.  `:1` may be occupied by
task1 services if that repo is co-deployed on the same pod.  To find a free display:

```bash
for d in 99 98 2 3; do
  if ! xdpyinfo -display :$d >/dev/null 2>&1; then
    echo ":$d is free"; break
  fi
done
```

---

## Known issues & lessons learned

### 1 — libGLU.so.1 missing on fresh pods (BLOCKER)

**Symptom:** MDL-SDK fails to load; Isaac Sim logs an error about `libGLU.so.1` not found.
RTX rendering silently produces black frames or crashes.

**Fix:**
```bash
# Build the stub once per pod (ephemeral — must re-run after each restart)
gcc -shared -fPIC -o /usr/local/lib/libGLU.so.1 scripts/libglu_stub.c && ldconfig
```

The stub (`scripts/libglu_stub.c`) provides the minimal symbol set that MDL-SDK
needs.  Full GLU functionality is not required.  **This step must be in every setup
script / pod-init routine.**

---

### 2 — `rep.orchestrator.step()` causes deadlock in live-physics renders (BLOCKER)

**Symptom:** The render process hangs indefinitely at the first frame capture.
No traceback; the process is alive but stuck.

**Root cause:**
`rep.orchestrator.step()` schedules a deferred render callback at k=0.
At k=1, `env.step()`'s internal `app.update()` fires that callback while
`app.update()` is still on the call stack → circular deadlock.

**Fix:** **Remove `rep.orchestrator.step()` entirely from live-physics render loops.**

```python
# WRONG — causes deadlock when env.step() is called
for state in states:
    env.step(action)
    rep.orchestrator.step()          # <-- remove this
    frame = annotator.get_data()

# CORRECT — annotators receive data directly from env.step()'s internal update
for state in states:
    env.step(action)
    frame = annotator.get_data()     # data already populated by app.update()
```

**Scope:** This applies only to **live-physics renders** (where `env.step()` drives
simulation).  Trajectory-replay renders that do **not** call `env.step()` can use
`rep.orchestrator.step()` safely.

---

### 3 — Missing `.rgs.hlsl` shaders are non-fatal (IGNORE)

**Symptom:** Isaac Sim logs errors like:
```
[Error] Failed to find shader: Translucency.rgs.hlsl
[Error] Failed to find shader: Reflections.rgs.hlsl
[Error] Failed to find shader: DirectLightingSampled.rgs.hlsl
```

**Impact:** None.  Isaac Sim falls back gracefully.  Rendering works correctly.
**Action: Do not spend time trying to fix these.**

---

### 4 — Shader cache pre-population is unnecessary (SKIP)

The `fetch_shadercache.py` script and any cache pre-population step add significant
startup time but are **not required**.  Isaac Sim renders correctly with a cold cache;
shaders compile on first use and are cached to `/workspace/persist/`.

Remove or skip any `fetch_shadercache.py` call from your workflow.

---

### 5 — NVIDIA known issue #36 — async rendering skips frames

`isaacsim.core.throttling` toggles `asyncRendering=True` between physics steps.
This can cause frames to be skipped in live-physics renders.

**Fix:** Disable async rendering via carb settings before starting the render loop:

```python
import carb
carb.settings.get_settings().set("/app/asyncRendering", False)
carb.settings.get_settings().set("/app/asyncRenderingLowLatency", False)
```

---

## File layout

```
scripts/
  setup_runtime.sh        # Run once per pod restart
  libglu_stub.c           # libGLU.so.1 stub source
  eval_rollout_task3.py   # Record rollout → forge_task3_states.npz
  render_task3.py         # Render states.npz → mp4

docs/
  RENDERING.md            # This file
  ISAAC_RTX_RENDERING.md  # Older notes (superseded by this file)
  videos/
    task3/
      eval_run_001.mp4    # First successful live-physics RTX render
      eval_run_NNN.mp4    # Naming scheme: zero-padded 3-digit run index
```

**Video naming scheme:** `docs/videos/task3/eval_run_NNN.mp4` where NNN is a
zero-padded 3-digit index (001, 002, …).  Each evaluation run that produces a
video gets the next index.  Do not reuse indices.

---

## Live-physics render pipeline (how render_task3.py works)

1. **Rollout** — `eval_rollout_task3.py` runs the trained policy in the Isaac Sim
   physics environment, recording joint states/actions to a `.npz` file.

2. **Render** — `render_task3.py` loads the `.npz`, replays the trajectory in a
   fresh Isaac Sim session with RTX ray-tracing enabled, and captures frames via
   Replicator annotators.  `env.step()` drives both physics and rendering;
   `rep.orchestrator.step()` is **not** called.

3. **Encode** — frames are assembled into an mp4 with `imageio`/`ffmpeg`.

---

## Reproducing from scratch on a new pod

```bash
# 0. Clone / ensure repo is at task3 branch
cd /workspace
git clone https://github.com/robot-team00/FORGE-plus FORGE-plus_task3
cd FORGE-plus_task3
git checkout task3

# 1. libGLU stub (MUST be first)
gcc -shared -fPIC -o /usr/local/lib/libGLU.so.1 scripts/libglu_stub.c && ldconfig

# 2. Runtime deps + display
bash scripts/setup_runtime.sh

# 3. Environment
export HOME=/workspace/persist/ovhome
export MPLBACKEND=Agg
export DISPLAY=:99
mkdir -p /workspace/persist/ovhome /workspace/persist/cudacache /workspace/persist/shadercache/gl

# 4. Rollout (uses existing checkpoint)
/workspace/.venv/bin/python scripts/eval_rollout_task3.py \
    --checkpoint checkpoints/task3_latest.pt \
    --out /tmp/task3_states.npz

# 5. Render
/workspace/.venv/bin/python scripts/render_task3.py \
    --states /tmp/task3_states.npz \
    --out docs/videos/task3/eval_run_001.mp4

# 6. Verify
ls -lh docs/videos/task3/eval_run_001.mp4
```

