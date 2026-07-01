# CLAUDE.md - FORGE-plus (read this first)

**FORGE-plus runs on a remote RunPod GPU pod, not on your local machine.** Isaac Sim RTX
rendering, training, and evals all happen on the pod; this repo is just the code. Connect to
the pod first.

## Orientation (start here)

- **Where it runs:** a shared RunPod pod with an NVIDIA RTX GPU. The pod's ID/URL **changes
  frequently** (it gets restarted/replaced), so never hardcode it — read the current pod URL
  from the RunPod console, or from an open Chrome tab (JupyterLab is usually already open at
  `https://<pod>-8888.proxy.runpod.net/lab`, port 8888).
- **Three clones on the pod, one shared venv, one shared assets dir:**
  - `/workspace/FORGE-plus_main` — branch `main` (kept synced to `origin/main`)
  - `/workspace/FORGE-plus_task1` — branch `task1`
  - `/workspace/FORGE-plus_task3` — branch `task3`
  - **`/workspace/.venv`** — shared Python/Isaac venv (outside the repo, git-ignored). Run
    Isaac/video code with `/workspace/.venv/bin/python`; the bare Jupyter kernel python lacks
    `pxr`/`imageio`.
  - **`/workspace/assets/franka/`** — shared Franka USD assets (outside the repo, git-ignored).
- **Connecting without GoTTY:** drive JupyterLab from the browser. Browse files with the Jupyter
  contents API (`GET /api/contents/<path>?content=1`); run code by starting a kernel
  (`POST /api/kernels`, pass the `_xsrf` cookie as the `X-XSRFToken` header) and talking to its
  `wss://<pod>-8888.proxy.runpod.net/api/kernels/<id>/channels` websocket.
- **Git/auth:** the pod can `git fetch`/`pull` without credentials but **cannot push** (no creds
  stored) — publish by pushing from a machine that has creds, or via the GitHub web UI. And
  **never `git add -A`** in a clone: it sweeps up untracked render mp4s (incl. the protected
  `docs/insertion_success.mp4`). Stage explicit paths.


## Headless Isaac Sim RTX rendering — live-physics pipeline (working as of 2026-06-23)

See **docs/RENDERING.md** for the complete fresh-pod guide.

### After every pod (re)start, run once:
```bash
cd /workspace/FORGE-plus_task3
gcc -shared -fPIC -o /usr/local/lib/libGLU.so.1 scripts/libglu_stub.c && ldconfig
bash scripts/setup_runtime.sh          # GLVND, Xvfb :99, HOME/MPLBACKEND/DISPLAY
```

### Record a rollout then render it (live-physics RTX path-traced):
```bash
export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99
/workspace/.venv/bin/python scripts/eval_rollout_task3.py \
    --checkpoint checkpoints/task3_latest.pt --out /tmp/states.npz
/workspace/.venv/bin/python scripts/render_task3.py \
    --states /tmp/states.npz --out docs/videos/task3/eval_run_NNN.mp4
```

Videos are saved to `docs/videos/task3/eval_run_NNN.mp4` (NNN = zero-padded run index).
Existing render: `docs/videos/task3/eval_run_001.mp4`.

### Hard-won facts — do NOT regress these (each cost hours to find):

1. **libGLU.so.1 required** — MDL-SDK fails silently without it. Rebuild stub on
   every pod restart: `gcc -shared -fPIC -o /usr/local/lib/libGLU.so.1 scripts/libglu_stub.c && ldconfig`

2. **Remove rep.orchestrator.step() from live-physics loops** — it causes a
   circular deadlock when env.step() is also called. Annotators get data directly
   from env.step()'s internal app.update(). This was the main blocker.

3. **Missing .rgs.hlsl shaders are non-fatal** — Translucency/Reflections/
   DirectLightingSampled log errors but rendering works. Do not chase these.

4. **Shader cache pre-population not needed** — fetch_shadercache.py removed
   from setup_runtime.sh. Cold cache is fine.

5. **Env vars before importing isaacsim**: HOME=/workspace/persist/ovhome,
   MPLBACKEND=Agg, DISPLAY=:99 (not :1 — that may be in use by task1).

6. **numpy MUST be 1.26.0** — isaacsim 5.1 pins `numpy==1.26.0`. numpy 2.x has an
   incompatible C-ABI; the compiled OmniGraph/replicator `.so` then read arrays as
   **size-0**, so `rgb.attach`/`activate_node_template` raise
   `TypeError: Unable to write from unknown dtype, kind=f, size=0` and every render
   product comes out 0x0 (rgb.get_data() -> shape=(0,), all frames EMPTY). Installing
   ultralytics/opencv silently upgrades numpy and breaks ALL RTX rendering. Fix:
   `/workspace/.venv/bin/pip install "numpy==1.26.0"`.

7. **After a FULL pod stop/start, the venv python3 symlink may break** — base image
   resets `/usr/bin/python3` to 3.10 while the venv needs 3.11. Symptom:
   `ModuleNotFoundError: No module named 'numpy'` from `/workspace/.venv/bin/python`.
   Fix: `ln -sf /usr/bin/python3.11 /workspace/.venv/bin/python3`.

8. **NGX (DLSS) is broken on this pod** ("Failed to create NGX context"). Do NOT enable
   DLSS (`/rtx/post/dlss/execMode`) or DLAA (`/rtx/post/aa/op=3`) — with NGX dead the
   render product is 0x0. Use `aa/op=1` (TAA) and keep the RT GI stack OFF (reflections/
   translucency/indirectDiffuse/AO/sampledLighting = false), exactly like render_task3.

## Rendering the learned FORGE insertion policy

Use **scripts/render_forge_min.py** (built on the proven render_task3 harness). The full
scripts/render_pick_place.py does NOT render on this pod — its kitchen USD + textured PBR
materials leave the render product 0x0. Also note: FrankaPickPlaceEnv no-ops `render()` for
training speed, so the render script restores `DirectRLEnv.render` before constructing the env
(else the RTX context never initializes). Output: docs/videos/task3/forge_insert.mp4.

## PyBullet fallback (no GPU, always works)
If Isaac/RTX is ever broken, scripts/eval_render_pybullet.py renders the same rollout with
PyBullet's CPU TinyRenderer (clear articulated arm, not photorealistic) -> docs/eval_episode_pybullet.mp4.
