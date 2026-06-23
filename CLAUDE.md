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

## PyBullet fallback (no GPU, always works)
If Isaac/RTX is ever broken, scripts/eval_render_pybullet.py renders the same rollout with
PyBullet's CPU TinyRenderer (clear articulated arm, not photorealistic) -> docs/eval_episode_pybullet.mp4.
