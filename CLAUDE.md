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


## Headless Isaac Sim RTX rendering WORKS on this pod (fixed 2026-06-20)

Full root-cause writeup + troubleshooting: **docs/ISAAC_RTX_RENDERING.md**

### After every pod (re)start, run once:

    cd /workspace/FORGE-plus_main      # or your task clone
    bash scripts/setup_runtime.sh     # restores libEGL (Vulkan), shader cache, Xvfb :1

### Render the eval rollout (photorealistic Franka Panda, RTX path-traced):

    cd /workspace/FORGE-plus_main      # or your task clone
    DISPLAY=:1 /workspace/.venv/bin/python scripts/render_eval_video.py
    # -> docs/eval_episode.mp4  (1920x1080, 30fps)
    # NOTE: the FIRST render after a restart is SLOW (~8 min) because Isaac compiles
    # the RTX pipeline shaders once; the cache then persists and later runs are fast.

### Hard-won facts - do NOT regress these (each cost hours to find):

1. Vulkan needs the GLVND dispatcher **libEGL.so.1** (apt: libegl1 + libglvnd0 ...).
   Missing => NVIDIA Vulkan fails to init (vkCreateInstance ERROR_INCOMPATIBLE_DRIVER)
   => RTX renders nothing: log says "Cannot load shader file GenerateMipMap.comp.hlsl"
   and every frame is an "EMPTY buffer". nvidia-smi/CUDA still work - only graphics is dead.
2. The Isaac **gpu_foundation shader cache ships as a stub**. scripts/fetch_shadercache.py
   restores the ~19 compiled-shader files from the wheel via HTTP range requests.
3. Frame capture MUST call **rep.orchestrator.step(rt_subframes=N)** before rgb.get_data().
   Bare app.update() never populates the annotator on this build => empty frames forever.
4. The camera uses **look_at=(...)**, NOT rotation=(...). Default Omniverse cameras look
   straight down at the floor, so rotation=(-25,0,0) renders an empty grey frame.
5. Use **/workspace/assets/franka/franka.usd** (shared assets, outside the repo; has real meshes). franka_visuals.usd has NO geometry.
   Reference it under a *parent* Xform and put your translate/rotate/scale on the parent,
   or you hit "xformOp:translate already exists" (the asset already has root xform ops).
6. Run Isaac/video code with **/workspace/.venv/bin/python** (the venv is shared at /workspace/.venv, outside the repo) - the JupyterLab kernel python lacks pxr/imageio.

## Environment
- GPU: NVIDIA RTX 2000 Ada (sm_89), driver 570.172.08. Isaac Sim 5.1.0. Vulkan 1.4.303.
- Vulkan is headless-only here: needs Xvfb on display :1 (no real X server).
- GitHub push credentials are NOT stored on the pod by default; push from a machine that has them.

## PyBullet fallback (no GPU, always works)
If Isaac/RTX is ever broken, scripts/eval_render_pybullet.py renders the same rollout with
PyBullet's CPU TinyRenderer (clear articulated arm, not photorealistic) -> docs/eval_episode_pybullet.mp4.
