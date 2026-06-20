# CLAUDE.md - FORGE-plus (read this first)

Guidance for Claude sessions working on this repo on the **shared RunPod pod**.

## Headless Isaac Sim RTX rendering WORKS on this pod (fixed 2026-06-20)

Full root-cause writeup + troubleshooting: **docs/ISAAC_RTX_RENDERING.md**

### After every pod (re)start, run once:

    cd /workspace/FORGE-plus
    bash scripts/setup_runtime.sh     # restores libEGL (Vulkan), shader cache, Xvfb :1

### Render the eval rollout (photorealistic Franka Panda, RTX path-traced):

    cd /workspace/FORGE-plus
    DISPLAY=:1 .venv/bin/python scripts/render_eval_video.py
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
5. Use **assets/franka/franka.usd** (has real meshes). franka_visuals.usd has NO geometry.
   Reference it under a *parent* Xform and put your translate/rotate/scale on the parent,
   or you hit "xformOp:translate already exists" (the asset already has root xform ops).
6. Run Isaac/video code with **.venv/bin/python** - the JupyterLab kernel python lacks pxr/imageio.

## Environment
- GPU: NVIDIA RTX 2000 Ada (sm_89), driver 570.172.08. Isaac Sim 5.1.0. Vulkan 1.4.303.
- Vulkan is headless-only here: needs Xvfb on display :1 (no real X server).
- GitHub push credentials are NOT stored on the pod by default; push from a machine that has them.

## PyBullet fallback (no GPU, always works)
If Isaac/RTX is ever broken, scripts/eval_render_pybullet.py renders the same rollout with
PyBullet's CPU TinyRenderer (clear articulated arm, not photorealistic) -> docs/eval_episode_pybullet.mp4.
