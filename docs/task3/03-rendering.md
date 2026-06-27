# 03 — Headless RTX rendering (live-physics)

How we turn a rollout into a path-traced demo mp4 on a headless pod. File:
`scripts/render_pick_place.py`. Also see the repo-root `CLAUDE.md` and `docs/RENDERING.md` for
the original fresh-pod setup. **Every numbered "gotcha" here cost hours — do not regress them.**

## 1. Fresh-pod setup (run once per pod restart)

```bash
cd /workspace/FORGE-plus_task3
gcc -shared -fPIC -o /usr/local/lib/libGLU.so.1 scripts/libglu_stub.c && ldconfig   # MDL-SDK needs it
bash scripts/setup_runtime.sh        # GLVND, Xvfb :99
export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 PYTHONPATH=/workspace/FORGE-plus_task3
```

`libGLU.so.1` is required — MDL-SDK fails *silently* without it. Set the env vars **before**
importing `isaacsim`.

## 2. SimulationApp flags

The renderer uses `SimulationApp({headless:True, width, height, extra_args})` with
`--/exts/isaacsim.core.throttling/enable_async=false` and a set of RTX feature toggles. The
async-off flag is critical: it makes `app.update()` **synchronous**, so annotator data is
committed before the call returns.

## 3. The big gotcha: `app.update()` AND `sim.render()` STEP PHYSICS

In this build, **both `app.update()` and `env.sim.render()` advance the physics sim**, with **no**
`_apply_action`/phase logic running between `env.step()` calls. Two consequences shaped the whole
pipeline:

- The ~220 pre-loop RTX-warmup `app.update()`s run *after* `env.reset()` and **drop the seated
  object** before the rollout. Fix: call `env.reset()` **again after** the warmups, right before
  the loop, then run the warmup-seat via `env.step()`s.
- Per captured frame you must drive the render with `app.update()` (which flushes physics→render),
  not `sim.render()` — see §4.

## 4. The render↔physics SYNC bug (the object froze on screen)

**Symptom:** the physics was correct (the seat put the bottle in the gripper, `d_xy≈0.04`), but
the **rendered** bottle stayed frozen on the shelf while the arm moved correctly.

**Root cause:** the object's pose is set via `write_root_pose_to_sim` (the warmup seat) and then
moved by the friction grip. **Neither `env.sim.render()` nor `env.step()` (which runs
`sim.step(render=False)`) flushes those PhysX changes to Fabric/RTX.** So the rendered mesh was
stuck at the object's last *simulated* pose (where it fell during the warmup). The **arm**
rendered fine because its motion is simulated every `app.update()`.

A previous "fix" made it worse: `_grab()` used `env.sim.render()` subframes **plus** a per-frame
capture+`write_root_pose_to_sim` **restore** of the object — which *froze* the object solid.

**Correct pattern (matches the proven `scripts/render_task3.py`):**

- `_grab()` = a couple of `app.update()`s then read the annotator. **No `sim.render()`. No pose
  capture/restore.** `app.update()` flushes PhysX→Fabric, so the friction-held object follows the
  gripper on screen. (`app.update()` flushing was proven by the warmup-fall rendering correctly.)
- The loop is **pure simulation**: `env.step(act)` → `_grab()`. Nothing writes object poses except
  the deliberate seat and the deliberate gripper-open (doc 02 §7).
- Before the capture loop, **flush** the seated pose to the render with ~20 `app.update()`s, or
  the first frame still shows the object on the shelf.

**Debugging rule:** the headless eval **cannot** reproduce the place (it stalls at `TRANSPORT`)
because it lacks `_grab()`'s per-frame `app.update()` settling. **Instrument the renderer itself**
— print finger width / object pose / EE pose inside the render loop. Don't trust a headless proxy
for render-coupled behavior.

## 5. Render loop structure (current)

```
env.reset()
... define camera, render product, overscan patch ...
for _ in range(110): app.update()       # warmup 1 (RTX init)
... attach rgb annotator ...
for _ in range(110): app.update()       # warmup 2 (settle annotator)
env.reset()                              # RE-SEAT after the warmups dropped the object
for 8: env.step(zeros)                   # warmup-seat: grip closes, friction holds (d_xy→~0.04)
for 20: app.update()                     # flush the seated pose to the render mesh
loop k:
    act = policy(obs) until placed; then scripted RETRACT (doc 02 §8)
    env.step(act)
    if placed: force fingers open (doc 02 §7)   # genuine release
    frame = _grab()                              # app.update()*2 + read rgb
    draw HUD; save frame
    end on first place + tail; break if the env auto-resets (don't capture episode 2)
ffmpeg frames -> mp4    # BEFORE app.close() (RTX shutdown hangs; a watchdog force-exits)
```

Render-only tweak: set `cfg.settle_steps` high (e.g. 400) so the env doesn't auto-reset the
instant the place succeeds — we want to capture the release + retract, not loop into episode 2.

**Wine-cellar insertion render (`place_strategy="insert"`, current — doc 05 §5):** the loop drives
**zero actions** (`act = zeros(7)`) until the bottle is inserted, then the scripted lateral
`RETRACT`. It does **not** call the policy — the env's base-aim + firm grip + cell geometry carry
the insertion, and the obs is now 37-dim (an `obj_up` triple was added) so the 34-dim shelf-place
policy would dim-mismatch. The renderer also binds a dark-wood PBR over the rack meshes
(`/World/envs/env_0/Rack`) and titles the HUD "Wine-Cellar Bottle Insertion". The kitchen
backdrop, wood floor, and wood counter are added for photorealism (§6, doc 05 §6).

## 6. Camera & HUD

- Camera: elevated front 3/4 view from `+x, -y` (the arm reaches base→shelf along `+x,+y`, so
  viewing from the side keeps it from blocking the object; a front view looked straight down the
  arm). Tune `eye`/`tgt`/focal so the **whole arm and the shelf** are in frame.
- HUD: title, object label (`OBJ_KEY="wine_bottle"`, not the budget class), phase `[n/7]`, and a
  **force gauge** (contact force vs `F_cmd` budget marker and `F_break` marker).
- Output: `docs/videos/task3/pick_place_eval_001.mp4`. ffmpeg at ~16 fps gives a gentle slow-mo
  fitting a fragile place; encode **before** `app.close()`.

## 7. Other non-fatal noise (don't chase)

- Missing `.rgs.hlsl` shaders (Translucency/Reflections/DirectLightingSampled) log errors but
  rendering works.
- `libMaterialX*`/`NGX` load errors and `Could not get NGX parameters` are non-fatal.
- First boot after a pod restart can take ~6–11 min (cold shader cache) — not a hang. But **only
  run one Isaac process at a time**; concurrent boots/kills wedge the GPU/caches and *do* hang.

## 8. PyBullet fallback

If RTX is ever broken, `scripts/eval_render_pybullet.py` renders the same rollout with PyBullet's
CPU renderer (clear articulated arm, not photorealistic).
