#!/usr/bin/env python3
"""Single-scene RTX snapshots of the task1 gear-insertion env.

Clone of render_forge_min.py's proven harness (same RTX flags, lights,
camera math, annotator patches), pointed at FrankaGearInsertEnv. Runs the
CURRENT checkpoint policy and saves stills at hand-off / mid-descent /
final state -> /workspace/snapshots_gear/gear_snap_{1,2,3}.png
"""
import os, sys, time as _time
import numpy as np
from pathlib import Path
from PIL import Image

_PC = "/workspace/persist"
os.environ.update({"HOME": _PC + "/ovhome", "MPLBACKEND": "Agg", "DISPLAY": ":99"})
os.makedirs(os.environ["HOME"], exist_ok=True)
sys.path.insert(0, "/workspace/FORGE-plus_task3")

_EXTRA = [
    "--/exts/isaacsim.core.throttling/enable_async=false",
    "--/rtx/raytracing/subsurface/enabled=false",
    "--/rtx/reflections/enabled=false",
    "--/rtx/translucency/enabled=false",
    "--/rtx/directLighting/sampledLighting/enabled=false",
    "--/rtx/indirectDiffuse/enabled=false",
    "--/rtx/ambientOcclusion/enabled=false",
    "--/rtx/raytracing/lightcache/spatialCache/enabled=false",
]
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 960, "height": 540,
                     "extra_args": _EXTRA})
print("booted", flush=True)
import torch, carb
torch.manual_seed(0)   # same trajectory as probe_grasp (seat press at k~220)
import omni.usd
from pxr import Gf, UsdGeom, UsdLux
import omni.replicator.core as rep
from forge_plus.isaac_gear_env import FrankaGearInsertEnv, GearInsertEnvCfg
from forge_plus.skills.policy_network import ForceConditionedPolicy, PolicyConfig

S = carb.settings.get_settings()
for _k in ["/rtx/reflections/enabled", "/rtx/translucency/enabled",
           "/rtx/indirectDiffuse/enabled", "/rtx/ambientOcclusion/enabled",
           "/rtx/directLighting/sampledLighting/enabled"]:
    S.set(_k, False)

cfg = GearInsertEnvCfg()
cfg.scene.num_envs = 1
cfg.forge_mode = True
cfg.forge_obj_cls = 0
cfg.forge_no_term = True
# the RTX view draws a live-tracking GHOST of the gear (dual-source probe:
# buffer == raw PhysX == upright, render tilted) — the duplicate comes from
# the physics-replication instancing path; a single env doesn't need it
cfg.scene.replicate_physics = False
from isaaclab.envs import DirectRLEnv as _DRL
FrankaGearInsertEnv.render = _DRL.render
env = FrankaGearInsertEnv(cfg)
print("env built", flush=True)

_CK = os.environ.get("SNAP_CKPT",
                     "/workspace/FORGE-plus_task3/checkpoints/task1_gear_sliprand.pt.it300")
ckpt = torch.load(_CK, map_location=env.device, weights_only=False)
pc = ckpt["policy_cfg"]
pcfg = pc if isinstance(pc, PolicyConfig) else PolicyConfig(**pc)
policy = ForceConditionedPolicy(pcfg).to(env.device)
policy.load_state_dict(ckpt["policy_state_dict"])
policy.eval()
print("policy loaded", flush=True)

out = env.reset()
obs = (out[0] if isinstance(out, tuple) else out)["policy"]
orig = env.scene.env_origins[0].cpu().numpy()
stage = omni.usd.get_context().get_stage()

dome = UsdLux.DomeLight.Define(stage, "/World/SkyDome")
dome.CreateIntensityAttr(750.0); dome.CreateColorAttr((0.95, 0.92, 0.86))
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(3200.0); sun.CreateColorAttr((1.0, 0.96, 0.88)); sun.CreateAngleAttr(0.6)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-48.0, 18.0, 0.0))
fill = UsdLux.SphereLight.Define(stage, "/World/Fill")
fill.CreateIntensityAttr(26000.0); fill.CreateRadiusAttr(0.4); fill.CreateColorAttr((1.0, 0.92, 0.82))
UsdGeom.Xformable(fill).AddTranslateOp().Set(
    Gf.Vec3d(float(orig[0]) - 0.8, float(orig[1]) - 1.2, float(orig[2]) + 1.6))
S.set("/rtx/sceneDb/ambientLightIntensity", 0.6)

gp = UsdGeom.Mesh.Define(stage, "/World/Ground")
_S = 8.0; gz = float(orig[2]) + 0.001
gp.CreatePointsAttr([(-_S, -_S, gz), (_S, -_S, gz), (_S, _S, gz), (-_S, _S, gz)])
gp.CreateFaceVertexCountsAttr([4]); gp.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
gp.CreateNormalsAttr([(0, 0, 1)] * 4)
gp.CreateDisplayColorAttr([(0.32, 0.30, 0.27)])

# framing: CLOSEUP=1 puts the camera ~0.45 m from the pinch point over the shaft
# (pad-on-hub contact fills the frame); default keeps the wide 3/4 tabletop view
CLOSEUP = os.environ.get("CLOSEUP", "0") == "1"
cam = UsdGeom.Camera.Define(stage, "/World/EvalCam")
if CLOSEUP:
    cam.CreateFocalLengthAttr(45.0)
    # the USD default near clip is 1.0 — at a 0.47 m eye distance the whole scene
    # falls inside the near plane and renders blank
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 10000.0))
    # SIDE view at hub height: pad-vs-hub vertical alignment reads directly off
    # the image, no projection math needed
    eye = Gf.Vec3d(float(orig[0]) + 0.449, float(orig[1]) - 0.42, float(orig[2]) + 0.47)
    tgt = Gf.Vec3d(float(orig[0]) + 0.449, float(orig[1]) + 0.120, float(orig[2]) + 0.455)
else:
    cam.CreateFocalLengthAttr(30.0)
    eye = Gf.Vec3d(float(orig[0]) + 1.25, float(orig[1]) - 0.85, float(orig[2]) + 0.95)
    tgt = Gf.Vec3d(float(orig[0]) + 0.40, float(orig[1]) + 0.10, float(orig[2]) + 0.42)
up = Gf.Vec3d(0, 0, 1)
fwd = (tgt - eye).GetNormalized()
rgt = Gf.Cross(fwd, up).GetNormalized()
tup = Gf.Cross(rgt, fwd).GetNormalized()
M = Gf.Matrix4d(rgt[0], rgt[1], rgt[2], 0, tup[0], tup[1], tup[2], 0,
                -fwd[0], -fwd[1], -fwd[2], 0, eye[0], eye[1], eye[2], 1)
UsdGeom.Xformable(cam).AddTransformOp().Set(M)
print("scene dressed", flush=True)

for _ in range(110): app.update()
rp = rep.create.render_product("/World/EvalCam", (960, 540))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])
try:
    from omni.replicator.core.scripts.utils import annotator_utils as _au
    _orig_fn = _au._resize_data_for_overscan
    def _safe(d, p):
        if not p or p.get("datawindow_overscan_z") is None: return d
        return _orig_fn(d, p)
    _au._resize_data_for_overscan = _safe
except Exception as ex:
    print("overscan skip: " + str(ex), flush=True)
for _ in range(110): app.update()
print("render ready shape=%s" % str(np.asarray(rgb.get_data()).shape), flush=True)

# Pause-capture (ported from render_recovery.py): the live-physics capture path
# draws STALE transforms — the closeup showed open fingers hovering above the gear
# while the physics buffers had the pads clamped on the hub (pad_c_z-gear_z at
# design). Freeze simulation for the capture and push the CURRENT world pose of
# the gear + every robot link (fingers included) into usdrt Fabric, which is what
# the renderer actually reads.
_rtf = {"xfs": None, "diag": False}
def _rt_flush_dynamics():
    from usdrt import Usd as RtUsd, Gf as RtGf, Rt
    from pxr import Usd as PUsd, Gf as PGf
    if _rtf["xfs"] is None:
        _rtstage = RtUsd.Stage.Attach(omni.usd.get_context().get_stage_id())
        gear_path = env._obj.cfg.prim_path.replace("env_.*", "env_0")
        gp_prim = _rtstage.GetPrimAtPath(gear_path)
        if not gp_prim.IsValid():
            # replicated-physics stages can ghost the root — fall back to the
            # first Xformable child (probe_pausecap's BADPATH case)
            print("flush: gear root INVALID at %s — probing children" % gear_path,
                  flush=True)
            u_prim = stage.GetPrimAtPath(gear_path)
            for child in PUsd.PrimRange(u_prim):
                cp = _rtstage.GetPrimAtPath(str(child.GetPath()))
                if cp.IsValid() and str(child.GetPath()) != gear_path:
                    print("flush: using child %s" % child.GetPath(), flush=True)
                    gp_prim = cp
                    break
        # gear world scale from USD (take-100 lesson: a missing world scale
        # can no-op or vanish the flushed visual)
        _M = UsdGeom.Xformable(stage.GetPrimAtPath(gear_path)) \
            .ComputeLocalToWorldTransform(PUsd.TimeCode.Default())
        _rtf["gear_scale"] = PGf.Transform(_M).GetScale()
        xfs = [(Rt.Xformable(gp_prim), None)]
        # ALSO flush the renderable DESCENDANTS: FSD composes a plain child
        # mesh from its own local + parent LOCAL, ignoring the parent's
        # world fast-path — writing only /Object left the visuals mesh at
        # its boot-time (falling, tilted) pose while the readback said fine.
        # The visuals' local xform is identity, so the body pose applies 1:1.
        u_root = stage.GetPrimAtPath(gear_path)
        for child in PUsd.PrimRange(u_root):
            cpath = str(child.GetPath())
            if cpath == gear_path or child.GetTypeName() != "Mesh":
                continue
            cp = _rtstage.GetPrimAtPath(cpath)
            if cp.IsValid():
                xfs.append((Rt.Xformable(cp), None))
                print("flush: + gear descendant %s" % cpath, flush=True)
        body_names = list(env._robot.data.body_names)
        for lp_ in env._robot.root_physx_view.link_paths[0]:
            prim = _rtstage.GetPrimAtPath(lp_)
            tail = lp_.rsplit("/", 1)[-1]
            if prim.IsValid() and tail in body_names:
                xfs.append((Rt.Xformable(prim), body_names.index(tail)))
        _rtf["xfs"] = xfs
        print("pause-cap flush set: gear + %d robot links" % (len(xfs) - 1), flush=True)
    d = env._robot.data
    lp = getattr(d, "body_link_pos_w", None)
    lq = getattr(d, "body_link_quat_w", None)
    if lp is None:
        lp, lq = d.body_pos_w, d.body_quat_w
    bp = env._obj.data.root_pose_w[0].tolist()
    for xf, li in _rtf["xfs"]:
        p = bp if li is None else (lp[0, li].tolist() + lq[0, li].tolist())
        if not xf.HasWorldXform():
            xf.SetWorldXformFromUsd()
        xf.GetWorldPositionAttr().Set(RtGf.Vec3d(p[0], p[1], p[2]))
        xf.GetWorldOrientationAttr().Set(
            RtGf.Quatf(p[3], RtGf.Vec3f(p[4], p[5], p[6])))
        if li is None:
            s = _rtf["gear_scale"]
            xf.GetWorldScaleAttr().Set(RtGf.Vec3d(s[0], s[1], s[2]))
    if not _rtf["diag"]:
        _rtf["diag"] = True
        gxf = _rtf["xfs"][0][0]
        print("flush DIAG: gear phys=%s fabric_readback=%s scale=%s" %
              (["%.3f" % v for v in bp[:3]],
               gxf.GetWorldPositionAttr().Get(),
               gxf.GetWorldScaleAttr().Get()), flush=True)

def _usd_flush_gear():
    # census v2: the gear's USD pose is never synced by the physics loop (it sits
    # far from the physics pose; only static prims render honestly). The renderer
    # demonstrably reads USD — pin the gear's USD xform to the physics buffer.
    from pxr import Usd as PUsd
    prim = stage.GetPrimAtPath(env._obj.cfg.prim_path.replace("env_.*", "env_0"))
    p = env._obj.data.root_pose_w[0].tolist()
    xf = UsdGeom.Xformable(prim)
    ops = {o.GetOpName(): o for o in xf.GetOrderedXformOps()}
    tr = ops.get("xformOp:translate") or xf.AddTranslateOp()
    orp = ops.get("xformOp:orient") or xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
    tr.Set(Gf.Vec3d(p[0], p[1], p[2]))
    q = Gf.Quatf(p[3], Gf.Vec3f(p[4], p[5], p[6]))
    orp.Set(q if orp.GetPrecision() == UsdGeom.XformOp.PrecisionFloat
            else Gf.Quatd(p[3], Gf.Vec3d(p[4], p[5], p[6])))

def grab(path, mode="flush"):
    """mode: 'live' = unpaused, no flush; 'pause' = paused, no flush;
    'flush' = paused + Fabric flush (the render_recovery pattern);
    'touch' = re-write the gear's CURRENT pose into PhysX (a physics no-op
    that marks the body dirty for the render sync — the only path proven to
    move this prim's visual: jump test + identity test both went through a
    write_root_pose_to_sim) then capture LIVE."""
    if mode == "touch":
        rs = env._obj.data.root_state_w.clone()
        env._obj.write_root_pose_to_sim(rs[:, :7])
        env._obj.write_root_velocity_to_sim(rs[:, 7:])
        env._obj.update(env.physics_dt)
        try:
            _rt_flush_dynamics()   # keep the link flush for the arm
        except Exception as fx:
            print("rt-flush FAILED: %r" % (fx,), flush=True)
        for _ in range(4):
            app.update()
        d = np.asarray(rgb.get_data())
        if d.ndim >= 3 and d.shape[0] > 1:
            Image.fromarray(d[..., :3]).save(path)
            print("saved %s (touch)" % path, flush=True)
        return
    if mode != "live":
        S.set_bool("/app/player/playSimulations", False)
    if mode == "flush":
        try:
            _rt_flush_dynamics()
        except Exception as fx:
            print("rt-flush FAILED: %r" % (fx,), flush=True)
        try:
            _usd_flush_gear()
        except Exception as fx:
            print("usd-flush FAILED: %r" % (fx,), flush=True)
    try:
        # 4 updates, not 2: with no per-step rendering between grabs the RTX
        # pipeline is a grab behind (frame N showed grab N-1's state)
        for _ in range(4):
            app.update()
        d = np.asarray(rgb.get_data())
    finally:
        if mode != "live":
            S.set_bool("/app/player/playSimulations", True)
    if d.ndim >= 3 and d.shape[0] > 1:
        Image.fromarray(d[..., :3]).save(path)
        print("saved %s (%s)" % (path, mode), flush=True)

OUT = "/workspace/snapshots_gear"
os.makedirs(OUT, exist_ok=True)

# warmup app.updates disturbed the reset; re-reset like render_forge_min
out = env.reset()
obs = (out[0] if isinstance(out, tuple) else out)["policy"]
_sfx = "_closeup" if CLOSEUP else ""
# 170 = carry over the shaft, 205 = mid-press, 220 = the seat moment (probe:
# gear z 0.406, Fins 12.8 N — capturing later photographs post-success drift,
# since this env runs forge_no_term)
snaps = {170: OUT + f"/gear_snap_1_handoff{_sfx}.png",
         205: OUT + f"/gear_snap_2_descent{_sfx}.png",
         220: OUT + f"/gear_snap_3_final{_sfx}.png"}
for k in range(230):
    with torch.no_grad():
        m, _ = policy(obs, env.f_cmd_norm().to(env.device))
    res = env.step(torch.clamp(m, -1, 1))
    obs = res[0]["policy"]
    if k == 170 and os.environ.get("CENSUS", "0") == "1":
        # v2: read USD-COMPOSED world transforms (census v1 proved no prim near
        # the gear has Fabric world data pre-flush — the renderer reads USD).
        # If the gear's USD rotation is big while physics says <=5 deg, the
        # physics->USD sync is the corruptor.
        from pxr import Usd as PUsd, Gf as PGf
        gw = env._obj.data.root_pose_w[0].tolist()
        print("census: physics gear pos=(%.3f,%.3f,%.3f) quat=(%.3f,%.3f,%.3f,%.3f)"
              % tuple(gw[:7]), flush=True)
        tc = PUsd.TimeCode.Default()
        for prim in stage.Traverse():
            if prim.GetTypeName() not in ("Mesh", "Cube", "Cylinder"):
                continue
            path = str(prim.GetPath())
            if "/Robot/" in path and "finger" not in path:
                continue   # skip arm links: only fingers matter here
            M = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(tc)
            t = M.ExtractTranslation()
            if (abs(t[0] - gw[0]) < 0.10 and abs(t[1] - gw[1]) < 0.10
                    and abs(t[2] - gw[2]) < 0.10):
                r = PGf.Transform(M).GetRotation()
                print("census HIT %-58s pos=(%.3f,%.3f,%.3f) rot: axis=%s ang=%.1f"
                      % (path, t[0], t[1], t[2],
                         tuple(round(v, 2) for v in r.GetAxis()), r.GetAngle()),
                      flush=True)
        print("census done", flush=True)
    if k == 222 and os.environ.get("IDTEST", "0") == "1":
        # frame-offset test: PhysX pose = identity quat, free space IN-FRAME
        # (side camera covers x~0.33-0.57), captured via the live sync (the one
        # path proven to draw). A tilted render of an identity-quat body proves
        # a constant offset between the body frame and the rendered frame.
        try:
            inert = env._obj.root_physx_view.get_inertias()[0].tolist()
            print("physx inertia tensor: %s" % (["%.3e" % v for v in inert],),
                  flush=True)
        except Exception as ex:
            print("inertia read failed: %r" % (ex,), flush=True)
        rs = env._obj.data.root_state_w.clone()
        rs2 = rs.clone()
        rs2[0, 0] = env.scene.env_origins[0, 0] + 0.52   # in-frame, right of shaft
        rs2[0, 1] = env.scene.env_origins[0, 1] + 0.12
        rs2[0, 2] = env.scene.env_origins[0, 2] + 0.50
        rs2[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=rs2.device)
        rs2[0, 7:] = 0.0
        env._obj.write_root_pose_to_sim(rs2[:, :7])
        env._obj.write_root_velocity_to_sim(rs2[:, 7:])
        env._obj.update(env.physics_dt)
        grab(OUT + "/idtest_identity.png", mode="live")
        print("identity-quat test frame saved", flush=True)
    if k == 170 and os.environ.get("MODETEST", "0") == "1":
        # which capture mode draws the truth? Physics here: gear upright (tilt
        # 0-5 deg), hub between the pads. Whichever frame shows that is honest.
        grab(OUT + "/mode_live.png", mode="live")
        grab(OUT + "/mode_pause.png", mode="pause")
        grab(OUT + "/mode_flush.png", mode="flush")
        print("mode test frames saved", flush=True)
    if k == 170 and os.environ.get("JUMPTEST", "0") == "1":
        # sync ground truth: photograph the physics gear, teleport it +8 cm,
        # photograph again, restore. If the rendered gear does not jump, the
        # object visual is not following physics AT ALL.
        rs = env._obj.data.root_state_w.clone()
        grab(OUT + "/jump_before.png")
        rs2 = rs.clone(); rs2[0, 2] += 0.08; rs2[0, 7:] = 0.0
        env._obj.write_root_pose_to_sim(rs2[:, :7])
        env._obj.write_root_velocity_to_sim(rs2[:, 7:])
        env._obj.update(env.physics_dt)
        grab(OUT + "/jump_after.png")
        env._obj.write_root_pose_to_sim(rs[:, :7])
        env._obj.write_root_velocity_to_sim(rs[:, 7:])
        env._obj.update(env.physics_dt)
        print("jump test frames saved", flush=True)
    if k in snaps:
        g = env._obj.data.root_pose_w[0, :3].cpu().numpy() - orig
        # grasp-geometry cross-check: pad-CENTER z minus gear-origin z should sit at
        # mug_grip_z (0.032) when the hub is seated between the pads; the finger BODY
        # origin is at the finger base, the pad centre ~0.045 below it (panda finger)
        bn = list(env._robot.data.body_names)
        lf = env._robot.data.body_pos_w[0, bn.index("panda_leftfinger")].cpu().numpy()
        rf = env._robot.data.body_pos_w[0, bn.index("panda_rightfinger")].cpu().numpy()
        pad_c_z = 0.5 * (lf[2] + rf[2]) - 0.045 - orig[2]
        gap_xy = ((0.5*(lf[0]+rf[0]) - orig[0] - g[0])**2
                  + (0.5*(lf[1]+rf[1]) - orig[1] - g[1])**2) ** 0.5
        print(f"snap at k={k}: gear=({g[0]:.3f},{g[1]:.3f},{g[2]:.3f}) "
              f"Fins={float(env._cf_insert[0]):.2f} pad_c_z-gear_z={pad_c_z - g[2]:+.4f} "
              f"(design +0.032) pinch_xy_offset={gap_xy:.4f}", flush=True)
        grab(snaps[k], mode="touch")
print("DONE", flush=True)
# skip app.close(): it hangs this pod's headless Kit after the captures are on disk
# (same as eval_gear_jam) — a hard exit releases the GPU cleanly
os._exit(0)
