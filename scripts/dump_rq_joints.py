# Dump every joint + authored PhysxMimicJointAPI (gearing/offset/reference) from the
# NVIDIA 2F-140 asset variants AND our merged combo asset, so the mimic tree we author
# in the merged articulation copies NVIDIA's exact axis conventions instead of guessing.
import os
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdPhysics, PhysxSchema

FILES = [
    "/workspace/assets/isaac51/Robots/Robotiq/2F-140/Robotiq_2F_140_physics_edit.usd",
    "/workspace/assets/isaac51/Robots/Robotiq/2F-140/Robotiq_2F_140_config.usd",
    "/workspace/assets/isaac51/Robots/Robotiq/2F-140/Robotiq_2F_140_base.usd",
    "/workspace/assets/isaac51/Robots/Robotiq/2F-140/configuration/Robotiq_2F_140_robot.usd",
    "/workspace/assets/franka/franka_robotiq_2f140.usd",
]

for fn in FILES:
    if not os.path.exists(fn):
        print(f"==== {fn}: MISSING", flush=True)
        continue
    try:
        st = Usd.Stage.Open(fn)
    except Exception as e:
        print(f"==== {fn}: OPEN FAIL {e}", flush=True)
        continue
    print(f"==== {fn}", flush=True)
    for pr in st.Traverse():
        t = pr.GetTypeName()
        is_joint = "Joint" in t
        mimics = []
        for axname in ("rotX", "rotY", "rotZ"):
            a = PhysxSchema.PhysxMimicJointAPI(pr, axname)
            g = a.GetGearingAttr()
            if g and g.HasAuthoredValue():
                refs = [str(x) for x in a.GetReferenceJointRel().GetTargets()]
                mimics.append(f"MIMIC[{axname}] ref={refs} gearing={g.Get()} offset={a.GetOffsetAttr().Get()}")
        if not is_joint and not mimics:
            continue
        info = [f"{pr.GetPath()} [{t}]"]
        if t == "PhysicsRevoluteJoint":
            rj = UsdPhysics.RevoluteJoint(pr)
            info.append(f"axis={rj.GetAxisAttr().Get()} lim=({rj.GetLowerLimitAttr().Get()},{rj.GetUpperLimitAttr().Get()})")
        if is_joint:
            j = UsdPhysics.Joint(pr)
            b0 = [str(x) for x in j.GetBody0Rel().GetTargets()]
            b1 = [str(x) for x in j.GetBody1Rel().GetTargets()]
            info.append(f"b0={b0} b1={b1}")
            q0, q1 = j.GetLocalRot0Attr().Get(), j.GetLocalRot1Attr().Get()
            info.append(f"rot0={q0} rot1={q1}")
            ax0, ax1 = j.GetLocalPos0Attr().Get(), j.GetLocalPos1Attr().Get()
            info.append(f"pos0={ax0} pos1={ax1}")
            drv = UsdPhysics.DriveAPI(pr, "angular")
            if drv and drv.GetStiffnessAttr().HasAuthoredValue():
                info.append(f"drive k={drv.GetStiffnessAttr().Get()} d={drv.GetDampingAttr().Get()} f={drv.GetMaxForceAttr().Get()} tgt={drv.GetTargetPositionAttr().Get()}")
        info.extend(mimics)
        print("  " + " | ".join(info), flush=True)

print("DUMP DONE", flush=True)
os._exit(0)
