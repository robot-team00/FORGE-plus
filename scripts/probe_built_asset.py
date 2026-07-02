#!/usr/bin/env python3
"""Diagnose the built franka_robotiq_2f140.usd composition."""
import os
os.environ.update({"HOME": "/workspace/persist/ovhome", "MPLBACKEND": "Agg", "DISPLAY": ":99"})
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, Sdf, Tf

FP = "/workspace/assets/isaac51/Robots/FrankaRobotics/FrankaPanda"

print("── ROOT layer text ──", flush=True)
lay = Sdf.Layer.FindOrOpen(f"{FP}/franka_robotiq_2f140.usd")
print(lay.ExportToString()[:1200], flush=True)

print("── CFG layer text (head) ──", flush=True)
lay2 = Sdf.Layer.FindOrOpen(f"{FP}/configuration/franka_Gripper_Robotiq_2F_140.usd")
print(lay2.ExportToString()[:900], flush=True)

print("── compose CFG alone ──", flush=True)
st = Usd.Stage.Open(f"{FP}/configuration/franka_Gripper_Robotiq_2F_140.usd")
robotiq = [str(p.GetPath()) for p in st.Traverse() if "Robotiq" in str(p.GetPath())]
print("robotiq prims:", len(robotiq), robotiq[:6], flush=True)

print("── compose ROOT ──", flush=True)
st2 = Usd.Stage.Open(f"{FP}/franka_robotiq_2f140.usd")
robotiq2 = [str(p.GetPath()) for p in st2.Traverse() if "Robotiq" in str(p.GetPath())]
print("robotiq prims:", len(robotiq2), robotiq2[:6], flush=True)
lf = st2.GetPrimAtPath("/panda/panda_leftfinger")
print("panda_leftfinger active:", lf.IsActive() if lf else "missing", flush=True)
from pxr import UsdPhysics
n_rb = n_j = 0
for prim in st2.Traverse():   # Traverse() skips inactive prims
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print("  body :", prim.GetPath(), flush=True); n_rb += 1
    t = prim.GetTypeName()
    if "Joint" in t and "Fixed" not in t:
        print("  joint:", prim.GetPath(), f"[{t}]", flush=True); n_j += 1
print(f"SANITY bodies={n_rb} joints={n_j}", flush=True)
# The three edits that silently failed before:
fj = st2.GetPrimAtPath("/panda/Robotiq_2F_140_edit/robotiq_base_link/AssemblerFixedJoint")
print("AssemblerFixedJoint:", "OK" if fj and fj.IsValid() and fj.GetTypeName() == "PhysicsFixedJoint" else "MISSING", flush=True)
ge = st2.GetPrimAtPath("/panda/Robotiq_2F_140_edit")
print("gripper holder apiSchemas:", ge.GetAppliedSchemas(), flush=True)
from pxr import UsdGeom
xf = UsdGeom.Xformable(ge).GetLocalTransformation()
print("gripper holder local translate:", tuple(round(v, 4) for v in xf.ExtractTranslation()), flush=True)
roots = [str(p.GetPath()) for p in st2.Traverse() if "PhysicsArticulationRootAPI" in p.GetAppliedSchemas()]
print("articulation roots:", roots, flush=True)
# composition errors
for e in st2.GetCompositionErrors():
    print("COMP ERROR:", e, flush=True)
os._exit(0)   # skip app.close() entirely — it hangs headless and pins the GPU
