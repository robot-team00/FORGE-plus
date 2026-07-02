#!/usr/bin/env python3
"""Inspect the local Franka 5.1 USD variant wiring + the Robotiq 2F-140 structure,
to author a franka_Gripper_Robotiq_2F_140 configuration. Read-only."""
import os, sys
os.environ.update({"HOME": "/workspace/persist/ovhome", "MPLBACKEND": "Agg", "DISPLAY": ":99"})

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, Sdf

ROOT = "/workspace/assets/isaac51/Robots"

def tree(stage, max_prims=60, show=("Joint", "Xform", "Scope")):
    n = 0
    for prim in stage.Traverse():
        t = prim.GetTypeName()
        if any(s in t for s in show) or "Root" in t:
            print("   ", prim.GetPath(), f"[{t}]", flush=True)
            n += 1
            if n >= max_prims:
                print("    ... (truncated)", flush=True)
                break

print("═" * 70)
print("1) franka.usd — variant sets", flush=True)
fp = f"{ROOT}/FrankaRobotics/FrankaPanda/franka.usd"
stage = Usd.Stage.Open(fp)
dp = stage.GetDefaultPrim()
print("default prim:", dp.GetPath(), flush=True)
for prim in [dp] + list(dp.GetChildren()):
    for name in prim.GetVariantSets().GetNames():
        vs = prim.GetVariantSets().GetVariantSet(name)
        print(f"  {prim.GetPath()} :: {name} -> {vs.GetVariantNames()} (cur={vs.GetVariantSelection()})", flush=True)

print("═" * 70)
print("2) franka.usd ROOT layer text (variant wiring)", flush=True)
lay = Sdf.Layer.FindOrOpen(fp)
txt = lay.ExportToString()
print(f"root layer chars: {len(txt)}", flush=True)
print(txt[:6000], flush=True)

print("═" * 70)
print("3) 2F-85 config layer (attachment template)", flush=True)
cp = f"{ROOT}/FrankaRobotics/FrankaPanda/configuration/franka_Gripper_Robotiq_2F_85.usd"
lay85 = Sdf.Layer.FindOrOpen(cp)
txt85 = lay85.ExportToString()
print(f"chars: {len(txt85)}", flush=True)
print(txt85[:8000], flush=True)

print("═" * 70)
print("4) Robotiq 2F-140 — candidate USDs, prim trees", flush=True)
for c in ["2f140_instanceable.usd", "Robotiq_2F_140_base.usd",
          "configuration/Robotiq_2F_140_robot.usd", "Robotiq_2F_140_physics_edit.usd"]:
    p = f"{ROOT}/Robotiq/2F-140/{c}"
    if not os.path.exists(p):
        print(f"  -- missing {c}", flush=True)
        continue
    st = Usd.Stage.Open(p)
    d = st.GetDefaultPrim()
    print(f"\n  == {c}  default={d.GetPath() if d else None}", flush=True)
    tree(st, max_prims=40)

app.close()
