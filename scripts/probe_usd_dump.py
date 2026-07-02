#!/usr/bin/env python3
"""Dump full USD layer texts needed to author the Franka + Robotiq 2F-140 variant."""
import os
os.environ.update({"HOME": "/workspace/persist/ovhome", "MPLBACKEND": "Agg", "DISPLAY": ":99"})

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, Sdf

ROOT = "/workspace/assets/isaac51/Robots"
OUT = "/workspace/logs/usd_dumps"
os.makedirs(OUT, exist_ok=True)

dumps = {
    "franka_root.usda": f"{ROOT}/FrankaRobotics/FrankaPanda/franka.usd",
    "cfg_2f85.usda": f"{ROOT}/FrankaRobotics/FrankaPanda/configuration/franka_Gripper_Robotiq_2F_85.usd",
    "r140_edit.usda": f"{ROOT}/Robotiq/2F-140/Robotiq_2F_140_physics_edit.usd",
    "r140_base.usda": f"{ROOT}/Robotiq/2F-140/Robotiq_2F_140_base.usd",
    "r85_edit.usda": f"{ROOT}/Robotiq/2F-85/Robotiq_2F_85_edit.usd",
}
for name, path in dumps.items():
    if not os.path.exists(path):
        print("missing:", path, flush=True)
        continue
    lay = Sdf.Layer.FindOrOpen(path)
    with open(f"{OUT}/{name}", "w") as fh:
        fh.write(lay.ExportToString())
    print(f"dumped {name} ({os.path.getsize(OUT+'/'+name)} chars) from {path}", flush=True)

# Also list 2F-85 dir for the file the config payloads
print("2F-85 files:", sorted(os.listdir(f"{ROOT}/Robotiq/2F-85")), flush=True)
print("2F-140 files:", sorted(os.listdir(f"{ROOT}/Robotiq/2F-140")), flush=True)

# Composed body/joint names of the 2F-140 (open the edit-or-base stage)
for cand in ["Robotiq_2F_140_physics_edit.usd", "Robotiq_2F_140_base.usd", "2f140_instanceable.usd"]:
    p = f"{ROOT}/Robotiq/2F-140/{cand}"
    if not os.path.exists(p):
        continue
    st = Usd.Stage.Open(p)
    print(f"\n== {cand} prims:", flush=True)
    for prim in st.Traverse():
        t = prim.GetTypeName()
        if "Joint" in t or t == "Xform":
            print("   ", prim.GetPath(), f"[{t}]", flush=True)
    break

app.close()
