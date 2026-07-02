#!/usr/bin/env python3
"""Probe the Nucleus Franka USD for a Robotiq 2F-140 gripper variant, and list what
Robotiq assets exist on the asset server. Read-only; no scene is built."""
import os, sys
os.environ.update({"HOME": "/workspace/persist/ovhome", "MPLBACKEND": "Agg", "DISPLAY": ":99"})
sys.path.insert(0, "/workspace/FORGE-plus_task3")

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
print("ISAAC_NUCLEUS_DIR =", ISAAC_NUCLEUS_DIR, flush=True)

fp = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
print("opening", fp, flush=True)
stage = Usd.Stage.Open(fp)
if stage is None:
    print("FAILED to open franka.usd", flush=True)
else:
    dp = stage.GetDefaultPrim()
    print("default prim:", dp.GetPath() if dp else None, flush=True)
    for prim in [dp] + list(dp.GetChildren() if dp else []):
        vsets = prim.GetVariantSets()
        for name in vsets.GetNames():
            vs = vsets.GetVariantSet(name)
            print(f"  prim={prim.GetPath()} variantSet={name} options={vs.GetVariantNames()} "
                  f"current={vs.GetVariantSelection()}", flush=True)

# Also check what standalone Robotiq assets exist
import omni.client
for d in ["Robots/Robotiq", "Robots"]:
    url = f"{ISAAC_NUCLEUS_DIR}/{d}"
    res, entries = omni.client.list(url)
    print(f"\nlist {url}: {res}", flush=True)
    if res == omni.client.Result.OK:
        for e in entries[:40]:
            print("   ", e.relative_path, flush=True)

app.close()
