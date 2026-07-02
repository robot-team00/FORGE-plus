#!/usr/bin/env python3
"""FK probe: panda_hand pose (in the robot root frame) at the env's init arm pose,
using the plain Franka asset. Used to author the 2F-140 flange placement."""
import os, sys
os.environ.update({"HOME": "/workspace/persist/ovhome", "MPLBACKEND": "Agg", "DISPLAY": ":99"})
sys.path.insert(0, "/workspace/FORGE-plus_task3")

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Robot")
cfg.spawn.usd_path = "/workspace/assets/franka/panda_instanceable.usd"
cfg.spawn.rigid_props.disable_gravity = True   # exact kinematic FK, no sag
jp = dict(cfg.init_state.joint_pos)
jp.update({"panda_joint2": -0.73, "panda_joint4": -2.46, "panda_joint6": 2.85})
cfg.init_state.joint_pos = jp
robot = Articulation(cfg)
sim.reset()
import torch
# exact kinematic write (safe: the plain franka has no loop constraints)
desired = torch.tensor([[0.0, -0.73, 0.0, -2.46, 0.0, 2.85, 0.741, 0.022, 0.022]],
                       device=robot.device)
robot.write_joint_state_to_sim(desired, torch.zeros_like(desired))
robot.set_joint_position_target(desired[:, :7], joint_ids=list(range(7)))
for _ in range(4):
    robot.write_data_to_sim(); sim.step(); robot.update(sim.get_physics_dt())

bn = list(robot.data.body_names)
h = bn.index("panda_hand")
hp = robot.data.body_pos_w[0, h] - robot.data.root_pos_w[0]
hq = robot.data.body_quat_w[0, h]   # root at identity orientation
jq = robot.data.joint_pos[0, :7]
print("arm joints:", [round(float(v), 4) for v in jq], flush=True)
print("panda_hand pos (root frame): (%.6f, %.6f, %.6f)" % (float(hp[0]), float(hp[1]), float(hp[2])), flush=True)
print("panda_hand quat wxyz: (%.6f, %.6f, %.6f, %.6f)" % (float(hq[0]), float(hq[1]), float(hq[2]), float(hq[3])), flush=True)
os._exit(0)   # skip app.close() entirely — it hangs headless and pins the GPU
