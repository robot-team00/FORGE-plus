#!/usr/bin/env python3
"""Verify the fractional-slip gate: arm rate ~= SLIP_FRAC across all envs,
and armed envs actually fire when the descent enters the funnel.

    HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 \
    PYTHONPATH=/workspace/FORGE-plus_task3 SLIP_DISTURB_MM=5 SLIP_FRAC=0.3 \
    /workspace/.venv/bin/python scripts/probe_slip_gate.py
"""
from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})

import os  # noqa: E402
import torch  # noqa: E402
from forge_plus.isaac_gear_env import FrankaGearInsertEnv, GearInsertEnvCfg  # noqa: E402
from forge_plus.skills.policy_network import ForceConditionedPolicy, PolicyConfig  # noqa: E402

cfg = GearInsertEnvCfg()
cfg.scene.num_envs = 128
cfg.forge_mode = True
cfg.forge_obj_cls = 0
print(f"[gate] cfg.slip_disturb_mm={cfg.slip_disturb_mm} cfg.slip_frac={cfg.slip_frac}",
      flush=True)
env = FrankaGearInsertEnv(cfg)

ck = torch.load("checkpoints/task1_gear_mixed_strict.pt",
                map_location=env.device, weights_only=False)
pc = ck["policy_cfg"]
pcfg = pc if isinstance(pc, PolicyConfig) else PolicyConfig(**pc)
policy = ForceConditionedPolicy(pcfg).to(env.device)
policy.load_state_dict(ck["policy_state_dict"])
policy.eval()

out = env.reset()
obs = (out[0] if isinstance(out, tuple) else out)["policy"]
fired_total = 0
ep_boundaries = 0
for k in range(1300):  # ~2 episodes
    with torch.no_grad():
        mean, _ = policy(obs, env.f_cmd_norm())
    res = env.step(mean.clamp(-1.0, 1.0))
    obs = res[0]["policy"]
    if k in (300, 590, 900, 1290):
        armed = int(env._slip_armed.sum()) if hasattr(env, "_slip_armed") else -1
        done = int(env._slip_done.sum()) if hasattr(env, "_slip_done") else -1
        print(f"[gate] step {k}: armed={armed}/128 fired(this-episode)={done}", flush=True)

print("[gate] PROBE_DONE", flush=True)
_app.close()
os._exit(0)
