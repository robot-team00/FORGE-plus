#!/usr/bin/env python3
# probe_policy_obs.py — OFFLINE policy probe (no Isaac): which observation feature
# drives the robotiq-phase flee? Feed franka-like vs robotiq-shimmed obs and sweep
# feature blocks one at a time.
import sys
sys.path.insert(0, "/workspace/FORGE-plus_task3")
import torch
from forge_plus.skills.policy_network import ForceConditionedPolicy, PolicyConfig

ck = torch.load("/workspace/FORGE-plus_task3/checkpoints/task3_forge_entrance.pt",
                map_location="cpu", weights_only=False)
pc = ck["policy_cfg"]
pcfg = pc if isinstance(pc, PolicyConfig) else PolicyConfig(**pc)
pol = ForceConditionedPolicy(pcfg)
pol.load_state_dict(ck["policy_state_dict"])
pol.eval()
print("obs_dim etc:", {k: getattr(pcfg, k) for k in dir(pcfg) if not k.startswith("_")
                       and isinstance(getattr(pcfg, k), (int, float))}, flush=True)

def obs_vec(jp, jv, ee_p, ee_q, ft, b2g, up, fcmd_feat):
    return torch.tensor([list(jp) + list(jv) + list(ee_p) + list(ee_q)
                         + list(ft) + list(b2g) + list(up) + [fcmd_feat]],
                        dtype=torch.float32)

JP  = [0.05, -0.57, 0.07, -2.38, -0.01, 2.86, 0.74]
JV  = [0.0] * 7
QU  = [0.004, 0.861, 0.082, 0.501]
FT  = [0.0] * 6
UP  = [0.0, 0.0, 1.0]

# franka reference state (from the instrumented franka run, insertion start)
FR = dict(ee_p=[0.451, 0.058, 0.641], b2g=[-0.006, 0.065, -0.090])
# robotiq state at the policy hand-off (smoke 27), after the shim
RQ = dict(ee_p=[0.452, 0.112, 0.684], b2g=[0.040, -0.055, -0.134])

for fcmd in (0.0, 0.5, 1.0):
    fc = torch.tensor([fcmd], dtype=torch.float32)
    for name, st in (("franka-like", FR), ("robotiq-shim", RQ)):
        o = obs_vec(JP, JV, st["ee_p"], QU, FT, st["b2g"], UP, fcmd)
        with torch.no_grad():
            m, s = pol(o, fc)
        print(f"fcmd={fcmd:.1f} {name:13s} mean={[round(float(v),2) for v in m[0]]}",
              flush=True)

# feature sweep: start franka-like, replace one block with robotiq's
print("--- sweep (fcmd=0.5) ---", flush=True)
fc = torch.tensor([0.5], dtype=torch.float32)
for name, kw in (
    ("base(franka)", FR),
    ("ee_p->rq", dict(ee_p=RQ["ee_p"], b2g=FR["b2g"])),
    ("b2g->rq", dict(ee_p=FR["ee_p"], b2g=RQ["b2g"])),
    ("b2g_x->rq", dict(ee_p=FR["ee_p"], b2g=[RQ["b2g"][0], FR["b2g"][1], FR["b2g"][2]])),
    ("b2g_y->rq", dict(ee_p=FR["ee_p"], b2g=[FR["b2g"][0], RQ["b2g"][1], FR["b2g"][2]])),
    ("b2g_z->rq", dict(ee_p=FR["ee_p"], b2g=[FR["b2g"][0], FR["b2g"][1], RQ["b2g"][2]])),
    ("both->rq", RQ),
):
    o = obs_vec(JP, JV, kw["ee_p"], QU, FT, kw["b2g"], UP, 0.5)
    with torch.no_grad():
        m, s = pol(o, fc)
    print(f"{name:14s} mean={[round(float(v),2) for v in m[0]]}", flush=True)
