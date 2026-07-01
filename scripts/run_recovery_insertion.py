#!/usr/bin/env python3
"""Closed-loop force-signature LLM recovery on the Isaac wine-cellar insertion.

Demonstrates the proposal's recovery layer running on the REAL contact-rich
Isaac env (no vision, force signature only):

    LLM sets F_max (identity)  ->  insert  ->  the bottle WEDGES on the cell rim
    ->  encode a force signature  ->  LLM picks a recovery (force only)
    ->  apply it within the same F_max  ->  retry  ->  seats.

The jam is induced with `cfg.jam_dx` (a lateral base-aim error). Run with
`--jam 0.0` to see the clean insertion (no recovery needed).

    export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 \
           PYTHONPATH=/workspace/FORGE-plus_task3
    /workspace/.venv/bin/python scripts/run_recovery_insertion.py --jam 0.05
"""

from __future__ import annotations

import argparse

# Isaac Sim must be initialised BEFORE any omniverse imports
from isaacsim import SimulationApp  # noqa: E402

_app = SimulationApp({"headless": True})

import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402

import torch  # noqa: E402

from forge_plus.isaac_pick_place_env import FrankaPickPlaceEnv, PickPlaceEnvCfg  # noqa: E402
from forge_plus.llm.client import HeuristicLLMClient, build_client  # noqa: E402
from forge_plus.llm.recovery_selector import RecoverySelector  # noqa: E402
from forge_plus.recovery.recovery_loop import RecoveryLoop  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jam", type=float, default=0.05,
                   help="induced lateral base-aim error (m); 0 = clean insertion")
    p.add_argument("--k_max", type=int, default=4, help="max recovery attempts")
    p.add_argument("--backend", choices=["heuristic", "local", "anthropic"],
                   default="heuristic", help="LLM backend for the recovery selector")
    p.add_argument("--policy", default="checkpoints/task3_forge_entrance.pt",
                   help="learned FORGE policy that drives the insertion skill (step_skill)")
    p.add_argument("--obj", type=int, default=2,
                   help="object class the learned policy seats: 2=metal_plate (break 180N; jam caught "
                        "~13N, WELL below break). 0=glass_bowl is fragile but the learned policy "
                        "cannot seat it (force-freeze stalls the descent — the known fragile-insert gap).")
    p.add_argument("--scripted", action="store_true",
                   help="drive the scripted (zero-action base-aim) insertion instead of the learned policy")
    p.add_argument("--out", default="/workspace/logs/recovery_insertion.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = PickPlaceEnvCfg()
    cfg.scene.num_envs   = 1
    cfg.place_strategy   = "insert"
    cfg.jam_dx           = args.jam
    # Keep the episode from auto-resetting mid-demo so the loop owns the timeline.
    cfg.episode_length_s = 120.0
    cfg.settle_steps     = 400
    # Drive the recovery on the LEARNED FORGE policy (not the scripted base-aim). The env's
    # step_skill runs env._skill_policy when set; forge_mode routes the EE through _forge_targets.
    if not args.scripted:
        cfg.forge_mode          = True
        cfg.forge_release_mode  = True    # the trained policy is 8-dim (arm + release)
        cfg.grasp_topdown       = False
        cfg.forge_obj_cls       = args.obj    # 0=fragile glass (recovery story), 2=robust

    env = FrankaPickPlaceEnv(cfg)
    print(f"\n{'='*64}\n  Force-Budgeted Recovery — Isaac wine-cellar insertion"
          f"\n{'='*64}", flush=True)

    # Load the learned FORGE policy to drive the insertion skill (step_skill).
    if not args.scripted:
        import torch as _torch
        from forge_plus.skills.policy_network import ForceConditionedPolicy, PolicyConfig
        ck = _torch.load(args.policy, map_location=env.device, weights_only=False)
        _pc = ck["policy_cfg"]
        _pcfg = _pc if isinstance(_pc, PolicyConfig) else PolicyConfig(**_pc)
        _pol = ForceConditionedPolicy(_pcfg).to(env.device)
        _pol.load_state_dict(ck["policy_state_dict"])
        _pol.eval()
        env._skill_policy = _pol
        print(f"  skill = LEARNED FORGE policy ({args.policy})", flush=True)
    else:
        print("  skill = SCRIPTED base-aim insertion (zero policy action)", flush=True)

    # Slow layer: recovery selector (force signature -> action). Budget F_max is
    # already set per object inside the env from the LLM budget cache.
    if args.backend == "heuristic":
        client = HeuristicLLMClient()
    else:
        client = build_client({"backend": args.backend})
    selector = RecoverySelector(client=client)
    loop = RecoveryLoop(selector=selector, k_max=args.k_max)

    print(f"  jam_dx = {args.jam:.3f} m   F_max = {env.f_max_n:.1f} N "
          f"(object budget, LLM)   backend = {client.name()}", flush=True)

    def _on_step(e, attempt, step):
        if step % 30 == 0:
            bx, by, bz = e._base_xyz0()
            cf = float(e._cf_filt[0].item())
            print(f"    [a{attempt} s{step:3d}] bz={bz:.3f} cf={cf:5.2f} "
                  f"rec={int(e._rec_steps[0])} jam_on={int(e._jam_on[0])} "
                  f"fail={e.is_failure()}", flush=True)

    result = loop.run(env, on_step=_on_step)

    print(f"\n{'-'*64}")
    print(f"  OUTCOME: {result.outcome.value}  in {result.attempts} attempt(s)")
    for a in result.log:
        if a.recovery_action:
            s = a.signature
            print(f"   attempt {a.attempt}: {a.result} "
                  f"[peak_axial={s['peak_axial_N']}N net_insert={s['net_insert_mm']}mm "
                  f"rising={s['axial_rising']} lateral={s['lateral_bias']}] "
                  f"-> {a.recovery_action}  ({a.recovery_rationale})")
        else:
            print(f"   attempt {a.attempt}: {a.result}")
    bx, by, bz = env._base_xyz0()
    print(f"  final bottle base (env frame): ({bx:.3f}, {by:.3f}, {bz:.3f})  "
          f"cell=({cfg.rack_x},{cfg.rack_y},{cfg.cell_floor_z})")
    print(f"{'-'*64}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({
            "outcome": result.outcome.value,
            "attempts": result.attempts,
            "f_max_n": result.f_max_n,
            "jam_dx": args.jam,
            "final_base": [bx, by, bz],
            "log": [a.__dict__ for a in result.log],
        }, fh, indent=2)
    print(f"  wrote {args.out}", flush=True)
    _app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
