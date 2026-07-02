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
    p.add_argument("--gripper", choices=["franka_panda", "robotiq_2f140"],
                   default="franka_panda",
                   help="end-effector; robotiq_2f140 uses the Franka+2F-140 combo asset "
                        "(doc 08: teleport contract + parse ghost)")
    p.add_argument("--out", default="/workspace/logs/recovery_insertion.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = PickPlaceEnvCfg()
    cfg.scene.num_envs   = 1
    cfg.gripper          = args.gripper
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
        # The forge recovery maneuver re-drives to the training hand-off at a gentle
        # 0.012/step — needs ~40 control steps (counter decrements per physics substep).
        cfg.rec_dur_steps       = 80
        cfg.forge_hybrid_retract = True   # post-release retract (matches the rendered demo)
        cfg.forge_no_term       = True    # demo owns the timeline: a drop is an honest FAIL,
                                          # never an auto-reset that fakes a later "success"
        if args.gripper == "robotiq_2f140":
            # TELEPORT CONTRACT: the robotiq path starts at the DEFAULT arm pose (no
            # reset teleport), so the scripted setup needs a longer window to drive
            # the ~0.5 m from the spawn pose to the cell-entrance hand-off.
            cfg.forge_setup_steps = 400
            # The gripper's four-bar loop joints only survive a RAW parse — the
            # physics-replicated clone path drops them (single env: nothing to
            # replicate anyway).
            cfg.scene.replicate_physics = False

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
    if args.gripper == "robotiq_2f140":
        # four-bar health diagnostics (doc 08: teleport contract + parse ghost)
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        n_ghost = sum(1 for pr in stage.Traverse() if "GhostGripper" in str(pr.GetPath()))
        bn = list(env._robot.data.body_names)
        sep0 = float((env._robot.data.body_pos_w[0, bn.index("left_inner_finger")]
                      - env._robot.data.body_pos_w[0, bn.index("right_inner_finger")]).norm())
        print(f"  [rq-diag] ghost prims={n_ghost} ghost_init={getattr(env, '_ghost', None) is not None} "
              f"four-bar sep at spawn={sep0:.4f} (healthy closed ~0.040)", flush=True)

    def _on_step(e, attempt, step):
        if step % 10 == 0:
            bx, by, bz = e._base_xyz0()
            cf  = float(e._cf_insert[0].item())
            eez = float(e._robot.data.body_pos_w[0, e._ee_idx, 2].item()
                        - e.scene.env_origins[0, 2].item())
            if e.cfg.gripper == "robotiq_2f140":
                # finger_joint angle (0=closed..0.785=open) + pad-body separation
                gap = float(e._robot.data.joint_pos[0, e._grip_ids[0]].item())
                sep = float((e._robot.data.body_pos_w[0, e._lf_idx]
                             - e._robot.data.body_pos_w[0, e._rf_idx]).norm().item())
                gtxt = f"ang={gap:.3f} sep={sep:.4f}"
            else:
                gap = float((e._robot.data.joint_pos[0, 7] + e._robot.data.joint_pos[0, 8]).item())
                gtxt = f"gap={gap:.4f}"
            print(f"    [a{attempt} s{step:3d}] base=({bx:.3f},{by:.3f},{bz:.3f}) eez={eez:.3f} "
                  f"{gtxt} cf={cf:5.2f} rec={int(e._rec_steps[0])} "
                  f"setup={int(e._setup_ctr[0])} rel={int(e._released[0])} "
                  f"fail={e.is_failure()}", flush=True)

    result = loop.run(env, on_step=_on_step)

    # Optional finale probe (RECOVERY_FINALE_PROBE=1): after the recovery seats the bottle,
    # keep stepping the LEARNED policy (arm mean + SAMPLED release, the render scheme) to see
    # whether the release head fires on this object's observation and the retract clears.
    if os.environ.get("RECOVERY_FINALE_PROBE") == "1" and not args.scripted \
            and result.outcome.value == "SUCCESS":
        pol = env._skill_policy
        env._jam_on[:] = False
        obs = env._get_observations()["policy"]
        for k in range(240):
            with torch.no_grad():
                m, s = pol(obs, env.f_cmd_norm())
            act = m.clone()
            act[:, 7] = m[:, 7] + s[:, 7] * torch.randn_like(s[:, 7])
            res = env.step(torch.clamp(act, -1, 1))
            obs = res[0]["policy"]
            if k % 10 == 0:
                bx, by, bz = env._base_xyz0()
                import isaaclab.utils.math as _lm
                upz = float(_lm.matrix_from_quat(env._obj.data.root_pose_w[:1, 3:7])[0, 2, 2])
                vel = float(env._obj.data.root_vel_w[0, :3].norm())
                eod = float((env._robot.data.body_pos_w[0, env._ee_idx]
                             - env._obj.data.root_pose_w[0, :3]).norm())
                print(f"    [finale s{k:3d}] m7={float(m[0,7]):+.3f} rel={int(env._released[0])} "
                      f"succ={int(env._succeeded[0])} base=({bx:.3f},{by:.3f},{bz:.3f}) "
                      f"upz={upz:.3f} vel={vel:.3f} eod={eod:.3f}", flush=True)
            if bool(env._succeeded[0]):
                print(f"    [finale] PLACED at step {k}", flush=True)
                break

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
