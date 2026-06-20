#!/usr/bin/env python3
"""Task 3 (fragile place / stack) -- evaluation report.

Answers issue #28's definition-of-done question:

    Does the budget-setter set a LOWER F_max for the glass bowl (F_break ~22 N)
    than for the sturdy mug (F_break ~160 N) from identity alone?

and quantifies the central claim: the "press harder" recovery is maximally
destructive here -- it should break fragile glass/ceramic at a high rate while
our force-budgeted method keeps breakage low and still seats the part.

Run (no GPU / API key needed -- the heuristic backend reasons from identity):

    python scripts/report_task3.py --n-episodes 80 --backend heuristic

Output: a console report + results/task3_report.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TASK = "task3_fragile_place"
FRAGILE = {"glass_bowl", "ceramic_plate"}
ROBUST = {"metal_plate", "sturdy_mug"}
OBJECTS = ["glass_bowl", "ceramic_plate", "sturdy_mug", "metal_plate"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 3 fragile place/stack report")
    p.add_argument("--n-episodes", type=int, default=80,
                   help="episodes per object per gripper")
    p.add_argument("--gripper", choices=["franka_panda", "robotiq_2f140", "both"],
                   default="both")
    p.add_argument("--backend", choices=["heuristic", "anthropic", "local", "mock"],
                   default="heuristic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="results")
    return p.parse_args()


def build_configs(n_per: int, grippers: list[str], seed: int):
    from forge_plus.envs.base_assembly_env import EpisodeConfig
    from forge_plus.envs.object_configs import OBJECT_REGISTRY
    rng = random.Random(seed)
    cfgs = []
    for obj in OBJECTS:
        oc = OBJECT_REGISTRY[obj]
        for g in grippers:
            for _ in range(n_per):
                cfgs.append(EpisodeConfig(
                    object_key=obj, task_name=TASK, gripper=g,
                    f_break_n=oc.sample_f_break(rng),
                    max_steps=1500, disturbance_seed=rng.randint(0, 10**6),
                ))
    return cfgs


def budget_table(backend: str):
    from forge_plus.llm.client import build_client
    from forge_plus.llm.budget_setter import BudgetSetter
    from forge_plus.envs.object_configs import OBJECT_REGISTRY
    setter = BudgetSetter(client=build_client({"backend": backend}))
    rows = []
    for obj in OBJECTS:
        oc = OBJECT_REGISTRY[obj]
        resp = setter.set_budget(oc.identity, TASK)
        rows.append({
            "object": obj,
            "material": oc.identity.material,
            "f_break_mean_n": oc.f_break_mean_n,
            "f_max_n": round(resp.F_max_N, 1),
            "margin_n": round(oc.f_break_mean_n - resp.F_max_N, 1),
            "confidence": resp.confidence,
            "rationale": resp.rationale,
            "fragile": obj in FRAGILE,
        })
    return rows


class SeatingController:
    """Canonical place/stack controller: press straight down at the allotted
    force ceiling to seat the object. Task 3 isolates *budget-setting* and
    *recovery*, so the reference controller presses to its budget rather than
    relying on a trained policy -- keeping the report about the force budget and
    reproducible without a GPU or checkpoint."""

    def act(self, obs, f_cmd: float):
        from forge_plus.control.force_clamp import Wrench
        return Wrench(0.0, 0.0, float(f_cmd), 0.0, 0.0, 0.0)


def run_method(method: str, cfgs, backend: str):
    from forge_plus.envs.place_stack_env import PlaceStackEnv, PlaceStackEnvConfig
    from forge_plus.recovery.recovery_actions import RecoveryActionExecutor
    from forge_plus.evaluation.metrics import compute_metrics

    env = PlaceStackEnv(PlaceStackEnvConfig())
    skill = SeatingController()
    rec = RecoveryActionExecutor()

    if method == "ours":
        from forge_plus.episode import EpisodeRunner
        from forge_plus.llm.client import build_client
        from forge_plus.llm.budget_setter import BudgetSetter
        from forge_plus.llm.recovery_selector import RecoverySelector
        from forge_plus.encoding.signature_encoder import SignatureEncoder
        client = build_client({"backend": backend})
        runner = EpisodeRunner(
            budget_setter=BudgetSetter(client=client),
            recovery_selector=RecoverySelector(client=client),
            signature_encoder=SignatureEncoder(),
            skill=skill, env=env, recovery_executor=rec, k_max=5,
        )
        results = [runner.run(c) for c in cfgs]
    else:
        from forge_plus.evaluation.baselines import BaselineRunner, BaselineType
        runner = BaselineRunner(baseline_type=BaselineType(method), skill=skill,
                                env=env, recovery_executor=rec, k_max=5)
        results = [runner.run(c) for c in cfgs]

    m = compute_metrics(results, fragile_object_keys=FRAGILE)
    frag = [r for r in results if r.object_key in FRAGILE]
    rob = [r for r in results if r.object_key in ROBUST]
    return {
        "method": method,
        "success_rate": m.closed_loop_success_rate,
        "breakage_rate": m.breakage_rate,
        "fragile_breakage_rate": (sum(r.broke for r in frag) / len(frag)) if frag else 0.0,
        "robust_breakage_rate": (sum(r.broke for r in rob) / len(rob)) if rob else 0.0,
        "mean_peak_contact_n": m.mean_peak_contact_n,
        "over_budget_rate": m.over_budget_rate,
        "failure_modes": m.failure_mode_distribution,
    }


def main() -> None:
    args = parse_args()
    grippers = (["franka_panda", "robotiq_2f140"] if args.gripper == "both"
                else [args.gripper])
    cfgs = build_configs(args.n_episodes, grippers, args.seed)

    print(f"\n{'='*72}\n  TASK 3 -- FRAGILE PLACE / STACK REPORT")
    print(f"  backend={args.backend}  episodes={len(cfgs)}  grippers={grippers}\n{'='*72}")

    budgets = budget_table(args.backend)
    print("\n[A] Force budget from object IDENTITY ALONE (F_break is hidden):\n")
    print(f"  {'object':16s}{'material':22s}{'F_break~':>9s}{'F_max':>8s}{'margin':>8s}")
    for b in budgets:
        tag = "fragile" if b["fragile"] else "robust"
        print(f"  {b['object']:16s}{b['material']:22s}"
              f"{b['f_break_mean_n']:>9.0f}{b['f_max_n']:>8.1f}{b['margin_n']:>8.1f}  ({tag})")
    glass = next(b for b in budgets if b["object"] == "glass_bowl")
    mug = next(b for b in budgets if b["object"] == "sturdy_mug")
    verdict = "PASS" if glass["f_max_n"] < mug["f_max_n"] else "FAIL"
    print(f"\n  >> glass_bowl F_max ({glass['f_max_n']} N) < sturdy_mug F_max "
          f"({mug['f_max_n']} N) ?  {verdict}")

    methods = ["press_harder", "no_ceiling", "oracle", "ours"]
    rows = [run_method(mth, cfgs, args.backend) for mth in methods]
    print("\n[B] Method comparison on Task 3:\n")
    print(f"  {'method':14s}{'success':>9s}{'break':>8s}{'fragileBrk':>11s}"
          f"{'robustBrk':>10s}{'peakN':>8s}")
    for r in rows:
        print(f"  {r['method']:14s}{r['success_rate']:>8.0%} {r['breakage_rate']:>7.0%} "
              f"{r['fragile_breakage_rate']:>10.0%} {r['robust_breakage_rate']:>9.0%} "
              f"{r['mean_peak_contact_n']:>7.1f}")
    ph = next(r for r in rows if r["method"] == "press_harder")
    ours = next(r for r in rows if r["method"] == "ours")
    print(f"\n  >> fragile breakage: press_harder {ph['fragile_breakage_rate']:.0%} "
          f"vs ours {ours['fragile_breakage_rate']:.0%}")
    print(f"  >> failure modes (ours): {ours['failure_modes']}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"config": vars(args), "budgets": budgets, "methods": rows}
    (out / "task3_report.json").write_text(json.dumps(payload, indent=2))
    _write_markdown(out / "task3_report.md", args, budgets, rows, glass, mug, verdict)
    print(f"\n  saved: {out/'task3_report.json'} and {out/'task3_report.md'}\n")


def _write_markdown(path, args, budgets, rows, glass, mug, verdict):
    L = []
    L.append("# Task 3 -- Fragile Place / Stack: Evaluation Report\n")
    L.append(f"_backend={args.backend}, {args.n_episodes} episodes/object/gripper, seed={args.seed}_\n")
    L.append("## A. Force budget from object identity alone\n")
    L.append("F_break is hidden from the agent; the budget-setter sees only the object "
             "identity (name, material, geometry tags, mass).\n")
    L.append("| object | material | F_break (mean) | F_max | margin | role |")
    L.append("|---|---|--:|--:|--:|---|")
    for b in budgets:
        L.append(f"| {b['object']} | {b['material']} | {b['f_break_mean_n']:.0f} | "
                 f"{b['f_max_n']:.1f} | {b['margin_n']:.1f} | "
                 f"{'fragile' if b['fragile'] else 'robust'} |")
    L.append(f"\n**Result: {verdict}** -- glass_bowl F_max = {glass['f_max_n']} N "
             f"< sturdy_mug F_max = {mug['f_max_n']} N, from identity alone.\n")
    L.append("## B. Method comparison\n")
    L.append("| method | success | breakage | fragile breakage | robust breakage | mean peak N |")
    L.append("|---|--:|--:|--:|--:|--:|")
    for r in rows:
        L.append(f"| {r['method']} | {r['success_rate']:.0%} | {r['breakage_rate']:.0%} | "
                 f"{r['fragile_breakage_rate']:.0%} | {r['robust_breakage_rate']:.0%} | "
                 f"{r['mean_peak_contact_n']:.1f} |")
    ph = next(r for r in rows if r["method"] == "press_harder")
    ours = next(r for r in rows if r["method"] == "ours")
    L.append(f"\n**Press-harder is maximally harmful here:** it breaks fragile objects at "
             f"{ph['fragile_breakage_rate']:.0%} vs {ours['fragile_breakage_rate']:.0%} for our "
             f"force-budgeted method, because raising the ceiling on a place/stack contact drives "
             f"the axial force past the breaking point instead of reallocating motion.\n")
    L.append(f"Failure-mode distribution (ours): `{ours['failure_modes']}`\n")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
