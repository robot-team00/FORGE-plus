#!/usr/bin/env python3
"""Full evaluation pipeline for Force-Budgeted Recovery.

Runs all baselines + our method across all tasks and grippers,
then prints the metrics table from §10.

Usage:
    python scripts/evaluate.py --task task1 --n-episodes 100
    python scripts/evaluate.py --all-tasks --n-episodes 200 --env mock
    python scripts/evaluate.py --task task1 --env isaac --num-envs 1024 --n-episodes 200
    python scripts/evaluate.py --task task1 --baseline oracle --n-episodes 50
"""

from __future__ import annotations

import argparse
import sys

# ── Bootstrap: SimulationApp must be the very first Isaac import ──────────────
#
# Isaac Sim registers carb/omni into sys.path inside SimulationApp.__init__.
# Any import of isaaclab.* before that call fails with:
#   ModuleNotFoundError: No module named 'carb'
#
# We pre-parse --env before any other setup so we know whether we need
# to boot Isaac Sim before the rest of the imports.
def _needs_isaac() -> bool:
    """Return True if the real Isaac Lab GPU path will be used."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--env", default="mock")
    ns, _ = p.parse_known_args()
    return ns.env == "isaac"


if _needs_isaac():
    # noqa: E402 — must precede all isaaclab imports
    from isaacsim import SimulationApp  # type: ignore[import]
    _SIM_APP = SimulationApp({"headless": True})

# ── Safe to import everything else now ───────────────────────────────────────
import json  # noqa: E402
import logging
import random
from pathlib import Path

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Force-Budgeted Recovery")
    p.add_argument("--task", choices=["task1", "task2", "task3", "all"], default="task1")
    p.add_argument("--gripper", choices=["franka_panda", "robotiq_2f140", "both"], default="both")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument(
        "--env",
        choices=["mock", "isaac"],
        default="mock",
        help=(
            "Physics backend for eval: "
            "'mock' (fast CPU, no GPU required, useful for sanity checks) or "
            "'isaac' (real Isaac Lab GPU sim — required for paper eval numbers). "
            "Isaac Lab requires a CUDA GPU and SimulationApp to be running."
        ),
    )
    p.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Parallel Isaac Lab envs (--env isaac only; ignored with --env mock)",
    )
    p.add_argument(
        "--device",
        default="cuda:0",
        help="Torch/Isaac Lab device (--env isaac only)",
    )
    p.add_argument(
        "--baseline",
        choices=["ours", "no_ceiling", "fixed_global", "press_harder", "heuristic", "vision_llm", "oracle", "all"],
        default="all",
    )
    p.add_argument("--backend", choices=["anthropic", "local", "mock"], default="mock",
                   help="LLM backend (mock for testing, anthropic or local for real runs)")
    p.add_argument("--local-url", default="http://localhost:11434/v1",
                   help="Base URL for local OpenAI-compatible server (--backend local)")
    p.add_argument("--local-model", default="qwen2.5:7b-instruct",
                   help="Model name served at --local-url (--backend local)")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def build_env(args: argparse.Namespace):
    """Instantiate the physics environment selected by --env.

    The Isaac Lab path requires SimulationApp to already be running (handled
    at module import time above). The mock path has no GPU dependency.
    """
    if args.env == "isaac":
        from forge_plus.envs.isaac_lab_env import IsaacLabAssemblyEnv, IsaacLabEnvConfig
        cfg = IsaacLabEnvConfig(num_envs=args.num_envs, device=args.device)
        log.info(f"Using IsaacLabAssemblyEnv: num_envs={args.num_envs}, device={args.device}")
        return IsaacLabAssemblyEnv(cfg)
    else:
        from forge_plus.envs.mock_assembly_env import MockAssemblyEnv, MockEnvConfig
        log.info("Using MockAssemblyEnv (--env mock)")
        return MockAssemblyEnv(MockEnvConfig())


def build_episode_configs(task: str, gripper: str, n_episodes: int, seed: int):
    from forge_plus.envs.base_assembly_env import EpisodeConfig
    from forge_plus.envs.object_configs import OBJECT_REGISTRY

    task_objects = {
        "task1": ["abs_round_connector", "steel_peg"],
        "task2": ["resin_planet_gear", "metal_planet_gear"],
        "task3": ["glass_bowl", "ceramic_plate", "metal_plate"],
    }
    grippers = ["franka_panda", "robotiq_2f140"] if gripper == "both" else [gripper]
    tasks = ["task1", "task2", "task3"] if task == "all" else [task]
    rng = random.Random(seed)

    configs = []
    for t in tasks:
        for g in grippers:
            obj_keys = task_objects[t]
            per_combo = max(1, n_episodes // (len(obj_keys) * len(grippers) * len(tasks)))
            for obj_key in obj_keys:
                obj_cfg = OBJECT_REGISTRY[obj_key]
                for i in range(per_combo):
                    f_break = obj_cfg.sample_f_break(rng)
                    configs.append(EpisodeConfig(
                        object_key=obj_key,
                        task_name=t,
                        gripper=g,
                        f_break_n=f_break,
                        disturbance_seed=rng.randint(0, 100000),
                    ))
    return configs


def run_evaluation(args: argparse.Namespace) -> None:
    from forge_plus.evaluation.baselines import BaselineRunner, BaselineType
    from forge_plus.evaluation.metrics import compute_metrics, print_metrics_table
    from forge_plus.episode import EpisodeRunner
    from forge_plus.llm.client import build_client
    from forge_plus.llm.budget_setter import BudgetSetter
    from forge_plus.llm.recovery_selector import RecoverySelector
    from forge_plus.encoding.signature_encoder import SignatureEncoder
    from forge_plus.recovery.recovery_actions import RecoveryActionExecutor
    from forge_plus.skills.forge_skill import FORGESkill, SkillConfig
    from forge_plus.skills.policy_network import PolicyConfig

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    env = build_env(args)
    skill = FORGESkill(SkillConfig(policy_cfg=PolicyConfig()))
    rec_exec = RecoveryActionExecutor()

    episode_cfgs = build_episode_configs(args.task, args.gripper, args.n_episodes, args.seed)
    log.info(f"Running {len(episode_cfgs)} episodes")

    all_results = {}

    baselines_to_run = (
        list(BaselineType) if args.baseline == "all" else [BaselineType(args.baseline)]
    )

    # Run baselines
    for bl_type in baselines_to_run:
        if bl_type.value == "ours":
            continue
        log.info(f"Running baseline: {bl_type.value}")
        runner = BaselineRunner(
            baseline_type=bl_type, skill=skill, env=env, recovery_executor=rec_exec, k_max=5
        )
        results = [runner.run(cfg) for cfg in episode_cfgs]
        metrics = compute_metrics(results)
        all_results[bl_type.value] = metrics
        if args.verbose:
            print(f"\n=== Baseline: {bl_type.value} ===")
            print_metrics_table(metrics)

    # Run ours
    if args.baseline in ("ours", "all"):
        log.info("Running: ours")
        llm_cfg: dict = {"backend": args.backend}
        if args.backend == "local":
            llm_cfg["base_url"] = args.local_url
            llm_cfg["model"] = args.local_model
        llm_client = build_client(llm_cfg)
        budget_setter = BudgetSetter(client=llm_client)
        recovery_selector = RecoverySelector(client=llm_client)
        sig_encoder = SignatureEncoder()

        runner = EpisodeRunner(
            budget_setter=budget_setter,
            recovery_selector=recovery_selector,
            signature_encoder=sig_encoder,
            skill=skill,
            env=env,
            recovery_executor=rec_exec,
            k_max=5,
            verbose=args.verbose,
        )
        results = [runner.run(cfg) for cfg in episode_cfgs]
        metrics = compute_metrics(results)
        all_results["ours"] = metrics
        if args.verbose:
            print("\n=== Ours ===")
            print_metrics_table(metrics)

    # Save results
    out_file = Path(args.output_dir) / f"metrics_{args.task}_{args.gripper}.json"
    serializable = {k: v.as_dict() for k, v in all_results.items()}
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2)
    log.info(f"Results saved to {out_file}")

    # Print comparison table
    _print_comparison(all_results)


def _print_comparison(all_results: dict) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        t = Table(title="Force-Budgeted Recovery — Method Comparison")
        t.add_column("Method", style="bold")
        t.add_column("Success%", justify="right")
        t.add_column("Breakage%", justify="right")
        t.add_column("FragileBreak%", justify="right")
        t.add_column("OverBudget%", justify="right")
        t.add_column("MeanMarginN", justify="right")

        for name, m in all_results.items():
            t.add_row(
                name,
                f"{m.closed_loop_success_rate:.1%}",
                f"{m.breakage_rate:.1%}",
                f"{m.fragile_breakage_rate:.1%}" if not isinstance(m.fragile_breakage_rate, float) or not (m.fragile_breakage_rate != m.fragile_breakage_rate) else "N/A",
                f"{m.over_budget_rate:.1%}",
                f"{m.mean_safety_margin_n:.1f}",
            )
        console.print(t)
    except ImportError:
        for name, m in all_results.items():
            print(f"{name}: success={m.closed_loop_success_rate:.1%} breakage={m.breakage_rate:.1%}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_evaluation(args)


if __name__ == "__main__":
    main()
