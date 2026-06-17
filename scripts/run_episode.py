#!/usr/bin/env python3
"""Run a single episode with optional verbose output.

Useful for debugging, demos, and checking individual LLM calls.

Usage:
    python scripts/run_episode.py --object abs_round_connector --task task1
    python scripts/run_episode.py --object steel_peg --task task1 --backend anthropic
    python scripts/run_episode.py --object glass_bowl --task task3 --gripper robotiq_2f140
"""

from __future__ import annotations

import argparse
import json
import logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single Force-Budgeted Recovery episode")
    p.add_argument("--object", default="abs_round_connector",
                   choices=["abs_round_connector", "steel_peg", "resin_planet_gear",
                            "metal_planet_gear", "glass_bowl", "ceramic_plate",
                            "metal_plate", "sturdy_mug"])
    p.add_argument("--task", choices=["task1", "task2", "task3"], default="task1")
    p.add_argument("--gripper", choices=["franka_panda", "robotiq_2f140"], default="franka_panda")
    p.add_argument("--k-max", type=int, default=5, help="Max recovery attempts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["anthropic", "local", "mock"], default="mock")
    p.add_argument("--local-url", default="http://localhost:11434/v1",
                   help="Base URL for local OpenAI-compatible server (--backend local)")
    p.add_argument("--local-model", default="qwen2.5:7b-instruct",
                   help="Model name served at --local-url (--backend local)")
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--show-json", action="store_true", help="Print full episode result as JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    from forge_plus.envs.mock_assembly_env import MockAssemblyEnv, MockEnvConfig
    from forge_plus.envs.object_configs import OBJECT_REGISTRY
    from forge_plus.envs.base_assembly_env import EpisodeConfig
    from forge_plus.episode import EpisodeRunner
    from forge_plus.llm.client import build_client
    from forge_plus.llm.budget_setter import BudgetSetter
    from forge_plus.llm.recovery_selector import RecoverySelector
    from forge_plus.encoding.signature_encoder import SignatureEncoder
    from forge_plus.recovery.recovery_actions import RecoveryActionExecutor
    from forge_plus.skills.forge_skill import FORGESkill, SkillConfig
    from forge_plus.skills.policy_network import PolicyConfig

    obj_cfg = OBJECT_REGISTRY[args.object]
    import random
    rng = random.Random(args.seed)
    f_break = obj_cfg.sample_f_break(rng)

    episode_cfg = EpisodeConfig(
        object_key=args.object,
        task_name=args.task,
        gripper=args.gripper,
        f_break_n=f_break,
        disturbance_seed=args.seed,
    )

    print(f"\n{'='*60}")
    print(f"  Force-Budgeted Recovery — Single Episode")
    print(f"{'='*60}")
    print(f"  Object:  {obj_cfg.identity.name}")
    print(f"  Task:    {args.task}")
    print(f"  Gripper: {args.gripper}")
    print(f"  F_break: {f_break:.1f} N  [EVALUATOR ONLY — hidden from agent]")
    print(f"  Backend: {args.backend}")
    print(f"{'='*60}\n")

    llm_cfg: dict = {"backend": args.backend}
    if args.backend == "local":
        llm_cfg["base_url"] = args.local_url
        llm_cfg["model"] = args.local_model
    llm_client = build_client(llm_cfg)
    env = MockAssemblyEnv(MockEnvConfig())
    skill = FORGESkill(SkillConfig(policy_cfg=PolicyConfig()))
    rec_exec = RecoveryActionExecutor()

    runner = EpisodeRunner(
        budget_setter=BudgetSetter(client=llm_client),
        recovery_selector=RecoverySelector(client=llm_client),
        signature_encoder=SignatureEncoder(),
        skill=skill,
        env=env,
        recovery_executor=rec_exec,
        k_max=args.k_max,
        verbose=args.verbose,
    )

    result = runner.run(episode_cfg)

    print(f"\n{'='*60}")
    print(f"  RESULT: {result.termination.value}")
    print(f"{'='*60}")
    print(f"  F_max set by LLM:   {result.f_max_n:.1f} N")
    print(f"  F_break (hidden):   {result.f_break_n:.1f} N")
    print(f"  Safety margin:      {result.safety_margin_n:.1f} N "
          f"({'SAFE' if result.safety_margin_n >= 0 else 'OVER-BUDGET'})")
    print(f"  Budget confidence:  {result.budget_confidence:.0%}")
    print(f"  Total attempts:     {result.total_attempts}")
    print(f"  Total steps:        {result.total_steps}")
    print(f"  Peak contact (N):   {result.peak_contact_n:.1f}")
    print(f"  Broke:              {result.broke}")
    print(f"  Succeeded:          {result.succeeded}")
    print(f"  Wall time:          {result.wall_time_s:.2f}s")

    if result.attempts:
        print(f"\n  Attempts:")
        for a in result.attempts:
            rec = a.recovery_action or "(none)"
            print(f"    [{a.attempt_idx}] steps={a.steps} outcome={a.outcome.value} "
                  f"recovery={rec} peak={a.peak_contact_n:.1f}N")

    if args.show_json:
        print("\n--- Full Episode Result (JSON) ---")
        d = result.__dict__.copy()
        d["termination"] = d["termination"].value
        d["attempts"] = [
            {**a._asdict(), "outcome": a.outcome.value}
            for a in d["attempts"]
        ]
        print(json.dumps(d, indent=2))


if __name__ == "__main__":
    main()
