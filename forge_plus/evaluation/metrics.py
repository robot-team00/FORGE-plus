"""Evaluation metrics for Force-Budgeted Recovery (§10 of the proposal).

All metrics are computed from a list of EpisodeResult objects.
The breakage metric is non-circular: F_break is only in EpisodeResult.f_break_n
(evaluator field) and never in the signature or any agent input.

Metrics:
  1. closed_loop_success_rate       — fraction seated within K attempts
  2. breakage_rate                  — fraction with |F_contact| > F_break
  3. budget_appropriateness         — distribution of m = F_break - F_max
  4. recovery_efficacy              — success | recovery invoked; attempts-to-success
  5. force_economy                  — peak and time-integrated contact force
  6. clamp_fidelity                 — clamp-vs-contact overshoot
  7. cross_gripper_transfer         — consistency of F_max and breakage across grippers
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from forge_plus.episode import EpisodeResult, EpisodeTermination


@dataclass
class EvaluationMetrics:
    # 1. Success
    closed_loop_success_rate: float = 0.0
    mean_attempts_to_success: float = 0.0

    # 2. Breakage (non-circular — uses hidden f_break only in post-hoc eval)
    breakage_rate: float = 0.0
    fragile_breakage_rate: float = 0.0
    robust_breakage_rate: float = 0.0

    # 3. Budget appropriateness
    mean_safety_margin_n: float = 0.0
    std_safety_margin_n: float = 0.0
    over_budget_rate: float = 0.0          # F_max > F_break (dangerous)
    under_budget_rate: float = 0.0         # F_max too low to seat (wasteful)
    median_safety_margin_n: float = 0.0

    # 4. Recovery efficacy
    recovery_invoked_rate: float = 0.0
    success_given_recovery: float = 0.0    # P(success | recovery invoked)
    action_distribution: dict[str, float] = field(default_factory=dict)

    # 5. Force economy
    mean_peak_contact_n: float = 0.0
    mean_mean_contact_n: float = 0.0

    # 6. Clamp fidelity
    mean_clamp_overshoot_n: float = 0.0
    max_clamp_overshoot_n: float = 0.0
    overshoot_rate: float = 0.0

    # 7. Cross-gripper transfer (filled by compare_grippers)
    gripper_success_rates: dict[str, float] = field(default_factory=dict)
    gripper_breakage_rates: dict[str, float] = field(default_factory=dict)
    f_max_consistency_n: float = 0.0      # |mean(F_max_panda) - mean(F_max_robotiq)|

    # Failure-mode breakdown (place/stack: over_press | edge_load | tip | under_seat)
    failure_mode_distribution: dict[str, float] = field(default_factory=dict)

    # Meta
    n_episodes: int = 0
    n_fragile: int = 0
    n_robust: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def compute_metrics(
    results: list[EpisodeResult],
    fragile_object_keys: set[str] | None = None,
    under_budget_min_margin_n: float = 5.0,  # if margin > this, consider under-budgeted
) -> EvaluationMetrics:
    """Compute all evaluation metrics from a list of episode results."""
    if not results:
        return EvaluationMetrics()

    m = EvaluationMetrics(n_episodes=len(results))

    # Split fragile vs. robust
    fragile_keys = fragile_object_keys or {
        "abs_round_connector", "resin_planet_gear", "glass_bowl", "ceramic_plate"
    }
    fragile = [r for r in results if r.object_key in fragile_keys]
    robust = [r for r in results if r.object_key not in fragile_keys]
    m.n_fragile = len(fragile)
    m.n_robust = len(robust)

    # 1. Success rate
    successes = [r for r in results if r.succeeded]
    m.closed_loop_success_rate = len(successes) / len(results)
    attempts_to_success = [r.total_attempts for r in successes]
    m.mean_attempts_to_success = float(np.mean(attempts_to_success)) if attempts_to_success else float("nan")

    # 2. Breakage rate
    broken = [r for r in results if r.broke]
    m.breakage_rate = len(broken) / len(results)
    m.fragile_breakage_rate = (
        len([r for r in fragile if r.broke]) / len(fragile) if fragile else float("nan")
    )
    m.robust_breakage_rate = (
        len([r for r in robust if r.broke]) / len(robust) if robust else float("nan")
    )

    # 3. Budget appropriateness
    margins = [r.safety_margin_n for r in results]
    m.mean_safety_margin_n = float(np.mean(margins))
    m.std_safety_margin_n = float(np.std(margins))
    m.median_safety_margin_n = float(np.median(margins))
    m.over_budget_rate = float(np.mean([r.over_budget for r in results]))
    # Under-budget: margin is positive but too large (left success on the table)
    # Proxy: episode failed (not broken, not success) and margin > threshold
    m.under_budget_rate = float(
        np.mean([
            not r.succeeded and not r.broke and r.safety_margin_n > under_budget_min_margin_n
            for r in results
        ])
    )

    # 4. Recovery efficacy
    with_recovery = [r for r in results if any(a.recovery_action for a in r.attempts)]
    m.recovery_invoked_rate = len(with_recovery) / len(results)
    m.success_given_recovery = (
        len([r for r in with_recovery if r.succeeded]) / len(with_recovery)
        if with_recovery else float("nan")
    )
    # Action distribution
    action_counts: dict[str, int] = {}
    for r in results:
        for a in r.attempts:
            if a.recovery_action:
                action_counts[a.recovery_action] = action_counts.get(a.recovery_action, 0) + 1
    total_actions = sum(action_counts.values())
    m.action_distribution = (
        {k: v / total_actions for k, v in action_counts.items()} if total_actions else {}
    )

    # 5. Force economy
    m.mean_peak_contact_n = float(np.mean([r.peak_contact_n for r in results]))
    m.mean_mean_contact_n = float(np.mean([r.mean_contact_n for r in results]))

    # 6. Clamp fidelity
    m.mean_clamp_overshoot_n = float(np.mean([r.clamp_overshoot_mean_n for r in results]))
    m.max_clamp_overshoot_n = float(np.max([r.clamp_overshoot_max_n for r in results]))
    m.overshoot_rate = float(np.mean([r.clamp_overshoot_max_n > 0 for r in results]))

    # 7. Cross-gripper transfer
    for gripper in set(r.gripper for r in results):
        g_results = [r for r in results if r.gripper == gripper]
        m.gripper_success_rates[gripper] = len([r for r in g_results if r.succeeded]) / len(g_results)
        m.gripper_breakage_rates[gripper] = len([r for r in g_results if r.broke]) / len(g_results)
    if "franka_panda" in m.gripper_success_rates and "robotiq_2f140" in m.gripper_success_rates:
        panda_fmax = np.mean([r.f_max_n for r in results if r.gripper == "franka_panda"])
        robotiq_fmax = np.mean([r.f_max_n for r in results if r.gripper == "robotiq_2f140"])
        m.f_max_consistency_n = float(abs(panda_fmax - robotiq_fmax))

    fm_counts: dict[str, int] = {}
    for r in results:
        fm = getattr(r, "failure_mode", None)
        if fm and not r.succeeded:
            fm_counts[fm] = fm_counts.get(fm, 0) + 1
    total_fm = sum(fm_counts.values())
    m.failure_mode_distribution = (
        {k: v / total_fm for k, v in fm_counts.items()} if total_fm else {}
    )

    return m


def print_metrics_table(m: EvaluationMetrics) -> None:
    """Pretty-print a metrics summary."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Force-Budgeted Recovery — Evaluation Metrics", show_header=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right")

    rows = [
        ("Episodes", str(m.n_episodes)),
        ("", ""),
        ("1. Closed-loop success rate", f"{m.closed_loop_success_rate:.1%}"),
        ("   Mean attempts to success", f"{m.mean_attempts_to_success:.2f}"),
        ("", ""),
        ("2. Breakage rate (all)", f"{m.breakage_rate:.1%}"),
        ("   Fragile breakage", f"{m.fragile_breakage_rate:.1%}"),
        ("   Robust breakage", f"{m.robust_breakage_rate:.1%}"),
        ("", ""),
        ("3. Mean safety margin (F_break - F_max)", f"{m.mean_safety_margin_n:.1f} N"),
        ("   Over-budget rate (F_max > F_break)", f"{m.over_budget_rate:.1%}"),
        ("   Under-budget rate", f"{m.under_budget_rate:.1%}"),
        ("", ""),
        ("4. Recovery invoked rate", f"{m.recovery_invoked_rate:.1%}"),
        ("   Success | recovery invoked", f"{m.success_given_recovery:.1%}"),
        ("", ""),
        ("5. Mean peak contact force", f"{m.mean_peak_contact_n:.1f} N"),
        ("", ""),
        ("6. Mean clamp overshoot", f"{m.mean_clamp_overshoot_n:.2f} N"),
        ("   Max clamp overshoot", f"{m.max_clamp_overshoot_n:.2f} N"),
    ]
    for name, val in rows:
        table.add_row(name, val)

    console.print(table)
