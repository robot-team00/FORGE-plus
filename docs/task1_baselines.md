# Task 1 — budget-setter baselines (clean insertion)

Date: 2026-07-08. Policy: `checkpoints/task1_gear_insert_franka.pt` (stage A,
trained on steel_gear at the identity-only LLM budget). Deterministic mean,
128 envs, ~200 episodes per cell (`scripts/eval_gear_insert.py --budget ...`).
Budget modes: **ours** = identity-only LLM (abs 10 N / steel 100 N, cached);
**oracle** = F_break − 5 N (evaluator-side cheat — the only place the hidden
F_break may reach a budget); **fixed_global** = 60 N for every object;
**no_ceiling** = 120 N (the controller hard cap, i.e. no per-object budget).

## abs_gear (fragile, F_break 38±5 N, floor 20 N)

| budget | success | breakage | peak Fins mean/p95/max (N) | margin (N) | over-budget eps |
|---|---|---|---|---|---|
| **ours** | 1.000 (204/204) | 0.000 | 0.61 / 2.76 / **7.91** | **+28.5** | **0/204** |
| oracle | 1.000 (200/200) | 0.000 | 0.43 / 2.00 / 9.92 | +5.0 | 0/200 |
| fixed_global | 1.000 (201/201) | 0.000 | 0.51 / 1.83 / 16.91 | **−21.4** | **201/201** |
| no_ceiling | 1.000 (200/200) | 0.000 | 5.61 / 16.31 / **21.74** | **−82.3** | 200/200 |

## steel_gear (robust, F_break 230±20 N)

| budget | success | breakage | peak Fins mean/p95/max (N) | margin (N) | over-budget eps |
|---|---|---|---|---|---|
| **ours** | 1.000 (200/200) | 0.000 | 2.41 / 11.26 / 16.93 | +131.5 | 0/200 |
| oracle | 1.000 (200/200) | 0.000 | 5.22 / 16.40 / 21.09 | +106.3 | 0/200 |
| fixed_global | 1.000 (200/200) | 0.000 | 0.63 / 8.18 / 17.79 | +170.9 | 0/200 |
| no_ceiling | 1.000 (200/200) | 0.000 | 5.59 / 16.05 / 20.01 | +110.1 | 0/200 |

## Reading vs the issue-#26 expected ordering

On **clean** insertions every budget mode seats the gear (the stage-A skill is
gentle by training), so closed-loop success does not separate the baselines.
The separation is exactly where the task's design predicts — **budget
appropriateness and force economy on the fragile object**:

- **ours ≈ oracle** (both 0 over-budget; ours actually keeps a larger margin
  and the lowest worst-case contact, 7.9 N vs 9.9 N) —
- **>> fixed_global**: every ABS episode runs with a budget 21 N *above* the
  breaking force; worst-case contact doubles to 16.9 N —
- **> no_ceiling**: the F_cmd-conditioned policy presses ~9× harder (mean
  5.6 N vs 0.6 N) for zero success benefit, and its worst case (21.7 N)
  reaches the ABS breaking-force *floor* (20 N). Breakage stayed 0 only
  because P(F_break < 21.7 N) ≈ 0.1% per episode at these margins.

Outcome-level (breakage/recovery) separation, and the remaining baselines
(press_harder, heuristic, vision_llm — recovery-policy variants), require the
**jam scenarios** (induced misalignment beyond the funnel capture radius),
where over-budget presses meet real resistance. That is the next eval phase,
together with the wedge-vs-friction signature discrimination.
