# Task 1 — budget-setter baselines under the STRICT seat criterion

Date: 2026-07-09 (supersedes the loose-criterion version; commit bedc9df).
Policy: `checkpoints/task1_gear_insert_franka.pt` (trained on abs_gear at the
10 N identity-only LLM budget, strict bore-on-shaft criterion, convex-bore
assets). Deterministic mean, 128 envs, ~200 episodes/cell.

## abs_gear (fragile, F_break 38±5 N, floor 20 N)

| budget | F_max | success | breakage | peak Fins mean/p95/max (N) | margin | over-budget eps |
|---|---|---|---|---|---|---|
| **ours** (LLM, identity-only) | 10 N | **1.000** | **0.000** | 14.9 / 16.8 / 20.2 | +28.0 | 0/200 |
| oracle (F_break−5) | ~33 N | 0.502 | **0.498** | 33.9 / 44.9 / 53.8 | +5.0 | 0/201 |
| fixed_global | 60 N | 0.000 | **1.000** | 34.6 / 42.2 / 47.1 | −21.4 | 205/205 |
| no_ceiling | 120 N | 0.000* | 0.000* | — | −82 | 200/200 |

\* no_ceiling: zero episode completions — F_cmd=1.0 conditioning is far
outside the policy's training distribution and behavior degenerates (no
descent to contact). Non-functional rather than safe.

## Findings vs the issue-#26 expected ordering

**ours >> oracle > fixed_global** — stronger separation than expected, and
one instructive inversion:

1. **Ours: 100% seat, 0% break.** The conservative identity-only budget
   (10 N vs 38 N F_break) absorbs the skill's funnel-entry overshoot
   (peaks ~1.5× budget) with room to spare.
2. **The oracle breaks half the gears.** F_break−ε is only "optimal" if
   clamp fidelity is perfect: conditioned at 33 N, the skill's overshoot
   peaks at 45-54 N — past the 38±5 N breaking force in half the episodes.
   Budget appropriateness must cover the overshoot distribution, not just
   sit under F_break. This is the clamp-fidelity metric made visible.
3. **Fixed-global 60 N destroys every part** (100% breakage) — the
   canonical argument for per-object budgets.
4. **Budget conditioning is part of the skill's operating envelope**: far-
   OOD budgets (120 N; also steel's 100 N) produce degenerate behavior,
   not just unsafe behavior. Mixed-budget training is queued to widen the
   envelope (then the steel row and a re-run of this table follow).
