# Task 1 — jam recovery: SOLVED, with the recovery-baseline table

Date: 2026-07-11. Commit 037bff1 line of work; policy =
`task1_gear_sliprand.pt.it300` (clean gate: 200/200 strict seats, 0 breaks,
Fins 13.8 N mean / 15.9 p95 / 17.6 max).

## The mechanism (what a slip actually does)

The in-grip slip disturbance (`--slip_mm 5`) teleports the gear 5 mm in the
pinch. The **position** snaps back within a substep (the pads re-center the
hub) — but the **orientation does not**: the asymmetric squeeze torques the
gear to a new wedged equilibrium, leaving it **tilted 7–10° in the grip**
permanently (probe: 4.6° → 10.1° at the kick, never decays). A 7–10° bore
cannot thread the 0.4 mm-clearance fit band (~3–4° max while threaded), so
the deterministic policy backs off and hovers with **zero contact force** —
it is correctly refusing a geometrically impossible insertion, which made
the failure invisible to any force-threshold jam detector.

## The recovery chain

All manipulation is the learned policy; recovery = discrete maneuvers from
the fixed menu; F_max is never raised.

1. **Contactless-hover branch** in `is_failure()`: near the cell, above the
   seat, no descent, peak < 1 N — as stuck as a wedge, previously invisible.
   (Windows straddling a recovery lift are excluded — settling is not
   hovering.)
2. **Signature-driven selection**: a first hover → `retract_and_reapproach`.
   A hover that *recurs* after the re-approach → **`regrasp`** — no arm
   maneuver changed anything, so the anomaly travels with the part. The
   re-seat restores the canonical upright in-grip pose (9.8° → 0.2°).
3. **Arrival-gated re-approach** after retract/regrasp: the scripted setup
   controller (the same one the benchmark uses at episode start) carries the
   hand back over the shaft; the learned policy resumes from its trained
   hand-off state and does the force-guided search + insertion itself.
4. The frozen post-recovery aim (`_fk_aim`) is retired: with regrasp
   removing the in-hand offset, a residual aim bias only displaced the
   policy's search center (a +3 mm aim pinned an otherwise healthy 275-step,
   12–15 N search outside the 1–1.5 mm capture radius).

Smoke (6 episodes): **6/6 recovered and seated, 0 breaks, 0 timeouts** (from
0/6 before the chain).

## Recovery-baseline table (issue #26)

`bash scripts/sweep_jam_recovery.sh` — fragile abs_gear (F_break 38±5 N),
5 mm in-grip slip every episode, 25 episodes/cell, deterministic policy
(`task1_gear_sliprand.pt.it300`), budget mode `ours` (identity-only LLM
ceiling; the ForceClamp caps commanded force).

| recovery      | success | breaks | timeouts | peak Fins mean / max |
|---------------|--------:|-------:|---------:|---------------------:|
| **ours**      | **64%** (16/25) | 12% (3) | 24% (6) | 23.2 / 31.5 N |
| heuristic     | 0%      | 0%     | 100%     | 0.0 / 0.0 N |
| vision_llm    | 32% (8/25) | 0%  | 68%      | 15.0 / 31.6 N |
| press_harder  | 4% (1/25) | **96% (24)** | 0% | 36.1 / 44.8 N |
| none          | 0%      | 0%     | 100%     | 0.0 / 0.0 N |

Reading the table:

- **ours** (force-signature → RecoverySelector): recovers two-thirds of
  slip jams. The signature chain is what finds `regrasp` — the only
  maneuver that fixes an in-grip tilt.
- **heuristic** (hand rule, never reads the full signature): maps every
  zero-force hover to `wiggle_search`; the tilt is never fixed; 0%.
- **vision_llm** (random menu proxy for pixels-only): succeeds roughly as
  often as it happens to pick `regrasp` — the force signature is worth the
  difference between 32% and 64%.
- **press_harder** (raise F_max ×1.25 per attempt): the anti-pattern the
  benchmark exists to demonstrate — **24 of 25 fragile gears destroyed**
  (peaks to 44.8 N vs F_break 38±5). More force is not a recovery strategy.
- **none**: the policy alone never escapes the post-slip state; the jam
  is real and recovery is doing the work.

## Honest caveats

- **`attempts` saturates at 5 in every cell** and is omitted from the
  table: residual settle-churn fires cosmetic hover attempts (including
  no-op `abort`s) before the productive descent, so the counter carries no
  signal yet. Tightening the hover window (or making `abort` terminal)
  would fix the metric, not the behavior.
- **Recovery episodes press harder than clean ones**: 23.2 N mean / 31.5 N
  max vs clean 13.8 / 17.6. That exposes low draws of the F_break
  distribution (38±5) — the 3 breaks in `ours`. The ceiling is never
  raised; this is the policy + clamp operating at its envelope during
  post-recovery seating. A gentler post-recovery descent profile is the
  obvious next refinement if the 12% break rate matters downstream.
- The 6/6 smoke vs 64% at n=25 is ordinary episode variance plus the
  timeout tail (6 episodes ran out of the 700-step budget mid-recovery).

## Reproduce

```bash
# single cell
/workspace/.venv/bin/python scripts/eval_gear_jam.py \
    --recovery ours --obj 0 --slip_mm 5 --episodes 25 \
    --ckpt checkpoints/task1_gear_sliprand.pt.it300
# full table
bash scripts/sweep_jam_recovery.sh checkpoints/task1_gear_sliprand.pt.it300 25
```
