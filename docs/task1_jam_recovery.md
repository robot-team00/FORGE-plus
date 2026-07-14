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

---

# Robotiq 2F-140 sweep (2026-07-14)

Same protocol on the ported gripper: fragile abs_gear, 5 mm in-grip slip
every episode, 25 episodes/cell, deterministic mean, budget mode `ours`.
Policy = `task1_gear_rq_uni.pt` (the bc3/bc7 weight-soup unification; clean
gates 256/256 both classes, 0 breaks). Recovery chain = the deferred-regrasp
rebuild (commit f81bb6a: post-lift seat pin, seat-arm hold, realized-ee fire
gate below the trained hand-off band, substep-true cooldown, protected
post-seat search window). Step cap **1600** (vs 700 for the Franka table):
one 2F-140 deferred-regrasp cycle — pend traverse + extended open-settle
seat + protected search — is ~450 steps, and the cap must fit ~3 cycles.

| recovery      | success | breaks | timeouts | attempts | peak Fins mean / max |
|---------------|--------:|-------:|---------:|---------:|---------------------:|
| **ours**      | **40%** (10/25) | 12% (3) | 48% (12) | 3.56 | 23.0 / 48.3 N |
| heuristic     | 0%      | 0%     | 100%     | 5.00 | 12.5 / 18.0 N |
| vision_llm    | 28% (7/25) | 12% (3) | 60% (15) | 3.88 | 19.0 / 43.6 N |
| press_harder  | 0%      | 0%     | 100%     | 5.00 | 11.3 / 16.5 N |
| none          | 0%      | **20% (5)** | 80% (20) | 0 | 10.2 / 50.5 N |

Reading the table:

- The mechanism transfers: the slip leaves the gear tilted **in the
  four-bar pinch**, and only the signature chain routes to `regrasp` (via
  the recurring-hover rule), the one maneuver that fixes an in-grip tilt.
  `ours` signatures: {hover 40, wedge 30, friction 19} — it is the only
  cell that ever reaches deep wedge contact, because it is the only one
  that ever restores an insertable grip.
- **heuristic** never reads the hover signature (its cells see only
  {hover, friction}), maps everything to wiggle/rotate, never regrasps: 0%.
- **vision_llm** (random menu) draws `regrasp` by luck: 28% vs our 40% —
  the force signature is again worth the gap, though narrower than on the
  Franka (64% vs 32%) because the 2F-140 regrasp cycle is long enough that
  wasted wrong-maneuver attempts burn the step budget.
- **press_harder shows a different face of the same anti-pattern.** On the
  Franka it destroyed 24/25 gears; here it breaks nothing — because its
  only maneuver (re-approach with an escalated ceiling) never fixes the
  in-grip tilt, so the tilted bore never takes load and the raised F_max
  never engages (peak 16.5 N, 100% timeouts). Escalation is futile here
  rather than destructive; the destructive face shows up in `none`.
- **none** is not benign on this gripper: with no recovery interrupting it,
  the policy's own descent eventually presses the tilted bore into the
  shaft tip — 5/25 fragile breaks at up to 50.5 N. The jam is real, and
  doing nothing is worse than a wrong maneuver.

## Honest caveats (2F-140)

- **The step cap differs from the Franka table** (1600 vs 700, reason
  above), so timeout rates are not directly comparable across the two
  tables. Within the 2F-140 table all cells share the cap.
- **48% timeouts in `ours`** is the current frontier, not breakage: the
  cap fits ~3 recovery cycles and some episodes need more (false-positive
  detector churn during the post-seat funnel search burns ~450-step
  cycles; in the 10-episode smoke, 3 of 5 timeouts were the older
  1000-step cap cutting a second cycle mid-flight — hence the raise).
- **Recovery episodes press harder than clean ones**: 23.0 N mean / 48.3 N
  max vs the uni clean gate's 15.9 / 36.7 — that envelope exposure is the
  3 fragile breaks (F_break 38±5). Same story as the Franka row; the
  ceiling is never raised.
- On **steel** (robust class) the same chain recovers 25/25 with 0 breaks
  and 0 timeouts, but recovery episodes press 85 N mean / 135 N max vs
  clean 35.6 / 62.5 — fine inside steel's 100 N budget, but the margin is
  consumed by the recovery seating press.
- `attempts` no longer saturates (3.56 mean in `ours` vs 5.00 on the
  Franka table): the deferred-regrasp pend state suppresses the cosmetic
  hover re-fires that inflated the Franka counter.

## Reproduce (2F-140)

```bash
# single cell
/workspace/.venv/bin/python scripts/eval_gear_jam.py \
    --gripper robotiq_2f140 --recovery ours --obj 0 --slip_mm 5 \
    --episodes 25 --ckpt checkpoints/task1_gear_rq_uni.pt
# full table
bash scripts/sweep_jam_recovery.sh checkpoints/task1_gear_rq_uni.pt 25 robotiq_2f140
```
