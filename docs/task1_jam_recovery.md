# Task 1 — jam-recovery status (machinery proven, completion pending)

Date: 2026-07-09. Commit 2c74e62 line of work; policy = `task1_gear_mixed_strict.pt`.

## What is demonstrated and working

- **Jam induction**: an in-grip slip disturbance (`cfg.slip_disturb_mm`,
  `eval_gear_jam.py --slip_mm`) fired as the gear enters the funnel. The
  slip itself snaps back within a substep (the pinch re-centers the hub),
  but its transient knocks the descent ~2 mm off at entry — producing a
  genuine wedge: bore rim caught in the funnel, sustained 9–15 N press,
  no descent.
- **Detection on the honest channel**: jams detect on the pairwise
  gear↔base insertion force (the whole-body channel is polluted by the
  ~10 N hub squeeze and produced constant false jams — fixed).
- **Signature discrimination**: wedge (steady lateral bias, often rising
  axial) vs friction (axial-only) signatures computed from the
  insertion-force window; both observed in eval logs.
- **Signature-driven selection with escalation**: rotate_align on a
  first wedge; a wedge that *survives* a realign escalates to regrasp
  (in-hand slip is not fixable by arm maneuvers); abort at attempt ≥4.

## What is not yet demonstrated

- **Recovery completion** (re-seat after a wedge): 0/6 in the latest
  smoke. The re-approach lands at the policy's natural ~2 mm precision;
  threading the funnel from a wedge needs either finer post-recovery
  alignment or a policy trained WITH jam/recovery experience (the current
  skill never saw a wedge state in training — recovery states are
  out-of-distribution). A 300-it mixed fine-tune on the ramp asset was
  tried and REJECTED: it raised the fragile-object force tail to 37 N and
  produced a break; the active checkpoint remains mixed-round-1.
- The recovery-policy baseline table (ours/heuristic/vision_llm/
  press_harder) therefore awaits completion-capable recovery.

## Next steps (in order of expected value)

1. Train WITH the slip disturbance active (jam states enter the training
   distribution; the policy learns its own re-approach) — the FORGE-native
   route, likely subsumes the scripted maneuvers.
2. Wedge-vs-friction discrimination demo can proceed independently: both
   signatures already appear in `eval_gear_jam.py` logs.
