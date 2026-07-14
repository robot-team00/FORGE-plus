# Checkpoint manifest — task1 gear insertion (FORGE GearMesh, issue #26)

All policies are `ForceConditionedPolicy` (obs 34, act 7, f_cmd-conditioned;
`forge_plus/skills/policy_network.py`). `.itN` files are training snapshots
every 100 iterations. Evaluate with `scripts/eval_gear_insert.py`
(deterministic clean gate) or `scripts/triage_rq_ckpts.py` (one-boot
multi-checkpoint triage). Filenames are load-bearing — scripts and session
notes reference them; do not rename.

## Deliverables (Robotiq 2F-140, final plant)

| ckpt | result |
|---|---|
| `task1_gear_rq_uni.pt` | **UNIFIED single ckpt, BOTH clean gates: abs 256/256, 0 breaks, peak 15.9 N mean / 18.5 p95; steel 256/256, 0 breaks, 35.6 N mean / 57.6 p95; 0 over-budget eps** |
| `task1_gear_rq_uni_rel.pt` | **uni + LEARNED RELEASE (act 8, `release_obs_head`): release gates (success = released + standing seated + hand clear) abs 256/256 0 brk 0 bad-rel, peak 15.7/18.5/20.1 N; steel 256/256 0 brk 0 bad-rel, 35.7/59.3/69.6 N. Arm dims + trunk BIT-IDENTICAL to uni (only the input-skip release Linear trains — `scripts/train_release_head.py`, data `scripts/collect_gear_release.py`); zero-FPR operating point, 255/256 window coverage** |
| `task1_gear_rq_bc3.pt` | abs clean gate: 256/256 seats, 0 breaks, peak 16.1 N (steel 1.6%) |
| `task1_gear_rq_bc4.pt` | steel clean gate: 256/256 seats, 0 breaks, peak 28.6 N (abs 0%) |
| `task1_gear_rq_bc7.pt` | best single pre-soup ckpt: steel 32/32 @ 18.9 N, abs 27/32 (84%) |

`task1_gear_rq_uni.pt` is a WEIGHT SOUP: `0.15*bc3 + 0.85*bc7`
(`scripts/make_soup.py`; valid because bc7 was warm-started from bc3 — same
basin; every alpha 0.10-0.75 passes triage, steel peak force is monotone in
alpha, 0.15 minimizes the joint force tails). Pod-only evidence lineages:
`task1_gear_rq_pp1.pt(.itN)` — tiny-std PPO polish of bc7 (std 0.05, lr 1e-4,
value warmup): abs destroyed 0/32 by it25 at every snapshot; 1/std^2 gradient
amplification makes tiny-std PPO move the mean too coarsely. `soup_b3b7_a*` /
`soup_b3b4_a*` — the alpha grids (b3b4 fails everywhere: cold-refit pair).

## BC / DAgger lineage (the working method: expert demos -> DAgger -> BC)

`task1_gear_rq_bc.pt` (plain BC — closed-loop failure, 30/32 broke) ->
`bc2` (+DAgger r1: first deterministic seats, 4/32) -> `bc3` (+r2) ->
`bc4` (+steel r3 + expert steel demos) -> `bc5`-`bc8` (balancing/warm-start
attempts at single-ckpt unification; bc7 = best, bc8 regressed — the abs
class sits on its own break floor). Demo data (pod-only):
`/workspace/logs/gear_demo_*.npz`; retrain with `scripts/bc_gear_mean.py`.

## Failed PPO-on-robotiq runs (kept as evidence: PPO cannot learn this task)

- `task1_gear_rq.pt` (+its) — steel-only fine-tune, broken plant
- `task1_gear_rq_abs.pt`, `task1_gear_rq_mixed.pt` (+its) — mixed rounds, broken plant
- `task1_gear_rq_v3.pt` (+its) — ori_k 1200 plant fix round, broken plant
- `task1_gear_rq_long.pt` (.it100-.it1200) — 1201-it consolidation, broken plant
- `task1_gear_rq_anneal.pt` (+its) — std-ceiling anneal 0.98->0.10, broken plant
- `task1_gear_rq_v4.pt` (+its) — FIXED plant, std 0.30->0.10: 0 seats
- `task1_gear_rq_v5.pt` (+its) — FIXED plant, std 0.50: ram-breaks -> disengagement
- `task1_gear_rq_v6.pt` (+its) — + contact leak: 0 seats
- `task1_gear_rq_v7.pt` (+its) — + slide-search authority: 0 seats

The funnel tolerance (~1.5 mm capture, 0.25 mm bore clearance at mu 1.0)
sits far below workable exploration noise (0.12 action noise already breaks
40/64 and seats 1/320 — DART collection).

## Franka lineage (bases; trained/evaluated on the franka gripper)

- `task1_gear_insert_franka.pt` — original franka insertion policy
  (`_LOOSE_INVALID` variant: seat criterion bug, do not use)
- `task1_gear_abs_strict.pt`, `task1_gear_mixed_strict.pt`, `task1_gear_mixed2.pt`,
  `task1_gear_ramp_ft.pt` — strict-seat-criterion retrain arc (d543102)
- `task1_gear_sliprand.pt` (+its) — slip-disturbance robust franka policy;
  **`.it300` is the resume/init base for every rq run and the BC lineage**

## task3_* checkpoints (bottle pick-place / insertion arcs)

Family-level map (see `docs/REPLICATION_PLAYBOOK.md` and the task3 session
notes for the full histories):

- `task3_place_*` — the FrankaPickPlace place-only arc (OSC / jacobian /
  smoothing iterations); `task3_place_solved_97succ.pt` = the ~97%
  gentle-place milestone, `task3_place_final.pt` / `task3_place_committed.pt`
  = the arc's endpoints.
- `task3_pick_place_franka.pt` — full pick+place
  (`.bak_weightsonly` = weights-only backup of the same).
- `task3_wine_bottle.pt`, `task3_wine_bottle_topdown.pt` — wine-cellar
  peg-in-hole insert (bottle into rack cell, ends vertical).
- `task3_faithful_grasp.pt` — real-friction grasp of the ceramic tumbler
  (flat-faces-only grip rule arc).
- `task3_extrinsic.pt`, `task3_extrinsic_A.pt` — extrinsic-dexterity place
  strategy variants.
- `task3_forge_entrance/handoff/handoff_descent/insert*` — the LEARNED
  FORGE insertion arc (entrance hand-off staging -> policy-driven insert).
- `task3_forge_release*.pt` — learned insert+release (8-dim action,
  finger-open fix; the PR #40 arc; `_v1`-`_v3` = iterations,
  `task3_forge_release.pt` = final).
- `task3_forge_robotiq.pt`, `task3_robotiq_2f140.pt`,
  `task3_rq_trainsmoke*.pt` — the 2F-140 bottle port + retrain arc
  (over-break force fixes, take-91 video; committed 0a80d6f).
- `task3_franka_panda.pt` — franka base for the bottle task.
- `task3_pp_smoke.pt`, `task3_forge_smoke.pt`, `task3_forge_dbg.pt` —
  pipeline smoke/debug checkpoints.
