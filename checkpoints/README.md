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
| `task1_gear_rq_bc3.pt` | **abs clean gate: 256/256 seats, 0 breaks, peak 16.1 N** (steel 1.6%) |
| `task1_gear_rq_bc4.pt` | **steel clean gate: 256/256 seats, 0 breaks, peak 28.6 N** (abs 0%) |
| `task1_gear_rq_bc7.pt` | best single ckpt: steel 32/32 @ 18.9 N, abs 27/32 (84%) |

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

## task3_* checkpoints

Bottle insertion / pick-place arcs (separate task; see the task3 session
notes and `docs/REPLICATION_PLAYBOOK.md`).
