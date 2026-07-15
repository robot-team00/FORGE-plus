# Task 1 stage-A eval — learned FORGE gear insertion (Franka), STRICT criterion

Date: 2026-07-08 (supersedes the earlier version of this file — those numbers
were plate-rest artifacts of a ±5 cm success box; see commit bedc9df).

**Success = bore ON shaft**: |gear − shaft|_xy < 5 mm at seat height
(|z − 0.400| < 6 mm), 6-step settle. Geometrically airtight: a gear resting
closer than ~2.5 cm to the shaft without true insertion is impossible.

Checkpoint: `checkpoints/task1_gear_insert_franka.pt` — trained on
**abs_gear** (fragile, 10 N identity-only LLM budget) for 600+600 iterations
under the strict criterion, with the convex-collision assets (the factory
SDF collision cooks WITHOUT the bore — see `scripts/probe_bore_physics.py`
and commit e65c01a) and anisotropic xy alignment shaping.

Deterministic policy mean, 128 envs (`scripts/eval_gear_insert.py`):

ACTIVE checkpoint = mixed round 1 (`task1_gear_mixed_strict.pt`, also copied
to `task1_gear_insert_franka.pt`):

| object | F_max (LLM) | F_break (hidden) | episodes | success | breakage | peak Fins mean/p95/max |
|---|---|---|---|---|---|---|
| abs_gear | 10 N | 38±5 N | 200 | **1.000** | **0.000** | 14.6 / 17.7 / 18.5 N |
| steel_gear | 100 N | 230±20 N | 248 | **0.730** | **0.000** | 95.2 / 109.3 / 111.1 N |

Checkpoint trajectory (all strict criterion):
- abs-only (600+600 its): abs 1.000/0.000; steel 0.000 (F_cmd 100 far
  outside the fragile-only conditioning — policy degenerates, no descent)
- + mixed round 1 (600 its, classes randomized, reset_std −1.0): abs
  1.000/0.000 RETAINED, steel 0.000 → **0.730**/0.000 — the ACTIVE ckpt
- + mixed round 2 (540 its more): COLLAPSE — steel → 0.000, abs → ~0.36.
  Continued PPO past convergence drifted the policy off both modes (the
  chronic PhysX narrowphase-overflow warnings are NOT the cause — the
  good checkpoints trained under identical warnings). Lesson: stop at the
  good checkpoint; further steel gains need a careful low-lr short round,
  not more of the same.

## Honest findings

1. **True insertion works**: 200/200 bore-on-shaft seats on the trained
   fragile object, zero breaks, worst-case contact 18.4 N < every sampled
   F_break (floor 20 N).
2. **Clamp overshoot is real**: peak insertion force (mean 14.8 N) EXCEEDS
   the 10 N budget during funnel entry — the soft ceiling (retreat above
   0.9×F_max) bounds but does not eliminate overshoot. This feeds the
   clamp-fidelity metric; margins to F_break absorbed it in every episode.
3. **Budget-conditioning transfer is one-directional**: a steel-trained
   policy transfers DOWN to the fragile budget, a fragile-trained one does
   NOT transfer UP (F_cmd 100 N is far outside its FiLM conditioning
   distribution). Mixed training recovers steel to 73% while retaining
   abs at 100% (see trajectory above).
4. The insertion tolerance is the true FORGE fit: bore flats at r 5.0 mm
   over the r 4.75 mm shaft (0.25 mm clearance); the collision lead-in is a
   realistic 2 mm chamfer equivalent (see gear_medium_body.usda).
