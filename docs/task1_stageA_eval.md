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

| object | F_max (LLM) | F_break (hidden) | episodes | success | breakage | peak Fins mean/p95/max |
|---|---|---|---|---|---|---|
| abs_gear (trained) | 10 N | 38±5 N | 200 | **1.000** | **0.000** | 14.8 / 16.6 / 18.4 N |
| steel_gear (transfer) | 100 N | 230±20 N | 200 begun | **0.000** | 0.000 | — (0 completions) |

## Honest findings

1. **True insertion works**: 200/200 bore-on-shaft seats on the trained
   fragile object, zero breaks, worst-case contact 18.4 N < every sampled
   F_break (floor 20 N).
2. **Clamp overshoot is real**: peak insertion force (mean 14.8 N) EXCEEDS
   the 10 N budget during funnel entry — the soft ceiling (retreat above
   0.9×F_max) bounds but does not eliminate overshoot. This feeds the
   clamp-fidelity metric; margins to F_break absorbed it in every episode.
3. **Budget-conditioning transfer is one-directional**: the earlier
   steel-trained policy transferred DOWN to the fragile budget, but this
   fragile-trained policy does NOT transfer UP (F_cmd 100 N is far outside
   its FiLM conditioning distribution; zero episodes complete). Fix queued:
   mixed-object training (`--forge_obj -1`).
4. The insertion tolerance is the true FORGE fit: bore flats at r 5.0 mm
   over the r 4.75 mm shaft (0.25 mm clearance); the collision lead-in is a
   realistic 2 mm chamfer equivalent (see gear_medium_body.usda).
