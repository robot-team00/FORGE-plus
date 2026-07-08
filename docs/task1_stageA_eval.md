# Task 1 stage-A eval — learned FORGE gear insertion (Franka)

Date: 2026-07-08. Checkpoint: `checkpoints/task1_gear_insert_franka.pt`
(train_gear_insert.py --forge --forge_obj 1 --num_envs 512 --iterations 600).

Deterministic policy mean, headless (`scripts/eval_gear_insert.py`, 128 envs):

| object | F_max (LLM, identity-only) | F_break (hidden) | episodes | success | breakage |
|---|---|---|---|---|---|
| steel_gear (obj 1, trained) | 100 N | 230±20 N | 200 | **1.000** (200/200) | 0.000 |
| abs_gear (obj 0, transfer)  | 10 N  | 38±5 N   | 203 | **1.000** (203/203) | 0.000 |

Success = gear bore seated down the middle shaft (|origin−seat| < 6 mm,
in-cell xy, 6-step settle). Breakage gated on the insertion-only
gear↔base contact force vs the hidden per-episode F_break sample.

Training-dynamics note: the skill converged only after the hand-off was
staged to FORGE-faithful few-mm noise (gear-aim + 3 mm arrival). The
learned behavior descends gently (insertion force fell from a 19 N
shaft-tip press to <1 N hover-and-seat after the it-120 breakthrough)
and every terminating episode from it 120 onward was a success.
