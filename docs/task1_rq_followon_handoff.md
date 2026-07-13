# Task1 Robotiq 2F-140 — follow-on session: single-ckpt unification, recovery tuning, sweep row

All work in `/workspace/FORGE-plus_task3`, branch `task3` (do NOT touch the
task1 clone/branch — parallel session). Read `checkpoints/README.md` first
(full checkpoint manifest), plus the auto-memory `task1-robotiq-gear-port.md`
— it has the entire history: the 19-probe plant diagnosis, why PPO-only
provably fails on this task, and the working demos+DAgger pipeline.

## Current state (all committed+pushed through `44040e0`)

The plant is fixed and locked (symmetric knuckle drives, anchored
command-continuous target, slide-search contact authority — details in
`forge_plus/isaac_gear_env.py` comments). Both clean gates passed:

- abs 256/256, 0 breaks, peak 16.1 N — `checkpoints/task1_gear_rq_bc3.pt`
- steel 256/256, 0 breaks, peak 28.6 N — `checkpoints/task1_gear_rq_bc4.pt`
- best single ckpt: `task1_gear_rq_bc7.pt` (steel 32/32 @ 18.9 N, abs 27/32)

Jam smoke done: the 2F-140 jam mode is HOVER-dominant (not the franka's
slip-tilt); recovery=ours lifts 5 mm-slip success 12% -> 48% with 44%
timeouts remaining. Demo data: `/workspace/logs/gear_demo_*.npz` (pod-only).

## Goals, in order

1. **Single-ckpt unification** — one policy >=95% on BOTH classes, 0 breaks.
   Start from `bc7`. Ideas ranked:
   (a) tiny-std PPO polish on bc7 (`--reset_std` <= -3.0, i.e. std <= 0.05 —
       NEVER >= 0.12: that noise level breaks 40/64 and seats 1/320);
   (b) class-balanced BC batches / loss weighting in `scripts/bc_gear_mean.py`;
   (c) per-class heads.
   Warm-start refits only (cold refits from sliprand see-saw the classes).
   Gate with `scripts/eval_gear_insert.py --gripper robotiq_2f140`
   (abs `--obj 0`, steel `--obj 1`).
2. **Recovery tuning vs the hover signature** —
   `scripts/eval_gear_jam.py --gripper robotiq_2f140 --ckpt <best> --obj 1
   --recovery ours --slip_mm 5`. The hover branch fires
   (retract_and_reapproach -> regrasp) but 44% still time out; diagnose with
   `--debug` + the jam attempt logs. The franka hover-branch lessons are in
   memory (`task1-recovery-arc`).
3. **Sweep row** once recovery is strong.

## Standing rules

- One Isaac process at a time (kill leftovers by explicit PID;
  `_app.close()` hangs — scripts use `os._exit(0)`).
- Long jobs via `nohup <script> & disown`, watch the log file.
- Deterministic-mean eval protocol; NEVER script manipulation in
  evals/renders (scripted-expert demos for TRAINING are fine — that's the
  established pipeline: `scripts/collect_gear_demos.py --driver` for DAgger).
- numpy stays 1.26.0.
- Commit explicit paths only, never `git add -A`; push with plain
  `git push origin task3`; creds in `/workspace/.jr_notes` (never print).
- Checkpoints ARE committed to git (user-authorized) — keep the
  `checkpoints/README.md` manifest updated with anything new.

## Traps that cost hours last session (don't re-derive)

- Trainer `succ` counters are /32 inflated under `forge_no_term` —
  deterministic triage (`scripts/triage_rq_ckpts.py`, one boot, multi-ckpt)
  is the only truth.
- The abs class sits on its own break floor (f_break min 20 N vs ~15-25 N
  contact transients), so its behavior is knife-edged under refits.
- Standing errors in the arm are load-bearing (command continuity at
  hand-off; don't "fix" them).
- The no-gravity-gripper experiment is a documented dead end unless you
  budget a full staging recalibration.

Work autonomously between gates; show results at each gate.
