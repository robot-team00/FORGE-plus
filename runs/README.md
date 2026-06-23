# runs/ -- one self-contained folder per experiment

Naming: runs/<YYYY-MM-DD>_<short-slug>/   e.g. 2026-06-21_forge-baseline
Each run holds:
  checkpoints/<gripper>.pt     rollouts/<gripper>.npz     videos/eval_<gripper>.mp4     run.md

Scripts take --run <run_id> (default = today's date):
  python scripts/train_place.py   --gripper franka_panda --run <id>
  python _scratch/diagnostics/record_rollout.py <gripper> <run_id>
  python scripts/render_eval_mpl.py <run_id>
