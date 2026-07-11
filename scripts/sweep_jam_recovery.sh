#!/usr/bin/env bash
# Task 1 (issue #26) — jam-recovery baseline sweep, one cell at a time
# (single Isaac process rule). Writes per-cell logs to /workspace/jam_sweep/.
#
#   bash scripts/sweep_jam_recovery.sh [CKPT] [EPISODES]
#
# Cells: ours / heuristic / vision_llm / press_harder / none, all on the
# fragile abs_gear with the honest in-grip 5 mm slip inducer.
set -u
# default: the sliprand snapshot that passes the clean gate (200/200, 0 breaks)
CKPT="${1:-checkpoints/task1_gear_sliprand.pt.it300}"
EP="${2:-25}"
OUT=/workspace/jam_sweep
mkdir -p "$OUT"
cd /workspace/FORGE-plus_task3
export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 \
       PYTHONPATH=/workspace/FORGE-plus_task3

for MODE in ours heuristic vision_llm press_harder none; do
    LOG="$OUT/jam_${MODE}.log"
    echo "=== $MODE -> $LOG ==="
    /workspace/.venv/bin/python scripts/eval_gear_jam.py \
        --recovery "$MODE" --obj 0 --slip_mm 5 --episodes "$EP" \
        --ckpt "$CKPT" > "$LOG" 2>&1
    grep -E "SUCCESS|BREAK|TIMEOUT|attempts|peak Fins|signatures" "$LOG" \
        || tail -5 "$LOG"
done
echo "SWEEP_DONE"
