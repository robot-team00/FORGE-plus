#!/bin/bash
# Render the fragile-object recovery episode repeatedly until a take ends in
# RESULT SUCCESS (recovery seated it AND the finale placed it: learned release +
# retract clear, no break). Takes encode to versioned scratch files
# (/workspace/render_takes/forge_recovery_take_NNN.mp4) — the stable README-linked
# docs/videos/task3/forge_recovery.mp4 is only overwritten by the approved take.
#   bash scripts/render_recovery_until_success.sh [max_tries]
set -u
cd /workspace/FORGE-plus_task3
export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 PYTHONPATH=/workspace/FORGE-plus_task3
MAX=${1:-6}
FINAL=docs/videos/task3/forge_recovery.mp4
for i in $(seq 1 "$MAX"); do
  LOG=/workspace/logs/render_recovery_take$i.log
  TAKEMP4=$(printf "/workspace/render_takes/forge_recovery_take_%03d.mp4" "$i")
  echo "=== render take $i/$MAX -> $LOG ==="
  TAKE=$i /workspace/.venv/bin/python scripts/render_recovery.py > "$LOG" 2>&1
  if grep -qa "RESULT SUCCESS" "$LOG" && grep -qa "FFMPEG ok" "$LOG" && [ -s "$TAKEMP4" ]; then
    echo "=== SUCCESS on take $i ==="
    grep -a "RESULT SUCCESS\|FFMPEG ok" "$LOG"
    cp "$TAKEMP4" "$FINAL"
    echo "approved take copied -> $FINAL"
    exit 0
  fi
  echo "--- take $i: $(grep -a 'RESULT' "$LOG" | tail -1 || echo 'no result (crash?)') ---"
done
echo "=== NO SUCCESS in $MAX takes ==="
exit 1
