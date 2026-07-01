#!/bin/bash
# Render the fragile-object recovery episode repeatedly until a take ends in
# RESULT SUCCESS (seated, no break) and encodes. Keeps the first successful clip.
#   bash scripts/render_recovery_until_success.sh [max_tries]
set -u
cd /workspace/FORGE-plus_task3
export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 PYTHONPATH=/workspace/FORGE-plus_task3
MAX=${1:-6}
for i in $(seq 1 "$MAX"); do
  LOG=/workspace/logs/render_recovery_take$i.log
  echo "=== render take $i/$MAX -> $LOG ==="
  /workspace/.venv/bin/python scripts/render_recovery.py > "$LOG" 2>&1
  if grep -qa "RESULT SUCCESS" "$LOG" && grep -qa "FFMPEG ok" "$LOG"; then
    echo "=== SUCCESS on take $i ==="
    grep -a "RESULT SUCCESS\|FFMPEG ok" "$LOG"
    exit 0
  fi
  echo "--- take $i: $(grep -a 'RESULT' "$LOG" | tail -1 || echo 'no result (crash?)') ---"
done
echo "=== NO SUCCESS in $MAX takes ==="
exit 1
