#!/bin/bash
# Render the fragile-object recovery episode repeatedly until a take ends in
# RESULT SUCCESS (recovery seated it AND the finale placed it: learned release +
# retract clear, no break). Takes encode to versioned scratch files
# (/workspace/render_takes/forge_recovery_<gripper>_take_NNN.mp4) — the stable
# README-linked docs/videos/task3/forge_recovery_<gripper>.mp4 is only overwritten
# by the approved take.
#   bash scripts/render_recovery_until_success.sh [max_tries] [gripper]
#   gripper: franka (default) | robotiq
set -u
cd /workspace/FORGE-plus_task3
export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 PYTHONPATH=/workspace/FORGE-plus_task3
MAX=${1:-6}
GTAG=${2:-franka}
case "$GTAG" in
  franka)  export GRIPPER=franka_panda ;;
  robotiq) export GRIPPER=robotiq_2f140 ;;
  *) echo "unknown gripper '$GTAG' (franka|robotiq)"; exit 2 ;;
esac
FINAL=docs/videos/task3/forge_recovery_${GTAG}.mp4
for i in $(seq 1 "$MAX"); do
  LOG=/workspace/logs/render_recovery_${GTAG}_take$i.log
  TAKEMP4=$(printf "/workspace/render_takes/forge_recovery_%s_take_%03d.mp4" "$GTAG" "$i")
  echo "=== render take $i/$MAX ($GTAG) -> $LOG ==="
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
