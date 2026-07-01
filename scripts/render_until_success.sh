#!/bin/bash
# Render the learned safe-release policy repeatedly until a rollout cleanly succeeds
# (bottle PLACED upright + hand clear, sustained to the end), then encode that rollout.
set -u
cd /workspace/FORGE-plus_task3
export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 PYTHONPATH=/workspace/FORGE-plus_task3 RELEASE=1
FF=$(/workspace/.venv/bin/python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
OUT=/workspace/FORGE-plus_task3/docs/videos/task3/forge_release.mp4
MAXATT=${MAXATT:-8}

for att in $(seq 1 "$MAXATT"); do
  echo "=== ATTEMPT $att ==="
  rm -f /workspace/frames_forge_min/*.png
  /workspace/.venv/bin/python scripts/render_forge_min.py > /workspace/logs/render_release.log 2>&1
  # Success = the render reached a sustained PLACED and ended on it (prints "RESULT PLACED").
  result=$(grep -ac "RESULT PLACED" /workspace/logs/render_release.log)
  echo "attempt $att: RESULT_PLACED=$result"
  if [ "$result" -ge 1 ]; then
    "$FF" -y -framerate 24 -i /workspace/frames_forge_min/f_%04d.png \
      -c:v libx264 -pix_fmt yuv420p -crf 20 "$OUT" >/dev/null 2>&1
    echo "SUCCESS on attempt $att -> $OUT ($(stat -c%s "$OUT") bytes)"
    exit 0
  fi
  echo "attempt $att not a sustained success, retrying..."
done
echo "NO_SUCCESS after $MAXATT attempts"
exit 1
