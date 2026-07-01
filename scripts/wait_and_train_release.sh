#!/bin/bash
# Politely wait until the shared GPU is free (teammate's main-dir job done), then run the
# retract-aware safe-release fine-tune. Does NOT touch any process it didn't launch.
set -u
cd /workspace/FORGE-plus_task3
export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 PYTHONPATH=/workspace/FORGE-plus_task3 WANDB_MODE=disabled

NEED_FREE=${NEED_FREE:-11000}     # MiB free required before we start (leave room; don't contend)
MAX_WAIT_MIN=${MAX_WAIT_MIN:-240} # give up after this many minutes
waited=0
echo "[wait] need >= ${NEED_FREE} MiB free; polling every 60s (max ${MAX_WAIT_MIN} min)"
while :; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  free=${free:-0}
  echo "[wait] t=${waited}min  free=${free} MiB"
  if [ "$free" -ge "$NEED_FREE" ]; then
    echo "[wait] GPU free enough -> launching training"
    break
  fi
  if [ "$waited" -ge "$MAX_WAIT_MIN" ]; then
    echo "[wait] TIMEOUT after ${MAX_WAIT_MIN} min — GPU still busy, not starting"
    exit 2
  fi
  sleep 60
  waited=$((waited + 1))
done

cp -f checkpoints/task3_forge_release.pt checkpoints/task3_forge_release_v2.pt 2>/dev/null
echo "[train] starting retract fine-tune"
/workspace/.venv/bin/python scripts/train_pick_place.py \
  --forge_release --forge_obj 2 \
  --resume checkpoints/task3_forge_release.pt \
  --ckpt checkpoints/task3_forge_release.pt \
  --num_envs 512 --iterations 250 --rollout 32
echo "[train] DONE"
