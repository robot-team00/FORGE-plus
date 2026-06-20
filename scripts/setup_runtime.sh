#!/usr/bin/env bash
# setup_runtime.sh - make this pod able to run the Isaac Sim RTX render.
# Idempotent. Run once after every pod (re)start:
#     bash scripts/setup_runtime.sh
#
# Restores the two things that do NOT survive a RunPod restart:
#   1. GLVND userspace libs (libEGL.so.1) - apt installs land in /usr (ephemeral).
#      Without libEGL.so.1 the NVIDIA Vulkan driver fails to init and RTX renders
#      nothing (vkCreateInstance -> ERROR_INCOMPATIBLE_DRIVER).
#   2. Xvfb on :1 (Vulkan needs an X display in this headless container).
# The Isaac shader cache lives under .venv (persists); we verify/restore it too.
set -u
REPO="/workspace/FORGE-plus"
VENV_PY="$REPO/.venv/bin/python"
L=/usr/lib/x86_64-linux-gnu

echo "[setup] 1/3 GLVND (libEGL.so.1) ..."
if [ ! -e "$L/libEGL.so.1" ]; then
  apt-get update -qq 2>/dev/null
  if apt-get install -y -qq libegl1 libglvnd0 libgl1 libglx0 libopengl0 libegl1-mesa >/dev/null 2>&1; then
    echo "[setup]   installed GLVND via apt"
  elif [ -d "$REPO/.runtime_libs" ]; then
    cp -av "$REPO/.runtime_libs/"* "$L/" 2>/dev/null && ldconfig
    echo "[setup]   restored GLVND from $REPO/.runtime_libs"
  else
    echo "[setup]   WARNING: could not install GLVND (no apt, no cached libs)"
  fi
else
  echo "[setup]   libEGL.so.1 already present"
fi

echo "[setup] 2/3 Isaac gpu_foundation shader cache ..."
if [ -x "$VENV_PY" ]; then
  "$VENV_PY" "$REPO/scripts/fetch_shadercache.py" || echo "[setup]   WARNING: shader cache fetch failed"
else
  echo "[setup]   WARNING: venv python not found at $VENV_PY"
fi

echo "[setup] 3/3 Xvfb display :1 ..."
if ! pgrep -x Xvfb >/dev/null 2>&1; then
  Xvfb :1 -screen 0 1920x1080x24 >/dev/null 2>&1 &
  sleep 2
  echo "[setup]   started Xvfb on :1"
else
  echo "[setup]   Xvfb already running"
fi
export DISPLAY=:1

echo "[setup] sanity: vulkaninfo device ->"
DISPLAY=:1 vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName" || \
  echo "[setup]   (vulkaninfo not conclusive - install vulkan-tools to verify)"

echo "[setup] DONE. Render with:"
echo "    cd $REPO && DISPLAY=:1 .venv/bin/python scripts/render_eval_video.py"
