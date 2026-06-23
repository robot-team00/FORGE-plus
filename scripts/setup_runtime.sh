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
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="/workspace/.venv"
VENV_PY="$VENV/bin/python"
L=/usr/lib/x86_64-linux-gnu

echo "[setup] 0/3 persistent shader/compute cache on /workspace ..."
export HOME=/workspace/persist/ovhome
export MPLBACKEND=Agg
export CUDA_CACHE_PATH=/workspace/persist/shadercache/cuda
export __GL_SHADER_DISK_CACHE=1
export __GL_SHADER_DISK_CACHE_PATH=/workspace/persist/shadercache/gl
mkdir -p "$HOME" "$CUDA_CACHE_PATH" "$__GL_SHADER_DISK_CACHE_PATH"
echo "[setup]   HOME -> $HOME (Kit/OV + driver shader caches now persist on /workspace)"

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


echo "[setup] 1b/3 libGLU.so.1 stub (required by MDL-SDK) ..."
LIBGLU=/usr/local/lib/libGLU.so.1
if [ ! -e "$LIBGLU" ]; then
  if gcc -shared -fPIC -o "$LIBGLU" "$REPO/scripts/libglu_stub.c" 2>/dev/null; then
    ldconfig
    echo "[setup]   built and installed libGLU.so.1 stub"
  else
    echo "[setup]   WARNING: failed to build libGLU stub — MDL-SDK may fail to load"
  fi
else
  echo "[setup]   libGLU.so.1 already present"
fi

echo "[setup] 2/3 (shader cache — skipped, not required) ..."
# Shader cache pre-population is unnecessary — rendering works with a cold cache.
# fetch_shadercache.py has been removed from the critical path. (See docs/RENDERING.md)
echo "[setup]   skipped (not needed — shaders compile on first use, cached to /workspace/persist/)"

echo "[setup] 3/3 Xvfb display :99 ..."
if ! pgrep -x Xvfb >/dev/null 2>&1; then
  Xvfb :99 -screen 0 1920x1080x24 >/dev/null 2>&1 &
  sleep 2
  echo "[setup]   started Xvfb on :99"
else
  echo "[setup]   Xvfb already running (check: pgrep Xvfb)"
fi
export DISPLAY=:99

echo "[setup] sanity: vulkaninfo device ->"
DISPLAY=:1 vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName" || \
  echo "[setup]   (vulkaninfo not conclusive - install vulkan-tools to verify)"

echo "[setup] DONE. Render with:"
echo "    cd $REPO && DISPLAY=:99 /workspace/.venv/bin/python scripts/render_task3.py"

# -- Ollama startup (added by FORGE-plus task3 setup) -----------------------
# Binary: /workspace/bin/ollama  (persists; do NOT reinstall from internet)
# Models: /workspace/ollama_models  (persists across pod restarts)
if ! pgrep -f "workspace/bin/ollama" > /dev/null 2>&1; then
    export OLLAMA_MODELS=/workspace/ollama_models
    export OLLAMA_HOME=/workspace
    nohup /workspace/bin/ollama serve >/tmp/ollama.log 2>&1 &
    sleep 2
    echo "ollama serve started PID $!"
else
    echo "ollama already running"
fi
# ---------------------------------------------------------------------------

