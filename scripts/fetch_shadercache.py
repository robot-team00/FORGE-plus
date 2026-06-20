#!/usr/bin/env python3
"""
fetch_shadercache.py - restore the Isaac Sim gpu_foundation Vulkan shader cache.

Why: on a fresh 'pip install isaacsim', the extension
omni.gpu_foundation.shadercache.vulkan is sometimes installed as a STUB
(only py/toml/md, missing the compiled .v shaders + cache/shadercache/common/version).
Without those, the RTX renderer logs
  "Cannot load shader file 'rtx/system/GenerateMipMap.comp.hlsl'"
and every captured frame is an EMPTY buffer.

This script pulls just the ~19 missing cache files out of the 3 GB
isaacsim-extscache-kit wheel via HTTP range requests (no full download).
Idempotent: does nothing if the cache is already present.

Run with the Isaac venv python:
  /workspace/.venv/bin/python scripts/fetch_shadercache.py
"""
import os, sys, glob, subprocess

VENV_SP = "/workspace/.venv/lib/python3.11/site-packages"
WHEEL = ("https://pypi.nvidia.com/isaacsim-extscache-kit/"
         "isaacsim_extscache_kit-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl")
EXT_GLOB = os.path.join(VENV_SP, "isaacsim/extscache/omni.gpu_foundation.shadercache.vulkan-*")

def main():
    dirs = sorted(glob.glob(EXT_GLOB))
    if not dirs:
        print("[fetch] extension dir not found:", EXT_GLOB, "- is isaacsim installed? aborting.")
        return 1
    ext_dir = dirs[-1]
    ext_name = os.path.basename(ext_dir)
    version_file = os.path.join(ext_dir, "cache", "shadercache", "common", "version")
    if os.path.exists(version_file):
        print("[fetch] shader cache already present:", version_file)
        return 0
    print("[fetch] shader cache MISSING - restoring from wheel via range requests...")
    try:
        from remotezip import RemoteZip  # noqa
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "remotezip"], check=True)
        from remotezip import RemoteZip  # noqa
    import shutil
    with RemoteZip(WHEEL) as z:
        want = [n for n in z.namelist() if ext_name in n and "/cache/" in n]
        print("[fetch] extracting", len(want), "files")
        for n in want:
            sub = n.split(ext_name + "/", 1)[1]
            target = os.path.join(ext_dir, sub)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(n) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    ok = os.path.exists(version_file)
    nv = len(glob.glob(os.path.join(ext_dir, "cache", "shadercache", "*.v")))
    print("[fetch] done. version file present:", ok, "| .v shaders:", nv)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
