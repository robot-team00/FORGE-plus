#!/usr/bin/env python3
"""Environment check script for FORGE-plus on RunPod."""
import os
import subprocess
import sys


def check(label, ok, detail=""):
    status = "[OK]" if ok else "[FAIL]"
    msg = f"{status} {label}"
    if detail:
        msg += f": {detail}"
    print(msg)
    return ok


def main():
    all_ok = True

    # GPU check
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        gpu_info = result.stdout.strip()
        is_blackwell = "Blackwell" in gpu_info or "PRO 4000" in gpu_info
        all_ok &= check("GPU", bool(gpu_info), gpu_info)
        if is_blackwell:
            print("  [WARN] Blackwell GPU detected — Vulkan ICD may be broken, rendering may fail")
    except Exception as e:
        all_ok &= check("GPU", False, str(e))

    # NVIDIA_DRIVER_CAPABILITIES check
    caps = os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "")
    has_graphics = "graphics" in caps or caps == "all"
    all_ok &= check(
        "NVIDIA_DRIVER_CAPABILITIES includes 'graphics'",
        has_graphics,
        caps if caps else "(not set — rendering will fail)"
    )

    # EGL check
    egl_paths = [
        "/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0",
        "/usr/lib/libEGL_nvidia.so.0",
    ]
    egl_found = next((p for p in egl_paths if os.path.exists(p)), None)
    all_ok &= check("EGL", bool(egl_found), egl_found or "not found")

    # Vulkan ICD check
    vulkan_icd = "/etc/vulkan/icd.d/nvidia_icd.json"
    all_ok &= check("Vulkan ICD", os.path.exists(vulkan_icd), vulkan_icd)

    # libGLU stub check
    glu_paths = ["/usr/local/lib/libGLU.so.1", "/usr/lib/x86_64-linux-gnu/libGLU.so.1"]
    glu_found = next((p for p in glu_paths if os.path.exists(p)), None)
    all_ok &= check("libGLU.so.1", bool(glu_found), glu_found or "not found — run: gcc -shared -fPIC -o /usr/local/lib/libGLU.so.1 scripts/libglu_stub.c && ldconfig")

    # venv check
    venv_python = "/workspace/.venv/bin/python"
    all_ok &= check("venv", os.path.exists(venv_python), venv_python)

    print()
    if all_ok:
        print("All checks passed. Ready for headless rendering.")
    else:
        print("Some checks failed. See docs/rendering.md for fixes.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
