"""00_setup – GPU report, project directories, default Bangla script.

Run first on a fresh Kaggle kernel:
    python scripts/00_setup.py [--base /kaggle/working/project]
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import base_arg, ensure_dirs, gpu_report, print_vram, p, DEFAULT_BANGLA_SCRIPT

def main() -> None:
    base_arg()
    ensure_dirs()
    gpu_report()
    print_vram("boot")

    # Persist the default Bangla script so downstream scripts can load it.
    script_path = p("work", "localized_script.txt")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(DEFAULT_BANGLA_SCRIPT)
    print(f"Bangla script saved ({len(DEFAULT_BANGLA_SCRIPT)} chars) -> {script_path}")

    # Detect video candidates under /kaggle/input
    import glob as _glob
    cands = _glob.glob("/kaggle/input/**/*.mp4", recursive=True)
    if cands:
        print(f"Auto-detected video: {cands[0]}")
    else:
        print("No .mp4 found in /kaggle/input – upload your Hindi reel first.")

    print("Setup complete ✓")

if __name__ == "__main__":
    main()
