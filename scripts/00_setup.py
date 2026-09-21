"""00_setup – GPU report, project directories, default Bangla script.

Run first on a fresh Kaggle kernel:
    python scripts/00_setup.py [--base project]

Default project root is <repo>/project, overridable with --base or REELAI_BASE.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    base_arg, ensure_dirs, gpu_report, print_vram, p,
    autodetect_video, DEFAULT_BANGLA_SCRIPT, DEFAULT_BASE,
)
import hf_utils

def main() -> None:
    base = base_arg()
    ensure_dirs()
    gpu_report()
    print_vram("boot")
    hf_utils.load_kaggle_secret()
    hf_utils.report_token()

    # Persist the default Bangla script so downstream scripts can load it.
    script_path = p("work", "localized_script.txt")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(DEFAULT_BANGLA_SCRIPT)
    print(f"Bangla script saved ({len(DEFAULT_BANGLA_SCRIPT)} chars) -> {script_path}")

    video = autodetect_video()
    origin = "REELAI_BASE" if os.environ.get("REELAI_BASE") else \
        ("default" if base == DEFAULT_BASE else "--base")
    print(f"Project root : {base} ({origin})")
    print(f"Input video  : {video}")
    for sub in ("work", "output", "previews", "tts_tests"):
        print(f"  {sub:<10} -> {p(sub)}")

    print("Setup complete")

if __name__ == "__main__":
    main()
