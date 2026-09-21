"""04_tts_shootout – Generate the same Bangla sample with every TTS engine.

Writes ~12 s wav samples to tts_tests/ using the shared engines in
tts_engines.py, then reports which engines worked. Listen to the wavs and
pass the winner to 05_voice.py --model.

Outputs:
    tts_tests/<engine>.wav
    tts_tests/shootout_report.json

Usage:
    python scripts/04_tts_shootout.py [--ref-voice session.wav] [--base ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import base_arg, ensure_dirs, p, print_vram, cleanup
import tts_engines

SAMPLE_TEXT = (
    "আপনি থাকুন, অথবা বাইরে থাকুন... "
    "আপনার ভাটা চলবে নিজের গতিতে।"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TTS engine shootout")
    ap.add_argument("--ref-voice", default=None,
                    help="Reference wav for voice cloning (cosyvoice needs it)")
    ap.add_argument("--base", default="/kaggle/working/project",
                    help="Project root")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base_arg()
    ensure_dirs()

    device = "cuda"
    print_vram("pre-tts")
    report: dict[str, str] = {}

    for name in tts_engines.NAMES:
        out = p("tts_tests", f"{name}.wav")
        t0 = time.time()
        try:
            ok = tts_engines.gen(name, SAMPLE_TEXT, out, args.ref_voice, device)
            if ok and os.path.exists(out):
                size = os.path.getsize(out)
                report[name] = f"ok ({size/1024:.0f} KB, {time.time()-t0:.0f}s)"
            else:
                report[name] = "fail (engine returned False)"
        except Exception as e:
            report[name] = f"fail: {type(e).__name__}: {e}"
            cleanup()
        print(f"  {name}: {report[name]}")

    report_path = p("tts_tests", "shootout_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    working = [k for k, v in report.items() if v.startswith("ok")]
    print(json.dumps(report, indent=1))
    print(f"Working engines: {working or 'NONE'}")
    print_vram("post-tts-shootout")
    if working:
        print("Listen to tts_tests/*.wav, then run:")
        print(f"  python scripts/05_voice.py --model {working[0]}")
    else:
        print("No engine produced audio – check the per-engine errors above.")


if __name__ == "__main__":
    main()
