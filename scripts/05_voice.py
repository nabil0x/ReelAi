"""05_voice – Generate the full Bangla narration with the chosen engine.

Generates the complete script via tts_engines, writes the natural result to
work/bangla_voice_raw.wav, then time-stretches it to match the original video
duration and writes work/bangla_voice.wav.

Usage:
    python scripts/05_voice.py --model chatterbox [--ref-voice wav] [--base ...]
    # models: chatterbox | cosyvoice | vits | mms_fallback | jongy5
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    base_arg, ensure_dirs, p, print_vram, load_meta, load_bangla_script,
    DEFAULT_BASE,
)
import tts_engines

MAX_STRETCH = 1.12


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Full Bangla narration")
    ap.add_argument("--model", default="mms_fallback", choices=tts_engines.NAMES)
    ap.add_argument("--ref-voice", default=None, help="Reference wav for cloning")
    ap.add_argument("--base", default=DEFAULT_BASE)
    return ap.parse_args()


def _load_mono(path: str):
    import librosa

    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr


def _fit_duration(y, sr: int, target: float):
    import librosa

    dur = len(y) / sr
    rate = dur / target if target > 0 else 1.0
    if 0.9 <= rate <= MAX_STRETCH:
        return librosa.effects.time_stretch(y.astype("float32"), rate=rate)
    if rate > MAX_STRETCH:
        print(f"WARNING: {dur:.1f}s vs target {target:.1f}s - capping at "
              f"{MAX_STRETCH}x. Shorten the Bangla wording for a natural fit.")
        return librosa.effects.time_stretch(y.astype("float32"), rate=MAX_STRETCH)
    return y.astype("float32")


def main() -> None:
    args = parse_args()
    base_arg()
    ensure_dirs()

    print_vram("pre-voice")
    target = load_meta()["DUR"]
    script = load_bangla_script()
    raw = p("work", "bangla_voice_raw.wav")
    final = p("work", "bangla_voice.wav")
    print(f"Engine: {args.model} | target {target:.1f}s | script {len(script)} chars")

    try:
        ok = tts_engines.gen(args.model, script, raw, args.ref_voice, "cuda")
    except Exception as e:
        ok = False
        print(f"Full generation raised: {type(e).__name__}: {e}")

    if not ok or not os.path.exists(raw):
        sample = p("tts_tests", f"{args.model}.wav")
        if not os.path.exists(sample):
            print(f"ERROR: generation failed and {sample} is missing. "
                  f"Run 04_tts_shootout.py first.")
            sys.exit(1)
        print(f"FALLBACK: using shootout sample {sample}. "
              f"Regenerate the full script for best quality.")
        raw = sample

    import librosa
    import soundfile as sf

    y, sr = _load_mono(raw)
    print(f"Raw narration: {len(y)/sr:.1f}s @ {sr} Hz")
    sf.write(final, _fit_duration(y, sr, target), sr)
    y2, _ = librosa.load(final, sr=None, mono=True)
    print(f"Final narration: {len(y2)/sr:.1f}s -> {final}")
    print_vram("post-voice")


if __name__ == "__main__":
    main()
