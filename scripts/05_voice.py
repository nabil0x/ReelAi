"""05_voice – Generate full Bangla voice with the winning TTS model.

Note: The shootout (04) used a short sample. For production quality,
re-run the winning model's generation logic on the full BANGLA_SCRIPT.
This script uses the sample wav and stretches/compresses it to match
the original video duration.

Outputs:
    work/bangla_voice.wav  – timing-matched Bangla narration

Usage:
    python scripts/05_voice.py --model vits [--base ...]
    # --model choices: chatterbox | cosyvoice | vits | mms_fallback | jongy5
"""
from __future__ import annotations
import argparse, json, os, sys
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from common import base_arg, ensure_dirs, p, print_vram, cleanup, load_meta

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Full Bangla voice with winning TTS model")
    ap.add_argument("--model", default="vits",
                    choices=["chatterbox", "cosyvoice", "vits", "mms_fallback", "jongy5"],
                    help="TTS model name from the shootout")
    ap.add_argument("--base", default="/kaggle/working/project", help="Project root")
    return ap.parse_args()

def _generate_full(model_name: str) -> tuple[int, int]:
    """Generate Bangla voice for the full script.

    Returns (waveform_length, sample_rate).
    For now this reuses the shootout wav as a placeholder.
    TODO: re-run the winning model on the full BANGLA_SCRIPT for best quality.
    """
    import librosa

    src = p("tts_tests", f"{model_name}.wav")
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} not found – run 04_tts_shootout.py first")

    y, sr = librosa.load(src, sr=24000)
    return len(y), sr

def main() -> None:
    args = parse_args()
    base_arg()
    ensure_dirs()

    print(f"Using model: {args.model}")
    print_vram("pre-voice")

    meta = load_meta()
    target_dur = meta["DUR"]
    print(f"Target duration: {target_dur:.1f}s")

    # NOTE: For full quality, replace the placeholder with model-specific
    # generation on the entire BANGLA_SCRIPT (see module docstring).
    import librosa
    src = p("tts_tests", f"{args.model}.wav")
    if not os.path.exists(src):
        print(f"ERROR: {src} not found – run 04_tts_shootout.py first")
        sys.exit(1)

    y, sr = librosa.load(src, sr=24000)
    raw_dur = len(y) / sr
    print(f"Raw duration: {raw_dur:.1f}s")

    rate = raw_dur / target_dur if target_dur > 0 else 1.0
    print(f"Stretch rate: {rate:.3f}")

    if 0.9 <= rate <= 1.12:
        y_out = librosa.effects.time_stretch(y, rate=rate)
    elif rate > 1.12:
        print("WARNING: audio too long – capping stretch at 1.12x. "
              "Shorten wording then regenerate.")
        y_out = librosa.effects.time_stretch(y, rate=1.12)
    else:
        y_out = y

    out_path = p("work", "bangla_voice.wav")
    sf.write(out_path, y_out, sr)
    print(f"Saved -> {out_path} ({len(y_out) / sr:.1f}s)")

    print_vram("post-voice")

if __name__ == "__main__":
    main()
