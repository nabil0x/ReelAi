"""05_voice – Generate full Bangla voice from the winning TTS model.

Generates the COMPLETE Bangla script (not a 12s sample stretch).
Writes work/bangla_voice_raw.wav (natural, unstretched), then
timing-matches to meta["DUR"] via librosa and writes work/bangla_voice.wav.

Falls back to the shootout sample wav if full generation fails.

Usage:
    python scripts/05_voice.py --model vits [--base ...]
    # --model choices: chatterbox | cosyvoice | vits | mms_fallback | jongy5
"""
from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    base_arg, ensure_dirs, p, print_vram, cleanup,
    load_meta, load_bangla_script, TTS_MODELS, pip_install,
)

CHOICES = ["chatterbox", "cosyvoice", "vits", "mms_fallback", "jongy5"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Full Bangla voice generation")
    ap.add_argument("--model", default="vits", choices=CHOICES,
                    help="TTS model name from the shootout")
    ap.add_argument("--base", default="/kaggle/working/project",
                    help="Project root")
    return ap.parse_args()


# ------------------------------------------------------------------
# Per-model full-script generators
# ------------------------------------------------------------------

def _gen_chatterbox(text: str, out_wav: str, model_id: str) -> bool:
    pip_install("chatterbox-tts")
    from chatterbox.tts import ChatterboxTTS
    import soundfile as sf
    model = ChatterboxTTS.from_pretrained(model_id, device="cuda")
    wav = model.generate(text, language_id="bn")
    sf.write(out_wav, wav.squeeze().cpu().numpy(), model.sr)
    del model; cleanup()
    return True


def _gen_cosyvoice(text: str, out_wav: str) -> bool:
    from transformers import AutoModelForTextToWaveform, AutoProcessor
    import soundfile as sf, torch
    pid = TTS_MODELS["cosyvoice"]
    proc = AutoProcessor.from_pretrained(pid, trust_remote_code=True)
    model = AutoModelForTextToWaveform.from_pretrained(
        pid, trust_remote_code=True, torch_dtype=torch.float16,
    ).to("cuda")
    inp = proc(text=[text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        result = model.generate(**inp)
    sr = getattr(proc, "sampling_rate", 24000)
    sf.write(out_wav, result.cpu().numpy().squeeze(), sr)
    del model, proc; cleanup()
    return True


def _gen_vits(text: str, out_wav: str) -> bool:
    try:
        from TTS.api import TTS
    except ImportError:
        pip_install("coqui-tts", no_deps=True)
        from TTS.api import TTS
    model = TTS(TTS_MODELS["vits"], gpu=True)
    model.tts_to_file(text, file_path=out_wav)
    del model; cleanup()
    return True


def _gen_mms(text: str, out_wav: str) -> bool:
    from transformers import VitsModel, AutoTokenizer
    import soundfile as sf, torch
    tok = AutoTokenizer.from_pretrained(TTS_MODELS["mms_fallback"])
    model = VitsModel.from_pretrained(
        TTS_MODELS["mms_fallback"], torch_dtype=torch.float16,
    ).to("cuda")
    inp = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        wav = model(**inp).waveform
    sf.write(out_wav, wav.cpu().numpy().squeeze(), 16000)
    del model, tok; cleanup()
    return True


GENERATORS = {
    "chatterbox":   lambda t, o: _gen_chatterbox(t, o, TTS_MODELS["chatterbox"]),
    "cosyvoice":    _gen_cosyvoice,
    "vits":         _gen_vits,
    "mms_fallback": _gen_mms,
    "jongy5":       lambda t, o: _gen_chatterbox(t, o, TTS_MODELS["jongy5"]),
}


def main() -> None:
    args = parse_args()
    base_arg()
    ensure_dirs()

    model_name = args.model
    print(f"Using model: {model_name}")
    print_vram("pre-voice")

    meta = load_meta()
    target_dur = meta["DUR"]
    print(f"Target duration: {target_dur:.1f}s")

    script = load_bangla_script()
    raw_wav = p("work", "bangla_voice_raw.wav")
    final_wav = p("work", "bangla_voice.wav")
    gen_ok = False

    print(f"Generating full script ({len(script)} chars) ...")
    try:
        gen_ok = GENERATORS[model_name](script, raw_wav)
    except Exception as e:
        print(f"WARNING: full generation failed: {e}")
        gen_ok = False

    if gen_ok and os.path.exists(raw_wav):
        import soundfile as sf, librosa
        y, sr = sf.read(raw_wav)
        if hasattr(y, "ndim") and y.ndim > 1:
            y = y.mean(axis=1)
        raw_dur = len(y) / sr
        print(f"Raw duration: {raw_dur:.1f}s (target {target_dur:.1f}s)")

        rate = raw_dur / target_dur if target_dur > 0 else 1.0
        print(f"Stretch rate: {rate:.3f}")

        if 0.9 <= rate <= 1.12:
            y_out = librosa.effects.time_stretch(y.astype("float32"), rate=rate)
        elif rate > 1.12:
            print("WARNING: audio too long – capping stretch at 1.12x.")
            print("         Shorten wording then regenerate.")
            y_out = librosa.effects.time_stretch(y.astype("float32"), rate=1.12)
        else:
            y_out = y

        sf.write(final_wav, y_out, sr)
        print(f"Saved -> {final_wav} ({len(y_out) / sr:.1f}s)")
    else:
        print("FALLBACK: full generation failed – using shootout sample wav")
        import soundfile as sf, librosa
        src = p("tts_tests", f"{model_name}.wav")
        if not os.path.exists(src):
            print(f"ERROR: {src} not found – run 04_tts_shootout.py first")
            sys.exit(1)
        y, sr = librosa.load(src, sr=24000)
        raw_dur = len(y) / sr
        print(f"Fallback sample duration: {raw_dur:.1f}s")

        rate = raw_dur / target_dur if target_dur > 0 else 1.0
        if 0.9 <= rate <= 1.12:
            y_out = librosa.effects.time_stretch(y, rate=rate)
        elif rate > 1.12:
            y_out = librosa.effects.time_stretch(y, rate=1.12)
        else:
            y_out = y

        sf.write(final_wav, y_out, sr)
        print(f"Saved -> {final_wav} ({len(y_out) / sr:.1f}s)")

    print_vram("post-voice")


if __name__ == "__main__":
    main()
