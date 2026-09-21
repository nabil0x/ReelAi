"""04_tts_shootout – Load/generate/unload each TTS model on the same sample.

Tests five Bengali TTS models and writes ~12 s wav samples to tts_tests/.
After listening, set the winner name in 05_voice.py via --model.

Outputs:
    tts_tests/chatterbox.wav
    tts_tests/cosyvoice.wav
    tts_tests/vits.wav
    tts_tests/mms_fallback.wav
    tts_tests/jongy5.wav

Usage:
    python scripts/04_tts_shootout.py [--base ...]
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from common import base_arg, ensure_dirs, p, print_vram, cleanup, TTS_MODELS, pip_install

SAMPLE_TEXT = (
    "আপনি থাকুন, অথবা বাইরে থাকুন... "
    "আপনার ভাটা চলবে নিজের গতিতে।"
)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TTS model shootout")
    ap.add_argument("--base", default="/kaggle/working/project", help="Project root")
    return ap.parse_args()

def _try_chatterbox(out: str) -> str:
    """Chatterbox Bangla TTS (EMTIAZZ)."""
    pip_install("chatterbox-tts")
    from chatterbox.tts import ChatterboxTTS
    import soundfile as sf
    import torch
    model = ChatterboxTTS.from_pretrained(TTS_MODELS["chatterbox"], device="cuda")
    wav = model.generate(SAMPLE_TEXT, language_id="bn")
    sf.write(out, wav.squeeze().cpu().numpy(), model.sr)
    del model; cleanup()
    return "ok"

def _try_cosyvoice(out: str) -> str:
    """CosyVoice3 Bengali via transformers."""
    from transformers import AutoModelForTextToWaveform, AutoProcessor
    import soundfile as sf
    import torch
    pid = TTS_MODELS["cosyvoice"]
    proc = AutoProcessor.from_pretrained(pid, trust_remote_code=True)
    model = AutoModelForTextToWaveform.from_pretrained(
        pid, trust_remote_code=True, torch_dtype=torch.float16,
    ).to("cuda")
    inp = proc(text=[SAMPLE_TEXT], return_tensors="pt").to("cuda")
    with torch.no_grad():
        result = model.generate(**inp)
    sr = getattr(proc, "sampling_rate", 24000)
    sf.write(out, result.cpu().numpy().squeeze(), sr)
    del model, proc; cleanup()
    return "ok"

def _try_vits(out: str) -> str:
    """Bangladeshi VITS (EMTIAZZ) via coqui-tts."""
    try:
        from TTS.api import TTS
    except ImportError:
        pip_install("coqui-tts", no_deps=True)
        from TTS.api import TTS
    model = TTS(TTS_MODELS["vits"], gpu=True)
    model.tts_to_file(SAMPLE_TEXT, file_path=out)
    del model; cleanup()
    return "ok"

def _try_mms(out: str) -> str:
    """MMS-TTS Bengali (Facebook) – guaranteed fallback."""
    from transformers import VitsModel, AutoTokenizer
    import soundfile as sf
    import torch
    tok = AutoTokenizer.from_pretrained(TTS_MODELS["mms_fallback"])
    model = VitsModel.from_pretrained(
        TTS_MODELS["mms_fallback"], torch_dtype=torch.float16,
    ).to("cuda")
    inp = tok(SAMPLE_TEXT, return_tensors="pt").to("cuda")
    with torch.no_grad():
        wav = model(**inp).waveform
    sf.write(out, wav.cpu().numpy().squeeze(), 16000)
    del model, tok; cleanup()
    return "ok"

def _try_jongy5(out: str) -> str:
    """Chatterbox Bangla TTS (jongy5) – same ChatterboxTTS API, different weights."""
    pip_install("chatterbox-tts")
    from chatterbox.tts import ChatterboxTTS
    import soundfile as sf
    import torch
    model = ChatterboxTTS.from_pretrained(TTS_MODELS["jongy5"], device="cuda")
    wav = model.generate(SAMPLE_TEXT, language_id="bn")
    sf.write(out, wav.squeeze().cpu().numpy(), model.sr)
    del model; cleanup()
    return "ok"

RUNNERS = {
    "chatterbox":   ("chatterbox",   _try_chatterbox),
    "cosyvoice":    ("cosyvoice",    _try_cosyvoice),
    "vits":         ("vits",         _try_vits),
    "mms_fallback": ("mms_fallback", _try_mms),
    "jongy5":       ("jongy5",       _try_jongy5),
}

def main() -> None:
    args = parse_args()
    base_arg()
    ensure_dirs()

    print_vram("pre-tts")
    report: dict[str, str] = {}

    for key in ("chatterbox", "cosyvoice", "vits", "mms_fallback", "jongy5"):
        name, fn = RUNNERS[key]
        out = p("tts_tests", f"{name}.wav")
        try:
            result = fn(out)
            report[name] = result
            print(f"  {name}: {result}")
        except Exception as e:
            report[name] = f"fail: {e}"
            print(f"  {name}: fail – {e}")
            cleanup()

    report_path = p("tts_tests", "shootout_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    print(json.dumps(report, indent=1))
    print_vram("post-tts-shootout")
    print("Listen to tts_tests/*.wav, then set --model in 05_voice.py")

if __name__ == "__main__":
    main()
