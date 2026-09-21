"""Bangla TTS engines for the reel pipeline.

Every engine exposes the same contract:

    gen(text, out_wav, model_id, ref_audio=None, device="cuda") -> bool

Returns True only when ``out_wav`` was written; otherwise False with an
actionable message so the shootout can continue down the fallback chain.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import chatterbox_loader as cb
from audio_utils import concat_with_pause, split_sentences, trim_silence
from common import cleanup, p, pip_install, sh
from hf_utils import default_reference, hf as _hf

CHATTERBOX_BANGLA = cb.EMTIAZZ
JONGY5_CHATTERBOX = cb.JONGY5
VITS_MODEL = "EMTIAZZ/bangladeshi-bangla-tts-vits"
MMS_MODEL = "facebook/mms-tts-ben"
COSYVOICE_MODEL = "kawshikbuet17/bengali-cosyvoice3-tts"
CB_PARAMS = dict(temperature=0.3, exaggeration=0.5, cfg_weight=0.5,
                 repetition_penalty=1.2, min_new_tokens=150)


# ---------------------------------------------------------------- chatterbox

def gen_chatterbox(text: str, out_wav: str, model_id: str,
                   ref_audio: str | None = None, device: str = "cuda") -> bool:
    if not cb.ensure_package():
        return False
    import soundfile as sf

    tts = cb.load(model_id, device)
    chunks = [trim_silence(tts.generate(text=s, audio_prompt_path=ref_audio,
                                        **CB_PARAMS).squeeze().cpu().numpy(),
                           tts.sr)
              for s in split_sentences(text)]
    sf.write(out_wav, concat_with_pause(chunks, tts.sr), tts.sr)
    del tts
    cleanup()
    return True


# ------------------------------------------------------------------- coqui

def _ensure_coqui() -> bool:
    try:
        import TTS  # noqa: F401
        return True
    except ImportError:
        pass
    pip_install("coqui-tts", no_deps=True)
    pip_install("coqui-tts-trainer", no_deps=True)
    pip_install("bangla", "bnnumerizer", "bnunicodenormalizer")
    try:
        import TTS  # noqa: F401
        return True
    except ImportError:
        print("  coqui TTS could not be imported")
        return False


def gen_vits(text: str, out_wav: str, model_id: str,
             ref_audio: str | None = None, device: str = "cuda") -> bool:
    if not _ensure_coqui():
        return False
    import numpy as np
    import soundfile as sf

    ckpt = _hf(model_id, "pytorch_model.pth", p("models", "bangla_vits"))
    cfg_path = _hf(model_id, "config.json", p("models", "bangla_vits"))

    try:
        from TTS.utils.synthesizer import Synthesizer
        syn = Synthesizer(tts_checkpoint=ckpt, tts_config_path=cfg_path)
        sr = getattr(syn, "output_sample_rate", 22050)
        sf.write(out_wav, np.array(syn.tts(text), dtype="float32"), sr)
        cleanup()
        return True
    except Exception as e:
        print(f"  Synthesizer path failed ({e}); trying TTS.api")
    from TTS.api import TTS as CoquiTTS
    tts = CoquiTTS(model_path=ckpt, config_path=cfg_path, gpu=(device == "cuda"))
    tts.tts_to_file(text, file_path=out_wav)
    del tts
    cleanup()
    return True


# --------------------------------------------------------------------- mms

def gen_mms(text: str, out_wav: str, model_id: str = MMS_MODEL,
            ref_audio: str | None = None, device: str = "cuda") -> bool:
    import soundfile as sf
    import torch
    from transformers import AutoTokenizer, VitsModel

    tok = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id).to(device).eval()
    inputs = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        wav = model(**inputs).waveform
    sf.write(out_wav, wav.cpu().numpy().squeeze(), model.config.sampling_rate)
    del model, tok
    cleanup()
    return True


# ---------------------------------------------------------------- cosyvoice

def gen_cosyvoice(text: str, out_wav: str, model_id: str,
                  ref_audio: str | None = None, device: str = "cuda") -> bool:
    if not ref_audio or not os.path.exists(ref_audio):
        print("  cosyvoice: needs a reference wav (cross-lingual clone). "
              "Pass --ref-voice <wav>.")
        return False
    import numpy as np
    import soundfile as sf

    repo = p("models", "CosyVoice")
    if not os.path.isdir(repo):
        sh(f"git clone --recursive --depth 1 "
           f"https://github.com/FunAudioLLM/CosyVoice {repo}")
    for extra in (repo, os.path.join(repo, "third_party", "Matcha-TTS")):
        if os.path.isdir(extra):
            sys.path.insert(0, extra)

    try:
        from cosyvoice.cli.cosyvoice import CosyVoice3
    except Exception as e:
        print(f"  cosyvoice repo import failed: {type(e).__name__}: {e}")
        print("  needs pynini/WeTextProcessing + sox, which rarely install on "
              "Kaggle. Use the official Space for this engine.")
        return False

    model_dir = p("models", "Fun-CosyVoice3-0.5B")
    if not os.path.isdir(model_dir):
        from huggingface_hub import snapshot_download
        snapshot_download(model_id, local_dir=model_dir)

    engine = CosyVoice3(model_dir)
    pieces = [o["tts_speech"].squeeze().cpu().numpy()
              for o in engine.inference_cross_lingual(text, ref_audio,
                                                      stream=False)]
    sr = getattr(engine, "sample_rate", 24000)
    sf.write(out_wav, np.concatenate(pieces), sr)
    del engine
    cleanup()
    return True


REGISTRY = {
    "chatterbox": lambda t, o, r, d: gen_chatterbox(t, o, CHATTERBOX_BANGLA, r, d),
    "jongy5": lambda t, o, r, d: gen_chatterbox(t, o, JONGY5_CHATTERBOX, r, d),
    "vits": lambda t, o, r, d: gen_vits(t, o, VITS_MODEL, r, d),
    "mms_fallback": lambda t, o, r, d: gen_mms(t, o, MMS_MODEL, r, d),
    "cosyvoice": lambda t, o, r, d: gen_cosyvoice(t, o, COSYVOICE_MODEL, r, d),
}

NAMES = ["chatterbox", "cosyvoice", "vits", "mms_fallback", "jongy5"]


def gen(name: str, text: str, out_wav: str,
        ref_audio: str | None = None, device: str = "cuda") -> bool:
    if ref_audio is None and name != "mms_fallback":
        ref_audio = default_reference()
    return REGISTRY[name](text, out_wav, ref_audio, device)
