"""Load Bangla Chatterbox engines (EMTIAZZ / jongy5).

ChatterboxTTS.from_local() builds ``T3()`` with its default 704-token vocab and
then loads ``t3_cfg`` with strict=True. Every Bangla fine-tune extends that
vocab, so from_local always raises a size mismatch. The fix used by the model
authors: load the ResembleAI base, then swap in a T3 sized to the fine-tuned
checkpoint plus the fine-tuned tokenizer.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import p, pip_install
from hf_utils import hf as _hf

BASE_REPO = "ResembleAI/chatterbox"
EMTIAZZ = "EMTIAZZ/chatterbox-bangla-tts"
JONGY5 = "jongy5/chatterbox-bangla"
BASE_FILES = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors",
              "conds.pt", "tokenizer.json"]
FT_WEIGHTS = {EMTIAZZ: "t3_bangla_888k.safetensors", JONGY5: "t3_cfg.safetensors"}


def ensure_package() -> bool:
    """chatterbox-tts pins torch==2.6.0 / transformers==4.46.3 / safetensors==0.5.3,
    none of which can resolve against the Kaggle image. Install --no-deps and add
    only the runtime deps the image lacks."""
    try:
        import chatterbox  # noqa: F401
        return True
    except ImportError as first:
        print(f"  chatterbox pre-check: {first}")

    pip_install("chatterbox-tts==0.1.2", no_deps=True)
    pip_install("s3tokenizer", "resemble-perth==1.0.1", "conformer==0.3.2")
    try:
        import diffusers  # noqa: F401
    except ImportError:
        pip_install("diffusers")

    try:
        import chatterbox  # noqa: F401
        return True
    except ImportError as e:
        print(f"  chatterbox-tts unusable: {e}")
        return False


def _model_dir(model_id: str) -> str:
    dest = p("models", model_id.split("/")[-1])
    os.makedirs(dest, exist_ok=True)
    return dest


def _base_dir() -> str:
    dest = p("models", "chatterbox_base")
    os.makedirs(dest, exist_ok=True)
    for name in BASE_FILES:
        _hf(BASE_REPO, name, dest)
    return dest


def _swap_t3(tts, weights_path: str, tokenizer_path: str, device: str) -> None:
    from chatterbox.models.t3.t3 import T3
    from chatterbox.models.tokenizers import EnTokenizer
    from safetensors.torch import load_file

    state = load_file(weights_path, device="cpu")
    if "model" in state:
        state = state["model"][0]
    if any(k.startswith("t3.") for k in state):
        state = {k[len("t3."):]: v for k, v in state.items()
                 if k.startswith("t3.")}

    hp = tts.t3.hp
    hp.text_tokens_dict_size = state["text_emb.weight"].shape[0]
    new_t3 = T3(hp=hp)
    new_t3.load_state_dict(state, strict=True)

    tts.t3 = new_t3
    tts.tokenizer = EnTokenizer(tokenizer_path)
    tts.t3.to(device).eval()
    tts.s3gen.to(device).eval()
    tts.ve.to(device).eval()
    tts.device = device


def load(model_id: str, device: str = "cuda"):
    dest = _model_dir(model_id)
    weights = _hf(model_id, FT_WEIGHTS[model_id], dest)
    tokenizer = _hf(model_id, "tokenizer.json", dest)
    base = _base_dir()

    from chatterbox.tts import ChatterboxTTS

    tts = ChatterboxTTS.from_local(base, device="cpu")
    _swap_t3(tts, weights, tokenizer, device)
    return tts
