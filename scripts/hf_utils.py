"""Hugging Face download helpers, including the demo reference voice."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import p

DEMO_REF_REPO = "jongy5/chatterbox-bangla"
DEMO_REF_FILE = "audios/refs/001.wav"


def hf(repo: str, filename: str, dest: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename, local_dir=dest)


def default_reference() -> str | None:
    """Fetch the demo Bangla reference voice published with jongy5's model.

    For production use pass your own --ref-voice; cloning a speaker requires
    that speaker's permission.
    """
    try:
        path = hf(DEMO_REF_REPO, DEMO_REF_FILE, p("models", "refs"))
        print(f"  demo reference voice: {DEMO_REF_REPO}/{DEMO_REF_FILE}")
        print("  (pass --ref-voice <wav> to clone your own speaker instead)")
        return path
    except Exception as e:
        print(f"  demo reference unavailable ({e})")
        return None
