"""Hugging Face helpers: authenticated downloads and the demo reference voice."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import p

DEMO_REF_REPO = "jongy5/chatterbox-bangla"
DEMO_REF_FILE = "audios/refs/001.wav"
ENV_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")


def get_token() -> str | None:
    """Resolve an HF token from the environment or a cached huggingface login."""
    for key in ENV_KEYS:
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    try:
        from huggingface_hub import get_token as _cached
        return _cached()
    except Exception:
        return None


def load_kaggle_secret(key: str = "HF_TOKEN") -> bool:
    """Promote a Kaggle Secret into the environment. No-op off Kaggle."""
    if os.environ.get(key, "").strip():
        return True
    try:
        from kaggle_secrets import UserSecretsClient
    except Exception:
        return False
    try:
        os.environ[key] = UserSecretsClient().get_secret(key)
        print(f"{key} loaded from Kaggle Secrets")
        return True
    except Exception as e:
        print(f"no Kaggle secret named {key} ({type(e).__name__})")
        return False


def report_token() -> None:
    if get_token():
        print("HF token: found (authenticated downloads)")
    else:
        print("HF token: NOT set - anonymous downloads (slower, rate-limited). "
              "Add a Kaggle Secret named HF_TOKEN.")


def hf(repo: str, filename: str, dest: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename, local_dir=dest,
                           token=get_token())


def snapshot(repo: str, local_dir: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo, local_dir=local_dir,
                             token=get_token())


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
