"""Shared utilities for the Hindi->Bangla reel pipeline.

Every numbered script should ``from common import ...`` rather than
re-defining these helpers.  All paths are Kaggle-native
(``/kaggle/working/project``) but overridable via ``--base``.
"""

from __future__ import annotations

import gc
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BASE = "/kaggle/working/project"
SUBDIRS = ["input", "work", "models", "output", "previews", "tts_tests"]

# 1080x1920 @ 30 fps – used as fallback when cv2 cannot read the video
FALLBACK_W, FALLBACK_H, FALLBACK_FPS = 1080, 1920, 30

# White/red caption band thresholds (fraction of frame height)
CAPTION_TOP_FRAC = 0.22      # top 22 %
CAPTION_BOT_FRAC = 0.75      # bottom starts at 75 %

# Noto Sans Bengali Bold font
NOTO_URL = (
    "https://github.com/google/fonts/raw/main/ofl/"
    "notosansbengali/NotoSansBengali-Bold.ttf"
)
NOTO_LOCAL = "/tmp/NotoSansBengali-Bold.ttf"

# TTS model IDs (preserved from the notebook)
TTS_MODELS = {
    "chatterbox": "EMTIAZZ/chatterbox-bangla-tts",
    "cosyvoice":  "kawshikbuet17/bengali-cosyvoice3-tts",
    "vits":       "EMTIAZZ/bangladeshi-bangla-tts-vits",
    "mms_fallback": "facebook/mms-tts-ben",
}

# Default Bangla script (from localized_script.txt)
DEFAULT_BANGLA_SCRIPT = """\
আপনি যদি আজ ৭ দিনের জন্য আপনার ভাটা থেকে দূরে থাকেন...

তাহলে কি আপনার ভাটা ঠিকভাবে চলবে?

ভাবুন তো...

আপনি না থাকলেও যদি ব্যবসার গুরুত্বপূর্ণ কাজগুলো চলতে থাকে?

কর্মীদের কাজের হিসাব, উৎপাদনের তথ্য, আর বিক্রির হিসাব—

সবকিছু যদি এক জায়গা থেকে সহজেই জানা যায়?

তাহলে ব্যবসা আর শুধু আপনার ওপর নির্ভর করবে না।

আপনি থাকুন, অথবা বাইরে থাকুন...

আপনার ভাটা চলবে নিজের গতিতে।

আর আপনি নিশ্চিন্তে নজর রাখতে পারবেন—যেকোনো জায়গা থেকে।"""

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _base() -> str:
    """Return current BASE (may be mutated by ``set_base``)."""
    return globals().get("_BASE", DEFAULT_BASE)


def set_base(path: str) -> str:
    globals()["_BASE"] = path
    return path


def p(*parts: str) -> str:
    """Shortcut: ``p('work', 'meta.json')`` -> ``<BASE>/work/meta.json``."""
    return os.path.join(_base(), *parts)


def ensure_dirs() -> None:
    """Create every pipeline sub-directory under BASE."""
    for d in SUBDIRS:
        os.makedirs(os.path.join(_base(), d), exist_ok=True)

# ---------------------------------------------------------------------------
# VRAM / GPU helpers
# ---------------------------------------------------------------------------

def print_vram(tag: str) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / 1e9
            resv = torch.cuda.memory_reserved(0) / 1e9
            print(f"[VRAM {tag}] alloc={alloc:.2f}GB reserved={resv:.2f}GB")
        else:
            print(f"[VRAM {tag}] CPU-only")
    except ImportError:
        print(f"[VRAM {tag}] torch not installed")


def cleanup() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def gpu_report() -> None:
    """Print GPU / VRAM info at pipeline start."""
    try:
        import torch
        print(torch.__version__, "| cuda:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))
            total = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"VRAM total: {total:.1f} GB")
    except ImportError:
        print("torch not installed – CPU mode")

# ---------------------------------------------------------------------------
# Shell helper
# ---------------------------------------------------------------------------

def sh(cmd: str) -> str:
    """Run a shell command, return combined stdout+stderr."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.strip() + r.stderr.strip()

# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def autodetect_video() -> str:
    """Find the first ``.mp4`` under ``/kaggle/input``, else a sensible default."""
    import glob as _glob
    cands = _glob.glob("/kaggle/input/**/*.mp4", recursive=True)
    return cands[0] if cands else "/kaggle/working/hindi_reel.mp4"


def read_video_meta(video_path: str) -> dict:
    """Return ``{W, H, FPS, N, DUR}`` using OpenCV (with fallbacks)."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or FALLBACK_FPS
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        dur = n / fps if fps else 0
        return {"W": w, "H": h, "FPS": fps, "N": n, "DUR": dur}
    except Exception:
        return {
            "W": FALLBACK_W,
            "H": FALLBACK_H,
            "FPS": FALLBACK_FPS,
            "N": 0,
            "DUR": 0.0,
        }


def save_meta(meta: dict) -> None:
    with open(p("work", "meta.json"), "w") as f:
        json.dump(meta, f)


def load_meta() -> dict:
    with open(p("work", "meta.json")) as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Script loader
# ---------------------------------------------------------------------------

def load_bangla_script() -> str:
    """Read the Bangla script from work/ or return the built-in default."""
    path = p("work", "localized_script.txt")
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return DEFAULT_BANGLA_SCRIPT

# ---------------------------------------------------------------------------
# Font helper
# ---------------------------------------------------------------------------

def ensure_font() -> str:
    """Download Noto Sans Bengali Bold if missing; return local path."""
    if os.path.exists(NOTO_LOCAL):
        return NOTO_LOCAL
    sh(f"curl -sL -o {NOTO_LOCAL} {NOTO_URL}")
    return NOTO_LOCAL

# ---------------------------------------------------------------------------
# Argparse convenience
# ---------------------------------------------------------------------------

def base_arg() -> str:
    """Parse ``--base`` from argv; return the chosen base path.

    Consumes ``--base <path>`` if present, otherwise uses DEFAULT_BASE.
    Also sets ``globals()["_BASE"]`` so ``p()`` works immediately.
    """
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base", default=DEFAULT_BASE, help="Project root directory")
    args, _ = parser.parse_known_args()
    return set_base(args.base)
