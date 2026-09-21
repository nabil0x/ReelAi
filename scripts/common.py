"""Shared utilities for the Hindi->Bangla reel pipeline."""
from __future__ import annotations
import gc, glob, json, os, subprocess, sys
from pathlib import Path

DEFAULT_BASE = "/kaggle/working/project"
SUBDIRS = ["input", "work", "models", "output", "previews", "tts_tests"]
FALLBACK_W, FALLBACK_H, FALLBACK_FPS = 1080, 1920, 30

CAPTION_TOP_FRAC = 0.22
CAPTION_BOT_FRAC = 0.75
CAPTION_MASK_DILATE = 9
CAPTION_MIN_W = 60
CAPTION_MIN_H = 14

NOTO_URL = (
    "https://github.com/google/fonts/raw/main/ofl/"
    "notosansbengali/NotoSansBengali-Bold.ttf"
)
NOTO_LOCAL = "/tmp/NotoSansBengali-Bold.ttf"

TTS_MODELS = {
    "chatterbox": "EMTIAZZ/chatterbox-bangla-tts",
    "cosyvoice":  "kawshikbuet17/bengali-cosyvoice3-tts",
    "vits":       "EMTIAZZ/bangladeshi-bangla-tts-vits",
    "mms_fallback": "facebook/mms-tts-ben",
    "jongy5":     "jongy5/chatterbox-bangla",
}

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

# -- Path helpers --

def _base() -> str:
    return globals().get("_BASE", DEFAULT_BASE)

def set_base(path: str) -> str:
    globals()["_BASE"] = path
    return path

def p(*parts: str) -> str:
    return os.path.join(_base(), *parts)

def ensure_dirs() -> None:
    for d in SUBDIRS:
        os.makedirs(os.path.join(_base(), d), exist_ok=True)

# -- VRAM / GPU --

def print_vram(tag: str) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            a = torch.cuda.memory_allocated(0) / 1e9
            r = torch.cuda.memory_reserved(0) / 1e9
            print(f"[VRAM {tag}] alloc={a:.2f}GB reserved={r:.2f}GB")
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
    try:
        import torch
        print(torch.__version__, "| cuda:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))
            t = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"VRAM total: {t:.1f} GB")
    except ImportError:
        print("torch not installed – CPU mode")

# -- Shell / pip helpers --

def sh(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.strip() + r.stderr.strip()

def pip_install(*pkgs: str, no_deps: bool = False, quiet: bool = True) -> int:
    """pip install with constraints file when available. Returns exit code."""
    cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd.append("-q")
    if no_deps:
        cmd.append("--no-deps")
    constraints = "/kaggle/working/constraints.txt"
    if os.path.isfile(constraints):
        cmd += ["-c", constraints]
    cmd += list(pkgs)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"pip_install failed ({r.returncode}): {r.stderr.strip()[-300:]}")
    return r.returncode

# -- Video helpers --

def autodetect_video() -> str:
    import glob as _glob
    cands = _glob.glob("/kaggle/input/**/*.mp4", recursive=True)
    return cands[0] if cands else "/kaggle/working/hindi_reel.mp4"

SRC_H264_NAME = "source_h264.mp4"


def normalize_video(video: str) -> str:
    """Transcode the input (AV1/HEVC/...) to H.264 in work/.

    OpenCV's bundled FFmpeg frequently cannot decode AV1, which makes
    every downstream ``cv2.VideoCapture`` stage read zero frames. ffmpeg
    itself decodes AV1 fine, so we normalise once and let all later
    stages read the H.264 copy.
    """
    out = p("work", SRC_H264_NAME)
    if os.path.exists(out):
        print(f"Normalized source already present -> {out}")
        return out
    sh(
        f'ffmpeg -y -v error -i "{video}" -an '
        f'-c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p '
        f'"{out}"'
    )
    if os.path.exists(out):
        print(f"Normalized source -> {out}")
        return out
    print("WARNING: H.264 normalisation failed – falling back to original input")
    return video


def source_video(fallback: str | None = None) -> str:
    """Video path every downstream stage should read.

    Prefers work/source_h264.mp4 (created by 01_extract.py) so that cv2 can
    decode it, regardless of the original codec.
    """
    norm = p("work", SRC_H264_NAME)
    if os.path.exists(norm):
        return norm
    return fallback or autodetect_video()


def read_video_meta(video_path: str) -> dict:
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or FALLBACK_FPS
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return {"W": w, "H": h, "FPS": fps, "N": n, "DUR": n / fps if fps else 0}
    except Exception:
        return {"W": FALLBACK_W, "H": FALLBACK_H, "FPS": FALLBACK_FPS, "N": 0, "DUR": 0.0}

def save_meta(meta: dict) -> None:
    with open(p("work", "meta.json"), "w") as f:
        json.dump(meta, f)

def load_meta() -> dict:
    with open(p("work", "meta.json")) as f:
        return json.load(f)

def keyframe_scale(keyframe_path: str, meta: dict | None = None) -> tuple[float, float]:
    """Return (sx, sy) to map keyframe coords -> full-res coords."""
    import cv2
    if meta is None:
        meta = load_meta()
    kf = cv2.imread(keyframe_path)
    if kf is None:
        return 1.0, 1.0
    kh, kw = kf.shape[:2]
    if kw == 0 or kh == 0:
        return 1.0, 1.0
    return meta["W"] / kw, meta["H"] / kh

def scale_boxes(boxes: list[dict], sx: float, sy: float) -> list[dict]:
    """Scale box coordinates from keyframe space to full-res space."""
    for b in boxes:
        b["x1"] *= sx
        b["y1"] *= sy
        b["x2"] *= sx
        b["y2"] *= sy
    return boxes

# -- Script / font --

def load_bangla_script() -> str:
    path = p("work", "localized_script.txt")
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return DEFAULT_BANGLA_SCRIPT

def ensure_font() -> str:
    if os.path.exists(NOTO_LOCAL):
        return NOTO_LOCAL
    sh(f"curl -sL -o {NOTO_LOCAL} {NOTO_URL}")
    return NOTO_LOCAL

# -- Argparse --

def base_arg() -> str:
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base", default=DEFAULT_BASE, help="Project root directory")
    args, _ = parser.parse_known_args()
    return set_base(args.base)

# -- Per-frame mask builder (used by 06_preview.py, 07_full.py) --

def build_frame_masks(
    W: int, H: int, fps: float, n_frames: int,
    tracks_path: str | None = None,
) -> list | None:
    """Build per-frame inpaint masks from tracked boxes. None = use band fallback."""
    import cv2
    try:
        import numpy as np
    except ImportError:
        return None
    if tracks_path is None:
        tracks_path = p("work", "text_tracks.json")
    if not os.path.isfile(tracks_path):
        return None
    with open(tracks_path, encoding="utf-8") as f:
        tracks = json.load(f)
    if not tracks:
        return None
    kf_boxes: dict[str, list[dict]] = {}
    for t in tracks:
        for fn in t.get("frames", []):
            kf_boxes.setdefault(fn, []).append(t["box"])
    fpkf = max(1, fps / 2.0)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CAPTION_MASK_DILATE,) * 2)
    masks = []
    for fi in range(n_frames):
        kf_name = f"f_{int(fi / fpkf) + 1:03d}.jpg"
        boxes = kf_boxes.get(kf_name, [])
        m = np.zeros((H, W), dtype=np.uint8)
        for b in boxes:
            x1, y1 = int(max(0, b["x1"])), int(max(0, b["y1"]))
            x2, y2 = int(min(W, b["x2"])), int(min(H, b["y2"]))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(m, (x1, y1), (x2, y2), 255, -1)
        masks.append(cv2.dilate(m, kern, iterations=1))
    return masks


def build_caption_anchors(
    H: int, fps: float, n_frames: int,
    tracks_path: str | None = None, smooth_win: int = 30,
) -> list[str]:
    """Per-frame anchor ('top'/'bottom') derived from tracked box centres.

    Smoothed over ``smooth_win`` frames to avoid jitter. Falls back to
    all-'top' when tracks are absent.
    """
    if tracks_path is None:
        tracks_path = p("work", "text_tracks.json")
    if not os.path.isfile(tracks_path):
        return ["top"] * n_frames
    with open(tracks_path, encoding="utf-8") as f:
        tracks = json.load(f)
    if not tracks:
        return ["top"] * n_frames

    kf_boxes: dict[str, list[dict]] = {}
    for t in tracks:
        for fn in t.get("frames", []):
            kf_boxes.setdefault(fn, []).append(t["box"])

    fpkf = max(1, fps / 2.0)
    mid = H / 2.0
    raw = []
    for fi in range(n_frames):
        kf_name = f"f_{int(fi / fpkf) + 1:03d}.jpg"
        boxes = kf_boxes.get(kf_name, [])
        if not boxes:
            raw.append(0.0)
            continue
        cy_mean = sum((b["y1"] + b["y2"]) / 2 for b in boxes) / len(boxes)
        raw.append(1.0 if cy_mean > mid else 0.0)

    anchors = []
    half_w = smooth_win // 2
    for fi in range(n_frames):
        lo = max(0, fi - half_w)
        hi = min(n_frames, fi + half_w + 1)
        avg = sum(raw[lo:hi]) / max(1, hi - lo)
        anchors.append("bottom" if avg > 0.5 else "top")
    return anchors
