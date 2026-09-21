"""07c_lama – LaMa isolated-frame cleaning for low-motion / tracked text regions.

For frames where text_tracks.json indicates isolated (few-overlap) boxes,
runs lama-cleaner with per-box masks.  Writes cleaned frames to work/lama_clean/.
Falls back to OpenCV Telea inpaint on any per-frame failure.

Outputs:
    work/lama_clean/f_%03d.png   – cleaned frames (PNG, lossless)

Usage:
    python scripts/07c_lama.py --video /kaggle/input/reel.mp4 [--base ...]
                                                 [--dilate 5]
"""
from __future__ import annotations
import argparse, json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    base_arg, ensure_dirs, p, print_vram, cleanup,
    autodetect_video, source_video, load_meta, FALLBACK_W, FALLBACK_H, FALLBACK_FPS,
    pip_install,
    DEFAULT_BASE,
)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LaMa per-box frame cleaner")
    ap.add_argument("--video", default=None, help="Input .mp4 path")
    ap.add_argument("--base", default=DEFAULT_BASE, help="Project root")
    ap.add_argument("--dilate", type=int, default=5,
                    help="Mask dilation kernel size (odd, default 5)")
    return ap.parse_args()

# ------------------------------------------------------------------
# Mask builder
# ------------------------------------------------------------------

def _build_box_mask(W: int, H: int, boxes: list[dict], dilate_k: int) -> np.ndarray:
    """White dilated boxes on black background -> single-channel mask."""
    mask = np.zeros((H, W), dtype=np.uint8)
    for b in boxes:
        x1 = int(max(0, b["x1"]))
        y1 = int(max(0, b["y1"]))
        x2 = int(min(W, b["x2"]))
        y2 = int(min(H, b["y2"]))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    if dilate_k > 0:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        mask = cv2.dilate(mask, kern, iterations=1)
    return mask

# ------------------------------------------------------------------
# LaMa cleaner
# ------------------------------------------------------------------

def _ensure_lama_pkg() -> bool:
    """Runtime-install lama-cleaner with --no-deps. Returns True if importable."""
    try:
        import lama_cleaner  # noqa: F401
        return True
    except ImportError:
        pass
    pip_install("lama-cleaner==1.2.5", no_deps=True)
    try:
        import lama_cleaner  # noqa: F401
        return True
    except ImportError:
        return False

def _try_lama_clean(frame: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Run lama-cleaner on a single frame+mask. Returns cleaned frame or None."""
    try:
        if not _ensure_lama_pkg():
            return None
        from lama_cleaner.model_manager import ModelManager
        from lama_cleaner.schema import Config
        import torch

        # Lazy singleton to avoid re-loading per frame
        if not hasattr(_try_lama_clean, "_mgr"):
            _try_lama_clean._mgr = ModelManager(
                name="lama", device="cuda",
                disable_nsfw_checker=True, no_half=False,
            )
            _try_lama_clean._cfg = Config(
                ldm_steps=20, hd_strategy="Original",
                hd_strategy_crop_margin=32, hd_strategy_crop_trigger_size=512,
            )
        mgr = _try_lama_clean._mgr
        cfg = _try_lama_clean._cfg

        # lama-cleaner expects RGB numpy; return RGB numpy
        from PIL import Image
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        msk_pil = Image.fromarray(mask)
        out_pil = mgr(img_pil, msk_pil, cfg)
        return cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None

def _telea_fallback(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """OpenCV Telea inpaint as fallback."""
    return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    base_arg()
    ensure_dirs()

    video = source_video(args.video)
    meta = load_meta()
    tracks_path = p("work", "text_tracks.json")

    if not os.path.exists(tracks_path):
        print("ERROR: work/text_tracks.json not found – run 03_track.py first")
        sys.exit(1)

    with open(tracks_path, encoding="utf-8") as f:
        tracks = json.load(f)

    # Build frame->boxes lookup
    frame_boxes: dict[str, list[dict]] = {}
    for t in tracks:
        for fname in t.get("frames", []):
            frame_boxes.setdefault(fname, []).append(t["box"])

    W = meta.get("W", FALLBACK_W)
    H = meta.get("H", FALLBACK_H)
    FPS = meta.get("FPS", FALLBACK_FPS)

    cap = cv2.VideoCapture(video)
    out_dir = p("work", "lama_clean")
    os.makedirs(out_dir, exist_ok=True)

    print_vram("pre-lama")
    idx = 0
    cleaned = 0
    telea_fb = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fname = f"f_{idx // max(1, int(FPS / 2)) + 1:03d}.jpg"
        boxes = frame_boxes.get(fname, [])

        if boxes:
            mask = _build_box_mask(W, H, boxes, args.dilate)
            # Try LaMa first
            result = _try_lama_clean(frame, mask)
            if result is not None:
                frame = result
                cleaned += 1
            else:
                frame = _telea_fallback(frame, mask)
                telea_fb += 1
        # else: no text detected -> keep original frame

        out_path = os.path.join(out_dir, f"f_{idx:04d}.png")
        cv2.imwrite(out_path, frame)
        idx += 1

        if idx % 50 == 0:
            print(f"  processed {idx} frames (lama={cleaned}, telea={telea_fb})")
            cleanup()

    cap.release()
    print(f"Done: {idx} frames -> {out_dir}")
    print(f"  LaMa cleaned: {cleaned}, Telea fallback: {telea_fb}, "
          f"unchanged: {idx - cleaned - telea_fb}")
    print_vram("post-lama")

if __name__ == "__main__":
    main()
