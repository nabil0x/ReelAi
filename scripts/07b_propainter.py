"""07b_propainter – ProPainter video-inpainting on tracked text regions.

Builds per-frame mask videos from text_tracks.json (white dilated boxes on
black, matching source W/H/FPS), clones the ProPainter repo on first run,
then invokes inference_propainter.py in chunks (--chunk_size default 8, fp16)
to produce a cleaned video.

Outputs:
    work/clean_propainter.mp4

Usage:
    python scripts/07b_propainter.py --video /kaggle/input/reel.mp4 [--base ...]
                                                 [--chunk_size 8] [--fp16]
"""
from __future__ import annotations
import argparse, json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    base_arg, ensure_dirs, p, print_vram, cleanup,
    autodetect_video, source_video, sh, load_meta, FALLBACK_W, FALLBACK_H, FALLBACK_FPS,
)

REPO_URL = "https://github.com/sczhou/ProPainter.git"
REPO_DIR_NAME = "ProPainter"

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ProPainter video-inpainting")
    ap.add_argument("--video", default=None, help="Input .mp4 path")
    ap.add_argument("--base", default="/kaggle/working/project", help="Project root")
    ap.add_argument("--chunk_size", type=int, default=8,
                    help="Frames per inference chunk (default 8)")
    ap.add_argument("--fp16", action="store_true", default=True,
                    help="Use fp16 inference (default: on)")
    ap.add_argument("--no-fp16", dest="fp16", action="store_false",
                    help="Disable fp16, use fp32")
    return ap.parse_args()

# ------------------------------------------------------------------
# Mask video builder
# ------------------------------------------------------------------

def _build_mask_video(video_path: str, tracks_path: str, meta: dict,
                      out_path: str) -> None:
    """Write a mask MP4: white dilated text boxes on black background."""
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or meta.get("W", FALLBACK_W)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or meta.get("H", FALLBACK_H)
    FPS = cap.get(cv2.CAP_PROP_FPS) or meta.get("FPS", FALLBACK_FPS)

    with open(tracks_path, encoding="utf-8") as f:
        tracks = json.load(f)

    # Build frame->boxes lookup
    frame_boxes: dict[str, list[dict]] = {}
    for t in tracks:
        for fname in t.get("frames", []):
            frame_boxes.setdefault(fname, []).append(t["box"])

    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    idx = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        mask = np.zeros((H, W), dtype=np.uint8)
        # Predict which f_%03d.jpg this frame index corresponds to
        fname = f"f_{idx // max(1, int(FPS / 2)) + 1:03d}.jpg"
        for box in frame_boxes.get(fname, []):
            x1 = int(max(0, box["x1"]))
            y1 = int(max(0, box["y1"]))
            x2 = int(min(W, box["x2"]))
            y2 = int(min(H, box["y2"]))
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        # Dilate to cover edges cleanly
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, kern, iterations=2)
        # 3-channel for VideoWriter
        vw.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        idx += 1
    cap.release()
    vw.release()
    print(f"Mask video -> {out_path} ({idx} frames, {W}x{H} @ {FPS:.1f}fps)")

# ------------------------------------------------------------------
# ProPainter clone + inference
# ------------------------------------------------------------------

def _clone_propainter(dest: str) -> str:
    """Clone ProPainter repo if not already present. Returns repo root path."""
    repo = os.path.join(dest, REPO_DIR_NAME)
    if os.path.isdir(os.path.join(repo, ".git")):
        print(f"ProPainter already cloned at {repo}")
        return repo
    os.makedirs(dest, exist_ok=True)
    sh(f"git clone --depth 1 {REPO_URL} {repo}")
    print(f"Cloned ProPainter -> {repo}")
    return repo

def _run_propainter(repo_dir: str, video: str, mask: str, out: str,
                    chunk_size: int, fp16: bool) -> None:
    """Invoke ProPainter inference_propainter.py."""
    script = os.path.join(repo_dir, "inference_propainter.py")
    if not os.path.isfile(script):
        raise FileNotFoundError(f"inference_propainter.py not found in {repo_dir}")

    fp16_flag = "--fp16" if fp16 else ""
    cmd = (
        f"cd {repo_dir} && python inference_propainter.py "
        f"--video {video} --mask {mask} --output {out} "
        f"--chunk_size {chunk_size} {fp16_flag}"
    )
    print(f"Running: {cmd}")
    result = sh(cmd)
    print(result)

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

    print_vram("pre-propainter")

    # 1. Build mask video
    mask_path = p("work", "mask_propainter.mp4")
    try:
        _build_mask_video(video, tracks_path, meta, mask_path)
    except Exception as e:
        print(f"ERROR building mask video: {e}")
        print("SKIP – ProPainter requires valid masks. Use 07_full.py instead.")
        sys.exit(0)

    # 2. Clone ProPainter
    try:
        repo_dir = _clone_propainter(p("models"))
    except Exception as e:
        print(f"ERROR cloning ProPainter: {e}")
        print("SKIP – install git or check network. Use 07_full.py instead.")
        sys.exit(0)

    # 3. Run inference
    out_path = p("work", "clean_propainter.mp4")
    try:
        _run_propainter(repo_dir, video, mask_path, out_path,
                        args.chunk_size, args.fp16)
        print(f"ProPainter output -> {out_path}")
    except Exception as e:
        print(f"ERROR ProPainter inference: {e}")
        print("SKIP – check GPU memory / model weights. Use 07_full.py instead.")

    print_vram("post-propainter")

if __name__ == "__main__":
    main()
