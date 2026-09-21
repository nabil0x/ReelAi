"""06_preview – 8-second temporal-inpaint preview with Bangla overlay.

Outputs:
    previews/preview.mp4  – short preview with audio

Usage:
    python scripts/06_preview.py --video /kaggle/input/reel.mp4 [--base ...]
"""
from __future__ import annotations
import argparse, os, sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    base_arg, ensure_dirs, p, print_vram, cleanup,
    autodetect_video, load_bangla_script, ensure_font,
    CAPTION_TOP_FRAC, CAPTION_BOT_FRAC,
)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="8s preview with Bangla overlay")
    ap.add_argument("--video", default=None, help="Input .mp4 path")
    ap.add_argument("--base", default="/kaggle/working/project", help="Project root")
    ap.add_argument("--seconds", type=int, default=8, help="Preview length in seconds")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    base_arg()
    ensure_dirs()

    video = args.video or autodetect_video()
    font_path = ensure_font()
    print_vram("pre-preview")

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_frames = int(args.seconds * fps)
    frames = []
    for _ in range(n_frames):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    print(f"Preview frames: {len(frames)}")

    # --- Temporal-median clean + Telea inpaint on caption bands ---
    top_px = int(h * CAPTION_TOP_FRAC)
    bot_px = int(h * CAPTION_BOT_FRAC)
    clean = []
    for i, f in enumerate(frames):
        j0 = max(0, i - 3)
        j1 = min(len(frames), i + 4)
        stack = np.stack(frames[j0:j1]).astype(np.float32)
        med = np.median(stack, axis=0).astype(np.uint8)
        m = np.zeros(f.shape[:2], np.uint8)
        m[:top_px, :] = 255
        m[bot_px:, :] = 255
        out = cv2.inpaint(med, m, 3, cv2.INPAINT_TELEA)
        clean.append(out)

    # --- Bangla overlay (first 2 caption lines) ---
    font = ImageFont.truetype(font_path, 44)
    script = load_bangla_script()
    lines = [l for l in script.split("\n") if l.strip()]
    caps = lines[:2] if len(lines) >= 2 else lines

    final = []
    for f in clean:
        pil = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        y = 60
        for c in caps:
            bb = draw.textbbox((0, 0), c, font=font)
            tw, th = bb[2] + 40, bb[3] + 24
            draw.rectangle([(w - tw) // 2, y, (w + tw) // 2, y + th], fill="white")
            draw.text(((w - tw) // 2 + 20, y + 12), c, font=font, fill="black")
            y += th + 10
        final.append(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))

    # --- Write silent video ---
    silent = p("previews", "preview_silent.mp4")
    vw = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in final:
        vw.write(f)
    vw.release()

    # --- Merge audio (H.264 + AAC) ---
    from common import sh
    voice = p("work", "bangla_voice.wav")
    out = p("previews", "preview.mp4")
    sh(
        f"ffmpeg -y -v error -i {silent} -i {voice} "
        f"-t {args.seconds} -c:v libx264 -pix_fmt yuv420p "
        f"-c:a aac -shortest {out}"
    )
    print(f"Preview saved -> {out}")
    print_vram("post-preview")

if __name__ == "__main__":
    main()
