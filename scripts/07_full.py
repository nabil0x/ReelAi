"""07_full – Chunked full-video clean + Bangla overlay + FFmpeg compose.

Outputs:
    output/bangladesh_version.mp4  – final localised reel

Usage:
    python scripts/07_full.py --video /kaggle/input/reel.mp4 [--base ...]

Note: For complex motion, swap the temporal-median approach with
ProPainter (chunk=8, fp16) on tracked masks from text_tracks.json.
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
    CAPTION_TOP_FRAC, CAPTION_BOT_FRAC, sh, FALLBACK_FPS,
)

CHUNK_SIZE = 30  # frames per processing chunk

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Full clean + overlay + compose")
    ap.add_argument("--video", default=None, help="Input .mp4 path")
    ap.add_argument("--base", default="/kaggle/working/project", help="Project root")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    base_arg()
    ensure_dirs()

    video = args.video or autodetect_video()
    font_path = ensure_font()
    script = load_bangla_script()
    lines = [l for l in script.split("\n") if l.strip()]

    print_vram("pre-full")
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or FALLBACK_FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        all_frames.append(f)
    cap.release()
    print(f"Total frames: {len(all_frames)}")

    top_px = int(h * CAPTION_TOP_FRAC)
    bot_px = int(h * CAPTION_BOT_FRAC)
    font = ImageFont.truetype(font_path, 44)
    per_line = max(1, len(all_frames) // max(1, len(lines)))
    out_frames = []

    n_chunks = (len(all_frames) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for ci in range(0, len(all_frames), CHUNK_SIZE):
        chunk = all_frames[ci:ci + CHUNK_SIZE]
        for j, f in enumerate(chunk):
            gi = min(ci + j, len(all_frames) - 1)
            j0 = max(0, gi - 2)
            j1 = min(len(all_frames), gi + 3)
            stack = np.stack(all_frames[j0:j1]).astype(np.float32)
            med = np.median(stack, axis=0).astype(np.uint8)
            m = np.zeros(f.shape[:2], np.uint8)
            m[:top_px, :] = 255
            m[bot_px:, :] = 255
            cf = cv2.inpaint(med, m, 3, cv2.INPAINT_TELEA)

            pil = Image.fromarray(cv2.cvtColor(cf, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            cap_line = lines[min((ci + j) // per_line, len(lines) - 1)][:42]
            bb = draw.textbbox((0, 0), cap_line, font=font)
            tw, th = bb[2] + 40, bb[3] + 24
            draw.rectangle(
                [(w - tw) // 2, 60, (w + tw) // 2, 60 + th], fill="white",
            )
            draw.text(((w - tw) // 2 + 20, 72), cap_line, font=font, fill="black")
            out_frames.append(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))

        chunk_num = ci // CHUNK_SIZE + 1
        print(f"Chunk {chunk_num}/{n_chunks}")
        cleanup()

    # --- Write silent video ---
    silent = p("work", "final_silent.mp4")
    vw = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in out_frames:
        vw.write(f)
    vw.release()

    # --- Merge audio (H.264 + AAC) ---
    voice = p("work", "bangla_voice.wav")
    final = p("output", "bangladesh_version.mp4")
    sh(
        f"ffmpeg -y -i {silent} -i {voice} "
        f"-c:v libx264 -pix_fmt yuv420p -c:a aac -shortest {final}"
    )
    info = sh(f"ls -lh {final}")
    print(f"FINAL: {info}")
    print_vram("post-full")

if __name__ == "__main__":
    main()
