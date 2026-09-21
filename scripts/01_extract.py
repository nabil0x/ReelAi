"""01_extract – Probe video metadata, extract mono 16 kHz WAV, sample frames.

Outputs:
    work/meta.json   – {W, H, FPS, N, DUR}
    work/orig_16k.wav – mono 16 kHz audio
    work/f_*.jpg     – 2 fps keyframes scaled to 540x960

Usage:
    python scripts/01_extract.py --video /kaggle/input/reel.mp4 [--base ...]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    base_arg, ensure_dirs, p, read_video_meta, save_meta, sh, print_vram,
    autodetect_video,
)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract audio + frames from Hindi reel")
    ap.add_argument("--video", default=None, help="Path to input .mp4")
    ap.add_argument("--base", default="/kaggle/working/project", help="Project root")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    base_arg()  # sets BASE from --base in argv

    video = args.video or autodetect_video()
    ensure_dirs()
    print(f"VIDEO_PATH = {video}")

    # --- ffprobe info ---
    probe = sh(
        "ffprobe -v error -select_streams v:0 "
        "-show_entries stream=width,height,avg_frame_rate,duration "
        f"-of default=noprint_wrappers=1 {video}"
    )
    print("ffprobe:\n", probe)

    # --- OpenCV metadata ---
    meta = read_video_meta(video)
    save_meta(meta)
    print(f"{meta['W']}x{meta['H']} @ {meta['FPS']:.1f}fps, "
          f"{meta['N']} frames, {meta['DUR']:.1f}s")

    # --- Extract audio (mono 16 kHz) ---
    wav_path = p("work", "orig_16k.wav")
    sh(f"ffmpeg -y -v error -i {video} -vn -ac 1 -ar 16000 {wav_path}")
    print(f"Audio -> {wav_path}")

    # --- Sample frames at 2 fps, scaled to 540x960 ---
    sh(f"ffmpeg -y -v error -i {video} -vf fps=2,scale=540:960 {p('work', 'f_%03d.jpg')}")
    frames = sorted(glob.glob(p("work", "f_*.jpg")))
    print(f"Frames extracted: {len(frames)}")

if __name__ == "__main__":
    main()
