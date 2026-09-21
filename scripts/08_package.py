"""08_package – Quality report + delivery ZIP.

Outputs:
    output/quality_report.json
    output/bangladesh_reel_adaptation.zip

Usage:
    python scripts/08_package.py [--base ...]
"""
from __future__ import annotations
import argparse, json, os, sys, zipfile

sys.path.insert(0, os.path.dirname(__file__))
from common import base_arg, ensure_dirs, p, load_meta, autodetect_video

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Package output + quality report")
    ap.add_argument("--model", default="vits", help="TTS model used (for report)")
    ap.add_argument("--base", default="/kaggle/working/project", help="Project root")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    base_arg()
    ensure_dirs()

    # --- Quality report ---
    video_path = autodetect_video()
    tracks_path = p("work", "text_tracks.json")
    n_tracks = 0
    if os.path.exists(tracks_path):
        with open(tracks_path) as f:
            n_tracks = len(json.load(f))

    meta = load_meta()
    report = {
        "video": video_path,
        "tts_winner": args.model,
        "tracks": n_tracks,
        "W": meta["W"],
        "H": meta["H"],
        "FPS": meta["FPS"],
        "duration_s": meta["DUR"],
    }
    rep_path = p("output", "quality_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Quality report -> {rep_path}")

    # --- ZIP ---
    zp = p("output", "bangladesh_reel_adaptation.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("bangladesh_version.mp4", "localized_script.txt",
                      "quality_report.json"):
            fp = p("output", name)
            if os.path.exists(fp):
                z.write(fp, name)
        voice = p("work", "bangla_voice.wav")
        if os.path.exists(voice):
            z.write(voice, "bangla_voice.wav")
        for name in ("text_detections.json", "text_tracks.json"):
            fp = p("work", name)
            if os.path.exists(fp):
                z.write(fp, name)
        preview = p("previews", "preview.mp4")
        if os.path.exists(preview):
            z.write(preview, "preview.mp4")

    from common import sh
    info = sh(f"ls -lh {zp}")
    print(f"ZIP: {info}")

if __name__ == "__main__":
    main()
