"""03_track – IoU-based tracking + EMA smoothing -> text_tracks.json.

Outputs:
    work/text_tracks.json  – [{id, box, frames}, ...]

Usage:
    python scripts/03_track.py [--base ...]
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from common import base_arg, ensure_dirs, p, DEFAULT_BASE

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="IoU text tracking across frames")
    ap.add_argument("--base", default=DEFAULT_BASE, help="Project root")
    return ap.parse_args()

def iou(a: dict, b: dict) -> float:
    """Intersection-over-union between two {x1,y1,x2,y2} boxes."""
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (
        (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
        + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        - inter + 1e-6
    )
    return inter / ua

def main() -> None:
    args = parse_args()
    base_arg()
    ensure_dirs()

    det_path = p("work", "text_detections.json")
    with open(det_path, encoding="utf-8") as f:
        dets = json.load(f)

    keys = sorted(dets.keys())
    tracks: list[dict] = []
    nxt_id = 0
    active: list[dict] = []

    for k in keys:
        cur_active: list[dict] = []
        for box in dets[k]:
            best_iou, best_idx = 0, -1
            for i, t in enumerate(active):
                s = iou(box, t["box"])
                if s > best_iou:
                    best_iou, best_idx = s, i
            if best_iou > 0.3:
                # EMA smoothing: 70 % old, 30 % new
                t = active[best_idx]
                for kk in ("x1", "y1", "x2", "y2"):
                    t["box"][kk] = 0.7 * t["box"][kk] + 0.3 * box[kk]
                t["frames"].append(k)
                cur_active.append(t)
            else:
                t = {"id": nxt_id, "box": dict(box), "frames": [k]}
                nxt_id += 1
                cur_active.append(t)
                tracks.append(t)
        active = cur_active

    # Drop tracks shorter than 2 frames (noise)
    tracks = [t for t in tracks if len(t["frames"]) >= 2]

    out = p("work", "text_tracks.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tracks, f, ensure_ascii=False)

    text_frame_count = sum(1 for k in keys if dets[k])
    print(f"Tracks: {len(tracks)} | frames with text: {text_frame_count}")
    print(f"Saved -> {out}")

if __name__ == "__main__":
    main()
