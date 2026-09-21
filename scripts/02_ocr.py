"""02_ocr – PP-OCRv5 Hindi detection on keyframes with white/red-box fallback.

Outputs:
    work/text_detections.json  – {frame_name: [{x1,y1,x2,y2,text,conf}, ...]}

Usage:
    python scripts/02_ocr.py [--base ...]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import base_arg, ensure_dirs, p, print_vram, cleanup

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OCR Hindi text in keyframes")
    ap.add_argument("--base", default="/kaggle/working/project", help="Project root")
    return ap.parse_args()

def _ocr_paddle(frames: list[str]) -> dict:
    """Run PP-OCRv5 Hindi on every frame. Returns detections dict."""
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="hi", use_textline_orientation=True)
    dets: dict = {}
    for fp in frames:
        res = ocr.predict(fp)
        boxes = []
        for page in res:
            for box, txt, conf in zip(
                page.get("rec_boxes", []),
                page.get("rec_texts", []),
                page.get("rec_scores", []),
            ):
                x1, y1, x2, y2 = map(float, box)
                boxes.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "text": txt, "conf": float(conf),
                })
        dets[os.path.basename(fp)] = boxes
        print(f"  {os.path.basename(fp)} -> {len(boxes)} detections")
    del ocr
    cleanup()
    return dets

def _fallback_heuristic(frames: list[str]) -> dict:
    """White/red colour-band heuristic when PaddleOCR is unavailable."""
    import cv2
    dets: dict = {}
    for fp in frames:
        img = cv2.imread(fp)
        if img is None:
            dets[os.path.basename(fp)] = []
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 40, 255]))
        red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(white, cv2.bitwise_or(red1, red2))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w > 60 and h > 14 and w / h > 2 and (y < 300 or y > img.shape[0] - 320):
                boxes.append({
                    "x1": float(x * 2), "y1": float(y * 2),
                    "x2": float((x + w) * 2), "y2": float((y + h) * 2),
                    "text": "", "conf": 0.5,
                })
        dets[os.path.basename(fp)] = boxes
        print(f"  {os.path.basename(fp)} -> {len(boxes)} (heuristic)")
    return dets

def main() -> None:
    args = parse_args()
    base_arg()
    ensure_dirs()

    frames = sorted(glob.glob(p("work", "f_*.jpg")))
    if not frames:
        print("No keyframes found – run 01_extract.py first"); return

    print_vram("pre-ocr")
    try:
        dets = _ocr_paddle(frames)
    except Exception as e:
        print(f"PaddleOCR failed, white/red-box fallback: {e}")
        dets = _fallback_heuristic(frames)

    out = p("work", "text_detections.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dets, f, ensure_ascii=False)
    print(f"Saved {out} ({len(dets)} frames)")
    print_vram("post-ocr")

if __name__ == "__main__":
    main()
