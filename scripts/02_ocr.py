"""02_ocr – Caption detection on keyframes.

Default engine is OpenCV (HSV white/red thresholding on full-res frames).
Set --engine paddle for PP-OCRv5 (requires `bash scripts/install_ocr.sh`).
Set --engine auto to try paddle first, fall back to opencv silently.

Outputs:
    work/text_detections.json  – {frame_name: [{x1,y1,x2,y2,text,conf}, ...]}

Usage:
    python scripts/02_ocr.py [--engine opencv|paddle|auto] [--base ...]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    base_arg, ensure_dirs, p, print_vram, cleanup,
    CAPTION_MIN_W, CAPTION_MIN_H,
    load_meta, keyframe_scale, scale_boxes,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Caption detection on keyframes")
    ap.add_argument("--engine", default="opencv",
                    choices=["opencv", "paddle", "auto"],
                    help="Detection engine (default: opencv)")
    ap.add_argument("--base", default="/kaggle/working/project",
                    help="Project root")
    return ap.parse_args()


# ------------------------------------------------------------------
# OpenCV engine – full-res HSV white/red threshold + contour merge
# ------------------------------------------------------------------

def _opencv_detect(frames: list[str], sx: float = 1.0, sy: float = 1.0) -> dict:
    import cv2

    WHITE_LO, WHITE_HI = np.array([0, 0, 180]), np.array([180, 40, 255])
    RED1_LO, RED1_HI = np.array([0, 70, 50]), np.array([10, 255, 255])
    RED2_LO, RED2_HI = np.array([170, 70, 50]), np.array([180, 255, 255])
    KERN = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 4))

    dets: dict = {}
    for fp in frames:
        img = cv2.imread(fp)
        if img is None:
            dets[os.path.basename(fp)] = []
            continue
        H, W = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        white = cv2.inRange(hsv, WHITE_LO, WHITE_HI)
        red1 = cv2.inRange(hsv, RED1_LO, RED1_HI)
        red2 = cv2.inRange(hsv, RED2_LO, RED2_HI)
        mask = cv2.bitwise_or(white, cv2.bitwise_or(red1, red2))

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERN, iterations=1)
        mask = cv2.dilate(mask, KERN, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        raw = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w < CAPTION_MIN_W or h < CAPTION_MIN_H:
                continue
            if w / max(h, 1) < 1.5:
                continue
            area = w * h
            if area < 800:
                continue
            raw.append((x, y, x + w, y + h, area))

        raw.sort(key=lambda b: b[1])

        merged: list[list[int]] = []
        for box in raw:
            x1, y1, x2, y2, _ = box
            cy = (y1 + y2) / 2
            placed = False
            for mb in merged:
                my1, my2 = mb[1], mb[3]
                if my1 <= cy <= my2 or abs(cy - (my1 + my2) / 2) < (H * 0.04):
                    mb[0] = min(mb[0], x1)
                    mb[1] = min(mb[1], y1)
                    mb[2] = max(mb[2], x2)
                    mb[3] = max(mb[3], y2)
                    placed = True
                    break
            if not placed:
                merged.append([x1, y1, x2, y2])

        boxes = []
        for mb in merged:
            boxes.append({
                "x1": float(mb[0]), "y1": float(mb[1]),
                "x2": float(mb[2]), "y2": float(mb[3]),
                "text": "", "conf": 0.5,
            })
        boxes = scale_boxes(boxes, sx, sy)
        dets[os.path.basename(fp)] = boxes
        print(f"  {os.path.basename(fp)} -> {len(boxes)} (opencv)")
    return dets


# ------------------------------------------------------------------
# PaddleOCR engine
# ------------------------------------------------------------------

def _paddle_detect(frames: list[str], sx: float = 1.0, sy: float = 1.0) -> dict:
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
        boxes = scale_boxes(boxes, sx, sy)
        dets[os.path.basename(fp)] = boxes
        print(f"  {os.path.basename(fp)} -> {len(boxes)} (paddle)")
    del ocr
    cleanup()
    return dets


# ------------------------------------------------------------------
# Engine router
# ------------------------------------------------------------------

def _try_paddle(frames: list[str], sx: float = 1.0, sy: float = 1.0) -> dict | None:
    try:
        return _paddle_detect(frames, sx, sy)
    except Exception as e:
        print(f"  PaddleOCR unavailable: {e}")
        return None


def main() -> None:
    args = parse_args()
    base_arg()
    ensure_dirs()

    frames = sorted(glob.glob(p("work", "f_*.jpg")))
    if not frames:
        print("No keyframes found – run 01_extract.py first")
        return

    meta_path = p("work", "meta.json")
    if not os.path.exists(meta_path):
        print("ERROR: work/meta.json not found – run 01_extract.py first")
        sys.exit(1)
    meta = load_meta()
    sx, sy = keyframe_scale(frames[0], meta)
    print(f"Scale: keyframe -> full-res: sx={sx:.4f} sy={sy:.4f}")

    print_vram("pre-ocr")
    engine = args.engine
    dets: dict | None = None

    if engine == "paddle":
        dets = _try_paddle(frames, sx, sy)
        if dets is None:
            print("FATAL: --engine paddle requested but PaddleOCR unavailable.")
            print("       Install with: bash scripts/install_ocr.sh")
            sys.exit(1)
    elif engine == "auto":
        dets = _try_paddle(frames, sx, sy)
        if dets is not None:
            engine = "paddle"
        else:
            print("  auto: falling back to opencv")
            engine = "opencv"
    if dets is None:
        dets = _opencv_detect(frames, sx, sy)

    print(f"Engine used: {engine}")
    out = p("work", "text_detections.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dets, f, ensure_ascii=False)
    print(f"Saved {out} ({len(dets)} frames)")
    print_vram("post-ocr")


if __name__ == "__main__":
    main()
