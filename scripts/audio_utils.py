"""Audio helpers shared by the TTS engines and the voice-stage scripts."""
from __future__ import annotations

import re


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.?!।])\s*", text.strip())
    cleaned = [p.strip() for p in parts]
    return [p for p in cleaned if p.strip(".?!। \t\n")]


def trim_silence(wav, sr: int, keep_ms: int = 120):
    import numpy as np

    if wav.size == 0:
        return wav
    win = max(1, int(sr * 0.02))
    n = len(wav) // win * win
    if n == 0:
        return wav
    frames = wav[:n].reshape(-1, win)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-9)
    thresh = max(rms.max() * 0.08, 1e-4)
    voiced = np.where(rms > thresh)[0]
    if voiced.size == 0:
        return wav
    pad = int(sr * keep_ms / 1000 / win)
    lo = max(0, voiced[0] - pad) * win
    hi = min(len(frames), voiced[-1] + 1 + pad) * win
    return wav[lo:hi]


def concat_with_pause(chunks, sr: int, pause_s: float = 0.25):
    import numpy as np

    if not chunks:
        return np.zeros(0, dtype="float32")
    gap = np.zeros(int(sr * pause_s), dtype="float32")
    out = []
    for c in chunks:
        out.append(c.astype("float32"))
        out.append(gap)
    return np.concatenate(out)
