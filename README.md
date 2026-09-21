# 🇧🇩 Hindi Reel → Bangla Reel

Automated pipeline to localise a Hindi 2D animated reel into Bangla for the Bangladesh market. Runs entirely on **Kaggle GPU** (≤16 GB VRAM, $0).

## Pipeline Overview

```
VIDEO/AUDIO split → PP-OCRv5 Hindi → IoU tracking
                                         ↓
                              Caption cleaning (choose one):
                                07_full.py  – temporal-median + Telea (fastest)
                                07b_propainter.py – ProPainter neural inpaint (best quality)
                                07c_lama.py – LaMa per-box cleaner (isolated frames)
                                         ↓
Bangla script → TTS shootout (5 models) → best voice
                                         ↓
Caption overlay → FFmpeg compose → final bangladesh_version.mp4
```

### Cleaning hierarchy

| Stage | Script | When to use |
|-------|--------|-------------|
| Temporal-median | `07_full.py` | Default – fast, CPU-friendly, good for simple caption bands |
| ProPainter | `07b_propainter.py` | Complex motion / occlusion – neural video inpainting (fp16, chunk=8) |
| LaMa | `07c_lama.py` | Isolated / low-motion frames – single-image inpainting per tracked box |

Run **one** of the three cleaning scripts before the overlay stage, or chain them:
`07b` (video-level) → `07c` (spot-fix remaining frames).

## Quickstart (Kaggle)

### 1. Clone this repo on a Kaggle GPU notebook

```python
!git clone https://github.com/nabil0x/ReelAi.git /kaggle/working/ReelAi
%cd /kaggle/working/ReelAi
!bash scripts/install.sh
```

> `install.sh` freezes the packages Kaggle already ships into a constraints file
> and installs with `-c`, so pip never re-resolves (and never backtracks on)
> `huggingface-hub` / `paddlex` / `modelscope`. Always use it instead of a bare
> `pip install -r requirements.txt`.

### 2. Upload your Hindi reel

Upload `*.mp4` as a Kaggle dataset attachment, or place it anywhere under `/kaggle/input/`.

### 3. Run each stage in order

```python
!python scripts/00_setup.py                          # GPU report + dirs
!python scripts/01_extract.py --video /kaggle/input/your_reel.mp4
!python scripts/02_ocr.py                            # PP-OCRv5 Hindi
!python scripts/03_track.py                          # IoU tracking
!python scripts/04_tts_shootout.py                   # Test 5 TTS models
```

### 4. Pick the best TTS model

Listen to the files in `tts_tests/` and note the winner (e.g. `vits`).

### 5. Generate voice, clean captions, preview, and final video

```python
!python scripts/05_voice.py --model vits             # ← replace with your winner
!python scripts/06_preview.py --video /kaggle/input/your_reel.mp4

# Pick ONE caption cleaner (or chain 07b → 07c):
!python scripts/07_full.py --video /kaggle/input/your_reel.mp4       # temporal-median (fast)
!python scripts/07b_propainter.py --video /kaggle/input/your_reel.mp4 # ProPainter (best)
!python scripts/07c_lama.py --video /kaggle/input/your_reel.mp4       # LaMa (isolated frames)

!python scripts/08_package.py --model vits
```

### 6. Download

Grab `output/bangladesh_reel_adaptation.zip` from the Kaggle output panel.

## Important Notes

| Note | Detail |
|------|--------|
| **BEST model** | Set `--model` in `05_voice.py` / `08_package.py` to the winner from the shootout. Choices: `chatterbox`, `cosyvoice`, `vits`, `mms_fallback`, `jongy5`. Default is `vits`. |
| **Preview first** | Run `06_preview.py` before the full render to verify overlay quality on 8 seconds. |
| **Video path** | Every script accepts `--video <path>`. If omitted, the first `.mp4` under `/kaggle/input/` is auto-detected. |
| **Project root** | Override with `--base <path>` (default: `/kaggle/working/project`). |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/common.py` | Shared utilities (VRAM, paths, shell, font) |
| `scripts/00_setup.py` | GPU report + directories + default Bangla script |
| `scripts/01_extract.py` | ffprobe metadata + mono WAV + keyframe extraction |
| `scripts/02_ocr.py` | PP-OCRv5 Hindi detection (with white/red-box fallback) |
| `scripts/03_track.py` | IoU-based text tracking + EMA smoothing |
| `scripts/04_tts_shootout.py` | TTS model shootout (Chatterbox / CosyVoice / VITS / MMS / jongy5) |
| `scripts/05_voice.py` | Full Bangla voice generation with winning model |
| `scripts/06_preview.py` | 8-second preview with temporal inpaint + Bangla overlay |
| `scripts/07_full.py` | Chunked temporal-median clean + overlay + H.264/AAC compose |
| `scripts/07b_propainter.py` | ProPainter neural video inpainting on tracked text masks |
| `scripts/07c_lama.py` | LaMa single-image cleaning for isolated / low-motion frames |
| `scripts/08_package.py` | Quality report + delivery ZIP |

## Requirements

See [`requirements.txt`](requirements.txt). Key packages: `paddlepaddle-gpu`, `paddleocr`, `opencv-python`, `librosa`, `soundfile`, `transformers`, `coqui-tts`.

## License

Public — see repo for details.
