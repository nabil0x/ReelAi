# 🇧🇩 Hindi Reel → Bangla Reel

Automated pipeline to localise a Hindi 2D animated reel into Bangla for the Bangladesh market. Runs entirely on **Kaggle GPU** (≤16 GB VRAM, $0).

## Pipeline Overview

```
VIDEO/AUDIO split → PP-OCRv5 Hindi → IoU tracking → temporal inpaint
                                ↓
Bangla script → TTS shootout (4 models) → best voice
                                ↓
Caption overlay → FFmpeg compose → final bangladesh_version.mp4
```

## Quickstart (Kaggle)

### 1. Clone this repo on a Kaggle GPU notebook

```python
!git clone https://github.com/nabil0x/ReelAi.git /kaggle/working/ReelAi
%cd /kaggle/working/ReelAi
!pip install -r requirements.txt
```

### 2. Upload your Hindi reel

Upload `*.mp4` as a Kaggle dataset attachment, or place it anywhere under `/kaggle/input/`.

### 3. Run each stage in order

```python
!python scripts/00_setup.py                          # GPU report + dirs
!python scripts/01_extract.py --video /kaggle/input/your_reel.mp4
!python scripts/02_ocr.py                            # PP-OCRv5 Hindi
!python scripts/03_track.py                          # IoU tracking
!python scripts/04_tts_shootout.py                   # Test 4 TTS models
```

### 4. Pick the best TTS model

Listen to the files in `tts_tests/` and note the winner (e.g. `vits`).

### 5. Generate voice, preview, and final video

```python
!python scripts/05_voice.py --model vits             # ← replace with your winner
!python scripts/06_preview.py --video /kaggle/input/your_reel.mp4
!python scripts/07_full.py --video /kaggle/input/your_reel.mp4
!python scripts/08_package.py --model vits
```

### 6. Download

Grab `output/bangladesh_reel_adaptation.zip` from the Kaggle output panel.

## Important Notes

| Note | Detail |
|------|--------|
| **BEST model** | Set `--model` in `05_voice.py` / `07_full.py` / `08_package.py` to the winner from the shootout. Default is `vits`. |
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
| `scripts/04_tts_shootout.py` | TTS model shootout (Chatterbox / CosyVoice / VITS / MMS) |
| `scripts/05_voice.py` | Full Bangla voice generation with winning model |
| `scripts/06_preview.py` | 8-second preview with temporal inpaint + Bangla overlay |
| `scripts/07_full.py` | Chunked full render + H.264/AAC compose |
| `scripts/08_package.py` | Quality report + delivery ZIP |

## Requirements

See [`requirements.txt`](requirements.txt). Key packages: `paddlepaddle-gpu`, `paddleocr`, `opencv-python`, `librosa`, `soundfile`, `transformers`, `coqui-tts`.

## License

Public — see repo for details.
