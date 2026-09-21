# 🇧🇩 Hindi Reel → Bangla Reel

Automated pipeline to localise a Hindi 2D animated reel into Bangla for the Bangladesh market. Runs entirely on **Kaggle GPU** (≤16 GB VRAM, $0).

## Pipeline Overview

```
VIDEO/AUDIO split → OpenCV caption detection → IoU tracking
                                                  ↓
                                    Caption cleaning (choose one):
                                      07_full.py  – temporal-median + Telea (fastest)
                                      07b_propainter.py – ProPainter neural inpaint (best quality)
                                      07c_lama.py – LaMa per-box cleaner (isolated frames)
                                                  ↓
Bangla script → TTS shootout (5 models) → full-script generation → timing match
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
> `huggingface-hub` / `accelerate` / `transformers`. Always use it instead of a
> bare `pip install -r requirements.txt`.

### 2. (Optional) Install PP-OCRv5

The default caption detector uses **OpenCV** (white/red HSV thresholding) and
requires no extra packages.  If you want PP-OCRv5 Hindi text recognition for
potentially better detection on complex scenes:

```python
!bash scripts/install_ocr.sh
```

This installs `paddlepaddle-gpu` and `paddleocr` from Paddle's own CUDA index.
If it fails (CUDA mismatch, network issue), the pipeline continues with OpenCV.

### 3. The input reel

The development reel ships in the repo at `assets/hindi_reel.mp4`, so no upload is
needed — `--video` can be omitted entirely. `autodetect_video()` checks
`/kaggle/input/**/*.mp4` first, then falls back to `assets/`.

To use a different reel, upload it as a Kaggle dataset and pass
`--video /kaggle/input/<dataset>/your_reel.mp4`.

### 4. Run each stage in order

```python
!python scripts/00_setup.py                          # GPU report + dirs
!python scripts/01_extract.py --video /kaggle/input/your_reel.mp4
!python scripts/02_ocr.py                            # OpenCV caption detection (default)
!python scripts/03_track.py                          # IoU tracking
!python scripts/04_tts_shootout.py                   # Test 5 TTS engines
!python scripts/04_tts_shootout.py --ref-voice your_voice.wav   # if you want cloning
```

> If you installed PP-OCRv5, you can try `!python scripts/02_ocr.py --engine paddle` or
> `--engine auto` (tries paddle, falls back to opencv silently).

### 5. Pick the best TTS model

Listen to the files in `tts_tests/` and note the winner (e.g. `vits`).

### 6. Generate voice, clean captions, preview, and final video

```python
!python scripts/05_voice.py --model vits             # ← replace with your winner
```

> `05_voice.py` generates the **full Bangla script** with the selected model
> (not a 12s sample stretch). It writes `bangla_voice_raw.wav` (natural, full-length),
> then timing-matches to the original video duration. If full generation fails,
> it falls back to the shootout sample wav with a loud warning.

```python
!python scripts/06_preview.py --video /kaggle/input/your_reel.mp4

# Pick ONE caption cleaner (or chain 07b → 07c):
!python scripts/07_full.py --video /kaggle/input/your_reel.mp4       # temporal-median (fast)
!python scripts/07b_propainter.py --video /kaggle/input/your_reel.mp4 # ProPainter (best)
!python scripts/07c_lama.py --video /kaggle/input/your_reel.mp4       # LaMa (isolated frames)

!python scripts/08_package.py --model vits
```

### 7. Download

Grab `output/bangladesh_reel_adaptation.zip` from the Kaggle output panel.

## Important Notes

| Note | Detail |
|------|--------|
| **Reference voice** | `--ref-voice <wav>` (3-6 s clean Bangla speech). If omitted, the demo reference published with `jongy5/chatterbox-bangla` (`audios/refs/001.wav`) is fetched at runtime and cached to `models/refs/` — nothing is bundled in git. **Cloning a speaker requires that speaker's permission**; supply your own wav for production. `cosyvoice` needs a reference (demo one is used automatically). |
| **Engine notes** | `chatterbox` (EMTIAZZ) needs base ResembleAI files + fine-tuned T3 swap — handled automatically. `jongy5` ships a complete dir and loads directly. `mms_fallback` is the always-works last resort. `cosyvoice` needs the CosyVoice repo and frequently SKIPs on Kaggle. |
| **BEST model** | Set `--model` in `05_voice.py` / `08_package.py` to the winner from the shootout. Choices: `chatterbox`, `cosyvoice`, `vits`, `mms_fallback`, `jongy5`. Default is `vits`. |
| **Preview first** | Run `06_preview.py` before the full render to verify overlay quality on 8 seconds. |
| **Caption detector** | Default is OpenCV (no extra packages). PP-OCRv5 is optional via `--engine paddle`. |
| **Video path** | Every script accepts `--video <path>`. If omitted, the first `.mp4` under `/kaggle/input/` is auto-detected. |
| **Project root** | Override with `--base <path>` (default: `/kaggle/working/project`). |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/common.py` | Shared utilities (VRAM, paths, shell, font, mask builder) |
| `scripts/00_setup.py` | GPU report + directories + default Bangla script |
| `scripts/01_extract.py` | ffprobe metadata + mono WAV + keyframe extraction |
| `scripts/02_ocr.py` | Caption detection: OpenCV (default) / PP-OCRv5 (optional) |
| `scripts/03_track.py` | IoU-based text tracking + EMA smoothing |
| `scripts/tts_engines.py` | The five TTS engines, one shared `gen()` contract |
| `scripts/audio_utils.py` | Sentence splitting, silence trim, chunk concat |
| `scripts/04_tts_shootout.py` | Runs every engine, reports which produced audio |
| `scripts/05_voice.py` | Full-script narration + timing match to video duration |
| `scripts/06_preview.py` | 8-second preview with temporal inpaint + Bangla overlay |
| `scripts/07_full.py` | Chunked temporal-median clean + overlay + H.264/AAC compose |
| `scripts/07b_propainter.py` | ProPainter neural video inpainting on tracked text masks |
| `scripts/07c_lama.py` | LaMa single-image cleaning for isolated / low-motion frames |
| `scripts/08_package.py` | Quality report + delivery ZIP |
| `scripts/install.sh` | Kaggle-safe dependency install with constraints file |
| `scripts/install_ocr.sh` | Optional PP-OCRv5 install (paddlepaddle-gpu + paddleocr) |

## Requirements

See [`requirements.txt`](requirements.txt). Core inference deps: `coqui-tts` (runtime, `--no-deps`), `anyascii`, `einops`, `loguru`, etc.  Kaggle already ships `torch`, `transformers`, `accelerate`, `opencv-python`, `librosa`, `soundfile` — these must NOT be listed in requirements to avoid pip backtracking.

## License

Public — see repo for details.
