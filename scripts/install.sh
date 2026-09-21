#!/usr/bin/env bash
# Kaggle-safe dependency install.
#
# Why this exists: the Kaggle image already ships a large, mutually-consistent
# set of ML packages (torch, transformers, accelerate, huggingface-hub, numpy,
# opencv, librosa, ...). A plain `pip install -r requirements.txt` lets pip
# re-resolve that whole set, and because paddleocr -> paddlex -> modelscope
# span different huggingface-hub ranges, pip falls into an endless
# "Downloading huggingface_hub-1.x.y ... 1.x.y-1 ..." backtracking crawl.
#
# Fix: freeze every already-installed package into a constraints file and pass
# it with -c. pip then treats them as fixed and only installs what's missing.
#
# Usage (from the repo root on Kaggle):
#     bash scripts/install.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONSTRAINTS="${REELAI_CONSTRAINTS:-$REPO_ROOT/constraints.txt}"
REQS="$(dirname "$0")/../requirements.txt"

# Snapshot installed versions, dropping any partial paddle installs so they
# stay upgradable, and dropping editable/URL lines that aren't valid pins.
python -m pip freeze \
  | grep -E '^[A-Za-z0-9_.-]+==' \
  | grep -viE '^(paddlepaddle|paddlepaddle-gpu|paddleocr|paddlex)==' \
  > "$CONSTRAINTS"

echo "Constrained $(wc -l < "$CONSTRAINTS") already-installed packages -> $CONSTRAINTS"
echo "Installing only missing packages from requirements.txt ..."

python -m pip install -q --prefer-binary -r "$REQS" -c "$CONSTRAINTS"

echo "install OK"
