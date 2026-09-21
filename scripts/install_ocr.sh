#!/usr/bin/env bash
# OPTIONAL PP-OCRv5 install for scripts/02_ocr.py --engine paddle.
#
# This is NOT required — the default caption detector (OpenCV) works without
# any Paddle packages.  Only run this if you want PaddleOCR's Hindi text
# recognition for potentially better detection on complex scenes.
#
# Usage:
#     bash scripts/install_ocr.sh
#
# Exit codes:
#     0  – success OR graceful skip (prints a message explaining why)

set -uo pipefail

CONSTRAINTS=/kaggle/working/constraints.txt
echo "=== Optional PP-OCRv5 install ==="

# Build constraints file if it doesn't exist yet
if [ ! -f "$CONSTRAINTS" ]; then
    python -m pip freeze \
      | grep -E '^[A-Za-z0-9_.-]+==' \
      > "$CONSTRAINTS" 2>/dev/null || true
fi

# Try installing paddlepaddle-gpu from Paddle's own CUDA index
echo "Attempting paddlepaddle-gpu==3.0.0 install ..."
if python -m pip install --prefer-binary \
    -c "$CONSTRAINTS" \
    paddlepaddle-gpu==3.0.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
    2>&1; then
    echo "paddlepaddle-gpu OK"
else
    echo "SKIP: paddlepaddle-gpu install failed (CUDA version mismatch or network issue)."
    echo "      PP-OCRv5 will not be available.  The default OpenCV detector still works."
    echo "      You can safely ignore this and continue with: python scripts/02_ocr.py"
    exit 0
fi

# Try installing paddleocr
echo "Installing paddleocr ..."
if python -m pip install --prefer-binary -c "$CONSTRAINTS" paddleocr 2>&1; then
    echo "paddleocr OK"
    echo ""
    echo "PP-OCRv5 installed.  You can now run:  python scripts/02_ocr.py --engine paddle"
else
    echo "SKIP: paddleocr install failed."
    echo "      The default OpenCV detector still works fine."
    echo "      Continue with: python scripts/02_ocr.py"
fi

exit 0
