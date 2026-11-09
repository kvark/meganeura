#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

# Figures and poster.pdf are build products of ../p3hpc tables and op-bench data.
uv run figures.py
chrome=${CHROME:-$(command -v google-chrome || command -v chromium)}
"$chrome" --headless=new --disable-gpu --no-pdf-header-footer \
    --print-to-pdf=poster.pdf "file://$PWD/poster.html" 2>/dev/null
