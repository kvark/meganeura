#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

# Regenerate from the source data; SVGs, HTML, PDF and notes are build products.
if [[ "${1:-}" == "--figures" ]]; then
    shift
fi
if [[ $# != 0 ]]; then
    printf 'Usage: bash build.sh [--figures]\n' >&2
    exit 2
fi

uv run figures.py
marp=(npx --yes @marp-team/marp-cli@4.5.1 slides.md --theme theme.css --html)
"${marp[@]}" -o slides.html
"${marp[@]}" --pdf --allow-local-files --pdf-outlines -o slides.pdf
"${marp[@]}" --notes -o slides-notes.txt
