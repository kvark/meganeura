#!/usr/bin/env bash
# Create/update the Python venv used by benchmarks.
#
# Usage:
#   bash bench/ensure_venv.sh          # create .venv if needed, install deps
#   source bench/ensure_venv.sh        # same, but also exports PYTHON
#
# After running, PYTHON is set to the venv interpreter path.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$DIR")"
VENV_DIR="$ROOT/.venv"

# Cross-platform python binary inside the venv
if [[ -f "$VENV_DIR/Scripts/python.exe" ]]; then
    PYTHON="$VENV_DIR/Scripts/python"
elif [[ -f "$VENV_DIR/bin/python" ]]; then
    PYTHON="$VENV_DIR/bin/python"
else
    PYTHON=""
fi

# Find a system python3 to bootstrap the venv
find_system_python() {
    for cmd in python3 python; do
        if command -v "$cmd" >/dev/null 2>&1; then
            echo "$cmd"
            return
        fi
    done
    return 1
}

is_nixos() {
    [[ -f /etc/NIXOS ]] || grep -qi nixos /etc/os-release 2>/dev/null
}

if is_nixos; then
    echo "WARNING: NixOS detected. venv/pip may install incompatible binaries."
    echo "  Consider using --no-venv with compare.sh instead."
    echo ""
fi

# If the venv already has working torch, nothing to do
if [[ -n "$PYTHON" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        if "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            echo "venv OK (torch with CUDA)"
            export PYTHON
            return 0 2>/dev/null || exit 0
        fi
    elif "$PYTHON" -c "import torch" 2>/dev/null; then
        echo "venv OK (torch)"
        export PYTHON
        return 0 2>/dev/null || exit 0
    fi
fi

echo "--- Setting up Python venv for benchmarks ---"
sys_python="$(find_system_python)" || {
    echo "ERROR: python3 not found — cannot create venv"
    return 1 2>/dev/null || exit 1
}

# Create venv if it doesn't exist
if [[ ! -d "$VENV_DIR" ]]; then
    echo "  Creating venv at $VENV_DIR ..."
    "$sys_python" -m venv "$VENV_DIR"
fi

# Resolve the venv python again after creation
if [[ -f "$VENV_DIR/Scripts/python.exe" ]]; then
    PYTHON="$VENV_DIR/Scripts/python"
else
    PYTHON="$VENV_DIR/bin/python"
fi

# Install PyTorch — pick CUDA build when nvidia-smi is available.
echo "  Installing PyTorch ..."
if command -v nvidia-smi >/dev/null 2>&1; then
    "$PYTHON" -m pip install --quiet --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
else
    "$PYTHON" -m pip install --quiet torch
fi

# Install remaining bench dependencies
echo "  Installing bench dependencies ..."
"$PYTHON" -m pip install --quiet -r "$DIR/requirements.txt"

echo "  Done."
echo ""

export PYTHON
