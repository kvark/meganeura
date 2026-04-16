#!/bin/bash
# Capture a meganeura example with NVIDIA Nsight Graphics.
#
# For compute-only workloads (no swapchain), this captures the Vulkan
# command stream. The capture can be opened in NSight Graphics UI for
# shader analysis, resource inspection, and API call review.
#
# Usage:
#   bash tools/ngfx_capture.sh matmul_throughput
#   bash tools/ngfx_capture.sh bench_smolvla_train -- --profile
#
# Output: ngfx_<name>.ngfx-capture in current directory
# Open with: ngfx-ui ngfx_<name>.ngfx-capture

set -e

NGFX="/c/Program Files/NVIDIA Corporation/Nsight Graphics 2026.1.0/host/windows-desktop-nomad-x64/ngfx-capture.exe"

if [ ! -f "$NGFX" ]; then
    echo "ERROR: ngfx-capture not found at $NGFX"
    echo "Install NSight Graphics from https://developer.nvidia.com/nsight-graphics"
    exit 1
fi

EXAMPLE="$1"
shift || true

if [ -z "$EXAMPLE" ]; then
    echo "Usage: $0 <example_name> [-- <args>]"
    echo ""
    echo "Examples:"
    echo "  $0 matmul_throughput"
    echo "  $0 bench_smolvla_train"
    exit 1
fi

# Build first
echo "Building $EXAMPLE..."
cargo build --release --example "$EXAMPLE" 2>&1

EXE="target/release/examples/${EXAMPLE}.exe"
if [ ! -f "$EXE" ]; then
    echo "ERROR: $EXE not found after build"
    exit 1
fi

OUTPUT="ngfx_${EXAMPLE}.ngfx-capture"
echo "Capturing with NSight Graphics..."
echo "  Will capture after 3s countdown"
echo "  Note: expect ~5-10x slowdown during capture"
echo "  Output: $OUTPUT"
echo ""

"$NGFX" \
    --exe "$EXE" \
    --args "$@" \
    --capture-countdown-timer 3000 \
    -n 1 \
    --terminate-after-capture \
    --no-hud \
    -o "$OUTPUT"

echo ""
echo "Done. Open with NSight Graphics UI:"
echo "  ngfx-ui $OUTPUT"
