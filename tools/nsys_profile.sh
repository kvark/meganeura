#!/bin/bash
# Profile a meganeura example with NVIDIA Nsight Systems.
#
# Requires: NSight Systems installed.
# On Windows: run from Admin PowerShell for Vulkan/WDDM GPU tracing.
#   Prefer tools/nsys_profile.ps1 which handles layer registration.
# On Linux:   run with sudo or set /proc/sys/kernel/perf_event_paranoid.
#
# Usage:
#   bash tools/nsys_profile.sh matmul_throughput
#   bash tools/nsys_profile.sh bench_smolvla_train -- --profile
#   bash tools/nsys_profile.sh smollm2_train_bench
#
# Output: nsys_<name>.nsys-rep in current directory
# Open with: nsys-ui nsys_<name>.nsys-rep

set -e

NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.2.1/target-windows-x64/nsys.exe"

if [ ! -f "$NSYS" ]; then
    echo "ERROR: nsys not found at $NSYS"
    echo "Install NSight Systems from https://developer.nvidia.com/nsight-systems"
    exit 1
fi

EXAMPLE="$1"
shift || true

if [ -z "$EXAMPLE" ]; then
    echo "Usage: $0 <example_name> [-- <args>]"
    echo ""
    echo "Examples:"
    echo "  $0 matmul_throughput"
    echo "  $0 bench_smolvla_train -- --profile"
    echo "  $0 profile_smollm2_decode"
    exit 1
fi

# Build first (so compile time isn't in the trace)
echo "Building $EXAMPLE..."
cargo build --release --example "$EXAMPLE" 2>&1

EXE="target/release/examples/${EXAMPLE}.exe"
if [ ! -f "$EXE" ]; then
    echo "ERROR: $EXE not found after build"
    exit 1
fi

OUTPUT="nsys_${EXAMPLE}"
echo "Profiling with NSight Systems..."
echo "  trace: vulkan API + GPU workloads"
echo "  output: ${OUTPUT}.nsys-rep"
echo ""

"$NSYS" profile \
    --trace=vulkan \
    --vulkan-gpu-workload=individual \
    --force-overwrite=true \
    --output="$OUTPUT" \
    "$EXE" "$@"

echo ""
echo "Done. Open with:"
echo "  nsys-ui ${OUTPUT}.nsys-rep"
echo ""
echo "Quick stats:"
"$NSYS" stats --report vulkan_gpu_workload "${OUTPUT}.nsys-rep" 2>/dev/null || true
