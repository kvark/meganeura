#!/usr/bin/env bash
# Run every shape in shapes.txt through Meganeura and PyTorch on one GPU,
# one process per shape and engine, and append rows to a TSV.
#
# Usage: run_all.sh <label> <meganeura-device-id> <python> <torch-device> [out.tsv]
#   <torch-device> is "cuda" for CUDA and ROCm builds, "xpu" for Intel.
# Example: run_all.sh 7900xt 29772 ~/venv/bin/python cuda results-7900xt.tsv
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../../../.." && pwd)"
label=$1 id=$2 py=$3 dev=$4 out=${5:-$here/results-$1.tsv}
cargo build --release --manifest-path "$root/Cargo.toml" --example op_bench
target_dir=$(cargo metadata --format-version 1 --no-deps --manifest-path "$root/Cargo.toml" \
  | "$py" -I -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')
bench="$target_dir/release/examples/op_bench"
for s in $(grep -v '^#' "$here/shapes.txt" | awk 'NF{print $2}'); do
  r=$(MEGANEURA_DEVICE_ID=$id MEGANEURA_TUNE=1 MEGANEURA_DISABLE_COOP=1 MEGANEURA_GPU_TIMING=1 \
      timeout 180 "$bench" "$here/shapes.txt" "$s" 2>/dev/null | grep -P '^(gemm|conv)\t' | tail -1) || true
  echo -e "$label\tmeganeura\t${r:-FAILED $s}" >> "$out"
  r=$(NVIDIA_TF32_OVERRIDE=0 timeout 180 "$py" -I "$here/torch_ops.py" "$here/shapes.txt" "$dev" "$s" 2>/dev/null \
      | grep -P '^(gemm|conv)\t' | tail -1) || true
  echo -e "$label\tpytorch\t${r:-FAILED $s}" >> "$out"
done
echo "wrote $out"
