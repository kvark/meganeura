#!/usr/bin/env bash
# Run every shape in shapes.txt (or $SHAPES, e.g. training-shapes.txt) through
# Meganeura and PyTorch on one GPU and append rows to a TSV. Meganeura runs in
# three processes per shape, since each process's kernel tuner can pick a
# different tile; the row with the median GPU time is kept. PyTorch runs once
# per shape, then once more untimed with the vendor libraries' logging on
# (cuDNN API log, MIOpen driver commands, oneDNN verbose), saved per shape
# under <out>-vendor/ because their algorithm and memory-format names are not
# all in the kernel names.
#
# Usage: run_all.sh <label> <meganeura-device-id> <python> <torch-device> [out.tsv]
#   <torch-device> is "cuda" for CUDA and ROCm builds, "xpu" for Intel.
# Example: run_all.sh 7900xt 29772 ~/venv/bin/python cuda results-7900xt.tsv
#          SHAPES=training-shapes.txt run_all.sh rtx5070 8433 ~/venv/bin/python cuda
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../../../.." && pwd)"
label=$1 id=$2 py=$3 dev=$4 out=${5:-$here/results-$1.tsv}
shapes=$here/${SHAPES:-shapes.txt}
logs=${out%.tsv}-vendor
mkdir -p "$logs"
rows='^(gemm|conv|dgrad|wgrad)\t'
cargo build --release --manifest-path "$root/Cargo.toml" --example op_bench
target_dir=$(cargo metadata --format-version 1 --no-deps --manifest-path "$root/Cargo.toml" \
  | "$py" -I -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')
bench="$target_dir/release/examples/op_bench"
for s in $(grep -v '^#' "$shapes" | awk 'NF{print $2}'); do
  runs=""
  for _ in 1 2 3; do
    r=$(MEGANEURA_DEVICE_ID=$id MEGANEURA_TUNE=1 MEGANEURA_DISABLE_COOP=1 MEGANEURA_GPU_TIMING=1 \
        timeout 180 "$bench" "$shapes" "$s" 2>/dev/null | grep -P "$rows" | tail -1) || true
    if [ -n "$r" ]; then runs+="$r"$'\n'; fi
  done
  n=$(printf '%s' "$runs" | grep -c . || true) r=""
  if [ "$n" -gt 0 ]; then
    r=$(printf '%s' "$runs" | sort -t$'\t' -k3,3g | sed -n "$(( (n + 1) / 2 ))p")
  fi
  echo -e "$label\tmeganeura\t${r:-FAILED $s}" >> "$out"
  r=$(NVIDIA_TF32_OVERRIDE=0 timeout 180 "$py" -I "$here/torch_ops.py" "$shapes" "$dev" "$s" 2>/dev/null \
      | grep -P "$rows" | tail -1) || true
  echo -e "$label\tpytorch\t${r:-FAILED $s}" >> "$out"
  NVIDIA_TF32_OVERRIDE=0 CUDNN_LOGLEVEL_DBG=3 CUDNN_LOGINFO_DBG=1 CUDNN_LOGDEST_DBG=stderr \
    MIOPEN_ENABLE_LOGGING_CMD=1 MIOPEN_LOG_LEVEL=6 ONEDNN_VERBOSE=2 \
    timeout 180 "$py" -I "$here/torch_ops.py" "$shapes" "$dev" "$s" --describe \
    > "$logs/$s.log" 2>&1 || true
done
echo "wrote $out and $logs/"
