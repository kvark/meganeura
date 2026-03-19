#!/usr/bin/env bash
# Compare meganeura vs PyTorch on SmolVLA action expert inference.
#
# Usage:  bash bench/compare_smolvla.sh [STEPS=10] [RUNS=5]
set -euo pipefail

STEPS="${STEPS:-10}"
RUNS="${RUNS:-5}"
WARMUP="${WARMUP:-3}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
VLM_SEQ_LEN="${VLM_SEQ_LEN:-16}"
PYTORCH_DTYPE="${PYTORCH_DTYPE:-float32}"

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$DIR")"
OUT_DIR="$ROOT/bench/results"
mkdir -p "$OUT_DIR"

echo "=== SmolVLA Action Expert Benchmark ==="
echo "  chunk_size:   $CHUNK_SIZE"
echo "  vlm_seq_len:  $VLM_SEQ_LEN"
echo "  denoise_steps: $STEPS"
echo "  warmup:       $WARMUP"
echo "  runs:         $RUNS"
echo ""

# --- meganeura ---
echo ">>> meganeura (blade-graphics, f32)"
cargo build --release --example bench_smolvla_meganeura --manifest-path "$ROOT/Cargo.toml" 2>&1 | tail -1
"$ROOT/target/release/examples/bench_smolvla_meganeura" \
    --steps "$STEPS" \
    --warmup "$WARMUP" \
    --runs "$RUNS" \
    > "$OUT_DIR/smolvla_meganeura.json" 2>/dev/stderr
echo "  -> $OUT_DIR/smolvla_meganeura.json"
echo ""

# --- PyTorch ---
echo ">>> PyTorch ($PYTORCH_DTYPE)"
if python3 -c "import torch, safetensors" 2>/dev/null; then
    python3 "$DIR/bench_smolvla_pytorch.py" \
        --steps "$STEPS" \
        --warmup "$WARMUP" \
        --runs "$RUNS" \
        --dtype "$PYTORCH_DTYPE" \
        --chunk-size "$CHUNK_SIZE" \
        --vlm-seq-len "$VLM_SEQ_LEN" \
        > "$OUT_DIR/smolvla_pytorch.json" 2>/dev/stderr
    echo "  -> $OUT_DIR/smolvla_pytorch.json"
else
    echo "  SKIPPED (torch or safetensors not installed)"
    echo '{"framework":"pytorch","error":"not installed"}' > "$OUT_DIR/smolvla_pytorch.json"
fi
echo ""

# --- Summary ---
echo "=== Results ==="
python3 -c "
import json, os

dir = '$OUT_DIR'
results = []
for name in ['smolvla_meganeura', 'smolvla_pytorch']:
    path = os.path.join(dir, name + '.json')
    with open(path) as f:
        results.append(json.load(f))

header = f'{\"Metric\":<28} {\"meganeura\":>14} {\"pytorch\":>14}'
print(header)
print('-' * len(header))

def get(r, k):
    v = r.get(k)
    if v is None: return 'N/A'
    if isinstance(v, float): return f'{v:.2f}'
    return str(v)

metrics = [
    ('avg_latency_ms', 'Avg latency (ms)'),
    ('median_latency_ms', 'Median latency (ms)'),
    ('stdev_latency_ms', 'Stdev (ms)'),
    ('avg_per_step_ms', 'Avg per step (ms)'),
    ('steps_per_second', 'Steps/second'),
    ('peak_memory_mb', 'Peak GPU memory (MB)'),
]

for key, label in metrics:
    vals = [get(r, key) for r in results]
    print(f'{label:<28} {vals[0]:>14} {vals[1]:>14}')
" 2>/dev/null || echo "(install python3 for summary table)"
