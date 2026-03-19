#!/usr/bin/env bash
# Compare meganeura vs PyTorch inference on SmolLM2-135M.
#
# Usage:  bash bench/compare.sh [--max-tokens 32] [--runs 5]
#
# Outputs JSON results for each framework, then a summary table.
set -euo pipefail

MAX_TOKENS="${MAX_TOKENS:-32}"
RUNS="${RUNS:-5}"
WARMUP="${WARMUP:-3}"
PROMPT="${PROMPT:-The meaning of life is}"
PYTORCH_DTYPE="${PYTORCH_DTYPE:-float32}"

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$DIR")"
OUT_DIR="$ROOT/bench/results"
mkdir -p "$OUT_DIR"

echo "=== SmolLM2-135M Inference Benchmark ==="
echo "  prompt:     \"$PROMPT\""
echo "  max_tokens: $MAX_TOKENS"
echo "  warmup:     $WARMUP"
echo "  runs:       $RUNS"
echo ""

# --- meganeura ---
echo ">>> meganeura (blade-graphics, f32)"
cargo build --release --example bench_meganeura --manifest-path "$ROOT/Cargo.toml" 2>&1 | tail -1
"$ROOT/target/release/examples/bench_meganeura" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --warmup "$WARMUP" \
    --runs "$RUNS" \
    > "$OUT_DIR/meganeura.json" 2>/dev/stderr
echo "  -> $OUT_DIR/meganeura.json"
echo ""

# --- PyTorch ---
echo ">>> PyTorch (transformers, $PYTORCH_DTYPE)"
if python3 -c "import torch, transformers" 2>/dev/null; then
    python3 "$DIR/bench_pytorch.py" \
        --prompt "$PROMPT" \
        --max-tokens "$MAX_TOKENS" \
        --warmup "$WARMUP" \
        --runs "$RUNS" \
        --dtype "$PYTORCH_DTYPE" \
        > "$OUT_DIR/pytorch.json" 2>/dev/stderr
    echo "  -> $OUT_DIR/pytorch.json"
else
    echo "  SKIPPED (torch or transformers not installed)"
    echo '{"framework":"pytorch","error":"not installed"}' > "$OUT_DIR/pytorch.json"
fi
echo ""

# --- Summary ---
echo "=== Results ==="
python3 -c "
import json, sys, os

dir = '$OUT_DIR'
results = []
for name in ['meganeura', 'pytorch']:
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
    ('tokens_per_second', 'Tokens/second'),
    ('latency_per_token_ms', 'Latency/token (ms)'),
    ('avg_ttft_ms', 'Avg TTFT (ms)'),
    ('median_ttft_ms', 'Median TTFT (ms)'),
    ('peak_memory_mb', 'Peak GPU memory (MB)'),
]

for key, label in metrics:
    vals = [get(r, key) for r in results]
    print(f'{label:<28} {vals[0]:>14} {vals[1]:>14}')
" 2>/dev/null || echo "(install python3 for summary table)"
