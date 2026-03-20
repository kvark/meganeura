#!/usr/bin/env bash
# Compare meganeura vs PyTorch inference.
#
# Usage:
#   bash bench/compare.sh                     # SmolVLA action expert (default)
#   bash bench/compare.sh --model smollm2     # SmolLM2-135M text generation
#   bash bench/compare.sh --model all         # both benchmarks
#
# Environment overrides:
#   RUNS=5  WARMUP=3  PYTORCH_DTYPE=float32
#   SmolLM2-specific:  MAX_TOKENS=32  PROMPT="The meaning of life is"
#   SmolVLA-specific:  STEPS=10  CHUNK_SIZE=50  VLM_SEQ_LEN=16
set -euo pipefail

MODEL="${MODEL:-smolvla}"
RUNS="${RUNS:-5}"
WARMUP="${WARMUP:-3}"
PYTORCH_DTYPE="${PYTORCH_DTYPE:-float32}"

# SmolLM2 defaults
MAX_TOKENS="${MAX_TOKENS:-32}"
PROMPT="${PROMPT:-The meaning of life is}"

# SmolVLA defaults
STEPS="${STEPS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
VLM_SEQ_LEN="${VLM_SEQ_LEN:-16}"

# Parse --model argument
for arg in "$@"; do
    case "$arg" in
        --model=*) MODEL="${arg#*=}" ;;
        --model) shift_next=1 ;;
        *)
            if [[ "${shift_next:-}" == "1" ]]; then
                MODEL="$arg"
                shift_next=
            fi
            ;;
    esac
done

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$DIR")"
OUT_DIR="$ROOT/bench/results"
mkdir -p "$OUT_DIR"

# Helper: format and print a comparison table
print_table() {
    local mega_json="$1" pytorch_json="$2"
    shift 2
    # Remaining args are "key:label" pairs
    python3 -c "
import json, os, sys

with open('$mega_json') as f:
    mega = json.load(f)
with open('$pytorch_json') as f:
    pytorch = json.load(f)

header = f'{\"Metric\":<28} {\"meganeura\":>14} {\"pytorch\":>14}'
print(header)
print('-' * len(header))

def get(r, k):
    v = r.get(k)
    if v is None: return 'N/A'
    if isinstance(v, float): return f'{v:.2f}'
    return str(v)

for pair in sys.argv[1:]:
    key, label = pair.split(':', 1)
    print(f'{label:<28} {get(mega, key):>14} {get(pytorch, key):>14}')
" "$@" 2>/dev/null || echo "(install python3 for summary table)"
}

# ============================================================
# SmolVLA action expert benchmark
# ============================================================
run_smolvla() {
    echo "=== SmolVLA Action Expert Benchmark ==="
    echo "  chunk_size:    $CHUNK_SIZE"
    echo "  vlm_seq_len:   $VLM_SEQ_LEN"
    echo "  denoise_steps: $STEPS"
    echo "  warmup:        $WARMUP"
    echo "  runs:          $RUNS"
    echo ""

    # meganeura
    echo ">>> meganeura (blade-graphics, f32)"
    cargo build --release --example bench_smolvla_meganeura --manifest-path "$ROOT/Cargo.toml" 2>&1 | tail -1
    "$ROOT/target/release/examples/bench_smolvla_meganeura" \
        --steps "$STEPS" \
        --warmup "$WARMUP" \
        --runs "$RUNS" \
        > "$OUT_DIR/smolvla_meganeura.json" 2>/dev/stderr
    echo "  -> $OUT_DIR/smolvla_meganeura.json"
    echo ""

    # PyTorch
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

    echo "=== SmolVLA Results ==="
    print_table "$OUT_DIR/smolvla_meganeura.json" "$OUT_DIR/smolvla_pytorch.json" \
        "avg_latency_ms:Avg latency (ms)" \
        "median_latency_ms:Median latency (ms)" \
        "stdev_latency_ms:Stdev (ms)" \
        "avg_per_step_ms:Avg per step (ms)" \
        "steps_per_second:Steps/second" \
        "peak_memory_mb:Peak GPU memory (MB)"
    echo ""
}

# ============================================================
# SmolLM2-135M text generation benchmark
# ============================================================
run_smollm2() {
    echo "=== SmolLM2-135M Inference Benchmark ==="
    echo "  prompt:     \"$PROMPT\""
    echo "  max_tokens: $MAX_TOKENS"
    echo "  warmup:     $WARMUP"
    echo "  runs:       $RUNS"
    echo ""

    # meganeura
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

    # PyTorch
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

    echo "=== SmolLM2 Results ==="
    print_table "$OUT_DIR/meganeura.json" "$OUT_DIR/pytorch.json" \
        "avg_latency_ms:Avg latency (ms)" \
        "median_latency_ms:Median latency (ms)" \
        "stdev_latency_ms:Stdev (ms)" \
        "tokens_per_second:Tokens/second" \
        "latency_per_token_ms:Latency/token (ms)" \
        "avg_ttft_ms:Avg TTFT (ms)" \
        "median_ttft_ms:Median TTFT (ms)" \
        "peak_memory_mb:Peak GPU memory (MB)"
    echo ""
}

# ============================================================
# Dispatch
# ============================================================
case "$MODEL" in
    smolvla)  run_smolvla ;;
    smollm2)  run_smollm2 ;;
    all)      run_smolvla; run_smollm2 ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Usage: bash bench/compare.sh [--model smolvla|smollm2|all]"
        exit 1
        ;;
esac
