#!/usr/bin/env bash
# Compare meganeura vs PyTorch inference.
#
# Usage:
#   bash bench/compare.sh                     # SmolVLA action expert (default)
#   bash bench/compare.sh --model smollm2     # SmolLM2-135M text generation
#   bash bench/compare.sh --model all         # both benchmarks
#
# Environment overrides:
#   RUNS=5  WARMUP=3  PYTORCH_DTYPE=float32  --no-venv
#   SmolLM2-specific:  MAX_TOKENS=32  PROMPT="The meaning of life is"
#   SmolVLA-specific:  STEPS=10  CHUNK_SIZE=50  VLM_SEQ_LEN=16
set -euo pipefail

MODEL="${MODEL:-smolvla_train}"
TRAIN_LR="${TRAIN_LR:-0.00001}"
RUNS="${RUNS:-5}"
WARMUP="${WARMUP:-3}"
PYTORCH_DTYPE="${PYTORCH_DTYPE:-float32}"
FORCE=""
NO_VENV=""

# SmolLM2 defaults
MAX_TOKENS="${MAX_TOKENS:-32}"
PROMPT="${PROMPT:-The meaning of life is}"

# SmolVLA defaults
STEPS="${STEPS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
VLM_SEQ_LEN="${VLM_SEQ_LEN:-16}"

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --model=*) MODEL="${arg#*=}" ;;
        --model) shift_next=model ;;
        --force) FORCE="--force" ;;
        --no-venv) NO_VENV=1 ;;
        *)
            if [[ "${shift_next:-}" == "model" ]]; then
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

# ============================================================
# Python venv setup — create .venv if missing, install deps,
# and always use its python for benchmarks.
# ============================================================
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

ensure_venv() {
    if [[ -n "$PYTHON" ]]; then
        # If we have an NVIDIA GPU, also verify torch has CUDA support
        if command -v nvidia-smi >/dev/null 2>&1; then
            if "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
                return 0
            fi
        elif "$PYTHON" -c "import torch" 2>/dev/null; then
            return 0
        fi
    fi

    echo "--- Setting up Python venv for benchmarks ---"
    local sys_python
    sys_python="$(find_system_python)" || {
        echo "ERROR: python3 not found — cannot create venv"
        return 1
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
    # --force-reinstall is needed when switching from a CPU-only build.
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
}

is_nixos() {
    [[ -f /etc/NIXOS ]] || grep -qi nixos /etc/os-release 2>/dev/null
}

if [[ -n "$NO_VENV" ]]; then
    # Use system python directly (e.g. on NixOS where venvs are problematic)
    if [[ -z "$PYTHON" ]]; then
        PYTHON="$(find_system_python)" || { echo "ERROR: python3 not found"; exit 1; }
    fi
else
    if is_nixos; then
        echo "WARNING: NixOS detected. venv/pip may install incompatible binaries."
        echo "  Consider: bash bench/compare.sh --no-venv"
        echo ""
    fi
    ensure_venv
fi

# Enable experimental Flash Efficient attention on AMD GPUs (ROCm/aotriton).
# Harmless on non-AMD systems; avoids detection issues (e.g. rocminfo not on PATH in nix-shell).
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Helper: format and print a comparison table
win_path() {
    # Convert MSYS2 paths (/c/foo) to Windows paths (C:/foo) for Python
    if [[ "${OSTYPE:-}" == msys* || "${OSTYPE:-}" == mingw* ]]; then
        case "$1" in
            /[a-zA-Z]/*) echo "$(echo "$1" | sed 's|^/\(.\)|\U\1:|')"; return ;;
        esac
    fi
    echo "$1"
}

print_table() {
    local mega_json pytorch_json
    mega_json="$(win_path "$1")"
    pytorch_json="$(win_path "$2")"
    shift 2
    # Remaining args are "key:label" pairs
    "$PYTHON" -c "
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
" "$@" || echo "(summary table failed — is python3 working?)"
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
        --force \
        > "$OUT_DIR/smolvla_meganeura.json" 2>/dev/stderr
    echo "  -> $OUT_DIR/smolvla_meganeura.json"
    echo ""

    # PyTorch
    echo ">>> PyTorch ($PYTORCH_DTYPE)"
    if "$PYTHON" -c "import torch, safetensors" 2>/dev/null; then
        "$PYTHON" "$DIR/bench_smolvla_pytorch.py" \
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
        "device:Device" \
        "avg_latency_ms:Avg latency (ms)" \
        "median_latency_ms:Median latency (ms)" \
        "stdev_latency_ms:Stdev (ms)" \
        "avg_per_step_ms:Avg per step (ms)" \
        "steps_per_second:Steps/second" \
        "peak_memory_mb:Peak GPU memory (MB)"
    echo ""

    # Output correctness comparison
    local mega_out="$OUT_DIR/smolvla_meganeura_output.json"
    local pytorch_out="$OUT_DIR/smolvla_pytorch_output.json"
    if [[ -f "$mega_out" && -f "$pytorch_out" ]]; then
        local mega_out_py pytorch_out_py
        mega_out_py="$(win_path "$mega_out")"
        pytorch_out_py="$(win_path "$pytorch_out")"
        echo "=== Output Comparison ==="
        "$PYTHON" -c "
import json, math, sys

with open('$mega_out_py') as f:
    mega = json.load(f)
with open('$pytorch_out_py') as f:
    pytorch = json.load(f)

if len(mega) != len(pytorch):
    print(f'ERROR: output length mismatch: meganeura={len(mega)}, pytorch={len(pytorch)}')
    sys.exit(1)

n = len(mega)
max_err = 0.0
sum_sq = 0.0
sum_abs = 0.0
for a, b in zip(mega, pytorch):
    d = abs(a - b)
    max_err = max(max_err, d)
    sum_sq += d * d
    sum_abs += d

l2 = math.sqrt(sum_sq)
mae = sum_abs / n
rmse = math.sqrt(sum_sq / n)

# Also compute relative error vs output magnitude
mag = math.sqrt(sum(v*v for v in pytorch) / n)

print(f'  output length:    {n}')
print(f'  max abs error:    {max_err:.6e}')
print(f'  mean abs error:   {mae:.6e}')
print(f'  RMSE:             {rmse:.6e}')
print(f'  L2 norm of diff:  {l2:.6e}')
print(f'  RMS output mag:   {mag:.6e}')
if mag > 0:
    print(f'  relative RMSE:    {rmse/mag:.6e}')

if max_err < 1e-3:
    print('  PASS: outputs match within 1e-3')
elif max_err < 1e-1:
    print('  WARN: outputs differ (max error > 1e-3, likely floating-point divergence)')
else:
    print('  FAIL: outputs diverge significantly')
" || echo "(output comparison failed — is python3 working?)"
        echo ""
    fi
}

# ============================================================
# SmolVLA action expert training benchmark (random weights)
# ============================================================
run_smolvla_train() {
    echo "=== SmolVLA Action Expert Training Benchmark ==="
    echo "  chunk_size:  $CHUNK_SIZE"
    echo "  vlm_seq_len: $VLM_SEQ_LEN"
    echo "  warmup:      $WARMUP"
    echo "  runs:        $RUNS"
    echo ""

    echo ">>> meganeura training (blade-graphics, f32, random weights)"
    cargo build --release --example bench_smolvla_train --manifest-path "$ROOT/Cargo.toml" 2>&1 | tail -1
    "$ROOT/target/release/examples/bench_smolvla_train" \
        --warmup "$WARMUP" \
        --runs "$RUNS" \
        --force \
        > "$OUT_DIR/smolvla_train_meganeura.json" 2>/dev/stderr
    echo "  -> $OUT_DIR/smolvla_train_meganeura.json"
    echo ""

    echo ">>> PyTorch training ($PYTORCH_DTYPE, random weights)"
    if "$PYTHON" -c "import torch" 2>/dev/null; then
        "$PYTHON" "$DIR/bench_smolvla_train_pytorch.py" \
            --warmup "$WARMUP" \
            --runs "$RUNS" \
            --dtype "$PYTORCH_DTYPE" \
            --chunk-size "$CHUNK_SIZE" \
            --vlm-seq-len "$VLM_SEQ_LEN" \
            > "$OUT_DIR/smolvla_train_pytorch.json" 2>/dev/stderr
        echo "  -> $OUT_DIR/smolvla_train_pytorch.json"
    else
        echo "  SKIPPED (torch not installed)"
        echo '{"framework":"pytorch","error":"not installed"}' > "$OUT_DIR/smolvla_train_pytorch.json"
    fi
    echo ""

    echo "=== SmolVLA Training Results ==="
    print_table "$OUT_DIR/smolvla_train_meganeura.json" "$OUT_DIR/smolvla_train_pytorch.json" \
        "device:Device" \
        "compile_time_s:Compile time (s)" \
        "fwd_avg_ms:Fwd avg (ms)" \
        "fwd_median_ms:Fwd median (ms)" \
        "train_step_avg_ms:Train step avg (ms)" \
        "train_step_median_ms:Train step median (ms)" \
        "approx_bwd_ms:Approx bwd (ms)"
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
    if "$PYTHON" -c "import torch, transformers" 2>/dev/null; then
        "$PYTHON" "$DIR/bench_pytorch.py" \
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
    smolvla)        run_smolvla ;;
    smolvla_train)  run_smolvla_train ;;
    smollm2)        run_smollm2 ;;
    all)            run_smolvla; run_smolvla_train; run_smollm2 ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Usage: bash bench/compare.sh [--model smolvla|smolvla_train|smollm2|all]"
        exit 1
        ;;
esac
