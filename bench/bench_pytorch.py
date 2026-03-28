#!/usr/bin/env python3
"""Benchmark SmolLM2-135M inference with PyTorch (HuggingFace transformers).

Measures per-step latency, tokens/second, time-to-first-token, and peak
memory for greedy autoregressive generation. Results are printed as JSON
so the Rust-side benchmark can parse and compare them.

Usage:
    pip install torch transformers
    python bench/bench_pytorch.py [--prompt "text"] [--max-tokens 32] [--warmup 3] [--runs 5]
"""

import argparse
import json
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch SmolLM2-135M benchmark")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", default=None, help="cuda / cpu / vulkan")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --- Resolve device ---
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    torch.set_float32_matmul_precision("high")
    print(f"device: {device}, dtype: {args.dtype}", file=sys.stderr)

    # --- Load model and tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype
    ).to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    # --- Helper: run one generation and return (elapsed_s, tokens_generated, output_ids) ---
    def run_once():
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,  # greedy
                pad_token_id=tokenizer.eos_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n_gen = out.shape[1] - prompt_len
        return elapsed, n_gen, out[0]

    # --- Helper: measure TTFT (time to first token) ---
    def measure_ttft():
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    # --- Warmup ---
    print(f"warming up ({args.warmup} runs)...", file=sys.stderr)
    for _ in range(args.warmup):
        run_once()

    # --- Reset peak memory ---
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # --- Benchmark ---
    print(f"benchmarking ({args.runs} runs, {args.max_tokens} tokens each)...", file=sys.stderr)
    latencies = []
    token_counts = []
    ttft_values = []

    for i in range(args.runs):
        elapsed, n_gen, out_ids = run_once()
        latencies.append(elapsed)
        token_counts.append(n_gen)

        ttft = measure_ttft()
        ttft_values.append(ttft)

        tps = n_gen / elapsed if elapsed > 0 else 0
        print(f"  run {i+1}: {elapsed*1000:.1f}ms, {n_gen} tokens, {tps:.1f} tok/s, ttft={ttft*1000:.1f}ms", file=sys.stderr)

    # --- Collect results ---
    avg_latency = statistics.mean(latencies)
    avg_tokens = statistics.mean(token_counts)
    tokens_per_sec = avg_tokens / avg_latency if avg_latency > 0 else 0
    latency_per_token_ms = (avg_latency / avg_tokens * 1000) if avg_tokens > 0 else 0

    result = {
        "framework": "pytorch",
        "model": args.model,
        "device": device,
        "dtype": args.dtype,
        "prompt": args.prompt,
        "prompt_tokens": prompt_len,
        "max_new_tokens": args.max_tokens,
        "runs": args.runs,
        "avg_latency_ms": avg_latency * 1000,
        "median_latency_ms": statistics.median(latencies) * 1000,
        "stdev_latency_ms": statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0,
        "tokens_per_second": tokens_per_sec,
        "latency_per_token_ms": latency_per_token_ms,
        "avg_ttft_ms": statistics.mean(ttft_values) * 1000,
        "median_ttft_ms": statistics.median(ttft_values) * 1000,
    }

    if device == "cuda":
        result["peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # --- Decode sample output for verification ---
    _, _, sample_out = run_once()
    result["sample_output"] = tokenizer.decode(sample_out, skip_special_tokens=True)

    # Print JSON to stdout for programmatic consumption
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
