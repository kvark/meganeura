#!/usr/bin/env python3
"""Benchmark SmolVLA action expert inference with PyTorch.

Measures per-step latency for the action expert denoising loop using
the lerobot/smolvla_base checkpoint. Uses synthetic inputs to match
the meganeura benchmark.

Usage:
    pip install torch transformers
    python bench/bench_smolvla_pytorch.py [--steps 10] [--runs 5]
"""

import argparse
import json
import math
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertAttention(nn.Module):
    def __init__(self, hidden, num_heads, num_kv_heads, head_dim, is_cross):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.is_cross = is_cross
        attn_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        self.q_proj = nn.Linear(hidden, attn_dim, bias=False)
        if is_cross:
            self.k_proj = nn.Linear(kv_dim, kv_dim, bias=False)
            self.v_proj = nn.Linear(kv_dim, kv_dim, bias=False)
        else:
            self.k_proj = nn.Linear(hidden, kv_dim, bias=False)
            self.v_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.o_proj = nn.Linear(attn_dim, hidden, bias=False)

    def forward(self, x, kv_input=None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        src = kv_input if self.is_cross and kv_input is not None else x
        kv_seq = src.shape[1]
        k = self.k_proj(src).view(B, kv_seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(src).view(B, kv_seq, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA repeat
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=not self.is_cross)
        attn = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn)


class ExpertLayer(nn.Module):
    def __init__(self, hidden, num_heads, num_kv_heads, head_dim, intermediate, eps, is_cross):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden, eps)
        self.self_attn = ExpertAttention(hidden, num_heads, num_kv_heads, head_dim, is_cross)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.mlp = SwiGLU(hidden, intermediate)

    def forward(self, x, kv_input=None):
        h = self.input_layernorm(x)
        x = x + self.self_attn(h, kv_input)
        h = self.post_attention_layernorm(x)
        x = x + self.mlp(h)
        return x


class ActionExpert(nn.Module):
    """SmolVLA action expert — matches meganeura's build_action_expert."""

    def __init__(self, expert_hidden=720, text_hidden=960, num_layers=16,
                 num_heads=15, num_kv_heads=5, head_dim=64,
                 intermediate=2048, action_dim=32, eps=1e-5,
                 self_attn_every_n=2):
        super().__init__()
        kv_dim = num_kv_heads * head_dim

        self.action_in_proj = nn.Linear(action_dim, expert_hidden)
        self.action_out_proj = nn.Linear(expert_hidden, action_dim)

        self.action_time_mlp_in = nn.Linear(expert_hidden * 2, expert_hidden)
        self.action_time_mlp_out = nn.Linear(expert_hidden, expert_hidden)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            is_cross = (i % self_attn_every_n != 0)
            self.layers.append(ExpertLayer(
                expert_hidden, num_heads, num_kv_heads, head_dim,
                intermediate, eps, is_cross,
            ))

        self.self_attn_every_n = self_attn_every_n
        self.kv_dim = kv_dim

    def forward(self, noisy_actions, timestep, vlm_kv):
        x = self.action_in_proj(noisy_actions)

        # Timestep conditioning (computed but broadcast-added is omitted for parity)
        t = self.action_time_mlp_in(timestep)
        t = F.silu(t)
        _t = self.action_time_mlp_out(t)

        for i, layer in enumerate(self.layers):
            is_cross = (i % self.self_attn_every_n != 0)
            kv = vlm_kv if is_cross else None
            x = layer(x, kv)

        return self.action_out_proj(x)


def load_expert_weights(expert, state_dict):
    """Map lerobot/smolvla_base checkpoint keys to our ActionExpert."""
    prefix = "model.vlm_with_expert.lm_expert.layers."
    mapping = {}

    # Top-level projections
    for name in ["action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"]:
        for suffix in ["weight", "bias"]:
            key = f"model.{name}.{suffix}"
            if key in state_dict:
                mapping[f"{name}.{suffix}"] = state_dict[key]

    # Layers
    for i in range(len(expert.layers)):
        src = f"{prefix}{i}"
        dst = f"layers.{i}"
        for part in [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]:
            key = f"{src}.{part}"
            if key in state_dict:
                mapping[f"{dst}.{part}"] = state_dict[key]

    expert.load_state_dict(mapping, strict=False)
    return expert


def main():
    parser = argparse.ArgumentParser(description="PyTorch SmolVLA action expert benchmark")
    parser.add_argument("--steps", type=int, default=10, help="denoising steps per run")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--vlm-seq-len", type=int, default=16)
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    print(f"device: {device}, dtype: {args.dtype}", file=sys.stderr)

    # --- Load checkpoint ---
    print("downloading model...", file=sys.stderr)
    from huggingface_hub import hf_hub_download
    import safetensors.torch

    # Download safetensors
    try:
        path = hf_hub_download("lerobot/smolvla_base", "model.safetensors")
        state_dict = safetensors.torch.load_file(path)
    except Exception:
        # May be sharded
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download("lerobot/smolvla_base")
        import glob, os
        st_files = sorted(glob.glob(os.path.join(model_dir, "model*.safetensors")))
        state_dict = {}
        for f in st_files:
            state_dict.update(safetensors.torch.load_file(f))

    # --- Build expert model ---
    print("building action expert...", file=sys.stderr)
    expert = ActionExpert(
        expert_hidden=720,
        text_hidden=960,
        num_layers=16,
        num_heads=15,
        num_kv_heads=5,
        head_dim=64,
        intermediate=2048,
        action_dim=32,
        eps=1e-5,
        self_attn_every_n=2,
    )

    load_expert_weights(expert, state_dict)
    expert = expert.to(device=device, dtype=torch_dtype)
    expert.eval()

    # --- Synthetic inputs ---
    kv_dim = 5 * 64  # num_kv_heads * head_dim
    noisy_actions = torch.randn(1, args.chunk_size, 32, device=device, dtype=torch_dtype) * 0.1
    timestep = torch.randn(1, 1, 720 * 2, device=device, dtype=torch_dtype) * 0.1
    vlm_kv = torch.randn(1, args.vlm_seq_len, kv_dim, device=device, dtype=torch_dtype) * 0.05

    def run_denoise():
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(args.steps):
                expert(noisy_actions, timestep, vlm_kv)
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    # --- Warmup ---
    print(f"warming up ({args.warmup} runs)...", file=sys.stderr)
    for _ in range(args.warmup):
        run_denoise()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # --- Benchmark ---
    print(f"benchmarking ({args.runs} runs, {args.steps} steps each)...", file=sys.stderr)
    latencies = []

    for i in range(args.runs):
        elapsed = run_denoise()
        latencies.append(elapsed)
        per_step = elapsed / args.steps
        print(f"  run {i+1}: {elapsed*1000:.1f}ms total, {per_step*1000:.2f}ms/step", file=sys.stderr)

    avg = statistics.mean(latencies)
    result = {
        "framework": "pytorch",
        "model": "lerobot/smolvla_base",
        "device": device,
        "dtype": args.dtype,
        "chunk_size": args.chunk_size,
        "vlm_seq_len": args.vlm_seq_len,
        "denoise_steps": args.steps,
        "runs": args.runs,
        "avg_latency_ms": avg * 1000,
        "median_latency_ms": statistics.median(latencies) * 1000,
        "stdev_latency_ms": statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0,
        "avg_per_step_ms": avg / args.steps * 1000,
        "steps_per_second": args.steps / avg,
    }

    if device == "cuda":
        result["peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
