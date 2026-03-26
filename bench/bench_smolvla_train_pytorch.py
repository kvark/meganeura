#!/usr/bin/env python3
"""Benchmark SmolVLA action expert training with PyTorch.

Implements the exact same graph as meganeura's build_action_expert_training():
  - Single-head attention (head_dim-sized Q/K/V/O projections, no GQA reshape)
  - No causal mask (plain softmax attention)
  - RMSNorm, SwiGLU FFN
  - MSE loss against target actions
  - SGD optimizer (lr=1e-5, matching meganeura's sgd_step_cpu)

This is NOT the full GQA SmolVLA model — it is an apples-to-apples comparison
with meganeura's training graph, which does not yet support multi-head reshape.

Usage:
    python bench/bench_smolvla_train_pytorch.py [--warmup 3] [--runs 5]
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


class SingleHeadAttention(nn.Module):
    """Single-head attention matching meganeura's primitive_self/cross_attention.

    Projects Q, K, V to head_dim (not num_heads * head_dim).
    No causal mask; plain softmax.
    """

    def __init__(self, hidden, head_dim, kv_dim, is_cross):
        super().__init__()
        self.head_dim = head_dim
        self.is_cross = is_cross
        self.q_proj = nn.Linear(hidden, head_dim, bias=False)
        kv_in = kv_dim if is_cross else hidden
        self.k_proj = nn.Linear(kv_in, head_dim, bias=False)
        self.v_proj = nn.Linear(kv_in, head_dim, bias=False)
        self.o_proj = nn.Linear(head_dim, hidden, bias=False)

    def forward(self, x, kv_input=None):
        # x: [B, seq, hidden], kv_input: [B, kv_seq, kv_dim] or None
        src = kv_input if self.is_cross and kv_input is not None else x
        q = self.q_proj(x)    # [B, seq, head_dim]
        k = self.k_proj(src)  # [B, kv_seq, head_dim]
        v = self.v_proj(src)  # [B, kv_seq, head_dim]
        # Plain softmax attention, no causal mask — matches meganeura graph
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.bmm(q, k.transpose(1, 2)) * scale  # [B, seq, kv_seq]
        probs = F.softmax(scores, dim=-1)
        out = torch.bmm(probs, v)                          # [B, seq, head_dim]
        return self.o_proj(out)


class ExpertLayer(nn.Module):
    def __init__(self, hidden, head_dim, kv_dim, intermediate, eps, is_cross):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden, eps)
        self.self_attn = SingleHeadAttention(hidden, head_dim, kv_dim, is_cross)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.mlp = SwiGLU(hidden, intermediate)

    def forward(self, x, kv_input=None):
        h = self.input_layernorm(x)
        x = x + self.self_attn(h, kv_input)
        h = self.post_attention_layernorm(x)
        x = x + self.mlp(h)
        return x


class ActionExpert(nn.Module):
    """Matches meganeura's build_action_expert_training() graph exactly."""

    def __init__(self, expert_hidden=720, num_layers=16,
                 head_dim=64, kv_dim=320,
                 intermediate=2048, action_dim=32, eps=1e-5,
                 self_attn_every_n=2):
        super().__init__()
        self.action_in_proj = nn.Linear(action_dim, expert_hidden, bias=True)
        self.action_time_mlp_in = nn.Linear(expert_hidden * 2, expert_hidden, bias=True)
        self.action_time_mlp_out = nn.Linear(expert_hidden, expert_hidden, bias=True)
        self.layers = nn.ModuleList([
            ExpertLayer(
                expert_hidden, head_dim, kv_dim, intermediate, eps,
                is_cross=(i % self_attn_every_n != 0),
            )
            for i in range(num_layers)
        ])
        self.action_out_proj = nn.Linear(expert_hidden, action_dim, bias=True)
        self.self_attn_every_n = self_attn_every_n

    def forward(self, noisy_actions, timestep, vlm_kv):
        # noisy_actions: [B, chunk, action_dim]
        # timestep:      [B, 1, expert_hidden*2]
        # vlm_kv:        [B, vlm_seq, kv_dim]
        x = self.action_in_proj(noisy_actions)
        t = F.silu(self.action_time_mlp_in(timestep))
        t = self.action_time_mlp_out(t)
        x = x + t  # broadcast add [B, 1, hidden] → [B, chunk, hidden]
        for i, layer in enumerate(self.layers):
            kv = vlm_kv if (i % self.self_attn_every_n != 0) else None
            x = layer(x, kv)
        return self.action_out_proj(x)


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--vlm-seq-len", type=int, default=16)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    print(f"device: {device}, dtype: {args.dtype}", file=sys.stderr)
    if device == "cpu":
        print("WARNING: CPU — not comparable to GPU meganeura", file=sys.stderr)

    chunk_size = args.chunk_size
    vlm_seq_len = args.vlm_seq_len
    expert_hidden = 720
    num_layers = 16
    head_dim = 64
    num_kv_heads = 5
    kv_dim = num_kv_heads * head_dim  # 320
    action_dim = 32
    intermediate = 2048

    print(
        f"SmolVLA single-head training: chunk={chunk_size}, vlm_seq={vlm_seq_len}, "
        f"layers={num_layers}, hidden={expert_hidden}, head_dim={head_dim}",
        file=sys.stderr,
    )
    print(
        "(single-head projection matching meganeura build_action_expert_training)",
        file=sys.stderr,
    )

    expert = ActionExpert(
        expert_hidden=expert_hidden,
        num_layers=num_layers,
        head_dim=head_dim,
        kv_dim=kv_dim,
        intermediate=intermediate,
        action_dim=action_dim,
        eps=1e-5,
        self_attn_every_n=2,
    ).to(device=device, dtype=dtype)

    # Deterministic init matching meganeura's bench (sin pattern × 0.1)
    with torch.no_grad():
        for i, p in enumerate(expert.parameters()):
            p.copy_(
                torch.sin(torch.arange(p.numel(), dtype=dtype) * 0.01 + i).view_as(p) * 0.1
            )

    optimizer = torch.optim.SGD(expert.parameters(), lr=1e-5)

    noisy_actions = torch.zeros(1, chunk_size, action_dim, device=device, dtype=dtype)
    timestep = torch.zeros(1, 1, expert_hidden * 2, device=device, dtype=dtype)
    vlm_kv = torch.zeros(1, vlm_seq_len, kv_dim, device=device, dtype=dtype)
    target = torch.zeros(1, chunk_size, action_dim, device=device, dtype=dtype)

    def fwd_only():
        sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            expert(noisy_actions, timestep, vlm_kv)
        sync(device)
        return time.perf_counter() - t0

    def train_step():
        sync(device)
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = expert(noisy_actions, timestep, vlm_kv)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        sync(device)
        return time.perf_counter() - t0

    print(f"warming up ({args.warmup} runs)...", file=sys.stderr)
    for _ in range(args.warmup):
        fwd_only()
        train_step()

    print(f"benchmarking forward ({args.runs} runs)...", file=sys.stderr)
    fwd_latencies = []
    for i in range(args.runs):
        t = fwd_only()
        fwd_latencies.append(t)
        print(f"  fwd run {i+1}: {t*1000:.2f}ms", file=sys.stderr)

    print(f"benchmarking training step ({args.runs} runs)...", file=sys.stderr)
    train_latencies = []
    for i in range(args.runs):
        t = train_step()
        train_latencies.append(t)
        print(f"  train run {i+1}: {t*1000:.2f}ms", file=sys.stderr)

    fwd_avg = statistics.mean(fwd_latencies)
    fwd_median = statistics.median(fwd_latencies)
    train_avg = statistics.mean(train_latencies)
    train_median = statistics.median(train_latencies)

    result = {
        "framework": "pytorch",
        "model": "smolvla_action_expert_single_head",
        "device": device,
        "dtype": args.dtype,
        "chunk_size": chunk_size,
        "vlm_seq_len": vlm_seq_len,
        "num_layers": num_layers,
        "fwd_avg_ms": round(fwd_avg * 1000, 2),
        "fwd_median_ms": round(fwd_median * 1000, 2),
        "train_step_avg_ms": round(train_avg * 1000, 2),
        "train_step_median_ms": round(train_median * 1000, 2),
        "approx_bwd_ms": round((train_avg - fwd_avg) * 1000, 2),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
