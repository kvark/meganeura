#!/usr/bin/env python3
"""SD U-Net training benchmark — PyTorch reference.

Matches the meganeura bench_sd_unet_train architecture for comparison.
JSON output to stdout, human-readable to stderr (same as meganeura bench).

Usage:
    python bench/bench_sd_unet_train_pytorch.py [--small] [--warmup 2] [--runs 5] [--steps 20]
"""

import argparse
import json
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, num_groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.res_conv = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return self.res_conv(x) + h


class SDUNet(nn.Module):
    def __init__(self, in_channels=4, base_channels=32, num_levels=3, num_groups=8):
        super().__init__()
        ch_mults = [2**i for i in range(num_levels)]
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False)

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        c_in = base_channels
        for level in range(num_levels):
            c_out = base_channels * ch_mults[level]
            self.encoder_blocks.append(ResBlock(c_in, c_out, num_groups))
            if level < num_levels - 1:
                self.downsamples.append(nn.Conv2d(c_out, c_out, 3, stride=2, padding=1, bias=False))
            else:
                self.downsamples.append(nn.Identity())
            c_in = c_out

        self.middle = ResBlock(c_in, c_in, num_groups)

        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(num_levels)):
            c_out = base_channels * ch_mults[level]
            skip_c = c_out
            if level < num_levels - 1:
                self.upsamples.append(nn.Upsample(scale_factor=2, mode="nearest"))
            else:
                self.upsamples.append(nn.Identity())
            self.decoder_blocks.append(ResBlock(c_in + skip_c, c_out, num_groups))
            c_in = c_out

        self.norm_out = nn.GroupNorm(num_groups, base_channels)
        self.conv_out = nn.Conv2d(base_channels, in_channels, 3, padding=1, bias=False)
        self.num_levels = num_levels

    def forward(self, x):
        x = self.conv_in(x)
        skips = []
        for i, (block, down) in enumerate(zip(self.encoder_blocks, self.downsamples)):
            x = block(x)
            skips.append(x)
            if i < self.num_levels - 1:
                x = down(x)
        x = self.middle(x)
        for i, (up, block) in enumerate(zip(self.upsamples, self.decoder_blocks)):
            level = self.num_levels - 1 - i
            if level < self.num_levels - 1:
                x = up(x)
            x = torch.cat([x, skips[level]], dim=1)
            x = block(x)
        x = F.silu(self.norm_out(x))
        return self.conv_out(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()

    if args.small:
        batch, in_c, res, base_ch, num_groups = 2, 4, 32, 64, 16
    else:
        batch, in_c, res, base_ch, num_groups = 4, 4, 32, 32, 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 1e-3

    model = SDUNet(in_c, base_ch, num_levels=3, num_groups=num_groups).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    config_name = "small" if args.small else "tiny"

    print(f"=== SD U-Net Training Benchmark (PyTorch) ===", file=sys.stderr)
    print(f"config:     {config_name} ({res}x{res} latent, batch {batch}, base_ch={base_ch})", file=sys.stderr)
    print(f"parameters: {num_params} ({num_params * 4 / 1e6:.2f} MB)", file=sys.stderr)
    print(f"device:     {device}", file=sys.stderr)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Warmup
    print(f"warmup ({args.warmup} runs)...", file=sys.stderr)
    for _ in range(args.warmup):
        x = torch.randn(batch, in_c, res, res, device=device)
        t = torch.randn(batch, in_c, res, res, device=device) * 0.5
        optimizer.zero_grad()
        pred = model(x)
        loss = F.mse_loss(pred, t)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    print(f"benchmarking ({args.runs} runs x {args.steps} steps)...", file=sys.stderr)
    run_times = []
    for r in range(args.runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        run_loss = 0.0
        for _ in range(args.steps):
            noisy = torch.randn(batch, in_c, res, res, device=device)
            noise = torch.randn(batch, in_c, res, res, device=device) * 0.5
            optimizer.zero_grad()
            pred = model(noisy)
            loss = F.mse_loss(pred, noise)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        run_times.append(elapsed)
        avg_loss = run_loss / args.steps
        print(f"  run {r+1}: {elapsed*1000:.2f}ms total, {elapsed*1000/args.steps:.2f}ms/step, avg_loss={avg_loss:.6f}", file=sys.stderr)

    run_times.sort()
    avg_s = sum(run_times) / len(run_times)
    median_s = run_times[len(run_times) // 2]
    step_avg_ms = avg_s * 1000 / args.steps
    step_median_ms = median_s * 1000 / args.steps
    samples_per_sec = args.steps * batch / avg_s

    device_name = str(device)
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)

    result = {
        "framework": "pytorch",
        "model": f"sd_unet_{config_name}",
        "device": device_name,
        "parameters": num_params,
        "batch_size": batch,
        "resolution": res,
        "compile_time_s": 0,
        "train_step_avg_ms": round(step_avg_ms, 2),
        "train_step_median_ms": round(step_median_ms, 2),
        "samples_per_sec": round(samples_per_sec, 1),
        "memory_mb": 0,
    }

    if device.type == "cuda":
        result["memory_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
