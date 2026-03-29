#!/usr/bin/env python3
"""Stable Diffusion U-Net training benchmark — PyTorch reference.

Matches the architecture and data pipeline of the meganeura sd_unet_train example
for an apples-to-apples comparison.

Usage:
    python bench/sd_unet_train_pytorch.py [--small] [--device cuda|cpu]
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """ResBlock: GroupNorm → SiLU → Conv3×3 → GroupNorm → SiLU → Conv3×3 + residual."""

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
    """Tiny SD U-Net: encoder → middle → decoder with skip connections."""

    def __init__(self, in_channels=4, base_channels=32, num_levels=3, num_groups=8):
        super().__init__()
        ch_mults = [2**i for i in range(num_levels)]

        # Input conv
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False)

        # Encoder
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

        # Middle
        self.middle = ResBlock(c_in, c_in, num_groups)

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(num_levels)):
            c_out = base_channels * ch_mults[level]
            skip_c = c_out  # skip connection has this many channels
            if level < num_levels - 1:
                self.upsamples.append(nn.Upsample(scale_factor=2, mode="nearest"))
            else:
                self.upsamples.append(nn.Identity())
            self.decoder_blocks.append(ResBlock(c_in + skip_c, c_out, num_groups))
            c_in = c_out

        # Output
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.small:
        batch, in_c, res, base_ch, num_groups = 2, 4, 32, 64, 16
    else:
        batch, in_c, res, base_ch, num_groups = 4, 4, 32, 32, 8

    epochs = 3
    steps_per_epoch = 50
    lr = 1e-3
    device = torch.device(args.device)

    model = SDUNet(in_c, base_ch, num_levels=3, num_groups=num_groups).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    print("=== SD U-Net Training Benchmark (PyTorch) ===")
    print(f"config:     {'small' if args.small else 'tiny'} ({res}×{res} latent, batch {batch}, 3 levels, base_ch={base_ch})")
    print(f"parameters: {num_params} ({num_params * 4 / 1e6:.2f} MB)")
    print(f"device:     {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Warmup
    x = torch.randn(batch, in_c, res, res, device=device)
    t = torch.randn(batch, in_c, res, res, device=device)
    pred = model(x)
    loss = F.mse_loss(pred, t)
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"\ntraining ({epochs} epochs × {steps_per_epoch} steps)...")
    t_train = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            noisy = torch.randn(batch, in_c, res, res, device=device)
            noise = torch.randn(batch, in_c, res, res, device=device) * 0.5

            optimizer.zero_grad()
            pred = model(noisy)
            loss = F.mse_loss(pred, noise)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch
        print(f"  epoch {epoch + 1}: avg_loss = {avg_loss:.6f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.time() - t_train

    total_steps = epochs * steps_per_epoch
    samples_per_sec = total_steps * batch / train_time
    steps_per_sec = total_steps / train_time

    print(f"\n=== Results ===")
    print(f"train time:      {train_time:.2f}s")
    print(f"total steps:     {total_steps}")
    print(f"throughput:      {samples_per_sec:.1f} samples/s ({steps_per_sec:.1f} steps/s)")
    print(f"per-step:        {train_time * 1000 / total_steps:.2f}ms")


if __name__ == "__main__":
    main()
