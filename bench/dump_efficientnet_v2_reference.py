#!/usr/bin/env python3
"""Dump EfficientNetV2-S features[0:6] reference for meganeura parity test.

Loads torchvision `efficientnet_v2_s` with ImageNet1K_V1 weights, fuses
each BatchNorm into the preceding conv weight + a per-channel bias,
and writes:

  * `bench/results/efficientnet_v2s.safetensors` — fused parameters
    keyed exactly the way `meganeura::models::efficientnet::weight_names`
    enumerates them.

  * `bench/results/efficientnet_v2s_reference.json` — input image and
    expected output `[1, 160, 12, 12]` from torchvision's forward at
    192×192 (no normalization — the loader feeds raw pixel values
    after dividing by 255 since the kindle path skips ImageNet
    normalization for now).

The Rust integration test reads these two files and compares meganeura's
forward output against torchvision's element-wise (max-abs error).

Usage:
    python bench/dump_efficientnet_v2_reference.py \\
      [--out-dir bench/results] [--seed 0]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torchvision
from safetensors.torch import save_file


# Stage descriptor — must match meganeura::models::efficientnet `weight_names`.
# Fields: (stage, block, expand_ratio, kind)
# kind="fused" → block.0.<conv,BN>, optional block.1.<conv,BN>
# kind="mbconv" → block.0.<conv,BN>, block.1.<conv,BN>, block.2.fc1/fc2, block.3.<conv,BN>
STAGES = [
    # features.1 (2× FusedMBConv e=1)
    (1, 0, 1, "fused"), (1, 1, 1, "fused"),
    # features.2 (4× FusedMBConv e=4)
    (2, 0, 4, "fused"), (2, 1, 4, "fused"),
    (2, 2, 4, "fused"), (2, 3, 4, "fused"),
    # features.3 (4× FusedMBConv e=4)
    (3, 0, 4, "fused"), (3, 1, 4, "fused"),
    (3, 2, 4, "fused"), (3, 3, 4, "fused"),
    # features.4 (6× MBConv e=4)
    (4, 0, 4, "mbconv"), (4, 1, 4, "mbconv"),
    (4, 2, 4, "mbconv"), (4, 3, 4, "mbconv"),
    (4, 4, 4, "mbconv"), (4, 5, 4, "mbconv"),
    # features.5 (9× MBConv e=6)
    (5, 0, 6, "mbconv"), (5, 1, 6, "mbconv"),
    (5, 2, 6, "mbconv"), (5, 3, 6, "mbconv"),
    (5, 4, 6, "mbconv"), (5, 5, 6, "mbconv"),
    (5, 6, 6, "mbconv"), (5, 7, 6, "mbconv"),
    (5, 8, 6, "mbconv"),
]


def fuse_bn_into_conv(conv_w, bn_w, bn_b, bn_mean, bn_var, eps):
    """Fuse a Conv2d + BatchNorm2d pair into (conv_w_fused, bias_per_channel).

    inv_std       = bn_w / sqrt(bn_var + eps)
    conv_w_fused  = conv_w * inv_std[:, None, None, None]
    bias_per_chan = bn_b - bn_mean * inv_std
    """
    inv_std = bn_w / torch.sqrt(bn_var + eps)
    conv_w_fused = conv_w * inv_std.view(-1, 1, 1, 1)
    bias_per_chan = bn_b - bn_mean * inv_std
    return conv_w_fused, bias_per_chan


def fuse_conv_bn(sd, prefix, eps=1e-3):
    """Fuse `<prefix>.0.<conv>` + `<prefix>.1.<bn>` from `sd` and return
    (flat_conv_weight, per_channel_bias). torchvision uses eps=0.001 for
    EfficientNetV2 BN."""
    conv_w = sd[f"{prefix}.0.weight"]
    bn_w = sd[f"{prefix}.1.weight"]
    bn_b = sd[f"{prefix}.1.bias"]
    bn_mean = sd[f"{prefix}.1.running_mean"]
    bn_var = sd[f"{prefix}.1.running_var"]
    fused_w, fused_b = fuse_bn_into_conv(conv_w, bn_w, bn_b, bn_mean, bn_var, eps)
    return fused_w.contiguous().reshape(-1), fused_b.contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="bench/results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading torchvision efficientnet_v2_s + ImageNet1K_V1 weights",
          file=sys.stderr)
    weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    full = torchvision.models.efficientnet_v2_s(weights=weights)
    full.eval()
    sd = full.state_dict()

    fused = {}

    # -------- Stem (features.0) --------
    w, b = fuse_conv_bn(sd, "features.0")
    fused["features.0.0.weight"] = w
    fused["features.0.bn.fused_bias"] = b

    # -------- Per-stage blocks --------
    for stage, block, expand_ratio, kind in STAGES:
        name = f"features.{stage}.{block}"
        if kind == "fused":
            # block.0 = (expand|single) 3×3 + BN
            w0, b0 = fuse_conv_bn(sd, f"{name}.block.0")
            fused[f"{name}.block.0.0.weight"] = w0
            fused[f"{name}.block.0.bn.fused_bias"] = b0
            if expand_ratio > 1:
                # block.1 = project 1×1 + BN
                w1, b1 = fuse_conv_bn(sd, f"{name}.block.1")
                fused[f"{name}.block.1.0.weight"] = w1
                fused[f"{name}.block.1.bn.fused_bias"] = b1
        elif kind == "mbconv":
            # block.0 = expand 1×1 + BN
            w0, b0 = fuse_conv_bn(sd, f"{name}.block.0")
            fused[f"{name}.block.0.0.weight"] = w0
            fused[f"{name}.block.0.bn.fused_bias"] = b0
            # block.1 = depthwise k×k + BN
            w1, b1 = fuse_conv_bn(sd, f"{name}.block.1")
            # Depthwise weight is stored [C, 1, kH, kW]; we save as [C*kH*kW] flat.
            fused[f"{name}.block.1.0.weight"] = w1.reshape(-1)
            fused[f"{name}.block.1.bn.fused_bias"] = b1
            # block.2 = SE (fc1, fc2, both 1×1 conv with bias)
            #   torchvision stores SE convs as Conv2d shape [out, in, 1, 1].
            #   meganeura expects MatMulBT-shaped weight [out, in].
            fc1_w = sd[f"{name}.block.2.fc1.weight"].squeeze(-1).squeeze(-1)
            fc1_b = sd[f"{name}.block.2.fc1.bias"]
            fc2_w = sd[f"{name}.block.2.fc2.weight"].squeeze(-1).squeeze(-1)
            fc2_b = sd[f"{name}.block.2.fc2.bias"]
            fused[f"{name}.block.2.fc1.weight"] = fc1_w.contiguous()
            fused[f"{name}.block.2.fc1.bias"] = fc1_b.contiguous()
            fused[f"{name}.block.2.fc2.weight"] = fc2_w.contiguous()
            fused[f"{name}.block.2.fc2.bias"] = fc2_b.contiguous()
            # block.3 = project 1×1 + BN
            w3, b3 = fuse_conv_bn(sd, f"{name}.block.3")
            fused[f"{name}.block.3.0.weight"] = w3
            fused[f"{name}.block.3.bn.fused_bias"] = b3
        else:
            raise ValueError(kind)

    # Save safetensors
    safetensors_path = out_dir / "efficientnet_v2s.safetensors"
    save_file(fused, str(safetensors_path))
    n_params = sum(t.numel() for t in fused.values())
    print(f"Wrote {safetensors_path} ({len(fused)} tensors, "
          f"{n_params:,} f32 = {n_params * 4 / 1e6:.1f} MB)", file=sys.stderr)

    # -------- Reference forward --------
    # Build features[0:6] sub-network and run on a deterministic test image.
    torch.manual_seed(args.seed)
    image = torch.rand(1, 3, 192, 192)  # raw [0, 1] pixel values
    sub = torch.nn.Sequential(*list(full.features[:6])).eval()
    with torch.no_grad():
        ref = sub(image)
    assert ref.shape == (1, 160, 12, 12), f"unexpected shape {ref.shape}"
    print(f"Reference output stats: shape={tuple(ref.shape)} "
          f"min={ref.min():.4f} max={ref.max():.4f} mean={ref.mean():.4f}",
          file=sys.stderr)

    ref_path = out_dir / "efficientnet_v2s_reference.json"
    with open(ref_path, "w") as f:
        json.dump({
            "input": image.flatten().tolist(),
            "input_shape": list(image.shape),
            "output": ref.flatten().tolist(),
            "output_shape": list(ref.shape),
            "seed": args.seed,
            "torchvision_version": torchvision.__version__,
            "torch_version": torch.__version__,
        }, f)
    n_input = image.numel()
    n_output = ref.numel()
    print(f"Wrote {ref_path} (input {n_input} f32, output {n_output} f32)",
          file=sys.stderr)

    # -------- Per-stage outputs (for bisection) --------
    stages = {}
    with torch.no_grad():
        x = image
        for i in range(6):
            x = full.features[i](x)
            stages[f"stage{i}"] = {
                "shape": list(x.shape),
                "data": x.flatten().tolist(),
            }
            print(f"  stage {i}: shape={tuple(x.shape)} "
                  f"min={x.min():.4f} max={x.max():.4f}", file=sys.stderr)

    stages_path = out_dir / "efficientnet_v2s_stages_reference.json"
    with open(stages_path, "w") as f:
        json.dump(stages, f)
    print(f"Wrote {stages_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
