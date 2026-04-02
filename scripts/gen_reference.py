#!/usr/bin/env python3
"""Generate reference outputs for ResNet-18 and Whisper-tiny with random weights.

Creates random weights, runs inference in PyTorch, saves weights + outputs
as npz files for comparison with meganeura.

Usage: python3 scripts/gen_reference.py
"""
import json
import sys
import torch
import torch.nn as nn
import numpy as np


def gen_resnet18():
    """ResNet-18 with random weights — compare conv+bn+relu+pool+fc pipeline."""
    print("=== ResNet-18 reference ===")
    torch.manual_seed(42)

    # Simple version: just test the stem + one block + GAP + FC
    # to verify the core pipeline without downloading 11M params

    # Build a minimal "ResNet-like" model:
    # conv7x7(3→64, stride=2) → BN → ReLU → MaxPool(3,stride=2) →
    # conv3x3(64→64) → BN → ReLU → conv3x3(64→64) → BN → residual+ReLU →
    # GAP → FC(64→10)
    class MiniResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(3, stride=2, padding=1)
            self.block_conv1 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
            self.block_bn1 = nn.BatchNorm2d(64)
            self.block_conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
            self.block_bn2 = nn.BatchNorm2d(64)
            self.fc = nn.Linear(64, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.bn1(self.conv1(x))))
            identity = x
            x = torch.relu(self.block_bn1(self.block_conv1(x)))
            x = self.block_bn2(self.block_conv2(x))
            x = torch.relu(x + identity)
            x = x.mean(dim=[2, 3])  # GAP
            return self.fc(x)

    model = MiniResNet()
    model.eval()

    # Fuse BN into conv for comparison with meganeura
    # (meganeura doesn't have runtime BN, it fuses at load time)
    def fuse_bn(conv_w, bn):
        """Returns (fused_weight, fused_bias_per_channel)"""
        scale = bn.weight.data / torch.sqrt(bn.running_var.data + 1e-5)
        w = conv_w * scale.view(-1, 1, 1, 1)
        b = bn.bias.data - bn.running_mean.data * scale
        return w, b

    # Create deterministic input
    x = torch.randn(1, 3, 64, 64)  # smaller than 224 for speed

    with torch.no_grad():
        output = model(x)

    print(f"  input shape: {x.shape}")
    print(f"  output shape: {output.shape}")
    print(f"  output[:5]: {output[0, :5].tolist()}")

    # Save state dict + input/output for meganeura comparison
    result = {
        "input": x.numpy().flatten().tolist(),
        "output": output.numpy().flatten().tolist(),
        "input_shape": list(x.shape),
        "output_shape": list(output.shape),
    }

    # Save fused weights
    w1, b1 = fuse_bn(model.conv1.weight.data, model.bn1)
    bw1, bb1 = fuse_bn(model.block_conv1.weight.data, model.block_bn1)
    bw2, bb2 = fuse_bn(model.block_conv2.weight.data, model.block_bn2)

    result["conv1_weight"] = w1.numpy().flatten().tolist()
    result["bn1_bias"] = b1.numpy().tolist()
    result["block_conv1_weight"] = bw1.numpy().flatten().tolist()
    result["block_bn1_bias"] = bb1.numpy().tolist()
    result["block_conv2_weight"] = bw2.numpy().flatten().tolist()
    result["block_bn2_bias"] = bb2.numpy().tolist()
    result["fc_weight"] = model.fc.weight.data.numpy().flatten().tolist()
    result["fc_bias"] = model.fc.bias.data.numpy().tolist()

    with open("bench/results/resnet_reference.json", "w") as f:
        json.dump(result, f)
    print(f"  saved bench/results/resnet_reference.json")
    return result


def gen_whisper_encoder():
    """Whisper-tiny encoder with random weights — test conv1d+gelu+transpose+attn+ffn."""
    print("\n=== Whisper encoder reference ===")
    torch.manual_seed(42)

    d_model = 32  # tiny for testing
    n_heads = 4
    n_layers = 1
    ffn_dim = 64
    n_mels = 16
    mel_len = 32

    class MiniWhisperEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(n_mels, d_model, 3, padding=1)
            self.conv2 = nn.Conv1d(d_model, d_model, 3, stride=2, padding=1)
            self.ln1 = nn.LayerNorm(d_model)
            self.q = nn.Linear(d_model, d_model)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model)
            self.out = nn.Linear(d_model, d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.fc1 = nn.Linear(d_model, ffn_dim)
            self.fc2 = nn.Linear(ffn_dim, d_model)
            self.final_ln = nn.LayerNorm(d_model)

        def forward(self, mel):
            # Conv stem
            x = torch.nn.functional.gelu(self.conv1(mel))
            x = torch.nn.functional.gelu(self.conv2(x))
            # [batch, d_model, seq] → [batch, seq, d_model]
            x = x.transpose(1, 2)

            # Single transformer layer (pre-norm)
            h = self.ln1(x)
            B, S, D = h.shape
            q = self.q(h).view(B, S, n_heads, D // n_heads).transpose(1, 2)
            k = self.k(h).view(B, S, n_heads, D // n_heads).transpose(1, 2)
            v = self.v(h).view(B, S, n_heads, D // n_heads).transpose(1, 2)
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).contiguous().view(B, S, D)
            h = self.out(attn)
            x = x + h

            h = self.ln2(x)
            h = torch.nn.functional.gelu(self.fc1(h))
            h = self.fc2(h)
            x = x + h

            return self.final_ln(x)

    model = MiniWhisperEncoder()
    model.eval()

    mel = torch.randn(1, n_mels, mel_len)

    with torch.no_grad():
        output = model(mel)

    seq_len = (mel_len + 2 - 3) // 2 + 1
    print(f"  mel shape: {mel.shape}")
    print(f"  output shape: {output.shape} (expected [1, {seq_len}, {d_model}])")
    print(f"  output[0,0,:5]: {output[0, 0, :5].tolist()}")

    result = {
        "input": mel.numpy().flatten().tolist(),
        "output": output.numpy().flatten().tolist(),
        "input_shape": list(mel.shape),
        "output_shape": list(output.shape),
        "d_model": d_model,
        "n_heads": n_heads,
        "n_mels": n_mels,
        "mel_len": mel_len,
        "seq_len": seq_len,
        "ffn_dim": ffn_dim,
    }

    # Save weights (transpose linears for meganeura: [out,in] → [in,out])
    sd = model.state_dict()
    for name, tensor in sd.items():
        result[name] = tensor.numpy().flatten().tolist()

    with open("bench/results/whisper_reference.json", "w") as f:
        json.dump(result, f)
    print(f"  saved bench/results/whisper_reference.json")
    return result


if __name__ == "__main__":
    import os
    os.makedirs("bench/results", exist_ok=True)
    gen_resnet18()
    gen_whisper_encoder()
    print("\nDone. Run the Rust correctness tests to compare.")
