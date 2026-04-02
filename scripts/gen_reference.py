#!/usr/bin/env python3
"""Generate reference outputs for ResNet-18 and Whisper conv stem.

Creates random weights, runs inference in PyTorch, saves weights + outputs
as JSON for comparison with meganeura.

Usage: python3 scripts/gen_reference.py
"""
import json
import os

import torch
import torch.nn as nn


def gen_resnet18():
    """ResNet-18 mini-model: conv+BN+ReLU+MaxPool+block+GAP+FC."""
    print("=== ResNet-18 reference ===")
    torch.manual_seed(42)

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

    def fuse_bn(conv_w, bn):
        scale = bn.weight.data / torch.sqrt(bn.running_var.data + 1e-5)
        w = conv_w * scale.view(-1, 1, 1, 1)
        b = bn.bias.data - bn.running_mean.data * scale
        return w, b

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        output = model(x)

    print(f"  input: {x.shape}, output: {output.shape}")
    print(f"  output[:5]: {output[0, :5].tolist()}")

    w1, b1 = fuse_bn(model.conv1.weight.data, model.bn1)
    bw1, bb1 = fuse_bn(model.block_conv1.weight.data, model.block_bn1)
    bw2, bb2 = fuse_bn(model.block_conv2.weight.data, model.block_bn2)

    result = {
        "input": x.numpy().flatten().tolist(),
        "output": output.numpy().flatten().tolist(),
        "conv1_weight": w1.numpy().flatten().tolist(),
        "bn1_bias": b1.numpy().tolist(),
        "block_conv1_weight": bw1.numpy().flatten().tolist(),
        "block_bn1_bias": bb1.numpy().tolist(),
        "block_conv2_weight": bw2.numpy().flatten().tolist(),
        "block_bn2_bias": bb2.numpy().tolist(),
        "fc_weight": model.fc.weight.data.numpy().flatten().tolist(),
        "fc_bias": model.fc.bias.data.numpy().tolist(),
    }
    with open("bench/results/resnet_reference.json", "w") as f:
        json.dump(result, f)
    print(f"  saved bench/results/resnet_reference.json")


def gen_whisper_conv_stem():
    """Whisper conv stem + LayerNorm + GELU FFN (no attention).

    Tests Conv1d, GELU, transpose, LayerNorm, Linear — the ops unique
    to Whisper. Attention is tested separately by existing gpu_smoke tests.
    """
    print("\n=== Whisper conv stem reference ===")
    torch.manual_seed(42)

    d_model = 64
    n_mels = 16
    mel_len = 32
    ffn_dim = 128

    class ConvStemFFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(n_mels, d_model, 3, padding=1)
            self.conv2 = nn.Conv1d(d_model, d_model, 3, stride=2, padding=1)
            self.ln = nn.LayerNorm(d_model)
            self.fc1 = nn.Linear(d_model, ffn_dim)
            self.fc2 = nn.Linear(ffn_dim, d_model)
            self.final_ln = nn.LayerNorm(d_model)

        def forward(self, mel):
            x = torch.nn.functional.gelu(self.conv1(mel))
            x = torch.nn.functional.gelu(self.conv2(x))
            x = x.transpose(1, 2)  # [B, C, L] -> [B, L, C]
            h = self.ln(x)
            h = torch.nn.functional.gelu(self.fc1(h))
            h = self.fc2(h)
            x = x + h
            return self.final_ln(x)

    model = ConvStemFFN()
    model.eval()

    mel = torch.randn(1, n_mels, mel_len)
    with torch.no_grad():
        output = model(mel)

    seq_len = output.shape[1]
    print(f"  mel: {mel.shape}, output: {output.shape}")
    print(f"  output[:5]: {output[0, 0, :5].tolist()}")

    result = {
        "input": mel.numpy().flatten().tolist(),
        "output": output.numpy().flatten().tolist(),
        "d_model": d_model,
        "n_mels": n_mels,
        "mel_len": mel_len,
        "seq_len": seq_len,
        "ffn_dim": ffn_dim,
    }
    sd = model.state_dict()
    for name, tensor in sd.items():
        result[name] = tensor.numpy().flatten().tolist()

    with open("bench/results/whisper_reference.json", "w") as f:
        json.dump(result, f)
    print(f"  saved bench/results/whisper_reference.json")


if __name__ == "__main__":
    os.makedirs("bench/results", exist_ok=True)
    gen_resnet18()
    gen_whisper_conv_stem()
    print("\nDone.")
