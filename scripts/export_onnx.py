#!/usr/bin/env python3
"""Export standard models to ONNX for Meganeura examples.

Usage:
  pip install torch torchvision onnx
  python3 scripts/export_onnx.py resnet18
  python3 scripts/export_onnx.py whisper-tiny
  python3 scripts/export_onnx.py all
"""

import argparse
import os

import torch
import torch.nn as nn


def export_resnet18(output_dir: str):
    """Export torchvision ResNet-18 to ONNX."""
    from torchvision.models import resnet18, ResNet18_Weights

    print("Loading ResNet-18 (ImageNet pretrained)...")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "resnet18.onnx")

    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
    )
    print(f"Done: {onnx_path} ({os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB)")

    # Reference inference
    with torch.no_grad():
        out = model(dummy)
        top5 = torch.topk(out[0], 5)
        print(f"Reference top-5 class indices: {top5.indices.tolist()}")
        print(f"Reference top-5 logits: {[f'{v:.3f}' for v in top5.values.tolist()]}")


def export_whisper_tiny_encoder(output_dir: str):
    """Export Whisper-tiny encoder to ONNX.

    Only the encoder (mel spectrogram -> hidden states), not the decoder.
    """
    try:
        from transformers import WhisperModel
    except ImportError:
        print("SKIP: `transformers` not installed (pip install transformers)")
        return

    print("Loading openai/whisper-tiny encoder...")
    model = WhisperModel.from_pretrained("openai/whisper-tiny")
    encoder = model.encoder
    encoder.eval()

    # Whisper encoder input: mel spectrogram [batch, 80, 3000] (30s audio)
    # Use shorter sequence for tractability
    mel_len = 300  # ~3s
    dummy = torch.randn(1, 80, mel_len)

    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "whisper-tiny-encoder.onnx")

    print(f"Exporting encoder to {onnx_path} (mel_len={mel_len})...")
    torch.onnx.export(
        encoder,
        dummy,
        onnx_path,
        input_names=["mel"],
        output_names=["hidden_states"],
        opset_version=17,
    )
    print(f"Done: {onnx_path} ({os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB)")


def export_mnist_mlp(output_dir: str):
    """Export dacorvo/mnist-mlp to ONNX."""
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    print("Downloading dacorvo/mnist-mlp weights...")
    path = hf_hub_download("dacorvo/mnist-mlp", "model.safetensors")
    state = load_file(path)

    class MnistMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layer = nn.Linear(784, 256)
            self.mid_layer = nn.Linear(256, 256)
            self.output_layer = nn.Linear(256, 10)

        def forward(self, x):
            x = torch.relu(self.input_layer(x))
            x = torch.relu(self.mid_layer(x))
            return self.output_layer(x)

    model = MnistMLP()
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 784)
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "mnist-mlp.onnx")

    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Done: {onnx_path} ({os.path.getsize(onnx_path)} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument(
        "model",
        choices=["mnist-mlp", "resnet18", "whisper-tiny", "all"],
        help="Which model to export",
    )
    parser.add_argument(
        "--output-dir", default="models/onnx", help="Output directory"
    )
    args = parser.parse_args()

    if args.model in ("mnist-mlp", "all"):
        export_mnist_mlp(args.output_dir)
    if args.model in ("resnet18", "all"):
        export_resnet18(args.output_dir)
    if args.model in ("whisper-tiny", "all"):
        export_whisper_tiny_encoder(args.output_dir)


if __name__ == "__main__":
    main()
