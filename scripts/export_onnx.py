#!/usr/bin/env python3
"""Export HuggingFace models to ONNX for Meganeura comparison testing.

Supports:
  - dacorvo/mnist-mlp  (simple 3-layer MLP)
  - HuggingFaceTB/SmolLM2-135M  (decoder-only transformer)

Usage:
  pip install torch transformers safetensors onnx
  python scripts/export_onnx.py mnist-mlp
  python scripts/export_onnx.py smollm2
"""

import argparse
import os
import sys

import torch
import torch.nn as nn


def export_mnist_mlp(output_dir: str):
    """Export dacorvo/mnist-mlp to ONNX."""
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    print("Downloading dacorvo/mnist-mlp weights...")
    path = hf_hub_download("dacorvo/mnist-mlp", "model.safetensors")
    state = load_file(path)

    print("Model tensors:")
    for name, tensor in state.items():
        print(f"  {name}: {tensor.shape} {tensor.dtype}")

    # Reconstruct the model architecture: Linear(784,256)+ReLU -> Linear(256,256)+ReLU -> Linear(256,10)
    class MnistMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layer = nn.Linear(784, 256)
            self.mid_layer = nn.Linear(256, 256)
            self.output_layer = nn.Linear(256, 10)

        def forward(self, x):
            x = torch.relu(self.input_layer(x))
            x = torch.relu(self.mid_layer(x))
            x = self.output_layer(x)  # raw logits (no softmax, for comparison)
            return x

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

    # Also run a quick reference inference for comparison
    with torch.no_grad():
        ref_input = torch.zeros(1, 784)
        ref_output = model(ref_input)
        print(f"Reference output (zeros input): {ref_output[0, :5].tolist()}")


def export_smollm2(output_dir: str):
    """Export HuggingFaceTB/SmolLM2-135M to ONNX.

    This exports the full causal LM with a fixed sequence length.
    The ONNX graph will contain decomposed ops (no custom CausalAttention etc).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo = "HuggingFaceTB/SmolLM2-135M"
    print(f"Downloading {repo}...")

    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32)
    model.eval()

    # Fixed sequence length for export
    seq_len = 16
    dummy_ids = torch.randint(0, 49152, (1, seq_len), dtype=torch.long)

    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "smollm2-135m.onnx")

    print(f"Exporting to {onnx_path} (seq_len={seq_len})...")
    torch.onnx.export(
        model,
        (dummy_ids,),
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=17,
    )
    print(f"Done: {onnx_path} ({os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB)")

    # Reference: encode a test prompt and get logits
    prompt = "The meaning of life is"
    enc = tokenizer(prompt, return_tensors="pt", padding="max_length",
                    max_length=seq_len, truncation=True)
    with torch.no_grad():
        out = model(enc["input_ids"])
        logits = out.logits
        # Print top-5 next token predictions from last real position
        input_ids = enc["input_ids"][0]
        num_tokens = (input_ids != tokenizer.pad_token_id).sum().item()
        if num_tokens > 0:
            last_logits = logits[0, num_tokens - 1]
            top5 = torch.topk(last_logits, 5)
            print(f"Reference top-5 next tokens for '{prompt}':")
            for i, (idx, val) in enumerate(zip(top5.indices, top5.values)):
                token = tokenizer.decode([idx.item()])
                print(f"  {i+1}. '{token}' (logit={val.item():.4f})")


def main():
    parser = argparse.ArgumentParser(description="Export HF models to ONNX")
    parser.add_argument("model", choices=["mnist-mlp", "smollm2", "all"],
                        help="Which model to export")
    parser.add_argument("--output-dir", default="models/onnx",
                        help="Output directory (default: models/onnx)")
    args = parser.parse_args()

    if args.model in ("mnist-mlp", "all"):
        export_mnist_mlp(args.output_dir)
    if args.model in ("smollm2", "all"):
        export_smollm2(args.output_dir)


if __name__ == "__main__":
    main()
