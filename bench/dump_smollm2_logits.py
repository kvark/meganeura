#!/usr/bin/env python3
"""Dump SmolLM2-135M reference logits from HuggingFace transformers.

Runs a single forward pass on a short prompt and writes the full logits
tensor to a binary file (f32 little-endian) plus a JSON sidecar with
metadata. This is used by the Rust-side correctness test to validate
meganeura's output against PyTorch.

Usage:
    python bench/dump_smollm2_logits.py [--prompt "text"] [--out-dir bench/results]
"""

import argparse
import json
import struct
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--out-dir", default="bench/results")
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model: {args.model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]  # [1, seq_len]
    seq_len = input_ids.shape[1]
    token_list = input_ids[0].tolist()

    print(f"Prompt: \"{args.prompt}\"", file=sys.stderr)
    print(f"Token IDs: {token_list}", file=sys.stderr)
    print(f"Seq len: {seq_len}", file=sys.stderr)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    logits_2d = logits[0]  # [seq_len, vocab_size]
    vocab_size = logits_2d.shape[1]
    print(f"Logits shape: [{seq_len}, {vocab_size}]", file=sys.stderr)

    # Print top-5 predictions at each position for quick inspection
    for pos in range(seq_len):
        pos_logits = logits_2d[pos]
        top5 = torch.topk(pos_logits, 5)
        tokens_str = ", ".join(
            f"{tokenizer.decode([idx.item()])!r}({idx.item()})={val.item():.4f}"
            for idx, val in zip(top5.indices, top5.values)
        )
        print(f"  pos {pos}: {tokens_str}", file=sys.stderr)

    # Also do greedy generation for a few tokens
    greedy_ids = []
    for pos in range(seq_len):
        next_id = logits_2d[pos].argmax().item()
        greedy_ids.append(next_id)

    # The token predicted at pos (seq_len-1) is the first generated token
    print(f"\nGreedy next tokens (from each position):", file=sys.stderr)
    for pos, tid in enumerate(greedy_ids):
        decoded = tokenizer.decode([tid])
        print(f"  pos {pos} -> token {tid} = {decoded!r}", file=sys.stderr)

    # Write binary logits: row-major [seq_len, vocab_size] as f32 LE
    logits_flat = logits_2d.contiguous().view(-1).numpy()
    bin_path = os.path.join(args.out_dir, "smollm2_ref_logits.bin")
    with open(bin_path, "wb") as f:
        f.write(struct.pack(f"<{len(logits_flat)}f", *logits_flat))
    print(f"\nWrote {len(logits_flat)} floats to {bin_path}", file=sys.stderr)

    # Write JSON metadata
    meta = {
        "model": args.model,
        "prompt": args.prompt,
        "token_ids": token_list,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "greedy_next_tokens": greedy_ids,
        "logits_file": "smollm2_ref_logits.bin",
    }
    meta_path = os.path.join(args.out_dir, "smollm2_ref_logits.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata to {meta_path}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
