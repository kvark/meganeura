#!/usr/bin/env python3
"""Trace SpectroStream encoder + decoder to learn the exact layer-by-layer structure.

Run after dump_codec_local.py has populated magenta_rt_codec_dump/. Walks the
TF concrete-function ops and prints them with input/output tensor shapes, so we
can reconstruct the architecture for the meganeura port.

Output:
  - prints a structured trace of decoder ops
  - writes `decoder_trace.json` with all ops + shapes for later reference

Run:
  nix-shell tools/magenta_rt/shell.nix --run "python tools/magenta_rt/trace_codec.py"
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import tensorflow as tf  # noqa: E402
import numpy as np  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

REPO = "google/magenta-realtime"


def load_codec_savedmodels():
    print("Resolving SpectroStream cache ...")
    root = snapshot_download(REPO, allow_patterns=["savedmodels/ssv2_48k_stereo/**"])
    base = Path(root) / "savedmodels" / "ssv2_48k_stereo"
    with tf.device("/cpu:0"):
        enc = tf.saved_model.load(str(base / "encoder"))
        dec = tf.saved_model.load(str(base / "decoder"))
    return enc, dec


def _shape(t):
    """Convert a tensor's shape to a list (or None if rank-unknown)."""
    if t.shape.dims is None:
        return None
    return [int(d) if d is not None else -1 for d in t.shape.as_list()]


def _op_info(op):
    return {
        "op": op.type,
        "name": op.name,
        "attrs": {k: str(op.get_attr(k))[:200] for k in op.node_def.attr},
        "inputs": [{"name": t.name, "shape": _shape(t), "dtype": t.dtype.name} for t in op.inputs],
        "outputs": [{"name": t.name, "shape": _shape(t), "dtype": t.dtype.name} for t in op.outputs],
    }


def trace_concrete_function(fn, name: str):
    """Dump ops from a tf.function's concrete graph, INCLUDING FunctionDef library
    bodies (the real work hides inside StatefulPartitionedCall sub-functions)."""
    graph = fn.graph
    ops_info = {"_main_": [_op_info(op) for op in graph.get_operations()]}

    # Walk every FunctionDef in the graph's library. These are the real
    # computation bodies that StatefulPartitionedCall dispatches to.
    gdef = graph.as_graph_def(add_shapes=True)
    for fdef in gdef.library.function:
        fname = fdef.signature.name
        # We don't have a live Graph object for the FunctionDef, but each
        # NodeDef has op type + name + attrs. That's enough for our purpose.
        nodes = []
        for nd in fdef.node_def:
            attrs = {}
            for k, v in nd.attr.items():
                # Stringify attr value compactly. Tensors get shape + dtype only.
                s = str(v)
                if len(s) > 300:
                    s = s[:300] + "..."
                attrs[k] = s
            nodes.append({
                "op": nd.op,
                "name": nd.name,
                "inputs": list(nd.input),
                "attrs": attrs,
            })
        ops_info[fname] = nodes
    main_n = len(ops_info["_main_"])
    fn_count = sum(len(v) for k, v in ops_info.items() if k != "_main_")
    print(f"  {name}: main graph={main_n} ops, function library={len(ops_info)-1} fns, {fn_count} ops total")
    return ops_info


INTERESTING = {
    "Conv2D", "Conv2DBackpropInput", "Conv1D", "Conv1DBackpropInput",
    "DepthwiseConv2dNative",
    "Mul", "Add", "AddV2", "BiasAdd",
    "Tanh", "Relu", "Elu", "Sigmoid", "LeakyRelu",
    "Reshape", "Transpose", "Pad", "MirrorPad",
    "Pack", "Unpack", "ConcatV2", "Split", "Slice", "StridedSlice",
    "Mean", "Variance", "BatchNorm", "FusedBatchNorm",
    "MatMul", "Einsum",
}


def print_op_summary(ops_info, name):
    """Print only the largest FunctionDef in the library (that's the main body)."""
    print(f"\n=== {name} ===")
    # Pick the FunctionDef with the most ops — that's the model body.
    candidates = {k: v for k, v in ops_info.items() if k != "_main_"}
    if not candidates:
        print("  (no FunctionDef library; nothing to dump)")
        return
    biggest = max(candidates, key=lambda k: len(candidates[k]))
    print(f"  (main body: {biggest}, {len(candidates[biggest])} ops)")
    for o in candidates[biggest]:
        if o["op"] not in INTERESTING:
            continue
        n = o["name"]
        if len(n) > 70:
            n = "…" + n[-68:]
        # Pull the most informative attr inline.
        info = ""
        for k in ("strides", "padding", "dilations", "ksize", "T"):
            if k in o.get("attrs", {}):
                v = o["attrs"][k].replace("\n", " ").strip()
                info += f"  {k}={v[:60]}"
        print(f"  [{o['op']:22s}] {n}{info}")


def main():
    out_dir = Path("magenta_rt_codec_dump")
    out_dir.mkdir(exist_ok=True)

    enc, dec = load_codec_savedmodels()

    # Decoder takes [B, S, D] embedding (B=1, S=50 frames, D=256).
    dummy_embed = tf.zeros((1, 50, 256), dtype=tf.float32)
    print("Tracing decoder with input [1, 50, 256] ...")
    with tf.device("/cpu:0"):
        # Resolve concrete function: dec is callable directly.
        cf_dec = dec.__call__.get_concrete_function(dummy_embed)
    dec_ops = trace_concrete_function(cf_dec, "decoder")
    print_op_summary(dec_ops, "DECODER")

    # Encoder takes [B, T, 2] = [1, 48000, 2] for 1 sec of stereo audio.
    dummy_audio = tf.zeros((1, 48000, 2), dtype=tf.float32)
    print("\nTracing encoder with input [1, 48000, 2] ...")
    with tf.device("/cpu:0"):
        cf_enc = enc.__call__.get_concrete_function(dummy_audio)
    enc_ops = trace_concrete_function(cf_enc, "encoder")
    print_op_summary(enc_ops, "ENCODER")

    (out_dir / "decoder_trace.json").write_text(json.dumps(dec_ops, indent=2))
    (out_dir / "encoder_trace.json").write_text(json.dumps(enc_ops, indent=2))
    print(f"\nFull traces written to {out_dir}/{{decoder,encoder}}_trace.json")


if __name__ == "__main__":
    main()
