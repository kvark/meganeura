#!/usr/bin/env python3
"""Walk body.keras_api.layers forward, calling each one and recording the
output shape. Lets us discover the TRUE per-block shape progression.

Strategy:
  - Start with embed [1, 50, 256] (the body's natural input).
  - For each layer i, try `body.keras_api.layers[i](x)`. If it works, record
    the output shape and use it as input to the next layer.
  - If a layer can't be called directly (e.g. needs multiple inputs from
    earlier non-adjacent layers — residual adds), skip it and continue.
  - For some layers (like activations), `__call__` may bypass; try fallbacks.
"""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402


def main():
    root = snapshot_download("google/magenta-realtime",
                             allow_patterns=["savedmodels/ssv2_48k_stereo/decoder/**"])
    with tf.device("/cpu:0"):
        dec = tf.saved_model.load(str(Path(root) / "savedmodels" / "ssv2_48k_stereo" / "decoder"))
    body = dec.keras_api.layers[0]
    body_layers = body.keras_api.layers
    print(f"body_layers count: {len(body_layers)}")

    embed = np.zeros((1, 50, 256), dtype=np.float32)
    # Start from layer 4 (conv1x1_a) with [1, 50, 1, 256].
    x = tf.constant(embed[:, :, None, :])
    print(f"\nstart at layer 4 with: {x.shape}\n")

    # Try each layer 4..96 in sequence; record successes.
    saved_outputs = {}  # i → tensor (for residual adds)
    start_i = 4
    for i in range(start_i, len(body_layers)):
        layer = body_layers[i]
        vars_ = list(getattr(layer, "variables", []))
        prefix = vars_[0].name.rsplit("/", 2)[0] if vars_ else "(no-vars)"
        # Try direct call.
        try:
            with tf.device("/cpu:0"):
                y = layer(x)
            sh = y.shape.as_list() if hasattr(y, "shape") else "unknown"
            print(f"  [{i:2d}] OK   {str(sh):28s}  {prefix}")
            saved_outputs[i] = y
            x = y
        except Exception as e:
            # Don't print the long error, just record it.
            print(f"  [{i:2d}] SKIP                              {prefix}")


if __name__ == "__main__":
    main()
