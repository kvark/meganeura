#!/usr/bin/env python3
"""Walk the loaded SpectroStream decoder's Keras layer list and print each
layer's name + signature. If the layers are in execution order, we can feed
the codebook embedding through them sequentially to capture intermediates."""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import tensorflow as tf  # noqa: E402
import numpy as np  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402


def main():
    root = snapshot_download("google/magenta-realtime", allow_patterns=["savedmodels/ssv2_48k_stereo/decoder/**"])
    with tf.device("/cpu:0"):
        dec = tf.saved_model.load(str(Path(root) / "savedmodels" / "ssv2_48k_stereo" / "decoder"))

    # Recursively walk the layer tree (Keras model has nested layers).
    def walk(obj, prefix="", depth=0, max_depth=4):
        if depth > max_depth: return
        ka = getattr(obj, "keras_api", None)
        if ka is None and hasattr(obj, "layers"):
            layers = obj.layers
        else:
            layers = getattr(ka, "layers", None)
        if layers is None: return
        for layer in layers:
            name = getattr(layer, "name", None) or getattr(layer, "_self_name", "?")
            n_vars = len(getattr(layer, "variables", []))
            has_call = hasattr(layer, "__call__")
            print(f"{prefix}  {name[:60]:<60s}  vars={n_vars:<3d}  callable={has_call}")
            walk(layer, prefix + "  ", depth + 1, max_depth)

    print(f"Layer tree:")
    walk(dec)


if __name__ == "__main__":
    main()
