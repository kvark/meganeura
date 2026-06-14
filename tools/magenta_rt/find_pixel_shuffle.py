#!/usr/bin/env python3
"""Find what pixel_shuffle-equivalent op TF uses between decoder_0 and decoder_1.

Print the perms of every Transpose op in the body's main function graph_def.
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
    cf = body.__call__.get_concrete_function(tf.TensorSpec([1, 50, 256], tf.float32))
    fdef = cf.graph.as_graph_def()
    funcs = sorted(fdef.library.function, key=lambda f: len(f.node_def), reverse=True)
    main_fn = funcs[0]

    print("=== All Reshape ops with their shape constant inputs ===")
    by_name = {n.name: n for n in main_fn.node_def}

    # Collect each Reshape's static shape Const inputs.
    for n in main_fn.node_def:
        if n.op == "Reshape" and "Reshape" in n.name:
            # The shape input is the second input
            if len(n.input) < 2:
                continue
            shape_input = n.input[1].split(":")[0]
            shape_node = by_name.get(shape_input)
            print(f"\n  {n.name}")
            if shape_node and shape_node.op == "Pack":
                # Pack's inputs are the individual dim values
                dims = []
                for di_name in shape_node.input:
                    di_name_clean = di_name.split(":")[0]
                    di_node = by_name.get(di_name_clean)
                    if di_node and di_node.op == "Const":
                        try:
                            v = tf.make_ndarray(di_node.attr["value"].tensor)
                            dims.append(str(v))
                        except Exception:
                            dims.append("?")
                    else:
                        dims.append(f"<{di_node.op if di_node else '??'}: {di_name_clean.rsplit('/', 1)[-1]}>")
                print(f"    shape pack: [{', '.join(dims)}]")

    print("\n=== All Transpose op perms ===")
    for n in main_fn.node_def:
        if n.op == "Transpose":
            print(f"\n  {n.name}")
            # Perm is second input.
            if len(n.input) >= 2:
                perm_name = n.input[1].split(":")[0]
                perm_node = by_name.get(perm_name)
                if perm_node and perm_node.op == "Const":
                    try:
                        v = tf.make_ndarray(perm_node.attr["value"].tensor)
                        print(f"    perm: {list(v)}")
                    except Exception:
                        print(f"    perm: <unreadable>")
                else:
                    print(f"    perm input: {perm_name}")


if __name__ == "__main__":
    main()
