#!/usr/bin/env python3
"""Capture TF body's intermediate tensors (base_conv_last + body output)
by re-building the inner function with extra outputs added to its signature.

Strategy: TF's saved concrete function exposes only the body's final output.
But the inner FunctionDef has every op as a named node. We import the
FunctionDef into a fresh tf.Graph as a callable, and add fetches for any
intermediate tensor by name.
"""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from safetensors.numpy import save_file as save_safetensors  # noqa: E402


DUMP = Path("magenta_rt_codec_dump")


def main():
    print("Loading decoder ...")
    root = snapshot_download("google/magenta-realtime",
                             allow_patterns=["savedmodels/ssv2_48k_stereo/decoder/**"])
    with tf.device("/cpu:0"):
        dec = tf.saved_model.load(str(Path(root) / "savedmodels" / "ssv2_48k_stereo" / "decoder"))
    body = dec.keras_api.layers[0]

    embed = np.zeros((1, 50, 256), dtype=np.float32)
    embed[0, 0, 0] = 1.0  # one non-zero so we can verify shapes/zeros aren't trivial

    # The body output is [1, 200, 480, 4]. Capture it as ground truth.
    print("Running body ...")
    with tf.device("/cpu:0"):
        body_out = body(tf.constant(embed))
    print(f"body_out: {body_out.shape}  range [{body_out.numpy().min():.4f}, {body_out.numpy().max():.4f}]")

    captured = {"embed": embed, "body_out": body_out.numpy()}

    # Try to also fetch base_conv_last output by intercepting it via the
    # function library. tf.compat.v1 lets us import and rebind outputs.
    cf = body.__call__.get_concrete_function(tf.TensorSpec([1, 50, 256], tf.float32))
    fdef = cf.graph.as_graph_def()
    funcs = sorted(fdef.library.function, key=lambda f: len(f.node_def), reverse=True)
    main_fn = funcs[0]

    # Find candidate intermediate nodes.
    intermediate_names = [
        "input_layer/base_conv_last/conv/StatefulPartitionedCall",  # 4D conv output
        "tf.reshape_8/Reshape",   # 5D [2, ?, T, 480, 2]
        "tf.compat.v1.transpose_3/transpose",   # transpose output
        "tf.reshape_9/Reshape",   # [-1, T, 480, 4]
    ]

    # Build a tf.Graph that fetches the intermediates. We use tf.compat.v1
    # graph editing: take the main function's nodes, register them in a
    # tf.Graph as if it were a normal graph (not a function), and use
    # placeholders for the variables. This is complex; a simpler trick is
    # to monkeypatch the body via tf.function tracing where we add hooks.
    #
    # Easiest practical approach: just call the body's __call__ via the
    # restored function and use tf.print to print intermediate tensors
    # during execution. But that doesn't return them programmatically.
    #
    # The SIMPLEST approach: register a tf.function that walks the function
    # library and calls the inner function but adds extra outputs.

    # Use tf.python.framework.function_def_to_graph + manual variable substitution.
    from tensorflow.python.framework import function_def_to_graph
    new_g = tf.Graph()
    try:
        with new_g.as_default():
            # Register all library functions first.
            for fn in fdef.library.function:
                tf.compat.v1.import_graph_def(
                    tf.compat.v1.GraphDef(library=tf.compat.v1.FunctionDefLibrary(function=[fn])),
                    name="")
            fg = function_def_to_graph.function_def_to_graph(main_fn)
            print(f"  imported main fn: {len(fg.get_operations())} ops")
            # Look up intermediate tensor shapes.
            for name in intermediate_names:
                try:
                    op = fg.get_operation_by_name(name)
                    for t in op.outputs:
                        sh = t.shape.as_list() if t.shape.dims is not None else "?"
                        print(f"  {name}[{t.value_index}]: shape={sh}")
                except Exception as e:
                    print(f"  {name}: not found ({str(e)[:80]})")
    except Exception as e:
        print(f"  function rebuild failed: {str(e)[:300]}")

    out = DUMP / "tf_body_intermediates.safetensors"
    save_safetensors(captured, str(out))
    print(f"\nSaved {len(captured)} tensors to {out}")


if __name__ == "__main__":
    main()
