#!/usr/bin/env python3
"""Extract the TF decoder body's graph_def and walk ops after base_conv_last
to figure out the 4-channel expansion that meganeura is missing.

The decoder body takes embed [1, S, 256] and outputs [1, 200, 480, 4]. The
last LEARNABLE op is base_conv_last (7x7 conv, 64 → 2 channels). After that,
there are parameter-free ops that reshape/expand to 4 channels. This script
prints the topology so we can mirror those ops in meganeura.
"""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import tensorflow as tf  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402


def main():
    print("Loading decoder ...")
    root = snapshot_download("google/magenta-realtime",
                             allow_patterns=["savedmodels/ssv2_48k_stereo/decoder/**"])
    with tf.device("/cpu:0"):
        dec = tf.saved_model.load(str(Path(root) / "savedmodels" / "ssv2_48k_stereo" / "decoder"))

    # Look at signatures and their concrete functions.
    print(f"\nDec signatures: {list(dec.signatures.keys())}")
    body = dec.keras_api.layers[0]

    # Get concrete function: `body` has __call__ which is a RestoredFunction.
    fn = body.__call__
    print(f"\nbody.__call__ type: {type(fn).__name__}")
    # Trigger a trace to get a concrete function.
    import numpy as np
    cf = fn.get_concrete_function(tf.TensorSpec([1, 50, 256], tf.float32))
    print(f"  concrete function: {cf}")
    g = cf.graph
    print(f"  graph: {len(g.get_operations())} ops")

    # Walk operations and find base_conv_last + everything after it.
    ops = list(g.get_operations())
    print("\n=== Looking for 'base_conv_last' producer ===")
    base_op_idx = None
    for i, op in enumerate(ops):
        if "base_conv_last" in op.name and op.type in ("Conv2D", "BiasAdd", "Add", "AddV2"):
            print(f"  [{i}] {op.type:15s} name={op.name}  outputs: {[t.shape.as_list() for t in op.outputs]}")
            if base_op_idx is None:
                base_op_idx = i

    # The body is wrapped in a StatefulPartitionedCall. Need to look at the
    # inner function in the graph_def's function library.
    print("\n=== Graph function library ===")
    fdef = cf.graph.as_graph_def()
    print(f"  Function library has {len(fdef.library.function)} functions")
    # Find the main body function (likely the largest by node count).
    funcs = sorted(fdef.library.function, key=lambda f: len(f.node_def), reverse=True)
    for f in funcs[:5]:
        print(f"  fn: {f.signature.name}  nodes={len(f.node_def)}")
    # Pick the biggest function and walk its nodes from the last (output)
    # backwards, listing types.
    main = funcs[0]
    print(f"\n=== Main function {main.signature.name} ===")
    print(f"  signature inputs: {[a.name for a in main.signature.input_arg]}")
    print(f"  signature outputs: {[a.name for a in main.signature.output_arg]}")
    # Walk nodes; print those matching 'base_conv_last' and everything that follows.
    nodes = list(main.node_def)
    # Index nodes by name for lookup.
    by_name = {n.name: n for n in nodes}
    # Find base_conv_last conv + its consumers/descendants.
    base_nodes = [n for n in nodes if "base_conv_last" in n.name]
    print(f"\n  base_conv_last related nodes ({len(base_nodes)}):")
    for n in base_nodes:
        attrs = list(n.attr.keys())[:5]
        print(f"    {n.op:18s} {n.name[:80]:80s} inputs={[i.split(':')[0] for i in n.input][:3]} attrs={attrs}")

    # Find the output node and trace BACKWARD using inputs.
    # Output name from signature.
    out_arg = main.signature.output_arg[0].name
    print(f"\n  output arg: {out_arg}")
    # The signature output is mapped via main.ret to an internal node.
    print(f"  ret map: {dict(main.ret)}")
    target_name = main.ret[out_arg].split(":")[0]
    print(f"  target node: {target_name}")
    # Walk from target back to a base_conv_last node and print the chain.
    if base_nodes:
        base_names = {n.name for n in base_nodes}
        print(f"\n  Backward walk from output until reaching base_conv_last:")
        visited = set()
        stack = [(target_name, 0)]
        chain = []
        while stack:
            name, depth = stack.pop()
            if name in visited: continue
            visited.add(name)
            node = by_name.get(name)
            if node is None: continue
            chain.append((depth, node))
            if name in base_names:
                continue
            for inp in node.input:
                inp_name = inp.split(":")[0].lstrip("^")
                if inp_name not in visited:
                    stack.append((inp_name, depth+1))
        # Print chain reverse (deepest first = closest to inputs).
        for depth, n in sorted(chain, key=lambda x: -x[0]):
            print(f"    d{depth} {n.op:18s} {n.name[:75]}")

    # Dump the actual Const values for any reshape-related node.
    print("\n=== Reshape constants ===")
    for n in nodes:
        if "tf.reshape" in n.name and n.op == "Const":
            try:
                val = tf.make_ndarray(n.attr["value"].tensor)
                print(f"  {n.name:70s} = {val}")
            except Exception:
                pass
        if "tf.reshape_9" in n.name or "temporal_cropping" in n.name:
            # print attrs
            if n.op == "Const":
                try:
                    val = tf.make_ndarray(n.attr["value"].tensor)
                    print(f"  {n.name:70s} = {val}")
                except Exception:
                    pass

    # Inspect what tf.reshape_9 actually does — find its inputs.
    print("\n=== tf.reshape_9/Reshape inputs ===")
    for n in nodes:
        if n.name == "tf.reshape_9/Reshape":
            print(f"  op: {n.op}")
            print(f"  inputs: {list(n.input)}")
        if "tf.reshape_9" in n.name and "Pack" in n.op:
            print(f"  Pack {n.name}: inputs={list(n.input)}")

    # === Inspect tensor SHAPES of key nodes by re-running the function ===
    # We build a TF callable that returns intermediate tensors.
    print("\n=== Inferring shapes by re-running body with hooks ===")
    # Strategy: import the function library function into a fresh graph,
    # then call it with our input. Pull intermediates by name.
    import tensorflow as tf
    nodes_of_interest = [
        "input_layer/base_conv_last/freq_dim_pad/PartitionedCall",
        "input_layer/base_conv_last/conv/StatefulPartitionedCall",
        "tf.reshape_8/Reshape",
        "tf.compat.v1.transpose_3/transpose",
        "tf.reshape_9/Reshape",
        "temporal_cropping//PartitionedCall",
    ]
    # Use tf.function to wrap the body call and add the intermediates as outputs.
    # We re-trace using lower-level tools.

    # Simpler: just print each node's output shape from the graph_def's
    # node_def attrs (TF stores output shape inference for some ops).
    print("\n=== Each interesting node's static shape ===")
    # Build a tf.Graph from the function so we can do shape inference.
    fn_graph = tf.Graph()
    with fn_graph.as_default():
        tf.graph_util.import_graph_def(fdef.library.function[0].SerializeToString() if False else fdef, name="")
    # Find tensors and print shapes.
    print(f"  fn_graph ops: {len(fn_graph.get_operations())}")
    return
    interesting = ("tf.reshape_5", "tf.reshape_6", "tf.reshape_7", "tf.reshape_8", "tf.reshape_9",
                   "tf.compat.v1.transpose_3", "tf.compat.v1.transpose_4",
                   "temporal_cropping", "base_conv_last/conv", "tf.__operators__.getitem_5",
                   "tf.__operators__.getitem_8", "tf.__operators__.getitem_9")
    for n in nodes:
        if any(t in n.name for t in interesting):
            attr_dict = {}
            for k in n.attr:
                a = n.attr[k]
                if a.HasField("tensor"):
                    try:
                        v = tf.make_ndarray(a.tensor)
                        attr_dict[k] = repr(v)
                    except Exception:
                        attr_dict[k] = f"<tensor {a.tensor.dtype}>"
                elif a.s:
                    attr_dict[k] = a.s.decode(errors="replace")[:60]
                elif a.i:
                    attr_dict[k] = a.i
                elif a.f:
                    attr_dict[k] = a.f
                elif a.b is not None:
                    attr_dict[k] = a.b
                elif a.HasField("list"):
                    if a.list.i:
                        attr_dict[k] = list(a.list.i)
            ins = [i.split(":")[0] for i in n.input][:5]
            print(f"  {n.op:18s} {n.name[:55]:<55s}  inputs={ins} attrs={list(attr_dict.items())[:3]}")
    return

    print("\n=== Body output tensor ===")
    body_output = cf.outputs[0]
    print(f"  shape: {body_output.shape}")
    print(f"  produced by op: {body_output.op.name} ({body_output.op.type})")

    print("\n=== Walking back from body output to base_conv_last ===")
    # BFS from body output, find the chain of ops.
    visited = set()
    to_visit = [body_output.op]
    chain = []
    while to_visit:
        op = to_visit.pop(0)
        if op.name in visited: continue
        visited.add(op.name)
        chain.append(op)
        for inp in op.inputs:
            if inp.op.name not in visited:
                to_visit.append(inp.op)
        if "base_conv_last" in op.name:
            break  # stop expanding past the conv

    # Now print the ops from the body output side back to base_conv_last,
    # filtering to only show non-trivial ones.
    print(f"  total ops in dep tree (BFS): {len(chain)}")
    # Sort topologically by walking output → input chain.
    seen_topo = set()
    def topo(op, depth=0):
        if op.name in seen_topo: return
        seen_topo.add(op.name)
        for inp in op.inputs:
            topo(inp.op, depth+1)
        # Print after recursion
        shapes = [t.shape.as_list() if t.shape.dims is not None else None for t in op.outputs]
        print(f"  {op.type:20s} {op.name[:70]:<70s}  shapes={shapes}")
    topo(body_output.op)


if __name__ == "__main__":
    main()
