"""Look inside the temporal_cropping function."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import tensorflow as tf
from pathlib import Path
from huggingface_hub import snapshot_download

root = snapshot_download("google/magenta-realtime",
                         allow_patterns=["savedmodels/ssv2_48k_stereo/decoder/**"])
with tf.device("/cpu:0"):
    dec = tf.saved_model.load(str(Path(root) / "savedmodels" / "ssv2_48k_stereo" / "decoder"))
body = dec.keras_api.layers[0]
cf = body.__call__.get_concrete_function(tf.TensorSpec([1, 50, 256], tf.float32))
fdef = cf.graph.as_graph_def()
funcs = list(fdef.library.function)

# Find functions whose name contains "temporal_cropping".
temporal_funcs = [f for f in funcs if "temporal_cropping" in f.signature.name]
print(f"Found {len(temporal_funcs)} temporal_cropping functions:")
for f in temporal_funcs:
    print(f"  {f.signature.name}  inputs={[a.name for a in f.signature.input_arg]}  outputs={[a.name for a in f.signature.output_arg]}  nodes={len(f.node_def)}")

for f in temporal_funcs[:2]:
    print(f"\n=== Function {f.signature.name} ===")
    by_name = {n.name: n for n in f.node_def}
    for n in f.node_def:
        attrs_str = []
        for k in n.attr:
            a = n.attr[k]
            if a.HasField("tensor"):
                try:
                    import numpy as np
                    v = tf.make_ndarray(a.tensor)
                    attrs_str.append(f"{k}={v}")
                except Exception:
                    pass
        attrs = "; ".join(attrs_str[:3])
        print(f"  {n.op:18s} {n.name[:50]:<50s} inputs={list(n.input)[:3]} {attrs[:80]}")
