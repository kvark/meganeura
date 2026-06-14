"""Extract the jax2tf_arg_N → architectural usage mapping from the TFLite file."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import re
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb

with open("/tmp/musiccoca_text_encoder.tflite", "rb") as f:
    buf = f.read()
m = schema_fb.ModelT.InitFromObj(schema_fb.Model.GetRootAs(buf, 0))

builtin_names = {}
for name in dir(schema_fb.BuiltinOperator):
    if name.startswith("_"):
        continue
    val = getattr(schema_fb.BuiltinOperator, name)
    if isinstance(val, int):
        builtin_names[val] = name


def desc_op(sg, op):
    oc = m.operatorCodes[op.opcodeIndex]
    name = builtin_names.get(oc.builtinCode, f"UNK_{oc.builtinCode}")
    if oc.customCode:
        name = f"CUSTOM:{oc.customCode.decode()}"
    return name


# For every subgraph, find every tensor that contains 'jax2tf_arg_' in its name,
# extract the arg index + shape.
all_var_info = {}  # idx -> (shape, where_found, where_used_op)
for sg_idx, sg in enumerate(m.subgraphs):
    print(f"subgraph {sg_idx}: {len(sg.tensors)} tensors, {len(sg.operators or [])} ops")
    for t_idx, t in enumerate(sg.tensors):
        if not t.name:
            continue
        name = t.name.decode(errors='replace')
        match = re.search(r"jax2tf_arg_(\d+)", name)
        if match:
            arg_idx = int(match.group(1))
            shape = tuple(int(x) for x in (t.shape if t.shape is not None else []))
            if arg_idx not in all_var_info:
                all_var_info[arg_idx] = {
                    "shape": shape,
                    "name": name,
                    "first_uses": [],
                }
            # Find ops that USE this tensor.
            for op_idx, op in enumerate(sg.operators):
                if op.inputs is not None and t_idx in op.inputs:
                    op_name = desc_op(sg, op)
                    out_tensors = [sg.tensors[oi].name.decode(errors='replace') if sg.tensors[oi].name else f"<{oi}>" for oi in (op.outputs or [])]
                    if len(all_var_info[arg_idx]["first_uses"]) < 3:
                        all_var_info[arg_idx]["first_uses"].append(
                            f"sg{sg_idx} op[{op_idx}] {op_name} → {out_tensors[0] if out_tensors else '?'}"
                        )

# Print sorted by arg index
print(f"# Variable mapping for MusicCoCa text encoder (text branch)")
print(f"# {len(all_var_info)} variables identified")
print()
print(f"{'arg':>4s} {'shape':<35s}  first uses")
for i in sorted(all_var_info.keys()):
    info = all_var_info[i]
    print(f"  {i:2d}  {str(list(info['shape'])):<35s}  {info['first_uses'][0] if info['first_uses'] else '(unused)'}")

# Aggregate by shape.
print(f"\n# Variables grouped by shape:")
by_shape = {}
for i, info in all_var_info.items():
    by_shape.setdefault(info["shape"], []).append(i)
for shape, ids in sorted(by_shape.items()):
    print(f"  {str(list(shape)):<35s} : {ids}")
