"""Match TFLite sg0 constants to the raw 80 vars by value comparison.

For each constant >1KB in TFLite sg0:
  - Convert its buffer to f32 array
  - For each raw safetensors var, check if any permutation/reshape matches
  - If match found, we have the jax2tf_arg_N → sg0 tensor mapping
"""
import os, struct, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb
from pathlib import Path


def load_safetensors(p):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
        raw = f.read()
    out = {}
    dtypes = {'F32': np.float32, 'I32': np.int32, 'F64': np.float64, 'I64': np.int64}
    for k, info in h.items():
        if k.startswith('__'): continue
        if info['dtype'] not in dtypes: continue
        s, e = info['data_offsets']
        sh = info['shape'] if info['shape'] else [1]
        out[k] = np.frombuffer(raw[s:e], dtype=dtypes[info['dtype']]).reshape(sh).copy()
    return out


WTS = "/x/Code/meganeura/magenta_rt_codec_dump/weights_musiccoca.safetensors"
TFLITE = "/tmp/musiccoca_text_encoder.tflite"

# Load raw vars (80 musiccoca + 12 quant codebooks + 1 quant counter).
print(f"Loading raw weights from {WTS}...")
raw = load_safetensors(WTS)
# Filter to musiccoca.tf_var_leaves.N
mc_vars = {}
import re
for k, v in raw.items():
    m_ = re.match(r"musiccoca\.tf_var_leaves\.(\d+)\.", k)
    if m_:
        idx = int(m_.group(1))
        mc_vars[idx] = v
print(f"  Loaded {len(mc_vars)} musiccoca vars (indices 0..{max(mc_vars):d})")

# Index raw vars by total bytes for quick filtering.
size_to_indices = {}
for i, arr in mc_vars.items():
    size = arr.nbytes
    size_to_indices.setdefault(size, []).append(i)

# Load TFLite.
print(f"\nLoading {TFLITE}...")
with open(TFLITE, "rb") as f:
    buf = f.read()
m = schema_fb.ModelT.InitFromObj(schema_fb.Model.GetRootAs(buf, 0))


def parse_buffer_f32(buf_bytes):
    return np.frombuffer(bytes(buf_bytes), dtype=np.float32).copy()


def compare_arrays_any_permutation(a_flat_sorted, b):
    """Quick check: does sorted-flat(b) match a_flat_sorted?"""
    b_flat_sorted = np.sort(b.flatten())
    if a_flat_sorted.shape != b_flat_sorted.shape:
        return False
    # Element-wise compare with f32 tolerance.
    return np.allclose(a_flat_sorted, b_flat_sorted, rtol=1e-5, atol=1e-7)


# Walk all subgraphs and look for f32 constants > 1KB.
print(f"\n=== Matching TFLite constants to raw vars ===")
matches = {}  # tflite_tensor_index → raw_var_index
sg0 = m.subgraphs[0]
candidates_checked = 0
for t_idx, t in enumerate(sg0.tensors):
    if not t.name:
        continue
    buf_idx = t.buffer
    if not buf_idx:
        continue
    buf_data = m.buffers[buf_idx].data
    if buf_data is None or len(buf_data) < 1024:
        continue
    # Get tensor's name + dtype.
    name = t.name.decode(errors='replace')
    # Default tensor type is FLOAT32 (0)
    is_f32 = (t.type == 0) if hasattr(t, 'type') else True
    if not is_f32:
        continue
    size = len(buf_data)
    # Pre-sort flat representation.
    arr = parse_buffer_f32(buf_data)
    arr_sorted = np.sort(arr)
    # Compare against raw vars of same size.
    for raw_idx in size_to_indices.get(size, []):
        raw_arr = mc_vars[raw_idx]
        if raw_arr.dtype != np.float32:
            continue
        if compare_arrays_any_permutation(arr_sorted, raw_arr):
            matches[t_idx] = (raw_idx, list(raw_arr.shape), list(t.shape))
            # Pretty short version of name for logging.
            short = name[:60]
            shape_str = str(list(t.shape))[:20]
            print(f"  match tflite[{t_idx:3d}] shape={shape_str:<22s} ⇔ arg{raw_idx:2d} (raw shape={list(raw_arr.shape)})  name={short}...")
            break
    candidates_checked += 1

print(f"\n{candidates_checked} sg0 constants checked, {len(matches)} matched to raw vars.")
# List unmatched raw vars.
matched_raws = {v[0] for v in matches.values()}
unmatched = sorted(set(mc_vars.keys()) - matched_raws)
print(f"\nUnmatched raw var indices: {unmatched}")
