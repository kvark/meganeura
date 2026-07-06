"""Dump MusicCoCa variables directly via TF checkpoint reader.

Bypasses tf.saved_model.load which fails because the SavedModel embeds
SentencepieceOp (from tensorflow-text). We only need the weight tensors,
not to RUN the model — the architecture is inferred from variable names.
"""
import os, json, shutil
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
import numpy as np
import tensorflow as tf
from huggingface_hub import snapshot_download
from safetensors.numpy import save_file as save_safetensors

DUMP = Path("magenta_rt_codec_dump")
DUMP.mkdir(exist_ok=True)


def dump_checkpoint_vars(prefix_in_safetensors, variables_prefix_path):
    """Read variables.data + variables.index → dict {name: ndarray}."""
    reader = tf.train.load_checkpoint(str(variables_prefix_path))
    shapes = reader.get_variable_to_shape_map()
    dtypes = reader.get_variable_to_dtype_map()
    out = {}
    for name in sorted(shapes.keys()):
        dt = dtypes[name]
        if dt.name in ("string", "resource"):
            print(f"  skip non-tensor: {name}  dtype={dt.name}")
            continue
        arr = reader.get_tensor(name)
        # Convert dtypes that safetensors doesn't natively support.
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        flat_name = name.replace("/", ".")
        out[f"{prefix_in_safetensors}.{flat_name}"] = np.ascontiguousarray(arr)
    return out


def main():
    root = Path(snapshot_download("google/magenta-realtime", allow_patterns=[
        "savedmodels/musiccoca_mv212f_cpu_novocab/variables/**",
        "savedmodels/musiccoca_mv212_quant/variables/**",
        "vocabularies/musiccoca_mv212f_vocab.model",
        "testdata/musiccoca_mv212/**",
    ]))
    print(f"root: {root}")

    weights = {}

    # MusicCoCa variables (the "novocab" variant has the same weights as compat,
    # just without the embedded SentencepieceOp tokenizer).
    mc_var_prefix = root / "savedmodels" / "musiccoca_mv212f_cpu_novocab" / "variables" / "variables"
    print(f"\nReading MusicCoCa variables from {mc_var_prefix}.*...")
    mc_w = dump_checkpoint_vars("musiccoca", mc_var_prefix)
    print(f"  loaded {len(mc_w)} tensors")
    weights.update(mc_w)

    # Quantizer.
    q_var_prefix = root / "savedmodels" / "musiccoca_mv212_quant" / "variables" / "variables"
    print(f"\nReading quantizer variables from {q_var_prefix}.*...")
    q_w = dump_checkpoint_vars("musiccoca_quant", q_var_prefix)
    print(f"  loaded {len(q_w)} tensors")
    weights.update(q_w)

    # Save.
    out_st = DUMP / "weights_musiccoca.safetensors"
    save_safetensors(weights, str(out_st))
    print(f"\nSaved {len(weights)} tensors to {out_st}  "
          f"({out_st.stat().st_size // 1024 // 1024} MB)")

    # SentencePiece vocab.
    vocab_src = root / "vocabularies" / "musiccoca_mv212f_vocab.model"
    vocab_dst = DUMP / "musiccoca_vocab.model"
    shutil.copy(vocab_src, vocab_dst)
    print(f"Copied vocab: {vocab_dst}  ({vocab_dst.stat().st_size // 1024} KB)")

    # Test data.
    td = root / "testdata" / "musiccoca_mv212"
    test_data = {}
    inputs_text = ""
    for f in td.iterdir():
        if f.suffix == ".npy":
            arr = np.load(f)
            test_data[f.stem] = arr
            print(f"  testdata.{f.stem}: shape={list(arr.shape)} dtype={arr.dtype}")
        elif f.suffix == ".txt":
            inputs_text = f.read_text()
    if test_data:
        save_safetensors(test_data, str(DUMP / "musiccoca_testdata.safetensors"))
        print(f"Saved test data to {DUMP / 'musiccoca_testdata.safetensors'}")
    if inputs_text:
        (DUMP / "musiccoca_inputs.txt").write_text(inputs_text)
        print(f"Saved prompts to {DUMP / 'musiccoca_inputs.txt'} ({len(inputs_text.splitlines())} lines)")

    # Manifest: variable name + shape summary, grouped by top-level prefix.
    from collections import Counter
    by_prefix = Counter()
    for name in weights:
        parts = name.split(".", 4)
        if len(parts) >= 4:
            by_prefix[".".join(parts[:3])] += 1
    print(f"\nTop-3-component prefixes ({len(by_prefix)} groups):")
    for k, n in sorted(by_prefix.items()):
        print(f"  {k:60s}  {n}")

    manifest = {
        "musiccoca_savedmodel": "musiccoca_mv212f_cpu_novocab",
        "quantizer_savedmodel": "musiccoca_mv212_quant",
        "vocab": "musiccoca_mv212f_vocab.model",
        "tensors": [
            {"name": k, "shape": list(v.shape), "dtype": str(v.dtype)}
            for k, v in sorted(weights.items())
        ],
        "test_prompts": inputs_text.splitlines() if inputs_text else [],
        "test_tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                         for k, v in test_data.items()},
    }
    (DUMP / "musiccoca_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved manifest to {DUMP / 'musiccoca_manifest.json'}")


if __name__ == "__main__":
    main()
