#!/usr/bin/env python3
"""Dump per-layer intermediate tensors from the official SpectroStream decoder.

Strategy: walk dec.keras_api.layers[0]'s sub-layers (the streamable_model_1
body). Each sub-layer's variables retain their original names from training,
which lets us identify which TF layer it is (input_layer, decoder_0, ...,
base_conv_last). We then call sub-layers in execution order, capturing the
output of each. The result lets meganeura's Rust port validate layer-by-layer
instead of debugging end-to-end.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.numpy import save_file as save_safetensors  # noqa: E402


def main():
    out_dir = Path("magenta_rt_codec_dump")
    root = snapshot_download("google/magenta-realtime", allow_patterns=["savedmodels/ssv2_48k_stereo/decoder/**"])
    with tf.device("/cpu:0"):
        dec = tf.saved_model.load(str(Path(root) / "savedmodels" / "ssv2_48k_stereo" / "decoder"))

    print(f"dec.keras_api.layers: {len(dec.keras_api.layers)}")
    body = dec.keras_api.layers[0]
    print(f"body: type={type(body).__name__}, vars={len(body.variables)}")

    # Look for body.layers — Keras Model attribute.
    body_layers = getattr(body, "layers", None) or getattr(getattr(body, "keras_api", None), "layers", None) or []
    print(f"body.layers: {len(body_layers)}")
    if not body_layers:
        # Try deeper
        for child_name in dir(body):
            if child_name.startswith("_"): continue
            v = getattr(body, child_name)
            tn = type(v).__name__
            if "Wrapper" in tn or "List" in tn:
                print(f"  {child_name}: {tn}, len={len(v) if hasattr(v, '__len__') else '?'}")

    # If we found body_layers, identify each one by its variables' names.
    if body_layers:
        print(f"\nLayer-by-layer variable identification:\n")
        for i, layer in enumerate(body_layers):
            vars_ = list(getattr(layer, "variables", []))
            tn = type(layer).__name__
            if vars_:
                # Pick the most-prefix-y variable name
                names = [v.name for v in vars_]
                common = os.path.commonprefix(names)
                if common.endswith("/"):
                    pass
                else:
                    common = common.rsplit("/", 1)[0] + "/"
                print(f"  [{i:3d}] {tn:25s} vars={len(vars_)}  prefix={common}")
            else:
                # Layer with no vars — print type and any input shape attr
                print(f"  [{i:3d}] {tn:25s} (no vars)")

    # Try calling the decoder with reference tokens and capture the output.
    # Even without per-layer access, we can capture the FINAL output for comparison.
    print("\nLoading reference tokens + codebooks ...")
    refs = {}
    with safe_open(str(out_dir / "reference_codec.safetensors"), framework="numpy") as f:
        for k in f.keys():
            refs[k] = f.get_tensor(k)
    weights = {}
    with safe_open(str(out_dir / "weights_spectrostream.safetensors"), framework="numpy") as f:
        for k in f.keys():
            weights[k] = f.get_tensor(k)

    tokens = refs["tokens"]  # [S, K]
    codebooks = weights["quantizer.rvq_codebooks"]
    # Dequantize using all 64 levels.
    embed = np.zeros((tokens.shape[0], codebooks.shape[2]), dtype=np.float32)
    for k in range(tokens.shape[1]):
        embed += codebooks[k, tokens[:, k]]
    print(f"dequantized embed: {embed.shape}")

    embed_b = embed[None].astype(np.float32)  # [1, S, D]
    print(f"calling dec(embed) ...")
    with tf.device("/cpu:0"):
        audio_out = dec(embed_b).numpy()
    print(f"dec output: {audio_out.shape}")

    ref_audio = refs["reconstructed_audio"]
    print(f"reference reconstructed: {ref_audio.shape}")
    err = np.sqrt(np.mean((audio_out[0] - ref_audio) ** 2))
    rms = np.sqrt(np.mean(ref_audio ** 2))
    print(f"end-to-end err (rms): {err:.6f}, ref rms: {rms:.6f}, ratio: {err/rms:.4f}")

    # ===== Monkey-patch layers to capture intermediates =====
    print("\nMonkey-patching key layers to capture intermediate tensors ...")
    captures: dict[str, np.ndarray] = {}

    def hook(layer_idx: int, label: str):
        layer = body_layers[layer_idx]
        original = layer.__call__
        def wrapped(*args, **kwargs):
            out = original(*args, **kwargs)
            # Capture INPUT (first positional arg) and OUTPUT.
            if args:
                captures[f"{label}.in"] = np.asarray(args[0])
            captures[f"{label}.out"] = np.asarray(out)
            return out
        layer.__call__ = wrapped
        return original

    # Pick the layers whose variables identify them clearly.
    interesting = []
    for i, layer in enumerate(body_layers):
        vars_ = list(getattr(layer, "variables", []))
        if not vars_:
            continue
        prefix = os.path.commonprefix([v.name for v in vars_]).rstrip("/")
        if "/" in prefix and not prefix.endswith("/"):
            prefix = prefix.rsplit("/", 1)[0]
        interesting.append((i, prefix))

    # Install hooks on interesting layers.
    originals = []
    for i, prefix in interesting:
        # Safe label for safetensors keys.
        label = prefix.replace("/", ".")
        originals.append((i, hook(i, label)))

    print(f"hooked {len(interesting)} layers; re-running dec(embed) ...")
    with tf.device("/cpu:0"):
        _ = dec(embed_b).numpy()
    print(f"captured {len(captures)} tensors")
    for k in sorted(captures):
        arr = captures[k]
        print(f"  {k:55s} shape={arr.shape}, dtype={arr.dtype}")

    # Save intermediates as safetensors for the Rust side.
    save_safetensors(captures, str(out_dir / "intermediates.safetensors"))
    print(f"\nSaved to {out_dir}/intermediates.safetensors")


if __name__ == "__main__":
    main()
