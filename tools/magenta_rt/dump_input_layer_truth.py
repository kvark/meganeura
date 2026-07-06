#!/usr/bin/env python3
"""Capture the TF SpectroStream decoder's actual intermediate tensors so we can
fix meganeura's host-side preprocessing.

Strategy:
  1. Load the decoder SavedModel.
  2. Walk dec.keras_api.layers to find the input_layer module.
  3. Wrap input_layer.__call__ so the input/output to it AND to its child
     `conv1x1_first` sub-module are captured.
  4. Run the decoder once with the reference tokens' dequantized embedding.
  5. Save captured tensors to safetensors.

The captured tensors then let us:
  * Verify meganeura's GPU-graph `preprocessed_input` (what we feed to InputLayer).
  * Fix the codebook→2560 host-side path by comparing to conv1x1_first's I/O.

Run:
  nix-shell tools/magenta_rt/shell.nix --run \
    "python tools/magenta_rt/dump_input_layer_truth.py"
"""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.numpy import save_file as save_safetensors  # noqa: E402


DUMP = Path("magenta_rt_codec_dump")


def find_input_layer(dec) -> tuple[object, object]:
    """Walk the decoder tree; return (input_layer_module, conv1x1_first_module)."""
    # Start from dec.keras_api.layers[0] (the decoder body).
    body = dec.keras_api.layers[0]
    # Body should have a .layers list with input_layer first.
    body_layers = getattr(body, "layers", None)
    if body_layers is None:
        body_layers = getattr(getattr(body, "keras_api", None), "layers", [])
    print(f"\nBody layers ({len(body_layers)} total):")
    for i, layer in enumerate(body_layers):
        vars_ = list(getattr(layer, "variables", []))
        names = [v.name for v in vars_][:3]
        print(f"  [{i}] type={type(layer).__name__} vars={len(vars_)} sample={names}")

    input_layer = None
    for layer in body_layers:
        vars_ = list(getattr(layer, "variables", []))
        for v in vars_:
            if "input_layer" in v.name:
                input_layer = layer
                break
        if input_layer is not None:
            break
    if input_layer is None:
        raise RuntimeError("could not find input_layer in body layers")
    print(f"input_layer: type={type(input_layer).__name__} vars={len(input_layer.variables)}")

    # Find conv1x1_first inside input_layer.
    cf = None
    inner_layers = getattr(input_layer, "layers", None) or \
                   getattr(getattr(input_layer, "keras_api", None), "layers", [])
    for layer in inner_layers:
        vars_ = list(getattr(layer, "variables", []))
        for v in vars_:
            if "conv1x1_first" in v.name:
                cf = layer
                break
        if cf is not None: break
    if cf is None:
        # Some Keras models hide layers in tracked attrs; try children.
        for name in dir(input_layer):
            if name.startswith("_"): continue
            try:
                obj = getattr(input_layer, name)
                ovars = list(getattr(obj, "variables", []))
                if ovars and any("conv1x1_first" in v.name for v in ovars):
                    cf = obj
                    break
            except Exception:
                pass
    if cf is None:
        print("WARN: conv1x1_first not found by direct walk — will only capture input_layer I/O")
    else:
        print(f"conv1x1_first: type={type(cf).__name__} vars={len(cf.variables)}")

    return input_layer, cf


def main():
    print("Loading decoder SavedModel ...")
    root = snapshot_download("google/magenta-realtime",
                             allow_patterns=["savedmodels/ssv2_48k_stereo/decoder/**"])
    with tf.device("/cpu:0"):
        dec = tf.saved_model.load(str(Path(root) / "savedmodels" / "ssv2_48k_stereo" / "decoder"))

    print("Locating input_layer ...")
    input_layer, conv1x1_first = find_input_layer(dec)
    body = dec.keras_api.layers[0]

    # Set up capture buffers.
    captured = {}

    # Load reference embed.
    print("Loading reference tokens ...")
    refs, weights = {}, {}
    with safe_open(str(DUMP / "reference_codec.safetensors"), framework="numpy") as f:
        for k in f.keys(): refs[k] = f.get_tensor(k)
    with safe_open(str(DUMP / "weights_spectrostream.safetensors"), framework="numpy") as f:
        for k in f.keys(): weights[k] = f.get_tensor(k)

    tokens = refs["tokens"]
    codebooks = weights["quantizer.rvq_codebooks"]
    embed = np.zeros((tokens.shape[0], codebooks.shape[2]), dtype=np.float32)
    for k in range(tokens.shape[1]):
        embed += codebooks[k, tokens[:, k]]
    embed_b = embed[None].astype(np.float32)
    print(f"embed shape: {embed_b.shape}")

    print("Running decoder (sanity check) ...")
    with tf.device("/cpu:0"):
        audio_out = dec(embed_b)
    print(f"audio_out shape: {audio_out.shape}")

    # Call input_layer directly with various candidate input shapes to figure out
    # what it expects. The body's overall input is the embed [1, S, 256], so the
    # input_layer probably accepts something similar but possibly with a freq dim
    # axis or with NHWC channels-last reshape.
    candidates = [
        ("embed_BSD",          embed_b),                          # [1, 50, 256]
        ("embed_B_S_1_D",      embed_b[:, :, None, :]),           # [1, 50, 1, 256] NHWC
    ]
    for name, candidate in candidates:
        print(f"Trying input_layer({name} shape={candidate.shape})")
        try:
            with tf.device("/cpu:0"):
                out = input_layer(tf.constant(candidate))
            arr = out.numpy() if hasattr(out, "numpy") else np.asarray(out)
            print(f"  → success! output shape: {arr.shape}")
            captured[f"input_layer_in_{name}"] = candidate
            captured[f"input_layer_out_{name}"] = arr
            break
        except Exception as e:
            msg = str(e)[:200]
            print(f"  → failed: {msg}")

    # Capture body output for comparison.
    print("\n=== body(embed) ===")
    try:
        with tf.device("/cpu:0"):
            body_out = body(tf.constant(embed_b.astype(np.float32)))
        arr = body_out.numpy() if hasattr(body_out, "numpy") else np.asarray(body_out)
        print(f"  body({embed_b.shape}) → {arr.shape}, dtype={arr.dtype}")
        print(f"  range [{arr.min():.4f}, {arr.max():.4f}] mean={arr.mean():.4f}")
        captured["body_out"] = arr.astype(np.float32)
    except Exception as e:
        print(f"  body failed: {str(e)[:200]}")

    # Walk body layers sequentially.
    print("\n=== walking body's layer-N attrs sequentially ===")
    x = tf.constant(embed_b.astype(np.float32))
    print(f"  start: {x.shape}")
    for i in range(40):
        attr_name = f"layer-{i}"
        layer = getattr(body, attr_name, None)
        if layer is None:
            print(f"  [{i:2d}] no body.{attr_name}, stop")
            break
        try:
            with tf.device("/cpu:0"):
                y = layer(x)
            sh = y.shape if hasattr(y, 'shape') else 'unknown'
            n_vars = len(getattr(layer, "variables", []))
            prefix = ""
            if n_vars > 0:
                vars_ = list(layer.variables)
                prefix = vars_[0].name.rsplit("/", 2)[0]
            print(f"  [{i:2d}] {sh}  vars={n_vars}  {prefix}")
            # Save intermediates we care about.
            if i in [4, 5, 6, 7, 14, 17, 18, 19]:
                captured[f"body_layer_{i:02d}_out"] = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
            if i > 25:
                break
            x = y
        except Exception as e:
            print(f"  [{i:2d}] FAILED: {str(e)[:150]}")
            # Try with verbose
            print(f"        layer type: {type(layer).__name__}, has __call__: {hasattr(layer, '__call__')}")
            break

    if conv1x1_first is not None:
        print("\nconv1x1_first variables (only directly-owned):")
        for v in conv1x1_first.variables:
            print(f"  {v.name}  shape={list(v.shape)}")
        print("\nconv1x1_first attrs:")
        for name in dir(conv1x1_first):
            if name.startswith("_"): continue
            try:
                v = getattr(conv1x1_first, name)
                if hasattr(v, "variables") or "conv" in name or hasattr(v, "__call__"):
                    nv = len(getattr(v, "variables", []))
                    print(f"  {name}: {type(v).__name__} vars={nv}")
            except Exception:
                pass
        print("\ninput_layer attrs:")
        for name in dir(input_layer):
            if name.startswith("_"): continue
            try:
                v = getattr(input_layer, name)
                if hasattr(v, "variables") or "conv" in name:
                    nv = len(getattr(v, "variables", []))
                    print(f"  {name}: {type(v).__name__} vars={nv}")
            except Exception:
                pass
        # Try calling conv1x1_first directly.
        # Likely input is post-ELU of [1, S, 1, 256].
        cf_candidates = [
            ("embed_B_S_1_D", embed_b[:, :, None, :]),
            ("elu_embed_B_S_1_D", np.where(embed_b[:, :, None, :] > 0, embed_b[:, :, None, :], np.exp(np.minimum(embed_b[:, :, None, :], 0)) - 1)),
            ("embed_BSD",     embed_b),
        ]
        for name, candidate in cf_candidates:
            try:
                with tf.device("/cpu:0"):
                    out = conv1x1_first(tf.constant(candidate.astype(np.float32)))
                arr = out.numpy() if hasattr(out, "numpy") else np.asarray(out)
                print(f"conv1x1_first({name} shape={candidate.shape}) → {arr.shape}")
                captured[f"cf_in_{name}"] = candidate.astype(np.float32)
                captured[f"cf_out_{name}"] = arr
                break
            except Exception as e:
                msg = str(e)[:200]
                print(f"conv1x1_first({name}) → failed: {msg}")

    # Compare end-to-end audio to reference.
    ref_audio = refs["reconstructed_audio"]
    audio_np = audio_out.numpy()[0] if audio_out.shape.rank == 3 else audio_out.numpy()
    n = min(audio_np.shape[0], ref_audio.shape[0])
    err = np.sqrt(np.mean((audio_np[:n] - ref_audio[:n]) ** 2))
    rms = np.sqrt(np.mean(ref_audio[:n] ** 2))
    print(f"end-to-end err / ref-rms = {err/rms:.6f}")

    # Save captures.
    out = {}
    for k, v in captured.items():
        if hasattr(v, "numpy"):
            arr = v.numpy()
        else:
            arr = np.asarray(v)
        out[k] = np.ascontiguousarray(arr.astype(np.float32))
        print(f"  {k}: {out[k].shape}")
    out["embed"] = embed_b.astype(np.float32)
    out["audio_tf"] = audio_np.astype(np.float32)

    save_safetensors(out, str(DUMP / "tf_intermediates.safetensors"))
    print(f"\nSaved to {DUMP/'tf_intermediates.safetensors'}")


if __name__ == "__main__":
    main()
