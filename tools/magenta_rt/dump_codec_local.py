#!/usr/bin/env python3
"""Dump SpectroStream codec weights + reference round-trip — local CPU, stable TF.

What this does:
  1. Downloads `savedmodels/ssv2_48k_stereo/{encoder,decoder,quantizer}` from
     `google/magenta-realtime` on HuggingFace Hub.
  2. Loads the three TF SavedModels on CPU (no GPU/TPU needed).
  3. Dumps all variables to `weights_spectrostream.safetensors` + `manifest.json`
     for meganeura's weight loader.
  4. Synthesizes a 2-second test signal (A-minor chord, stereo, with envelope),
     encodes via the official codec to tokens, decodes back to audio, and saves
     both as reference for end-to-end validation of meganeura's SpectroStream
     decoder.

Outputs (in ./magenta_rt_codec_dump/):
  weights_spectrostream.safetensors   encoder + decoder + RVQ codebooks
  manifest.json                       all variable names + shapes + dtypes
  reference_codec.safetensors         input_audio, tokens, reconstructed_audio
  input.wav                           the test signal (for ear-check)
  reconstructed.wav                   the codec round-trip output

Run:
  pip install tensorflow huggingface_hub safetensors numpy scipy
  python tools/magenta_rt/dump_codec_local.py

Tested with stable `tensorflow==2.17.x`. If you hit
"`tensorflow_text.core` not found" or similar, the SavedModel uses ops only in
tf-nightly — fall back to a fresh venv with `pip install tf-nightly`.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

# Quiet TF startup noise.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU
import tensorflow as tf  # noqa: E402

from huggingface_hub import snapshot_download  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402

REPO = "google/magenta-realtime"
SAMPLE_RATE = 48000
RVQ_DEPTH = 64
EMBED_DIM = 256
CODEBOOK_SIZE = 1024


def download_codec(cache_dir: str | None) -> str:
    print(f"[1/5] Downloading SpectroStream from {REPO} ...")
    root = snapshot_download(
        REPO,
        allow_patterns=["savedmodels/ssv2_48k_stereo/**"],
        cache_dir=cache_dir,
    )
    print(f"      cached at {root}")
    return root


def load_codec(root: str):
    print("[2/5] Loading TF SavedModels (CPU) ...")
    base = Path(root) / "savedmodels" / "ssv2_48k_stereo"
    with tf.device("/cpu:0"):
        enc = tf.saved_model.load(str(base / "encoder"))
        dec = tf.saved_model.load(str(base / "decoder"))
        qnt = tf.saved_model.load(str(base / "quantizer"))
    print(f"      encoder vars: {len(enc.variables)}, "
          f"decoder vars: {len(dec.variables)}, "
          f"quantizer.quantizers: {len(qnt._quantizers)}")
    return enc, dec, qnt


def dump_variables(module, prefix: str) -> dict[str, np.ndarray]:
    out = {}
    for v in module.variables:
        raw = v.name.split(":", 1)[0]
        name = raw.replace("/", ".")
        arr = np.asarray(v.numpy())
        # Skip non-weight scalars (e.g. bool training flags) that safetensors
        # can't serialize; weights are all float arrays.
        if arr.dtype == np.bool_ or arr.ndim == 0:
            print(f"      skip non-weight var: {prefix}.{name} ({arr.dtype}, shape {arr.shape})")
            continue
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        out[f"{prefix}.{name}"] = np.ascontiguousarray(arr)
    return out


def extract_rvq_codebooks(qnt) -> np.ndarray:
    """Walk the quantizer's internal RVQ codebooks. Matches `SpectroStreamSavedModel._rvq_codebooks`."""
    result = np.zeros((RVQ_DEPTH, CODEBOOK_SIZE, EMBED_DIM), dtype=np.float32)
    for i in range(RVQ_DEPTH):
        var = qnt._quantizers[i].embeddings
        result[i] = var.numpy().T  # stored as [D, V], we want [V, D]
    return result


def synthesize_test_audio() -> np.ndarray:
    """2 seconds stereo @ 48kHz: A-minor chord with fade-in/out envelope."""
    n = 2 * SAMPLE_RATE
    t = np.linspace(0, 2.0, n, dtype=np.float32)
    audio = np.zeros((n, 2), dtype=np.float32)
    for f in (220.0, 261.63, 329.63):  # A3, C4, E4
        audio[:, 0] += 0.1 * np.sin(2 * np.pi * f * t)
        audio[:, 1] += 0.1 * np.sin(2 * np.pi * f * t + 0.05)  # slight detune for stereo
    env = np.minimum(t * 4.0, 1.0) * np.minimum((2.0 - t) * 4.0, 1.0)
    audio *= env[:, None]
    return audio


def rvq_dequantize(tokens: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    """tokens: [S, K] int → embed: [S, D]. Sum the K codebook entries."""
    S, K = tokens.shape
    D = codebooks.shape[2]
    out = np.zeros((S, D), dtype=np.float32)
    for k in range(K):
        out += codebooks[k, tokens[:, k]]
    return out


def codec_round_trip(audio: np.ndarray, enc, dec, qnt, rvq_codebooks: np.ndarray):
    """Encode then decode `audio` via the official codec. Returns (tokens, reconstructed)."""
    audio_b = audio[None].astype(np.float32)  # [1, T, 2]
    with tf.device("/cpu:0"):
        embed = enc(audio_b).numpy()  # [1, S, D]
        tokens_tf, _ = qnt.inference_encoding_with_tf_function(embed, num_quantizers=RVQ_DEPTH)
        tokens = tf.transpose(tokens_tf, (1, 2, 0)).numpy()  # [1, S, 64]
        # Dequantize using the same codebooks we dumped (sanity: matches official).
        embed_back = rvq_dequantize(tokens[0], rvq_codebooks)
        # Decoder takes [B, S, D] → [B, T, 2]
        reconstructed = dec(embed_back[None].astype(np.float32)).numpy()[0]
    return tokens[0].astype(np.int32), reconstructed.astype(np.float32)


def write_wav(path: Path, sr: int, audio_f32: np.ndarray):
    """Write a 16-bit PCM WAV. Avoids scipy dep — uses tf.audio.encode_wav."""
    audio = np.clip(audio_f32, -1.0, 1.0)
    enc = tf.audio.encode_wav(tf.constant(audio, dtype=tf.float32), sample_rate=sr)
    path.write_bytes(enc.numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="magenta_rt_codec_dump", help="output directory")
    ap.add_argument("--cache-dir", default=None, help="HuggingFace cache override")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = download_codec(args.cache_dir)
    enc, dec, qnt = load_codec(root)

    print("[3/5] Dumping weights ...")
    weights = {}
    weights.update(dump_variables(enc, "encoder"))
    weights.update(dump_variables(dec, "decoder"))
    rvq = extract_rvq_codebooks(qnt)
    weights["quantizer.rvq_codebooks"] = rvq
    total_bytes = sum(a.nbytes for a in weights.values())
    print(f"      {len(weights)} tensors, {total_bytes / 1e6:.1f} MB")
    save_file(weights, str(out_dir / "weights_spectrostream.safetensors"))

    manifest = sorted(
        ({"name": n, "shape": list(a.shape), "dtype": str(a.dtype), "bytes": int(a.nbytes)}
         for n, a in weights.items()),
        key=lambda d: d["name"],
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("[4/5] Synthesizing test audio + codec round-trip ...")
    audio = synthesize_test_audio()
    tokens, reconstructed = codec_round_trip(audio, enc, dec, qnt, rvq)
    print(f"      tokens shape: {tokens.shape} (frames × rvq_depth)")
    print(f"      reconstructed shape: {reconstructed.shape}")

    # Truncate to common length (the codec may pad).
    n = min(audio.shape[0], reconstructed.shape[0])
    ref = {
        "input_audio": audio[:n].astype(np.float32),
        "tokens": tokens.astype(np.int32),
        "reconstructed_audio": reconstructed[:n].astype(np.float32),
    }
    save_file(ref, str(out_dir / "reference_codec.safetensors"))

    print("[5/5] Writing WAVs for ear-check ...")
    write_wav(out_dir / "input.wav", SAMPLE_RATE, audio[:n])
    write_wav(out_dir / "reconstructed.wav", SAMPLE_RATE, reconstructed[:n])

    # Tiny metadata for meganeura side.
    (out_dir / "reference_metadata.json").write_text(json.dumps({
        "sample_rate": SAMPLE_RATE,
        "num_channels": 2,
        "rvq_depth": RVQ_DEPTH,
        "embedding_dim": EMBED_DIM,
        "codebook_size": CODEBOOK_SIZE,
        "num_frames": int(tokens.shape[0]),
        "test_signal": "A-minor chord (220, 261.63, 329.63 Hz), 2 sec stereo, fade-in/out",
    }, indent=2))

    print(f"\nDone. Output in {out_dir}/")
    print(f"  Listen to input.wav vs reconstructed.wav to verify the official codec is OK.")
    print(f"  meganeura will then need to match reconstructed.wav given tokens from reference_codec.safetensors.")


if __name__ == "__main__":
    main()
