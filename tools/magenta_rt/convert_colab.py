"""Magenta-RT weight extraction for meganeura. Run cell-by-cell in Colab on a v2-8 TPU runtime.

Produces magenta_rt_export.zip containing:
  weights_spectrostream.safetensors  SpectroStream encoder + decoder + RVQ codebooks
  weights_musiccoca.safetensors      MusicCoCa text encoder + RVQ codebooks
  weights_llm.safetensors            Encoder-decoder Depthformer LLM (large)
  manifest.json                      All variable names, shapes, dtypes per model
  reference.safetensors              Intermediates from one fixed inference run
  reference.wav                      The 2s audio output (for ear check)
  reference_metadata.json            Prompt, seed, sampling params, shape contract

Cells are delimited with "# %%" so the file is jupytext-convertible.
"""

# %% [markdown]
# # Magenta-RT export for meganeura
#
# Open this notebook in Colab on a **v2-8 TPU** runtime (Runtime > Change runtime type).
# Run the install cell, **restart the session when prompted**, then continue.

# %%
# CELL 1: install magenta-realtime + tf-nightly. ~5 minutes. Colab will ask you to restart.
# After restart, do NOT re-run this cell — jump to CELL 2.
#
# NOTES:
# - magenta-rt's [tpu] extras pin `t5x[tpu]`, but t5x on PyPI is yanked 0.0.0 — install
#   t5x from git first, then install magenta-rt WITHOUT extras to skip re-resolution.
# - `jax[tpu]==0.8.1` is then installed separately so its outcome is visible.
# - Use Colab/IPython shell magic (!cmd) so pip output streams to the cell.
!git clone https://github.com/magenta/magenta-realtime.git
!pip install git+https://github.com/google-research/t5x.git
!pip install -e magenta-realtime/
!pip install "jax[tpu]==0.8.1" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
!pip uninstall -y tensorflow tf-nightly tensorflow-cpu tf-nightly-cpu tensorflow-tpu tf-nightly-tpu tensorflow-hub tf-hub-nightly tensorflow-text tensorflow-text-nightly
# Install nightlies in a SINGLE pip call so pip picks a consistent build date —
# tf-nightly and tensorflow-text-nightly ship matched C++ extensions and break
# (ModuleNotFoundError: tensorflow_text.core) if installed separately.
!pip install --upgrade --force-reinstall tf-nightly tensorflow-text-nightly tf-hub-nightly
!pip install safetensors
# Sanity-check the import works (uncomment if you want to verify before restart):
# import tensorflow_text; from tensorflow_text import core; print("tf-text OK")
print("Installed. RESTART THE SESSION (Runtime > Restart Session), then go to CELL 2.")

# %%
# CELL 2: Load Magenta-RT large. ~5 minutes (downloads checkpoints).
import json
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict
from safetensors.numpy import save_file as save_safetensors

from magenta_rt import system, audio, utils as mrt_utils

MRT = system.MagentaRT(tag="large", device="tpu:v2-8", lazy=False)
print("Loaded. SpectroStream sr=", MRT.codec.sample_rate, "MusicCoCa dim=", MRT.style_model.config.embedding_dim)

OUT = Path("magenta_rt_export")
OUT.mkdir(exist_ok=True)

# %%
# CELL 3: Helpers.

def _to_numpy(x):
    """Convert TF Variable / Tensor / jax Array to numpy, preserving dtype."""
    if hasattr(x, "numpy"):
        return x.numpy()
    if hasattr(x, "__array__"):
        return np.asarray(x)
    return np.array(x)


def dump_tf_module(module, prefix: str) -> dict:
    """Walk all variables in a TF Trackable/Module/SavedModel, return dict {name: ndarray}."""
    out = {}
    vars_iter = getattr(module, "variables", None) or getattr(module, "trainable_variables", [])
    for v in vars_iter:
        # Variable names look like "module/dense/kernel:0". Normalize to dots, drop trailing :N.
        raw = v.name.split(":", 1)[0]
        name = raw.replace("/", ".")
        arr = _to_numpy(v)
        out[f"{prefix}.{name}"] = arr
    return out


def dump_jax_params(params, prefix: str) -> dict:
    """Flatten a (frozen) dict tree of jax Arrays into {dotted.name: ndarray}."""
    flat = flatten_dict(jax.tree_util.tree_map(lambda x: np.asarray(x), dict(params)))
    return {f"{prefix}." + ".".join(map(str, k)): v for k, v in flat.items()}


def write_manifest(weights_by_model: dict[str, dict]) -> dict:
    manifest = {}
    for model_name, weights in weights_by_model.items():
        manifest[model_name] = sorted(
            [
                {"name": n, "shape": list(a.shape), "dtype": str(a.dtype), "bytes": int(a.nbytes)}
                for n, a in weights.items()
            ],
            key=lambda d: d["name"],
        )
    return manifest


# %%
# CELL 4: Extract SpectroStream (encoder, decoder, quantizer RVQ codebooks).
# The codec's SavedModels are accessed via MRT.codec but SpectroStreamJAX wraps them.
# The underlying SavedModelBase exposes _encoder/_decoder which are TF objects.
# We force load by referencing them.
codec = MRT.codec
# Force underlying TF models to load (parent class accessor).
from magenta_rt import spectrostream as ss
ss_tf = ss.SpectroStreamSavedModel(lazy=False)  # forces TF load
print("SS encoder vars:", len(ss_tf._encoder.variables))
print("SS decoder vars:", len(ss_tf._decoder.variables))
print("SS quantizer rvq_codebooks shape:", ss_tf.rvq_codebooks.shape)

ss_weights = {}
ss_weights.update(dump_tf_module(ss_tf._encoder, "encoder"))
ss_weights.update(dump_tf_module(ss_tf._decoder, "decoder"))
# All 64 RVQ codebooks (we'll slice to 16 in meganeura).
ss_weights["quantizer.rvq_codebooks"] = ss_tf.rvq_codebooks  # (64, 1024, 256) fp32
print(f"Total SpectroStream tensors: {len(ss_weights)}")

save_safetensors(ss_weights, str(OUT / "weights_spectrostream.safetensors"))

# %%
# CELL 5: Extract MusicCoCa (text encoder + RVQ codebooks). Also save the SentencePiece vocab.
import shutil

mc = MRT.style_model  # MusicCoCaV212F
_ = mc._encoder  # force load
_ = mc.rvq_codebooks  # force load

mc_weights = dump_tf_module(mc._encoder, "encoder")
mc_weights["quantizer.rvq_codebooks"] = mc.rvq_codebooks  # (12, 1024, 768) fp32
print(f"Total MusicCoCa tensors: {len(mc_weights)}")
print("Signatures:", list(mc._encoder.signatures.keys()))

save_safetensors(mc_weights, str(OUT / "weights_musiccoca.safetensors"))

# Copy the sentencepiece tokenizer alongside (needed for text encoding).
from magenta_rt import asset
vocab_path = asset.fetch(mc._vocab_path)
shutil.copy(vocab_path, OUT / "musiccoca_vocab.model")
print("Copied vocab:", (OUT / "musiccoca_vocab.model").stat().st_size, "bytes")

# %%
# CELL 6: Extract LLM (T5X Depthformer).
# MRT._llm is a compiled infer_fn; the params live on the InteractiveModel's train_state.
# We trigger the lazy load by touching _llm, then go through InteractiveModel.
_ = MRT._llm
# Re-create the interactive model handle to get train_state cleanly.
from magenta_rt.depthformer import model as dfmodel
checkpoint_dir = asset.fetch(f"checkpoints/llm_large_x3047_c1860k.tar", is_dir=True, extract_archive=True)
batch_size, num_partitions, model_parallel_submesh = MRT._device_params
_, _, interactive_model = dfmodel.load_pretrained_model(
    checkpoint_dir=checkpoint_dir,
    size="large",
    batch_size=batch_size,
    num_partitions=num_partitions,
    model_parallel_submesh=model_parallel_submesh,
)
params = interactive_model.train_state.params
llm_weights = dump_jax_params(params, "")
# Strip leading "."
llm_weights = {k.lstrip("."): v for k, v in llm_weights.items()}
print(f"Total LLM tensors: {len(llm_weights)}")
print("First 10 names:")
for k in sorted(llm_weights.keys())[:10]:
    print(f"  {k}  {llm_weights[k].shape}  {llm_weights[k].dtype}")
print("Total size (MB):", sum(v.nbytes for v in llm_weights.values()) / 1e6)

save_safetensors(llm_weights, str(OUT / "weights_llm.safetensors"))

# %%
# CELL 7: Write manifest.json (so meganeura side can plan name-mapping without re-running this).
manifest = write_manifest({
    "spectrostream": ss_weights,
    "musiccoca": mc_weights,
    "llm": llm_weights,
})
with open(OUT / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("manifest entries:", {k: len(v) for k, v in manifest.items()})

# %%
# CELL 8: Run reference inference + save all intermediates.
# Fixed prompt + seed for reproducibility. This is the oracle we compare against in meganeura.
PROMPT = "synthwave"
SEED = 0

# Style embedding + tokens
style_embedding = MRT.embed_style(PROMPT)          # (768,) fp32
style_tokens_full = MRT.style_model.tokenize(style_embedding)  # (12,) int32
style_tokens_llm_input = style_tokens_full[: MRT.config.encoder_style_rvq_depth]  # (6,)

# Build encoder inputs ourselves (mirrors system.py:617-680) so we can dump them.
state = MRT.init_state()
codec_tokens_lm = np.where(
    state.context_tokens >= 0,
    mrt_utils.rvq_to_llm(
        np.maximum(state.context_tokens, 0),
        MRT.config.codec_rvq_codebook_size,
        MRT.config.vocab_codec_offset,
    ),
    np.full_like(state.context_tokens, MRT.config.vocab_mask_token),
)
style_tokens_lm = mrt_utils.rvq_to_llm(
    style_tokens_llm_input,
    MRT.config.style_rvq_codebook_size,
    MRT.config.vocab_style_offset,
)
encoder_inputs_pos = np.concatenate(
    [
        codec_tokens_lm[:, : MRT.config.encoder_codec_rvq_depth].reshape(-1),
        style_tokens_lm,
    ],
    axis=0,
)
encoder_inputs_neg = encoder_inputs_pos.copy()
encoder_inputs_neg[-MRT.config.encoder_style_rvq_depth :] = MRT.config.vocab_mask_token
encoder_inputs = np.stack([encoder_inputs_pos, encoder_inputs_neg], axis=0)  # (2, 1006)

# Now actually run generate_chunk so we capture the LLM output too.
chunk, _ = MRT.generate_chunk(state=None, style=style_embedding, seed=SEED)

# To get the raw LLM output too, call the infer fn directly with the same inputs.
batch_size = MRT._device_params[0]
generated_tokens, _ = MRT._llm(
    {
        "encoder_input_tokens": encoder_inputs,
        "decoder_input_tokens": np.zeros(
            (batch_size, MRT.config.chunk_length_frames * MRT.config.decoder_codec_rvq_depth),
            dtype=np.int32,
        ),
    },
    {
        "max_decode_steps": np.array(
            MRT.config.chunk_length_frames * MRT.config.decoder_codec_rvq_depth, dtype=np.int32
        ),
        "guidance_weight": MRT._guidance_weight,
        "temperature": MRT._temperature,
        "topk": MRT._topk,
    },
    jax.random.PRNGKey(SEED),
)
generated_tokens = np.asarray(generated_tokens)  # (2, 800)
generated_rvq_tokens = mrt_utils.llm_to_rvq(
    generated_tokens[:1].reshape(MRT.config.chunk_length_frames, MRT.config.decoder_codec_rvq_depth),
    MRT.config.codec_rvq_codebook_size,
    MRT.config.vocab_codec_offset,
    safe=False,
)  # (50, 16) int32

# SpectroStream codec round-trip sanity (independent of LLM):
#   take the generated 2s audio, re-encode -> decode, compare. Tests codec only.
codec_roundtrip_tokens = MRT.codec.encode(chunk)           # (50, 64) int32
codec_roundtrip_audio = MRT.codec.decode(codec_roundtrip_tokens).samples  # (96000, 2)

ref = {
    "style_embedding": style_embedding.astype(np.float32),
    "style_tokens_full": style_tokens_full.astype(np.int32),
    "style_tokens_llm_input": style_tokens_llm_input.astype(np.int32),
    "encoder_inputs": encoder_inputs.astype(np.int32),
    "generated_tokens_raw": generated_tokens.astype(np.int32),
    "generated_rvq_tokens": generated_rvq_tokens.astype(np.int32),
    "chunk_audio": chunk.samples.astype(np.float32),
    "codec_roundtrip_tokens": codec_roundtrip_tokens.astype(np.int32),
    "codec_roundtrip_audio": codec_roundtrip_audio.astype(np.float32),
}
save_safetensors(ref, str(OUT / "reference.safetensors"))

# Write WAV for ear-checking.
import scipy.io.wavfile as wavfile
_wav_i16 = (np.clip(chunk.samples, -1.0, 1.0) * 32767.0).astype(np.int16)
wavfile.write(str(OUT / "reference.wav"), int(MRT.sample_rate), _wav_i16)

meta = {
    "prompt": PROMPT,
    "seed": SEED,
    "sample_rate": int(MRT.sample_rate),
    "num_channels": int(MRT.num_channels),
    "chunk_length_seconds": MRT.config.chunk_length,
    "chunk_length_frames": MRT.config.chunk_length_frames,
    "chunk_length_samples": MRT.config.chunk_length_samples,
    "decoder_codec_rvq_depth": MRT.config.decoder_codec_rvq_depth,
    "encoder_codec_rvq_depth": MRT.config.encoder_codec_rvq_depth,
    "encoder_style_rvq_depth": MRT.config.encoder_style_rvq_depth,
    "codec_rvq_codebook_size": MRT.config.codec_rvq_codebook_size,
    "vocab_size_pretrained": MRT.config.vocab_size_pretrained,
    "vocab_pad_token": MRT.config.vocab_pad_token,
    "vocab_mask_token": MRT.config.vocab_mask_token,
    "vocab_codec_offset": MRT.config.vocab_codec_offset,
    "vocab_style_offset": MRT.config.vocab_style_offset,
    "guidance_weight": MRT._guidance_weight,
    "temperature": MRT._temperature,
    "topk": MRT._topk,
    "tag": MRT._tag,
}
with open(OUT / "reference_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Reference saved.")

# %%
# CELL 9: Zip everything and download (or copy to Drive for the large LLM file).
zip_path = "magenta_rt_export.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
    for p in sorted(OUT.iterdir()):
        zf.write(p, p.name)
        print(f"  zipped {p.name}  {p.stat().st_size/1e6:.1f} MB")
zip_gb = Path(zip_path).stat().st_size / 1e9
print(f"\nTotal zip: {zip_gb:.2f} GB")

# files.download chokes above ~1GB. For the large export, mount Drive and copy.
if zip_gb > 1.0:
    from google.colab import drive  # noqa
    drive.mount("/content/drive")
    target = Path("/content/drive/MyDrive") / zip_path
    shutil.copy(zip_path, target)
    print(f"Copied to {target}")
else:
    from google.colab import files  # noqa
    files.download(zip_path)
