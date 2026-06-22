#!/usr/bin/env python3
"""NumPy reference for the Magenta-RT LLM (T5 base config).

Architecture (per the dumped HF checkpoint llm_base_x4286_c1860k):
- Encoder: 12 bidirectional pre-norm Transformer layers
  - GeGLU MLP (wi_0 + wi_1 → gate, multiply, → wo)
  - Self-attention (no rel-pos bias)
  - RMSNorm scale-only (T5LayerNorm)
- Token embedding: shared [vocab=29824, embed=768]
- Position embedding: sinusoidal absolute (not in checkpoint, computed)

Embed dim 768, heads 12 × head_dim 64, MLP dim 2048.

Status: encoder forward pass only. Decoder (temporal + depth + sampling)
is TODO.

This doubles as the **real-weight numeric reference** for the Rust encoder:
run with `MEGANEURA_LLM_WEIGHTS=<weights_llm_base.safetensors>` (and optionally
`MEGANEURA_ENC_REF_OUT=<out.bin>` to write the encoder output as raw little-
endian f32 for `tests/llm_encoder_real_weight.rs`). The deterministic input is
`ids[i] = (i*101 + 7) % vocab`, seq from `MEGANEURA_ENC_REF_SEQ` (default 32) —
the Rust test uses the identical formula. A match cross-validates the weight
mapping (flat copy, no transpose) and the encoder op composition at real weight
magnitudes; it does **not** settle the position *scheme* (both sides add the
same sinusoidal PE — see LLM_FINDINGS.md).
"""
import json
import os
import struct
import re
from pathlib import Path
import numpy as np


# Weights path: env override, else the conventional dump location.
DUMP = Path(
    os.environ.get(
        "MEGANEURA_LLM_WEIGHTS",
        str(Path(__file__).resolve().parents[2] / "magenta_rt_codec_dump" / "weights_llm_base.safetensors"),
    )
)

# Deterministic reference input — must match tests/llm_encoder_real_weight.rs.
REF_SEQ = int(os.environ.get("MEGANEURA_ENC_REF_SEQ", "32"))
REF_VOCAB = 29824


def ref_input_ids(seq, vocab=REF_VOCAB):
    """The deterministic encoder input shared with the Rust gate."""
    return np.array([(i * 101 + 7) % vocab for i in range(seq)], dtype=np.int32)


def load_safetensors(p):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
        raw = f.read()
    out = {}
    dtypes = {'F32': np.float32, 'I32': np.int32}
    for k, info in h.items():
        if k.startswith('__'):
            continue
        if info['dtype'] not in dtypes:
            continue
        s, e = info['data_offsets']
        sh = info['shape'] if info['shape'] else [1]
        out[k] = np.frombuffer(raw[s:e], dtype=dtypes[info['dtype']]).reshape(sh).copy()
    return out


def rms_norm(x, scale, eps=1e-6):
    """T5LayerNorm: RMS normalization, no centering, no bias."""
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return x / rms * scale


def gelu_tanh(x):
    """T5 GeGLU uses tanh-approximated GeLU on the gate."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def geglu(gate, up):
    """GeGLU: GeLU(gate) * up."""
    return gelu_tanh(gate) * up


def sinusoidal_pos_embed(seq_len, embed_dim, min_scale=1.0, max_scale=10000.0):
    """Fixed sinusoidal absolute PE — a faithful port of flaxformer's
    `components.initializers.sinusoidal()` (what `embedding.FixedEmbed` uses, and
    what the v1 gin config wires onto the encoder). **Split-half** layout: first
    `embed_dim//2` columns are sines, next `embed_dim//2` are cosines."""
    pe = np.zeros((seq_len, embed_dim), dtype=np.float32)
    position = np.arange(0, seq_len)[:, None]
    half = embed_dim // 2
    scale_factor = -np.log(max_scale / min_scale) / (half - 1)
    div_term = min_scale * np.exp(np.arange(0, half) * scale_factor)
    pe[:, :half] = np.sin(position * div_term)
    pe[:, half:2 * half] = np.cos(position * div_term)
    return pe


def encoder_layer(x, weights, layer_idx, num_heads=12, head_dim=64, mlp_dim=2048,
                  eps=1e-6):
    """One T5 1.1 encoder layer: pre-norm self-attn + pre-norm GeGLU MLP, both with residual."""
    prefix = f'target.encoder.layers_{layer_idx}'
    embed = num_heads * head_dim

    # Pre-attention RMSNorm.
    ln1_scale = weights[f'{prefix}.pre_attention_layer_norm.scale']
    h = rms_norm(x, ln1_scale, eps)

    # Self-attention: Q, K, V projections.
    wq = weights[f'{prefix}.attention.query.kernel']  # [embed, embed]
    wk = weights[f'{prefix}.attention.key.kernel']
    wv = weights[f'{prefix}.attention.value.kernel']
    wo = weights[f'{prefix}.attention.out.kernel']

    q = (h @ wq).reshape(*h.shape[:-1], num_heads, head_dim)
    k = (h @ wk).reshape(*h.shape[:-1], num_heads, head_dim)
    v = (h @ wv).reshape(*h.shape[:-1], num_heads, head_dim)

    # Scaled dot-product attention.
    scale = 1.0 / np.sqrt(head_dim)
    # Shape: [batch, seq, heads, head_dim]
    scores = np.einsum('bqhd,bkhd->bhqk', q, k) * scale
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)
    out_heads = np.einsum('bhqk,bkhd->bqhd', attn, v)
    attn_out = out_heads.reshape(*h.shape[:-1], embed) @ wo

    x = x + attn_out

    # Pre-MLP RMSNorm + GeGLU FFN.
    ln2_scale = weights[f'{prefix}.pre_mlp_layer_norm.scale']
    h = rms_norm(x, ln2_scale, eps)
    wi_0 = weights[f'{prefix}.mlp.wi_0.kernel']  # gate
    wi_1 = weights[f'{prefix}.mlp.wi_1.kernel']  # up
    wo_mlp = weights[f'{prefix}.mlp.wo.kernel']
    gate = h @ wi_0
    up = h @ wi_1
    ffn_out = geglu(gate, up) @ wo_mlp

    return x + ffn_out


def encoder_forward(ids, weights, num_layers=12, embed_dim=768, num_heads=12,
                    head_dim=64, mlp_dim=2048, eps=1e-6):
    """LLM encoder forward pass.
    ids: [batch, seq] int32 token ids.
    Returns: [batch, seq, embed_dim] encoder hidden states."""
    # Token embedding lookup.
    token_embed_table = weights['target.token_embedder.embedding']  # [vocab, embed]
    x = token_embed_table[ids]  # [batch, seq, embed]

    # Add sinusoidal position embedding.
    batch, seq, dim = x.shape
    pe = sinusoidal_pos_embed(seq, dim)
    x = x + pe[None]

    # 12 transformer layers.
    for layer_idx in range(num_layers):
        x = encoder_layer(x, weights, layer_idx, num_heads, head_dim, mlp_dim, eps)

    # Final encoder norm.
    final_scale = weights['target.encoder.encoder_norm.scale']
    x = rms_norm(x, final_scale, eps)
    return x


def main():
    print(f"Loading LLM weights from {DUMP} ...")
    weights = load_safetensors(DUMP)
    print(f"  {len(weights)} tensors")

    # Deterministic reference input (matches the Rust gate). The LLM vocab is
    # unified across SpectroStream codec tokens + MusicCoCa style tokens, so any
    # int in [0, 29824) is a valid token id.
    ids = ref_input_ids(REF_SEQ)[None]  # [1, seq]
    print(f"\nReference input: seq={REF_SEQ}, ids[:8]={ids[0, :8].tolist()}")

    out = encoder_forward(ids, weights)  # [1, seq, embed]
    print(f"Encoder output: shape={out.shape}  range=[{out.min():.4f}, {out.max():.4f}]")
    print(f"  rms={np.sqrt((out ** 2).mean()):.4f}")

    ref_out = os.environ.get("MEGANEURA_ENC_REF_OUT")
    if ref_out:
        out[0].astype("<f4").tofile(ref_out)
        print(f"Wrote encoder reference ({out[0].size} f32) to {ref_out}")


if __name__ == '__main__':
    main()
