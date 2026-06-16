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
"""
import json
import struct
import re
from pathlib import Path
import numpy as np


DUMP = Path('/x/Code/meganeura/magenta_rt_codec_dump')


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


def sinusoidal_pos_embed(seq_len, embed_dim):
    """Fixed sinusoidal absolute position embedding, standard T5 convention.
    Returns [seq_len, embed_dim] with interleaved sin/cos."""
    pos = np.arange(seq_len)
    inv_freq = 1.0 / np.power(10000.0, np.arange(0, embed_dim, 2) / embed_dim)
    sin_pos = np.sin(pos[:, None] * inv_freq[None, :])
    cos_pos = np.cos(pos[:, None] * inv_freq[None, :])
    pe = np.zeros((seq_len, embed_dim), dtype=np.float32)
    pe[:, 0::2] = sin_pos
    pe[:, 1::2] = cos_pos
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
    print("Loading LLM weights...")
    weights = load_safetensors(DUMP / 'weights_llm_base.safetensors')
    print(f"  {len(weights)} tensors")

    # Quick test: encoder with a dummy 16-token input.
    # The LLM vocab is unified across SpectroStream codec tokens + MusicCoCa
    # style tokens, so any int in [0, 29824) is a valid token id.
    ids = np.zeros((1, 16), dtype=np.int32)
    ids[0, 0] = 1   # SOS-like
    ids[0, 1] = 12345  # arbitrary
    ids[0, 15] = 7    # arbitrary
    print(f"\nTest input: ids={ids[0].tolist()}")

    out = encoder_forward(ids, weights)
    print(f"Encoder output: shape={out.shape}  range=[{out.min():.4f}, {out.max():.4f}]")
    print(f"  rms={np.sqrt((out ** 2).mean()):.4f}")
    print("\nTODO: decoder (temporal + depth) + autoregressive sampling")


if __name__ == '__main__':
    main()
