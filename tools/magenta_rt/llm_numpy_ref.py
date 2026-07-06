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


# ============================ DECODER REFERENCE ============================
#
# Faithful NumPy port of the v1 Depthformer decoder, mirroring
# `magenta_rt/depthformer/modules.py` (initial commit b35a850) and the
# flaxformer T5 components it composes. The gate interface is the flat
# `decoder_input_tokens` [T*Q] sequence the decoder actually receives (the
# upstream shift_right / feature conversion is identical for any consumer, so
# comparing at this interface isolates the decoder forward itself).


def relative_position_bucket(relative_position, bidirectional, num_buckets,
                             max_distance):
    """Port of flaxformer `RelativePositionBiases._relative_position_bucket`."""
    ret = np.zeros_like(relative_position)
    n = -relative_position
    if bidirectional:
        num_buckets //= 2
        ret += (n < 0).astype(np.int32) * num_buckets
        n = np.abs(n)
    else:
        n = np.maximum(n, 0)
    max_exact = num_buckets // 2
    is_small = n < max_exact
    val_if_large = max_exact + (
        np.log(n.astype(np.float32) / max_exact + np.finfo(np.float32).eps)
        / np.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).astype(np.int32)
    val_if_large = np.minimum(val_if_large, num_buckets - 1)
    ret += np.where(is_small, n, val_if_large)
    return ret


def relpos_bias(table, qlen, klen, bidirectional, max_distance):
    """Build the [heads, qlen, klen] additive bias from a [heads, buckets] table.

    relative_position = memory(k) - context(q); bucket then gather (flaxformer
    `RelativePositionBiases.__call__`)."""
    num_heads, num_buckets = table.shape
    context = np.arange(qlen)[:, None]
    memory = np.arange(klen)[None, :]
    rp = memory - context  # (qlen, klen)
    bucket = relative_position_bucket(rp, bidirectional, num_buckets, max_distance)
    # table[h, bucket[q,k]] -> [heads, qlen, klen]
    return table[:, bucket]


def attention(x_q, x_kv, wq, wk, wv, wo, num_heads, head_dim, causal,
              bias=None):
    """Multi-head attention. x_q [Sq, D], x_kv [Sk, D]. Scores scaled by
    1/sqrt(head_dim) — matching flaxformer's dot_product_attention and the
    encoder reference (which validated against real weights to 9.6e-7)."""
    sq = x_q.shape[0]
    sk = x_kv.shape[0]
    embed = num_heads * head_dim
    scale = 1.0 / np.sqrt(head_dim)
    q = (x_q @ wq).reshape(sq, num_heads, head_dim)
    k = (x_kv @ wk).reshape(sk, num_heads, head_dim)
    v = (x_kv @ wv).reshape(sk, num_heads, head_dim)
    scores = np.einsum('qhd,khd->hqk', q, k) * scale
    if bias is not None:
        scores = scores + bias  # [heads, qlen, klen]
    if causal:
        mask = np.tril(np.ones((sq, sk), dtype=bool))
        scores = np.where(mask[None], scores, -1e10)
    scores = scores - scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn = attn / attn.sum(axis=-1, keepdims=True)
    out = np.einsum('hqk,khd->qhd', attn, v).reshape(sq, embed)
    return out @ wo


def temporal_layer(x, enc, w, prefix, tbias, cfg):
    """One temporal decoder layer: causal self-attn (+rel-pos) + cross-attn + MLP."""
    nh, hd, eps = cfg['heads'], cfg['head_dim'], cfg['eps']
    g = lambda n: w[f'{prefix}.{n}']
    h = rms_norm(x, g('pre_self_attention_layer_norm.scale'), eps)
    x = x + attention(h, h, g('self_attention.query.kernel'),
                      g('self_attention.key.kernel'), g('self_attention.value.kernel'),
                      g('self_attention.out.kernel'), nh, hd, causal=True, bias=tbias)
    h = rms_norm(x, g('pre_cross_attention_layer_norm.scale'), eps)
    x = x + attention(h, enc, g('encoder_decoder_attention.query.kernel'),
                      g('encoder_decoder_attention.key.kernel'),
                      g('encoder_decoder_attention.value.kernel'),
                      g('encoder_decoder_attention.out.kernel'), nh, hd, causal=False)
    h = rms_norm(x, g('pre_mlp_layer_norm.scale'), eps)
    x = x + geglu(h @ g('mlp.wi_0.kernel'), h @ g('mlp.wi_1.kernel')) @ g('mlp.wo.kernel')
    return x


def depth_layer(x, w, prefix, dbias, cfg):
    """One depth decoder layer: causal self-attn (+rel-pos) + MLP, no cross-attn."""
    nh, hd, eps = cfg['heads'], cfg['head_dim'], cfg['eps']
    g = lambda n: w[f'{prefix}.{n}']
    h = rms_norm(x, g('pre_self_attention_layer_norm.scale'), eps)
    x = x + attention(h, h, g('self_attention.query.kernel'),
                      g('self_attention.key.kernel'), g('self_attention.value.kernel'),
                      g('self_attention.out.kernel'), nh, hd, causal=True, bias=dbias)
    h = rms_norm(x, g('pre_mlp_layer_norm.scale'), eps)
    x = x + geglu(h @ g('mlp.wi_0.kernel'), h @ g('mlp.wi_1.kernel')) @ g('mlp.wo.kernel')
    return x


def decoder_forward(decoder_input_tokens, encoder_out, weights, cfg,
                    add_position=True):
    """Faithful Depthformer decoder: decoder_input_tokens [T*Q] + encoder_out
    [enc_seq, D] -> logits [T*Q, vocab]. `add_position` toggles the FixedEmbed
    absolute PE (to quantify its effect)."""
    Q = cfg['num_levels']
    embed = cfg['embed']
    eps = cfg['eps']
    tq = decoder_input_tokens.shape[0]
    T = tq // Q

    # Base Decoder embed: token embedding + FixedEmbed sinusoidal PE (added).
    table = weights['target.token_embedder.embedding']
    embedded = table[decoder_input_tokens]  # [T*Q, D]
    if add_position:
        embedded = embedded + sinusoidal_pos_embed(tq, embed)

    # --- Temporal: _to_temporal_embedded_inputs (edge-pad (Q-1,1), reshape,
    #     drop last frame) then mean over Q. ---
    padded = np.pad(embedded, ((Q - 1, 1), (0, 0)), mode='edge')  # [(T+1)*Q, D]
    temporal_in = padded.reshape(T + 1, Q, embed)[:-1].mean(axis=1)  # [T, D]

    tbias = relpos_bias(
        weights['target.decoder.decoder.temporal_decoder.relpos_bias.rel_embedding'],
        T, T, bidirectional=False, max_distance=cfg['temp_max_dist'])
    x = temporal_in
    for i in range(cfg['num_temporal_layers']):
        x = temporal_layer(
            x, encoder_out, weights,
            f'target.decoder.decoder.temporal_decoder.layers_{i}', tbias, cfg)
    temporal_context = x  # [T, D]

    # --- Depth: _to_depth_embedded_inputs (concat temporal level-0 with the
    #     first Q-1 level embeddings of each frame). ---
    depth_in = np.concatenate(
        [temporal_context[:, None, :], padded.reshape(T + 1, Q, embed)[1:, :-1, :]],
        axis=1)  # [T, Q, D]

    dbias = relpos_bias(
        weights['target.decoder.decoder.depth_decoder.relpos_bias_depth.rel_embedding'],
        Q, Q, bidirectional=False, max_distance=cfg['depth_max_dist'])

    norm_scale = weights['target.decoder.decoder_norm.scale']
    logits_w = weights['target.decoder.logits_dense.kernel']
    logits = np.zeros((T, Q, logits_w.shape[1]), dtype=np.float32)
    for t in range(T):
        xd = depth_in[t]  # [Q, D]
        for i in range(cfg['num_depth_layers']):
            xd = depth_layer(
                xd, weights,
                f'target.decoder.decoder.depth_decoder.depth_layers_{i}', dbias, cfg)
        xd = rms_norm(xd, norm_scale, eps)
        logits[t] = xd @ logits_w
    return logits.reshape(T * Q, logits_w.shape[1])


BASE_CFG = dict(embed=768, heads=12, head_dim=64, mlp=2048, eps=1e-6,
                num_levels=16, num_temporal_layers=20, num_depth_layers=4,
                temp_max_dist=128, depth_max_dist=16)


def ref_decoder_input_tokens(tq, vocab=REF_VOCAB):
    """Deterministic decoder input grid — must match the Rust decoder gate."""
    return np.array([(i * 53 + 11) % vocab for i in range(tq)], dtype=np.int32)


def ref_encoder_out(enc_seq, embed):
    """Deterministic encoder output — must match the Rust decoder gate."""
    out = np.zeros((enc_seq, embed), dtype=np.float32)
    for i in range(enc_seq):
        for j in range(embed):
            out[i, j] = 0.1 * np.sin(0.7 * i + 0.013 * j)
    return out


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

    # Decoder reference (small grid for a tractable Rust GPU gate).
    dec_frames = int(os.environ.get("MEGANEURA_DEC_REF_FRAMES", "3"))
    Q = BASE_CFG['num_levels']
    enc_seq = int(os.environ.get("MEGANEURA_DEC_REF_ENCSEQ", "5"))
    dec_in = ref_decoder_input_tokens(dec_frames * Q)
    enc = ref_encoder_out(enc_seq, BASE_CFG['embed'])
    logits = decoder_forward(dec_in, enc, weights, BASE_CFG, add_position=True)
    print(f"\nDecoder ref: frames={dec_frames} Q={Q} -> logits {logits.shape}, "
          f"range=[{logits.min():.3f}, {logits.max():.3f}]")
    # Quantify the absolute-PE effect.
    no_pe = decoder_forward(dec_in, enc, weights, BASE_CFG, add_position=False)
    print(f"  abs-PE effect on logits: max|with-without| = "
          f"{np.abs(logits - no_pe).max():.3f}, argmax changes = "
          f"{int((logits.argmax(-1) != no_pe.argmax(-1)).sum())}/{logits.shape[0]}")
    dec_ref = os.environ.get("MEGANEURA_DEC_REF_OUT")
    if dec_ref:
        logits.astype("<f4").tofile(dec_ref)
        enc.astype("<f4").tofile(dec_ref + ".enc")
        print(f"Wrote decoder reference ({logits.size} f32) to {dec_ref}")


if __name__ == '__main__':
    main()
