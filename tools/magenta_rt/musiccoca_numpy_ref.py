"""NumPy reference for MusicCoCa text encoder; verifies against the SavedModel.

Uses the args-N → role mapping we extracted:
  - per-layer: 64-79 (alphabetical order within each block)
  - text token embed: 27
  - attn pool projections: 14, 17, 19, 21 + biases 13, 18, 20
  - pool query: 24
  - misc LN biases: 16, 22, 62

Unknowns to brute-force (small search space):
  - 4 LN params among args 68-71 (which is pre-attn-scale/bias, pre-mlp-scale/bias)
  - Q/K/V/O assignment for attn pool projections (4! = 24 options)
  - Contrastive projection location and shape
"""
import os, struct, json, re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import itertools
from pathlib import Path
import numpy as np


SM = Path("/x/Hub/models--google--magenta-realtime/snapshots/c05f8d6d608afd588469b7a8ef0929d5a1f8f6bb/savedmodels/musiccoca_mv212f_cpu_novocab")
WTS = "/x/Code/meganeura/magenta_rt_codec_dump/weights_musiccoca.safetensors"


def load_safetensors(p):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
        raw = f.read()
    out = {}
    dtypes = {'F32': np.float32, 'I32': np.int32}
    for k, info in h.items():
        if k.startswith('__'): continue
        if info['dtype'] not in dtypes: continue
        s, e = info['data_offsets']
        sh = info['shape'] if info['shape'] else [1]
        out[k] = np.frombuffer(raw[s:e], dtype=dtypes[info['dtype']]).reshape(sh).copy()
    return out


def load_args():
    raw = load_safetensors(WTS)
    args = {}
    for k, v in raw.items():
        m_ = re.match(r"musiccoca\.tf_var_leaves\.(\d+)\.", k)
        if m_:
            args[int(m_.group(1))] = v
    return args


def layer_norm(x, scale, bias, eps=1e-6, scale_offset=1.0):
    """Apply LayerNorm: mean-variance normalize last axis, then (scale+offset) + bias.
    flaxformer convention initializes scale to 0 with implicit +1.0 offset."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * (scale + scale_offset) + bias


def rms_norm(x, scale, eps=1e-6):
    """T5LayerNorm: RMS only, no mean subtraction, scale only (no bias)."""
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return x / rms * scale


def gelu(x):
    """tanh approximation of GeLU (flaxformer default)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def text_encoder_layer(x, paddings, args, layer_idx,
                       pre_attn_scale_arg, pre_attn_bias_arg,
                       pre_mlp_scale_arg, pre_mlp_bias_arg,
                       q_arg=77, k_arg=73, v_arg=79, o_arg=75,
                       qb_arg=76, kb_arg=72, vb_arg=78, ob_arg=74):
    """Run one transformer layer (pre-norm)."""
    # Pre-attn LayerNorm (flaxformer +1.0 scale offset).
    pre_attn_scale = args[pre_attn_scale_arg][layer_idx]  # [768]
    pre_attn_bias = args[pre_attn_bias_arg][layer_idx]    # [768]
    h = layer_norm(x, pre_attn_scale, pre_attn_bias, scale_offset=1.0)

    # Attention K/O/Q/V mapping is parameterized for brute-force search.
    Wq = args[q_arg][layer_idx]   # [768, 12, 64]
    Wk = args[k_arg][layer_idx]
    Wv = args[v_arg][layer_idx]
    Wo = args[o_arg][layer_idx]
    bq = args[qb_arg][layer_idx]  # [12, 64]
    bk = args[kb_arg][layer_idx]
    bv = args[vb_arg][layer_idx]
    bo = args[ob_arg][layer_idx]  # [768]

    # h: [batch, seq, 768]
    # q = h @ Wq  → [batch, seq, 12, 64]
    q = np.einsum("bsd,dnh->bsnh", h, Wq) + bq
    k = np.einsum("bsd,dnh->bsnh", h, Wk) + bk
    v = np.einsum("bsd,dnh->bsnh", h, Wv) + bv

    # Attention scores [batch, heads, seq_q, seq_k]
    scale = 1.0 / np.sqrt(64.0)
    scores = np.einsum("bqnh,bknh->bnqk", q, k) * scale
    # Mask: paddings broadcasting. paddings shape [batch, seq]: 1 = pad.
    mask = (paddings[:, None, None, :] > 0.5).astype(np.float32) * -1e9
    scores = scores + mask
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)

    # Output [batch, seq, 12, 64]
    o_heads = np.einsum("bnqk,bknh->bqnh", attn, v)
    # Project to [batch, seq, 768]
    o = np.einsum("bsnh,dnh->bsd", o_heads, Wo) + bo
    # Residual
    x = x + o

    # Pre-mlp LN (flaxformer +1.0 scale offset).
    pre_mlp_scale = args[pre_mlp_scale_arg][layer_idx]
    pre_mlp_bias = args[pre_mlp_bias_arg][layer_idx]
    h = layer_norm(x, pre_mlp_scale, pre_mlp_bias, scale_offset=1.0)

    # MLP: wi (65) + bias (64) → gelu → wo (67) + bias (66)
    Wi = args[65][layer_idx]   # [768, 3072]
    bi = args[64][layer_idx]   # [3072]
    Wo_mlp = args[67][layer_idx]  # [3072, 768]
    bo_mlp = args[66][layer_idx]  # [768]

    h2 = h @ Wi + bi
    h2 = gelu(h2)
    h2 = h2 @ Wo_mlp + bo_mlp

    # Residual
    return x + h2


def embed_text_numpy(ids, paddings, args,
                     pre_attn_scale_arg=71, pre_attn_bias_arg=70,
                     pre_mlp_scale_arg=69, pre_mlp_bias_arg=68,
                     q_arg=77, k_arg=73, v_arg=79, o_arg=75,
                     qb_arg=76, kb_arg=72, vb_arg=78, ob_arg=74):
    """Text encoder forward. ids: [batch, seq=128] int. paddings: [batch, seq]."""
    # Token embedding: arg 27 [768, 64000]. To use as lookup, transpose.
    embed_table = args[27].T  # [64000, 768]
    x = embed_table[ids]  # [batch, seq, 768]
    batch, seq, dim = x.shape
    # CoCa/T5 scale by sqrt(d) — confirmed necessary by ablation.
    x = x * np.sqrt(dim)
    # Sinusoidal PE: CONCATENATED [sin(...), cos(...)] (NOT interleaved!)
    # Confirmed via TFLite op[28] CONCATENATION(sin, cos) → [1, 1, 768].
    pos = np.arange(seq)
    inv_freq = 1.0 / np.power(10000.0, np.arange(0, dim // 2) * 2.0 / dim)  # [dim/2]
    angles = pos[:, None] * inv_freq[None, :]  # [seq, dim/2]
    pe = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)  # [seq, dim]
    x = x + pe[None].astype(np.float32)

    # 12-layer transformer.
    for li in range(12):
        x = text_encoder_layer(x, paddings, args, li,
                               pre_attn_scale_arg, pre_attn_bias_arg,
                               pre_mlp_scale_arg, pre_mlp_bias_arg,
                               q_arg=q_arg, k_arg=k_arg, v_arg=v_arg, o_arg=o_arg,
                               qb_arg=qb_arg, kb_arg=kb_arg, vb_arg=vb_arg, ob_arg=ob_arg)

    # Final LN at end of embed_text (per TFLite trace): scale = arg23 + 1.0, bias = arg22.

    # Attention pooling.
    # Pool query arg 24 [1, 768], single learned token attending over seq.
    # Alphabetical (K < O < Q < V):
    #   args[13]=K.bias, args[14]=K.kernel,
    #   args[16]=O.bias, args[17]=O.kernel,
    #   args[18]=Q.bias, args[19]=Q.kernel,
    #   args[20]=V.bias, args[21]=V.kernel
    # Confirmed by TFLite einsum signature on arg 17 (ABNH,DNH->ABD = OUT projection).
    pool_q = args[24]   # [1, 768]
    pool_Wk = args[14]
    pool_Wo = args[17]
    pool_Wq = args[19]
    pool_Wv = args[21]
    pool_bk = args[13]
    pool_bo = args[16]
    pool_bq = args[18]
    pool_bv = args[20]

    # Query: [batch=1, 1, 768] @ Wq → [batch, 1, 12, 256]
    pool_q_in = np.broadcast_to(pool_q, (batch, 1, 768))
    q = np.einsum("bsd,dnh->bsnh", pool_q_in, pool_Wq) + pool_bq
    k = np.einsum("bsd,dnh->bsnh", x, pool_Wk) + pool_bk
    v = np.einsum("bsd,dnh->bsnh", x, pool_Wv) + pool_bv

    scale = 1.0 / np.sqrt(256.0)
    scores = np.einsum("bqnh,bknh->bnqk", q, k) * scale
    mask = (paddings[:, None, None, :] > 0.5).astype(np.float32) * -1e9
    scores = scores + mask
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)
    o_heads = np.einsum("bnqk,bknh->bqnh", attn, v)

    # Project back to 768 via pool_Wo [768, 12, 256] (einsum ABNH,DNH->ABD).
    pool_out = np.einsum("bqnh,dnh->bqd", o_heads, pool_Wo) + pool_bo
    pool_emb = pool_out[:, 0, :]
    # Final LN at the end (TFLite confirms): scale = arg23 + 1.0, bias = arg22.
    pool_emb = layer_norm(pool_emb, args[23], args[22])
    return pool_emb


def main():
    import tensorflow as tf
    args = load_args()
    print(f"Loaded {len(args)} args")

    # Load SavedModel for oracle.
    print(f"Loading SavedModel...")
    with tf.device("/cpu:0"):
        mc = tf.saved_model.load(str(SM))
    embed_text_sig = mc.signatures["embed_text"]

    # Test input: SOS=1 at position 0, then 0s, paddings 1 from position 1 onward.
    batch = 1
    seq = 128
    ids = np.zeros((batch, seq), dtype=np.int32)
    ids[0, 0] = 1   # SOS
    paddings = np.ones((batch, seq), dtype=np.float32)
    paddings[0, 0] = 0.0  # only SOS is valid

    # Run SavedModel.
    with tf.device("/cpu:0"):
        result = embed_text_sig(inputs_0=tf.constant(ids), inputs_0_1=tf.constant(paddings))
    truth_raw = result["contrastive_txt_embed"].numpy()[0]   # [768]
    truth_l2 = result["contrastive_txt_embed_l2_normalized"].numpy()[0]
    print(f"Truth raw: range=[{truth_raw.min():.4f}, {truth_raw.max():.4f}]  norm={np.linalg.norm(truth_raw):.4f}")
    print(f"Truth L2:  range=[{truth_l2.min():.4f}, {truth_l2.max():.4f}]  norm={np.linalg.norm(truth_l2):.4f}")

    # Sweep all 24 orderings of attention K/O/Q/V kernels (args 73, 75, 77, 79).
    print(f"\n=== Sweep attention K/O/Q/V kernel ordering (24 permutations) ===")
    kernels = [73, 75, 77, 79]
    biases = [72, 74, 76, 78]
    # kernel→bias mapping: kernel 73 is paired with bias 72 (same alphabetical position),
    # so for any permutation of kernels, the bias permutation is shape-determined.
    best_cos = -2.0
    best_perm = None
    for perm_idx, perm in enumerate(itertools.permutations(range(4))):
        # perm[0]=K_index, perm[1]=O_index, perm[2]=Q_index, perm[3]=V_index
        k_arg = kernels[perm[0]]
        o_arg = kernels[perm[1]]
        q_arg = kernels[perm[2]]
        v_arg = kernels[perm[3]]
        # For each kernel, the matching bias is one of {72, 74, 76, 78}.
        # K, Q, V have biases [12, 12, 64]; O has bias [12, 768].
        # In the layer block: 72/76/78 are [12,12,64], 74 is [12,768]. So Wo's bias is 74.
        # K/Q/V biases come from {72, 76, 78} matching kernel positions.
        # Kernel at position 73 (KOQV index 0) pairs with bias at 72.
        # Kernel at 77 (idx 2) pairs with bias 76. Kernel at 79 (idx 3) pairs with 78.
        # Kernel at 75 (idx 1) is OUT — but OUT's bias is [12,768]=74, fixed.
        bias_for_kernel = {73: 72, 75: 74, 77: 76, 79: 78}
        kb = bias_for_kernel[k_arg]
        ob = bias_for_kernel[o_arg]  # this should be 74 if o_arg=75
        qb = bias_for_kernel[q_arg]
        vb = bias_for_kernel[v_arg]
        # But if o_arg != 75, the bias mapping is off. Skip those.
        if o_arg != 75:
            continue
        ours = embed_text_numpy(ids, paddings, args,
                                q_arg=q_arg, k_arg=k_arg, v_arg=v_arg, o_arg=o_arg,
                                qb_arg=qb, kb_arg=kb, vb_arg=vb, ob_arg=ob)[0]
        cos = np.dot(ours, truth_raw) / (np.linalg.norm(ours) * np.linalg.norm(truth_raw) + 1e-9)
        norm = np.linalg.norm(ours)
        if cos > best_cos:
            best_cos = cos
            best_perm = (k_arg, o_arg, q_arg, v_arg)
        print(f"  KOQV=({k_arg},{o_arg},{q_arg},{v_arg})  cos={cos:+.4f}  norm={norm:.4f}")
    print(f"\nBest KOQV: {best_perm}  cos={best_cos:.4f}")

    # Diagnostic.
    ours = embed_text_numpy(ids, paddings, args)[0]
    cos = np.dot(ours, truth_raw) / (np.linalg.norm(ours) * np.linalg.norm(truth_raw) + 1e-9)
    print(f"\nDefault: cos={cos:.4f} norm={np.linalg.norm(ours):.4f} truth_norm={np.linalg.norm(truth_raw):.4f}")

    # Sweep all 24 orderings of LN params 68-71 with KOQV fixed at best.
    print(f"\n=== Sweep LN ordering with corrected scale convention ===")
    best_cos2 = -2.0
    best_perm2 = None
    for perm in itertools.permutations([68, 69, 70, 71]):
        ours = embed_text_numpy(ids, paddings, args,
                                pre_attn_scale_arg=perm[0],
                                pre_attn_bias_arg=perm[1],
                                pre_mlp_scale_arg=perm[2],
                                pre_mlp_bias_arg=perm[3])[0]
        cos = np.dot(ours, truth_raw) / (np.linalg.norm(ours) * np.linalg.norm(truth_raw) + 1e-9)
        if cos > best_cos2:
            best_cos2 = cos
            best_perm2 = perm
    print(f"Best LN perm: {best_perm2}  cos={best_cos2:.6f}")


if __name__ == "__main__":
    main()
