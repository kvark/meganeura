#!/usr/bin/env python3
"""SpectroStream decoder reference v2 - corrected per ARCH_FINDINGS.md.

Key differences from v1:
1. conv2d_3x3 is CAUSAL in H: pad [2, 0] H + pad [1, 1] W + VALID 3x3 conv.
2. base_conv_last is CAUSAL in H: pad [6, 0] H + VALID 7x7 conv.
3. decoder_0 conv-T stride (2, 1); decoder_1 (2, 2); others as before.
4. conv-T post: trim H to H_in*stride_h causal (from END), then crop_freq_dim
   strips W [1, 1] (begin=1, end=-1).
5. input_layer parallel paths: conv1x1_b(ELU(conv1x1_a(x))) + conv1x1(x).

Works in NHWC throughout (matches TF; avoids transpose noise).
"""
import json
import os
import struct
from pathlib import Path
import numpy as np

DUMP_DIR = Path('/x/Code/meganeura/magenta_rt_codec_dump')

NUM_FRAMES = int(os.environ.get('NUM_FRAMES', 50))
TEMPORAL_PAD = 1   # pad [0, 1] on T (added at END)
INITIAL_CHANNELS = 512
INITIAL_FREQ_BINS = 5
EMBED_DIM = 256
CODEBOOK_SIZE = 1024
MAX_RVQ_DEPTH = 64


# ---------- safetensors I/O ----------

def load_safetensors(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
        raw = f.read()
    out = {}
    dtypes = {'F32': np.float32, 'I32': np.int32, 'BOOL': np.bool_, 'I64': np.int64}
    for k, info in header.items():
        if k.startswith('__'):
            continue
        s, e = info['data_offsets']
        if info['dtype'] not in dtypes:
            continue
        dtype = dtypes[info['dtype']]
        shape = info['shape'] if info['shape'] else [1]
        out[k] = np.frombuffer(raw[s:e], dtype=dtype).reshape(shape).copy()
    return out


def save_safetensors(path, tensors):
    header = {}
    offset = 0
    parts = []
    for k, v in tensors.items():
        v = np.ascontiguousarray(v.astype(np.float32))
        nbytes = v.nbytes
        header[k] = {'dtype': 'F32', 'shape': list(v.shape),
                     'data_offsets': [offset, offset + nbytes]}
        parts.append(v.tobytes())
        offset += nbytes
    hb = json.dumps(header).encode()
    hb += b' ' * ((8 - len(hb) % 8) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(hb)))
        f.write(hb)
        for p in parts:
            f.write(p)


# ---------- convs (NHWC throughout) ----------

def conv2d_nhwc_valid(x, kernel_hwio, bias=None, stride=(1, 1)):
    """x: [B, H, W, C_in], kernel: [kH, kW, C_in, C_out]. VALID padding, given stride.
    Returns [B, H_out, W_out, C_out].
    Fast: im2col via stride_tricks + matmul.
    """
    B, H, W, C_in = x.shape
    kH, kW, kC_in, C_out = kernel_hwio.shape
    assert C_in == kC_in
    sh, sw = stride
    H_out = (H - kH) // sh + 1
    W_out = (W - kW) // sw + 1
    # Build sliding window view: [B, H_out, W_out, kH, kW, C_in].
    x_c = np.ascontiguousarray(x)
    bs, hs, ws, cs = x_c.strides
    from numpy.lib.stride_tricks import as_strided
    windows = as_strided(
        x_c,
        shape=(B, H_out, W_out, kH, kW, C_in),
        strides=(bs, hs * sh, ws * sw, hs, ws, cs),
        writeable=False,
    )
    # Reshape to [B * H_out * W_out, kH*kW*C_in] and kernel to [kH*kW*C_in, C_out].
    win_flat = windows.reshape(B * H_out * W_out, kH * kW * C_in)
    k_flat = kernel_hwio.reshape(kH * kW * C_in, C_out)
    out = (win_flat @ k_flat).reshape(B, H_out, W_out, C_out).astype(np.float32)
    if bias is not None:
        out += bias
    return out


def conv2d_transpose_nhwc_valid(x, kernel_hwoi, bias=None, stride=(1, 1)):
    """x: [B, H, W, C_in], kernel: [kH, kW, C_out, C_in] (TF Conv2DTranspose layout).
    Output H = (H-1)*sh + kH, W = (W-1)*sw + kW (no padding).
    Returns [B, H_out, W_out, C_out]."""
    B, H, W, C_in = x.shape
    kH, kW, C_out, kC_in = kernel_hwoi.shape
    assert C_in == kC_in
    sh, sw = stride
    H_d = H * sh - (sh - 1)
    W_d = W * sw - (sw - 1)
    dilated = np.zeros((B, H_d, W_d, C_in), dtype=np.float32)
    dilated[:, ::sh, ::sw, :] = x
    # Forward conv with stride 1 and full pad, kernel flipped + transposed.
    kernel_fwd = kernel_hwoi[::-1, ::-1].transpose(0, 1, 3, 2)  # [kH, kW, C_in, C_out]
    padded = np.pad(dilated, ((0, 0), (kH - 1, kH - 1), (kW - 1, kW - 1), (0, 0)))
    out = conv2d_nhwc_valid(padded, kernel_fwd, bias=None, stride=(1, 1))
    if bias is not None:
        out += bias
    return out


def causal_conv2d_3x3(x, kernel_hwio, bias):
    """conv2d_3x3 as TF implements it:
      1. freq_dim_pad: pad W [1, 1] (NHWC axis 2)
      2. internal Pad: pad H [2, 0] (causal, axis 1)
      3. Conv2D VALID 3x3
    Net effect: H preserved (causal), W preserved (SAME).
    """
    # H pad causal [2, 0], W pad [1, 1].
    padded = np.pad(x, ((0, 0), (2, 0), (1, 1), (0, 0)))
    return conv2d_nhwc_valid(padded, kernel_hwio, bias, stride=(1, 1))


def causal_conv_77(x, kernel_hwio, bias):
    """base_conv_last:
      1. freq_dim_pad: pad W [3, 3] (SAME-like in W)
      2. internal Pad: pad H [6, 0] (causal)
      3. Conv2D VALID 7x7
    Net: H and W both preserved.
    """
    padded = np.pad(x, ((0, 0), (6, 0), (3, 3), (0, 0)))
    return conv2d_nhwc_valid(padded, kernel_hwio, bias, stride=(1, 1))


def elu(x):
    return np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1.0).astype(np.float32)


def rvq_dequantize(tokens, codebooks, depth):
    out = np.zeros((tokens.shape[0], codebooks.shape[2]), dtype=np.float32)
    for k in range(depth):
        out += codebooks[k, tokens[:, k]]
    return out


# ---------- decoder ----------

# Corrected per ARCH_FINDINGS.md: decoder_0 stride (2, 1), decoder_1 (2, 2)
DECODER_BLOCKS = [
    # name, kt_h, kt_w, stride_h, stride_w, in_c, out_c, has_shortcut_conv, w_crop_end
    # decoder_4 uses crop_freq_dim end=-2 (TF verified). All others end=-1.
    ('decoder_0', 4, 3, 2, 1, 512,  1024, True,  -1),
    ('decoder_1', 4, 4, 2, 2, 512,  256,  True,  -1),
    ('decoder_2', 3, 4, 1, 2, 256,  256,  False, -1),
    ('decoder_3', 3, 4, 1, 2, 256,  256,  False, -1),
    ('decoder_4', 3, 6, 1, 3, 256,  128,  True,  -2),
    ('decoder_5', 3, 4, 1, 2, 128,  128,  False, -1),
    ('decoder_6', 3, 4, 1, 2, 128,  64,   True,  -1),
]


def conv_T_block(x, kt_kernel, kt_bias, stride_h, stride_w, w_crop_end=-1):
    """TF-style ConvTranspose:
      1. Conv2DBackpropInput VALID with given strides
      2. Internal slice: trim H to H_in * stride_h (causal — strip from END)
      3. crop_freq_dim: W [1:w_crop_end] (decoder_4 uses w_crop_end=-2)
    """
    B, H_in, W_in, _ = x.shape
    out = conv2d_transpose_nhwc_valid(x, kt_kernel, kt_bias, stride=(stride_h, stride_w))
    # Internal H trim (causal).
    H_target = H_in * stride_h
    out = out[:, :H_target, :, :]
    # crop_freq_dim on W.
    out = out[:, :, 1:w_crop_end, :]
    return out


def upsample_nearest_nhwc(x, factor_h, factor_w):
    return np.repeat(np.repeat(x, factor_h, axis=1), factor_w, axis=2)


def preprocess_input_v2(tokens, weights, num_frames, temporal_pad):
    """tokens [S, 64] → preprocessed NHWC input [1, S+pad, 5, 512] for decoder body.
    Equivalent to TF: temporal_padding → expand_dims to [1, S+pad, 1, 256] →
    input_layer.conv1x1_first → reshape to [1, S+pad, 5, 512].
    """
    codebooks = weights['quantizer.rvq_codebooks']
    embed = rvq_dequantize(tokens, codebooks, depth=MAX_RVQ_DEPTH)  # [S, 256]
    # temporal_padding: pad [0, 1] on T (add 1 frame at END).
    embed_t = np.pad(embed, ((0, temporal_pad), (0, 0)))  # [S+pad, 256]
    # Expand to NHWC: [1, S+pad, 1, 256].
    h = embed_t[None, :, None, :]

    # input_layer.conv1x1_first parallel paths:
    #   main_a = conv1x1_a(h)    [1, T, 1, 2560]
    #   main_b = conv1x1_b(ELU(main_a))   [1, T, 1, 2560]
    #   parallel = conv1x1(h)    [1, T, 1, 2560]
    #   out = main_b + parallel
    k_a = weights['decoder.input_layer.conv1x1_first.conv1x1_a.weight_norm.rescaled.kernel']
    b_a = weights['decoder.input_layer.conv1x1_first.conv1x1_a.weight_norm.bias']
    main_a = conv2d_nhwc_valid(h, k_a, b_a)  # 1x1 VALID = identity spatially
    # NOTE: TF activation/PartitionedCall before conv1x1_b — is it ELU?
    main_b_in = elu(main_a)
    k_b = weights['decoder.input_layer.conv1x1_first.conv1x1_b.weight_norm.rescaled.kernel']
    b_b = weights['decoder.input_layer.conv1x1_first.conv1x1_b.weight_norm.bias']
    main_b = conv2d_nhwc_valid(main_b_in, k_b, b_b)
    k_p = weights['decoder.input_layer.conv1x1_first.conv1x1.weight_norm.rescaled.kernel']
    b_p = weights['decoder.input_layer.conv1x1_first.conv1x1.weight_norm.bias']
    parallel = conv2d_nhwc_valid(h, k_p, b_p)
    summed = main_b + parallel  # [1, T, 1, 2560]
    # reshape_5: [1, T, 1, 2560] → [1, T, 5, 512].
    B, T, _, _ = summed.shape
    out = summed.reshape(B, T, INITIAL_FREQ_BINS, INITIAL_CHANNELS)
    return out


def decoder_forward_v2(preprocessed, weights, intermediates):
    """preprocessed [B, T_pad, 5, 512] NHWC → body output [1, T_body, 480, 4]."""
    intermediates['preprocessed_input'] = preprocessed
    x = preprocessed

    # input_layer residual block (causal conv2d_3x3 a/b).
    residual = x
    for sub in ['conv2d_3x3_a', 'conv2d_3x3']:
        h = elu(x)
        k = weights[f'decoder.input_layer.{sub}.weight_norm.rescaled.kernel']
        b = weights[f'decoder.input_layer.{sub}.weight_norm.bias']
        x = causal_conv2d_3x3(h, k, b)
    x = x + residual
    intermediates['stage_input_layer'] = x.copy()

    # 7 decoder blocks
    for idx, (name, kt_h, kt_w, sh, sw, in_c, out_c, has_shortcut, w_end) in enumerate(DECODER_BLOCKS):
        residual_block_input = x
        # Main: ELU → conv-T (sh, sw) → conv2d_3x3
        h = elu(x)
        kt = weights[f'decoder.{name}.conv2dtranspose_{kt_h}x{kt_w}.weight_norm.rescaled.kernel']
        ktb = weights[f'decoder.{name}.conv2dtranspose_{kt_h}x{kt_w}.weight_norm.bias']
        main = conv_T_block(h, kt, ktb, sh, sw, w_crop_end=w_end)
        if idx == 0:
            intermediates['block0_after_convT_crop'] = main.copy()
        main = elu(main)
        c3k = weights[f'decoder.{name}.conv2d_3x3.weight_norm.rescaled.kernel']
        c3b = weights[f'decoder.{name}.conv2d_3x3.weight_norm.bias']
        main = causal_conv2d_3x3(main, c3k, c3b)
        if idx == 0:
            intermediates['block0_main_after_3x3'] = main.copy()

        # Shortcut: conv1x1 (if has_shortcut) → upsample by (sh, sw).
        sc = residual_block_input
        if has_shortcut:
            sck = weights[f'decoder.{name}.shortcut.conv1x1.weight_norm.rescaled.kernel']
            scb = weights[f'decoder.{name}.shortcut.conv1x1.weight_norm.bias']
            sc = conv2d_nhwc_valid(sc, sck, scb)
        sc = upsample_nearest_nhwc(sc, sh, sw)
        if idx == 0:
            intermediates['block0_shortcut'] = sc.copy()

        if sc.shape != main.shape:
            print(f"  WARN block {name}: shape mismatch main={main.shape} sc={sc.shape}")
        x = main + sc

        # 2-way fold after decoder_0: split C in half → 2 sub-batches.
        # Verified by total-element / decoder_1.in_c=512 constraint.
        if idx == 0:
            B, T, W, C = x.shape
            x_r6 = x.reshape(B, T, W, 2, C // 2)
            x_t2 = x_r6.transpose(3, 0, 1, 2, 4)
            x = x_t2.reshape(-1, T, W, C // 2)

        intermediates[f'stage_block_{idx}'] = x.copy()

    # base_conv_last (causal 7x7).
    x = elu(x)
    bcl_k = weights['decoder.input_layer.base_conv_last.conv.kernel']
    bcl_b = weights['decoder.input_layer.base_conv_last.conv.bias']
    x = causal_conv_77(x, bcl_k, bcl_b)
    intermediates['stage_output'] = x.copy()
    return x


def main():
    print("Loading weights + tokens...")
    weights = load_safetensors(DUMP_DIR / 'weights_spectrostream.safetensors')

    use_tokens = '--tokens' in os.sys.argv or os.environ.get('USE_TOKENS', '0') == '1'

    if use_tokens:
        refs = load_safetensors(DUMP_DIR / 'reference_codec.safetensors')
        tokens = refs['tokens'].astype(np.int32)
    else:
        # Synthetic input: same as TF intermediates capture
        tokens = None

    if tokens is not None:
        preprocessed = preprocess_input_v2(tokens, weights, NUM_FRAMES, TEMPORAL_PAD)
    else:
        # Use TF's il_out + reshape to bypass conv1x1 paths (test body alone).
        tf_int = load_safetensors(DUMP_DIR / 'tf_intermediates.safetensors')
        il_out = tf_int['input_layer_out_embed_B_S_1_D']  # [1, 50, 1, 2560]
        # reshape_5 layout: TF stores conv1x1 outputs in NHWC [1, T, 1, 2560].
        # Reshape to [1, T, 5, 512] for body input.
        # NOTE: this only feeds the conv1x1_a output (not the full input_layer with
        # conv1x1_b + parallel conv1x1). Use --tokens to feed the full path.
        # Also pad T by [0, 1] for temporal_pad.
        B, T, _, _ = il_out.shape
        # OBS: conv1x1_a output may NOT be the same as full input_layer output.
        # Pad before reshape.
        padded = np.pad(il_out, ((0, 0), (0, TEMPORAL_PAD), (0, 0), (0, 0)))
        preprocessed = padded.reshape(B, T + TEMPORAL_PAD, INITIAL_FREQ_BINS, INITIAL_CHANNELS)

    print(f"preprocessed: shape={preprocessed.shape}  range=[{preprocessed.min():.3e}, {preprocessed.max():.3e}]")

    print("Running v2 decoder...")
    intermediates = {}
    out = decoder_forward_v2(preprocessed, weights, intermediates)

    print("\nStage ranges:")
    for k, v in intermediates.items():
        if v.dtype == np.float32:
            rms = np.sqrt((v.astype(np.float64) ** 2).mean())
            print(f"  {k:30s} shape={str(v.shape):30s} range=[{v.min():9.3e}, {v.max():9.3e}]  rms={rms:.3e}")

    out_path = DUMP_DIR / 'decoder_reference_v2.safetensors'
    save_safetensors(out_path, intermediates)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
