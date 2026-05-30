#!/usr/bin/env python3
"""Naive SpectroStream decoder reimplementation.

Loads the dumped safetensors weights + reference tokens, runs a from-scratch
decoder forward pass using TF ops for the heavy lifting (Conv2D / Conv2DTranspose),
and compares the output waveform against `reconstructed_audio` from the official
codec.

The goal isn't speed — it's a layer-by-layer ground-truth implementation we
can port to meganeura. Each call site prints the running shape so we can
bisect any mismatch.

Run:
  nix-shell tools/magenta_rt/shell.nix --run "python tools/magenta_rt/naive_decoder.py"
"""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from safetensors import safe_open  # noqa: E402

DUMP = Path("magenta_rt_codec_dump")

# Per-block spec: (block_name, conv_transpose_kernel_size "kHxkW", stride_w, has_shortcut_conv1x1)
# Derived from manifest + trace. stride_h is always 1 (time dim isn't upsampled
# per block — total time upsample comes from interpreting the 2D spectrogram).
DECODER_BLOCKS = [
    ("decoder_0", "4x3", 2, True),
    ("decoder_1", "4x4", 2, True),
    ("decoder_2", "3x4", 2, False),
    ("decoder_3", "3x4", 2, False),
    ("decoder_4", "3x6", 3, True),
    ("decoder_5", "3x4", 2, False),
    ("decoder_6", "3x4", 2, True),
]


def load_weights():
    w = {}
    with safe_open(str(DUMP / "weights_spectrostream.safetensors"), framework="numpy") as f:
        for k in f.keys():
            w[k] = f.get_tensor(k)
    return w


def load_reference():
    r = {}
    with safe_open(str(DUMP / "reference_codec.safetensors"), framework="numpy") as f:
        for k in f.keys():
            r[k] = f.get_tensor(k)
    return r


def W(weights, name):
    """Fetch the post-weight-norm 'rescaled' kernel (the actually-used one) and bias."""
    k = weights[name + ".rescaled.kernel"]
    b = weights[name.replace(".weight_norm", "") + ".weight_norm.bias"]
    return k, b


def elu(x):
    return tf.nn.elu(x)


def conv2d_valid(x, kernel, bias, stride_h=1, stride_w=1):
    y = tf.nn.conv2d(x, kernel, strides=[1, stride_h, stride_w, 1], padding="VALID")
    return tf.nn.bias_add(y, bias)


def conv2d_transpose_valid(x, kernel, bias, stride_h=1, stride_w=1):
    """x: NHWC [N, H_in, W_in, C_in]. kernel: [kH, kW, C_out, C_in] (TF Conv2DTranspose layout)."""
    in_shape = tf.shape(x)
    kH, kW = kernel.shape[0], kernel.shape[1]
    C_out = kernel.shape[2]
    out_h = (in_shape[1] - 1) * stride_h + kH
    out_w = (in_shape[2] - 1) * stride_w + kW
    out_shape = tf.stack([in_shape[0], out_h, out_w, C_out])
    y = tf.nn.conv2d_transpose(
        x, kernel, output_shape=out_shape,
        strides=[1, stride_h, stride_w, 1], padding="VALID",
    )
    return tf.nn.bias_add(y, bias)


def freq_dim_pad(x, before=1, after=1):
    """Zero-pad the W (freq) axis by (before, after) elements."""
    return tf.pad(x, [[0, 0], [0, 0], [before, after], [0, 0]])


def crop_freq_dim(x, before=1, after=1):
    """Crop the W (freq) axis by (before, after)."""
    return x[:, :, before:x.shape[2] - after, :]


def upsample_freq(x, scale_w):
    """ResizeNearestNeighbor along W only (H scale = 1)."""
    H = tf.shape(x)[1]
    new_w = tf.shape(x)[2] * scale_w
    return tf.image.resize(x, [H, new_w], method="nearest")


def decoder_block(x, w, prefix, ksize, stride_w, has_shortcut_conv):
    """One residual upsampling block."""
    main = x
    # --- conv-transpose path ---
    main = elu(main)
    kt_k, kt_b = W(w, f"{prefix}.conv2dtranspose_{ksize}.weight_norm")
    main = conv2d_transpose_valid(main, kt_k, kt_b, stride_h=1, stride_w=stride_w)
    # crop_freq_dim: standard "VALID -> SAME" pattern crops kW-1 total
    # (so for kW=3 stride=2: input W=5 -> out W=11 -> crop 2/2 = (1,1) -> 9; for kW=4: crop 3 total)
    # We make this consistent with stride_w. Concrete crop comes from trace.
    kW = int(ksize.split("x")[1])
    crop_before, crop_after = kW // 2, kW - 1 - kW // 2
    main = crop_freq_dim(main, crop_before, crop_after)

    # --- conv2d_3x3 refinement ---
    main = elu(main)
    main = freq_dim_pad(main, 1, 1)  # 3x3 with VALID needs (1,1) freq pad to keep size
    c3_k, c3_b = W(w, f"{prefix}.conv2d_3x3.weight_norm")
    main = conv2d_valid(main, c3_k, c3_b)

    # --- shortcut path ---
    short = x
    if has_shortcut_conv:
        sc_k, sc_b = W(w, f"{prefix}.shortcut.conv1x1.weight_norm")
        short = conv2d_valid(short, sc_k, sc_b)
    short = upsample_freq(short, scale_w=stride_w)
    # Sizes may differ slightly between main and short due to crop choices —
    # truncate short to match main's W.
    main_w = main.shape[2] if main.shape[2] is not None else tf.shape(main)[2]
    short = short[:, :, :main_w, :]
    return main + short


def input_layer(x, w):
    """input_layer:
      conv1x1_first: gated expand 256 → 2560 via (conv1x1, conv1x1_a, conv1x1_b)
      reshape 2560 = 5 * 512 → [B, T, F=5, C=512]
      residual conv2d_3x3 block
    """
    # Gated conv1x1 expansion. Order from trace:
    #   a_branch = activation(x) -> conv1x1_a    [256 -> 2560]
    #   m_branch = activation(x) -> conv1x1      [256 -> 2560]   (these come first)
    #   b_branch = activation(a_branch) -> conv1x1_b   [2560 -> 2560]
    #   merged = m_branch + b_branch
    a_post_act = elu(x)
    k, b = W(w, "decoder.input_layer.conv1x1_first.conv1x1_a.weight_norm")
    a = conv2d_valid(a_post_act, k, b)
    k, b = W(w, "decoder.input_layer.conv1x1_first.conv1x1.weight_norm")
    m = conv2d_valid(a_post_act, k, b)  # activation is shared since conv1x1_b/activation precedes conv1x1
    b_post_act = elu(a)
    k, b = W(w, "decoder.input_layer.conv1x1_first.conv1x1_b.weight_norm")
    bp = conv2d_valid(b_post_act, k, b)
    merged = m + bp  # tentative

    # Reshape from [B, T, 1, 2560] → [B, T, 5, 512]
    B, T = merged.shape[0], merged.shape[1]
    merged = tf.reshape(merged, [B if B is not None else -1, T if T is not None else -1, 5, 512])

    # Residual conv2d_3x3 block in input_layer
    main = elu(merged)
    main = freq_dim_pad(main, 1, 1)
    k, b = W(w, "decoder.input_layer.conv2d_3x3_a.weight_norm")
    main = conv2d_valid(main, k, b)
    main = elu(main)
    main = freq_dim_pad(main, 1, 1)
    k, b = W(w, "decoder.input_layer.conv2d_3x3.weight_norm")
    main = conv2d_valid(main, k, b)
    # shortcut is identity (no conv1x1 because channels match)
    return merged + main


def base_conv_last(x, w):
    """Final 7x7 conv: 64 → 2 channels (this is NOT weight_norm — bare conv)."""
    x = elu(x)
    x = freq_dim_pad(x, 3, 3)  # 7x7 with VALID
    k = w["decoder.input_layer.base_conv_last.conv.kernel"]
    b = w["decoder.input_layer.base_conv_last.conv.bias"]
    y = tf.nn.conv2d(x, k, strides=[1, 1, 1, 1], padding="VALID")
    return tf.nn.bias_add(y, b)


def dequantize(tokens, codebooks, levels):
    """tokens: [S, K] int; codebooks: [K_full, V, D]. Sum the first `levels` codebook entries."""
    S, K = tokens.shape
    K_use = min(K, levels)
    D = codebooks.shape[2]
    out = np.zeros((S, D), dtype=np.float32)
    for k in range(K_use):
        out += codebooks[k, tokens[:, k]]
    return out


def main():
    print(f"Loading weights from {DUMP}/")
    w = load_weights()
    refs = load_reference()
    print(f"  loaded {len(w)} tensors")

    tokens = refs["tokens"]  # [S, K=64]
    print(f"tokens shape: {tokens.shape}, dtype: {tokens.dtype}")

    codebooks = w["quantizer.rvq_codebooks"]
    print(f"codebooks shape: {codebooks.shape}")
    embed = dequantize(tokens, codebooks, levels=tokens.shape[1])
    print(f"dequantized embedding shape: {embed.shape}")

    # NHWC: [B, T, F=1, C=256]
    x = embed[None, :, None, :].astype(np.float32)
    x = tf.constant(x)
    print(f"\nDecoder input NHWC: {x.shape}")

    # NOTE: temporal_padding is part of streaming inference. For non-streaming
    # we tentatively skip it and crop at the end. If shapes don't line up we
    # add temporal_padding back.

    x = input_layer(x, w)
    print(f"after input_layer:   {x.shape}")

    for prefix, ksize, stride_w, has_sc in DECODER_BLOCKS:
        x = decoder_block(x, w, f"decoder.{prefix}", ksize, stride_w, has_sc)
        print(f"after {prefix}:        {x.shape}")

    x = base_conv_last(x, w)
    print(f"after base_conv_last: {x.shape}")
    # Expected: [1, T_final, F_final, 2]. F_final is the STFT freq bin count.
    # iSTFT will collapse (T_final, F_final, 2_re_im) → audio.

    # iSTFT — best guess at params: frame_length=1920, frame_step=1920//2=960, Hann window.
    # The 2 channels are (re, im) of a stereo-merged spectrogram. Stereo decoding:
    # we'd expect 2 separate iSTFTs but trace only showed 1 — defer until shapes work.
    spec = x.numpy()[0]  # [T_final, F_final, 2]
    print(f"\nSpectrogram for iSTFT: {spec.shape}")
    print(f"Reference reconstructed audio shape: {refs['reconstructed_audio'].shape}")

    # iSTFT attempt — placeholder, likely needs adjustment after shapes are known.
    re_im = spec[..., 0] + 1j * spec[..., 1]  # [T, F] complex
    try:
        import scipy.signal as ss
        _, audio = ss.istft(re_im.T, fs=48000, nperseg=1920, noverlap=1920 - 480, window="hann")
        print(f"iSTFT output: {audio.shape}")
    except Exception as e:
        print(f"iSTFT failed: {e}")
        return

    # Compare to reference (truncate to common length)
    target = refs["reconstructed_audio"]
    n = min(target.shape[0], audio.shape[0])
    err_l2 = np.sqrt(np.mean((audio[:n] - target[:n, 0]) ** 2))
    rms_t = np.sqrt(np.mean(target[:n] ** 2))
    print(f"\nRMS error vs ref (channel 0): {err_l2:.6f}, ref RMS: {rms_t:.6f}, ratio: {err_l2/rms_t:.3f}")


if __name__ == "__main__":
    main()
