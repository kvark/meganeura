#!/usr/bin/env python3
"""NumPy reference implementation of the SpectroStream decoder.

Reads the dumped weights (`weights_spectrostream.safetensors`) and runs the
decoder forward pass layer-by-layer in NumPy. Saves intermediate activations
so meganeura's GPU implementation can be validated layer-by-layer.

Run inside `nix-shell -p python3Packages.numpy` (no TF needed).

Outputs:
  decoder_reference.safetensors
    preprocessed_input       [1, 512, 74, 5]    fed into meganeura GPU graph
    stage_input_layer        [1, 512, 70, 5]    after input_layer residual
    stage_block_0            [1, 512, 70, 80]   after decoder_0 + pixel_shuffle
    stage_block_1            [1, 256, 68, 160]  after decoder_1
    ...
    stage_block_6            [1, 64, 56, 1920]  after decoder_6
    stage_output             [1, 2, 50, 1920]   after base_conv_last (= audio)
    audio                    [96000, 2]         flattened/transposed for comparison
"""
import json
import os
import struct
from pathlib import Path
import numpy as np

DUMP_DIR = Path('/x/Code/meganeura/magenta_rt_codec_dump')

# Match SpectroStream config used by meganeura.
NUM_FRAMES = int(os.environ.get('NUM_FRAMES', 50))
TEMPORAL_PAD = 1
INITIAL_CHANNELS = 512
INITIAL_FREQ_BINS = 5
EMBED_DIM = 256
CODEBOOK_SIZE = 1024
MAX_RVQ_DEPTH = 64
LLM_RVQ_DEPTH = 16  # Magenta-RT LLM uses only the first 16 codebooks


# ---------- safetensors I/O ----------

def load_safetensors(path: Path) -> dict[str, np.ndarray]:
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
            continue  # skip unsupported dtype
        dtype = dtypes[info['dtype']]
        shape = info['shape'] if info['shape'] else [1]
        out[k] = np.frombuffer(raw[s:e], dtype=dtype).reshape(shape).copy()
    return out


def save_safetensors(path: Path, tensors: dict[str, np.ndarray]):
    header = {}
    offset = 0
    payload_parts = []
    for k, v in tensors.items():
        v = np.ascontiguousarray(v.astype(np.float32))
        nbytes = v.nbytes
        header[k] = {
            'dtype': 'F32',
            'shape': list(v.shape),
            'data_offsets': [offset, offset + nbytes],
        }
        payload_parts.append(v.tobytes())
        offset += nbytes
    header_bytes = json.dumps(header).encode()
    # Pad header to 8-byte alignment.
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b' ' * pad
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(header_bytes)))
        f.write(header_bytes)
        for part in payload_parts:
            f.write(part)


# ---------- weight access helpers ----------

class WeightNormConv:
    """conv kernel with weight-norm: kernel_normalized = g * v / ||v||_per_out_c.
    Both stored: `.kernel` (v) and `.rescaled.kernel` (already-normalized).
    Use `.rescaled.kernel` for forward.
    """
    def __init__(self, weights, prefix):
        self.kernel = weights[f'{prefix}.weight_norm.rescaled.kernel']
        self.bias = weights[f'{prefix}.weight_norm.bias']


def conv2d_nchw(x: np.ndarray, kernel_hwio: np.ndarray, bias: np.ndarray | None,
                stride: int = 1, padding=(0, 0)) -> np.ndarray:
    """x: [N, C_in, H, W], kernel: [kH, kW, C_in, C_out] (TF layout, NHWC kernel order).
    Returns [N, C_out, H_out, W_out]."""
    N, C_in, H, W = x.shape
    kH, kW, kC_in, C_out = kernel_hwio.shape
    assert C_in == kC_in, f"channel mismatch: x has {C_in}, kernel expects {kC_in}"
    pH, pW = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (pH, pH), (pW, pW)))
    H_out = (H + 2 * pH - kH) // stride + 1
    W_out = (W + 2 * pW - kW) // stride + 1
    out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
    # Transpose kernel to [C_out, C_in, kH, kW] for easier indexing.
    k = kernel_hwio.transpose(3, 2, 0, 1)
    for oh in range(H_out):
        for ow in range(W_out):
            ih = oh * stride
            iw = ow * stride
            patch = x_padded[:, :, ih:ih + kH, iw:iw + kW]  # [N, C_in, kH, kW]
            # einsum: oc,ic,kh,kw * N,ic,kh,kw → N,oc
            out[:, :, oh, ow] = np.einsum('oikm,nikm->no', k, patch)
    if bias is not None:
        out += bias[None, :, None, None]
    return out


def conv2d_transpose_nchw(x: np.ndarray, kernel_hwoi: np.ndarray, bias: np.ndarray | None,
                          stride=(1, 1)) -> np.ndarray:
    """ConvTranspose with TF VALID padding.
    x: [N, C_in, H, W]. kernel: [kH, kW, C_out, C_in] (TF Conv2DTranspose filter layout).
    Output H = (H-1)*stride_h + kH, W = (W-1)*stride_w + kW (no padding).
    Returns [N, C_out, H_out, W_out].
    Implemented as dilate+conv2d for clarity (matches meganeura's rewrite)."""
    N, C_in, H, W = x.shape
    kH, kW, C_out, kC_in = kernel_hwoi.shape
    assert C_in == kC_in
    sh, sw = stride
    H_out = (H - 1) * sh + kH
    W_out = (W - 1) * sw + kW
    # Dilate input with zeros: [N, C_in, H_d, W_d] where H_d = H*sh - (sh-1), W_d = W*sw - (sw-1).
    H_d = H * sh - (sh - 1)
    W_d = W * sw - (sw - 1)
    dilated = np.zeros((N, C_in, H_d, W_d), dtype=np.float32)
    dilated[:, :, ::sh, ::sw] = x
    # Pad by (kH-1, kW-1) on each side, then forward conv with stride 1, no padding,
    # with kernel transposed and spatially flipped.
    # kernel_hwoi[h, w, oc, ic] → kernel_fwd[h, w, ic, oc] flipped: kernel_fwd[h, w, ic, oc] = kernel_hwoi[kH-1-h, kW-1-w, oc, ic].
    kernel_fwd = kernel_hwoi[::-1, ::-1].transpose(0, 1, 3, 2)  # [kH, kW, ic, oc]
    out = conv2d_nchw(dilated, kernel_fwd, bias=None, stride=1, padding=(kH - 1, kW - 1))
    if bias is not None:
        out += bias[None, :, None, None]
    return out


def elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1.0).astype(np.float32)


def rvq_dequantize(tokens: np.ndarray, codebooks: np.ndarray, depth: int) -> np.ndarray:
    """tokens: [S, K], codebooks: [K, V, D] → embed: [S, D], summing first `depth` codebooks."""
    S, K = tokens.shape
    D = codebooks.shape[2]
    out = np.zeros((S, D), dtype=np.float32)
    for k in range(depth):
        out += codebooks[k, tokens[:, k]]
    return out


# ---------- decoder forward ----------

DECODER_BLOCKS = [
    # (name, kt_h, kt_w, stride_h, stride_w, in_c, mid_c, out_c, has_shortcut_conv)
    # ALL stride_h = 1. T doubling in the visible TF output comes from the
    # tail reshape_9 absorbing the 4-fold batch dim into the new T.
    ('decoder_0', 4, 3, 2, 2, 512, 1024, 1024, True),
    ('decoder_1', 4, 4, 1, 2, 512, 256, 256, True),
    ('decoder_2', 3, 4, 1, 2, 256, 256, 256, False),
    ('decoder_3', 3, 4, 1, 2, 256, 256, 256, False),
    ('decoder_4', 3, 6, 1, 3, 256, 128, 128, True),
    ('decoder_5', 3, 4, 1, 2, 128, 128, 128, False),
    ('decoder_6', 3, 4, 1, 2, 128, 64, 64, True),
]


def preprocess_input(tokens: np.ndarray, weights: dict, num_frames: int, temporal_pad: int):
    """tokens [S, 64] → preprocessed input [1, 512, S+pad, 5]."""
    # 1. Dequantize via first 16 codebooks (Magenta-RT LLM uses only 16 of 64).
    # NOTE: codec round-trip in dump_codec_local.py uses ALL 64. So we match that.
    codebooks = weights['quantizer.rvq_codebooks']  # [64, 1024, 256]
    embed = rvq_dequantize(tokens, codebooks, depth=MAX_RVQ_DEPTH)  # [S, 256]
    # 2. Input-layer expansion: 256 → 2560 via a SINGLE conv1x1.
    #   Verified via TF capture: TF's `input_layer(embed)` and `conv1x1_first(embed)`
    #   both equal `embed @ conv1x1_a.weight + conv1x1_a.bias`. The `conv1x1`
    #   and `conv1x1_b` weights in the safetensors file exist but are unused
    #   by the production decoder (likely artifacts of an alternative arch).
    k = weights['decoder.input_layer.conv1x1_first.conv1x1_a.weight_norm.rescaled.kernel'].reshape(-1, 2560)
    b = weights['decoder.input_layer.conv1x1_first.conv1x1_a.weight_norm.bias']
    out_2560 = embed @ k + b  # [S, 2560]
    # 3. Reshape [S, 2560] = [S, 5, 512] and transpose to NCHW [1, 512, S, 5].
    out_5x512 = out_2560.reshape(num_frames, INITIAL_FREQ_BINS, INITIAL_CHANNELS)
    # Want [1, C, S, W] with W=5, C=512.
    nchw = out_5x512.transpose(2, 0, 1)[None]  # [1, 512, S, 5]
    # 4. Temporal-pad H by temporal_pad/2 on each side.
    half = temporal_pad // 2
    padded = np.pad(nchw, ((0, 0), (0, 0), (half, temporal_pad - half), (0, 0)))
    return padded.astype(np.float32)


def decoder_forward(preprocessed: np.ndarray, weights: dict, intermediates: dict):
    """Run the full decoder forward pass. preprocessed [1, 512, T_pad, 5] → [1, 2, T_pad*4, 1920]."""
    intermediates['preprocessed_input'] = preprocessed
    x = preprocessed
    # --- input_layer residual block: SAME padding both axes (preserves H, W). ---
    residual = x
    for sub in ['conv2d_3x3_a', 'conv2d_3x3']:
        h = elu(x)
        k = weights[f'decoder.input_layer.{sub}.weight_norm.rescaled.kernel']
        b = weights[f'decoder.input_layer.{sub}.weight_norm.bias']
        x = conv2d_nchw(h, k, b, stride=1, padding=(1, 1))
    x = x + residual
    intermediates['stage_input_layer'] = x.copy()

    # --- 7 decoder blocks ---
    for idx, (name, kt_h, kt_w, stride_h, stride_w, in_c, mid_c, out_c, has_shortcut) in enumerate(DECODER_BLOCKS):
        # Main path: ELU → ConvTranspose(stride_h, stride_w) → crop → ELU → conv2d_3x3(SAME).
        main = elu(x)
        if idx == 0:
            intermediates['block0_after_elu'] = main.copy()
        kt_prefix = f'decoder.{name}.conv2dtranspose_{kt_h}x{kt_w}'
        kt_kernel = weights[f'{kt_prefix}.weight_norm.rescaled.kernel']
        kt_bias = weights[f'{kt_prefix}.weight_norm.bias']
        main = conv2d_transpose_nchw(main, kt_kernel, kt_bias, stride=(stride_h, stride_w))
        if idx == 0:
            intermediates['block0_after_convT'] = main.copy()
        # crop_freq_dim: trim (kt_h - stride_h, kt_w - stride_w) cells total on H/W, centered.
        h_excess = kt_h - stride_h
        w_excess = kt_w - stride_w
        main = main[:, :, h_excess // 2 : main.shape[2] - (h_excess - h_excess // 2),
                          w_excess // 2 : main.shape[3] - (w_excess - w_excess // 2)]
        if idx == 0:
            intermediates['block0_after_crop'] = main.copy()
        main = elu(main)
        c3_prefix = f'decoder.{name}.conv2d_3x3'
        c3_kernel = weights[f'{c3_prefix}.weight_norm.rescaled.kernel']
        c3_bias = weights[f'{c3_prefix}.weight_norm.bias']
        # SAME padding on both axes (preserves H, W).
        main = conv2d_nchw(main, c3_kernel, c3_bias, stride=1, padding=(1, 1))
        if idx == 0:
            intermediates['block0_main_after_3x3'] = main.copy()

        # Shortcut: optional 1x1 conv → nearest-neighbor upsample by (stride_h, stride_w).
        shortcut = x
        if has_shortcut:
            sc_prefix = f'decoder.{name}.shortcut.conv1x1'
            sc_kernel = weights[f'{sc_prefix}.weight_norm.rescaled.kernel']
            sc_bias = weights[f'{sc_prefix}.weight_norm.bias']
            shortcut = conv2d_nchw(shortcut, sc_kernel, sc_bias, stride=1, padding=(0, 0))
        if stride_h > 1:
            shortcut = np.repeat(shortcut, stride_h, axis=2)
        if stride_w > 1:
            shortcut = np.repeat(shortcut, stride_w, axis=3)
        if idx == 0:
            intermediates['block0_shortcut'] = shortcut.copy()

        if not has_shortcut and shortcut.shape[1] != main.shape[1]:
            raise ValueError(f"block {name}: no shortcut conv but channels mismatch {shortcut.shape[1]} != {main.shape[1]}")
        x = main + shortcut

        # After block 0, TF body applies a 4-way fold via reshape_6 + transpose_2 +
        # reshape_7. In NHWC [1, T, W, C=1024]: split T into 2 halves AND C into 2
        # halves, giving 4 batches. The new tensor shape is [4, T, W/2, C/2] with
        # the interleaving formula:
        #   batch = c_half * 2 + t_half
        #   t_out = 2 * (orig_t mod T/2) + (orig_w // W/2)
        #   w_out = orig_w mod W/2
        #   c_out = orig_c mod C/2
        # Verified bit-exact via test_4fold.py.
        if idx == 0:
            # 4-way fold ONLY (no pixel_shuffle). Splits both C and W in half,
            # producing 4 batches. decoder_1 expects in_c=512, which matches
            # C/2 after the fold. Verified bit-exact via test_4fold.py.
            x_nhwc = x.transpose(0, 2, 3, 1)  # [1, T, W, C]
            B, T, W, C = x_nhwc.shape
            x_r6 = x_nhwc.reshape(-1, T, W // 2, 2, C // 2)
            x_t2 = x_r6.transpose(3, 0, 1, 2, 4)
            x_r7 = x_t2.reshape(-1, T, W // 2, C // 2)
            x = x_r7.transpose(0, 3, 1, 2)

        intermediates[f'stage_block_{idx}'] = x.copy()

    # --- base_conv_last: SAME padding both axes (preserves H, W). ---
    x = elu(x)
    bcl_kernel = weights['decoder.input_layer.base_conv_last.conv.kernel']
    bcl_bias = weights['decoder.input_layer.base_conv_last.conv.bias']
    x = conv2d_nchw(x, bcl_kernel, bcl_bias, stride=1, padding=(3, 3))
    intermediates['stage_output'] = x.copy()

    # Reshape to audio (best-effort): [1, 2, T_final, W_final] → [T_final * W_final, 2].
    N, C, S, W = x.shape
    audio = x[0].transpose(1, 2, 0).reshape(S * W, C)
    intermediates['audio'] = audio
    return audio


def main():
    import sys
    use_tokens = '--tokens' in sys.argv

    print("Loading weights...")
    weights = load_safetensors(DUMP_DIR / 'weights_spectrostream.safetensors')
    print(f"  {len(weights)} tensors loaded")

    if use_tokens:
        print("Loading reference codec data (will dequantize tokens via host-side preprocessing)...")
        ref = load_safetensors(DUMP_DIR / 'reference_codec.safetensors')
        tokens = ref['tokens'].astype(np.int32)
        reconstructed = ref['reconstructed_audio']
        print(f"  tokens {tokens.shape}, reconstructed {reconstructed.shape}")
        print("Preprocessing input...")
        preprocessed = preprocess_input(tokens, weights, NUM_FRAMES, TEMPORAL_PAD)
    else:
        # Synthetic deterministic input — small magnitude (~ELU/sigmoid scale).
        # Same generator must be used on the meganeura side so the two
        # implementations get bit-identical input.
        in_size = 1 * INITIAL_CHANNELS * (NUM_FRAMES + TEMPORAL_PAD) * INITIAL_FREQ_BINS
        idx = np.arange(in_size, dtype=np.float32)
        flat = np.sin(idx * 1e-7).astype(np.float32)
        preprocessed = flat.reshape(1, INITIAL_CHANNELS, NUM_FRAMES + TEMPORAL_PAD, INITIAL_FREQ_BINS)
        reconstructed = None
        print("Using synthetic input (sin(i * 1e-7), same as test s10).")
    print(f"  preprocessed_input shape: {preprocessed.shape}")

    print("Running decoder forward pass (this is slow in NumPy — minutes for full input_layer)...")
    intermediates = {}
    audio = decoder_forward(preprocessed, weights, intermediates)

    print(f"NumPy decoder audio shape: {audio.shape}")
    print(f"  range [{audio.min():.4f}, {audio.max():.4f}]")
    if reconstructed is not None:
        print(f"Reference reconstructed shape: {reconstructed.shape}")
        print(f"  range [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")

    # Compare against TF reference if available.
    if reconstructed is not None:
        min_len = min(audio.shape[0], reconstructed.shape[0])
        diff = audio[:min_len] - reconstructed[:min_len]
        print(f"Max abs diff: {np.abs(diff).max():.6f}")
        print(f"Mean abs diff: {np.abs(diff).mean():.6f}")
        print(f"Relative diff: {np.abs(diff).mean() / (np.abs(reconstructed[:min_len]).mean() + 1e-9):.4f}")

    out_path = DUMP_DIR / 'decoder_reference.safetensors'
    save_safetensors(out_path, intermediates)
    print(f"\nSaved intermediates to {out_path}")


if __name__ == '__main__':
    main()
