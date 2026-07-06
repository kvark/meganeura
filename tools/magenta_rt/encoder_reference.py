#!/usr/bin/env python3
"""NumPy reference for the SpectroStream encoder (audio STFT → embed [50,256]).

Mirror of the verified decoder (decoder_reference_v2.py). Architecture +
strides/paddings extracted from the TF encoder SavedModel graph:

  STFT [1,T,480,4] → fold 4ch → batch [2,T,480,2]
  → base_conv_first (causal H[6,0], W[3,3], VALID 7x7, 2→32)
  → encoder_0..6 (downsample), with a batch→channel fold before encoder_6
  → bottleneck (reshape 5*256→1280, gated conv1x1_last → 256) → embed [1,50,256]

Each encoder block (mirror of a decoder block):
  main = conv2d_3x3(ELU(down_conv(ELU(x))));  down_conv = causal H[2,0] + W[1,1]
         VALID strided conv (stride_h, stride_w)
  shortcut = downsample(x by (sh,sw)) (+ conv1x1 if channels change)
  x = main + shortcut
"""
import os
import numpy as np
from pathlib import Path
from safetensors.numpy import load_file, save_file

DUMP = Path(os.environ.get("MAGENTA_RT_DUMP", "magenta_rt_codec_dump"))


def conv2d_valid(x, k, b=None, sh=1, sw=1):
    # x NHWC [B,H,W,Ci]; k [kh,kw,Ci,Co]; VALID, stride (sh,sw)
    B, H, W, Ci = x.shape
    kh, kw, _, Co = k.shape
    Ho = (H - kh) // sh + 1
    Wo = (W - kw) // sw + 1
    out = np.zeros((B, Ho, Wo, Co), np.float32)
    # im2col
    for i in range(kh):
        for j in range(kw):
            patch = x[:, i:i + sh * Ho:sh, j:j + sw * Wo:sw, :]  # [B,Ho,Wo,Ci]
            out += np.einsum('bhwc,co->bhwo', patch, k[i, j])
    if b is not None:
        out += b
    return out


def elu(x):
    return np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1.0).astype(np.float32)


def causal_3x3(x, k, b):
    x = np.pad(x, ((0, 0), (2, 0), (1, 1), (0, 0)))
    return conv2d_valid(x, k, b)


# encoder block specs: name, (kh,kw)_down, sh, sw, wpad, has_shortcut
EBLOCKS = [
    ('encoder_0', 3, 4, 1, 2, (1, 1), True),
    ('encoder_1', 3, 4, 1, 2, (1, 1), False),
    ('encoder_2', 3, 6, 1, 3, (1, 2), True),
    ('encoder_3', 3, 4, 1, 2, (1, 1), False),
    ('encoder_4', 3, 4, 1, 2, (1, 1), False),
    ('encoder_5', 4, 4, 2, 2, (1, 1), True),
    ('encoder_6', 4, 3, 2, 1, (1, 1), True),
]
DOWN_CONV = {'encoder_0': 'conv2d_3x4_a', 'encoder_1': 'conv2d_3x4_a', 'encoder_2': 'conv2d_3x6_a',
             'encoder_3': 'conv2d_3x4_a', 'encoder_4': 'conv2d_3x4_a', 'encoder_5': 'conv2d_4x4_a',
             'encoder_6': 'conv2d_4x3_a'}


def down_shortcut(x, sh, sw):
    # shortcut downsample = AvgPool2D window=stride=(sh,sw), VALID (TF graph).
    B, H, W, C = x.shape
    Ho, Wo = H // sh, W // sw
    xc = x[:, :Ho * sh, :Wo * sw, :].reshape(B, Ho, sh, Wo, sw, C)
    return xc.mean(axis=(2, 4)).astype(np.float32)


def encoder_forward(stft, W, inter):
    # stft [1,T,480,4] -> fold to [2,T,480,2]: channels [L_re,L_im,R_re,R_im]
    B, T, F, C = stft.shape
    x = stft.reshape(B, T, F, 2, 2).transpose(3, 0, 1, 2, 4).reshape(2, T, F, 2)
    inter['folded'] = x.copy()
    # base_conv_first: causal H[6,0], W[3,3], VALID 7x7
    xp = np.pad(x, ((0, 0), (6, 0), (3, 3), (0, 0)))
    k = W['encoder.base_conv_first.weight_norm.rescaled.kernel']
    b = W['encoder.base_conv_first.weight_norm.bias']
    x = conv2d_valid(xp, k, b)
    inter['base_conv_first'] = x.copy()

    for ei, (name, kh, kw, sh, sw, wpad, has_sc) in enumerate(EBLOCKS):
        if name == 'encoder_6':
            # batch→channel fold (inverse of decoder_0's C→batch fold): [2,T,W,C]->[1,T,W,2C]
            sb, t, w, c = x.shape
            x = x.transpose(1, 2, 0, 3).reshape(1, t, w, 2 * c)
            inter['pre_encoder_6_fold'] = x.copy()
        res = x
        # main (inverse order of the decoder block: conv2d_3x3 THEN downsample)
        h = elu(x)
        c3k = W[f'encoder.{name}.conv2d_3x3.weight_norm.rescaled.kernel']
        c3b = W[f'encoder.{name}.conv2d_3x3.weight_norm.bias']
        h = causal_3x3(h, c3k, c3b)
        h = elu(h)
        dc = DOWN_CONV[name]
        kk = W[f'encoder.{name}.{dc}.weight_norm.rescaled.kernel']
        kb = W[f'encoder.{name}.{dc}.weight_norm.bias']
        hp = np.pad(h, ((0, 0), (2, 0), (wpad[0], wpad[1]), (0, 0)))
        main = conv2d_valid(hp, kk, kb, sh=sh, sw=sw)
        # shortcut
        sc = down_shortcut(res, sh, sw)
        if has_sc:
            sck = W[f'encoder.{name}.shortcut.conv1x1.weight_norm.rescaled.kernel']
            scb = W[f'encoder.{name}.shortcut.conv1x1.weight_norm.bias']
            sc = conv2d_valid(sc, sck, scb)
        if sc.shape != main.shape:
            print(f"  WARN {name}: main {main.shape} sc {sc.shape}")
        x = main + sc
        inter[f'stage_{name}'] = x.copy()

    # bottleneck: reshape [1,T,5,256]->[1,T,1,1280], conv2d_3x3 residual + gated conv1x1_last
    b1, t1, w1, c1 = x.shape
    # conv2d_3x3_a then conv2d_3x3 residual (causal)
    res = x
    h = elu(x)
    kb2 = W['encoder.bottleneck.conv2d_3x3.weight_norm.rescaled.kernel']
    bb2 = W['encoder.bottleneck.conv2d_3x3.weight_norm.bias']
    h = causal_3x3(h, kb2, bb2)
    h = elu(h)
    ka = W['encoder.bottleneck.conv2d_3x3_a.weight_norm.rescaled.kernel']
    ba = W['encoder.bottleneck.conv2d_3x3_a.weight_norm.bias']
    h = causal_3x3(h, ka, ba)
    x = res + h
    inter['bottleneck_resid'] = x.copy()
    # reshape freq*chan -> 1280, gated conv1x1_last
    x = x.reshape(b1, t1, 1, w1 * c1)  # [1,T,1,1280]
    ka = W['encoder.bottleneck.conv1x1_last.conv1x1_a.weight_norm.rescaled.kernel']
    ba = W['encoder.bottleneck.conv1x1_last.conv1x1_a.weight_norm.bias']
    main_a = conv2d_valid(elu(x), ka, ba)
    kb3 = W['encoder.bottleneck.conv1x1_last.conv1x1_b.weight_norm.rescaled.kernel']
    bb3 = W['encoder.bottleneck.conv1x1_last.conv1x1_b.weight_norm.bias']
    main_b = conv2d_valid(elu(main_a), kb3, bb3)
    kp = W['encoder.bottleneck.conv1x1_last.conv1x1.weight_norm.rescaled.kernel']
    bp = W['encoder.bottleneck.conv1x1_last.conv1x1.weight_norm.bias']
    parallel = conv2d_valid(x, kp, bp)
    out = main_b + parallel  # [1,T,1,256]
    embed = out[:, :, 0, :]  # [1,T,256]
    inter['embed'] = embed.copy()
    return embed


def stft_features(audio):
    """SpectroStream encoder STFT: tf.signal.stft(frame 960, hop 480, fft 960,
    periodic Hann, pad_end=True), NO causal pre-pad — frame f = samples
    [f*480, f*480+960) (the last frame zero-padded at the END). Keep the first
    480 bins (DC..479, Nyquist dropped). Output [1, T, 480, 4] = [L_re,L_im,R_re,R_im].
    """
    from numpy.fft import rfft
    fl, fs = 960, 480
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(fl) / fl)).astype(np.float32)
    nsamp = audio.shape[0]
    nfr = nsamp // fs  # pad_end gives ceil; for nsamp multiple of fs this is exact
    feats = np.zeros((1, nfr, 480, 4), np.float32)
    for ch in range(2):
        sig = np.pad(audio[:, ch], (0, fl))  # pad end so the last frame is full
        for f in range(nfr):
            frame = sig[f * fs:f * fs + fl] * win
            spec = rfft(frame)
            feats[0, f, :, ch * 2] = spec[:480].real
            feats[0, f, :, ch * 2 + 1] = spec[:480].imag
    return feats


def main():
    W = load_file(str(DUMP / 'weights_spectrostream.safetensors'))
    er = load_file(str(DUMP / 'encoder_ref.safetensors'))
    # STFT features from the reference audio — use the saved TF body input if present,
    # else compute. Here we feed the same audio the TF encoder used.
    # The encoder body input is the STFT [1,T,480,4]; compute it to match the TF stft.
    import importlib.util
    audio = er['audio']  # [96000,2]
    feats = stft_features(audio)
    inter = {}
    embed = encoder_forward(feats, W, inter)
    tf_embed = er['embed'][0] if er['embed'].ndim == 3 else er['embed']
    print("numpy embed:", embed.shape, "TF embed:", tf_embed.shape)
    m = min(embed.shape[1], tf_embed.shape[0])
    d = np.abs(embed[0, :m] - tf_embed[:m]).max()
    rel = d / (np.abs(tf_embed[:m]).max() + 1e-9)
    print(f"embed max abs diff {d:.4e}  rel {rel:.4e}")
    for k, v in inter.items():
        print(f"  {k:22s} {str(v.shape):24s} range [{v.min():.2f},{v.max():.2f}]")


if __name__ == '__main__':
    main()
