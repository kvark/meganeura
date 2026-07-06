#!/usr/bin/env python3
"""Full body→audio pipeline test on Python NumPy ref output.

Steps:
  1. Run decoder ref with reference tokens, num_frames=50.
  2. Take stage_output [2, 2, 204, 1920] (= NCHW from 2-fold path).
  3. Convert to NHWC and apply TF's reshape_8/transpose_3/reshape_9 chain.
  4. temporal_cropping: strip 4 from front of T → [..., 200, 480, 4].
  5. iSTFT (sqrt-hann, hop 480, fft 960, window^2 OLA norm).
  6. Compare audio to reference_codec.safetensors's reconstructed_audio.
"""
import os
os.environ.setdefault("NUM_FRAMES", "50")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import struct, json
from pathlib import Path
import numpy as np

# Import the decoder ref functions.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "decoder_reference",
    "/x/Code/meganeura/tools/magenta_rt/decoder_reference.py",
)
ref_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref_mod)


def load_st(p):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
        raw = f.read()
    out = {}
    dtypes = {'F32': np.float32, 'I32': np.int32}
    for k, info in header.items():
        if k.startswith('__'): continue
        if info['dtype'] not in dtypes: continue
        s, e = info['data_offsets']
        shape = info['shape'] if info['shape'] else [1]
        out[k] = np.frombuffer(raw[s:e], dtype=dtypes[info['dtype']]).reshape(shape).copy()
    return out


def istft(stft, fft_size, hop, window):
    n_frames, n_freq = stft.shape
    out_len = (n_frames - 1) * hop + fft_size
    audio = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)
    for i in range(n_frames):
        full = np.zeros(fft_size, dtype=np.complex64)
        full[:n_freq] = stft[i]
        full[fft_size - n_freq + 1: fft_size] = np.conj(stft[i, 1:n_freq][::-1])
        time = np.fft.ifft(full).real
        audio[i * hop : i * hop + fft_size] += time * window
        norm[i * hop : i * hop + fft_size] += window * window
    norm = np.maximum(norm, 1e-10)
    return audio / norm


def main():
    DUMP = Path("/x/Code/meganeura/magenta_rt_codec_dump")

    print("Loading weights + tokens ...")
    weights = ref_mod.load_safetensors(DUMP / 'weights_spectrostream.safetensors')
    refs = ref_mod.load_safetensors(DUMP / 'reference_codec.safetensors')

    tokens = refs['tokens'].astype(np.int32)
    reconstructed = refs['reconstructed_audio']  # [96000, 2]

    preprocessed = ref_mod.preprocess_input(tokens, weights, ref_mod.NUM_FRAMES, ref_mod.TEMPORAL_PAD)
    print(f"preprocessed_input: {preprocessed.shape}")

    print("Running NumPy decoder ...")
    intermediates = {}
    _ = ref_mod.decoder_forward(preprocessed, weights, intermediates)
    body_pre_crop = intermediates['stage_output']
    print(f"stage_output: {body_pre_crop.shape}  range [{body_pre_crop.min():.3e}, {body_pre_crop.max():.3e}]")

    # 2-fold output is [B=2, C=2, T_pre_crop=204, W=1920] in NCHW.
    # Convert to NHWC + apply TF's tail chain.
    B, C, T_pre, W = body_pre_crop.shape
    print(f"B={B} C={C} T_pre={T_pre} W={W}")
    nhwc = body_pre_crop.transpose(0, 2, 3, 1)  # [B, T_pre, W, C]
    # reshape_8: [2, -1, T_pre, 480, 2]
    r8 = nhwc.reshape(2, -1, T_pre, 480, 2)
    print(f"reshape_8: {r8.shape}")
    # transpose_3 perm [1, 2, 3, 0, 4]
    t3 = r8.transpose(1, 2, 3, 0, 4)
    print(f"transpose_3: {t3.shape}")
    # reshape_9: TF actually uses T_dyn = 2 * T_pre (absorbs the leading 2 from
    # reshape_8 into T). So target [-1, 2*T_pre, 480, 4] which gives [1, 2*T_pre, 480, 4].
    r9 = t3.reshape(-1, 2 * T_pre, 480, 4)
    print(f"reshape_9: {r9.shape}")
    # temporal_cropping: strip 4 from front of T
    cropped = r9[:, 4:, :, :]
    print(f"temporal_cropping: {cropped.shape}")
    # Take batch 0 for audio (or sum batches?).
    body_eq = cropped[0]  # [200, 480, 4]
    print(f"body-equivalent: {body_eq.shape}  range [{body_eq.min():.3e}, {body_eq.max():.3e}]")

    # iSTFT
    Lr = body_eq[..., 0]; Li = body_eq[..., 1]
    Rr = body_eq[..., 2]; Ri = body_eq[..., 3]
    L_c = Lr + 1j * Li
    R_c = Rr + 1j * Ri
    fft_size = 960; hop = 480
    sqrt_hann = np.sqrt(np.hanning(fft_size)).astype(np.float32)
    L_audio = istft(L_c, fft_size, hop, sqrt_hann)
    R_audio = istft(R_c, fft_size, hop, sqrt_hann)
    audio = np.stack([L_audio[:96000], R_audio[:96000]], axis=-1)
    if audio.shape[0] < 96000:
        audio = np.pad(audio, ((0, 96000 - audio.shape[0]), (0, 0)))
    print(f"audio: {audio.shape}  range [{audio.min():.4f}, {audio.max():.4f}]")
    print(f"reference: {reconstructed.shape}  range [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")

    rms = np.sqrt(np.mean(reconstructed ** 2))
    err = np.sqrt(np.mean((audio - reconstructed) ** 2))
    print(f"rms-ratio: {err/rms:.4f}")

    # Also try summing all batches (in case TF wants the sum).
    body_sum = cropped.sum(axis=0)
    Lr2 = body_sum[..., 0]; Li2 = body_sum[..., 1]
    Rr2 = body_sum[..., 2]; Ri2 = body_sum[..., 3]
    L_c2 = Lr2 + 1j * Li2; R_c2 = Rr2 + 1j * Ri2
    L_audio2 = istft(L_c2, fft_size, hop, sqrt_hann)
    R_audio2 = istft(R_c2, fft_size, hop, sqrt_hann)
    audio2 = np.stack([L_audio2[:96000], R_audio2[:96000]], axis=-1)
    if audio2.shape[0] < 96000:
        audio2 = np.pad(audio2, ((0, 96000 - audio2.shape[0]), (0, 0)))
    err2 = np.sqrt(np.mean((audio2 - reconstructed) ** 2))
    print(f"summed batches: rms-ratio: {err2/rms:.4f}")

    # Try mean across batches.
    body_mean = cropped.mean(axis=0)
    L_c3 = body_mean[..., 0] + 1j * body_mean[..., 1]
    R_c3 = body_mean[..., 2] + 1j * body_mean[..., 3]
    L_audio3 = istft(L_c3, fft_size, hop, sqrt_hann)
    R_audio3 = istft(R_c3, fft_size, hop, sqrt_hann)
    audio3 = np.stack([L_audio3[:96000], R_audio3[:96000]], axis=-1)
    if audio3.shape[0] < 96000:
        audio3 = np.pad(audio3, ((0, 96000 - audio3.shape[0]), (0, 0)))
    err3 = np.sqrt(np.mean((audio3 - reconstructed) ** 2))
    print(f"mean batches: rms-ratio: {err3/rms:.4f}")


if __name__ == "__main__":
    main()
