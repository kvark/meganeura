#!/usr/bin/env python3
"""Capture TF body output for reference tokens, then try applying iSTFT to
see if body output → audio is a clean iSTFT operation.

TF body output: [1, 200, 480, 4] for [1, 50, 256] embed input.
Reference audio: [96000, 2] = 2 seconds stereo at 48kHz.

If body output's 4 channels are (L_real, L_imag, R_real, R_imag) and we apply
a standard iSTFT (window 960, hop 480), we should recover audio close to the
reference.
"""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from safetensors import safe_open  # noqa: E402


DUMP = Path("magenta_rt_codec_dump")


def main():
    root = snapshot_download("google/magenta-realtime",
                             allow_patterns=["savedmodels/ssv2_48k_stereo/decoder/**"])
    with tf.device("/cpu:0"):
        dec = tf.saved_model.load(str(Path(root) / "savedmodels" / "ssv2_48k_stereo" / "decoder"))
    body = dec.keras_api.layers[0]

    # Load reference tokens and codebooks.
    refs = {}
    with safe_open(str(DUMP / "reference_codec.safetensors"), framework="numpy") as f:
        for k in f.keys():
            refs[k] = f.get_tensor(k)
    weights = {}
    with safe_open(str(DUMP / "weights_spectrostream.safetensors"), framework="numpy") as f:
        for k in f.keys():
            weights[k] = f.get_tensor(k)

    tokens = refs["tokens"]
    codebooks = weights["quantizer.rvq_codebooks"]
    embed = np.zeros((tokens.shape[0], codebooks.shape[2]), dtype=np.float32)
    for k in range(tokens.shape[1]):
        embed += codebooks[k, tokens[:, k]]
    embed_b = embed[None].astype(np.float32)

    print(f"Running body(embed{embed_b.shape}) and dec(embed{embed_b.shape}) ...")
    with tf.device("/cpu:0"):
        body_out = body(tf.constant(embed_b)).numpy()
        audio_out = dec(tf.constant(embed_b)).numpy()[0]
    print(f"body_out: {body_out.shape}  range [{body_out.min():.4f}, {body_out.max():.4f}]")
    print(f"audio_out: {audio_out.shape}  range [{audio_out.min():.4f}, {audio_out.max():.4f}]")
    print(f"reference: {refs['reconstructed_audio'].shape}  "
          f"range [{refs['reconstructed_audio'].min():.4f}, {refs['reconstructed_audio'].max():.4f}]")

    # Validate dec output matches reference (sanity).
    err = np.sqrt(np.mean((audio_out - refs['reconstructed_audio']) ** 2))
    rms = np.sqrt(np.mean(refs['reconstructed_audio'] ** 2))
    print(f"dec vs reference rms ratio: {err/rms:.6f}")

    # Now try applying iSTFT to body_out manually.
    # body_out shape [1, 200, 480, 4]. Treat the 4 channels as
    # (L_real, L_imag, R_real, R_imag). Build complex STFT per stereo channel.
    body_squeezed = body_out[0]   # [200, 480, 4]
    L_real = body_squeezed[..., 0]
    L_imag = body_squeezed[..., 1]
    R_real = body_squeezed[..., 2]
    R_imag = body_squeezed[..., 3]
    L_complex = L_real + 1j * L_imag   # [200, 480]
    R_complex = R_real + 1j * R_imag

    # Try standard iSTFT parameters: window 960 (FFT size 960), hop 480.
    # 480 freq bins means input is (FFT_size / 2) = 480, FFT_size = 960.
    fft_size = 960
    hop = 480
    # Use a Hann window.
    window = np.hanning(fft_size).astype(np.float32)

    def istft(stft, fft_size, hop, window):
        """stft: [n_frames, n_freq] complex. Returns [n_frames * hop + fft_size - hop] real."""
        n_frames, n_freq = stft.shape
        out_len = (n_frames - 1) * hop + fft_size
        audio = np.zeros(out_len, dtype=np.float32)
        # Construct full spectrum from half (Hermitian symmetry).
        # If n_freq = fft_size / 2 = 480: NO Nyquist, NO DC handling.
        # Need to be careful about whether n_freq includes DC and Nyquist.
        # Assume n_freq = fft_size / 2 (positions 0..479 of full spectrum 0..959, NO Nyquist).
        for i in range(n_frames):
            # Build full FFT bin set: positions 0..n_freq + conjugate of n_freq..0
            full = np.zeros(fft_size, dtype=np.complex64)
            full[:n_freq] = stft[i]
            # Hermitian symmetry: full[fft_size - k] = conj(full[k])
            full[fft_size - n_freq + 1: fft_size] = np.conj(stft[i, 1:n_freq][::-1])
            time = np.fft.ifft(full).real
            audio[i * hop : i * hop + fft_size] += time * window
        # Normalize for the overlap-add of the window.
        # COLA condition for Hann + hop = fft_size / 2 is satisfied.
        return audio

    ref_len = refs['reconstructed_audio'].shape[0]

    def compare(label, L_audio, R_audio):
        a = np.stack([L_audio[:ref_len], R_audio[:ref_len]], axis=-1)
        if a.shape[0] < ref_len:
            a = np.pad(a, ((0, ref_len - a.shape[0]), (0, 0)))
        e = np.sqrt(np.mean((a - refs['reconstructed_audio']) ** 2))
        print(f"  {label}: rms-ratio {e/rms:.4f}")

    # Try various scaling/normalization with sqrt_hann (best so far).
    print(f"\n--- scaling sweep with sqrt_hann ---")
    Lr = body_squeezed[..., 0]; Li = body_squeezed[..., 1]
    Rr = body_squeezed[..., 2]; Ri = body_squeezed[..., 3]
    L_c0 = Lr + 1j * Li
    R_c0 = Rr + 1j * Ri
    sqrt_hann = np.sqrt(np.hanning(fft_size)).astype(np.float32)
    for scale in [1.0, 0.5, 0.25, 1.0/fft_size, 2.0/fft_size, 1.0/hop, 1.0/np.sqrt(fft_size), 1.0/np.sqrt(hop)]:
        L_audio = istft(L_c0, fft_size, hop, sqrt_hann) * scale
        R_audio = istft(R_c0, fft_size, hop, sqrt_hann) * scale
        compare(f"sqrt_hann scale={scale:.6f}", L_audio, R_audio)

    # Try conjugation, sign flips.
    print(f"\n--- sign and conjugation experiments ---")
    for sign_l, sign_r, conj_l, conj_r in [
        (1, 1, False, False),
        (1, 1, True, False),
        (1, 1, False, True),
        (1, 1, True, True),
        (-1, 1, False, False),
        (1, -1, False, False),
        (-1, -1, False, False),
    ]:
        L_c = sign_l * (Lr + 1j * Li * (-1 if conj_l else 1))
        R_c = sign_r * (Rr + 1j * Ri * (-1 if conj_r else 1))
        L_audio = istft(L_c, fft_size, hop, sqrt_hann)
        R_audio = istft(R_c, fft_size, hop, sqrt_hann)
        compare(f"sl={sign_l} sr={sign_r} cl={conj_l} cr={conj_r}", L_audio, R_audio)

    # Try with explicit window-norm: divide by sum of window^2 per output sample.
    print(f"\n--- with overlap normalization ---")
    def istft_normalized(stft, fft_size, hop, window):
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
    L_audio = istft_normalized(L_c0, fft_size, hop, sqrt_hann)
    R_audio = istft_normalized(R_c0, fft_size, hop, sqrt_hann)
    compare("sqrt_hann + window^2 norm", L_audio, R_audio)
    L_audio = istft_normalized(L_c0, fft_size, hop, np.hanning(fft_size).astype(np.float32))
    R_audio = istft_normalized(R_c0, fft_size, hop, np.hanning(fft_size).astype(np.float32))
    compare("hann + window^2 norm", L_audio, R_audio)

    # Try Hann (not sqrt) with proper normalization, also Vorbis window.
    def vorbis_window(N):
        n = np.arange(N)
        return np.sin(np.pi / 2 * np.sin(np.pi * (n + 0.5) / N) ** 2).astype(np.float32)
    for wn, w in [("hann", np.hanning(fft_size).astype(np.float32)),
                   ("vorbis", vorbis_window(fft_size))]:
        L_audio = istft_normalized(L_c0, fft_size, hop, w)
        R_audio = istft_normalized(R_c0, fft_size, hop, w)
        compare(f"{wn} + sqrt(window) synth norm", L_audio, R_audio)
        # Pure inverse with no window on synthesis
        L_audio2 = istft(L_c0, fft_size, hop, np.ones(fft_size).astype(np.float32))
        R_audio2 = istft(R_c0, fft_size, hop, np.ones(fft_size).astype(np.float32))
        compare(f"no-window synth", L_audio2, R_audio2)

    # Try different freq-bin interpretations: include DC + Nyquist (n_freq=481)
    # with the last bin being Nyquist (= take 480 of the 480 bins as 0..479, with Nyquist=0).
    print(f"\n--- variants with explicit DC/Nyquist ---")
    # Maybe bins are 0..479 representing DC..nyquist-1 (no Nyquist)
    # Or 0..479 representing pos 1..480 (skipping DC)

    # Skip DC: shift bins by 1, with DC=0
    L_c_skip_dc = np.concatenate([np.zeros((L_c0.shape[0], 1), dtype=L_c0.dtype), L_c0[:, :-1]], axis=-1)
    R_c_skip_dc = np.concatenate([np.zeros((R_c0.shape[0], 1), dtype=R_c0.dtype), R_c0[:, :-1]], axis=-1)
    L_audio = istft(L_c_skip_dc, fft_size, hop, sqrt_hann)
    R_audio = istft(R_c_skip_dc, fft_size, hop, sqrt_hann)
    compare("skip-DC sqrt_hann", L_audio, R_audio)

    # Write audio_attempt to WAV for listening.
    L_audio = istft(L_c0, fft_size, hop, sqrt_hann)
    R_audio = istft(R_c0, fft_size, hop, sqrt_hann)
    audio = np.stack([L_audio[:ref_len], R_audio[:ref_len]], axis=-1)
    if audio.shape[0] < ref_len:
        audio = np.pad(audio, ((0, ref_len - audio.shape[0]), (0, 0)))
    audio_clip = np.clip(audio, -1.0, 1.0)
    wav = tf.audio.encode_wav(tf.constant(audio_clip, dtype=tf.float32), sample_rate=48000)
    (DUMP / "naive_istft_attempt.wav").write_bytes(wav.numpy())
    print(f"\nWrote {DUMP / 'naive_istft_attempt.wav'} for ear-check")
    # Also write the reference + TF dec output as side-by-side comparators.
    wav = tf.audio.encode_wav(tf.constant(refs['reconstructed_audio'], dtype=tf.float32), sample_rate=48000)
    (DUMP / "tf_reconstructed.wav").write_bytes(wav.numpy())
    wav = tf.audio.encode_wav(tf.constant(audio_out, dtype=tf.float32), sample_rate=48000)
    (DUMP / "tf_dec_output.wav").write_bytes(wav.numpy())
    print(f"Wrote {DUMP / 'tf_reconstructed.wav'} (ground truth)")
    print(f"Wrote {DUMP / 'tf_dec_output.wav'} (TF dec function)")

    # Try various channel orderings and windows.
    print(f"\n--- naive numpy iSTFT, channel order experiments ---")
    chs = body_squeezed
    # Order options: (L_real, L_imag, R_real, R_imag) vs (L_real, R_real, L_imag, R_imag) etc.
    orderings = [
        ("LriRri",  chs[..., 0], chs[..., 1], chs[..., 2], chs[..., 3]),
        ("LrLiRrRi", chs[..., 0], chs[..., 1], chs[..., 2], chs[..., 3]),  # same
        ("LrRrLiRi", chs[..., 0], chs[..., 2], chs[..., 1], chs[..., 3]),
        ("LrRiLiRr", chs[..., 0], chs[..., 3], chs[..., 1], chs[..., 2]),
    ]
    for label, Lr, Li, Rr, Ri in orderings:
        L_c = Lr + 1j * Li
        R_c = Rr + 1j * Ri
        for win_name, win in [("hann", np.hanning(fft_size).astype(np.float32)),
                               ("sqrt_hann", np.sqrt(np.hanning(fft_size)).astype(np.float32)),
                               ("rect", np.ones(fft_size, dtype=np.float32))]:
            L_audio = istft(L_c, fft_size, hop, win)
            R_audio = istft(R_c, fft_size, hop, win)
            compare(f"{label} {win_name}", L_audio, R_audio)

    # Also try TF's official inverse_stft.
    print(f"\n--- tf.signal.inverse_stft ---")
    # tf.signal.inverse_stft expects [..., n_frames, n_freq] complex.
    # n_freq for an iSTFT with frame_length 960 = 960/2+1 = 481. We have 480.
    # Try padding to 481 with zero.
    for n_freq_target, label in [(480, "as-is"), (481, "pad-1-bin")]:
        # Build STFT
        if n_freq_target == 481:
            L_stft = np.pad(L_complex, ((0, 0), (0, 1)))
            R_stft = np.pad(R_complex, ((0, 0), (0, 1)))
        else:
            L_stft = L_complex
            R_stft = R_complex
        for window_fn, wname in [(tf.signal.hann_window, "hann"),
                                  (tf.signal.hamming_window, "hamming")]:
            for frame_len, frame_step in [(960, 480), (480, 240)]:
                try:
                    L_audio = tf.signal.inverse_stft(
                        tf.constant(L_stft, dtype=tf.complex64),
                        frame_length=frame_len, frame_step=frame_step,
                        window_fn=tf.signal.inverse_stft_window_fn(frame_step, forward_window_fn=window_fn),
                    ).numpy()
                    R_audio = tf.signal.inverse_stft(
                        tf.constant(R_stft, dtype=tf.complex64),
                        frame_length=frame_len, frame_step=frame_step,
                        window_fn=tf.signal.inverse_stft_window_fn(frame_step, forward_window_fn=window_fn),
                    ).numpy()
                    compare(f"tf {label} {wname} fft={frame_len} hop={frame_step}",
                            L_audio, R_audio)
                except Exception as e:
                    print(f"  tf {label} {wname} fft={frame_len} hop={frame_step}: FAIL {str(e)[:60]}")


if __name__ == "__main__":
    main()
