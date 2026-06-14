#!/usr/bin/env python3
"""Validate the 4-way batch fold formula against TF's actual reshape chain
on the SpectroStream decoder.

If we get the 4-fold right, applying decoder_1..6 + base_conv_last to a
[4, 50, 5, 512] tensor (instead of meganeura's [2, T_meg, 10, 512]) should
yield TF body output.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import tensorflow as tf


def my_4fold(x_nhwc):
    """Apply the TF reshape_6 + transpose_2 + reshape_7 chain to [B=1, T, W, C].

    Output: [4, T, W/2, C/2] where the 4 batches are:
      batch = c_half * 2 + t_half
      (c_half = orig_c // (C/2),  t_half = orig_t // (T/2))

    Inside each batch:
      t_new = 2 * (orig_t mod (T/2)) + (orig_w // (W/2))
      w_new = orig_w mod (W/2)
      c_new = orig_c mod (C/2)
    """
    B, T, W, C = x_nhwc.shape
    assert B == 1
    T_half = T // 2; W_half = W // 2; C_half = C // 2
    out = np.zeros((4, T, W_half, C_half), dtype=x_nhwc.dtype)
    for batch in range(4):
        c_half = batch // 2
        t_half = batch % 2
        for t_new in range(T):
            for w_new in range(W_half):
                for c_new in range(C_half):
                    orig_t = (t_new // 2) + t_half * T_half
                    orig_w = (t_new % 2) * W_half + w_new
                    orig_c = c_half * C_half + c_new
                    out[batch, t_new, w_new, c_new] = x_nhwc[0, orig_t, orig_w, orig_c]
    return out


def tf_4fold(x_nhwc):
    """TF's exact reshape_6 + transpose_2 + reshape_7 chain."""
    B, T, W, C = x_nhwc.shape
    x = tf.constant(x_nhwc)
    r6 = tf.reshape(x, [-1, T, W // 2, 2, C // 2])
    t2 = tf.transpose(r6, perm=[3, 0, 1, 2, 4])
    r7 = tf.reshape(t2, [-1, T, W // 2, C // 2])
    return r7.numpy()


def main():
    # Test on a tractable size.
    np.random.seed(0)
    B, T, W, C = 1, 8, 8, 16
    x = np.random.randn(B, T, W, C).astype(np.float32)

    my_out = my_4fold(x)
    tf_out = tf_4fold(x)
    print(f"my:  {my_out.shape}  range [{my_out.min():.4f}, {my_out.max():.4f}]")
    print(f"tf:  {tf_out.shape}  range [{tf_out.min():.4f}, {tf_out.max():.4f}]")
    print(f"max diff: {np.abs(my_out - tf_out).max():.6e}")

    # Verify on a position-by-position basis.
    for batch in [0, 1, 2, 3]:
        diff = np.abs(my_out[batch] - tf_out[batch]).max()
        print(f"  batch {batch}: max diff = {diff:.6e}")


if __name__ == "__main__":
    main()
