#!/usr/bin/env python3
"""Verify exactly what TF's reshape_6 + transpose_2 + reshape_7 chain does
by applying it to a controlled tensor with unique values per position.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import tensorflow as tf


def main():
    # Create a controlled NHWC tensor that decoder_0 would produce.
    # Use small sizes for tractability: B=1, T=4, F=4 (= W in NHWC), C=8.
    # Each position gets a unique encoded value: t*1000000 + w*10000 + c.
    B, T, F, C = 1, 4, 4, 8
    x_np = np.zeros((B, T, F, C), dtype=np.float32)
    for t in range(T):
        for w in range(F):
            for c in range(C):
                x_np[0, t, w, c] = t * 1_000_000 + w * 10_000 + c

    print(f"Input shape: {x_np.shape}")
    print(f"Input layout: row-major NHWC, flat[B,T,F,C] = t*F*C + w*C + c")
    print()

    # Mimic reshape_6: [-1, T_dyn, F/2, 2, C/2]
    # In our toy: F=4, C=8 → [-1, T, 2, 2, 4]
    F_half = F // 2  # 2
    C_half = C // 2  # 4
    x = tf.constant(x_np)
    # The actual TF reshape_6 has T_dyn from getitem_6. Let's try BOTH interpretations:
    # (i) T_dyn = input.shape[1] = T (= our 4)
    # (ii) T_dyn = something else
    # Try (i) first.
    r6 = tf.reshape(x, [-1, T, F_half, 2, C_half])
    print(f"reshape_6 → [-1, T={T}, F_half={F_half}, 2, C_half={C_half}]: {r6.shape}")
    r6_np = r6.numpy()

    # Decode: which (orig_t, orig_w, orig_c) does each position correspond to?
    print("\nreshape_6 sample positions:")
    for (b_, t_, d1_, d2_, c_) in [(0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0),
                                     (0, 1, 0, 0, 0), (1, 0, 0, 0, 0)]:
        try:
            v = r6_np[b_, t_, d1_, d2_, c_]
            t_orig = int(v / 1_000_000)
            w_orig = int((v % 1_000_000) / 10_000)
            c_orig = int(v % 10_000)
            print(f"  r6[{b_},{t_},{d1_},{d2_},{c_}] = orig (t={t_orig}, w={w_orig}, c={c_orig})")
        except IndexError as e:
            print(f"  r6[{b_},{t_},{d1_},{d2_},{c_}]: out of range")

    # transpose_2 perm [3, 0, 1, 2, 4]
    t2 = tf.transpose(r6, perm=[3, 0, 1, 2, 4])
    print(f"\ntranspose_2 → {t2.shape}")

    # reshape_7: [-1, T_dyn, F_half, C_half]
    r7 = tf.reshape(t2, [-1, T, F_half, C_half])
    print(f"reshape_7 → [-1, T={T}, F_half={F_half}, C_half={C_half}]: {r7.shape}")
    r7_np = r7.numpy()

    print("\nreshape_7 sample positions (the final 4D batch=? layout):")
    for (b_, t_, d1_, c_) in [(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0),
                                (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]:
        v = r7_np[b_, t_, d1_, c_]
        t_orig = int(v / 1_000_000)
        w_orig = int((v % 1_000_000) / 10_000)
        c_orig = int(v % 10_000)
        print(f"  r7[{b_},{t_},{d1_},{c_}] = orig (t={t_orig}, w={w_orig}, c={c_orig})")


if __name__ == "__main__":
    main()
