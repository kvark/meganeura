#!/usr/bin/env python3
"""Discover the exact shape chain at the tail of the TF decoder body.

Strategy: rebuild the tail manually using known reshape constants from
extract_body_graph.py output, fed by the base_conv_last output (whose shape
we infer by running the body and computing back from the body output).
"""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402


def main():
    root = snapshot_download("google/magenta-realtime",
                             allow_patterns=["savedmodels/ssv2_48k_stereo/decoder/**"])
    with tf.device("/cpu:0"):
        dec = tf.saved_model.load(str(Path(root) / "savedmodels" / "ssv2_48k_stereo" / "decoder"))
    body = dec.keras_api.layers[0]

    embed = np.zeros((1, 50, 256), dtype=np.float32)
    with tf.device("/cpu:0"):
        body_out = body(tf.constant(embed))
    print(f"body_out: {body_out.shape}  range [{body_out.numpy().min():.4f}, {body_out.numpy().max():.4f}]")

    # body_out is [1, 200, 480, 4]. Working back through reshape_9 → transpose_3 → reshape_8 → base_conv_last:
    # reshape_9 input: [a, 200, 480, ?] where ? * 4 / 4 = 4 ⇒ input shape gives (-1, 200, 480, 4).
    # transpose_3 perm = [1, 2, 3, 0, 4]: dim 0 was something, dim 1 → 200, ..., dim 3 → batch'.
    # Reverse perm: inverse_perm of [1, 2, 3, 0, 4] is [3, 0, 1, 2, 4]. So transpose_3 INPUT was [batch', 200, 480, ?, 4].
    # But it's 5D. Wait — 200 = (batch' originally? No batch' was first dim).
    # Trying with body_out [1, 200, 480, 4] and considering temporal_cropping does a crop:
    # If reshape_9 output is [-1, T_pre_crop, 480, 4] then temporal_cropping → [1, 200, 480, 4]
    # T_pre_crop > 200, maybe 200 + 2*crop = 200 + 24 = 224.
    # transpose_3 input shape (before transpose) at perm [1, 2, 3, 0, 4]:
    #   input.shape[1] = 224, input.shape[2] = 480, input.shape[3] = ?, input.shape[0] = ?, input.shape[4] = 4
    # Total output (reshape_9 view) = 1 * 224 * 480 * 4 = 430080 (= total reshape_9 in)
    # And the reshape combines last two dims (?, 4) → final dim 4 (since transpose_3 has 5 dims, reshape_9 has 4 dims)
    # So [batch', T, F, X, 2] OR [batch', T, F, 2, 2]: the dim "?" should be 1 to give reshape_9 output last dim = 1*2 = 2? But final dim is 4.
    # Try ? = 2: transpose_3 input [batch', 224, 480, 2, 2], output (after perm [1,2,3,0,4]) shape [shape[1]=batch', shape[2]=224, shape[3]=480, shape[0]=2, shape[4]=2] = [batch', 224, 480, 2, 2]
    # Wait that's the same. Let me redo. transpose perm [1, 2, 3, 0, 4]:
    #   output.shape[i] = input.shape[perm[i]]
    #   perm[0]=1, perm[1]=2, perm[2]=3, perm[3]=0, perm[4]=4
    #   So output = [input.shape[1], input.shape[2], input.shape[3], input.shape[0], input.shape[4]]
    # If we want transpose output to combine via reshape_9 to [1, 224, 480, 4],
    # and reshape_9 sees the transpose output as 5D [-1, 224, 480, ?, ?] and reshape (collapse last 2) to [1, 224, 480, 4],
    # then ? * ? = 4 (likely 2*2) and -1 = 1.
    # So transpose output: [1, 224, 480, 2, 2]. Inverse perm: input.shape[0] = output.shape[3] = 2, input.shape[1] = output.shape[0] = 1, input.shape[2] = output.shape[1] = 224, input.shape[3] = output.shape[2] = 480, input.shape[4] = output.shape[4] = 2.
    # So reshape_8 output = [2, 1, 224, 480, 2]. Total = 2*1*224*480*2 = 430080.
    # base_conv_last total = 430080. Final base_conv_last shape:
    #   reshape_8 was [2, -1, getitem_8, 480, 2] with getitem_8 = base_conv_last.shape[1].
    #   getitem_8 = 224 (from reshape_8 output dim 2 = getitem_8 = 224).
    #   So base_conv_last shape[1] = 224.
    # If NHWC, base_conv_last output shape = [1, 224, ?, ?] with total 430080.
    # 224 * ? * ? = 430080 → ? * ? = 1920 → could be (1920, 1) or (480, 4) or (960, 2).
    # Given base_conv_last out_c is 2 (from manifest), shape is [1, 224, 960, 2]. So H=224, W=960.

    # Verify by inferring base_conv_last input shape.
    # base_conv_last kernel = 7x7, output channels 2 (no input freq_dim pad in H).
    # If H_out = 224 then H_in = 224 + 6 = 230 (VALID conv).
    # W_out = 960 with pad_w = 3 (SAME-ish W pad): W_in = 960.
    # So base_conv_last input: [1, 230, 960, 64].
    print("\nInferred (assuming temporal_cropping crops 12 from each side: 224→200):")
    print("  base_conv_last output: [1, 224, 960, 2]  (NHWC, H=224 time, W=960 freq, C=2)")
    print("  reshape_8 output:      [2, 1, 224, 480, 2]")
    print("  transpose_3 output:    [1, 224, 480, 2, 2]")
    print("  reshape_9 output:      [1, 224, 480, 4]")
    print("  temporal_cropping:     [1, 200, 480, 4]")

    print("\nMeganeura currently produces base_conv_last output [1, 2, 50, 1920].")
    print("That's H=50 time, W=1920 freq. So meganeura's TIME dim is ~4.5× too short")
    print("and FREQ dim is 2× too large vs TF. Most likely cause: temporal_pad and the")
    print("time-axis upsampling pattern through decoder blocks differ.")
    print("\nRoot cause hypothesis: meganeura treats the embed S=50 frames as the time")
    print("axis, but TF treats it as encoded frames that get UPSAMPLED ~4x through the")
    print("decoder blocks. Meganeura's decoder shrinks time by 14 (from VALID conv2d_3x3),")
    print("so its preprocessing prepads by 24 to compensate. TF doesn't shrink time —")
    print("conv2d_3x3 likely uses SAME padding on both H and W (not just W).")


if __name__ == "__main__":
    main()
