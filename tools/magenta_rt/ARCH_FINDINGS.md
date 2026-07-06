# SpectroStream decoder architecture: corrections from TF graph inspection

The current Python ref (`decoder_reference.py`) and meganeura
(`src/models/magenta_rt/spectrostream.rs`) implementations have several
**architectural bugs** that account for the ~35× value-magnitude mismatch
against TF (range `[-1777, 4072]` vs TF body `[-51, 47]` on the same input).

The diagnostic that nailed it:
1. Per-stage rms growth in `tools/magenta_rt/per_stage_ranges.py` shows
   biggest amplifiers at decoder_1 (7.5×) and decoder_5 (6.2×).
2. TF `concrete_function.graph.library.function` was dumped to reveal real
   per-op padding constants and conv-T strides.

## Discovered facts about TF graph

### conv2d_3x3 (every occurrence)
Padding sequence:
1. `freq_dim_pad`: `Pad/paddings = [[0,0],[0,0],[1,1],[0,0]]` → pad W [1,1]
2. internal `weight_norm/Pad`: `Pad/paddings = [[0,0],[2,0],[0,0],[0,0]]` →
   pad H **[2, 0]** (causal — only past frames)
3. `Conv2D` `padding=VALID`, kernel 3×3, stride 1

Net effect: H is preserved (51 → 53 via causal pad → 51 via VALID conv) and
W is preserved (5 → 7 → 5). **Causal in H**, SAME-like in W.

### base_conv_last (input_layer/base_conv_last/conv)
- `Pad/paddings = [[0,0],[6,0],[0,0],[0,0]]` → pad H **[6, 0]** (causal)
- Conv2D VALID, kernel 7×7
- W has **no padding** → W shrinks by 6

### conv-T strides (Conv2DBackpropInput) per block, NHWC `[B,H,W,C]`:
| block      | strides         | (stride_h, stride_w) |
|------------|-----------------|----------------------|
| decoder_0  | `[1, 2, 1, 1]`  | **(2, 1)** — H only  |
| decoder_1  | `[1, 2, 2, 1]`  | **(2, 2)**           |
| decoder_2  | `[1, 1, 2, 1]`  | (1, 2)               |
| decoder_3  | `[1, 1, 2, 1]`  | (1, 2)               |
| decoder_4  | `[1, 1, 3, 1]`  | (1, 3)               |
| decoder_5  | `[1, 1, 2, 1]`  | (1, 2)               |
| decoder_6  | `[1, 1, 2, 1]`  | (1, 2)               |

Current `decoder_reference.py` has decoder_0 (2,2) and decoder_1 (1,2) — WRONG.

### Conv-T internal slice
The Conv2DBackpropInput op's output is then sliced internally. The slice
appears to trim H to `H_in * stride_h` from the END (causal trim) and may
leave W untouched. The `crop_freq_dim` function strips 1 each side of W
afterwards.

### crop_freq_dim (decoder_0)
`StridedSlice` begin=`[0,0,1,0]`, end=`[0,0,-1,0]`, strides=`[1,1,1,1]`.
Strips 1 from each side of W. Confirmed for decoder_0; needs re-check for
others (per-block sizes may vary).

### Shortcut upsample2d
ResizeNearestNeighbor by `Const value=[2,1]` (decoder_0) or `[2,2]`
(decoder_1) etc. — matches conv-T strides.

### input_layer/conv1x1_first
Has **two parallel paths** that are SUMMED:
1. `conv1x1_a(x) → ELU → conv1x1_b` (residual main branch)
2. `conv1x1(x)` (parallel branch)
3. `add_1 = main + parallel` → reshape_5 → conv2d_3x3 residual block

My `decoder_reference.py` currently uses ONLY conv1x1_a, dropping conv1x1_b
and conv1x1 — but the test `il_out == conv1x1_a output` matched because the
TF intermediate I dumped (`input_layer_out_embed_B_S_1_D`) is the
conv1x1_a output specifically, not the input_layer's final output.

### temporal_padding
`Pad/paddings = [[0,0],[0,1],[0,0]]` → pad 1 frame at END of T axis. ✓ (matches)

### temporal_cropping
`StridedSlice` begin=`[0,4]`, strips 4 frames from front of T. ✓ (matches)

### Tail chain (reshape_8 → transpose_3 → reshape_9)
- transpose_3 perm: `[1, 2, 3, 0, 4]`. ✓ (matches)

## What needs to change

1. **conv2d_3x3 helper** in Python + meganeura: switch from current
   `padding=(1,1)` SAME to causal H pad `[2,0]` + SAME W pad `[1,1]`. Use
   `conv2d` with explicit asymmetric padding.

2. **base_conv_last**: causal H pad `[6,0]`, VALID 7×7. Currently uses
   SAME padding `(3,3)`.

3. **conv-T strides**: fix decoder_0 to (2,1), decoder_1 to (2,2).

4. **conv-T crop**: switch from current centered `[h_excess//2, h_excess-h_excess//2]`
   to TF's actual semantics — trim H to `H_in * stride_h` from END (causal)
   and crop W via `crop_freq_dim` `[1, 1]`.

5. **input_layer**: add the parallel `conv1x1_b` and `conv1x1` paths so the
   input_layer output is `conv1x1_b(ELU(conv1x1_a(x))) + conv1x1(x)`.

6. **4-fold / pixel-shuffle between decoder_0 and decoder_1**: verify
   semantics match TF reshape_6/transpose_2/reshape_7 (currently believed
   bit-exact via `test_4fold.py` but the in/out shapes need to recompute
   given new conv-T sizes).

## Verification plan

1. Rewrite `decoder_reference.py` with the corrections above.
2. Run `body_compare_tf.py` (Python ref against TF body_out for real
   tokens). Target: <1% relative error.
3. Mirror corrections into `src/models/magenta_rt/spectrostream.rs`.
4. Verify meganeura matches TF body output too (new test).
5. Then iSTFT (which already gives 0.90 rms ratio when fed TF body
   directly) gives near-bit-exact audio output.
