# Plan: cooperative-matrix flash attention forward

## Why this is the next big lever

After the `layer_norm` rewrite (commit `8ede280`), the Whisper-tiny
inference profile collapses to one dominant cost:

```
FlashAttention[1500]  4x   4.60 ms  (74.8% of 6.15 ms total)
MatMul[1500x384x*]   24x   1.04 ms  (16.9%)
Conv2dGemm            2x   0.18 ms  ( 2.8%)
LayerNorm             9x   0.07 ms  ( 1.2%)
(small ops)               ~0.27 ms  ( 4.3%)
```

`FlashAttention[1500]` — flash-2 forward, head_dim=64, q_seq=1500,
non-causal — is running at **~3.4 GFLOPs per layer / 1.15 ms ≈ 6 % of
RTX 5080 FP32 peak**. The QKV-projection matmuls in the same model
(`MatMul[1500x384x384]`) cost 26 µs/dispatch via `matmul_coop_*`,
hitting tensor cores at vastly higher arithmetic intensity. The
flash inner loop is plain scalar arithmetic (`pdot += q{e} *
shared_k[...]` in `generate_flash_attention_module`), which Volta+
hardware can't auto-promote to MMA.

Closing this gap means rewriting the inner QK^T and PV products to
use `coop_mat` ops, the way `gen_matmul_coop_wgsl_prologue` already
does for the projections.

## Expected payoff

| dispatch                 | now      | coop-MMA target | rough gain |
|--------------------------|----------|-----------------|------------|
| Whisper inf (FA total)   | 4.60 ms  | ~0.5–1.0 ms     | ~5x        |
| Whisper inference end    | 6.15 ms  | ~2.0–2.5 ms     | ~2.5x      |
| SmolLM2 train (FA fwd+bwd) | 5.5 ms (8 + 7.4 + 22.7%) | half | ~1.2x training |

The training case is murkier because the bwd kernels also run; this
plan covers forward only.

## Why it is not a one-afternoon refactor

The existing flash kernel co-mingles three concerns the coop_mat
rewrite has to disentangle:

1. **Online softmax**. The kernel keeps `max_score` and `sum_exp` in
   per-thread registers across the KV-tile loop and rescales the O
   accumulator after every tile. Coop matrix accumulators are
   workgroup-cooperative `coop_mat<f32, C>` tiles, not per-thread
   scalars — the rescale step has to operate on whole accumulator
   tiles and must coordinate with the per-row max held by
   subgroup-cooperative threads.

2. **Tile shape mismatch**. Coop matrix on Blackwell wants 16x16 tiles.
   Flash today uses BQ × BKV = 128 × 8. To produce the score tile via
   `coop_mat`, BKV needs to grow to ≥16 (and ideally a multiple of 16
   for the QK^T side and the PV side both). Increasing BKV to 16
   doubles shared K storage (8 KB → 16 KB at hd=64) and, more
   importantly, doubles the size of the per-thread softmax/normaliser
   state that crosses the inner-loop boundary.

3. **K layout for QK^T**. The score is `Q @ K^T`. coop_mat MMAs are
   `C += A @ B`, so the K matrix wants to be staged as `[BKV × hd]`
   (already what `shared_k` is), but the *coop_mat* type for it is
   `coop_mat16x16<f16,B>` — so we need `gen_matmul_coop_wgsl(BT)`'s
   transposed-vec4 staging, not the direct vec4 staging we have now.
   PV (the second matmul) needs the opposite — V staged in normal
   B-form. Two staging styles in one kernel, both vec4-aligned.

4. **f16 input + f32 accum**. The matmul_coop kernels cast f32→f16 at
   the staging step and the C accumulator stays f32. Flash's
   numerical model assumes the per-tile score is f32 throughout; we
   need to verify gradient correctness when intermediate softmax
   factors round-trip through f16. Likely fine for inference, riskier
   for backward.

5. **Non-square output tile**. matmul_coop emits a 32×32 output tile
   (output_tile=2 × 16). Flash's BQ=128 / BKV=16 score tile is 128×16,
   not square. We'd build it from 8 separate `coop_mat` outputs —
   doable but the bookkeeping isn't free.

## Suggested approach (in priority order)

* **Phase 0 — unit-test scaffold.** Stand up a flash-coop variant
  behind `MEGANEURA_FLASH_FWD_COOP=1`, dispatched only when
  `gpu.capabilities().cooperative_matrix.f16_tile >= 16`. Existing
  scalar kernel stays the default until the variant proves out.

* **Phase 1 — `coop_mat` for QK^T only.** Keep the softmax + PV in
  scalar form; only replace the score computation with a single
  `coop_mat16x16<f16>` MMA per BKV chunk of 16. Validate vs
  `flash_attention_seq128_correctness` first. This alone should claim
  most of the gain because QK^T is the inner-loop hot spot.

* **Phase 2 — `coop_mat` for PV.** Stage P (the post-softmax weights)
  into shared memory, then issue a second MMA. This is where the
  staging-style mismatch shows up; expect a day on the layout.

* **Phase 3 — kill the scalar fallback** once vendor coverage is
  demonstrated (NVIDIA + AMD RDNA3 + Intel Xe-HPG; Apple Silicon
  takes the simdgroup_matrix path which we already exercise for
  `matmul_coop_8x8_f32`).

## Why this commit doesn't do it

The above is honestly two-three focused days of work plus a careful
correctness / numerical validation pass. Squeezing it into the
in-flight pipeline-stats branch would either land a half-finished
kernel (high risk of correctness regressions in a load-bearing
pass) or starve the rest of the plan of attention. The supporting
infrastructure in this branch — pipeline-stats measurement, register
budgets, per-kernel EPT, the e-graph cost model — is what the
coop-matrix kernel will lean on once it exists, so landing those
first is the right ordering.

## Reference: what already works for tensor-core kernels

* `src/codegen.rs:gen_matmul_coop_wgsl_prologue` — production
  cooperative matmul, both vec4 staging styles, f16/f32 paths.
* `src/shaders/conv2d_grad_input_gemm_coop_3x3.wgsl` — hand-written
  3×3 conv-backward kernel using `coop_mat`. **Currently dead code:**
  no compile.rs path emits `Conv2dGradInputGemmCoop3x3` even though
  the ResNet-50 profile (commit `7ebcd48`) shows
  `Conv2dGradInputGemm` eating 34% of training time. Wiring this in
  is a smaller win than flash but a lot less work — a candidate
  side-quest.
* `examples/analyze_shaders.rs` — the existing sweep already prints
  register counts for these kernels; comparing the coop-flash variant
  against the scalar one will be one extra entry.
