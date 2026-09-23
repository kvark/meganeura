# Shader and lowering audit — September 23, 2026

A read of every hand-written WGSL shader, the generated kernels they sit
beside, and the lowering and autodiff that feed them. Base: `3015e1d`.

Evidence boundary: every change was checked on lavapipe (Mesa 25.2, LLVM 20)
against CPU f64 references or central finite differences. The regressions
live in [`tests/shader_audit.rs`](../tests/shader_audit.rs), and each one was
confirmed to fail before its fix where a fix applies. The existing library,
regression and smoke suites pass, apart from the two fused-versus-expanded
bit-exact tests that also fail on `main` under lavapipe (CI skips them there
with `MEGANEURA_SKIP_BIT_EXACT`). **No change was timed on a real GPU.** The
performance changes remove memory traffic, dispatches or serialization by
construction; how much each is worth on a given device is still to be
measured. None of them is claimed as a speedup here.

## Wrong results

| Finding | Consequence | Fix |
|---|---|---|
| Scalar attention backward ran 64 lanes with one dimension each | Heads wider than 64 on sequences shorter than the flash query block got zero gradient past dimension 63. The forward was right, so finite differences agreed with a loose tolerance. | Each lane owns up to four dimensions (heads ≤ 256); wider heads are rejected at compile time. |
| LayerNorm weight gradient skipped its row reduction below four rows | Two or three rows kept row 0 only and wrote past the `[cols]` output. | Only a single row writes directly. |
| MaxPool2d backward was an average-pool approximation over `stride²` consecutive flat elements | Wrong window and no argmax: every gradient upstream of ResNet's stem pool. | `MaxPool2dGrad` gathers from covering windows and routes to the first maximum (PyTorch's tie rule). |
| BCE backward divided by `p(1 − p)` unclamped | A sigmoid saturating to exactly 1.0 in f32 gave NaN while the loss stayed finite. | Floor at 1e-12, as PyTorch does. |
| GroupNorm backward variance as `E[x²] − mean²` | 8% input-gradient error at mean 50 and spread 0.1; forward and backward disagreed on `inv_std`. | Statistics computed once, with the forward kernel's two-pass order. |
| Depthwise convolution staged filters in a 49-entry array; the grid Z axis held `batch × channels` | Larger kernels overflowed shared memory; large batches exceeded the portable grid. | Rejected at compile time. |

The plan cache format is bumped, since cached plans can carry the old
lowerings.

## Traffic, dispatches and serialization

- **Uniform constants become literals.** Autodiff seeds ReLU masks, mean
  gradients and row broadcasts with full `zeros[n]`/`scale[n]` tensors,
  pinned in host-visible memory and read every step. They also used up
  input slots under the three-binding cap, so a ReLU backward took two
  dispatches.
- **The ReLU mask comes from the output.** The pre-activation then has one
  reader, can be freed after the forward pass, and fuses.
- **Broadcast pointwise loads.** `AddPerChannel`, `BiasAdd` and `BiasMul`
  are pointwise DAGs, so a conv → bias → ReLU block writes one activation
  instead of three.
- **The per-channel bias gradient no longer transposes.** It reshaped,
  materialized a transpose and ran a tall `SumRows`. That is ~50 reductions
  per ResNet-50 step, and the shape the recent `SumRows` row split worked
  around. `SumInner` reduces the contiguous planes directly.
- **GroupNorm backward.** The weight/bias kernel re-derived group
  statistics per channel, reading each activation `channels_per_group + 1`
  times (11× for a 320-channel, 32-group UNet block).
- **Attention dK/dV.** `dot(dO, O)` was recomputed per query in every KV
  workgroup, a third of the inner loop's products plus O's staging. It is
  now reduced once.
- **Gradient clipping.** Global clipping ran one single-workgroup dispatch
  per parameter with a barrier after each (~270 for SmolLM2). Adaptive
  clipping streamed each parameter through one workgroup. Both now use
  per-workgroup partial slots and a fixed-order finish.
- **Smaller items.** Norm weight gradients fold row blocks into partials.
  `GlobalAvgPool` gives wide planes a workgroup each. Transposes stage
  through a shared tile. Cross-entropy reads the logits twice. Large
  `SumAll`/`MeanAll` split across workgroups. Softmax backward broadcasts
  without a K = 1 matmul.

## Abstraction notes

The recurring pattern is a reduction whose parallelism was picked for one
shape and then left alone: one workgroup per tensor (clipping, whole-tensor
sums), one thread per row (pooling), or recomputation per consumer
(GroupNorm statistics, attention row dots). The generated reduction template
already expresses most of these. Where a hand-written kernel remains, it
should say which axis it parallelizes and why.

"Partials, then a reduction" now exists in several forms: matmul split-K,
convolution weight split-K, the `SumRows` row split, norm row blocks, clip
partials and whole-tensor sums. Each chooses its own split count and
scratch accounting. A single partial-reduction lowering that owns both,
and exposes the split to the measured search, would replace five
heuristics. Two are new in this audit (norm blocks keep at least 1024
workgroups; whole-tensor sums use 16K elements per workgroup), and the
`SumRows` split constants are another. None is measured.

## Not done

Ranked by expected value:

1. **Fuse bias and activation into the convolution store.** The block is now
   conv plus one pointwise pass; an epilogue on the implicit-GEMM store
   removes the remaining round trip. It needs a bias binding in the conv
   kernels and their tuner classes.
2. **Sparse cross-entropy labels.** SmolLM2 training feeds dense one-hot
   `[tokens, vocab]` f32 labels: a vocabulary-sized tensor per token,
   uploaded and read every step. A class-index form removes it.
3. **Unify partial reductions** as above, with the split count measured.
4. **Fold the clip scale into the optimizer.** Scaling gradients in place is
   a full read and write of every gradient; the optimizer could apply
   `min(1, max_norm / norm)` as it reads them. That changes what
   `read_param_grad` returns after a clipped step, so it needs an API
   decision.
5. **Gradient accumulation waits on the CPU every step.** A mid-step
   submit-and-wait works around an observed write/read hazard; the pass
   boundary should already order it. Find the missing barrier and remove
   the wait.
6. **Grid limits.** 1D dispatches exceed 65,535 workgroups past ~16.7M
   elements. That works on NVIDIA and AMD and happens to on lavapipe, but
   Intel ANV and lavapipe both report 65,535, so it is undefined there.
   Fold large grids into two dimensions.
7. **Chunked GroupNorm forward statistics** combine raw `(sum, sumsq)` per
   slice, so they cancel for large means (clamped, so finite). Per-slice
   shifted sums with Chan's combination need a per-row shift input in the
   reduction template.
8. **Save norm statistics from the forward.** RMSNorm and LayerNorm backward
   recompute per-row statistics in both the input- and weight-gradient
   kernels, as attention saves its LSE.
9. **Folded constants stay allocated** (host-visible, pinned) so `read_node`
   can return them. Allocate them lazily, or device-local, once no dispatch
   reads them.

## Lavapipe hazard

Indexing a shared-memory array with a value derived from a runtime loop
bound, inside that loop, crashed lavapipe's multithreaded JIT (SIGSEGV;
single-threaded `LP_NUM_THREADS=0` was fine). The same index under a
constant loop bound with an early `break` works. The norm weight-gradient
kernels use that form; keep it in mind for new kernels, since CI runs on
lavapipe.
