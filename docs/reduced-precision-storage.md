# Reduced-precision storage: status and plan

## Native BF16 foundation (October 11, 2026)

The experimental BF16 path uses the native Naga extension and Blade Vulkan
capabilities, not packed-u32 emulation. `Graph::to_bf16` stores two bytes per
element with nearest-even GPU rounding; `from_bf16` widens exactly. Explicit
BF16 operands work with `matmul`, `matmul_at` and `matmul_bt` (and an F32
addend), producing F32 outputs and accumulation. Master parameters and their
gradients remain F32. By default, backward widens the quantized forward operands
and uses the existing F32 derivative path. Opt-in
`CompileOptions::bf16_matrix_gradients` uses BF16 derivative operands for
explicit BF16 forward matrices only, with F32 accumulation and master gradients.
It does not demote other F32 operations or enable F16 derivative inputs.
The graph transforms are currently intended for training. Inference would
convert every F32 parameter to BF16 on each run; persistent BF16 parameter
storage and direct BF16 model loading are not implemented yet.

Dense BF16 products use the same `MatMul`/`MatMulAT`/`MatMulBT` shader groups,
addend layouts and bindings as F32/F16 products. `Kernel::CooperativeTyped`
records the operand, multiplication and output types (`codegen::MatrixTypes`)
and K partitions. Its checked cooperative loop stages operands in the
multiplication type, handles partial tiles and rounds a BF16 output to nearest
even; split-K partials remain F32 until reduction. Kernel selection and tuning
preserve this storage contract. Typed products are reported as cooperative
matrix kernels in profiles.

This is not autocast or complete mixed-precision training. BF16 graph inputs,
parameters, outputs and unsupported operations reject explicitly; only internal
conversion, pointwise, dense-matrix and convolution storage is supported. Check
Blade's `shader_bfloat16` and `bf16_f32_shapes`; cooperative products select
`[16,16,16]` or `[8,16,16]` in M×N×K order. The latter enables Xe2 (B570):
eight rectangular tiles cover the same 32×32 output block, with the same
checked tails and 32-element K partitions. Dense products, fused addends and
implicit-GEMM convolutions use this shared loop. Devices without those features retain the existing
F32/F16 paths. No default graph is automatically narrowed. Explicit BF16
matrix stores and pointwise conversion boundaries fuse: pointwise arithmetic
widens inside the kernel and rounds once at its store. Shared values needed by
F32 consumers or external outputs stay available. Reshaped BF16 activations
retain F32 master-gradient views. `Graph::into_bf16_matmuls` and
`Graph::into_bf16_convolutions` select supported regions before autodiff;
general layouts, attention and automatic whole-model precision selection remain
work. This is not universal BF16 activation storage.

On RTX5080/580.178.04, 1,035 exact conversion cases and 18 F64-referenced matrix /
F32-gradient fixtures pass (three layouts, optional addend, full/partial/scalar
dimensions). The driver's plain BF16 conversion truncates; the narrowing kernel
rounds before casting. An odd-length GPU clear originally failed Vulkan's
multiple-of-four requirement; clears now round within existing padded buffers.
The pre-existing workgroup-layout VUID10684 remains recorded. These are
primitive checks, not a speedup claim. `tests/bf16_storage.rs` covers the path;
unsupported hardware skips only its GPU-specific cases.

The connected follow-up passes 18 reduced-gradient matrix cases (three layouts,
optional addend, derivative scales 1e-20/1/1e20), both fused/unfused F64 checks
and 32 synthetic Adam updates. Its two-layer clamped-activation fixture's loss
falls .16861932 -> .03437764; master parameters and both moments remain finite.
One additional GPU guard passes in 9.76s of child execution, with no allocation
warnings; the known workgroup-layout message remains. CPU checks include 16
pointwise storage combinations, 12 matrix store/layout/addend combinations,
protected outputs and aliased BF16 views. This is not a full learner or a
performance/learning-quality claim.

## Historical F16 weight-storage work

PR #242, branch `experiment/f16-weight-storage-2026-10-10`, based on `09fd74f`.

## What exists

- `DType::F16` tensors, `Graph::parameter_f16`, and `Op::ToF16` (a cast whose
  backward is the identity, so an f32 master parameter receives the f32
  gradient).
- Matrix kernels read f16 **weights** (`WeightFormat::F16`) in the scalar,
  split-K and matrix-vector paths, and `Embedding` reads an f16
  table. Uploads convert f32 data to f16 (`runtime/transfer.rs`).
- The accelerated contract feeds f16 *inputs* to cooperative matrix tiles,
  converting f32 operands inside the shader; storage stays f32.

## What did not exist in that F16 study

- **Activations are always f32.** Every intermediate buffer, attention,
  normalization, pointwise and convolution kernel reads and writes f32.
- **No bf16 at that revision.** GGUF bf16 tensors were widened to f32. The
  native path documented above was added later.
- **f16 weights disabled cooperative tiles** at `3102683`: `Pipelines::key`
  checked reduced storage before cooperative kernels. Fixed in this branch.
- **No f16 training path.** Nothing keeps f32 master weights with f16 copies
  for forward and backward, and there is no loss scaling.

## In this branch

1. `Graph::store_weights_f16()` converts every f32 parameter whose only uses
   are the weight operand of `MatMul`, `MatMulBT`, `FusedMatMulAdd`,
   `FusedMatMulBTAdd` or an embedding table. Inferena's runner applies it to
   inference sessions under `INFERENA_F16_WEIGHTS=1` (accelerated contract
   only). Training is untouched.
2. f16-input cooperative tiles read f16 weights directly: the plain and
   prologue pipeline keys carry the weight format, the generators declare B as
   `array<vec4<f16>>`, and `kernels::matmul` admits f16 weights for plain
   (not compensated) f16 tiles. `tests/coop_f16_weights.rs` checks plain and
   transposed-B products and ReLU epilogues with partial M/K tiles against an
   f64 reference, and mixed f32/f16 RmsNorm projections against scalar
   execution after uploads.
   Hardware capabilities determine whether cooperative selection is expected;
   a missing cooperative dispatch on supported hardware is a failure, not a skip.

Native-f32 cooperative kernels still require f32 weight storage. Session policy
prefers those tiles when available, including under `AllowF16`, so stored f16
weights use scalar kernels on those devices. Half-sized storage is not a promise
of faster execution; widening stored halves into native-f32 tiles remains work
for a separate change.

Measurements: `/x/Code/inferena-results/f16-weights-20261010`.

## RX 7900 XT validation (October 11, 2026)

RADV exposes 16-wide f16 cooperative tiles and no native-f32 tiles on this
device. The new path is exercised, not a scalar fallback. The targeted f16,
RmsNorm-prologue, GEMV-parity and skinny-cooperative suites pass (33 tests).
Ragged f16 products, including fused ReLU, have relative L2 error below
`8.3e-4` against the f64 oracle using f16-rounded weights. Mixed f32/f16
prologues use distinct pipelines and remain correct after parameter updates.

Release microbenchmarks compare f32 and f16 **storage**, both under `AllowF16`,
on one shared context with search and GPU timestamps disabled. Each variant
warms for 250 ms; 12 alternating ABBA/BAAB rounds time completed batches of
16 replays. Construction, uploads and readbacks are excluded. The table shows
the median latency across three independent runs and the run-to-run speedup
range; times include CPU recording/submission. No foreign GPU clients were
observed. These repeatedly reuse the same weights and are not whole-model
latency measurements.

| Case | M × K × N | f32 µs | f16 µs | f32/f16 speedup |
| --- | --- | ---: | ---: | ---: |
| dense | 512 × 512 × 512 | 31.52 | 31.16 | 1.01–1.02× |
| dense-bt | 512 × 512 × 512 | 35.86 | 33.25 | 1.07–1.08× |
| ragged | 257 × 65 × 512 | 10.88 | 10.62 | 1.02–1.08× |
| rms-projection | 257 × 576 × 1536 | 91.27 | 86.93 | 1.05–1.05× |
| relu-epilogue | 512 × 512 × 512 | 34.97 | 34.41 | 1.01–1.02× |
| small-projection | 32 × 576 × 192 | 21.56 | 41.47 | 0.51–0.52× |
| vocab-prefill | 128 × 576 × 49152 | 353.04 | 335.52 | 1.05–1.06× |
| vocab-decode | 1 × 576 × 49152 | 168.67 | 33.76 | 4.96–5.02× |

Cooperative cases match the f32-storage outputs exactly: both kernels feed
f16-rounded operands to the tiles. Scalar cases have relative L2 below
`2.2e-4`. The small projection loses the f32 small-tile path and falls back to
the general f16-weight kernel, making it about 1.9× slower. Keep this an
explicit storage/accuracy tradeoff, not a blanket latency optimization.

Debug Vulkan validation reports `VUID-StandaloneSpirv-None-10684` for workgroup
array layout. The same error reproduces on base `09fd74f` in
`coop_matmul_skinny::matmul_non_aligned_k18`; numerical tests pass, but this is
not a validation-clean run. It is a separate pre-existing SPIR-V issue.

## Remaining stages from the F16 study

1. **Search over storage.** Let measured search choose f32 or f16 storage
   per weight instead of a session-wide switch.
2. **f16 activations at matrix boundaries.** A storage dtype per buffer in the
   memory plan; matrix kernels write f16 outputs with f32 accumulation;
   normalization, softmax and reductions read f16 and accumulate in f32.
   Pointwise kernels are generated, so f16 I/O is a codegen parameter there.
3. **Mixed-precision training.** f32 master parameters with f16 copies made
   once per step (`ToF16` already has the right backward), f16 activations
   saved for backward, f32 parameter gradients, and a static loss scale
   folded into the loss gradient. Derivative regions that need full precision
   keep their current marking.
4. **bf16.** The native foundation above supersedes the proposed packed-u32
   emulation. Automatic whole-model selection and broader operation coverage
   remain unfinished.

Each stage needs its own numerical gate class: bf16 autocast in PyTorch
already misses the 1% sampled-output gate on SmolLM2 (1.02% on the RTX 5070).
