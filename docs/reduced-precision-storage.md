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
and K partitions. Square dense products share the ordinary FP16 generator's
vector staging and 2x2 register tiles, with operands staged in their
multiplication type. Vector bindings are used only when both operands contain
whole vectors; odd strides and K tails use masked per-lane loads. Incomplete
bindings retain the checked kernel: scalar staging in the fast kernel was
slower on the tested ragged AMD products. F32 outputs with whole 16x16 tiles use direct
cooperative stores; output tails, addends and reduced outputs use checked
stores from shared memory. BF16 outputs round to nearest even; split-K
partials remain F32 until reduction. Kernel selection and tuning
preserve this storage contract. Typed products are reported as cooperative
matrix kernels in profiles.

The staged square kernel and the rectangular/implicit-GEMM fallback share the
same 32-element K partition boundaries. Pipeline keys include vector-binding
and direct-output specialization, independently of the uniform partition
count. The generic code-generation API retains the checked implementation;
runtime construction specializes from the actual product dimensions.

Two explicit F16 operands select the same typed kernel with F16 multiplication,
F32 accumulation and the same K partitions, given Blade's `shader_float16` and
an F16/F32 cooperative shape. The F16 contract also includes subnormal operands
(2^-24 to 2^-14), which a listed shape does not promise: session construction
runs a one-time exact probe per context and rejects devices that flush them.
Blade requests SPIR-V `DenormPreserve 16` where supported, which lets B570
preserve these values. `runtime::typed_matrix_supported` reports all three
requirements. BF16 subnormals lie in F32's range and follow its handling. The
ordinary dense kernels honor reduced storage
only in B (`WeightFormat`), so any other F16 left operand, and any reduced
addend, is rejected at compile time instead of being read as F32. There is no
F16 widening for derivatives, so these products are forward-only.

Typed products are not schedule-locked: kernel selection keeps their arithmetic
and storage contract, and bounded tuning varies their tile geometry and K
partition count. On devices with workgroup cooperative matrices, candidates
include square tiles and both orientations of rectangular tiles through
128×256 and 256×128, with 32- or 64-element K stages. Matching F16/BF16 operands
load directly; operands needing conversion stage in shared memory. NN, AT and
BT products include fused addends and F32/F16/BF16 outputs. Reduced outputs
round once after accumulation and addition. Device granularity, invocation
count, shared memory (including driver reservations), alignment and binding
capacity filter candidates before compilation. Ragged and unsupported forms
retain the checked subgroup kernel, including B570's 8×16×16 native shape.

A candidate must keep the product's SumRows contract (split
products stay split) and fit the compiled partial buffer, so tuning can lower
but not raise a compiled count. A product without any applicable partition or
tile candidate is left out before the class limit and pending-work checks.
Qualification uses private scratch in the operands' storage types and
references built from operands rounded to the multiplication type; F16
multiplication uses a 1/1024 small-input probe instead of 1e-12.

Automatic F16 kernels also compare workgroup tiles with identical arithmetic.
This does not introduce F16 multiplication into an F32 incumbent. Strict F32
and protected derivative regions retain their existing precision. A kernel
winner still needs whole-program qualification when used by measured build
search; isolated kernel correctness and speed are not model acceptance.

Measured construction automatically considers BF16 multiplication under
`CoopPolicy::Auto`, alongside its existing F32/F16 paths. Checked F16 candidates
also reach layouts left scalar by the ordinary path, on devices with square
or rectangular native tiles.
`NativeF32` and `Disabled` retain their ordinary precision/selection restrictions;
`AllowF16` adds typed F16 candidates alongside its existing F16 paths.
There is no separate BF16 opt-in. `BuildSearchReport::matrix_inputs` records the
additional arithmetic allowed by the runtime policy. Device probes filter
unsupported types and report them in `unsupported_matrix_inputs`.

BF16 alternatives cover eligible forward dense products and convolutions,
keeping F32 storage, output and protected derivatives. The ordinary program
remains the first fallback. Arithmetic alternatives enter under the existing
time/program bounds and require the caller's full-program accuracy and timing
gates. Smaller scalar incumbents can expose workgroup tiles through these typed
alternatives. Permitting BF16 is not a claim that it is selected or faster.
Arithmetic alternatives may replace scalar layouts extracted during the same
search, preserving split-K partial buffers and reductions. Graphs with
caller-supplied matrix layouts keep those layouts; protected derivatives and
unsupported fusions are also left unchanged.

This is not autocast or complete mixed-precision training. BF16 graph inputs,
parameters, outputs and unsupported operations reject explicitly; only internal
conversion, pointwise, dense-matrix and convolution storage is supported. Check
Blade's `shader_bfloat16` and `bf16_f32_shapes`; cooperative products select
`[16,16,16]`, `[8,16,16]` or `[8,8,8]` in M×N×K order. The rectangular tile enables Xe2 (B570):
with 32-lane subgroups, four logical 16×16 tiles each keep two accumulators, sharing
staging and barriers across both halves. Apple's 8×8×8 products additionally
keep two column accumulators and advance K in eight-element steps. They cover the same 32×32 output
block, with checked tails and 32-element K partitions. Dense products, fused addends and
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

## M3 checked cooperative search (October 22, 2026)

The shared checked kernel accepts Apple's native 8×8×8 tile for F32, F16/F32
and BF16/F32. Each 16×16 logical quadrant uses four native accumulators on the
M3's 32-lane subgroups; staging, F32 accumulation, checked stores, convolutions
and 32-element K partition boundaries remain shared with the wider tiles.
Existing Blade/Naga pins supply the Metal capabilities and native BF16 shader
support. Kernel memo revision 12 discards decisions from the older generator.

Inferena's five-model native diagnostic passes all 14 session qualifications,
including full outputs/gradients and final repeatability. Neither reduced
format is reported unsupported; 36 BF16 candidate programs qualify. Twelve
selected programs use cooperative matrices (the two small latency sessions
keep scalar kernels). BF16 improves ResNet inference from 42.00 to 36.72 ms
in paired search, before F16 improves further to 34.53 ms and wins. None of
the final selected programs use BF16. This establishes usable search coverage
and a selective improvement over F32, not a BF16 advantage over F16 or a new
P3HPC publication cohort. Fresh compiled PyTorch qualification and independent
F32-oracle comparisons also pass for all five models' outputs and gradients.
Separate-process timings were variable.

BF16 remains useful when range disqualifies F16: the existing Auto-search
regression multiplies 65536-valued operands by 1/65536, producing exact 32s.
The F16 candidate fails the unchanged gate, while BF16 qualifies on this M3.
Both formats retain their own storage and arithmetic contracts; neither is
silently substituted for the other.

Source archives, executable hashes, raw samples and search/qualification
receipts are preserved outside Git under Inferena's
`inferena-results/m3-coopmat-20261022-{initial,models,reference}`. See Inferena's
`EXPERIMENT.md` for reproduction and the comparison scope.

## Shared dense staging on RX 7900 XT (October 16, 2026)

Release microbenchmarks against PR #257's `9b6b41a`, using process-local
Mesa 26.2.4 with `RADV_EXPERIMENTAL=bfloat16`, put a 1024-cubed BF16
multiply at approximately 100 us instead of 350 us. Including the graph's
F32-to-BF16 conversions, the GPU dispatch sum is approximately 115 us instead
of 360 us. Explicit F16 reaches the same approximately 100 us kernel range
as the ordinary promoted-F16 path. These are repeated-input microbenchmarks,
not whole-model speedups, and clocks were not pinned. Odd-sized bindings keep
the checked fallback after measuring a scalar-staging regression there.

RADV pipeline statistics for the large typed kernel change from 8192 to 2048
bytes of LDS and 1204 to 584 instructions, with no spills. The ordinary FP16
kernel's statistics are unchanged. F16 and BF16 typed kernels have matching
statistics. Kernel partition tuning remains bounded; the tested challengers
qualified but kept their original counts.

The AMD capability query lists only 16x16x16, subgroup-scoped cooperative
matrices: no rectangular or workgroup-scoped tiles. Experimental Mesa exposes
BF16 types and cooperative matrices, but `shaderBFloat16DotProduct` is false.
The system Mesa 26.0.3 does not expose BF16. Subgroup-size control (32--64),
64 KiB workgroup shared memory and pipeline executable statistics are exposed.
Blade already exposes statistics through capture-enabled contexts, but does
not request an explicit subgroup size. No driver installation or device-wide
settings were changed for this work.

Initial staging validation: 449 library tests passed. The AMD regression run
passed 147 tests with 18 ignored and five numerical failures also present before
the staging commit (four BF16 tests and this PR's own F16 exactness test).
That comparison did not resolve the failures: the F16 test's stronger-than-native
arithmetic assumption is addressed below, not attributed to experimental Mesa.
New poisoned-buffer checks cover 126 typed shape/layout/
addend/output combinations, including vector and scalar bindings, K tails,
checked/direct output stores and split-K. The ignored BF16 convolution test
also passes. Rectangular code generation is validated through SPIR-V emission;
it was not measured on a rectangular-matrix GPU here.

## Native matrix numerical contract

Explicit F16/BF16 products opt into native cooperative arithmetic. Operand
storage and conversion retain their specified format/rounding; accumulators
and split partials are F32. **F32 accumulator storage is not a promise of
bit-exact scalar IEEE F32 FMA arithmetic.** The Vulkan specification leaves
cooperative matrix internal precision and operation order implementation-dependent:
[Precision and operation of SPIR-V instructions](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html).
Results need not be bit-identical across devices, tile shapes or partition counts.

On RX 7900 XT, a single native WMMA instruction computing `1*1 + (-1)*0`
returns `0x3f7fffff` (0.9999999403953552), not `0x3f800000` (1). The same
behavior occurs with F16 and BF16 operands in standalone HIP and through both
Meganeura implementations. System Mesa 26.0.3 and private Mesa 26.2.4 produce
identical F16 results. The original 33x65x1031 exactness fixture has maximum
absolute error 3.948807716369629e-7. Uploaded inputs and the split reduction
are correct; the error already exists in matrix partials. This is native
instruction behavior, not an experimental-Mesa-only issue or accumulation
ordering. Whether AMD considers it an erratum remains unconfirmed.

Meganeura's finite, non-overflowing/non-underflowing native-matrix validation
budget is defined by `reference::native_matrix::error_bound`:

```
u = 2^-24                         # F32 unit roundoff
q = 2*K + 1
gamma(q) = q*u / (1 - q*u)        # requires q*u < 1
allowed = gamma(q) * (sum(abs(a[k] * b[k])) + abs(addend))
```

The reference uses operands already rounded to the multiplication format.
This deliberately adopts the conservative serial multiply/add rounding budget
as an **acceptance policy**, not a theorem about undocumented matrix hardware
or a bound inferred from one observed error. It uses F32, not F16/BF16 epsilon.
The sum of absolute products handles cancellation; no absolute or tensor-wide
floor excuses loss of small results. Each partial is checked with its own K
range and the final output with the full K (not K times the number of splits).
An invalid or uninformative budget is rejected. Reduced outputs are checked
against the rounded endpoints of the allowed interval: native arithmetic can
cross a BF16 rounding midpoint even when the conversion itself is correct.

This budget applies to primitive native matrix validation, not automatically
to a whole differentiated graph, nonlinear pipeline, or benchmark. Those have
separate reference/magnitude checks. Special values and underflow need separate
coverage. Storage conversion remains bit-exactly tested; positive exact layout
fixtures retain strict equality; signed/cancellation/scaled fixtures exercise
the native budget. F16 subnormal-input preservation remains a separate required
capability, with its existing rejection probe and regression unchanged. The
probe is not an exhaustive numerical qualification of arbitrary inputs.

`CoopPolicy` controls implicit cooperative selection for ordinary F32 graphs;
it does not widen explicit typed products into a strict fallback. Callers who
need to avoid native matrix arithmetic must use F32 operands and
`CoopPolicy::Disabled`. This does not promise identical reduction order across
all scalar implementations. No exact-native option or automatic fallback is
currently provided. Inferena's precision policy and benchmark thresholds are
unchanged by this contract.

Follow-up validation on RX 7900 XT: all eight typed-matmul tests pass with
system Mesa (BF16-specific cases skip there); the private Mesa full regression
run passes 165 tests, ignores 21, and still fails three existing BF16
connected-graph/reduced-gradient tests. Typed signed arithmetic and split-K
primitive checks now pass. Those three higher-level failures are not waived
by this primitive budget and their assertions remain unchanged. All-feature
library tests pass (609 passed, 7 ignored), as does strict Clippy. RTX 5070 and
B570 hardware follow-up remains outstanding.

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
