# Reduced precision work plan

Updated October 25, 2026. First priority: let measured search retain reduced
operand and activation storage, with F32 accumulation. Keep the staged F32
candidate available and select by qualified complete-program time.

## Survey complete

- [x] Query RTX 5070 / NVIDIA 595.91.07 and Arc B570 / Mesa 26.0.3 directly.
- [x] Compare advertised types with Blade's filters and Meganeura's generators.
- [x] Inspect executed PyTorch instructions with Nsight and a Level Zero
  native-module capture, rather than infer accumulation from tensor dtypes.
- [x] Measure isolated accumulation changes and two-matmul storage chains on
  both GPUs, with independent numerical checks and retained samples.

Both drivers expose F16 accumulation; B570 also exposes BF16 accumulation.
NVIDIA additionally exposes FP8 and both drivers expose integer cooperative
products. The inspected default PyTorch F16/BF16 GEMMs accumulate in F32.
NVIDIA's explicit F16-accumulation option helps larger products, but changing
only Meganeura's accumulator yields much smaller gains. Reduced storage is the
stronger immediate lead. These are microbenchmarks, not full-model results.

Survey baseline: Meganeura `5b232bda2b8393ac3f51d4d8e596732b07544310`,
Blade `eae0c6f7aa3167c79d1135050cf8e17c17fabd0a`, PyTorch 2.13.0
`cf30153c`. Raw queries, native binaries, disassembly, sources and samples are
retained outside the repository in `matrix-survey-20261025/survey-summary.json`
under the local Inferena results directory. Keep that evidence separate from
the frozen paper cohort.

## 1. Automatic matrix storage — qualified implementation

The first implementation, which materialized reduced operands in separate
conversion dispatches, regressed the measured model inference cases on both
GPUs. Sharing only reused casts did not establish a useful win either. Keep
these rejected samples in the survey archive. The producer-only candidate narrows stores
inside existing private producers, including pointwise and normalization
reduction epilogues, without adding conversion dispatches.

The producer-only candidate passes the existing output/gradient gates and
20-step SGD checks on both GPUs. It regresses NVIDIA SmolLM2 inference by 8%; matched dense retuning does not
remove that regression. Keep the measured staged fallback. The Intel SmolVLA
gain in the early study did not reproduce with the direct-output backend;
retain both samples rather than claim that gain.

- [x] Evaluate sharing F16/BF16 operand conversions. Standalone conversion
  dispatches did not produce a useful model speedup; keep their evidence.
- [x] Implement an experimental pass to narrow private matrix outputs in their producer when every reader
  already needs the same reduced format. Preserve F32 outputs needed by
  backward or another consumer.
- [x] Keep graph inputs, parameters, outputs, optimizer state, protected
  precision regions and explicitly pinned values at their declared types.
- [x] Add this as a measured implementation choice alongside staged F32
  storage. Do not spend the entire tuning budget on a larger type frontier.
- [x] Check full outputs and gradients on both GPUs, then compare complete
  inference/training steps on representative models. Retain losing cases.

Use existing shaders and serialization where possible. F32 accumulation and
split-K partials stay unchanged. The producer-store candidate keeps parameter
loads in F32 and adds no per-step casts. Persistent weight caching is separate
work with explicit invalidation requirements.

With the rectangular load fix below, a fixed-plan paired comparison improves
B570 SmolLM2 inference from 8.764 to 8.213 ms (6.3%). NVIDIA still rejects the
producer-store candidate on performance: 3.759 to 4.069 ms (8.2% slower).
Fresh 30-second searches reach and select the Intel storage choice and retain
F32 storage on NVIDIA. A separate paired holdout of the saved winners gives
8.770 to 8.239 ms on Intel (6.1% faster) and 3.744 to 3.713 ms on NVIDIA (0.8%,
within the small changes seen between tuning runs). Both searches finish in
31–33 seconds including setup. Submission counts match in each holdout pair.

## 2. Broaden storage coverage — active

The next prototype specializes private SwiGLU/GeGLU output stores in the
consumer's F16/BF16 type, without another dispatch. Forward arithmetic stays
F32 and backward readers prevent narrowing. GPU output/gradient/update checks
pass on both cards; complete-model timing is in progress. Forward attention
and batched/block storage remain subsequent opportunities.

The forward attention generator currently requires square 16x16 matrices, so
B570's 8x16 product is not admitted there; the rectangular scaled-backward path
is separate. Its cooperative forward kernel also multiplies QK on matrices
but computes PV with scalar F32 operations. Investigate a native 8/16-row QK+PV
candidate, retaining online softmax/statistics and accumulation in F32, and
qualify F16 P/V operands against the unchanged output and training gates.
This is a separate algorithm experiment from narrowing the attention output.
Keep tails, causal/window masks, GQA, head padding and LSE storage explicit.

- [ ] Extend useful storage chains through pointwise operations and compatible
  fusion without adding redundant widening/narrowing dispatches.
- [ ] Cover additional NN/AT/BT, batched/block and split-reduction forms where
  profiles justify the generator and qualification work.
- [ ] Investigate cooperative matrix conversions for direct reduced result
  stores, avoiding an F32 shared-memory round trip. Naga currently rejects
  matrix casts; each intermediate matrix type must be supported by the device.
- [ ] Consider immutable reduced weight caches only with upload, sharing,
  checkpoint and optimizer-update invalidation defined and tested. Measure
  device-local and host-visible placement independently.

## 3. Intel loading and tile geometry

- [x] Repair dropped direct-output layout information in checked rectangular
  GEMMs and enable direct stores for aligned checked convolutions: [PR #286](https://github.com/kvark/meganeura/pull/286).
  The same-plan model comparison improves B570 SmolLM2 inference by 2.6% and
  SD U-Net by 1.3%; other affected cases change by at most 1%.
- [x] Test native vector bindings as a load-alignment optimization. All twelve
  measured NVIDIA native/mixed-operand cases are within 0.4%; do not ship this
  as a performance change. Cooperative load strides count pointer elements,
  so a vector pointer also requires changing the stride, not just its index.
- [ ] Match shapes, operand precision and accumulation before comparing the
  XPU and Vulkan GEMMs' instructions, loads, staging and register use.
- [x] Investigate the observed XPU SIMD16 / no-shared-memory kernel versus
  our SIMD32 staged kernel. A diagnostic Blade patch enables Vulkan subgroup
  control and explicitly requests 16: shaders compile and qualify. On matched
  128x576x576 products, fully native F16/BF16 loads are 23–92% slower at SIMD16;
  staged results vary slightly with layout/type. Keep the default width.
  The `cs16` override failure alone did not establish lack of support.
- [x] Evaluate direct operand loads with whole-model timing.
  Keep this independent of storage-policy changes for attribution.

Direct loads of native F16/BF16 operands, retaining F32 accumulation and direct
F32 result stores, reduce isolated Intel GEMM time by 20–59% across NN/AT/BT
and two reduction lengths. NVIDIA also benefits on these isolated products.
The implementation in [PR #287](https://github.com/kvark/meganeura/pull/287)
bypasses staging independently for each native operand on aligned checked
products; F32 operands retain conversion staging. On B570, the same reduced
SmolLM2 plan improves 8.671 to 8.207 ms (5.4%). Routing NVIDIA's specialized
16x16 products through this generic path regresses the whole model by 10%, so
the final patch retains that specialization (measured change below 0.01%).
The Intel native-load shader remains SIMD32, with 281 instructions versus 400
for the staged shader and no register spills; forcing `cs16` also fails pipeline creation for the native-load shader.
The valid subgroup-control experiment above does execute at SIMD16; its native
NN shader has 347 instructions and 58 non-SSA registers, versus 281 and 24 at
SIMD32. Subgroup width alone does not reproduce XPU's kernel efficiency.

The NVIDIA workgroup regression is concentrated in mixed native/F32 operand
loads. Staging both operands recovers 7.6% on the narrowed SmolLM2 plan but is
neutral against the original F32-storage model. Deferring both matrix loads
until conversions finish also recovers part of the loss, but does not improve
the original model. Neither prototype is proposed as a production change.
Their code and positive/negative model samples remain in the survey archive.

## 4. Selective F16 accumulation

- [ ] Retain advertised accumulator/result types through Blade and lowering.
- [x] Compare F32 and F16 accumulators at fixed storage, shape and schedule.
- [x] Check long reductions, cancellation, overflow and small contributions.
- [ ] Qualify any profitable bounded candidate with unchanged model checks
  and training trajectories before admission.
- [ ] Admit only profitable, qualified candidates. BF16 accumulation is lower
  priority: the Intel probe lost accuracy without a useful latency gain.

The cancellation fixture returns zero with F32 accumulation and nonfinite
values in all 1,024 outputs with F16 accumulation on both GPUs. Summing small
contributions through K=65,536 loses 50% of the result with F16 accumulation;
Intel BF16 loses 50% at K=8,192. Random K=4,096 operands produce about 0.19%
relative L2 error with F16 and 1.58% with Intel BF16. Keep F32 as the default.
These are adversarial fixtures and isolated timings, not proof that every
reduced-accumulation workload fails. Do not expand Blade's public capability
surface until a bounded, profitable workload justifies the additional path.

## 5. FP8 and integer matrix paths

- [ ] Define scaling, rounding, packing and output contracts before adding
  new types. These are quantization policies, not substitutes for autocast.
- [ ] Start from a workload whose executed kernels or profile support the
  opportunity. No inspected default PyTorch floating-point GEMM used these.

Each implementation should remain a small reviewable change with its own
numerical and complete-workload evidence. Update this checklist as work lands;
do not silently change the paper's results, qualification bounds or submission
policy while pursuing it.
