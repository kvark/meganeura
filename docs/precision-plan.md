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

Private SwiGLU/GeGLU stores now specialize to the consumer's F16/BF16 type,
without another dispatch. Forward arithmetic stays F32 and backward readers
prevent narrowing. GPU output/gradient/update checks pass on both cards.
Fresh 30-second searches plus a separate paired holdout improve Intel
SmolLM2 from 8.228 to 8.068 ms (1.94%) over the first producer pass. NVIDIA
retains F32 storage; its holdout change is 0.84%, treated as tuning variation.
Fixed-plan complete-model results put the combined Intel storage improvement
at 7.85% (8.751 to 8.064 ms); other measured models are neutral or regress.

- [x] Extend private reduced stores through SwiGLU/GeGLU, preserving F32
  consumers and avoiding standalone conversions.

The forward attention generator currently requires square 16x16 matrices, so
B570's 8x16 product is not admitted there; the rectangular scaled-backward path
is separate. Its cooperative forward kernel also multiplies QK on matrices
but computes PV with scalar F32 operations. Investigate a native 8/16-row QK+PV
candidate, retaining online softmax/statistics and accumulation in F32, and
qualify F16 P/V operands against the unchanged output and training gates.
This is a separate algorithm experiment from narrowing the attention output.
Keep tails, causal/window masks, GQA, head padding and LSE storage explicit.

The 8/16-row prototype now uses native F16 products for both QK and PV,
with F32 online softmax, LSE, output storage and accumulation. Same-plan
inference checks pass against the independent F32 oracle on both cards.
The production-identity rerun improves NVIDIA Whisper by 10.1% and Intel SD
U-Net by 5.6%, but regresses Intel Whisper by 66.5%; these fixed-plan results
require measured selection. Other inference changes range from a 2.9% gain
to a 1.6% regression.

Training admission is being evaluated separately from the inference candidate.
Independent F32 trajectories show that
the selected baseline also drifts: at step 20 the NVIDIA baseline has 69.7%
gradient error and 5.62% loss error; Intel has 2.48% gradient error while its
loss remains close. Comparing two independently updated reduced programs did
not isolate the new attention algorithm's contribution.

At identical weights from the F32 step-20 checkpoint, the NVIDIA baseline has
20.5% gradient error and 4.46% loss error. Replacing only forward attention with
scalar F32 barely changes this. Replacing dense products with F32 restores the
fixed-state gradient/loss bounds (0.49% gradient error); a diagnostic three-F16-
product residual correction also passes (0.58%). Correcting both dense products
and attention gives 0.25% gradient error. F32 accumulation and storage remain
unchanged. These are accuracy results, not latency claims or production code.
On Intel, correcting both QK and PV restores the fixed-state bounds; the plain
candidate fails the maximum-gradient bound even though relative L2 is 0.61%.

The extra 1% gradient comparison between independently updated states also
rejects an all-F32 layout change (4.39% at step 20). Retain it as a stress
diagnostic, but do not interpret it alone as reduced-precision error. The
benchmark's existing contract uses 1% maximum/RMS gradient bounds at identical
weights and 2% per-step training/probe/held-out loss bounds over 256 real-text
updates. Run that frozen workload, with its parameter-update and learning
checks, before extending training admission. Keep all thresholds unchanged.

The frozen 256-update workload now passes on both GPUs for all six variants:
native F32, the selected native program, plain QK/PV, residual-corrected dense
and attention with F32 derivatives, PyTorch F32 and PyTorch FP16. Every
training/probe/held-out loss comparison passes against both F32 references,
with parameter updates and learning checks satisfied. The synthetic checkpoint
still exposes a real input-precision limitation: PyTorch FP16 at those exact
weights has 34.5% gradient error and 7.97% loss error too. Its F32 loss agrees
with the native F32 oracle. Neither result establishes accuracy at every state.

The compensated workgroup prototype does not establish a performance win on
this model. Its best combined dense/attention variant is 2.4% slower than the
qualified F32 program. Holding forward attention identical in each paired run,
dense correction alone is 2.9–3.0% slower (same weights, F32 derivatives and
submission count; 100 paired samples per arm after 10 seconds of warmup).
Keep the prototype and its accuracy results, but do not add a production
precision option for this result. Investigate larger shapes only if a model
profile supports the opportunity.

- [x] Validate rectangular QK/PV shaders, padded heads, query/key tails,
  GQA, causal/sliding/partial-causal masks and full F64 output/LSE fixtures.
  The long cancellation fixture (128 queries, 1024 keys, width 64) misses the
  probe's provisional 1% relative-L2 bound at 1.31%; PyTorch F16 has the same
  error on the identical inputs. Preserve that failure beside the model results.
- [x] Measure complete inference/training plans on both cards; retain losing
  models and failed trajectory checks, not only favorable kernel timings.
- [x] Give the candidate a distinct shader identity so numerical qualification
  does not mistake two algorithms for repeated executions of one plan.
- [x] Verify inference-only measured admission with fresh 30-second searches
  and independent paired holdouts, including Intel Whisper's losing case.
  Holdout: NVIDIA Whisper 4.336→3.904 ms (9.95% faster); Intel SD U-Net
  3.279→3.086 ms (5.89% faster). Intel Whisper retains the old path and changes
  by 0.03%. Submission counts match in every pair; all plans qualify before
  and after timing. The six searches finish in 22–30.4 seconds including setup.
- [x] Investigate training drift against independent F32 trajectories and
  distinguish fixed-state error from optimizer-trajectory sensitivity.
- [x] Complete the frozen 256-update real-text training gate on both GPUs,
  retaining plain, corrected and strict references and every failed gate.
- [x] Measure compensated workgroup dense products after numerical qualification,
  against a qualified F32 alternative at the same weights. The matched dense-only
  pairs regress, so no production precision API or default change is proposed.
- [x] Qualify the remaining models with their declared 32-update synthetic
  workload. SmolVLA, SD U-Net and Whisper pass on both GPUs: unchanged loss
  curves and initial full-gradient gates, nonzero parameter updates, and
  gradients checked again at identical F32 step-32 weights.
- [x] Admit training as a separate measured choice, preserving independent
  forward/backward search coverage. All eight saved winners from fresh
  30-second searches pass their declared trajectories: 32 synthetic updates
  for three models and 256 real-text updates for SmolLM2 on both GPUs. Keep
  the independent F32 references, parameter updates and thresholds unchanged.
  Same-plan holdouts, changing only forward attention, improve NVIDIA Whisper
  by 4.5% at both 4 and 8 submissions (19.953→19.054 ms at 8) and Intel SD
  U-Net by 1.84% at one submission (8.553→8.395 ms). Fresh-search holdouts
  select QK/PV for those two cases and retain other paths for the other six;
  Intel SmolVLA regresses 1.24% after search without choosing QK/PV. Do not
  attribute the other models' tuning variation to the attention algorithm.
  All 16 searches finish in 30–34.1 seconds including setup. Timing covers
  forward, loss and backward; optimizer updates are verified separately.

Dense derivative controls for transposed AT/BT products now separate
multiplication precision from storage.
They retain F32 inputs, outputs, accumulators, split partials and protection
flags while explicitly testing the existing F16/BF16 kernels on saved layouts.
Paired complete-model training changes, with fixed submission counts:

| GPU | Model | Plain F16 | BF16 |
| --- | --- | ---: | ---: |
| B570 | SmolLM2 | -6.84% | -5.65% |
| B570 | SmolVLA | +2.67% | +4.01% |
| B570 | SD U-Net | -2.32% | -1.94% |
| B570 | Whisper | rejected | -1.16% |
| RTX 5070 | SmolLM2 | -10.74% | -3.68% |
| RTX 5070 | SmolVLA | -8.27% | -4.35% |
| RTX 5070 | SD U-Net | -1.45% | +0.08% |
| RTX 5070 | Whisper | rejected | +61.91% |

Plain F16 fails Whisper's unchanged maximum-gradient gate on both GPUs
(1.963e-5 error, 7.696e-6 limit), so those candidates are not timed. The
SmolLM2 F16 plans pass all 256 real-text updates on both GPUs against native
and PyTorch F32 references. Six selected 32-update cases also pass: F16 for
SmolVLA/SD on both cards, BF16 for Intel Whisper and scaled F16 for NVIDIA
Whisper. Preserve the losing Intel SmolVLA timing beside its passing accuracy
checks. These are saved-plan opportunities, not bounded-search results.

- [x] Prototype per-K32 binary scaling with F32 bindings and accumulation.
  Vectorization and single-factor restoration improve the prototype, but its
  complete-model gains are smaller than plain F16 where plain F16 qualifies.
  Scaled NVIDIA SmolLM2 improves 2.80% and passes 256 updates; Intel regresses.
  Scanning once per partition is slower and fails a block-varying range fixture
  that per-K32 scaling passes. Mixed-range stress still rejects per-K32 scaling;
  the F32 baseline also fails the subnormal-input fixture. Retain all failures.
A separate NN control covers the long-K vocabulary-input gradient in
SmolLM2 (128×576×49,152, retaining 48 split partitions). Plain F16 fails the
unchanged gradient gate on both cards: RMS error is about 7.3e-5 against a
3.68e-5 limit. Consequently, extending the passing AT/BT candidate to every
derivative product is not valid. BF16 qualifies before/after timing at fixed weights
and changes complete-step time by -3.23% on Intel and -0.27% on NVIDIA.
Combining BF16 NN with the qualified F16 AT/BT plan improves another 3.72% on
Intel and 0.36% on NVIDIA. Both combined plans pass the unchanged 256-update
real-text training/probe/held-out checks against native and PyTorch F32.
Keep the NN scope separate from the AT/BT gains in the table above.

A diagnostic fixed-scale F16 NN alternative multiplies both inputs by 2^8
before conversion, retains F32 accumulation and restores the result once at
the existing F32 store. It improves the same F16 AT/BT baseline by 3.99% on
Intel and 0.93% on NVIDIA; both 256-update trajectories pass the same frozen
checks. Powers 8 and 12 give similar timings. These are separate controls,
not a paired BF16-versus-scaled comparison. The diagnostic uses the generic
cooperative generator on both cards (including NVIDIA) and requires its
matching engine/environment; its saved plans are not production artifacts.
Before choosing an additional NN policy, test Whisper's rejected plain-F16
products and overflow/mixed-range cases, and serialize any admitted scaling
as part of kernel identity. Raw controls and trajectories are archived in
`static-dense-model-controls` and `static-dense-real-text` beside the survey.

- [x] Separate automatic derivative protection from explicit precision pins:
  [PR #294](https://github.com/kvark/meganeura/pull/294). Ordinary construction
  keeps the combined F32 guard, including after extraction, fusion and splitting.
  Explicit pins propagate into derivatives; older metadata stays conservative.
- [x] Add measured AT/BT derivative-input choices using the existing kernels:
  [PR #295](https://github.com/kvark/meganeura/pull/295).
  Forward and derivative types remain independent. Bindings, accumulators and
  split partials stay F32; strict policy and explicit pins remain in force.
  Keep the derivative guard and the caller's whole-gradient/trajectory gates.
  No mixed-storage profile or new backend capability API is needed.
- [x] Reach the Intel SmolLM2 derivative opportunity within 30 seconds without
  displacing its stronger layout, combining #295 with the exact-repeat port in
  [Inferena #83](https://github.com/kvark/inferena/pull/83). The arithmetic-only
  comparison below remains neutral; the qualified combined result follows.

Fresh 30-second search winners, measured in separate 100-sample paired
holdouts after 10 seconds of warmup (forward + loss + backward, milliseconds;
optimizer updates are qualified separately):

| GPU | Model | Baseline | Derivative search | Change |
| --- | --- | ---: | ---: | ---: |
| B570 | SmolLM2 | 50.383 | 50.421 | +0.08% |
| B570 | SmolVLA | 14.690 | 14.692 | +0.01% |
| B570 | SD U-Net | 9.342 | 9.361 | +0.21% |
| B570 | Whisper | 59.172 | 58.995 | -0.30% |
| RTX 5070 | SmolLM2 | 17.982 | 16.437 | -8.59% |
| RTX 5070 | SmolVLA | 6.637 | 6.096 | -8.15% |
| RTX 5070 | SD U-Net | 3.798 | 3.716 | -2.15% |
| RTX 5070 | Whisper | 19.288 | 19.300 | +0.06% |

NVIDIA's three gains select F16 AT/BT inputs. Intel SD also selects them,
without a whole-search gain; the other Intel winners and NVIDIA Whisper keep
F32 derivative products. Intel SmolLM2 spends the budget on graph coverage,
qualification and existing submission trials before reaching reduced arithmetic.
Keep its fixed-plan gain separate from the neutral search result.

Submission counts match except NVIDIA SmolLM2: its new winner chooses four
instead of eight. At matched counts, the saved plans give 18.568→16.350 ms
(-11.94%, four submissions) and 17.966→15.721 ms (-12.50%, eight). Only artifact
submission metadata changes for these comparisons; neither physical plan is
retuned. Search still leaves about 0.63 ms available at eight submissions.
The submission policy itself is unchanged.

All sixteen searches complete in 30.35–34.05 seconds including setup. All eight
holdouts qualify before/after timing, and every winner passes its declared
training gate: six 32-update synthetic cases and both 256-update real-text
cases, with initial/final fixed-state gradients, parameter updates and the
unchanged native/PyTorch F32 loss bounds as applicable. All 677 unit tests and
all-target/all-feature lint pass. Earlier search orderings and their neutral
results remain in the external archive.

Preserve these additional leads for follow-up:

- Repeated full-gradient qualification on Intel: the 134.5M-element SmolLM2
  check transfers roughly 538 MB each time. In the first 30-second derivative
  search, repeated checks prevent reaching the new precision trial. Investigate
  comparing every output/gradient bit on the GPU against an already CPU-qualified
  snapshot. Any difference must use the existing full CPU check; every new plan
  still needs its independent F32 qualification. This could avoid duplicate
  transfers without dropping gates or approximating their numerical reductions.
  Include packed/combined gradients, buffer lifetimes, changed inputs and a
  bounded snapshot allocation before considering production admission.

The exact-repeat prototype reaches the Intel opportunity: nine→twelve trials
within the 30-second budget, with eleven exact GPU repeats replacing duplicate
538 MB readbacks. The independent CPU gate still checks every new plan. The
normal holdout harness (shortcut disabled) gives 50.401→40.713 ms (-19.22%) for
the saved winners. At matched submission counts, the gain is 17.87% at one
(50.378→41.373 ms) and 18.07% at four (49.736→40.749 ms). Both GPU winners pass
the unchanged 256-update real-text gate against native/PyTorch F32 references.
NVIDIA is neutral in this comparison (-0.44%). This is a qualification/search
prototype, separate from the arithmetic admission in #295. The port is now
[Inferena #83](https://github.com/kvark/inferena/pull/83). A build of the unmodified
production framework sources with #295 plus #296 also reaches the typed
AT/BT choice in fresh 30-second searches: training is 40.956 ms on Intel and
15.989 ms on NVIDIA; inference is 7.812/3.660 ms. Every ordinary qualification
passes, without internal-buffer hooks.

The standalone port on current main plus #296 also improves selection: Intel
training changes 49.738→45.664 ms and NVIDIA 18.462→18.042 ms, while total native
setup/search changes 78.365→75.521 s and 68.189→67.821 s. These are separate
ordinary 100-sample collections, not paired fixed-plan effects; submission
counts change under the existing policy. Keep this experiment distinct from
the precision-stack holdouts above.

The prototype also passes explicit GPU tests on both cards: loss-preserving
gradient mutations, final-element/large-array tails, CPU fallback on changed
gradients and exact restoration. Snapshot data is bounded by 1 GiB and half the
available GPU budget. [PR #296](https://github.com/kvark/meganeura/pull/296),
standalone on main, supplies the gradient-buffer interop accessor so the port
can avoid internal-buffer hooks. The shortcut changes no numerical bounds or
fresh-execution counts.

- Explicit F16 quantization before residual subtraction: a small GPU probe
  retains the F32→F16→F32 conversions in emitted SPIR-V, yet the Intel driver
  returns zero residuals. `quantizeToF16` preserves the intended correction.
  Check production compensated GEMMs before changing their generator; the
  isolated NVIDIA cast probe does not reproduce this Intel behavior.
- Shared-memory stride/padding: tested 18 matched NN/AT/BT cases, including
  mixed storage, BF16 and tails on both GPUs. All numerical checks pass, but
  padding by 8 or 16 half-elements regresses the staged Intel 128×576×576
  products by 5–11%. Native direct loads are unchanged. Retain the samples;
  do not change production staging or spend model time on this losing variant.
- Larger key blocks for QK/PV: the prototype processes 16 keys between shared
  memory exchanges. Test 32/64-key blocks at fixed arithmetic and storage before
  attributing Intel Whisper's loss to matrix hardware. Measure barriers,
  staging/register pressure and complete models; preserve masks/tails, F32
  online softmax and the smaller portable fallback.
- Per-attention-class selection: SD U-Net and SmolVLA contain multiple
  attention shapes. A whole-program switch can hide a profitable subset;
  evaluate class-specific alternatives with complete-program qualification.
  The collected Whisper fixture is encoder-only: all four attention dispatches
  are 1500×1500 with width 64, so no decoder subset exists in that workload.
- Scaled F16 backward GEMMs: preserve very small derivatives and dynamic
  range explicitly, as the existing attention derivative path does. Do not
  clear full-precision protection merely to reach matrix hardware.
  Before another per-K32 scan, test a fixed power-of-two operand scale that is
  constant over the whole split partition. Restore the F32 accumulator once
  before its existing store, retaining F32 bindings/partials and avoiding
  per-block scans/restoration. Compare NN/Whisper cases that plain F16 rejects
  against BF16, and keep overflow/mixed-range failures and frozen trajectory
  checks. This is a bounded measured candidate, not a universal loss scale.
- NVIDIA's raw NV2 query also advertises reductions, per-element operations
  and conversions. Survey Blade/Naga support before exploring softmax on
  cooperative fragments or accumulator-to-operand conversion; these might
  avoid shared score/PV round trips. Retain the portable path and require a
  complete-model gain before adding APIs for them. The pinned Blade enables
  only NV2 workgroup scope and flexible dimensions; reduction/conversion/
  per-element feature bits remain disabled. The pinned Naga IR has matrix
  loads, multiply-add and stores, but no matrix reduction or per-element call.
  These require compiler/backend work before a model experiment can use them.

- [ ] Extend useful storage chains through pointwise operations and compatible
  fusion without adding redundant widening/narrowing dispatches.
- [ ] Cover additional NN/AT/BT, batched/block and split-reduction forms where
  profiles justify the generator and qualification work.
- [ ] Investigate cooperative matrix conversions for direct reduced result
  stores, avoiding an F32 shared-memory round trip. Both queried GPUs expose
  F16 accumulator/result matrices, so test F32 accumulation followed by a
  same-role F16 conversion and direct store. This does not require doing the
  reduction in F16. [Standard KHR matrix conversions](https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/KHR/SPV_KHR_cooperative_matrix.asciidoc#34211-conversion-instructions)
  preserve scope, dimensions and use; NV2's accumulator-to-operand conversions
  are separate. Naga currently rejects matrix casts. Each intermediate matrix
  type still needs device support; require qualified model gains before adding
  compiler/capability APIs.
- [ ] Consider immutable reduced weight caches only with upload, sharing,
  checkpoint, external-buffer and optimizer-update invalidation defined and
  tested. Inference parameters are still mutable through these interfaces.
  Measure
  device-local and host-visible placement independently.

## 3. Intel loading and tile geometry

- [x] Extend checked typed NN/AT/BT split products to fused addends. Add the
  F32 bias only to the first partition, keep F32 partials and the existing
  reduction, and distinguish this addressing in the pipeline key. Ordinary
  shaders remain byte-identical to the baseline. Independent F64 partial/final
  fixtures, poisoned-buffer reuse and private tuner checks pass on both GPUs.
  Fixed-plan training holdouts isolating this extension improve B570 SmolLM2
  47.860→45.802 ms (4.30%) and RTX 5070 18.314→17.905 ms (2.23%), with matched
  submission counts and independent F32 qualification before/after timing.
  All six 32-update model cases and both 256-update real-text variants pass
  unchanged training gates. These are saved-layout opportunities; bounded
  search does not yet reliably combine the best layout and arithmetic. A fresh
  30-second NVIDIA search reaches all 60 new split/add products, but its
  holdout is neutral (18.382→18.362 ms): another class retains eight partitions
  instead of the older winner's two. Preserve this result beside the fixed-plan
  gains rather than claim an automatic end-to-end improvement.
- [x] Test earlier rectangular arithmetic coverage. The old early-coverage
  condition recognizes 8×8×8, while B570 exposes 8×16×16. Extending it reaches
  reduced products earlier, but loses 5.62% on SmolLM2 and is neutral on
  SmolVLA by displacing a better graph layout. Retain this rejected experiment.
- [ ] Improve layout/arithmetic ordering using a cost estimate that covers the
  complete plan. On the saved B570 SmolLM2 layout, the previously eligible 121
  dense products improve 4.93%; adding forward QK/PV improves 5.44%. The
  instrumented profile is dominated by protected F32 backward products. Do
  not confuse earlier candidate coverage with a better selected program.
  The current cost fallback charges multiply FLOPs only for dispatches with
  `mnk()`; attention otherwise contributes mostly buffer traffic. Account for
  attention work and staging before using this estimate to rank graph forms.

The split-count regression above is not explained by missing coverage: both
searches measure the two-partition candidate before their deadlines. In the
losing race, its median paired gain exceeds the configured 5% improvement plus
twice the median absolute deviation of paired differences. An additional check
on the difference of two independent medians rejects it under timing drift.
Using paired gains/noise consistently, with the same margin, sample count and
numerical qualification, retains two partitions in all six NVIDIA SmolLM2
searches (three per mode), versus one of six with the old rule. Independent
holdouts improve inference by 2.98–5.29%; the partition-only ablation gives
4.59% at fixed graph, arithmetic, storage and submission count. Training
winners cluster at 18.01–18.03 ms, versus 18.04–21.51 ms; against the stronger
saved baseline the gain is 2.24%, and the winner is within 0.18% of the manually
assembled fast plan. This is more reliable selection, not a new kernel gain.

All 16 two-GPU/model/mode holdouts qualify before and after timing. Other
models change by -1.19% to +1.29%; the Intel Whisper inference loss repeats
at +1.25%, and its winning decisions would also pass the old timing rule.
Retain that losing selection beside the gains. All six 32-update and both
256-update selected-winner trajectories pass unchanged gates against the
independent F32 references. Intel SmolLM2 training remains neutral: improving
late layout/arithmetic combinations still needs the complete-plan work above.

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
