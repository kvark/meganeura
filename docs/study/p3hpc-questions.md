# Workshop questions and rehearsal

Use the short answer first. Offer the technical detail when asked. This is a
study aid, not a script that must be memorized word for word. Numeric evidence
and source provenance live in [the final P3HPC evidence guide](../../paper/p3hpc/RESULTS.md); competitor sources live
in [alternatives](alternatives.md).

## Positioning

### 1. What is the actual contribution?

Short: a compact native training/inference layer over Vulkan and Metal, with
a correctness-gated cross-machine performance study and real embedded
deployment.

Detail: the graph, autodiff, specialization, memory and runtime path are
shared. The contribution is not invention of graph compilation, portable
autodiff, Rust ML, autotuning or graphics-API inference. It is the integration
and the evidence about what that implementation recovers and misses.

### 2. Does the evaluation cover HPC hardware and larger models?

Short: H100 completes the five-model cohort; 360M and 1.7B have valid strict
forward/backward results from one process per completed condition.

Detail: the larger-model extension stops in searched 1.7B PyTorch capture,
so it supplies preliminary scaling evidence, not full replicated coverage.
Batch/sequence shapes remain small, with no optimizer or distributed work.
MI300X is an adverse availability result: the supplied report has working
ROCm but no validated Vulkan model execution. The graphics substrate has
limits as well as deployment advantages.

### 3. Why Rust? Is avoiding Python the speedup?

Short: Rust helps native embedding and a cohesive implementation; the
performance question is mostly about compiled kernels and execution plans.

Detail: compiled PyTorch is not a Python loop doing GPU arithmetic. Both
systems ultimately execute native GPU programs. Rust does not guarantee
correct shader indexing, synchronization or numerical behavior. The owned
runtime is also substantially helped by Blade and Naga.

### 4. Why Vulkan/Metal instead of CUDA everywhere?

Short: CUDA does not provide the same cross-vendor graphics-device reach or
native renderer integration.

Detail: the cost is supplying kernels and tooling that vendor compute stacks
already maturely provide. Driver capabilities and performance vary. We did
not demonstrate that graphics APIs expose every hardware optimization or
that they have no performance ceiling.

### 5. Why not Burn, tinygrad, TVM or IREE?

Short: these are serious close alternatives, not missing prior art.

Detail: Burn is broad Rust training with JIT fusion and portable backends;
tinygrad is a minimal end-to-end compiler with search; TVM provides mature
schedule-search ideas; IREE compiles scheduling and computation into native
deployment artifacts. Meganeura chooses a smaller owned static interface over
one graphics boundary. Only PyTorch has matched measurements here.

### 6. Is one train-to-deploy framework unique?

Short: no. The useful result is that this compact implementation carries the
path onto the demonstrated native graphics targets.

Detail: other frameworks unify training and inference, and export-based
systems deploy very small runtimes. Our footprint comparison is not a claim
of minimal size among all compiled runtimes. Quantized fine-tuning on Apple
is also already offered by MLX-LM.

## Architecture

### 7. What happens when I call build?

Short: obtain device capabilities, optimize the forward graph, differentiate
if training, compile and select kernels, schedule and allocate, then create
the session's pipelines and buffers.

Detail: the compiler can be inspected without a GPU, but the public build
path needs the actual context. The plan cache stores a pre-runtime plan.
Late selection changes geometry/padding/fusions, then aliasing uses the final
scheduling constraints. See the [architecture diagram](architecture.md).

### 8. What does static mean? Do you replay a CUDA Graph?

Short: shapes and tensor execution structure are fixed for a session. We
re-encode commands, not replay a native captured CUDA Graph.

Detail: this avoids graph rediscovery and per-step tensor allocation. It
does not eliminate all host work or CPU allocation. General runtime shape
changes need explicit plan support; a changing KV position is not an arbitrary
dynamic shape. Native command capture may or may not help particular cases;
the final PyTorch replay ablation measures its effect on the reference, not a native command-replay implementation.

### 9. How can the system be general if it has specialized attention kernels?

Short: specialize reusable tensor algorithms, not model identities.

Detail: online softmax is an algorithmic improvement over materializing a
score matrix, not simply a tile-size knob. A minimal compiler still needs a
way to express that algorithm. Meganeura uses archetypes and generators but
also retains templates and operator-specific derivative rules. "No
handcrafted kernels" would be inaccurate.

### 10. What does equality saturation buy you?

Short: the ability to explore equivalent graph forms; no measured runtime
advantage over greedy selection for the current paper rule set.

Detail: repeated-region outlining bounds saturation work, and traffic-aware
extraction ranks representations. Costs are approximations, with fallback
cases. Greedy is the default because additional optimizer machinery should
earn its compilation cost through useful choices.

### 11. How is buffer aliasing safe across parallel work?

Short: lifetimes are separated by barrier groups, not just dispatch indices.

Detail: if dispatches can overlap in the same group, their buffers cannot be
reused merely because one appears earlier in a vector. Parameters, gradients,
outputs and persistent state are pinned where needed. Pinning prevents alias
reuse; it does not require CPU-mapped storage. Padding and optimizer reads
extend the relevant physical/lifetime contract.

### 12. How much autotuning exists now?

Short: bounded measured kernel selection runs inside `build`. In the final
cohort it improves strict 135M prefill 1.38× on H100 and 1.09× on RTX 5070.

Detail: the light policy disables search; the searched policy enables it.
Both use greedy graph rewriting. Exact classes cover scalar matmul/convolution
tiles and eligible native-f32 cooperative matmuls, with private scratch,
numerical qualification, timing/noise guards, and resource bounds.
F16-input, complex-fusion, GEMV, representation search, persistent winners,
and automatic whole-step confirmation remain opportunities.

The final recorded preparation difference amortizes after about 111 H100
or 238 RTX 5070 prefills. These estimates charge all three benchmark sessions,
not an inference-only deployment, and assume naturally warm driver state.
Searched PyTorch is still faster in every CUDA phase comparison.
The older six-pair 1.092× experiment and its 80-call break-even are separate
development evidence, not interchangeable numbers for this cohort.

One preparation API combines compilation and measured choice; graph rewriting
and kernel selection are not jointly searched today. The separate compiler
microstudy measures 141–172 µs WGSL parsing and roughly 31 ms native versus
100–167 ms Triton preparation for another cold GEMM candidate after compiler
warmup. Native driver work is included in the latter comparison. The useful
claim is integrated portable deployment with a measured search budget, not
invention of autotuning or a 10,000× end-to-end compiler advantage.
[Compilation and search](performance-plan.md#cheap-compilation-as-a-search-budget).

### 13. Does strict f32 mean identical results?

Short: no. It constrains arithmetic permissions, not operation ordering or
bitwise identity.

Detail: f32 storage/output does not prevent reduced-input matrix arithmetic.
The strict harness disables TF32 and all cooperative-matrix paths, including native-f32 cooperative tiles.
Different reductions and contraction behavior can still differ. Accelerated
mode deliberately permits different fast input formats on each engine and
must independently pass the gates.

### 14. Why did compensated f16 backward get rolled back?

Short: preserving extra mantissa bits did not preserve f32 exponent range.

Detail: with `hi=f16(x)` and `lo=f16(x-f32(hi))`, both can vanish for tiny
values. F32 accumulation cannot reconstruct discarded inputs. The August 28
fix retains scalar f32 for protected derivatives on f16-only matrix devices.
bf16 or scaling needs its own capability plumbing and numerical contract.

### 15. Are the gradients verified element by element?

Short: CUDA replay is checked that way against uncaptured PyTorch; the
cross-engine comparison uses total and per-parameter gradient norms.

Detail: replay qualification covers every participating element, checks
fixed RMS/maximum bounds, and tests two consecutive replays. Accelerated
training first checks eight ordinary repeats. Cross-engine norms can still
miss a sign error (`g` and `-g`), and 256 output samples can miss localized
errors. All 366 completed pairs individually pass the 5% gradient gate.
This is not full cross-engine elementwise equivalence or convergence evidence.

### 16. Did you find a PyTorch bug on the 780M?

Short: the original submission had a reference inconsistency, not a proven
root cause. It does not recur in the final cohort.

Detail: the older cross-backend norm audit justified a symmetric exclusion
under its declared analysis. The new pinned ROCm cohort passes Whisper in
both contracts and all three processes. No oracle exclusion or old failure
count belongs in the camera-ready tables.

### 17. Do incomplete campaigns or exclusions bias the aggregate?

Short: the complete-GPU score is conditional, and the missing coverage is
reported explicitly.

Detail: the same six complete GPU-reference systems enter every primary
workload and phase. CPU support, partial Windows, and the H100 extension
are separate populations. Unreached conditions are not assigned failures,
and failed conditions are not given replacement timings. Requiring support
across every attempted GPU gives both stacks a known hole, hence zero
universal-set portability; reporting only the conditional score would mislead.

## Performance and methodology

### 18. In one sentence, do you beat PyTorch?

Short: on selected workloads, especially Radeon and small transformer shapes,
but the stronger CUDA baseline usually wins.

Detail: across six complete GPU-reference systems under light preparation,
strict inference/minimal/training median ratios are 1.66/1.26/2.41, with
nominal wins 5/30, 10/30, and 4/30. One inference win is essentially a tie.
Searched PyTorch wins all 60 CUDA phase comparisons. Intel CPU and partial
campaigns are excluded from these counts.

### 19. Isn't Intel GPU versus CPU unfair?

Short: it answers support availability, not a GPU-to-GPU efficiency question.

Detail: RPL-U explicitly uses eager CPU PyTorch and is excluded from all GPU
scores. Native Vulkan wins its five full-inference comparisons but loses the
two minimal transformer shapes. Arc B570 supplies a separate real XPU
comparison: default/no-graph completes 30 pairs. Its embedding backward
requires a shape-probed dense `index_add` equivalent, recorded in the results.
State this qualified workaround rather than calling the reference unmodified
native-XPU execution.

### 20. Did the final comparison use PyTorch max-autotune and CUDA Graphs?

Short: yes on both complete NVIDIA campaigns, with explicit whole-phase
replay validation and an uncaptured ablation.

Detail: the original submission bypassed capture and the reviewer was right
to question it. The camera-ready now uses the repaired common-source cohort.
H100 135M one-token PyTorch time falls 4.244 → 1.276 ms with default replay
alone; diffusion training falls 12.035 → 4.035 ms. These controls can reverse
an apparent native win.

Max-autotune was requested during bring-up on every applicable backend.
It failed in diffusion compilation on both Radeon systems and did not finish
the first B570 qualification within an hour. Their final default-only
campaigns label the omissions. This is an automatic-compilation portability
finding; no eager timing substitutes for the failed request.
ROCm replay itself was not qualified in this protocol.

Pinned Inductor's 68-SM gate declines GEMM template search on 5070/3050;
H100 executes that search. Other compiler choices still help some small-GPU
workloads. H100 strict ResNet inference gains 1.28× from searched preparation
but compilation grows 15.14 → 708.33 seconds, giving about 1.14 million calls
to repay the recorded extra setup. That does not make search unhelpful for
every workload or phase.

Light/searched is separate from strict/accelerated arithmetic. Both primary
CUDA preparation policies use replay. Fresh processes and private
TorchInductor/Triton caches prevent reuse of a saved compiler winner; persistent
driver/library caches remain naturally warm. The frozen protocol treats this
as a threat to validity, not independent first-use cold starts.

Windows and the H100 extension stop in `cublasSgemm` during training capture.
The preceding validated pairs survive, with actual replicate counts;
there are no replacement timing claims for the failed conditions.
[Methodology and failures](../../paper/p3hpc/CUDA-GRAPHS.md).

### 21. Are the training times complete training steps?

Short: the matrix times forward, scalar loss and backward; it excludes the
optimizer update.

Detail: optimizer support is demonstrated separately. These results do not
measure convergence, data loading, communication, checkpoint cost or steps
to target quality. Avoid translating a F+L+B speed ratio into training-job
time or energy savings without those measurements.

### 22. Is the one-token result LLM decoding throughput?

Short: no, it is stateless minimal-shape forward latency without KV cache.

Detail: real decode needs matched cache length/layout, token position,
quantization, batching and memory accounting. Serving adds request scheduling
and tail latency. llama.cpp or vLLM comparisons need that expanded protocol,
not a relabeling of the current column.

### 23. What does a portability score of 0.39 mean?

Short: the mean of five workload-level harmonic efficiencies for strict
light-policy training over six complete GPU-reference systems.

Detail: the comparator is the faster valid result of the two engines on each
machine. It is not 39% of peak or a median time ratio. The corresponding
PyTorch score is 0.98. Every workload uses the same six systems; the final
cohort has no oracle-dispute exclusion. The CPU comparison and partial
campaigns remain separate. Expanding to every attempted GPU exposes
availability holes and gives both stacks zero under that support requirement.

### 24. Do profiles prove the remaining gap is just missing kernels?

Short: they identify where to work, not prove the absence of API or driver
limits.

Detail: the separate resident-parameter 5070 Nsight experiment puts 1.7B
prefill's native grouped GPU span at 52.67 ms, versus 27.91 ms of CUDA
kernels; native host recording takes another 2.56 ms. These are different
timing boundaries, not additive components of an exact partition.
A longer token control directly measures CPU downclock sensitivity while
GPU time stays near 1.88 ms. Convolution and GEMV experiments identify
general choices that improve qualified whole-step execution.

Those diagnostics have separate revisions and placement controls; they do
not explain every final-cohort platform. Wall minus summed kernels is not
barrier cost, and shader workgroup barriers are not resource dependencies
between dispatches. The historical M3 attention profile is also separate.
See the [paper's gap analysis](../../paper/p3hpc/main.tex).

### 25. Why can a tensor-core fast path become slower?

Short: matrix throughput is only one term in the cost.

Detail: operand staging, padding, layouts, occupancy, reduction shape and
dispatch structure can dominate. Accelerated RX 7900 XT SmolVLA also regresses in the final light cohort.
This motivates measured choice among correct candidates; it does not imply
cooperative matrices are generally bad or always profitable above a fixed
threshold.

### 26. Does 12.8 MiB prove greater productivity?

Short: it proves a small historical deployment artifact under a stated accounting
basis, not developer effort or feature equivalence.

Detail: the comparison excludes weights, Python itself, OS and drivers as
documented. Source counts exclude large dependencies. We did not record
implementation/maintenance time by platform or perform a user study.

### 27. Was training performed on the Quest headset?

Short: no. Host training produced the decoder checkpoint; the headset runs
inference integrated with graphics.

Detail: the case demonstrates a physical native deployment path. Its source
and artifacts are separately versioned, outside the paired matrix, and there
is no matched PyTorch Android speedup claim.

### 28. What is the next convincing result?

Short: the current evidence is enough to finish this paper; further engineering
should broaden useful legal choices with whole-step confirmation.

Detail: final compile-time search already has measured gains. Separate
convolution-specialization and GEMV experiments identify general opportunities.
A future production change should charge preparation, retain failures, and
confirm held-out end-to-end performance. Another Intel server rental is not a
prerequisite for camera-ready; larger replicated models and multi-platform
kernel attribution remain clear research limits.

## Observability and debugging

### 29. Isn't a static graph much harder to debug than eager PyTorch?

Short: eager PyTorch is more convenient interactively. We preserve names and
dispatch provenance and offer materialized debug sessions, growing-graph
evaluation, plan/shader dumps and structured profiles.

Detail: inspection must distinguish a fused-away value from an aliased one.
Debug mode disables dispatch fusion/aliasing, but graph rewrites and precision
policy are separate controls. We do not claim arbitrary gradient hooks, Python
breakpoints inside shaders, or identical eager/compiler tooling. See the
[debugging comparison](observability.md).

### 30. Does first_bad identify the operation that caused a NaN?

Short: it identifies the first *reported* nonfinite output prefix in plan
order, not necessarily the root cause.

Detail: `step_debug` scans at most 65,536 floats of each primary output after
the complete step, skips aliased outputs outside debug mode, and can miss
overwritten values and extra outputs. A poisoned input can implicate its first
consumer; finite wrong answers and underflow need independent checks. Active
optimizer/KV updates still happen during this diagnostic step.

### 31. Can I use your GPU profile as the end-to-end benchmark?

Short: no. Per-dispatch instrumentation changes the pass/barrier structure.

Detail: keep raw profile samples and the overhead relative to normal grouped
execution. Reset state before every retained profile run. The structured
collector cannot assign appended optimizer passes to graph metadata, so capture
without those passes and time full optimizer-backed training separately.
Likewise, the new tuner's isolated scratch result needs whole-step confirmation.

### 32. Can a bad checkpoint corrupt a running session, and is it portable?

Short: format 3 validates the entire logical restore before writing anything.
Matching parameter names, shapes and storage types can use different padding.

Detail: malformed files leave parameters, gradients, moments and counters
unchanged and do not allocate moments. This is not rollback after device loss
or allocation failure. Training-to-inference validates and ignores moments;
legacy files retain partial-load behavior. Optimizer configuration, in-flight
accumulation windows, clipping cadence and application RNG are not saved.
Cross-padding GPU tests pass; cross-backend qualification is still due.

### 33. What memory did lazy optimizer state actually save?

Short: two unused F32 moment buffers: 8 MiB for the tested 1,048,576-element
parameter. Adam itself still needs that storage when selected.

Detail: the regression checks retained buffer allocation requests, not peak
driver VRAM. Graph buffers, moments, accumulators and diagnostics are reported
separately; staging and driver objects are outside that sum. Reads of
uninitialized moments return zeros without allocating; clearing the optimizer
retains initialized state for later reuse. See
[checkpoint and memory contracts](checkpoints-and-memory.md).

### 34. How do you distinguish a tuning gain from a lucky session or timing drift?

Short: compare untuned sessions first, then reverse which session owns the
selected kernels while keeping buffers and training history in place.

Detail: the new six-process experiment uses four symmetric crossover blocks
and balanced starting roles. It requires a quiet A/A control and the same
5% + twice-MAD guard in both orientations and pooled pairs. Dense inference
passes all six processes (median 1.177×); MLP+Adam has four inconclusive results
and two unstable controls; ResNet never changes selection. Keep every attempt.
This is descriptive counterbalancing, not randomized causal proof or a
confidence interval. A/A runs before search only, and telemetry is coarse.

### 35. Why can search be expensive when the candidate kernels are fast?

Short: correctness qualification, transfers and host work cost time too.

Detail: the first phase profile put 98% of ResNet's search in qualification.
The follow-up separates CPU work and transfer/dispatch/wait, then changes
only the private staging allocation. With Blade 0.9 on RTX 5070, median CPU
readback allocation/copy falls from 582 to 2 ms; unchanged CPU validation
still takes about 6 ms. Total search falls from 606 to 39 ms despite increased
preparation and transfer costs. Keep the ordinary/tiny patterns, full scans
and sampled f64 checks: they were not the dominant cost. See the
[six-process protocol and results](../experiments.md#readback-2026-09-06).

### 36. Does read-optimized staging mean faster kernels or faster ResNet?

Short: neither. It means cheaper kernel search on the tested device.

Detail: candidate code, bindings, precision and selection guards are identical.
Download is the default for one private upload/readback buffer, not the live
model's allocations. All 108 comparisons qualify, with bit-exact state checks
through Adam step 178; continuation was untimed. Both arms used Blade 0.9,
so this is not a Blade-version benchmark either. The original Shared option
remains available and tuning remains opt-in. Source policies differ on Vulkan,
but Shared and Download map to the same Metal storage mode; fleet performance
and automatic whole-step confirmation are still open.

### 37. What exactly is reused, and can it change validation or the memory budget?

Short: one private staging allocation, at the same exact size, within one
tuning call. Not model buffers, qualified outputs, inputs or selected winners.

Detail: a size change releases the old buffer before allocation; all binding
buffers and encoders remain fresh. Every ordinary/tiny input upload, NaN poison,
readback, finite/parity scan and sampled f64 check still runs. Full simultaneous
requests count against the same byte cap, and nothing is retained on return.
Reports expose allocation/reuse/release counts and both comparison/final cleanup.
Six process pairs reduce dense and MLP search costs by median paired ratios
1.378× and 1.414×, with bit-exact state checks through Adam step 178. Tuning
remains opt-in; these are not model or cross-engine gains.

### 38. Why does a no-reuse control still show timing differences?

Short: equal implementations do not guarantee equal observed times.

Detail: ResNet uses one allocation in each arm. Fresh-first process ratios have
median 1.105×; reuse-first ratios have median 0.914×. All six are retained,
and the overall gain/regression guards both reject a claim. Even the ratio of
cost medians (0.981×) and median process ratio (1.007×) differ in direction.
Balanced order is a control, not proof of constant clocks or causal isolation;
the 250 ms telemetry cannot explain every short operation. See the
[complete resource protocol and results](../experiments.md#staging-reuse-2026-09-06).

### 39. Can all full-state comparisons pass while a gradient is still wrong?

Short: yes, if both executions share a bug or the cases miss its domain.

Detail: the new profiler preserved all 45 full states exactly. Reviewing the
convolution family still found dX using the wrong padding outside same padding.
A direct f64 oracle failed where same-padding models could not expose the
error. Both scalar and generated cooperative indexing are fixed, with full
odd/asymmetric/batched/tiny derivative checks. Control-session parity measures
preservation; an independent oracle addresses correctness. Neither replaces
domain coverage or convergence evidence.

### 40. What do the new whole-step profiles justify doing next?

Short: broaden reusable derivative implementations and measured selection,
not the matmul search budget or the numerical tolerance.

Detail: ResNet backward convolution accounts for 60.66–60.77% of instrumented
dispatch time; SmolLM2 backward attention accounts for 36.25–40.58%. A long
weight-gradient reduction with ten output workgroups motivates testing split-K
with charged temporary storage. These are F+L+B-only RTX profiles, not optimizer
or Metal evidence. Intrusive pass timing and substantial short-case drift mean
the shares are localization evidence, not guaranteed whole-step speedup bounds.
Exact convolution classes/qualification now cover the existing scalar
tiles; accept any new schedule only after independent edge tests and matched
whole-step confirmation. [Protocol and limitations](../experiments.md#training-profile-2026-09-06).

### 41. Why isn't convolution tuning just another M/N/K lookup?

Short: equal contraction dimensions can gather different physical tensors.

Detail: dX uses M=Ci, N=H×W, K=Co×Kh×Kw, with one dispatch plane per batch;
dW uses M=Co, N=Ci×Kh×Kw, K=batch×Oh×Ow. Kernel aspect ratio, stride and both
padding dimensions affect addresses even when M/N/K match. Our exact key
records all of them, plus precision, binding capacity and placement. Scratch
is physical NCHW data, not an im2col allocation. The existing two scalar tiles
share one bounded runner and unchanged decision guard; new options expose
Dense, ConvDerivatives and All scopes. No model name enters selection.

### 42. What does convolution qualification prove, and what doesn't it?

Short: structural legality and numerical testing are separate checks.

Detail: checked extents prevent index overflow, and shared convolution kernels
now decompose indices with exact integer arithmetic. The earlier reciprocal-domain
filter only excluded unsafe shapes from tuning; it did not fix ordinary execution.
Both variants must still produce finite, matching full
outputs on ordinary/tiny synthetic inputs and match 32 f64 contractions,
including dX batch edges. Separate full f64 scatter oracles cover padding,
stride and tile edges. Shared bugs, untested domains and convergence still
require independent evidence. A qualified isolated winner is not automatically
a whole-step win. [Contract and experiment](../experiments.md#conv-tiles-2026-09-06).

### 43. Can two bit-exact training sessions still be an invalid benchmark?

Short: yes—matching zeros and advancing a counter do not prove useful work.

Detail: the first small convolution-chain builder supplied 4-D operands to
an API documented for flat NCHW arrays. Its first forward dispatch had zero
workgroups. Both sessions agreed on zero loss, gradients and Adam moments,
so the original comparisons passed. We retained/disqualified those twelve
cases, corrected the builder, added early operand rejection and full forward
oracles, and required nonzero prefix signals plus actual parameter changes.
The corrected six-process cohort passes these checks. This is why workload
validity, independent correctness, state preservation and performance are
separate gates. [Correction and archive](../experiments.md#conv-tiles-corrected-2026-09-06).

### 44. Did convolution autotuning win?

Short: four isolated dX classes win; whole-step confirmation remains inconclusive.

Detail: eight ResNet dispatches change from scalar 64 to 32 tiles in every
corrected process. The median whole-step ratio is 1.05056×, with observed
time reductions 4.60–4.85%. That is below our 5% requirement even before the
noise margin. All six A/A screens pass, but no confirmed gain follows. The
small Adam/SGD chains make real updates and retain their starting tiles.
The eight-class structural budget visits only 8/45 ResNet derivative classes;
it is not a measured cost ranking. The next experiment is bounded split-K dW
with charged partial storage/reduction and explicit search coverage, not a
lower threshold or an unconditional performance claim.
Ordinary convolution indexing is now repaired separately: exact integer arithmetic
replaces unsafe float reciprocal multiplication, with full adversarial GPU oracles.

### 45. How does exact indexing fit a minimal, general engine?

Short: fix the shared arithmetic, not individual shapes.

Detail: the old f32 reciprocal mapped 41/41 to zero, selecting the wrong batch
in a weight gradient. Exact division repaired this across scalar and generated
convolution kernels and removed the tuner-only interval filter. Its retained
cost check increased ResNet F+L+B from about 17.55 to 21.55 ms.

The follow-up uses an **integer** reciprocal: for `d > 1`, precompute
`m = floor(2^32/d)`, take the high word of `n*m`, and increment if the remainder
is still at least `d`. The estimate is at most one low for every u32 numerator;
`d = 1` returns `n`. One shared WGSL helper and four derived u32 uniforms serve
all convolution directions. No float addressing, width exceptions or new tuning
knob is needed. GPU raw-u32 checks and full f64 forward/dX/dW oracles pass.

A new six-process RTX cohort shows only about 2% lower ResNet F+L+B time, with
bit-identical full states. Most of the old cost remains, and short-case drift
prevents a no-regression claim. A correctness proof does not prove profitability;
this local comparison is not a new PyTorch result or paired-MAD tuning decision.
[Repair](../experiments.md#conv-indexing-2026-09-06),
[proof and follow-up measurements](../experiments.md#conv-divisor-2026-09-06).

### 46. Why isn't split-K just another tile size?

Short: it changes both the reduction order and the execution plan.

Detail: splitting a long contraction creates more independent workgroups, but
also writes partial results and launches a reduction before consumers run.
Those bytes, barriers and dispatch costs belong to the candidate. The explicit
prototype now partitions the existing scalar dW template into `[split,M,N]`
partials and uses existing f32 SumRows to write the original gradient. It runs
as a plan transformation before allocation, with an all-or-nothing logical-byte
cap; ordinary scheduling and memory reuse handle the intermediate lifetime.
Labels/origins preserve attribution, but profiling only the convolution family
would miss the final generic reduction.

Full independent f64 partial/final oracles and short SGD/Adam trajectories test
the sequence. A long tiny-gradient three-way partial fails the unchanged accuracy
gate even though its final gradient passes: it remains unqualified. The live
tile search still cannot change allocation layouts. Explicit bounded sequence
probes now reuse its scratch, timers and decision guard without installing a
choice. A four-process cohort finds an isolated 6.93× eight-way gain on the
synthetic long case, including SumRows, but both profiled large shapes reject
before timing. No whole-step training gain is established; defaults are unchanged.
[Prototype](../experiments.md#split-k-2026-09-06),
[sequence measurements](../experiments.md#split-k-sequence-2026-09-06).

### 47. How can the unsplit f32 control fail qualification?

Short: f32 arithmetic does not guarantee a fixed error bound for every long sum.

Detail: two long dW control elements pass the sampled checks but fail the full
f64 scan, one on ordinary and one on tiny inputs. Independent input-coordinate
scatter with sequential CPU f32 FMA reproduces their GPU bits exactly. These
particular failures are accumulation rounding, not an integer-indexing discrepancy.
We retain the rejections and collect no timings for those comparisons.

The earlier three-way partial failure and a later pass on different synthetic
inputs of the same shape also coexist. “Qualified” means the executed tests
passed, not that every possible gradient is covered. Stronger shared accumulation
and broader input coverage come before training promotion; loosening tolerances
or claiming two same-order scalar tiles are independent references would not
resolve this. The probe's full validation can cost seconds, so its cost is reported
separately from sequence timing. [Evidence and limits](../experiments.md#split-k-sequence-2026-09-06).

### 48. Why stop after a mostly successful accuracy improvement?

Short: the acceptance rule was declared before the test, and a deferred feature
is a valid outcome.

Detail: summing 16-term tiles locally and compensating their outer sum fixes
the known control and partial failures. But only 230/240 broader rows qualify;
long tiny cancellation inputs fail for both tiles and every tested count.
A CPU reproduction confirms the reported error even with compensated outer
addition. That step cannot recover rounding already lost within the tile.

We remove the unqualified arithmetic, retain the source and rejection evidence,
and defer split-K promotion without a whole-step benchmark. We do not add a
shape exception, relax the gate, or turn the next possible algorithm into another
mandatory deadline item. The tuning foundation is useful without universal
kernel coverage. This closes the engineering milestone and lets paper review
proceed; it does not prove split-K can never succeed.
[Decision and evidence](../experiments.md#compensated-dw-2026-09-06).

## Talk outline

The actual workshop slot length is not confirmed here. For a 12-minute talk,
prepare approximately eight slides and leave detail in backup:

| Time | Slide and message |
|---|---|
| 0–1 min | Question: can a compact graphics-API layer support useful training and deployment? |
| 1–2.5 min | Architecture diagram: shared graph/autodiff/plan, native embedding boundary. |
| 2.5–4 min | Protocol: eight machines, common PyTorch source, explicit replay and arithmetic/preparation policies. |
| 4–6 min | Results: 2.41× strict training median; replay reverses apparent wins; native H100 prefill search gains 1.38×. |
| 6–7.5 min | Numerical gates, intermittent capture failures, CPU support and MI300X limitation. |
| 7.5–9 min | H100 partial scaling and the separate host/GPU timeline; explain what the traces establish. |
| 9–10 min | Deployment and productivity scope; Quest inference, not headset training. |
| 10–12 min | Limits, alternatives, bounded autotuning direction and takeaway. |

Backup slides: complete strict/accelerated tables, portability equation,
partial-campaign inventory, precision rollback, preparation/break-even accounting,
variant-selection contract, API/operator limitations, and measured replay controls. For a longer slot, expand method and architecture before
adding more performance claims.

## Self-test exercises

1. Sketch the backward shapes for `[32,784] × [784,128]`, then mark which
   nodes need full-precision operand protection.
2. Explain why two buffers whose dispatch indices do not overlap can still
   be unsafe to alias.
3. Compute harmonic efficiency for `[1,1,0.5]` and distinguish it from the
   arithmetic mean and a median time ratio.
4. Give a gradient pair that passes norm gates but is elementwise wrong.
5. Explain why a 2× improvement in a family using 73% of true step time is
   not a 2× whole-step improvement.
6. Name three stateful objects a tuner must restore, and three hardware or
   software changes that invalidate a stored performance winner.
7. Distinguish the six-system GPU training result from the seven complete
   campaigns and the total number of retained valid pairs.

Answers: (1) `dX=[32,784]`, `dW=[784,128]`, with derivative matmuls protected;
(2) same-group execution may overlap without a separating barrier;
(3) 0.75 versus arithmetic 0.833; (4) `g` versus `-g`;
(5) `1/(0.27+0.73/2)=1.57×`; (6) optimizer moments/counter, KV state and
accumulated gradients; device/driver, generator revision and numerical policy;
(7) six complete GPU-reference systems: 4/30 strict training wins and 2.41×
median; seven complete campaigns: 330 pairs; adding partial Windows and H100
extension records gives 366 valid pairs. Different denominators answer different questions.
