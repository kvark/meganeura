# Workshop questions and rehearsal

Use the short answer first. Offer the technical detail when asked. This is a
study aid, not a script that must be memorized word for word. Numeric evidence
and source provenance live in [results](results.md); competitor sources live
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

### 2. Why is this an HPC paper if the machines are consumer GPUs?

Short: the question is performance portability across heterogeneous stacks,
including whether a usable compute path exists at all.

Detail: the portability unit is the machine plus installed software, not only
an ISA. Graphics-driver availability, build/deployment closure and numerical
contracts affect usable performance. We do not claim a supercomputer-scale
evaluation, distributed training or exascale scaling evidence.

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
the present experiment does not isolate that effect.

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

Short: an opt-in exact-shape search over scalar and legal native-f32
cooperative matmul tiles, using private scratch. Scalar GPU qualification has
passed; this is new engineering, not part of the frozen paper's results.

Detail: legal implementations can win regardless of the initial occupancy or
native-8 large-shape profitability threshold;
qualification, paired samples, a noise guard and resource bounds are explicit.
It replaces the state-mutating family demotion tuner. Native padding must fit
existing allocations. F16-input/complex-fusion/GEMV search, persistent winners
and automatic whole-step confirmation remain future work. Native-f32 GPU
qualification needs a different device: our RTX 5070 advertises f16 tiles only.
Capability probing is named `auto_tune` but does not time kernels. Neither
this new search nor new speedups are part of the frozen paper evidence.

The [separate five-process transfer pilot](../experiments/tuning-2026-09-05/README.md)
shows 1.15×/1.13× whole-step gains on two synthetic chains and unchanged choices
on two smaller ones. Search amortizes over roughly 1,600–2,850 steps in the
winning cases; do not turn that into a general model or PyTorch claim.

The [six-case holdouts](../experiments/holdouts-2026-09-06/README.md) then tested
inference, Adam, SGD and F+L+B across five processes. All full control-session
tensor/state comparisons passed through step 78, but none cleared the whole-step
guard. An unchanged ResNet control even showed a misleading 1.071× ratio of
medians, rejected by the paired-gain gate. The honest conclusion is stronger
state evidence and limited transfer, not “automatic tuning makes training
faster.” More representative kernel families and controlled confirmation come
before default-on or persistent winners.

## Numerical behavior

### 13. Does strict f32 mean identical results?

Short: no. It constrains arithmetic permissions, not operation ordering or
bitwise identity.

Detail: f32 storage/output does not prevent reduced-input matrix arithmetic.
The strict harness disables TF32 and reduced-input cooperative paths.
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

Short: not in the frozen full-model matrix. It compares total and
per-parameter gradient norms.

Detail: `g` and `-g` pass a norm-only check identically. The output gate also
uses 256 sampled values. Smaller gradcheck tests provide complementary
evidence, but are not a proof of every large graph. Add element samples or
seeded projections and optimizer trajectories in the next protocol.

### 16. Did you find a PyTorch bug on the 780M?

Short: a reference inconsistency, not a proven root cause.

Detail: four PyTorch backends and five Meganeura backends cluster tightly in
their parameter-norm vectors; that local PyTorch backward result is about
16.3% from the other references. We retain the records and symmetrically omit
that pair from training comparisons. Root cause could be framework, runtime,
driver or another configuration detail; consensus norms alone do not decide.

### 17. Does changing the reference after seeing results bias the aggregate?

Short: it is a post-hoc analytic choice, disclosed with a sensitivity result.

Detail: both engines lose the same training cell; valid forward results stay.
The former one-sided assignment changes Meganeura's strict training score
from 0.62 to 0.52, while PyTorch remains 0.93 rounded. The reader can evaluate
the impact. An independent numerical reference is the next resolution step.

## Performance and methodology

### 18. In one sentence, do you beat PyTorch?

Short: on selected frozen workloads, yes; universally, no.

Detail: strict GPU-referenced minimal shapes win 12/20; full inference wins
8/20; valid training wins 5/19 with median time ratio 1.78. Apple training
loses even against eager MPS. Those populations exclude Intel's CPU reference.

### 19. Isn't Intel GPU versus CPU unfair?

Short: it answers support availability, not a GPU-to-GPU efficiency question.

Detail: the installed reference exposes no usable XPU. We label that fallback
and provide GPU-reference-only ratios separately. The complete-machine
portability score includes CPU because that is the working reference on that
machine, not because CPU time measures the iGPU's hardware potential.

### 20. Why not PyTorch max-autotune or CUDA graphs?

Short: they are important stronger automatic baselines that were not tested.

Detail: the frozen protocol uses default `torch.compile` on Linux and no
additional manual capture. `reduce-overhead`/`max-autotune` can change capture
and selection without model-specific kernels. The paper now states this
limitation explicitly. A new sweep must verify actual activation and charge
compile/search cost, memory and accuracy, not silently replace the old data.

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

### 23. What does a portability score of 0.62 mean?

Short: the mean of workload-level harmonic efficiencies relative to the
better valid time of two engines on each complete machine.

Detail: it is not 62% of peak or a median speed ratio. Invalid/unsupported
implementation coverage yields zero for the specified set. Whisper training
uses four machines because its fifth reference is disputed. Platform set,
denominator and final averaging must be named together.

### 24. Do profiles prove the remaining gap is just missing kernels?

Short: they identify where to work, not prove the absence of API or driver
limits.

Detail: NVIDIA ResNet backward convolution dominates its timestamp profile;
Apple Whisper backward attention dominated a pre-optimization profile.
Instrumentation changes grouping and synchronization attribution. Amdahl
estimates are conditional, and only normal whole-step measurements establish
an improvement.

### 25. Why can a tensor-core fast path become slower?

Short: matrix throughput is only one term in the cost.

Detail: operand staging, padding, layouts, occupancy, reduction shape and
dispatch structure can dominate. Frozen accelerated AMD SmolVLA regressed.
This motivates measured choice among correct candidates; it does not imply
cooperative matrices are generally bad or always profitable above a fixed
threshold.

### 26. Does 12.8 MiB prove greater productivity?

Short: it proves a small frozen deployment artifact under a stated accounting
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

Short: a bounded, state-safe shape tuner that improves held-out end-to-end
workloads against stronger automatic baselines without weakening accuracy.

Detail: convolution derivatives and Metal attention have evidence-backed
priority. Report compile/tune amortization, regressions, memory and multiple
processes. A new default-on tuner before state isolation, or a fast single
kernel without a whole-step gain, would not settle the question.

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
execution. Reset state for the timestamp-ring advance runs too. The structured
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

Detail: ResNet's only eligible classifier outer product spends median 538 ms
in qualification out of 546 ms total search; sampling is about 7 ms. The phase
timer has not yet separated CPU checks from uploads/readbacks and GPU work.
Read-optimized staging is a testable hypothesis, not an established cause.
Do not remove tiny-operand checks or reinterpret a cheaper search as a model
speedup. See the [complete protocol and results](../experiments/crossover-2026-09-06/README.md).

## Talk outline

The actual workshop slot length is not confirmed here. For a 12-minute talk,
prepare approximately eight slides and leave detail in backup:

| Time | Slide and message |
|---|---|
| 0–1 min | Question: can a compact graphics-API layer support useful training and deployment? |
| 1–2.5 min | Architecture diagram: shared graph/autodiff/plan, native embedding boundary. |
| 2.5–4 min | Protocol: five machines, explicit precision, minimal-shape and F+L+B definitions. |
| 4–6 min | Results: 12/20 minimal wins, 1.78× training median; show losses as prominently as wins. |
| 6–7.5 min | Numerical gates and 780M dispute; explain sampled outputs and norm vectors. |
| 7.5–9 min | Profile localization: convolution derivatives and pre-optimization Metal attention. |
| 9–10 min | Deployment and productivity scope; Quest inference, not headset training. |
| 10–12 min | Limits, alternatives, bounded autotuning direction and takeaway. |

Backup slides: complete strict/accelerated tables, portability equation,
oracle sensitivity, precision rollback example, compile/closure accounting,
variant-selection contract, API/operator limitations, stronger PyTorch
baseline plan. For a longer slot, expand method and architecture before
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
7. Say the main result once using the GPU-only reference set and once using
   the full five-machine set without mixing denominators.

Answers: (1) `dX=[32,784]`, `dW=[784,128]`, with derivative matmuls protected;
(2) same-group execution may overlap without a separating barrier;
(3) 0.75 versus arithmetic 0.833; (4) `g` versus `-g`;
(5) `1/(0.27+0.73/2)=1.57×`; (6) optimizer moments/counter, KV state and
accumulated gradients; device/driver, generator revision and numerical policy;
(7) GPU-only training 5/19 wins and 1.78× median, all-machine training 8/24
wins and 1.50× median. These are frozen observations, not today's code.
