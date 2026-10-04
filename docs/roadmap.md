# Roadmap

Updated October 4, 2026. Keep optimizations reusable across operators and shapes,
with numerical qualification and whole-step measurements determining adoption.
The [compiler-search design](compiler-search.md) describes the current direction;
[experiments](experiments.md) records measurements and rejected candidates.

The September tuning-foundation milestone is closed. The bounded compensated
weight-gradient candidate failed 10 of 240 accuracy rows, its arithmetic was
reverted, and automatic split-K promotion was deferred. The work below is an
engineering backlog, not another condition on the publication freeze. Published
results retain their recorded revisions and
[artifact definitions](../paper/p3hpc/artifact/README.md#original-submission-artifact-legacy).

## Compiler and shader generation

Repeated-region outlining, bounded saturation, traffic-based extraction and
matmul epilogues are implemented. Construction-time measured search now retains
alternative graph forms and scalar matrix schedules together, lowers them through
the normal compiler, and qualifies private sessions before selecting a winner.
The earlier plan to select one graph by traffic before tuning its kernels has
been superseded; see [the search API and its limits](compiler-search.md#api-and-current-boundary).

Remaining work:

- Broaden useful graph and implementation choices within explicit construction
  budgets. Keep a legal incumbent when a comparison is incomplete or invalid.
- Complete pipeline, binding, geometry, precision and padding contracts for
  additional kernel families; simplify remaining runtime fallback hierarchies.
- Explore cross-region fusion where opaque cut edges currently prevent it.
- Simplify runtime-specialized shader generation. Keep WGSL/Naga while replacing
  exact-text rewrites with explicit specialization points; evaluate alternative
  frontends against the [shader investigation](shader-generation.md).

## Memory and persistence

Lifetime analysis, buffer aliasing, device-local intermediates, lazy optimizer
state and logical format-3 checkpoints are implemented. Checkpoint restores
preflight all records, and memory reports separate graph buffers, moments,
accumulators and auxiliary allocations. Logical sizes govern optimizer work;
physical padding is not training state.

Remaining work:

- Measure driver-peak memory on larger workloads and qualify restores across
  Metal and Vulkan. Report allocation failures in a structured way.
- Make checkpoint file replacement crash-atomic. Design a complete training-loop
  snapshot separately from tensor/optimizer persistence if applications need it.
- Investigate activation rematerialization using liveness and a memory budget.
  Outlined regions can bound the extraction problem, but recomputation and
  barrier costs must be measured. No activation-checkpointing policy is exposed
  by the current session API.

## Precision

Automatic compensated-f16 derivative selection was rolled back. Splitting a
value into f16 high/low parts improves mantissa precision but does not preserve
f32's exponent range: both parts can underflow. Protected derivatives use native
f32 cooperative tiles or scalar f32; `AllowF16` explicitly relaxes that policy.
Cooperative backward remains experimental and default-off.

Potential work requires independent numerical evidence:

- bf16 cooperative matrices need end-to-end type, tile, feature and backend
  support. Their exponent range addresses underflow, not f32 mantissa accuracy.
- Mixed-precision training needs an explicit graph precision policy, f32 master
  weights and loss-scaling behavior. Compare gradients and training trajectories
  against f32 before making it an automatic choice.

## Dispatch latency and convolution

Horizontal matmul fusion is implemented. Use whole-step measurements and current
profiles to establish whether dispatch count, synchronization, submission or
individual kernels limit a workload.

Convolution indexing is exact. The invariant-divisor follow-up recovered some
of the cost of replacing the incorrect floating reciprocal, but did not erase
it. Scalar derivative tile search and explicit split-K probes exist; the
[split-K qualification failures](experiments.md#split-k-sequence-2026-09-06)
prevent automatic promotion of the rejected weight-gradient candidates.

Remaining work:

- Recover convolution indexing and derivative costs without weakening address or
  numerical correctness. Qualify the complete split/reduction sequence and its
  temporary storage before considering a new split-K candidate.
- Revisit parked cooperative and staging variants on the hardware where they
  would be used. Keep rejected measurements in the experiment record.
- Explore layout-tagged NCHW/NHWC alternatives, including conversion costs, only
  when profiles justify the extra compiler and kernel surface.
- Treat persistent decode kernels as research. Vulkan/WGSL forward-progress and
  synchronization constraints need an explicit design; compare with existing
  fusion before expanding the runtime around them.

## Measured selection

Exact-class kernel probes use private scratch, numerical qualification,
interleaved timing and explicit budgets. Construction-time search reuses that
machinery while also comparing alternative complete plans. Tuning stays opt-in.
The [September holdouts](experiments.md#holdouts-2026-09-06) and
[crossover](experiments.md#crossover-2026-09-06) established state parity and a
repeatable gain on one dense chain, not general model or fleet speedups.

Remaining work:

- Qualify native-f32 cooperative execution and broader device coverage. Confirm
  selected plans end to end on representative inputs and persistent state.
- Extend candidates to packed weights, additional cooperative layouts, complex
  fusion and attention only with complete interface and precision contracts.
- Persist winners with device, driver, backend/compiler/generator revision,
  numerical policy and validation provenance. Semantic plan caching alone is
  insufficient for reusing a performance decision.

## Quantized fine-tuning

Frozen quantized weights, trainable dense adapters and `StopGradient` provide
building blocks for LoRA-style training. A useful application-facing next step
would add a small graph helper, an example and a benchmark that verify gradient
flow and memory use. The opportunity is a common Vulkan/Metal path, with
numerical and memory qualification preceding broader training claims.
