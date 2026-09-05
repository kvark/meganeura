# How to win while staying general

This is a proposed engineering and experiment plan, not a claim of new
speedups. No GPU work was run during the September audit. Do not execute the
GPU portions until the user releases the busy device.

## The objective

Win useful workloads under a matched numerical and workload contract, without
embedding model names or vendor-name performance exceptions throughout the
compiler. Preserve one native graph/training/deployment path. Universal
superiority over mature libraries, serving engines and distributed training
systems is not a realistic near-term acceptance criterion.

Minimalism should constrain the *description of the search space*, not forbid
specialization. A reusable tiled contraction with a few layouts can cover many
models. A source branch saying "if Whisper on this GPU" cannot. Equally,
automatic tuning over two bad implementations cannot discover an algorithm
the generator cannot express. Improving reusable kernel families and improving
selection are complementary work.

## What is automatic today, and what is missing

Graph construction helpers, autodiff, greedy fusion, shader specialization,
scheduling, allocation aliases and device capability selection already work.
Measured tuning is optional, family-wide and demotion-only: it compares an
already-promoted cooperative family with saved scalar fallbacks. Heuristic-
rejected cooperative candidates are absent from the search. Winners are not
persisted. The misleadingly named `runtime::auto_tune` is capability probing.

The CPU-only shape census now exposes the amount of repeated work. A full
SmolLM2-135M training graph at sequence 128 has 1,123 pre-runtime dispatches
and 19 `(entry,M,N,K,precision)` matmul classes. Cached decode at maximum
sequence 256 has 423 pre-runtime dispatches and six such classes. Many classes
repeat 30 or 60 times. These are static counts, not runtime dispatch counts,
measurements, or a complete tuning key. They make bounded class-level tuning
plausible without tuning each transformer layer independently.

Reproduce the structural inspection without a GPU:

```sh
cargo run --example diagnose_fusion -- smollm2-train smollm2-decode
```

The diagnostic's old fixed barrier-cost estimate was removed. The absence of
matches in its narrow legacy fusion matcher does not establish absence of
optimization opportunities.

## Proposed minimal tuner contract

Separate three stages rather than folding all decisions into one cost model:

```text
Generator candidates
  → hardware legality and binding/geometry consistency
  → precision policy and numerical qualification
  → bounded measurements in an equivalent session context
  → whole-step confirmation
  → versioned, persistent winner
```

Start with a small Rust-owned representation using existing plan and variant
types. Avoid introducing a new compiler framework just to search a few knobs.

1. Enumerate a scalar baseline and at most a few alternatives for an exact
   class. Include ordinary, small-tile, cooperative and GEMV where meaningful;
   unsupported candidates never reach timing. Keep heuristic thresholds as
   initial priorities, not hard profitability exclusions.
2. Key classes by operation/direction, dimensions, layout/strides, storage and
   operand precision, epilogue/prologue, horizontal pack, alias/alignment
   constraints and relevant generator knobs. Do not initially bucket different
   shapes by an approximate class and assume a winner transfers.
3. Build candidates with their complete geometry, padding, pipeline and
   bindings together. Pure exact-key lookup should fail visibly on a missing
   variant rather than silently choose a geometrically incompatible fallback.
4. Use scratch state or a complete snapshot/restore contract. Optimizer
   moments, step counters, gradient accumulation, derived weights, KV caches
   and external buffers must not advance during measurement. Zero-filled
   build-time tensors are not a representative correctness workload.
5. Validate representative nonzero, adversarial inputs before accepting a
   candidate. Include tiny derivative operands, odd dimensions, long
   reductions and epilogue paths. A finite result alone is not sufficient.
   Qualification is evidence for a defined domain, not a proof for all inputs.
6. Measure interleaved baseline/candidate trials with bounded warmup and
   sample budgets. Use medians/dispersion and require a margin over noise;
   avoid irreversible decisions from a single sequential A-then-B run.
7. Prefer frequent, costly classes first; stop when the compile/tune budget is
   spent. Cache enough evidence to report why a choice won. Revalidate the
   combined plan, since individually good fusions and kernels can interact.
8. Persist a winner only with device identity, driver/backend, advertised
   capabilities, compiler/generator revision, numerical contract, validation
   version and full class key. Capabilities and package version alone are
   insufficient. Keep a cheap deterministic path when tuning is disabled.

TVM's search-space/runner/database separation is an established model worth
borrowing in small form. Triton's autotuning API explicitly includes reset and
restore hooks, reinforcing that stateful kernels cannot be timed by blindly
executing them repeatedly.
[TVM MetaSchedule](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html),
[Triton autotune](https://triton-lang.org/main/python-api/generated/triton.autotune.html)

Tuning should expose compile cost, search cost and steady-state cost separately.
The break-even number of uses is approximately
`extra_build_and_tune_time / time_saved_per_use`. If search takes 2 s and
saves 0.1 ms, amortization needs about 20,000 uses. This is an illustrative
calculation, not an observed Meganeura result. An interactive renderer and a
long training run should be able to request different budgets.

## Priority order and decisive experiments

| Priority | General mechanism | Why investigate | Evidence required before claiming a win |
|---|---|---|---|
| 0 | Tuner state isolation, exact variants, stronger correctness gates | A fast wrong or state-mutating candidate is unusable. | CPU invariants plus GPU parity and state-preservation tests on legal candidates. |
| 1 | Shape-level bidirectional variant search | Repeated classes; frozen accelerated regressions; remaining occupancy vetoes. | Holdout shapes, repeated processes, whole-step gains beyond noise, bounded tuning budget. |
| 1 | Reusable convolution derivative tiling/layout | NVIDIA ResNet profile concentrates in backward convolution. | Correct odd/stride/channel cases, dX/dW parity; F+L+B and optimizer-backed step improve. |
| 1 | Metal attention backward schedules | Frozen profile identified dK/dV and dQ costs. | Tune EPT/tiles/workgroup layout jointly; several sequence/head widths, end-to-end memory and time. |
| 2 | Eliminate materializations via existing prologue/epilogue/reduction generators | Common traffic costs; small-kernel populations. | Dispatch/traffic reduction plus wall-time improvement; avoid register-spill and fanout regressions. |
| 2 | Lazy optimizer state and logical checkpoint layout | Memory limits larger training workloads independently of ALU speed. | Peak allocated bytes drop without changed updates or restore behavior. |
| 3 | Layout search, rematerialization | May unlock convolution performance or larger models. | Account for conversion/recompute traffic and complete backward dependencies. |
| Research | bf16, scaled/compensated low-precision derivatives | Scalar f32 can leave matrix hardware idle. | Device/compiler support and exponent-sensitive accuracy, then convergence and speed. |

For convolution, first reuse the contraction generator with explicit indexing
maps for forward/dX/dW. Add a small number of tilings and layouts, not separate
ResNet rules. Split-K or multi-pass weight-gradient reduction is a candidate
only if its temporary storage, reduction ordering and synchronization are
included in the plan and measurement. For attention, tune scalar and legal
cooperative variants under separate precision contracts; memory-efficient
attention need not mean reduced-precision attention.

Do not start by replacing greedy rewrites with a larger e-graph. The current
rule set has not shown a runtime win from that complexity. Add searchable
semantic alternatives only when the generator and an observed workload give
them a useful choice to make. A bytes-moved estimate is a good prior, not a
complete machine model; register pressure can still explain losses even if
prior register-count heuristics predicted poorly.

Do not assume command replay has no possible value forever. Existing profiles
bound what was important for specific revisions and shapes. Measure host
encoding/submission, queue synchronization and kernel time independently when
the device is available. A global spin-barrier persistent megakernel is not a
portable drop-in optimization: inter-workgroup forward progress and residency
can make it deadlock. Backend synchronization changes belong in Blade and
need their own correctness evidence.

## Precision research without weakening the result

The abandoned automatic compensated-f16 derivative route is an important
lesson. Splitting `x` into `hi=f16(x)` and `lo=f16(x-f32(hi))` improves some
mantissa errors but both parts can underflow for a sufficiently small f32
operand. Three MMA products do not recreate missing exponent range. The
August 28 rollback and tiny-gradient test are the correct baseline.

bf16 has f32-like exponent range but much less mantissa precision. Vulkan
separately exposes bf16 types, dot products and cooperative-matrix support;
the relevant feature and operand/tile combination must pass through Blade
and Naga. It is not just one Meganeura enum change. Block scaling or loss
scaling introduces scale bookkeeping, overflow detection, optimizer ordering
and reproducibility questions. Treat those as explicit new contracts.
[Vulkan bf16 proposal](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_shader_bfloat16.html),
[Ootomo and Yokota's mixed-precision analysis](https://arxiv.org/abs/2203.03341)

## Deferred measurement protocol

When the GPU is released, start with correctness, not a full timing sweep:

1. Exercise current regression tests for packed matmuls, scalar fallbacks,
   tiny gradients, odd convolution sizes, device-local memory and checkpoints
   on real cooperative-capable Vulkan and Metal devices. Software Vulkan
   cannot establish these hardware paths.
2. Establish a new development baseline in a new result directory and revision
   pair; never overwrite `paper/results`. Record GPU isolation, clock/power
   policy, drivers, inputs, precision, warmups, compiler settings and seeds.
3. Compare PyTorch default, `reduce-overhead`, `max-autotune`, and appropriate
   eager/provider paths automatically. Check actual graph capture/recompiles
   and kernel selection, not just the option string. Include relevant
   cuDNN autotuning policy. Keep all engines under the same accuracy gates.
4. Split full inference, real cached decoding, F+L+B and optimizer-backed
   training. For serving claims add concurrent request/KV-memory metrics;
   isolated batch-one latency alone is insufficient.
5. Use multiple independent processes with alternating engine order, retain
   raw samples and show uncertainty. Monitor interference rather than silently
   collecting timings on a busy device.
6. Add holdout operator shapes and at least one unfitted model family. Report
   geometric/median ratios with explicit populations, individual regressions,
   peak memory, build/search cost and correctness failures, not only wins.
7. Ablate candidate generation, tuning, fusion, aliasing and backend changes
   independently. Re-run the entire acceptance set after combining winners.

Proposed acceptance: no changed validity decision or persistent state; no
material repeated regression without an explicit documented tradeoff; lower
training gap and preserved minimal-latency wins on GPU-to-GPU references;
finite search budget; winners selected without model-name branches; and a
fresh run able to explain and reproduce its selection. Numeric regression
tolerances should be set from measured process-level noise, not invented now.

For the camera-ready deadline, stronger wording and evidence replay are
mandatory; a new performance sweep or unproven architecture overhaul is not.
If new measurements cannot be completed cleanly, retain the frozen results
and present this work as future direction.
