# How to win while staying general

This separates the implemented search from the remaining engineering and
experiment plan. The initial September audit was CPU-only. After the user
released the GPU, the scalar search passed device qualification on an RTX
5070 (driver 595.71.05). This does not establish a performance improvement or
native-f32 cooperative coverage; that device advertises only f16 matrix tiles.
New tuning experiments are separate from the frozen paper evidence.

The [first whole-step transfer experiment](../experiments/tuning-2026-09-05/README.md)
now retains five independent processes: two of four synthetic dense chains
improved by median 1.151× and 1.127×, while the two smaller cases kept their
initial tiles and showed no benefit. All outputs matched their untuned
references exactly. This is scoped Meganeura-versus-Meganeura evidence, not a
PyTorch or model-training result; search amortization takes thousands of steps.

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
Measured tuning is optional. The September follow-up replaces the old
family-wide cooperative demotion tuner with bounded exact-class f32-matmul
search. Scalar 32/64 tiles and advertised, smoke-tested native-f32 cooperative
tiles can compete. Cooperative occupancy and native-8 large-shape thresholds
choose the initial implementation, not the legal challengers in this domain.
Other kernel families still use heuristics. Winners are not persisted. The misleadingly named
`runtime::auto_tune` is capability probing.

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

## Implemented search: scalar and native-f32 matmul tiles

`Session::tune()` uses defaults; `Session::tune_with(TuneOptions)` returns a
serializable `TuneReport`. `SessionConfig { tune: true }` invokes the safe
default search on both a cache hit and a newly compiled plan. Core code is
Rust, with no new dependency or per-model/per-vendor rule.
[Search types and decision logic](../../src/tune.rs),
[runtime runner](../../src/runtime/tuning.rs)

The old API entry point remains, but `TuneOutcome` now contains class/tile
evidence instead of `family/coop_ms/scalar_ms/kept_coop`. Consumers reading
those former fields must migrate. The old live-step tuner is removed, not left
behind as a second unsafe default path.

| Contract | Current implementation |
|---|---|
| Search space | Existing 32×32 and 64×64 scalar-f32 implementations, plus the device's native-f32 cooperative 8×8 or 16×16 primitive (2×2 output primitives/workgroup), for plain MatMul, MatMulAT, MatMulBT and forward MatMul+Add. |
| Eligibility | Contiguous row-major f32; fixed supported binding arity; nonzero checked extents; portable workgroup limits; non-overlapping physical bindings. Cooperative candidates honor session policy, capability/smoke tests and existing binding capacity. No f16-input/compensated cooperative, GEMV, horizontal packs, arbitrary prologues/epilogues or reduced storage. |
| Exact class | Entry/direction, M/N/K, derivative precision requirement, declared binding capacities and A/B/addend/output placement. Different initial choices are separate searches. No device-to-device transfer. |
| Complete candidate | Exact pipeline key, variant flags, scalar fallback and recalculated X/Y/Z geometry together; cooperative and scalar axes differ. Logical extents, bindings, access sets and allocation plan stay fixed. Lazy shader rejection is reported; no fallback lookup during trials. |
| State isolation | Private per-class scratch and command encoder. No `Session::step`, no live tensor bindings, no reads/writes of model state. Matching Shared/DeviceTransient placement, with bounded upload/readback staging. |
| Qualification | Two deterministic signed, nonzero, full-mantissa patterns: ordinary magnitudes and tiny `1e-12` A/addend operands. Full logical-output finite and cross-variant comparisons; 32 f64 reference dots including 8/16/32/64 tile edges. Scratch padding is zeroed for inputs and NaN-poisoned for outputs; only logical outputs are checked. |
| Numerical tolerance | `abs(reference-actual) <= scale*1e-5 + abs(reference)*2e-4`, with scale 1 or `1e-12`. A domain-specific screen, not a proof or training convergence test. |
| Timing | Default six complete pairs, alternating AB/BA order; each sample encodes/submits/waits for 16 barrier-delimited repeated dispatches. Upload/qualification/readback excluded from samples, included in total cost. |
| Decision | Up to two sequential challenger comparisons per class. Each uses the last accepted winner as incumbent. Candidate median and median paired gain must beat a 5% margin plus twice the MAD of paired differences. Noise guard is not a confidence interval. Invalid/incomplete comparisons retain the incumbent, including a completed earlier winner. |
| Budget | Default eight classes, 64 MiB GPU scratch including staging, two-second soft total deadline, one warmup per variant. An in-flight driver/validation/submission operation can overrun. Pipeline/CPU memory is not charged to the scratch byte cap. |
| Priority/reporting | Descending repetition×M×N×K structural prior, stable ties; resolved settings, eligible/visited classes, per-comparison and pipeline costs, exclusions, scratch/time/device-memory skips, qualification/rejection details and raw timings. This prior orders searches, never declares a winner. |
| Lifetime | Choices affect this session only. Semantic plan cache and frozen results stay untouched. |

The former small-tile occupancy cutoff is now an **initial choice**, not a
profitability veto for eligible tuned classes: either tile can win. Untuned,
unsupported or budget-limited work keeps deterministic selection. The default
small-tile geometry now uses exact ceil-divided extents instead of doubling
already-rounded 64-tile counts. Native-f32 cooperative profitability thresholds
are likewise challenged where the candidate fits. GEMV, f16-input cooperative
and complex fusion thresholds remain outside this search; the system is not
fully autotuned.

Cooperative padding is a legality constraint, not a performance veto. N must
be divisible by 16; normal/add matmuls require K≥4; output and addend need
full-output-tile capacity. The tuner cannot enlarge a session or borrow unused
slack from another tenant of an aliased allocation. Thus an unpadded scalar
class may have only one challenger; an already-padded cooperative class may
have both scalar challengers. Capacity is part of the class key. The search
never re-enables cooperative matrices disabled by policy or rejected by the
session's smoke test.

Why start here? Both scalar variants already share a precision, layout,
allocation and access contract. This gives useful search/qualification/report
infrastructure without simultaneously changing precision or late-fusion
semantics. It also removes the old tuner's optimizer/KV side effects without
copying an entire model and all hidden training state.

Limits: scratch input/cache history and isolated barriers are not the live
graph's context; repeated warm buffers can favor a different implementation
than streaming weights. Memory-placement matching does not reproduce every
external allocator property or concurrent workload. The paired noise guard
cannot prove an idle GPU. More input distributions, native-f32 hardware and
Vulkan/Metal fleet qualification remain required. The runtime does not yet
automatically confirm or roll back choices using a whole-step measurement.

On an idle GPU, inspect a report with explicit bounds:

```rust
let report = session.tune_with(meganeura::TuneOptions {
    max_classes: 8,
    max_time: std::time::Duration::from_secs(10),
    ..Default::default()
})?;
println!("{}", serde_json::to_string_pretty(&report)?);
```

The regression target is intentionally ignored by default because it performs
timings. Compile it without running: `cargo test --test tune --no-run`.
Portable scalar qualification:

```sh
cargo test --release --test tune -- --ignored --skip tune_native_cooperative_f32 --test-threads=1
```

All four tests passed on the RTX 5070: output preservation, active-training
tensor/moment/counter preservation plus subsequent optimizer updates against
an untuned control, all four entries across three rectangular/edge shapes,
and budget skips. This is not a complete KV/external-buffer/optimizer matrix.

`cargo run --release --example tune_session -- --device` prints actual matrix
capabilities. Only on a native-f32 device, run
`cargo test --release --test tune tune_native_cooperative_f32 -- --ignored --test-threads=1`.
That test requires real native execution, including below-threshold shapes,
padded rows and a dimension above the native-8 veto; it deliberately fails on
unsupported hardware. Native shaders passed offline Naga/SPIR-V checks here,
but that is not a passed native GPU test.

For an independent whole-step experiment, commit tracked source changes,
then run `cargo run --release --example tune_session -- new-results.json` on
an idle device. The Rust harness builds matched untuned/tuned dense chains,
records the exact revision, lockfile/executable hashes, device and driver,
search costs/decisions, output parity, and 40 alternating whole-step pairs
after warmup. Repeat in independent processes. It refuses to overwrite an
existing file and uses explicit f32 policy, not environment-selected f16
staging. Synthetic chains are a transfer experiment, not a PyTorch comparison
or a new result for the paper's model matrix.

## Target contract for broader tuning

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
| 2, implemented; fleet/peak qualification open | Lazy optimizer state and logical checkpoint layout | Memory limits larger training workloads independently of ALU speed. | RTX tests check actual allocation deltas, preflight rejection and next-update parity; measure peak process memory on larger workloads/backends. |
| 3 | Layout search, rematerialization | May unlock convolution performance or larger models. | Account for conversion/recompute traffic and complete backward dependencies. |
| Research | bf16, scaled/compensated low-precision derivatives | Scalar f32 can leave matrix hardware idle. | Device/compiler support and exponent-sensitive accuracy, then convergence and speed. |

The checkpoint/memory follow-up now avoids Adam/LaProp moments for SGD and
F+L+B-only sessions, accounts for accumulators once, and serializes logical
tensors with whole-file preflight. Optimizer/clip/accumulation lengths ignore
allocation padding, with poisoned-tail regressions. The 1,048,576-element F32 test verifies
8 MiB of unused moment allocations are absent. Different-padding restores
preserve the next Adam update, and eight training-correctness tests pass on
RTX 5070. These are storage/correctness results, not a speedup or measured
driver-peak reduction. Detailed contracts and reproduction live in
[checkpoints and memory](checkpoints-and-memory.md). Remaining work includes
cross-backend restore qualification, large-workload peak measurements and
structured allocation failures; it does not require a new kernel family.

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
