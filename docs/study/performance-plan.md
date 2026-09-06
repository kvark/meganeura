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

The [six-case holdout follow-up](../experiments/holdouts-2026-09-06/README.md)
now includes nonlinear MLP/SmolLM2 inference, Adam trajectories, Whisper SGD
and ResNet F+L+B. All 30 case runs pass full control-session tensor/state parity,
but **none clears the whole-step guard**. Median ratios range from 0.977× for
MLP Adam to 1.052× for MLP inference. These are synthetic engineering results,
not new cross-engine or convergence evidence. This limits the pilot's scope;
it does not justify default-on tuning or lowering the threshold.

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
Scalar convolution dX/dW now also expose their existing 32/64 tiles to this
bounded search. Other kernel families still use heuristics. Winners are not persisted. The misleadingly named
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

## Implemented search: dense tiles and scalar convolution derivatives

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
| Search space | Existing 32×32 and 64×64 scalar-f32 implementations, plus the device's native-f32 cooperative 8×8 or 16×16 primitive (2×2 output primitives/workgroup), for plain MatMul, MatMulAT, MatMulBT and forward MatMul+Add. Scalar tiles also cover convolution dX/dW, as detailed below. |
| Eligibility | Contiguous row-major f32; fixed supported binding arity; nonzero checked extents; portable workgroup limits; non-overlapping physical bindings. Cooperative candidates honor session policy, capability/smoke tests and existing binding capacity. No f16-input/compensated cooperative, GEMV, horizontal packs, arbitrary prologues/epilogues or reduced storage. |
| Exact class | Entry/direction, M/N/K, derivative precision requirement, declared binding capacities and A/B/addend/output placement. Different initial choices are separate searches. No device-to-device transfer. |
| Complete candidate | Exact pipeline key, variant flags, scalar fallback and recalculated X/Y/Z geometry together; cooperative and scalar axes differ. Logical extents, bindings, access sets and allocation plan stay fixed. Lazy shader rejection is reported; no fallback lookup during trials. |
| State isolation | Private per-class scratch and command encoder. No `Session::step`, no live tensor bindings, no reads/writes of model state. Matching Shared/DeviceTransient bindings, with one bounded upload/readback buffer: Download by default, Shared as an explicit control. |
| Qualification | Two deterministic signed, nonzero, full-mantissa patterns: ordinary magnitudes and tiny `1e-12` A/addend operands. Full logical-output finite and cross-variant comparisons; 32 f64 reference dots including 8/16/32/64 tile edges. Scratch padding is zeroed for inputs and NaN-poisoned for outputs; only logical outputs are checked. |
| Numerical tolerance | `abs(reference-actual) <= scale*1e-5 + abs(reference)*2e-4`, with scale 1 or `1e-12`. A domain-specific screen, not a proof or training convergence test. |
| Timing | Default six complete pairs, alternating AB/BA order; each sample encodes/submits/waits for 16 barrier-delimited repeated dispatches. Upload/qualification/readback excluded from samples, included in total cost. |
| Decision | Up to two sequential challenger comparisons per class. Each uses the last accepted winner as incumbent. Candidate median and median paired gain must beat a 5% margin plus twice the MAD of paired differences. Noise guard is not a confidence interval. Invalid/incomplete comparisons retain the incumbent, including a completed earlier winner. |
| Budget | Default eight classes, 64 MiB GPU scratch including staging, two-second soft total deadline, one warmup per variant. An in-flight driver/validation/submission operation can overrun. Pipeline/CPU memory is not charged to the scratch byte cap. |
| Priority/reporting | Descending repetition×M×N×K×Z structural prior (Z is batch for dX, otherwise 1), stable ties; resolved settings, eligible/visited classes, per-comparison and pipeline costs, exclusions, scratch/time/device-memory skips, qualification/rejection details and raw timings. This prior orders searches, never declares a winner. |
| Lifetime | Choices affect this session only. Semantic plan cache and frozen results stay untouched. |

The dense contract above now extends to scalar NCHW convolution dX/dW, with
`TuneScope::{Dense, ConvDerivatives, All}` selecting the experiment domain.
New options use All; tuning remains default-off. Old missing scope settings
deserialize as Dense, and historical runners explicitly keep Dense.

`TuneClass.conv2d` records all twelve convolution parameters. Equal M/N/K is
not enough: kernel aspect ratio, stride and padding change gathered values,
and batch changes physical NCHW storage. dX uses M=Ci, N=H×W,
K=Co×Kh×Kw and Z=batch; dW uses M=Co, N=Ci×Kh×Kw, K=batch×Oh×Ow and Z=1.
Priority includes Z, and scratch/readback sizes use physical tensors without
allocating im2col. Tile choices change the distinct Small/large shader entry
and geometry together; dense `use_small_tiles` is not a convolution flag.

Both scalar variants retain f32 arithmetic. The full finite/cross-variant
scans include all dX batches, while the 32 f64 dots use convolution indexing
and explicit first/last/scattered batches. Checked integer products and signed
coordinates and padded K loops reject overflow. Convolution now decomposes
indices using integer division, including ordinary forward/cooperative paths
outside the tuner. The earlier f32 reciprocal interval filter is obsolete and
removed, along with four uniform fields. This admits formerly excluded shapes
without relaxing validation. Forward and cooperative convolution remain outside
the search, but receive the shared correctness repair.

This distinction matters: multiplying by the rounded f32 reciprocal mapped
`41 / 41` to zero. A batch-2, width-41 GPU regression produced `dW[0]=0.9391`
instead of the independent f64 oracle's `0.8370`. Full oracles now cover fourteen
scalar shapes, both tiles and ordinary/tiny gradients; six generated cooperative
shapes execute on this GPU with exactly representable f16 operands. Native-f32
execution still needs hardware qualification. The [separate cost protocol](../experiments/conv-indexing-2026-09-06/README.md)
keeps correctness and performance evidence distinct.

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

## Lessons from optimizer-backed holdouts

The fixed [September 6 protocol and records](../experiments/holdouts-2026-09-06/README.md)
keep normal optimization and compare complete parameter, gradient and allocated
moment arrays at matched training ages through step 78. All 140 isolated
comparisons qualify; 31 choose challengers. Nevertheless no process/case
clears the descriptive gain or regression guard, and four MLP Adam processes
have lower whole-step ratios. Kernel qualification and measured isolated
profitability are necessary screens, not end-to-end acceptance.

Coverage is narrow even for these graphs. Only 1/512 ResNet plan dispatches is
eligible (the classifier weight gradient); none of its convolutions is searched.
Its median search cost is 1.12 s for no changed choice. SmolLM2 Adam and Whisper
SGD reach the eight-class cap. These counts do not measure time shares and do
not establish that raising the cap would help. Search phase profiling and
whole-step profiles should precede a larger budget or a new kernel family.

The immediate follow-up adds structured preparation/qualification/warmup/
sampling wall times to new `TuneOutcome` reports, with partial early-exit
accounting. This instrumentation postdates the measured source; the old raw
records remain unchanged and deserialize with `phase_times: None`.
It does not change selection policy or explain the old ResNet cost after the
fact. Use it for the next separately recorded profile; see
[phase boundaries and missing-data semantics](observability.md).

The unchanged ResNet control in one process has a 1.071× ratio of medians but
only 0.01470 ms median paired gain. The paired guard correctly rejects it.
This makes A/A controls, randomized/crossover session roles and interference/
clock telemetry concrete next work for automatic confirmation. Keep every
attempt and retain one predeclared acceptance policy; do not pick a favorable
process or quietly loosen the margin. Cross-session bitwise equality here
does not strengthen the frozen paper's sampled-output/gradient-norm contract.

The [controlled crossover cohort](../experiments/crossover-2026-09-06/README.md)
is retained separately: six processes, three diagnostic repeat cases, an
untuned/untuned control and four role-reversed blocks with matched evolving
states. It uses a checked selection-only swap instead of resetting training
from an incomplete snapshot. Confirmation requires a stable control and the
existing guard in both winner-side orientations and pooled pairs. Device
telemetry and the new search phase timers are retained; this does not change
runtime selection policy or enable tuning by default. Dense inference passes
both orientations in all six processes (median 1.177×); MLP+Adam has no
confirmed gain, with two unstable controls; ResNet remains unchanged. All
declared tensor comparisons are bit-exact through Adam step 178.

Qualification took 98.26–98.65% of ResNet's search wall time in that cohort
(median 538 of 546 ms); sampling took about 7 ms. That motivated the separately
recorded experiment below, not a retrospective change to those measurements.

## Qualification cost: measured readback staging

The [six-process Shared/Download experiment](../experiments/readback-2026-09-06/README.md)
uses published Blade 0.9 and Naga 30 in both arms. New nested timers separate
input preparation, CPU upload copy, upload transfer/wait, dispatch/wait,
readback transfer/wait, CPU readback allocation/copy and numerical validation.
These are accumulated host wall times, not GPU timestamps. The staging buffer
alone changes memory policy; candidate bindings, capacities, ordinary/tiny
patterns, complete finite/parity scans and sampled f64 dots remain identical.

| Diagnostic repeat case | Median total search, Shared → Download | Median qualification, Shared → Download | Median per-process search ratio |
|---|---:|---:|---:|
| Dense inference | 73.20 → 43.98 ms | 51.82 → 6.79 ms | 1.661× |
| MLP+Adam | 175.44 → 63.53 ms | 144.88 → 10.42 ms | 2.803× |
| ResNet F+L+B | 606.46 → 38.85 ms | 598.38 → 20.59 ms | 15.567× |

ResNet's CPU readback allocation/copy accounts for 582.24 ms with Shared versus
2.03 ms with Download; its unchanged numerical validation takes 6.03 versus
6.01 ms. Download is not free: preparation increases from 0.65 to 10.41 ms,
upload transfer/wait from 1.12 to 3.59 ms and readback transfer/wait from 0.54
to 2.33 ms. Preparation also increases on both other cases. The first dense
Download search is slower overall and is included, not discarded.

All 18 case runs and 108 class comparisons qualify. Full tensor/counter checks
are bit-exact through Adam step 178 and allocation requests match. ResNet's
total/qualification/readback-copy gain guards pass, and neither other case
has a total-search regression guard. This meets the predeclared promotion
rule: `TuneOptions::default()` now selects Download for private staging.
Explicit Shared and historical missing-field deserialization preserve the
old policy. Tuning remains default-off. Six process pairs on one GPU do not
establish fleet behavior; Metal maps both policies to shared storage.

The [allocation and exact-size reuse follow-up](../experiments/staging-reuse-2026-09-06/README.md)
now splits preparation and times cleanup. A tagged localization profile puts
19.60/20.79 ms dense preparation and 32.05/33.98 ms MLP preparation in staging
allocation, while candidate binding allocations cost hundredths of a millisecond.
That motivates one reusable staging slot, not a broader binding/encoder pool.

`TuneStagingReuse::SameSize` now defaults on **within a tuning call** after a
separate six-process comparison passes its predeclared gates. Sizes must match
exactly; a change releases the old buffer before new allocation. Full binding
plus staging requests still count against the same cap, and nothing remains
after return. Every input upload, poison, readback and numerical check repeats.
Fresh remains an explicit control and the historical missing-field policy.

| Case | Median total search, Fresh → SameSize | Staging allocations per search | Median paired process ratio |
|---|---:|---:|---:|
| Dense inference | 44.01 → 31.84 ms | 3 → 1 | 1.378× |
| MLP+Adam | 64.27 → 45.46 ms | 5 → 2 | 1.414× |
| ResNet F+L+B | 38.51 → 39.24 ms | 1 → 1 | 1.007×; no guarded change |

The ratio column is not a ratio of medians. ResNet is strongly order-sensitive;
the first dense SameSize run is slower overall. Both remain in the records.
All 108 comparisons qualify with bit-exact state checks through Adam step 178,
identical per-comparison/peak scratch requests and zero retained staging at
return. Cleanup includes the final retained-buffer release; it is not moved
outside total search. No qualification or validation gain guard passes.

This closes the measured staging-allocation opportunity at this scope. The
[subsequent whole-step profiles](../experiments/training-profile-2026-09-06/README.md)
put 60.66–60.77% of instrumented ResNet dispatch time in backward convolution
and 36.25–40.58% of SmolLM2's in backward attention. Full profiled state agrees,
but ordinary before/after timing drift is up to 27% for SmolLM2 and 100% for
Whisper. These F+L+B-only profiles rank work; they neither measure optimizer
passes nor establish candidate gains. A newly exposed non-same-padding dX bug
is fixed before widening the search, with full independent f64 derivative tests.

Exact-class qualification and selection for the existing 32/64 convolution
derivative tiles are implemented. The [corrected crossover](../experiments/conv-tiles-corrected-2026-09-06/README.md)
retains six ResNet F+L+B and small Adam/SGD convolution-chain processes.
ResNet changes eight dX dispatches and observes median 17.5808→16.7293 ms,
ratio 1.05056×, but all six decisions are inconclusive under the unchanged
5%+noise guard. The small chains keep their original choices. All 84 isolated
comparisons qualify, full compared state is bit-exact, and both optimizer
sessions actually update parameters through age 178. Search costs median
640/37/38 ms respectively; sampling dominates ResNet search. Its structural
prior visits only 8/45 exact derivative classes, not the expensive stem dW.

The original small-case builder violated the documented flat operand contract;
both controls agreed on zero loss/gradients after a zero-workgroup forward
dispatch. Those twelve cases remain archived and disqualified. The public
helper now rejects malformed operand shapes, and the runner requires nonzero
training signals and actual parameter updates. Independent full
f64 scatter oracles cover forward outputs, eight shapes and both derivative tiles before/after search,
ordinary/tiny gradients, state isolation and budget skips. Distinct-state
Adam swaps cover subsequent accumulation/clipping updates as well.
The long-reduction, small-output dW classes motivate the next bounded split-K
candidate, including its partial storage and final reduction. It needs a small
extension from one dispatch to a candidate sequence with explicit scratch
lifetime: the current geometry-only swap promises a fixed allocation plan.
Reuse class keys, qualification and decision logic; do not hide persistent
partials outside the budget or create another tuner. Keep these measured
choices about algebra, not model names.
Do not infer that cross-call pools, larger budgets
or weaker checks follow from this result. Cheaper search neither supplies
convolution candidates nor turns MLP's earlier inconclusive whole-step ratios
into a training gain. Tuning remains opt-in.

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
