# Tuning allocation and bounded staging reuse

Separate from the readback cohort and frozen P3HPC evidence. The next question
is the preparation cost that increased with Download staging, and whether
one private staging buffer can be reused within a single tuning call without
changing candidate bindings, validation or retained-byte limits.

## Allocation profile before implementation

First commit instrumentation only. Nested preparation timers separate checks,
pipeline setup, scratch binding allocations, staging allocation, command
encoder creation and binding/geometry construction. Pipeline setup is the
same measurement as `compile_time`, not an additional cost. Comparison cleanup
is timed separately, including early exits; it is not part of preparation.
All are host wall times, not GPU timestamps. Historical reports deserialize
missing phases as `None`, not measured zero.

Run one fresh process of the existing `tune_readback` harness, seed 1, on this
instrumented source and retain it as `profile-01.json` here. Its metadata keeps
the readback protocol name because the case, policy and state checks are that
protocol's: three diagnostic cases, Shared/Download staging, full qualification
and matched state through step 178. This is a separately tagged localization
profile, not a seventh member of the old cohort or a performance promotion
decision. Keep every attempt. No other GPU test, build or heavy analysis may
overlap it.

Use that profile to choose the smallest reuse scope. Any reuse comparison
must be separately committed/predeclared before measurement, charge actual
retained staging capacity to each comparison, and release retained resources
before returning from `tune_with`. No live buffers, qualified results, inputs,
or kernel winners may be reused in place of the existing validation work.

## Profile result and chosen scope

The single localization process completed with all comparisons qualified and
bit-exact state checks through Adam step 178. [profile-01.json](profile-01.json.gz)
retains source `42fda56`, tag `evidence/allocation-profile-2026-09-06`, executable
SHA-256 `e1f4a4c36c314195173a3448ca3fb416616f594056006a2c26b49a98bdfd82cd`.
The unchanged [registry lockfile](../readback-2026-09-06/Cargo.lock.gz) has SHA-256
`72cecddd0f090e82d5db46bc6f7f80429a058b2afa0eaec84a9b27e7340a4a80`.
These are one process's Download costs, not medians or a promotion result:

| Case | Preparation | Staging allocation | Binding allocations | Encoder creation | Cleanup |
|---|---:|---:|---:|---:|---:|
| Dense inference | 20.787 ms | 19.596 ms | 0.009 ms | 0.027 ms | 2.095 ms |
| MLP+Adam | 33.979 ms | 32.054 ms | 0.017 ms | 0.042 ms | 3.478 ms |
| ResNet F+L+B | 9.986 ms | 9.317 ms | 0.003 ms | 0.009 ms | 1.351 ms |

Staging allocation dominates preparation. Test only a **one-slot exact-size
staging cache within `tune_with`**. Keep candidate binding buffers and command
encoders fresh. Different requested sizes release the previous buffer before
any new scratch allocation; smaller requests do not keep excess capacity.
Only storage is reused: all uploads, poisoning, readbacks and validation still
run. No staging remains in the session or after a bounded/failed search returns.

## Predeclared reuse comparison

Commit the implementation and this decision before measuring. Use six fresh
serial processes, seeds 1..6, with the same three diagnostic builders and full
prefix/search/step-33/step-178 comparisons as the readback protocol. Both arms
use Download, strict scalar f32, normal fusion/aliasing/device placement,
eight classes, 64 MiB scratch, ten-second soft deadline, one warmup and six
pairs of sixteen dispatches. Session 0 uses Fresh, session 1 SameSize. Alternate
which searches first using `(seed + case_index) % 2`; each arm starts first
three times per case. Each starts with the same heuristic plan on a new session.
Every later loss pair is checked. Continuation is untimed.

Every report retains per-comparison binding/staging requests, reuse flags,
allocation/release counts, peak simultaneous scratch requests and zero staging
bytes retained at return. Device-budget checks charge only new allocations
because retained staging is already resident; the user scratch cap charges
the complete simultaneous binding-plus-staging request. Exact-size reuse must
keep per-comparison requests and cohort peak requests identical between arms.
Memory summaries before/after search must still match; they are not peak VRAM.
Expected staging allocation counts are dense 3 → 1, MLP 5 → 2, ResNet 1 → 1.
The single-comparison ResNet case is the no-reuse control, not a promised win.

Preparation's staging timer includes any previous-size release. Per-comparison
cleanup destroys bindings/encoder and Fresh staging. SameSize's last staging
release is recorded as `TuneReport::final_cleanup`, inside total search time.
Report `cleanup` as the sum of comparison and final cleanup, and
`staging_and_cleanup` as preparation's staging management plus **all** scratch
cleanup, not a falsely isolated staging-only destruction cost. Do not move
allocation or final release outside the end-to-end search cost.

Apply the existing descriptive 5% + twice paired-difference MAD guards to the
six process pairs, reporting all cost medians/ratios and both search orders.
Allow SameSize as the new option default only if all numerical/state/coverage
and memory/count invariants pass, dense and MLP each pass gain guards for both
total search and `staging_and_cleanup`, and ResNet passes no total-search
regression guard. Keep every failed/noisy attempt, with no new cases, exclusions
or threshold changes after measurement. This is one-device search-cost evidence,
not confidence intervals, fleet behavior, model speedup or a Blade comparison.
Fresh remains available and missing historical settings retain Fresh. Tuning
itself stays default-off regardless of the outcome.

```sh
cargo build --release --locked --example tune_staging_reuse
for seed in 1 2 3 4 5 6; do
    target/release/examples/tune_staging_reuse "new-reuse-${seed}.json" "$seed"
done
```

The runner shares the matched diagnostic/state-check code with `tune_readback`;
its explicit options keep that older experiment on Fresh even if defaults change.
It refuses dirty tracked source or existing outputs, saves completed cases
incrementally and retains optional 250 ms GPU telemetry/host boundaries. Keep
the GPU free of concurrent tests, builds, other benchmarks and heavy analysis.
Telemetry cannot establish constant clocks or resolve every short pair.

## Retained comparison results

All six serial processes completed with no retries or discarded attempts:
[1](run-01.json.gz), [2](run-02.json.gz), [3](run-03.json.gz), [4](run-04.json.gz),
[5](run-05.json.gz), [6](run-06.json.gz). [summary.json](summary.json) retains all
cost-level paired medians/noise guards and every process search ratio.

| Identity | Measured value |
|---|---|
| Source | `875c501af4d9b9b5e63274758fa5cb610a629c3e` |
| Immutable tag | `evidence/staging-reuse-2026-09-06` |
| Executable SHA-256 | `13d937c3fdade2488889c26ee1280f6188db35e76d629a6b05b309c5e34fcb78` |
| GPU / driver | NVIDIA GeForce RTX 5070 / 595.71.05 |
| Host / build | Linux x86_64, Intel Core i5-12400F, release Rust 1.98.0, no RUSTFLAGS |
| Lockfile | The unchanged archived readback lockfile identified above |

The compute-process query was empty before the cohort. No builds, GPU tests,
other benchmarks or heavy analysis overlapped it; resident graphics processes
remained. The 453 device samples span 41–61 °C, graphics clocks 180–2902 MHz
and memory clocks 405–14001 MHz over the whole processes, including idle and
validation. These coarse observations do not establish isolation or constant
clocks for each short search.

### Decision and accounting

Times are medians of six per-process costs, in milliseconds. The last column
is the median of the six paired **process ratios**, not a ratio of cost medians.

| Case | Total search, Fresh → SameSize | Preparation, Fresh → SameSize | Staging management + all cleanup, Fresh → SameSize | Median process ratio |
|---|---:|---:|---:|---:|
| Dense inference | 44.005 → 31.837 | 17.586 → 6.272 | 18.273 → 6.097 | 1.378× |
| MLP+Adam | 64.271 → 45.459 | 28.687 → 12.783 | 30.319 → 12.084 | 1.414× |
| ResNet F+L+B | 38.510 → 39.239 | 10.562 → 10.570 | 11.380 → 11.367 | 1.007× |

Dense and MLP pass both required total-search and staging-plus-cleanup gain
guards. ResNet passes neither a total-search gain nor regression guard. All
18 case runs and 36 searches complete; **108 comparisons qualify**, with
bit-exact full tensor/counter comparisons through Adam step 178 and all 175 later training
loss pairs per case passing. Per-search state is unchanged. Candidate classes,
directions, initial/challenger implementations, dispatch multiplicity and
binding placement/capacity match between arms. This is control-session parity,
not an independent engine oracle or a convergence result.

| Case | Staging allocations, Fresh → SameSize | Reuses, Fresh → SameSize | Peak simultaneous scratch requests, both arms |
|---|---:|---:|---:|
| Dense inference | 3 → 1 | 0 → 2 | 2,621,440 bytes |
| MLP+Adam | 5 → 2 | 0 → 3 | 2,486,272 bytes |
| ResNet F+L+B | 1 → 1 | 0 → 0 | 16,396,192 bytes |

Allocation and release counts balance, per-comparison byte requests match,
and retained staging is zero at return in every search. Post-search model
buffer requests also match. This establishes requested-byte accounting, not
driver peak VRAM. MLP's size-change release is inside preparation; every last
release is inside total search. Nothing is hidden in untimed teardown.

All predeclared gates pass. A later commit promotes SameSize in
`TuneOptions::default()`; the measured tag stays fixed. Fresh remains explicit,
omitted historical reuse settings still deserialize as Fresh, and tuning
itself stays opt-in. No validation, binding, candidate, memory-cap or kernel
selection gate changes accompany promotion.

Qualification medians are dense 6.516 → 6.385 ms, MLP 10.826 → 10.207 ms and
ResNet 20.256 → 20.611 ms. Validation alone is 0.888 → 0.869 ms, 2.052 →
2.022 ms and 6.135 → 6.129 ms, respectively. Neither qualification nor validation
passes a gain/regression guard on any case. The measured improvement is in
allocation/lifetime management, not reduced correctness work. Other small
component guards remain visible in the summary; do not inflate them into
additional independent discoveries.

### Every process, including startup and order effects

Each ratio is Fresh total search divided by SameSize total search. The first
column after the seed lists dense / MLP / ResNet search order.

| Seed | First arm | Dense | MLP+Adam | ResNet |
|---|---|---:|---:|---:|
| 1 | Reuse / Fresh / Reuse | 0.698× | 3.013× | 0.931× |
| 2 | Fresh / Reuse / Fresh | 1.420× | 1.409× | 1.084× |
| 3 | Reuse / Fresh / Reuse | 1.324× | 1.422× | 0.914× |
| 4 | Fresh / Reuse / Fresh | 1.415× | 1.393× | 1.105× |
| 5 | Reuse / Fresh / Reuse | 1.368× | 1.418× | 0.909× |
| 6 | Fresh / Reuse / Fresh | 1.389× | 1.409× | 1.128× |

Fresh-first versus reuse-first subgroup median ratios are dense 1.415× / 1.324×,
MLP 1.422× / 1.409× and ResNet 1.105× / 0.914×. ResNet's clear order reversal
is a warning against claiming causality or zero harm from a balanced median.
Its ratio of cost medians is 0.981×, while its median process ratio is 1.007×;
neither is a passed gain. These are six descriptive process pairs, not fleet
samples or confidence intervals.

In process 1, dense reuse costs 62.643 ms versus Fresh 43.732 ms; pipeline
setup is 32.105 versus 0.462 ms. Staging-plus-cleanup still falls 19.038 →
6.035 ms. Conversely, MLP Fresh has 71.628 ms pipeline setup versus reuse's
1.013 ms, inflating its first ratio. Both observations are retained. Driver
caches were not cleared, and no cold-cache or guaranteed one-shot claim is made.

### Reproduction and remaining scope

```sh
cargo test --test staging_reuse_evidence -- --nocapture
cargo test --test tuning_evidence --test holdout_evidence --test crossover_evidence --test readback_evidence --test staging_reuse_evidence
cargo test --example tune_staging_reuse
```

CPU replay recomputes costs, every guard, exact-size hit/miss counts, per-binding
and peak scratch requests, final release accounting, state summaries and
identities for both the profile and cohort. Mutation tests reject altered
policies, missing release, excess capacity, stale reuse flags, corrupt timing,
state, memory or promotion decisions. Raw tensor vectors are not archived:
replay checks the retained summaries, not independent producer authenticity.

GPU tests cover both staging placements and both policies, stale NaN payloads,
grow/shrink/reuse sequences, early-return cleanup, a later oversized class
after a completed comparison, and subsequent optimizer/accumulation updates.
After default promotion, the release unit/integration suite, all three ignored
GPU library regressions and all five scalar tuning tests pass on this host.
Formatting, all-target/all-feature Clippy, strict rustdoc, the Rust 1.92 library
check and both diagnostic examples' CPU tests also pass. The frozen paper
verifier passes six tests and reproduces every table byte-for-byte; external
fixtures and unavailable native-f32 hardware are not newly qualified here.

Native-f32 hardware and fleet coverage remain due. The next decision should
come from whole-step profiles and reusable kernel-family coverage, not an
unmeasured cross-call pool, larger budget or weaker validation. None of these
new results changes the frozen P3HPC tables or establishes faster model execution.
