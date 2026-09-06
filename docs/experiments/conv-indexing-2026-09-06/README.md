# Exact convolution indexing: correctness and cost

## Correctness repair

The scalar forward/dX/dW templates and generated cooperative forward/dX now
use integer division for index decomposition. Four reciprocal uniforms and
the tuner's f32 interval filter are removed; checked products, signed coordinate
bounds, padded-K loop bounds, declared capacities and dispatch limits remain.
There is no approximate-division fallback, new precision policy or model rule.
The transposed-weight and flattened-upstream loads also reuse their original
remainders instead of decomposing coordinates only to reconstruct them.

Before the repair, the new full f64 oracle reproduced a GPU error on the
RTX 5070: batch 2, Ci=3, H=1, W=41, Co=5, 1×1 kernel, stride 1, no padding,
tile 32, ordinary upstream gradients. `dW[0]` was `0.93909097` versus reference
`0.8370159872050777`. This is a deterministic regression test, not an error-rate
estimate. The original arithmetic was at source `dc68693`; the new test was
added before changing it. The old reciprocal maps 41/41 to zero and crosses
the wrong batch boundary. Divisors 47 and 55 have the same CPU counterexample.

Independent full forward scatter oracles cover all forward outputs and both
derivatives, both scalar tiles, ordinary and `1e-12` gradients. The new six
shapes exercise spatial/batch and rectangular kernel/channel boundaries at
41/47/55, in addition to eight padding/stride/tile-edge shapes. The tuner runs
on every new shape with its original numerical gates, full-state isolation and
zero-byte/zero-time skips. Generated forward and dX actually execute on three
additional batch-2, H=16, W=41/47/55, Ci=64, Co=128, 1×1 cases. This device's
cooperative path is f16-input only: exactly representable bounded inputs isolate
indexing; native-f32 execution remains unqualified here. Naga validates native
f32 tile-8/tile-16 generation independently.

## Predeclared cost check

Commit this protocol and the runner before measurements. Freeze separate source
tags `evidence/conv-indexing-baseline-2026-09-06` (old arithmetic, new runner)
and `evidence/conv-indexing-exact-2026-09-06` (repair, same runner). Keep all six
fresh-process records. Source changes between these tags are the correctness
repair and its tests/documentation, not workloads, timing or precision settings.
This is a correctness-cost diagnostic, not a new engine or paper speedup claim.

Reuse `profile_training --conv-indexing` with the existing three fixed-input,
optimizer-free F+L+B cases: SmolLM2, Whisper, ResNet-50. Rotate case order by seed
1..3 as in the original profiling cohort. Build both release executables before
timing, using the [archived Blade 0.9 lock](../readback-2026-09-06/Cargo.lock.gz),
SHA-256 `72cecddd0f090e82d5db46bc6f7f80429a058b2afa0eaec84a9b27e7340a4a80`.
No local Blade edits or other benchmark, build, qualification or heavy host work
runs concurrently with the retained cohort.

Run in this fixed order: baseline-1, exact-1, exact-2, baseline-2, baseline-3,
exact-3. All cases retain strict scalar f32, default greedy fusion/aliasing,
no tuning, normal GPU timestamp capability enabled, fixed initialization and
inputs. Warm 30 steps; settle five before each block; collect 20 normal
`step+wait` samples before and after five instrumented full-step captures.
The profiler's pass-per-dispatch costs are localization, not normal wall time.
All normal timed losses and full profiled states are checked outside timers,
including before the profiling ring advances. Keep the original nonzero-gradient
preflight, finite/tolerance checks, full-state rosters, immutable dispatch keys,
resident-buffer requests and continuous 250 ms bounded telemetry.

The new mode also records SHA-256 of the complete reference and final declared
snapshots: sorted tensor names and lengths, little-endian f32 bits, Adam counter
and moment allocation size, with explicit length delimiters and a version prefix.
These hashes run outside timers. Require equality within each process and across
both revisions for each case before interpreting its timings. Hashes supplement,
not replace, the in-memory numerical checks; raw vectors are not retained and
cannot be reconstructed by replay. Dispatch contracts and requested tensor memory
must also match across revisions. Any failure disqualifies that case's timing;
keep the record and investigate without silently relaxing the requirement.

Report every process's before/after normal medians, within-process drift and
seed-matched baseline/exact ratios separately for both normal blocks. Report
instrumented convolution forward/dX/dW sums separately, never as Amdahl or
whole-step gains. SmolLM2 is a no-convolution control; Whisper also has convolution
front-end work. These are sequential source-level process pairs, not interleaved
same-process A/B observations: do not apply the tuner's paired-MAD test to them,
invent confidence intervals, pool models or interpret near-unity as equivalence.
No acceptance threshold, default or schedule changes are chosen from these data.
The correctness fix stays even if exact division costs time.

## Lean split-K follow-up

Split-K cannot be represented by a tile flag: it changes the reduction order,
writes `[split, M, N]` partials and needs a second dispatch before consumers run.
The existing tuner promises identical bindings/allocation plans and geometry-only
swaps. Do not smuggle persistent partials into that contract or omit their bytes
and reduction time. Extend a candidate to a bounded dispatch sequence with an
explicit scratch lifetime, reusing the current class key, qualification, paired
measurement and decision code. The existing [SumRows kernel](../../../src/shaders/sum_rows.wgsl)
already reduces `[split, M*N]` partials with a fixed f32 tree and is used for
normalization weight gradients. Reuse it after a scalar split along reduction
tiles; no new reduction shader, float atomics, model names, im2col buffer or
separate tuning framework is needed. Test uneven K,
multiple batches, tiny gradients and next optimizer updates before a separately
predeclared whole-step cohort. The exact-division cost check is not split-K evidence.

## Retained results

All six fixed-order attempts ran on September 6, 19:01:34.673–19:06:08.413 UTC,
RTX 5070 / driver 595.71.05, i5-12400F, Rust 1.98.0. No run was discarded or
repeated. [Baseline 1](baseline-01.json.gz), [exact 1](exact-01.json.gz),
[exact 2](exact-02.json.gz), [baseline 2](baseline-02.json.gz),
[baseline 3](baseline-03.json.gz), [exact 3](exact-03.json.gz) retain every raw sample,
full dispatch contract, numerical comparison and telemetry sample.
Records are losslessly compressed with `gzip -n`; decompression reproduces the
original bytes, checked against [raw SHA-256 digests](RAW-SHA256SUMS). Replay uses
the existing flate2 dependency, with no new production dependency or benchmark
framework. This keeps the source checkout small without dropping evidence.

| Variant | Source / immutable tag | Executable SHA-256 |
|---|---|---|
| Baseline | `aa1344e784a37462b8261aed97cb804c11ce8ba3` / `evidence/conv-indexing-baseline-2026-09-06` | `f4e681345ac30729ce45bfdbdbd0dd3fce395c4dbfa6f6a9fe8016fa5b015a59` |
| Exact | `45304b884d821875c476f6d28446051ad22fe35f` / `evidence/conv-indexing-exact-2026-09-06` | `7f0c7fbae450e8a16b553a4a6de3bfed2c8e3c4ae7ea505ca831371344feabde` |

During setup, sharing a Cargo target directory between worktrees caused an old
library artifact to be reused by a new regression test. That test reproduced
the old failure again. Before any retained timing, the package's build artifacts
were cleared, the repaired code rebuilt, its embedded shader sources checked,
and the complete ordinary suite, all six convolution suites, ignored state tests,
profiling preflight and Clippy passed. The baseline executable had already been
copied out and retained its hash. For reproduction, use separate target
directories per revision; do not rely on a shared target's freshness heuristic.

All eighteen cases pass, including all 90 full profiled-state checks and 720
normal timed-loss checks. Reference/final hashes agree within each process and
across both revisions, covering all declared parameters, gradients and loss
partials/scalar, with zero optimizer state. Dispatch contracts, pipeline keys
and requested tensor memory also agree. This establishes bit identity on these
fixed workloads, not correctness of the old arithmetic for arbitrary shapes.
The width-41 independent oracle is the counterexample to that broader claim.

Each cell below is the median of three process medians, in milliseconds. Before
and after refer to the intervening instrumented capture, not old/new source.

| Workload | Baseline before | Exact before | Baseline after | Exact after |
|---|---:|---:|---:|---:|
| SmolLM2 F+L+B | 2.7089 | 2.7298 | 2.8941 | 2.9306 |
| Whisper F+L+B | 2.8545 | 2.9044 | 3.1764 | 3.5513 |
| ResNet-50 F+L+B | 17.5526 | 21.5507 | 17.5482 | 21.5627 |

ResNet shows a consistent cost increase: 22.78% in the before aggregate and
22.88% after. Its seed-matched baseline/exact ratios are 0.81593/0.81342/0.81418
before, and 0.81457/0.81361/0.81383 after. All six within-process ResNet drifts
are below 0.43% in magnitude. This is a local correctness-cost regression,
not a paired-MAD decision or proof about another device. The earlier convolution
tile cohort measured different, reciprocal-indexed source; its historical
timings do not describe this repaired implementation, tuned or untuned.

SmolLM2 has no convolution dispatches, yet drifts as much as 23.69% within a
process. Whisper drifts up to 90.53%; its after-block baseline/exact ratios span
0.7160–1.5315. Neither short workload supports a stable no-regression/equivalence
claim. [The replayed summary](summary.json) retains every process median, drift
and source-pair ratio; no model pooling or confidence interval is reported.

Instrumented ResNet median sums, kept separate from normal wall time, are
forward 4.9324→5.3787 ms, dX 6.3193→9.3808 ms and dW 5.1860→6.5679 ms. They
localize most additional instrumented work to dX, but are not an Amdahl model.
All 1,073 telemetry samples remain: utilization 0–89%, graphics 217–2902 MHz,
memory clocks 405–14001 MHz, temperature 43–56°C, device memory used
271–710 MiB, power 7.04–68.90 W. These coarse observations do not identify the
cause of short-case drift or measure peak memory.

[CPU replay](../../../tests/training_profile_evidence.rs) reuses the original
profile validator for both cohorts and checks exact source/executable/lock,
rotations, nonoverlapping process windows, all numerical rosters, dispatch
geometry, profile arithmetic, normal medians/losses, cross-source state hashes,
and unchanged memory. Mutation tests reject changed identities, attribution,
boundaries, samples and cross-source hashes. Raw vectors are not archived;
replay verifies recorded observations, not a new oracle execution.

Keep the exact repair. Recovering this cost calls for a separately qualified,
reusable exact indexing implementation, not restoring the old approximation or
fitting model thresholds. The subsequent split-K prototype can reuse SumRows,
but still needs explicit candidate-sequence/scratch accounting and independent
numerical and whole-step evidence. Tuning remains opt-in; frozen paper tables
and prior evidence tags are untouched.

Follow-up: the [shared integer-divisor cohort](../conv-divisor-2026-09-06/README.md)
qualifies an all-integer high-product/remainder correction without width
exceptions. It observes only about 2% lower ResNet time than exact division,
with bit-identical full states. Most of this repair's cost remains; the two
cohorts are separate source/process comparisons, not a jointly measured triple.
