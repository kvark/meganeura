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
timing, using the [archived Blade 0.9 lock](../readback-2026-09-06/Cargo.lock),
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
measurement and decision code. A scalar split along reduction tiles with a fixed
ordered f32 sum is sufficient for the first experiment; no float atomics, model
names, im2col buffer or separate tuning framework is needed. Test uneven K,
multiple batches, tiny gradients and next optimizer updates before a separately
predeclared whole-step cohort. The exact-division cost check is not split-K evidence.
