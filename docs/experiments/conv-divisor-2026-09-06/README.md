# Shared exact invariant-divisor experiment

## Predeclared comparison

The [preceding repair](../conv-indexing-2026-09-06/README.md) exposed a roughly
23% ResNet F+L+B cost from exact runtime integer division. Keep that repair's
semantics and test a shared, all-integer replacement for invariant divisors.
This is a local implementation comparison, not a new engine or paper claim.

For a positive u32 divisor `d > 1`, compute `m = floor(2^32 / d)` on the host.
On the GPU, take `q = high32(n * m)` and increment once if `n - q*d >= d`.
The truncated reciprocal underestimates `n/d` by less than one for every u32
numerator, so the corrected quotient is exact. Handle `d = 1` by returning `n`.
Implement high multiplication using four 16-bit limb products in shared WGSL;
no GPU u64, floating-point addressing, width exceptions or shape thresholds.
Use this helper for the scalar forward/dX/dW decompositions and generated
cooperative spatial decomposition. Constant-divisor generated expressions stay
integer divisions. Four derived u32 uniforms are the only binding-layout change.

Before timing, execute the shared WGSL arithmetic against a CPU u64 oracle,
including quotient boundaries, limb carries, values beyond 2^24, the full u32
endpoints and deterministic random inputs. Require the existing full f64
convolution oracles for both scalar tiles, ordinary and tiny gradients, padding,
stride, batch and rectangular kernels; run actual generated cooperative paths
where supported. Preserve all tuner legality, numerical and state-isolation gates.

Commit this protocol/runner with unchanged exact-division production code and
tag `evidence/conv-divisor-baseline-2026-09-06`; freeze the qualified helper as
`evidence/conv-divisor-reciprocal-2026-09-06`. Build each revision in a separate
target directory before measurement with the
[archived Blade 0.9 lock](../readback-2026-09-06/Cargo.lock), SHA-256
`72cecddd0f090e82d5db46bc6f7f80429a058b2afa0eaec84a9b27e7340a4a80`.
Keep all attempts, including failures. Do not build, qualify, benchmark another
workload or run heavy host analysis concurrently with the retained cohort.

Reuse `profile_training --conv-divisor` without changing its three fixed-input,
optimizer-free strict-f32 F+L+B cases, seed rotations, 30 warmups, five settling
steps, 20 normal step+wait samples before and after five instrumented captures,
250 ms telemetry, nonzero-gradient preflight, finite/full-state comparisons,
ring-advance checks or timing boundaries. No tuning or precision changes.
Run exactly six fresh processes in order: baseline-1, reciprocal-1, reciprocal-2,
baseline-2, baseline-3, reciprocal-3. Require identical reference/final full-state
hashes within/across sources, dispatch contracts, pipeline keys and requested
tensor memory before interpreting a case's timings. The extra 16 uniform bytes
per convolution binding are not tensor allocation and must be reported separately.

Replay using the existing profile validator. Report all process before/after
medians, within-process drifts and seed-matched source ratios. Report directional
instrumented sums separately, not as whole-step speedups or Amdahl estimates.
These sequential source-level pairs are not the tuner's interleaved paired-MAD
observations. Do not pool models, invent confidence intervals, infer equivalence
from noisy controls or compare with old cohorts as if measurements were paired.
Choose no new performance threshold, workload rule or tuning default from this
cohort. A regression is a retained negative result, not permission to return to
approximate indexing. Split-K remains a separate candidate-sequence change.

## Implementation and correctness argument

[One private Rust helper](../../../src/divisor.rs) derives each multiplier;
[one shared WGSL implementation](../../../src/shaders/divisor.wgsl) is injected
into both scalar templates and generated cooperative source. No new public API,
dependency, pipeline key, tuning option or numerical tolerance is introduced.
The convolution uniform grows from 48 to 64 bytes. Multipliers are derived when
binding the dispatch, so their host cost is included in normal step+wait samples.

For `d > 1`, let `B = 2^32`, `m = floor(B/d)` and `0 <= n < B`.
Then `0 <= n/d - n*m/B <= n/B < 1`. Thus `floor(n*m/B)` is the true quotient
or one less. Its product with `d` cannot exceed `n`; both the product and
remainder fit u32, and comparing the remainder with `d` supplies the exact
single correction. Division by one returns `n`; zero divisors are rejected.
There is no f32 range condition or correction retry loop.

The high product splits each operand into two base-65536 limbs. The two middle
accumulations are at most 4,294,901,759 and 4,294,901,760, respectively, so they
fit u32; their carries plus the high-limb product reconstruct `floor(a*b/2^32)`.
The GPU test executes this same source, reads raw u32 outputs and checks both
the quotient against CPU integer division and high product against CPU u64
multiplication. Inputs cover divisors 1..256, powers of two and neighbors,
quotient boundaries, u32 endpoints, and 100,000 deterministic random pairs
alternating full-width and 16-bit divisors. It runs in the ordinary GPU suite,
including CI; it is not a floating-point norm comparison or a translated CPU
implementation substituted for device execution.

This is the general invariant-divisor strength-reduction idea also used by
[libdivide](https://libdivide.com/), not an integration or copy of that library's
magic-number algorithms. Here a truncated reciprocal and one correction keep
the implementation small. Whether this portable limb expansion is profitable
depends on backend code generation and requires measurements on each device.

## Retained results

All six processes completed September 6, 20:11:24.293–20:16:00.821 UTC, on
RTX 5070 / driver 595.71.05, i5-12400F, Rust 1.98.0. Every predeclared attempt
is retained: [baseline 1](baseline-01.json.gz), [reciprocal 1](reciprocal-01.json.gz),
[reciprocal 2](reciprocal-02.json.gz), [baseline 2](baseline-02.json.gz),
[baseline 3](baseline-03.json.gz), [reciprocal 3](reciprocal-03.json.gz).
Lossless `gzip -n` compression keeps the cohort around 1.1 MiB; decompressed
bytes match [all six original SHA-256 digests](RAW-SHA256SUMS). There were no
discarded or repeated attempts and no concurrent build/qualification work.

| Implementation | Source / immutable tag | Executable SHA-256 |
|---|---|---|
| Exact division | `1b7a09f2140c886dc9b28504e06f15f7d0d9a67c` / `evidence/conv-divisor-baseline-2026-09-06` | `8b91fda6ac63fae73883f18710860f395360be1488852c4731e2fb7fefb32e17` |
| Integer reciprocal + correction | `1f2c7740231fc4998be1de244678f247843533dd` / `evidence/conv-divisor-reciprocal-2026-09-06` | `76e9ead4c7d3412738c54de8ef15bf995cf822a432d35fe81bea6115909e508f` |

The [shared CPU replay](../../../tests/training_profile_evidence.rs) checks all
18 cases, 90 profiled full-state checks, 720 timed loss checks, source/executable/
lock identity, process windows, numerical rosters, telemetry, profile arithmetic
and normal medians. Reference/final state hashes agree within and across sources,
as do dispatch contracts, pipeline keys and requested tensor memory. The extra
16 bytes per convolution uniform binding remain explicitly outside tensor-memory
accounting. Cross-source hash mutation checks cover both this and the prior
indexing cohort. Raw vectors are not archived; replay checks the recorded
observations, not a fresh independent operator oracle.

Each value below is the median of three process medians, in milliseconds.
Before/after refer to instrumented capture between the normal blocks.

| Workload | Division before | Reciprocal before | Division after | Reciprocal after |
|---|---:|---:|---:|---:|
| SmolLM2 F+L+B | 2.8120 | 2.8396 | 2.8361 | 2.7876 |
| Whisper F+L+B | 2.8775 | 2.8615 | 3.2447 | 6.4303 |
| ResNet-50 F+L+B | 21.5986 | 21.1966 | 21.6076 | 21.1521 |

ResNet time falls 1.86% before and 2.11% after profiling. Seed-matched
division/reciprocal ratios are 1.01902/1.01705/1.02240 before and
1.02026/1.02181/1.02791 after. All six within-process ResNet drifts are below
0.43% in magnitude. This is a small, consistent local observation, not the
tuner's paired-MAD acceptance test or a cross-engine/optimizer-backed gain.
It does **not** recover most of the preceding exact-indexing regression.
The earlier 17.55 ms approximate implementation is historical context, not
a third arm measured alongside this cohort.

The short cases remain nonstationary. SmolLM2, which has no convolution, drifts
up to 28.24%. Whisper's after-block source ratios are 0.48507/0.50597/1.01437:
the first two candidate processes deteriorate substantially, not merely by a
rounding-sized difference. The largest within-process drift is 252.34%.
In reciprocal seed 2, the first eleven after-block samples are 9.75–11.12 ms,
whereas the last eight are 3.47–3.54 ms. Five predeclared settling steps did
not establish a stable regime for this short workload. Keep every sample;
these data neither establish a steady-state regression cause nor support a
no-regression/equivalence claim. [The summary](summary.json) retains all process
medians, drifts and ratios. A separate stability investigation is still due.

Instrumented ResNet median directional sums are forward 5.3772→5.1938 ms,
dX 9.3832→9.1067 ms and dW 6.5672→6.5210 ms. These localize small changes,
not additive whole-step savings. All 1,082 telemetry samples remain: utilization
0–90%, graphics 217–2902 MHz, memory clocks 405–14001 MHz, device memory used
271–710 MiB, power 7.51–71.83 W and temperature 43–57°C. These are coarse
observations, not a diagnosis of short-case transitions or a peak-memory measurement.

Before the cohort, all ordinary integration tests passed, as did all six full
convolution oracle suites, all 259 library tests including ignored GPU/state
tests, the profiling preflight, all-target/all-feature Clippy, Rust 1.92 library
check, strict docs, package verification and frozen-paper artifact replay.
The primitive GPU arithmetic test is in the ordinary suite. Native-f32
cooperative execution and other-device profitability remain unqualified here.

Retain the shared exact implementation and its modest result without new
thresholds or a growing indexing-variant menu. Tuning stays opt-in and the
frozen paper tables stay untouched. The next scheduling experiment remains
bounded split-K dW using existing SumRows, with explicit partial-buffer lifetime,
charged bytes and full-sequence timing. Better lowering of exact arithmetic and
short-case timing stability remain separate open items; neither is solved by
this cohort.
