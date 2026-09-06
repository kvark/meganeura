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
