# Exact-class convolution derivative tiles

**Validity correction:** the small Adam/SGD case builder used 4-D operands,
violating the documented flat convolution API. Its first forward dispatch had
zero workgroups; loss, gradients and moments were all zero. The original runner
reported matching states but did not reject this vacuous workload. All six
original raw files remain unchanged and these optimizer cases are disqualified
as training/performance evidence. ResNet's nonzero workload is unaffected.
See the [separately predeclared corrected cohort](../conv-tiles-corrected-2026-09-06/README.md).
The protocol below is preserved as the original intention, not a claim that
the malformed cases fulfilled it.

## Predeclared protocol

This is a separate Meganeura-versus-Meganeura engineering cohort, not a new
paper, PyTorch, convergence or fleet comparison. Commit source and this protocol
before retaining measurements. Keep all six serial fresh-process attempts,
including skips, invalid outputs, unchanged choices and unstable controls.
Do not alter candidates, budgets, cases or gates after seeing results.

The shared `tune_crossover` runner gains an explicit `--conv-derivatives` mode;
its original mode and earlier diagnostic runners remain dense-only. New tuning
options default to `All`, but tuning itself remains opt-in. Missing historical
scope settings deserialize as `Dense`, and absent convolution shapes as None.

Candidates are the existing scalar 32/64 tiles for dX and dW. The full key adds
batch, input/output channels and spatial extents, kernel shape, stride and both
padding dimensions to direction, logical M/N/K, precision requirement, declared
binding capacities and placements. Small/large convolution shader entries are
canonicalized for grouping, then installed together with exact XYZ geometry.
No shader arithmetic, reduced-precision policy, forward convolution or
cooperative convolution is changed. Legal integer products and signed spatial
coordinates are checked. Existing f32 reciprocal decompositions must match
integer division throughout the domain: monotonicity plus both endpoints of
each quotient interval, within exactly representable integer inputs. This is
a conservative legality restriction, not a performance threshold. Excluded
shapes retain their original dispatch; this does not repair their old kernels.

Scratch holds physical NCHW operands, not materialized im2col matrices. Full
logical output scans include every dX batch. Both ordinary full-mantissa and
tiny `1e-12` upstream gradients retain full finite/elementwise cross-variant
checks and 32 f64 reference contractions, now with convolution indexing and
first/last/scattered batches. Keep the same scale-aware tolerances. Independent
full f64 scatter oracles test eight shapes, stride 1/2, non-square/even kernels,
asymmetric/non-same padding, odd channel/tile edges and batches >1, with both
initial tiles and both gradient magnitudes. Budget skips and search/swap state
isolation are checked separately. These tests do not make sampled production
qualification a proof of arbitrary kernel correctness.

Three fixed synthetic cases, in order:

1. The existing folded-BN ResNet-50, batch 1, image 224², 1,000 classes, F+L+B,
   no optimizer. This is a diagnostic repeat of the profiled workload.
2. Two convolution layers with Adam: NCHW input `[2,5,17,19]`, channels 17/33,
   kernels 3×2 / 2×3, strides 1/2, padding (2,0)/(0,1), no activation,
   mean-squared output loss. The second layer's dX and both dW are searched.
3. The same two-layer graph with SGD.

Use shared deterministic fan-in-scaled initialization and fixed inputs. Adam:
LR 1e-4, beta .9/.999, epsilon 1e-8; SGD LR 1e-3; both clip norm 1 every step,
no accumulation or decay. Normal greedy fusion, allocation aliasing and
device-local placement. Strict f32; cooperative attention disabled; on f16-only
matrix hardware the session's cooperative policy is Disabled.

The search is `ConvDerivatives` only: eight-class, 64 MiB scratch (including
one exact-size Download staging slot), ten-second soft wall deadline, one warmup,
six alternating pairs of sixteen barrier-delimited dispatches. Priority is
repetition×M×N×K×dispatch-batches (dW's K already includes batch). Preserve the
5% improvement and twice-paired-MAD decision guard. No class/model overrides.

Reuse the [crossover protocol](../crossover-2026-09-06/README.md): three prefix
steps, thirty warmup, five settling, forty alternating untuned A/A pairs. Tune
one session at age 78. Four blocks of twenty pairs, with five settling steps
each, selected plan on left/right/right/left or its mirror. Six process seeds
1..6 balance the starting side by `(seed + case_index) % 2`. Both sessions keep
their own allocations/history. Exact full declared state before/after search
and swaps; full parameters, logical gradients, allocated moments and loss at
prefix/warmup/control/block endpoints; every timed loss pair is checked outside
timers. Adam must reach step 178. Cross-session tolerance stays relative L2
≤2e-4 and elementwise `1e-6 + 2e-4*abs(reference)`. Archive complete before/after
dispatches and declared buffers to replay which legal choices changed.

Whole-step timing is normal `step+wait`, including optimizer/clip when present;
search, swaps, readbacks and settling are excluded. A/A stability requires
absolute difference of medians, absolute paired median difference and twice
paired MAD each ≤5% of the left median. Confirm only a numerically valid,
changed plan with stable A/A and 5%+twice-MAD gains in both orientations and
pooled samples. Report all decisions and process ratios; no reruns to quiet
noise, pooled cross-workload speedup, confidence-interval or equivalence claim.
Report search cost and amortization only for confirmed gains.

Continuous 250 ms bounded NVIDIA telemetry and host boundary samples remain
active. No other benchmark, qualification, build or heavy host analysis runs
concurrently. Clocks/caches are not locked/reset; graphics processes remain
resident. Record clean tracked source, revision, executable/lock hashes, runtime
and compile settings. Incremental create-new raw files retain failures. Full
vectors are checked in memory, only summaries archived; replay verifies their
consistency, not missing vectors.

Use the [Blade 0.9 archived lock](../readback-2026-09-06/Cargo.lock), SHA-256
`72cecddd0f090e82d5db46bc6f7f80429a058b2afa0eaec84a9b27e7340a4a80`.
Do not use the sibling Blade checkout.

```sh
cargo build --release --locked --example tune_crossover
for seed in 1 2 3 4 5 6; do
    target/release/examples/tune_crossover "new-conv-${seed}.json" "$seed" --conv-derivatives
done
```

Regardless of outcome, split-K dW remains a separate next implementation and
cohort: charge partial storage and reduction dispatches, retain full-precision
validation and whole-step confirmation. Tile selection alone cannot create
more parallel work than these existing algorithms expose.
