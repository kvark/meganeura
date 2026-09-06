# Bounded split-K weight-gradient plan prototype

## Scope

The [preceding profiles and tile search](../conv-tiles-corrected-2026-09-06/README.md)
motivate more parallelism for long-reduction, small-output convolution weight
gradients. This change supplies a legal two-pass plan to experiment with. It
does **not** automatically search, select or install split-K, and has no retained
performance measurements. The compiler's ordinary choices, live tile tuner's
fixed-allocation contract and frozen paper results are unchanged.

The implementation uses the existing scalar 32/64 dW template, exact integer
indexing, SumRows reduction, scheduler and memory planner. There is no new
dependency, model-name rule, float atomic, precision mode or tuning framework.
The public experimental entry point is
[`ExecutionPlan::split_conv_weight_gradients`](../../../src/compile/split_k.rs).
Call it on a plan **before** constructing a session, with explicit
`(current_dispatch_index, splits)` selections and a partial-byte cap. It returns
the charged bytes or an error, not a qualified winner.

## Execution and memory contract

For dW, `M = Co`, `N = Ci*Kh*Kw`, `K = batch*Oh*Ow`. A split count `S` produces:

```text
upstream gradient + input
  → scalar dW partials [S, M, N]
  → dependency barrier
  → existing SumRows [S, M*N]
  → original logical weight-gradient buffer → existing consumers / optimizer
```

Partition `ceil(K/16)` complete reduction tiles into contiguous ranges, balanced
to within one tile. Only the last tile can be padded. The producer's Z workgroup
coordinate selects the partition; `num_workgroups.z` supplies `S`, so the 64-byte
convolution uniform and original input bindings do not grow. Within a partition
the existing gather, f32 FMA and output-tile code is shared. The unsplit generated
shader contains no runtime partition logic. SumRows uses its existing f32 row
lanes and fixed tree, without atomics. The overall accumulation order changes.

Preflight all selections before modifying the plan. Reuse the tile tuner's
exact convolution geometry, binding-capacity and modifier checks. Additionally:

- Reject duplicates, missing dispatches, aliased logical inputs/output and
  non-scalar or already-split entries.
- Require `2 <= S <= min(65535, ceil(K/16))`, so every partition is nonempty.
- Check `M*N*S` against u32 shader indexing and its f32 byte size against usize.
- Require `ceil(M*N/32) <= 65535` for the existing SumRows dispatch geometry.
- Charge the sum of **new logical partial capacities** `4*M*N*S` across this
  call's selections, before any alias reuse. Reject the whole call if it exceeds
  the caller's cap; no earlier valid selection is left installed on error.

The cap is conservative across nonoverlapping partial lifetimes. It is not a
whole-session memory limit, isolated-tuning scratch cap or peak-VRAM estimate.
The ordinary session preflight still checks the actual planned allocations;
search staging and simultaneously resident comparison sessions would need their
own complete accounting. The scheduler sees ordinary producer/consumer edges,
and the existing memory planner can reuse partial storage after its reduction.
A CPU regression constructs two dependent sequences: they occupy four barrier
groups and share one physical partial allocation instead of two.

The reduction writes the original logical gradient. Parameter/gradient pairs,
node-buffer mappings, graph outputs and optimizer roles remain unchanged.
Both dispatches retain the original origin and precision/fusion-barrier metadata.
Their labels distinguish `split-K partials (S)` and `split-K reduction`. The
partial producer profiles as convolution; SumRows still profiles as a generic
reduction. Summing only convolution-family timings would omit part of this
candidate's cost.

## Numerical qualification, including a rejection

The [full convolution tests](../../../tests/conv_derivatives.rs) use the existing
independent f64 forward-scatter oracle, not the GPU implicit-GEMM gather formula.
They check every forward output, dX and final dW element. For split-K they also
mask the upstream gradient to each partition's reduction range and check **every
partial** against that same independent oracle. Ordinary random f32 inputs use
all mantissa bits; a `1e-12` upstream-gradient pass follows the ordinary pass to
expose lost tiny signals and stale partial storage.

Keep the original finite checks, elementwise bound
`abs(error) <= scale*1e-5 + abs(reference)*2e-4`, and relative-L2 bound `2e-4`
with nonzero reference norm and no norm floor. Full control-session parity alone
does not replace these checks. Nor does a correct final sum erase a failed
individual-partial check.

On September 6, RTX 5070 / driver 595.71.05, both output tiles pass four
rectangular/batched/padded/strided fixtures, with split counts 2, 3, 7 and 16
where legal. Their `(batch,Ci,H,W,Co,Kh,Kw,stride,padH,padW)` tuples are:

```text
(3,  5, 7,  9,  7, 2, 3, 1, 1, 0)
(2, 17, 9, 11, 19, 3, 3, 2, 0, 1)
(2, 65, 5, 13, 33, 2, 2, 1, 1, 0)
(2,  3, 1, 41,  5, 1, 1, 1, 0, 0)
```

A longer fixture `(2,3,1,32771,5,1,1,1,0,0)`, `K=65542`, exposes a limit.
The unsplit control and all tested final split gradients pass. Counts 2, 7 and
16 also pass every partial. Count 3 fails the tiny-gradient partial check on
both tiles: partial 1 (zero-based), element 13, is `-8.967522e-14` versus f64
reference `-8.963817608922598e-14`. Its approximately `3.70e-17` error exceeds
the approximately `2.79e-17` elementwise bound. Retain this as an unqualified
configuration on this device, not a universal ban on three-way splits. The
test reports all partition-count rejections, and a separate fixed-value CPU
regression proves this observed discrepancy still fails the unchanged gate.
No tolerance or model-specific exception was added to admit it.

Three-way splitting on the first, shorter fixture also passes four actual SGD
and four Adam updates, both with and without aliasing, against an unsplit
control. These checks include clipping, changing upstream inputs, every
parameter/gradient, both Adam moments, loss and its partials, update counter and
moment bytes.
Both parameters actually change; SGD never allocates Adam state. These are
short control trajectories plus independent operator oracles, not long-run
convergence or independent optimizer qualification.

The split entries stay outside `TuneClass` collection. Zero-budget live tuning
still sees the remaining unsplit dX class and leaves state/keys unchanged.
Swapping a split and unsplit session is rejected without changing keys, tensors
or optimizer age. A future sequence search must not bypass this layout guard.

## Verification and remaining work

Use the archived [Blade 0.9 lock](../readback-2026-09-06/Cargo.lock), SHA-256
`72cecddd0f090e82d5db46bc6f7f80429a058b2afa0eaec84a9b27e7340a4a80`.
The local GPU qualification uses Rust 1.98.0; Rust 1.92 remains the MSRV.
Run the full convolution suite, including the existing cooperative and live
tile-tuning tests, with:

```sh
cargo test --release --locked --test conv_derivatives -- --include-ignored --test-threads=1 --nocapture
```

All ten tests pass locally; the long case's reported three-way rejection is
intentional, not a passing accuracy result for that configuration. CPU checks
cover atomic plan failure, integer/dispatch bounds, partition coverage, shader
generation/SPIR-V/bindings, origin preservation, serialization and partial reuse.
Other backends and native-f32 cooperative hardware are not newly qualified here.

The ordinary release unit/integration suite also passes, as do all 266 library
tests with ignored GPU/state checks enabled, all-target/all-feature Clippy,
the Rust 1.92 library check, strict rustdoc and package verification. Frozen-paper
replay still passes six verifier tests and all 50 cells / 165 files with unchanged
tables. Existing ignored/data-dependent coverage limits remain; this is not a
new external-engine or fleet qualification.

Next, use this plan shape in bounded **isolated sequence** qualification and
measurement, reusing exact class keys and the existing decision guard. Explicitly
charge inputs, final output, all partials, upload/readback staging and simultaneous
lifetimes. Time both dispatches and their dependency, not just the more-parallel
producer. Reject numerically unqualified choices before timing; measure search
cost as well as kernel cost. Only then compare normal whole-step sessions built
from selected plans, with optimizer-state checks and a predeclared, source-frozen
protocol retaining every attempt. Whole-step confirmation needs distinct layouts;
the current geometry-only live swap is not that mechanism. No speedup, automatic
adoption or profitable split count is claimed by this prototype.

The [subsequent isolated sequence cohort](../split-k-sequence-2026-09-06/README.md)
now measures both passes with complete scratch accounting. It finds a synthetic
long-reduction gain but rejects both profiled large shapes before timing; full
scans expose unsplit-control accuracy failures. Read it as a separate source and
qualification result, not a retrospective speed claim for this initial prototype.
