# Design turns and the lessons they leave

This is a source-history reconstruction, not a claim about the author's
unrecorded intentions. Commit dates are author dates and can appear out of
merge order. The cited commits are in the ancestry of the audited checkout.
Performance observations from development notes remain historical unless
they were reproduced in the frozen paper matrix.

## 1. Own the training-to-deployment path

The central choice is to own a typed graph, autodiff and native runtime while
delegating GPU API machinery to Blade and shader compilation to Naga. This
keeps model and checkpoint handling in the same system from training to
deployment. Shared-context support became explicit in
[`814f78e`](https://github.com/kvark/meganeura/commit/814f78e), and later
renderer-facing work exposed GPU outputs in
[`256b906`](https://github.com/kvark/meganeura/commit/256b906).

Benefit: embeddable execution and graphics integration without a vendor ML
runtime. Cost: Meganeura must supply operator coverage, backward correctness,
kernel quality, memory planning and diagnostics that mature stacks already
have. The dependency boundary reduces owned code; it does not make driver
and compiler behavior disappear. One source implementation still needs
device-specific qualification.

## 2. From fused operators toward reusable generators

Early optimizations added concrete fused forms. Schedule-based pointwise
generation appeared in
[`3c90455`](https://github.com/kvark/meganeura/commit/3c90455); later
consolidation in
[`94cccb7`](https://github.com/kvark/meganeura/commit/94cccb7) and August
generator refactors reduced repeated entry implementations.

The important trade is between a primitive IR that exposes composition and
high-level algorithms that avoid bad materializations. Lowering attention
too early to separate matmul/softmax/matmul can hide its online algorithm;
keeping every fused model pattern as its own operator proliferates cases.
Meganeura's practical middle is tensor-level semantics plus reusable kernel
archetypes and pointwise/prologue/epilogue composition. It is incomplete:
templates and numerous entry variants remain.

Lesson: add an implementation by the tensor algebra and dataflow it supports,
then demonstrate reuse in multiple shapes. A shorter shader list is not a
win if it forces a slower algorithm or obscures legality.

## 3. Equality saturation became real, then ceased to be the default story

Traffic-aware extraction and repeated-region outlining arrived in
[`94ba086`](https://github.com/kvark/meganeura/commit/94ba086).
[`3d24850`](https://github.com/kvark/meganeura/commit/3d24850) made extracted
terms the mechanism actually producing optimized graphs, rather than merely
using an e-graph beside separate global pattern application. Subsequent
topology repair is visible in
[`a7ced10`](https://github.com/kvark/meganeura/commit/a7ced10).

The paper ablation found the same selected graph for the current rewrite
set under greedy and e-graph modes. The current default is greedy; egglog
is research infrastructure for richer competing alternatives. Outlining
remains a way to bound expensive optimization on repeated transformer blocks.

Benefit: equality saturation can explore choices without committing early.
Cost: saturation/extraction overhead, boundary handling, graph reconstruction
and an imperfect cost model. Do not claim it caused the measured speedups.
The older register-pressure model and the current traffic estimate are
engineering hypotheses, not universally valid hardware laws.

## 4. Fusion needed one owner for the complete variant

The historical RMSNorm/matmul fusion could lose cooperative acceleration;
[`88f485f`](https://github.com/kvark/meganeura/commit/88f485f) disabled a
losing path. Later work unified pipeline keys
([`7663f7c`](https://github.com/kvark/meganeura/commit/7663f7c)) and made
small/cooperative tiling modifiers instead of proliferating shader groups
([`3dce143`](https://github.com/kvark/meganeura/commit/3dce143),
[`19a1f00`](https://github.com/kvark/meganeura/commit/19a1f00)).

The invariant is stronger than "pick the same shader name": shader bindings,
workgroup dimensions, tile types, epilogues, padding and packing must all
describe the same variant. A fallback with the wrong geometry can leave rows
unwritten, not merely run slowly. Current selection is centralized, but
pipeline lookup still has a fallback hierarchy.

September audit example: horizontal packing retained a scalar fallback whose
bindings and `z=1` geometry described only one original matmul. The audit now
drops that fallback and conservatively propagates precision from all siblings.
A future tuner must build a complete packed scalar candidate to restore that
search choice safely. This is a correctness fix; no performance benefit was
measured.

## 5. Memory reuse changed the correctness boundary

Lifetime-based aliasing landed in
[`8e31f2f`](https://github.com/kvark/meganeura/commit/8e31f2f), and optional
device-local intermediates in
[`6693e62`](https://github.com/kvark/meganeura/commit/6693e62).
[`65ee3c1`](https://github.com/kvark/meganeura/commit/65ee3c1) expanded
device-local gradients/optimizer state and horizontal fusion. Metal host
access then needed explicit staging
([`3dc9621`](https://github.com/kvark/meganeura/commit/3dc9621)); constant
gradient seeds needed to remain host-visible
([`5069246`](https://github.com/kvark/meganeura/commit/5069246)).

Benefit: fewer physical allocations and less host-visible traffic on discrete
GPUs. Cost: liveness, synchronization and host accessibility become correctness
properties. Pinned state is not synonymous with mapped state. Reused padded
storage must not expose a previous tenant's data to an unguarded load.

Large-graph host-memory and allocation hardening followed on August 28
([`6761ef6`](https://github.com/kvark/meganeura/commit/6761ef6),
[`a0d822d`](https://github.com/kvark/meganeura/commit/a0d822d)). The September 5
Blade pin adds the `DeviceTransient` allocation path
([`bd6be08`](https://github.com/kvark/meganeura/commit/bd6be08)). None of
these post-freeze changes can inherit the paper's performance numbers.
The September 6 migration replaces the git pins with published Blade 0.9,
blade-macros 0.3 and Naga 30, raises the Rust minimum to 1.92, and restores
full `cargo package` verification. This closes the registry-dependency gap;
it is not a new publication or cross-backend performance result.

## 6. The precision turn: accurate forward was not enough

[`b1405a3`](https://github.com/kvark/meganeura/commit/b1405a3) repaired
cooperative training accuracy by protecting derivative operands and improved
Apple performance before the matrix freeze. An August 15 experiment then
introduced compensated f16 staging for full-precision work. On August 28,
[`3326f39`](https://github.com/kvark/meganeura/commit/3326f39) removed its
automatic use to preserve exponent range; the tiny-gradient regression is
[`f2a0108`](https://github.com/kvark/meganeura/commit/f2a0108).

Why compensation was insufficient: for a tiny `x`, both `f16(x)` and the
f16 residual can be zero. Accumulating three products in f32 cannot recover
information discarded before multiplication. Mantissa compensation is not
range preservation. Current `Auto` uses native-f32 tiles when legal or scalar
f32 for protected derivatives; raw f16 on those regions needs explicit opt-in.

Lesson: precision is part of legality for a given numerical contract.
Do not let a timing search "discover" that a wrong derivative is fast.
Loss/gradient scales and long training trajectories are not covered by
forward-only parity.

## 7. Hermetic configuration and inspectability

The August work added provenance/eager debugging
([`94cccb7`](https://github.com/kvark/meganeura/commit/94cccb7)), centralized
the environment registry
([`a4e8687`](https://github.com/kvark/meganeura/commit/a4e8687)), and moved
environment reads to explicit constructors
([`031350d`](https://github.com/kvark/meganeura/commit/031350d)).

Benefit: embedding applications get typed, reproducible behavior and can trace
generated work back to model nodes. Cost: options must travel through every
build/cache layer correctly. The September audit found rewrite settings absent
from the build-cache fingerprint and tuning skipped on cache hits; both are
now repaired. "All configuration is typed" does not itself prove that every
cache key includes it.

## 8. What should survive the next design turn

Keep semantic optimization, arithmetic permission and hardware scheduling
conceptually separate. Keep generator and runtime variant descriptions in
agreement. Keep frozen evidence immutable. Keep debug paths on the same
kernels with explicit observability costs. Prefer a few measured reusable
choices to new model-specific conditions.

Do not turn [rejected experiments](../rejected-optimizations.md) into eternal
prohibitions. Driver versions, shapes, dependency behavior and precision
contracts change. Retesting needs a reason and a controlled protocol, but an
old failure is evidence about that experiment, not a proof of impossibility.

## 9. A shared baseline is not an independent oracle

The [September 6 whole-step profiles](../experiments/training-profile-2026-09-06/README.md)
prioritize convolution derivatives. Reviewing their indexing then uncovered
a stride-1 dX bug outside same padding. Forward cross-correlation uses
`ih = oh*stride + kh - padding_h`; inversion requires
`oh = (ih + padding_h - kh)/stride`, with divisibility/bounds checks. Both
scalar and generated cooperative paths incorrectly used `kernel_h-1-padding_h`
while leaving the weight indices unflipped. Width had the same error.

Same padding satisfies `2*padding == kernel-1`, masking this substitution.
The model profiles' full-state parity therefore passed while a newly added
unpadded 3×3 f64 oracle failed. The oracle scatters forward contributions
directly, independently of the kernels' implicit-GEMM gather indexing. Fixing
the padding expression preserves bindings, geometry, precision and budgets;
it is not a new performance schedule or a justification to loosen validation.

The regression covers eight shapes, both scalar tiles, full dX/dW and tiny
upstream derivatives. A separate test requires actual generated cooperative
execution. On f16-only hardware it uses bounded, exactly representable operands
to isolate indexing, not to certify f32 derivative range. Native-f32 modules
validate offline; that is different evidence from native-f32 device execution.

Lesson: full control-session comparisons establish preservation against that
control, not absolute operator correctness. Independent oracles and unaligned,
asymmetric, tiny and otherwise uncovered cases are complementary requirements
before an automatic search can safely choose a faster implementation.

## 10. Exact arithmetic is shared machinery, not a model exception

Replacing floating-point index reciprocals with integer division fixed a
width-41 batch-boundary error, but added about 23% to a retained ResNet F+L+B
cohort. The next implementation precomputes `floor(2^32/d)` for each positive
invariant divisor greater than one, multiplies to obtain the high word, then
corrects by at most one. A short proof covers all u32 numerators, and a shared
WGSL implementation uses only 32-bit integer operations. It needs neither
special widths nor approximate-addressing fallbacks. Division by one is the
identity; zero is rejected.

This is semantic strength reduction, not a learned performance threshold or
a new runtime search space. The four derived integer uniforms and their host
binding work are real costs. Raw-u32 GPU tests exercise the same shared helper;
full f64 convolution oracles and state-preservation checks remain unchanged.
The [six-process follow-up](../experiments/conv-divisor-2026-09-06/README.md)
finds only about 2% lower ResNet step time on RTX 5070, leaving most of the
earlier regression. Two candidate Whisper after-blocks worsen substantially.
Retain those observations: exactness, backend profitability, timing stability
and whole-model generality are separate questions. A growing arithmetic-variant
menu is not yet justified. Split-K remains a distinct, explicitly budgeted
dispatch-sequence experiment using the existing SumRows reduction.
