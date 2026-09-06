# Bounded dW accumulation attempt

## Finish line agreed with the author

This closes the current tuning-foundation milestone, not the entire research
backlog. Make one shared accumulation attempt; if it qualifies, run one separately
predeclared whole-step acceptance cohort and adopt or defer split-K. A numerical
rejection or an inconclusive performance result is a valid stopping point.
Attention schedules, persistent winners, new precision and layout search are
deferred to a later phase. Paper review and packaging do not depend on a new win.

## Accuracy experiment, declared before execution

Development control: `cd53d57`, with the earlier
[sequence cohort](../split-k-sequence-2026-09-06/README.md) left unchanged.
Try exactly one arithmetic strategy in the shared scalar convolution dW template:
accumulate each existing 16-term K tile locally, then Kahan-add its result to the
running total. Both 32/64 output tiles and split/unsplit dW use the same generator.
There is no shape cutoff, split-count exception, extra GPU buffer, precision
permission or tolerance change. Forward, dX, dense matmul, cooperative kernels
and final SumRows arithmetic are unchanged. Additional registers and arithmetic
are costs to measure, not presumed free.

Qualification includes the two previously rejected profiled shapes, the padded
rectangular fixture and the long uneven-K fixture. Test unsplit and counts
2/3/4/8, both tile sizes, full final and partial f64 oracles, ordinary and tiny
upstream inputs. Reuse the original hashed patterns and broaden with LCG and
structured cancellation inputs. Retain the old erroneous values in gate tests;
passing new output must not retroactively qualify the old records. Existing
independent full forward/dX/dW, indexing, scratch-state, alias and clipped SGD/Adam
trajectory tests also remain required. Timings from tests are not performance
evidence. Keep all qualification failures and do not tune the strategy after
seeing them. A failure defers split-K and ends this attempt.

The ignored reporting test executes all 240 rows even after a rejection. Its
test-process success means the matrix completed, not that every row qualified;
the JSON `status` and `qualified` count are the numerical decision. Source and
executable identities are recorded. Reproduce into a new output path:

```sh
MEGANEURA_ACCUMULATION_RECORD=new-accuracy.json cargo test --release --locked --test conv_derivatives report_bounded_weight_accumulation_qualification -- --ignored --nocapture --test-threads=1
```

Run from the frozen candidate tag below to reproduce the retained candidate.
The active branch keeps the reporting test but restores the pre-attempt
arithmetic, so running there tests a different source and can give other results.

Only after this gate passes may a whole-step protocol be frozen. That experiment
must include the cost of the changed unsplit arithmetic and the complete split
sequence, rebuilt plans with charged partial storage, full live-state checks,
stable controls, repeated processes, search/build cost and unchanged gain guards.
It must not compare a new split sequence only against a newly slowed control and
call that a gain over the previous development revision. Defaults and frozen
paper measurements are not silently promoted.

## Result: defer, milestone closed

The single candidate is frozen at
`evidence/compensated-dw-accuracy-2026-09-06`, source
`4142e29c4975910738b2a0b16158ac853fa04047`. The
[complete accuracy record](accuracy.json.gz) identifies executable SHA-256
`e24f57f7ed704b7453604bc4dfc953cff5688cdabc636f08a9b15273440a6ef2`,
the unchanged archived Blade 0.9 lock, RTX 5070 and driver 595.71.05.
The source was clean. All 240 declared rows completed in one invocation, with
no retries or alternate arithmetic strategies. There are no performance samples.
[Digest](SHA256SUMS).

**230/240 rows qualify; 10 reject.** All hashed and LCG rows pass, including
the original two profiled-control failures and the earlier long three-way tiny
partial. All three shorter shapes pass every input pattern. The long shape's
ordinary cancellation inputs also pass. However, its tiny cancellation inputs
fail for both scalar tiles and all five counts (unsplit, 2, 3, 4, 8). Every one
of those ten final gradients fails; split sequences also have partial relative-L2
failures. Full-array checks are attempted for every final and every partial;
an individual array check may return at its first bad element. The retained
record contains extents and failures, not the original vectors or a universal
correctness proof.

For unsplit output element 6, the candidate returns `1.9869085e-15` versus
f64 `1.9731522921401375e-15`. The error is about `1.38e-17`, beyond the
unchanged approximately `1.04e-17` elementwise bound. A subsequent CPU-only
regression reproduces these GPU bits with 16-term f32 FMA blocks and compensated
outer addition; independent f64 agrees with the retained reference. Compensating
tile sums cannot recover rounding already lost within a tile. This diagnoses
the reproduced case, not every device or possible implementation.

Per the predeclared stopping rule, no whole-step experiment follows this
rejection. The candidate's production generator/template changes are removed;
active arithmetic is exactly the pre-attempt `cd53d57` arithmetic. The candidate
remains recoverable at its immutable tag. Keep the broader reporting fixture,
CPU oracle/gate regressions and evidence replay. No tolerance, precision policy,
default tuning policy, dependency or frozen paper result changes.

The [CPU replay](../../../tests/split_k_evidence.rs) checks source/executable/lock
identity, the complete shape/pattern/scale/tile/count roster, array extents and
all 230 passes/10 rejections. Mutation tests reject dropped or promoted failures.
It checks the recorded decision, not absent full vectors; the separate CPU
arithmetic regression reproduces the one reported final element above.

Final active-tree verification passes: the ordinary release test suite, all
268 library tests including ignored state/staging tests, all 13 non-reporting
convolution tests including ignored GPU/optimizer checks, evidence replay and
mutation tests, Clippy, Rust 1.92 library check, strict rustdoc and package
verification. The six paper-verifier tests and frozen 50-cell/165-file replay
also pass with unchanged tables. These checks do not qualify native-f32 matrix
hardware absent from this device or turn the rejected candidate into a winner.

This is an explicit **defer split-K promotion** decision, not a claim that split-K
cannot work or that compensated accumulation is never useful. The earlier
synthetic speedups remain scoped to their earlier source and inputs. The
tuning-foundation engineering phase is closed. The next active work is the
[paper's reviewer-response, author review and packaging checklist](../../../paper/p3hpc/REVISION.md),
not another accumulation, attention or layout experiment.
