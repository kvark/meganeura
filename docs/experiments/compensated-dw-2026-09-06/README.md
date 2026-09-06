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

Only after this gate passes may a whole-step protocol be frozen. That experiment
must include the cost of the changed unsplit arithmetic and the complete split
sequence, rebuilt plans with charged partial storage, full live-state checks,
stable controls, repeated processes, search/build cost and unchanged gain guards.
It must not compare a new split sequence only against a newly slowed control and
call that a gain over the previous development revision. Defaults and frozen
paper measurements are not silently promoted.
