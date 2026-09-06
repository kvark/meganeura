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
