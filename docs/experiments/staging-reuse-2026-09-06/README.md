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

## Profile result and chosen scope

The single localization process completed with all comparisons qualified and
bit-exact state checks through Adam step 178. [profile-01.json](profile-01.json)
retains source `42fda56`, tag `evidence/allocation-profile-2026-09-06`, executable
SHA-256 `e1f4a4c36c314195173a3448ca3fb416616f594056006a2c26b49a98bdfd82cd`.
The unchanged [registry lockfile](../readback-2026-09-06/Cargo.lock) has SHA-256
`72cecddd0f090e82d5db46bc6f7f80429a058b2afa0eaec84a9b27e7340a4a80`.
These are one process's Download costs, not medians or a promotion result:

| Case | Preparation | Staging allocation | Binding allocations | Encoder creation | Cleanup |
|---|---:|---:|---:|---:|---:|
| Dense inference | 20.787 ms | 19.596 ms | 0.009 ms | 0.027 ms | 2.095 ms |
| MLP+Adam | 33.979 ms | 32.054 ms | 0.017 ms | 0.042 ms | 3.478 ms |
| ResNet F+L+B | 9.986 ms | 9.317 ms | 0.003 ms | 0.009 ms | 1.351 ms |

Staging allocation dominates preparation. Test only a **one-slot exact-size
staging cache within `tune_with`**. Keep candidate binding buffers and command
encoders fresh. Different requested sizes release the previous buffer before
any new scratch allocation; smaller requests do not keep excess capacity.
Only storage is reused: all uploads, poisoning, readbacks and validation still
run. No staging remains in the session or after a bounded/failed search returns.

## Predeclared reuse comparison

Commit the implementation and this decision before measuring. Use six fresh
serial processes, seeds 1..6, with the same three diagnostic builders and full
prefix/search/step-33/step-178 comparisons as the readback protocol. Both arms
use Download, strict scalar f32, normal fusion/aliasing/device placement,
eight classes, 64 MiB scratch, ten-second soft deadline, one warmup and six
pairs of sixteen dispatches. Session 0 uses Fresh, session 1 SameSize. Alternate
which searches first using `(seed + case_index) % 2`; each arm starts first
three times per case. Each starts with the same heuristic plan on a new session.
Every later loss pair is checked. Continuation is untimed.

Every report retains per-comparison binding/staging requests, reuse flags,
allocation/release counts, peak simultaneous scratch requests and zero staging
bytes retained at return. Device-budget checks charge only new allocations
because retained staging is already resident; the user scratch cap charges
the complete simultaneous binding-plus-staging request. Exact-size reuse must
keep per-comparison requests and cohort peak requests identical between arms.
Memory summaries before/after search must still match; they are not peak VRAM.
Expected staging allocation counts are dense 3 → 1, MLP 5 → 2, ResNet 1 → 1.
The single-comparison ResNet case is the no-reuse control, not a promised win.

Preparation's staging timer includes any previous-size release. Per-comparison
cleanup destroys bindings/encoder and Fresh staging. SameSize's last staging
release is recorded as `TuneReport::final_cleanup`, inside total search time.
Report `cleanup` as the sum of comparison and final cleanup, and
`staging_and_cleanup` as preparation's staging management plus **all** scratch
cleanup, not a falsely isolated staging-only destruction cost. Do not move
allocation or final release outside the end-to-end search cost.

Apply the existing descriptive 5% + twice paired-difference MAD guards to the
six process pairs, reporting all cost medians/ratios and both search orders.
Allow SameSize as the new option default only if all numerical/state/coverage
and memory/count invariants pass, dense and MLP each pass gain guards for both
total search and `staging_and_cleanup`, and ResNet passes no total-search
regression guard. Keep every failed/noisy attempt, with no new cases, exclusions
or threshold changes after measurement. This is one-device search-cost evidence,
not confidence intervals, fleet behavior, model speedup or a Blade comparison.
Fresh remains available and missing historical settings retain Fresh. Tuning
itself stays default-off regardless of the outcome.

```sh
cargo build --release --locked --example tune_staging_reuse
for seed in 1 2 3 4 5 6; do
    target/release/examples/tune_staging_reuse "new-reuse-${seed}.json" "$seed"
done
```

The runner shares the matched diagnostic/state-check code with `tune_readback`;
its explicit options keep that older experiment on Fresh even if defaults change.
It refuses dirty tracked source or existing outputs, saves completed cases
incrementally and retains optional 250 ms GPU telemetry/host boundaries. Keep
the GPU free of concurrent tests, builds, other benchmarks and heavy analysis.
Telemetry cannot establish constant clocks or resolve every short pair.
