# Controlled crossover: predeclared protocol

Separate from the frozen paper, September 5 pilot and September 6 holdouts.
This is a Meganeura-versus-Meganeura confirmation experiment, not an engine
comparison. Do not change the tuner, generators, guard or cases after seeing
the results. Keep every attempt, including unstable controls and no changes.

Three fixed synthetic cases: the pilot's promising 128×512, eight-layer dense
inference shape (fan-in-scaled initialization here); the same MLP+Adam as the
six-case holdouts; and the same folded-BN ResNet-50 F+L+B control, no optimizer.
Workloads, initialization and full-state comparison code are shared in Rust.
These are repeat/diagnostic workloads, not new unseen holdouts.

Use strict f32 with cooperative attention disabled and f16-only matrix hardware
disabled. Keep normal fusion, aliasing and device-local placement. Adam uses
LR 1e-4, beta .9/.999, epsilon 1e-8, clipping 1.0 every step, no accumulation
or decay. Inputs remain fixed. The model-quality/independent-oracle and memory
accounting limitations of the prior holdouts still apply.

## Matched state and role reversal

Two sessions retain their original tensor allocations and evolving training
state throughout. Run three prefix steps and compare full parameters, logical
gradients, allocated moments and output/loss. Warm for thirty steps, compare,
perform a no-change selection swap, settle for five steps, then record forty
alternating left/right A/A pairs. Both sessions are untuned here (age 78).

Tune exactly one session using the existing eight-class, 64 MiB, ten-second
soft budget. Require bitwise preservation of its declared state. Record the
new phase timers without inferring a phase from old experiment costs.

Measure four blocks of twenty pairs, with five settling steps before each.
The selected plan resides on left/right/right/left, or its mirror. A checked
selection swap changes only legal f32 tile choices; tensors, optimizer history
and allocations stay put. Require exact before/after-swap state and complete
cross-session comparisons after every block. Adam reaches steps 103, 128, 153
and 178. Read and validate every loss pair outside both timers. Cross-session
tolerance remains relative L2 ≤2e-4 and elementwise `1e-6 + 2e-4*abs(reference)`.
No checkpoint resets or training-state transplantation occur.

Six fresh processes, seeds 1..6, run serially. Initial winner side is
`(seed + case_index) % 2`, so each case starts on either side three times.
Execution order also alternates and is recorded independently of winner side.
This is deterministic counterbalancing, not a claim of random assignment.
The symmetric four-block sequence reduces linear period bias; it cannot
eliminate nonlinear training-age/temperature effects or carryover. A/A is
measured before search only, not after every crossover block.

## Acceptance and telemetry

An A/A control is considered stable only if all three are ≤5% of its left
median: absolute difference of medians, absolute median paired difference,
and twice the paired-difference MAD. This is a descriptive stability screen,
not an equivalence test or confidence interval. Keep unstable controls.

A confirmed gain requires numerical validity, changed selections, a stable
A/A control, and the existing 5% + twice-MAD gain guard in **both** winner-side
orientations and the pooled pairs. Apply the analogous regression guard too.
Otherwise report unchanged selection, unstable control or inconclusive. Do
not average unlike workloads or enable tuning by default from this cohort.
No automatic runtime confirmation/rollback or persistent winner is added.

Optional NVIDIA telemetry samples clocks, power, temperature, memory and GPU
utilization every 250 ms in a separate process. Its overhead is present for
both sessions. Record the raw stream (bounded to 40,000 samples), availability
and timestamps, plus host CPU ticks/load/frequency at block boundaries outside
timers. Missing telemetry is missing evidence, not proof of no interference.
Clocks are not locked, caches not cleared, and graphics processes remain
resident. No other benchmark, qualification or compilation runs concurrently.

Reserve new output paths, require clean tracked source, commit the protocol
before measurement, record source/lockfile/executable identity, and retain
all completed cases incrementally. Full vectors are compared in memory but
only summaries are archived. CPU replay checks consistency, not omitted vectors.

```sh
cargo build --release --locked --example tune_crossover
for seed in 1 2 3 4 5 6; do
    target/release/examples/tune_crossover "new-crossover-${seed}.json" "$seed"
done
```

Use the identical [archived dependency resolution](../tuning-2026-09-05/Cargo.lock):
SHA-256 `4a84951f05631821a4dedb57f87195f82e59dad6eaf93bca015848b8a44eb874`.
Do not use the sibling Blade checkout for retained measurements.

Before measurement, CPU gates passed (246 library tests, four tests in each
workload example), along with all-target/all-feature Clippy, strict rustdoc
and a locked Rust 1.88 library check. On the RTX 5070, the distinct-state Adam
swap regression, all three crossover numerical prefixes/no-op swaps, all six
prior holdout prefixes and all four scalar tuning regressions passed. Native
f32 cooperative geometry is covered offline, not on this f16-only device.

Results and the measured-source tag will be added after measurement.
