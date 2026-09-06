# Controlled crossover: protocol and results

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

Use the identical [archived dependency resolution](../tuning-2026-09-05/Cargo.lock.gz):
SHA-256 `4a84951f05631821a4dedb57f87195f82e59dad6eaf93bca015848b8a44eb874`.
Do not use the sibling Blade checkout for retained measurements.

Before measurement, CPU gates passed (246 library tests, four tests in each
workload example), along with all-target/all-feature Clippy, strict rustdoc
and a locked Rust 1.88 library check. On the RTX 5070, the distinct-state Adam
swap regression, all three crossover numerical prefixes/no-op swaps, all six
prior holdout prefixes and all four scalar tuning regressions passed. Native
f32 cooperative geometry is covered offline, not on this f16-only device.

## Retained results

Measured source: `c789bfa95dd2e645022e1c58b5bffab128f66fac`, tagged
`evidence/crossover-2026-09-06`. It rebases onto upstream `230cab0` (Q4 SmolLM2
projections); the strict-f32 workload choices remain unchanged. The harness
and protocol were committed before measurement. Prior evidence tags and the
frozen paper were not moved or rewritten.

Linux x86_64, Intel Core i5-12400F, Rust 1.98.0, RTX 5070 / NVIDIA 595.71.05,
f32 tile 0 / f16 tile 16. All six fresh, serial processes used executable
SHA-256 `264f1fcaa8364ddbd23b19464f4396f9c21f0888b0b30a620fa026de3390d2b4`
and the archived lockfile above. Runs began at 05:20 UTC on September 6:
[run 1](run-01.json.gz), [run 2](run-02.json.gz), [run 3](run-03.json.gz),
[run 4](run-04.json.gz), [run 5](run-05.json.gz), [run 6](run-06.json.gz).
No attempts were discarded, replaced or rerun. Graphics processes remained
resident; the pre-cohort compute-process query was empty.

| Case | Whole-step ms, baseline → selected¹ | Median ratio² (process range) | Stable A/A | Confirmation, runs 1–6 | Changed dispatches per process |
|---|---:|---:|---:|---|---:|
| Dense inference | 0.25306 → 0.21527 | 1.177× (1.162–1.181×) | 6/6 | Six confirmed gains | 8 |
| MLP + Adam | 0.36255 → 0.36888 | 0.984× (0.980–0.996×) | 4/6 | Four inconclusive; two unstable controls | 2 |
| ResNet-50 F+L+B | 17.79465 → 17.79284 | 1.000× (0.999786–1.000237×) | 6/6 | Six unchanged selections | 0 |

¹ Median of six per-process pooled medians. ² Median of six within-process
ratios, not the ratio of aggregated times. Include all processes, including
unstable controls; do not aggregate different workloads. No orientation or
pooled sample passes the regression guard. This is not evidence of zero harm:
the small MLP slowdown remains visible, and absence of a guard-passing change
is not an equivalence result.

Dense inference passes the unchanged 5% + twice-MAD guard with the selected
plan on **each** side in every process. This strengthens evidence that its
scalar tile choice transfers beyond a fixed session assignment. It is still
one synthetic shape on one GPU, not a PyTorch or fleet claim. Estimated
search-only break-even is median 1,903 uses (range 1,716–2,979), calculated
within each confirmed process. It excludes experimental confirmation costs;
there is no corresponding accepted amortization gain for MLP or ResNet.

MLP's two failed A/A screens are informative: twice-MAD is 0.03076/0.03653 ms
against allowed 0.01885/0.01876 ms. Neither the difference of medians nor median
paired difference fails those screens. The record retains this noise instead
of interpreting “no significant A/A difference” as proof of a quiet control.

All 18 case runs are numerically valid, all 54 isolated comparisons qualify,
and 30 class decisions choose challengers. Full compared tensors match bitwise
at every declared stage, including immediately around search and selection
swaps. Adam reaches the expected step 178, not merely equal counters. All
720 A/A and 1,440 crossover pairs retain execution order and, for training,
finite matching losses. These are control-session comparisons, not an
independent model oracle or convergence study.

## Where search time goes

Per-process sums over classes, then medians across all six processes, in ms:

| Case | Total search | Preparation | Qualification | Warmup | Sampling | Pipeline setup (inside preparation) |
|---|---:|---:|---:|---:|---:|---:|
| Dense inference | 73.40 | 2.42 | 51.07 | 1.87 | 16.88 | 0.86 |
| MLP + Adam | 168.47 | 1.52 | 139.57 | 1.95 | 21.14 | 1.00 |
| ResNet-50 F+L+B | 546.41 | 0.63 | 538.10 | 0.49 | 7.05 | 0.52 |

Independent medians need not add up. Phase sums exclude cleanup/final
bookkeeping; pipeline setup is already inside preparation. These are host
wall times, not GPU timestamp attribution. Pipeline caches were not cleared:
for example, dense run 1 includes 32.45 ms of pipeline setup, unlike most
later processes. Do not present the median as a cold-compilation cost.

The ResNet classifier outer product is still the only eligible dispatch
(1/512). Qualification consumes 98.26–98.65% of search wall time across the
six processes. The new timers localize **this cohort's** cost; they cannot
retroactively partition the older holdout's uninstrumented 1.12-second median.
Increasing sampling counts or deleting validation is not the supported next
step. Separate upload/readback, CPU checks and GPU work within qualification.
One source-based hypothesis is the single `Memory::Shared` staging buffer:
the pinned Blade also exposes read-optimized `Memory::Download`. Investigate
placement and transfers in a separately declared experiment while preserving
scratch binding placement, full comparisons and tiny-operand checks. No
readback optimization or causal attribution is claimed here.

## Telemetry and validation limits

All six monitors completed without a reported error or sample-cap hit,
retaining 138–142 rows per process at a requested 250 ms cadence. Across the
whole processes, graphics clocks span 180–2,902 MHz and temperature 44–63°C;
these include construction, readbacks, idle gaps and timing blocks, not just
steady-state execution. Short dense/MLP blocks can fall between samples.
This does not establish constant clocks, per-pair device state or absence of
interference. It also does not identify the cause of MLP's unstable controls.
Host load/ticks/frequency samples are retained outside timers.

Before the cohort, the rebased full release `--tests` suite passed, plus both
workload examples including their GPU preflights, the distinct-state swap
regression and four scalar-tuning regressions. Default ignored tests remain
ignored; absent external-reference fixtures and disabled hub features do not
constitute cross-engine qualification. Native-f32 hardware remains unavailable.

CPU replay is shared with the prior holdouts for tensor/memory/search checks:

```sh
cargo test --test crossover_evidence retained_crossover_runs_replay -- --nocapture
cargo test --test crossover_evidence --test holdout_evidence --test tuning_evidence
```

It recomputes all controls, orientation/pooled guards, selected pipeline-change
counts, tensor-summary arithmetic, expected counters and phase accounting;
checks process identities, ordering, telemetry presence and complete pair
rosters; and rejects mutated/missing records. It cannot recompute omitted
vectors, prove the producer executed faithfully, or turn descriptive guards
into confidence intervals. The frozen paper verifier still passes all 50 cells
and 165 retained files with byte-identical generated tables.
