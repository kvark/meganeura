# Qualification readback: predeclared staging experiment

Separate from all prior retained experiments and the frozen P3HPC matrix.
Both arms use published Blade 0.9.0, blade-macros 0.3.0 and Naga 30.0.1;
this is **not** a comparison of Blade versions. The [archived lockfile](Cargo.lock)
has SHA-256 `72cecddd0f090e82d5db46bc6f7f80429a058b2afa0eaec84a9b27e7340a4a80`.
Meganeura now requires Rust 1.92, matching Blade's published minimum.

## Question and controlled change

The prior crossover located 98.26–98.65% of ResNet's search cost in
qualification but could not distinguish readback from CPU validation. New
nested, accumulated timers separate input preparation, CPU upload copy,
upload encode/transfer/wait, candidate encode/dispatch/wait, readback
encode/transfer/wait, CPU allocation/copy from mapped staging, and CPU
validation. They are disjoint **host wall times within qualification**, not
GPU timestamps. Unreached fields are `None`; early exits retain partial times.
Bookkeeping/deallocation remains in the enclosing phase. Historical reports
have no breakdown, not a measured zero. Do not add nested times twice.

Compare the original single `Memory::Shared` staging buffer with a single
`Memory::Download` buffer of the same size. Both handle uploads and readbacks.
Neither is a live graph buffer or candidate kernel binding. Scratch binding
placement, capacities, scratch budget, generators, precision, ordinary/tiny
patterns, NaN poisoning, finite scans, full cross-candidate comparisons,
sampled f64 dots, search samples and selection guard remain unchanged.
The staging enum is an explicit diagnostic control, not a model-name rule.

The published [Vulkan backend's allocation flags](https://github.com/kvark/blade/blob/866de2c37acbcf1e54c3a21f3213dae4f2f45746/blade-graphics/src/vulkan/resource.rs)
request fast device access for Shared in addition to host upload/download;
Download omits the fast-device preference. The published
[Metal backend](https://github.com/kvark/blade/blob/866de2c37acbcf1e54c3a21f3213dae4f2f45746/blade-graphics/src/metal/resource.rs)
maps both to shared storage. These are source-level
policies, not proof of a particular allocation's physical location or cache
properties on this machine.

## Fixed cohort and validity

Reuse the exact three crossover builders/initialization: eight-layer 128×512
dense inference, 127-row 384→640→256 MLP+Adam with bias/GELU, and folded-BN
ResNet-50 batch-one F+L+B without an optimizer. These are diagnostic repeats,
not unseen models, pretrained weights, or a PyTorch comparison.

Six fresh processes, seeds 1..6, run serially. Each constructs two matched
sessions before either search. Session 0 uses Shared staging, session 1 uses
Download. Search first on `(seed + case_index) % 2`, so each arm runs first
three times per case. Each search starts from the same heuristic plan on a
fresh session, not a previous measured winner. No driver-cache clearing or
clock locking; balanced order reduces but cannot eliminate cache/period bias.

Use strict scalar f32, cooperative matmul and attention disabled, normal
fusion/aliasing/device-local placement. Search uses eight classes, 64 MiB,
ten-second soft deadline, one warmup and six pairs of sixteen dispatches.
After three matched live steps, compare full parameters, logical gradients,
allocated moments and output/loss. Each arm must preserve these tensors and
the expected counter bitwise across its search. Continue both normal sessions
to steps 33 and 178, compare full tensors, and validate every subsequent loss
pair. Adam settings and relative L2/elementwise tolerances are exactly the
prior crossover's. Untimed continuation is a correctness check, not a
whole-step latency experiment. Saved summaries are not full tensor dumps.

Retain both complete search reports, selected keys, memory summaries,
comparisons, trajectories, resolved settings, process/order/timestamps and
source/lockfile/executable identity. Each search measures preparation,
qualification, warmup and sampling; nested breakdown is for qualification
only. Keep every failed or noisy attempt, with no threshold/case changes.
Optional 250 ms GPU telemetry and host boundary samples retain the same
availability/coverage limitations as the crossover experiment. No concurrent
compilation, GPU qualification, heavy analysis or other benchmark.

## Decision fixed before measurement

For each case, pair the six Shared and Download process observations. Apply
the existing descriptive 5% + twice paired-difference MAD gain/regression
guard to total search, qualification and each nested cost. These are six
process pairs, not confidence intervals or fleet samples. Report all process
ratios and both search orders. Do not pool unlike workloads.

Promoting Download as the default **private tuning staging** is allowed only
if all cases/arms preserve numerical/state validity, exact class/binding
coverage and memory request counts; ResNet's total search, qualification and
CPU readback-copy costs all pass the gain guard; and neither other case
passes a total-search regression guard. The numerical gates, public tuning
default-off policy and kernel candidate/selection policy must not change.
The original Shared setting remains available, and missing historical settings
must still deserialize as Shared after any default change. This single-device
result would not establish cross-backend performance or whole-model speedup.

Commit source/protocol and build once before all six measurements:

```sh
cargo build --release --locked --example tune_readback
for seed in 1 2 3 4 5 6; do
    target/release/examples/tune_readback "new-readback-${seed}.json" "$seed"
done
```

The runner refuses existing outputs and dirty tracked source. The protocol
and measured source were committed before the cohort below; prior records
are immutable.

Before measurement, the Blade 0.9 full release test suite passed, including
247 library tests (two GPU tests ignored by default). The two GPU tests then
passed explicitly: distinct-state tuning swaps and all-bit staging round trips
over odd, rectangular and 8.192 MB outputs on both shared/device-local bindings.
All four scalar tuner regressions passed, with rectangular four-entry and
optimizer/accumulation-state checks exercising both staging choices. All six
old holdout prefixes and three crossover prefixes passed too. Native-f32 GPU
and absent external-reference-fixture coverage are not implied. CPU evidence
replay, all-feature Clippy, strict rustdoc and Rust 1.92 library checks pass.

## Retained cohort — September 6, 2026

All six fresh processes completed serially with no failed/discarded attempts
or retries. Raw records: [1](run-01.json), [2](run-02.json), [3](run-03.json),
[4](run-04.json), [5](run-05.json), [6](run-06.json).
[summary.json](summary.json) contains all cost-level paired arithmetic and
per-process total-search ratios; CPU replay recomputes it from the raw reports.

| Identity | Measured value |
|---|---|
| Source | `2abeff93b6ae5d9714a698cdc64d942ca965d2ff` |
| Immutable source tag | `evidence/readback-2026-09-06` |
| Executable SHA-256 | `306e52acccefb9ad8d346aae7370f213d953c8e91c581aa201c53395e7644ce1` |
| GPU / driver | NVIDIA GeForce RTX 5070 / 595.71.05 |
| Host / build | Linux x86_64, Intel Core i5-12400F, release Rust 1.98.0, no RUSTFLAGS |
| Matrix capability | f32 tile 0 / f16 tile 16; both cooperative paths disabled |

No build, qualification test or heavy analysis overlapped the retained cohort.
The pre-run compute-process query was empty; graphics processes remained
resident. Telemetry retained 465 samples across the six processes, with
42–62 °C, graphics clocks 217–2902 MHz and memory clocks 405–14001 MHz over
the **whole processes**, including idle/session-construction/validation intervals. There was
no clock lock or cache clearing. At 250 ms, this is neither per-pair telemetry
nor proof that external activity could never interfere.

### Total cost and the promotion decision

Costs below are medians of six per-process sums over all class comparisons,
in milliseconds. The ratio column is the median of six paired process ratios,
not the ratio of the two cost medians. Ratios greater than one favor Download.

| Case | Total search, Shared → Download | Qualification, Shared → Download | Median process search ratio |
|---|---:|---:|---:|
| Dense inference | 73.202 → 43.977 | 51.825 → 6.791 | 1.661× |
| MLP+Adam | 175.439 → 63.532 | 144.883 → 10.421 | 2.803× |
| ResNet F+L+B | 606.464 → 38.853 | 598.378 → 20.592 | 15.567× |

All three cases pass the total-search gain guard; none passes a total-search
regression guard. ResNet also passes both targeted qualification and CPU
readback-copy gain guards. All 18 case runs, 36 searches and **108 class
comparisons qualify**. Both arms search identical classes, candidates,
dispatch multiplicities and binding capacities/placements, with matching
retained tensor-buffer requests. Prefix, per-search, step-33 and step-178
full tensor/counter comparisons are bit-exact, including Adam moments and
the expected 178 updates. All 175 subsequent training loss pairs per case
pass. These are control-session comparisons, not a cross-engine oracle or
convergence result; no whole-step latency was timed.

The fixed promotion rule passes. A follow-up commit changes only the new
`TuneOptions` staging default to Download, alongside replay/tests/docs.
`TuneStaging::Shared` remains explicit, and omitted historical staging fields
still deserialize as Shared. The measured tag does not move to that later
commit. Numerical gates, candidates, live-buffer allocation policy, search
budgets and default-off tuning are unchanged.

### Where qualification spent time

ResNet's nested qualification medians, in milliseconds:

| Disjoint cost within qualification | Shared | Download |
|---|---:|---:|
| CPU input/padding/sentinel preparation | 3.017 | 3.058 |
| CPU upload copy | 3.598 | 2.852 |
| Upload encode/transfer/submit/wait | 1.124 | 3.592 |
| Candidate encode/dispatch/submit/wait | 0.562 | 0.664 |
| Readback encode/transfer/submit/wait | 0.535 | 2.331 |
| CPU mapped-memory-to-vector allocation/copy | 582.238 | 2.035 |
| CPU finite/parity scans and f64 reference checks | 6.027 | 6.006 |
| Other qualification bookkeeping/destruction | 0.006 | 0.007 |

These are not GPU timestamp intervals. Medians of components need not add
to the median total. CPU readback allocation/copy also falls from 45.280 to
0.142 ms on dense inference and from 134.656 to 0.466 ms on MLP+Adam. Validation
is still performed at full declared strength; its gain/regression guard passes
in **neither direction on any case**. The changed staging policy addresses the
measured host-copy cost without claiming a measured physical heap/cache cause.

The tradeoff is visible. Download increases preparation on every case:
dense 2.280 → 17.235 ms, MLP 1.454 → 28.142 ms, ResNet 0.648 → 10.408 ms.
Preparation includes pipeline setup, allocation and binding work; these timers
do not yet separate all of it. Upload transfer/wait passes a regression guard
on every case; ResNet's readback transfer/wait does too. Those costs are
outweighed here by CPU readback savings, not assumed to be free.

### Every process and both orders

Each cell below is that process's Shared total search divided by Download
total search. The first-search column lists dense / MLP / ResNet arms.

| Seed | First search | Dense | MLP+Adam | ResNet F+L+B |
|---|---|---:|---:|---:|
| 1 | Download / Shared / Download | 0.853× | 3.430× | 15.976× |
| 2 | Shared / Download / Shared | 1.542× | 2.784× | 16.004× |
| 3 | Download / Shared / Download | 1.781× | 3.044× | 11.678× |
| 4 | Shared / Download / Shared | 1.845× | 2.822× | 15.828× |
| 5 | Download / Shared / Download | 1.515× | 2.488× | 15.307× |
| 6 | Shared / Download / Shared | 1.819× | 2.408× | 12.116× |

For Shared-first versus Download-first subgroups (three observations each),
median ratios are dense 1.819× / 1.515×, MLP 3.044× / 2.784× and ResNet
15.828× / 15.307×. These descriptive subgroups are not extra independent
cohorts or confidence intervals.

The first dense Download search is **slower overall**, 78.824 versus 67.216 ms.
Its preparation costs 51.947 ms, including 32.647 ms pipeline setup; Shared
costs 2.075 ms, including 0.498 ms pipeline setup. Qualification still falls
from 45.847 to 6.682 ms. The first MLP Shared search also has 70.4 ms pipeline
setup. These observations stay in the cohort. Balanced order does not erase
startup effects, and no cold-cache or guaranteed one-shot latency claim is made.

### Replay and next question

CPU-only consistency and mutation tests:

```sh
cargo test --test readback_evidence retained_readback_runs_and_promotion_gates_replay -- --nocapture
cargo test --test tuning_evidence --test holdout_evidence --test crossover_evidence --test readback_evidence
cargo test --example tune_readback
```

The replay checks identities, complete case/search/phase rosters, search order,
class/binding equivalence, selected keys and guards, counters/full-tensor
comparison summaries, loss trajectories, memory requests, elapsed costs and
summary arithmetic. Mutations exercise missing data and corrupted policy,
coverage, timing, state, memory and decisions. Raw vectors are not archived;
replay cannot independently reproduce them or authenticate the producer.
Quality CI runs the replay without a GPU. The frozen paper still passes all
six verifier tests, 50 cells / 165 files, and byte-identical generated tables.

After promotion, the full release test suite and the six explicitly enabled
GPU tuning/staging regressions pass again. Full `cargo package` verification
passes; a fresh consumer builds and runs on Rust 1.92 using the assembled
package and registry dependencies, including the new/default and historical
staging settings. This is packaging/API qualification, not a new GPU cohort.

Next, split preparation's allocation/binding and pipeline work, and time final
scratch cleanup separately before testing bounded reuse. Do not reduce the validation
coverage or quietly add uncharged staging buffers. Fleet measurements remain
due, especially because Metal maps these two policies identically. Cheaper
search is not faster GPU math, a ResNet model speedup, a Blade-version win,
or grounds to enable tuning by default.
