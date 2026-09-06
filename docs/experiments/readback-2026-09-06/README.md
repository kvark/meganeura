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

The published Vulkan backend's Shared allocation requests fast device access
in addition to host upload/download; Download omits the fast-device preference.
The published Metal backend maps both to shared storage. These are source-level
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

The runner refuses existing outputs and dirty tracked source. Results and
the measured-source tag will be added afterward; prior records are immutable.

Before measurement, the Blade 0.9 full release test suite passed, including
247 library tests (two GPU tests ignored by default). The two GPU tests then
passed explicitly: distinct-state tuning swaps and all-bit staging round trips
over odd, rectangular and 8.192 MB outputs on both shared/device-local bindings.
All four scalar tuner regressions passed, with rectangular four-entry and
optimizer/accumulation-state checks exercising both staging choices. All six
old holdout prefixes and three crossover prefixes passed too. Native-f32 GPU
and absent external-reference-fixture coverage are not implied. CPU evidence
replay, all-feature Clippy, strict rustdoc and Rust 1.92 library checks pass.
