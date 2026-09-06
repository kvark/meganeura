# Whole-step tuning holdouts: inference and optimizer-backed training

This is a Meganeura-versus-Meganeura engineering experiment, separate from the
frozen P3HPC matrix. It tests the existing search without changing its candidate
space, selection thresholds, or generators in response to the outcomes.
Weights and inputs are synthetic and deterministic, not pretrained checkpoints
or a model-quality dataset. “Holdout” means new workloads/shapes relative to
the September 5 dense-chain pilot, not an unseen-device generalization study.

## Fixed cases

| Case | Graph and shape | Timed work |
|---|---|---|
| `mlp-inference` | 127 rows, 384 → 640 → 256, bias + GELU | Forward |
| `mlp-adam` | Same MLP, synthetic one-hot labels | Forward + cross entropy + backward + clipping + Adam |
| `smollm2-inference` | Existing medium-test builder: 8 layers, width 128, FFN 256, 2 heads, vocabulary 64, sequence 127 | Forward logits; no KV cache |
| `smollm2-adam` | Same decoder graph and synthetic next-token labels | Forward + cross entropy + backward + clipping + Adam |
| `whisper-sgd` | Existing Whisper-tiny encoder: batch 1, 100 mel frames, width 384, 4 layers | Forward + mean squared encoder output + backward + clipping + SGD |
| `resnet50-flb` | Existing folded-BN ResNet-50 builder, batch 1, 224² image | Forward + cross entropy + backward, **no optimizer** |

The convolution-heavy case is retained even if the matmul-only tuner has no
useful candidates. No case may be removed because it shows no gain, hits a
search budget, or regresses. Different training boundaries are labeled, not
aggregated into a universal “training speedup.”

## Numerical and state contract

F32 storage/operands throughout; native-f32 matmul is allowed only when
advertised. Disable cooperative attention explicitly; on the available f16-
tile-only RTX, disable cooperative matmul too. Leave ordinary graph rewrites,
fusion and allocation enabled. The search remains opt-in, at most 8 classes,
64 MiB scratch, six paired samples × 16 dispatches and a 10-second soft deadline.

Initialize parameters deterministically, using graph roles for normalization
scales, biases and convolution fan-in; initialize original graph parameters
so the runtime builds consistent derived weights. No model-name kernel rules.
Adam uses LR 1e-4, beta1 0.9, beta2 0.999, epsilon 1e-8. SGD uses LR 1e-3.
Both use global gradient clipping at 1.0, every update. No decay, accumulation,
dropout, data-loader/RNG advancement, or external/KV state.

Two matched sessions receive the same initialization and static inputs. Run
three identical prefix steps before tuning. Compare all parameter and gradient
elements, allocated moment elements, output/loss and Adam counters. Snapshot
the tuned session immediately before/after search and require bitwise equality
of these tensors/counter, including finite values. This is a declared probe
set, not a claim that every hidden runtime object is snapshotted.

Cross-session checks run after the prefix, after warmup and after timed steps.
Each tensor must be finite, have relative L2 error ≤ 2e-4 and satisfy
`abs(error) ≤ 1e-6 + 2e-4 * abs(reference)` elementwise. Tiny gradients are not
accepted merely because their absolute difference is small. Both sessions
must have the expected optimizer-update counts, not just equal counters:
Adam steps 3 after the prefix/search, 33 after warmup, and 78 at the end; other
cases keep the Adam counter at zero. Untuned Meganeura is the control,
not an independent model oracle; the tuner's sampled f64 dots are separate.

## Timing and records

Thirty warmups per session, a full state comparison, five settling steps
after those validation readbacks, then forty alternating AB/BA whole-step pairs.
Each pair advances both sessions once, preserving matched training age. Do
not restore weights between timed steps: this measures an evolving, fixed-
input training trajectory. Timing includes normal encoding, submission, wait,
backward and the stated optimizer/clip work. Exclude construction, uploads,
search, validation readbacks and logging. Read training loss after each pair,
outside both timers, and retain the trajectory. Each loss pair must also
satisfy the finite-value and numerical contract; a later recovery cannot hide
a nonfinite or divergent timed loss.

Run five fresh processes serially on an available GPU. Retain every attempt,
including errors. Record source revision, clean tracked status, executable and
lockfile hashes, device/driver/capabilities, pipeline keys, search reports,
all forty raw timing pairs, loss trajectories and complete tensor-comparison
summaries. Full tensor values are compared in memory but are not archived;
CPU replay verifies summary arithmetic, not the omitted full vectors.

Use the pilot's descriptive 5% + twice paired-difference MAD guard for gains;
report regressions using the same baseline-relative threshold. Neither is a
confidence interval. Report every process and per-case medians/ranges; no
discarding slow processes. Amortization uses search time / whole-step time
saved, and is not meaningful where there is no demonstrated gain.

Session memory records count retained tensor-buffer requests. API-level
usage/budget samples describe the process with **both** matched sessions
resident; these are stage samples, not peak VRAM or per-session driver usage.
Clocks are not locked and shader caches are not cleared; process repetition
does not imply independent fleet samples. Frozen paper files remain untouched.

## Reproduction

Use the measured revision and its recorded dependency resolution on an idle
GPU. The runner refuses a dirty tracked checkout or an existing output path.
The dependency resolution is identical to the retained
[September 5 lockfile](../tuning-2026-09-05/Cargo.lock.gz), SHA-256
`4a84951f05631821a4dedb57f87195f82e59dad6eaf93bca015848b8a44eb874`.
Use it as the root `Cargo.lock` when rebuilding the measured revision.
Each process rebuilds both sessions and reruns search for every case. It writes
completed case records incrementally; a failed case remains in the document
and causes a nonzero exit after the other cases have run.

```sh
cargo test --example tune_holdouts
cargo test --release --example tune_holdouts holdout_prefixes -- --ignored --nocapture --test-threads=1
cargo build --release --locked --example tune_holdouts
target/release/examples/tune_holdouts --list
for run in 01 02 03 04 05; do
    target/release/examples/tune_holdouts "new-holdout-${run}.json"
done
```

Do not run qualification, compilation, another benchmark, or heavy host work
concurrently with measurements. Source/configuration changes require a new
revision and an explicitly separate experiment, not replacement of a slow or
failed record. The ignored GPU preflight checks the three-step prefix only;
it does not measure search or steady-state performance.

## Results — September 6, 2026

Measured source: `8bcd6b924e6dbb2d40b825171f89e8d19785d3d8`, retained by
`evidence/holdouts-2026-09-06`. The protocol and harness were committed before
measurement. This source rebases the development work onto `a455409` (including
upstream Q4 fixes); it does not relabel the older pilot or frozen paper source.
Rust 1.98.0, Linux x86_64, Intel Core i5-12400F, RTX 5070 / NVIDIA 595.71.05,
f32 tile 0 / f16 tile 16. All five processes used the same executable hash
`7ada9ad886d956fe6da80818d95d03bbe8b75833632c35237bc3fbccefa0db57`
and the lockfile above. Raw records:
[run 1](run-01.json.gz), [run 2](run-02.json.gz), [run 3](run-03.json.gz),
[run 4](run-04.json.gz), [run 5](run-05.json.gz). No attempts were discarded or rerun.

**No case/process cleared the predeclared whole-step improvement or regression
guard.** All 30 completed successfully, all 140 scalar comparisons qualified,
and 31 class decisions accepted a challenger. Isolated selection does not by
itself establish an end-to-end win. No default, threshold or generator changed.

| Case | Whole-step ms, baseline → tuned¹ | Median ratio² (process range) | Gain / regression guards | Changed dispatches, runs 1–5 | Median search cost |
|---|---:|---:|---:|---|---:|
| MLP inference | 0.08292 → 0.07882 | 1.052× (1.012–1.053×) | 0/5, 0/5 | 1, 1, 1, 0, 1 | 40.5 ms |
| MLP + Adam | 0.36663 → 0.37357 | 0.977× (0.969–1.001×) | 0/5, 0/5 | 0, 2, 2, 1, 2 | 172.6 ms |
| SmolLM2 inference | 0.72905 → 0.70737 | 1.026× (1.001–1.044×) | 0/5, 0/5 | 8, 8, 8, 0, 8 | 42.6 ms |
| SmolLM2 + Adam | 5.03869 → 5.00575 | 1.010× (1.006–1.016×) | 0/5, 0/5 | 16, 16, 16, 8, 16 | 86.4 ms |
| Whisper encoder + SGD | 9.28370 → 9.30853 | 0.997× (0.992–1.005×) | 0/5, 0/5 | 8, 0, 8, 4, 8 | 495.4 ms |
| ResNet-50 F+L+B | 17.75295 → 17.76358 | 1.000× (0.999–1.071×) | 0/5, 0/5 | 0, 0, 0, 0, 0 | 1,117.6 ms |

¹ Median of five per-process medians. ² Median of five within-process ratios,
not the ratio of the aggregated times. Do not pool the different training
boundaries. Since no gain passed the guard, no amortization win is asserted.
The small negative ratios are retained, not described as demonstrated safety
or erased because they are below the regression threshold.

All compared vectors matched **bitwise** in this cohort: parameters (including
loadable source/derived weights), logical gradients, allocated Adam moments,
outputs and complete loss partials. Search preserved the declared tensors and
counter exactly. Both Adam cases reached update 78; the other cases retained
counter zero. All 800 timed training-loss pairs were finite and equal. The
full gradient comparisons cover 410,496 MLP, 1,321,088 SmolLM2, 7,632,384 Whisper
and 25,530,472 ResNet elements per session at each validation stage. This is
control-session parity, **not independent gradient correctness or convergence**.
Static synthetic labels are learned by the Adam cases; that is not language
model quality. Folded BN and no optimizer remain part of the ResNet boundary.

### Coverage, cost and measurement limits

| Case | Visited / eligible classes | Eligible / total plan dispatches |
|---|---:|---:|
| MLP inference | 2 / 2 | 2 / 5 |
| MLP + Adam | 5 / 5 | 5 / 18 |
| SmolLM2 inference | 4 / 4 | 25 / 83 |
| SmolLM2 + Adam | 8 / 11 | 90 / 284 |
| Whisper + SGD | 8 / 9 | 40 / 227 |
| ResNet F+L+B | 1 / 1 | 1 / 512 |

The eight-class cap binds for SmolLM2 Adam and Whisper SGD in all processes.
No time/scratch/device-memory/qualification skip occurred. These are dispatch
counts, not fractions of runtime; optimizer work is outside the plan count.
The sole ResNet candidate is the classifier's `MatMulAT(2048,1000,1)` weight
gradient, not a convolution. Search costs 1.02–3.41 seconds there and never
changes the tile. Most of that cost is outside the retained timed kernel
batches and the sub-millisecond pipeline-setup timer; the present report does
not separately localize the remaining qualification/upload/readback/warmup
cost. A structural M×N×K prior alone does not estimate search amortization.

The GPU reported 0% utilization and 271 MiB used before the experiment; no
other tests, compilation or benchmarks were launched alongside it. Graphics
processes remained resident. A lightweight spot check during the fifth process
found no other compute application; that is not continuous interference or
clock monitoring. The runner records `nvidia-smi` before/after each process.
Temperature rose from 43°C to 52°C. Clocks were not locked, so these data do
not isolate the cause of timing drift.

In particular, run 3's unchanged ResNet control has a **1.071× ratio of
medians**, yet its median paired gain is only **0.01470 ms**, with a 0.09023 ms
noise margin. The pairwise gate rejects it. Run 4's MLP inference baseline
median is 0.14008 ms versus about 0.082–0.083 ms in the others, and its search
keeps the incumbent. Both stay in the table. Neither discarding run 4 nor
crediting run 3's unchanged choices would be defensible.

Memory records confirm zero moment/accumulator allocation for inference,
Whisper SGD and ResNet F+L+B; Adam uses 3,283,968 bytes for MLP moments and
10,568,704 for SmolLM2 moments. Retained buffer requests are 70,691,088 bytes
for Whisper SGD and 311,439,916 for ResNet F+L+B. These extend accounting
checks to larger graphs; they do **not** measure a peak-VRAM reduction.

### What this changes in the roadmap

Keep tuning opt-in. Do not lower the guard or fit thresholds to these results.
First add A/A controls and plan-selection crossover/randomized session roles
to a separately declared confirmation protocol, plus clock/host interference
telemetry. Alternating execution order alone does not randomize allocation
order or reproduce steady clocks. Profile the whole step and search phases
separately before spending a larger budget. Then extend reusable convolution
derivative schedules and attention/fusion coverage where those profiles
justify it; searching more scalar matmul tiles cannot optimize excluded work.
Automatic whole-step rollback, persistent winners, native-f32 hardware and
fleet qualification remain open. No new PyTorch or frozen-paper claim follows.

### Qualification and CPU replay

`cargo test --release --tests -- --test-threads=1` passed on this source,
including the 242 library tests and upstream Q4 regression. Default ignored
tests stayed ignored. The fixture-dependent ResNet/Whisper PyTorch-reference
tests returned early because their files are absent; the hub-only SmolLM2
reference target has zero tests without that feature. These are not new
cross-engine correctness results. Separately, all six ignored holdout prefixes
and all four scalar tuner tests passed; native-f32 was deliberately excluded.
All-feature clippy, strict rustdoc, Rust 1.88 library check and the six-test /
50-cell frozen-paper replay also pass locally. Hosted CI was not run here.

```sh
cargo test --test holdout_evidence
cargo test --test holdout_evidence retained_holdout_runs_replay -- --nocapture
```

The Rust replay checks source/settings, tensor rosters and logical extents,
expected counters, numerical-summary arithmetic, every loss pair, raw timing
medians/guards, search decisions/budgets, memory accounting and pipeline-change
counts. Mutation tests reject incomplete or inconsistent evidence. It also
prints the aggregate table values and retains the unchanged-control example
as a regression. It cannot reconstruct omitted full vectors or independently
prove GPU execution from JSON. Both this cohort and the prior pilot are in
CPU CI; frozen publication artifacts remain byte-identical.

Post-measurement engineering: new reports now include `TuneOutcome.phase_times`
for preparation, qualification, warmup and sampling, including partial early
exits. This responds to the unexplained search-cost gap above. These records
predate that instrumentation and have no phase measurements; the tagged source
remains the reproduction target. No measured cost is retroactively attributed
to a phase. See the [observability contract](../../study/observability.md).
