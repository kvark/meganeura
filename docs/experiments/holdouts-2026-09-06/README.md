# Whole-step tuning holdouts: protocol

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
[September 5 lockfile](../tuning-2026-09-05/Cargo.lock), SHA-256
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

## Status

Protocol and Rust harness are being prepared before retained measurements.
Results and reproduction commands will be added after qualification.
