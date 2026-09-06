# Whole-step training localization

Separate from frozen P3HPC evidence and all tuning promotion cohorts. This is
a baseline localization study, not a candidate comparison or a speedup claim.
Commit the harness and this protocol before retaining measurements. Keep every
attempt, including numerical/capture errors and unstable normal timing blocks.

## Fixed boundaries and protocol

Reuse the holdout builders, deterministic fan-in initialization, fixed inputs,
ordinary fusion, aliasing and device-local placement. Force scalar f32 matmul
and attention, with no tuning. Three cases, all **forward + loss + backward**:

- `smollm2-flb`: medium-test decoder, eight layers, width 128, FFN 256, two
  heads, vocabulary 64, sequence 127; synthetic next-token cross entropy.
- `whisper-flb`: Whisper-tiny encoder, batch 1, 100 mel frames, width 384,
  four layers; mean squared encoder output.
- `resnet50-flb`: folded-BN ResNet-50, batch 1, 224² RGB image, 1,000 classes;
  cross entropy. Not running-statistic BN training.

Unlike the earlier SmolLM2/Whisper holdouts, **no optimizer, clipping or gradient
accumulation** is configured. Weights do not evolve. These are not Adam/SGD
step profiles: the existing structured collector cannot attribute runtime-
appended passes. Do not compare their wall times with those different training
boundaries. There are no pretrained weights or independent engine oracles.

Three fresh serial processes, seeds 1..3, rotate the fixed case order so each
case appears in every position. Use the published Blade 0.9 resolution from
the [archived readback lockfile](../readback-2026-09-06/Cargo.lock), SHA-256
`72cecddd0f090e82d5db46bc6f7f80429a058b2afa0eaec84a9b27e7340a4a80`.
Do not use the sibling Blade checkout for retained runs.

For each case:

1. Build/initialize once. Record selected pipeline keys, full dispatch contracts
   (including uniform words and bindings), requested memory, and a zero-time
   search census. The census must visit no class and change no selection.
2. Warm for 30 ordinary steps and retain a reference summary over all full
   parameter/gradient/loss tensors. Require finite values and nonzero gradients.
3. Settle for five steps, then retain 20 normal `step + wait` wall samples and
   every loss outside the timers. Compare the complete state with the reference.
4. Settle for five ordinary steps, then collect five hardware-timestamp profiles
   using one pass per dispatch. After each retained step, compare its full
   parameter/gradient/loss state **before** the two ordinary ring-advance steps
   can overwrite it. Readbacks use separate encoders and fall outside the
   retained wall timer. The two ring-advance steps also precede the next sample.
5. Settle for five steps, retain a second 20-sample normal block and complete
   final state comparison. Require unchanged keys and requested memory.

Numerical tolerance is the existing full-tensor relative L2 ≤2e-4 and
elementwise `1e-6 + 2e-4*abs(reference)` contract. Retain exactness separately;
do not weaken the thresholds or substitute finite losses for full gradients.
Check zero Adam counters, zero moment/accumulator requests, the existing unused
four-byte clip scalar, and all five profiled state comparisons. No clipping
pass runs. Raw vectors are compared in memory; only summaries are saved.

The timestamp context is enabled for both normal blocks, but only the capture
splits every dispatch into a pass. Retain instrumented wall times and their
ratio to the first normal median. Vulkan top-of-pipe pass intervals include
the next pass's preceding barrier; these are not pure instruction/kernel times.
Report before/after normal medians and drift without discarding unstable runs.
Neither an instrumentation ratio nor a family share establishes an Amdahl bound
for ordinary execution. Capture readbacks interrupt steady-state residency and
clock behavior between samples; the following ring steps are not a clock lock.

Retain per-dispatch raw timestamp samples, family/phase aggregates, pipeline
statistics where available, source/executable/lock identities and optional
250 ms NVIDIA telemetry. No builds, GPU tests, other benchmarks or heavy host
analysis overlap any retained process. Do not clear driver caches, lock clocks,
or imply graphics processes and unobserved interference are absent.

```sh
cargo test --example profile_training
cargo test --release --example profile_training -- --ignored --test-threads=1
cargo build --release --locked --example profile_training
for seed in 1 2 3; do
    target/release/examples/profile_training "new-training-profile-${seed}.json" "$seed"
done
```

The runner refuses dirty tracked source and existing outputs and saves completed
cases incrementally. Profiles rank the next reusable operation/direction/shape
family to investigate. They do not authorize model-name thresholds, larger tune
budgets, reduced derivative precision or deletion of validation. Any candidate
implementation and matched whole-step performance decision need a separately
committed protocol and new source identity after localization.

Before measurement, the two-capture GPU preflight passes for all three cases,
including full profiled-result comparisons before ring advancement and exact
timestamp counts. The roster/arithmetic CPU tests and all-target/all-feature
Clippy pass. The preflight corrected an initial accounting assumption: training
sessions reserve a four-byte clip scalar even with clipping disabled; it is
retained in the protocol, not erased from memory accounting.
