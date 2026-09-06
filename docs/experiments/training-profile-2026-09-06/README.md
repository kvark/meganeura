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

## Retained results and what they support

All three fresh serial processes completed, with no discarded or repeated
attempts: [run 1](run-01.json), [run 2](run-02.json), [run 3](run-03.json).
[summary.json](summary.json) retains every before/after median, drift and
instrumentation ratio, plus recomputable family shares.

Measured source is `48e23154419e2098db1109be6e59d8cd4d265b6a`, tagged
`evidence/training-profile-2026-09-06`. Executable SHA-256 is
`563f10841be0738ef1dc3546040c841a093ee83d3ba4847032b0260de4b114c4`.
The archived lockfile above is unchanged. This is Linux x86_64, Rust 1.98.0
release, Intel Core i5-12400F, RTX 5070 / driver 595.71.05; f32 tile 0,
f16 tile 16, cooperative execution explicitly disabled, no RUSTFLAGS.

All nine case runs pass. **All 45 retained profiled executions have bit-exact
full parameter/gradient/loss comparisons** with their grouped-execution
references, before ordinary ring advancement. The normal before/after state
checks and all 360 retained losses also match exactly. Adam counters remain
zero. Requested resident model buffers are unchanged: 21,185,284 bytes for
SmolLM2, 70,691,088 for Whisper, 311,439,916 for ResNet, including each unused
four-byte clip scalar. These are requests, not peak VRAM. The zero-time census
visits no class, allocates no scratch and changes no selection.

### Localization, not an optimization decision

Shares below use the sum of dispatch medians; each range spans all three
processes. These are shares of the instrumented plan, not normal wall time.

| Case | Main attributed work | Dispatches | Share range | Median sum of dispatch medians |
|---|---|---:|---:|---:|
| ResNet F+L+B | Backward convolution | 105 | 60.66–60.77% | 11.565 ms |
| ResNet F+L+B | Forward convolution/spatial | 54 | 25.68–25.75% | 4.903 ms |
| SmolLM2 F+L+B | Backward attention | 16 | 36.25–40.58% | 1.257 ms |
| SmolLM2 F+L+B | Backward matrix | 90 | 24.30–25.63% | 0.753 ms |
| Whisper F+L+B | Backward matrix | 40 | 31.51–32.28% | 0.886 ms |
| Whisper F+L+B | Forward matrix | 17 | 19.59–19.95% | 0.543 ms |

ResNet's 52 dX dispatches contribute 6.378–6.384 ms, and its 53 dW dispatches
5.181–5.188 ms. Its largest single dispatch is the stem weight gradient:
`C[64,147] = A[64,12544] * B[12544,147]`, represented implicitly without
materializing im2col. The selected 32×32 tile launches just ten workgroups;
process 1 attributes 0.889 ms to it. Other small-output, long-reduction dW
classes recur across the plan. This motivates a bounded split-K candidate,
not a new hardcoded occupancy threshold or a claim that low occupancy was
measured. No pipeline executable statistics were returned despite the request;
register use, spilling and hardware occupancy remain unobserved.

SmolLM2's eight `MultiHeadAttnGradKV` dispatches contribute 0.712–0.727 ms;
eight `MultiHeadAttnGradQ` dispatches contribute 0.541–0.549 ms. Both are
outside the existing dense-matmul search. Whisper is less attention-dominated
at this short encoder sequence; do not transfer SmolLM2's ranking to it or
to Metal without measurement. The exact census remains 11/9/1 eligible
matmul classes and 90/40/1 eligible dispatches for SmolLM2/Whisper/ResNet.
Dispatch coverage is not time coverage.

The profiler's forward/backward labels split the scheduled plan at its last
loss-producing dispatch. Independent derivative seeds can be scheduled in
that prefix; these labels are not a replacement for node provenance or the
`requires_full_precision` contract. The dX/dW and attention-gradient entries
above explicitly identify derivative operations.

### Every normal block and the disturbance it exposes

All cells are normal **before → after** medians in milliseconds. Both blocks
use the same unchanged session and timestamp-enabled context; they are not
candidate/baseline pairs or a profiler-disabled production benchmark.

| Case | Process 1 | Process 2 | Process 3 | Profiled/first-normal ratios, processes 1–3 |
|---|---:|---:|---:|---|
| SmolLM2 F+L+B | 2.654 → 2.693 | 2.661 → 3.383 | 2.626 → 2.849 | 1.315×, 1.611×, 1.371× |
| Whisper F+L+B | 2.797 → 5.582 | 2.767 → 3.095 | 2.742 → 3.124 | 1.091×, 1.250×, 1.210× |
| ResNet F+L+B | 17.617 → 17.647 | 17.607 → 17.591 | 17.568 → 17.630 | 1.260×, 1.218×, 1.191× |

ResNet drift is −0.09% to +0.35%. SmolLM2 is +1.48%, +27.14%, +8.48%;
Whisper is +99.55%, +11.83%, +13.93%. The first Whisper profile also contains
a 6.416 ms wall sample among roughly 3 ms samples. Keep these observations.
Full numerical parity does not prove timing stationarity. No before/after
pair is interpreted as a gain, and no outlier is replaced by a rerun.

The compute-process query was empty before capture; resident graphics processes
remained. The 533 telemetry rows span 39–50 °C, graphics clocks 180–2917 MHz,
memory clocks 405–14001 MHz, and GPU utilization 0–88% over whole processes.
Readbacks, construction, idle periods and ordinary steps are included. These
samples neither establish isolation nor identify the cause of short-block drift.
No builds, tests, other benchmarks or heavy host analysis overlapped the cohort.

## Correctness finding before a new performance candidate

Reviewing the prioritized convolution family uncovered a pre-existing dX
padding error, reproduced by a new independent f64 scatter oracle. Forward
cross-correlation addresses `ih = oh*stride + kh - padding_h`; its input
gradient must gather `oh = (ih + padding_h - kh)/stride` when divisible.
The stride-1 scalar and generated cooperative kernels instead substituted
`kernel_h - 1 - padding_h`, without also flipping the weight indices. The
same issue applied to width. It is masked when `2*padding == kernel-1` and
does not affect the already-correct general-stride branch.

On the pre-fix code, the new unpadded 3×3 scalar test reports dX[0] = 0.353679
versus oracle 0.425886. The actual generated cooperative path also fails its
unpadded case (−0.641583 versus 0.198180). Commit `fc6be7a` fixes the common
scalar template and cooperative generator. This is a correctness repair,
not a speedup. The measured profile tag is unchanged. Replay verifies that
every stride-1 dX dispatch in this particular cohort satisfies the masking
same-padding identity; that is a source-level scope check, not a rerun of
the corrected executable or a general convolution correctness claim.

[The new oracle regression](../../../tests/conv_derivatives.rs) checks full
dX/dW through autodiff against direct f64 accumulation of forward contributions:
eight shapes × both 32/64 scalar tiles × ordinary/tiny `1e-12` upstream
gradients, including zero/asymmetric/large padding, even kernels, odd extents,
batches, stride 1/2 and 32/64 row/column edges. Generated cooperative dX is
also required to execute on three shapes. On this f16-only GPU, those operands
are pre-rounded to exactly representable f16 and remain bounded; tiny f32
derivatives are **not** qualified on f16. Native-f32 8/16 modules validate
offline; native-f32 execution and fleet coverage remain due.

This explains why control-session parity and an independent oracle are both
needed: two sessions can agree perfectly while sharing an indexing error,
and a same-padding model roster does not test unpadded derivatives.

## Next engineering boundary and replay

The next tuning extension should key convolution derivatives by their complete
direction, batch, spatial/channel/kernel dimensions, stride/padding, placement
and capacity, using the existing 32/64 scalar implementations as legal controls.
Then evaluate split-K dW with an explicitly charged partial buffer and final
reduction. It must preserve full f32 operands and pass the new independent
edge/tiny tests, state isolation and matched whole-step confirmation before any
default change. Do not specialize on ResNet's name, borrow an uncharged buffer,
or widen the numerical gate. Attention scheduling remains the next separately
qualified family; these RTX profiles are not Metal evidence.

```sh
cargo test --test training_profile_evidence
cargo test --test training_profile_evidence retained_whole_step_profiles_replay -- --nocapture
cargo test --release --test conv_derivatives -- --test-threads=1
cargo test --release --test conv_derivatives -- --ignored --test-threads=1
```

CPU replay validates all record identities, rosters, state/loss summaries,
zero-search work, memory requests, timing labels and geometry against dispatch
contracts, convolution shapes/bytes, raw timestamp quartiles/medians, additive
family shares, before/after drift, instrumentation ratios and summary arithmetic.
Mutation tests reject missing profile states, misattributed dispatches, changed
workload/precision, corrupt timing, geometry, memory and summary values. Replay
cannot recover omitted full tensors or prove the producer's authenticity.

After the padding repair, the full release unit/integration suite, generated
cooperative oracle, three-case capture preflight, three ignored library GPU
regressions and five scalar tuning tests pass. All six evidence replay targets,
all-target/all-feature Clippy, strict rustdoc and the locked Rust 1.92 library
check pass. The frozen verifier passes six tests and reproduces all tables
byte-for-byte. Missing external-reference fixtures, disabled hub-only tests,
native-f32 execution and other backends are not newly qualified here.
