# Persistent Session qualification — October 24, 2026

Milestone 1 passes correctness and harness integration. It does **not** establish
a performance improvement: the production ordinary backend wins these paired
comparisons and remains the default. These are generated pointwise/reduction
regions with explicit ordinary fallback, not a device-scheduled graph.

## Configuration and boundaries

Source `d3b167ba632ae2446ec935f3112a1c0b843a0b5c`, clean working tree;
Blade `7a20f25eb0d63591611c9622cb8391b5f88f88a4`, Naga
`a489554c96f87cbc5252ee00350730f6e93869f9`. Linux x86-64, Rust 1.98.0
(`88d9e12ae`, LLVM 22.1.8), release builds, RTX 5080, NVIDIA 595.91.07.
The device was idle before measurement; clocks were not locked. No competing
GPU test or benchmark ran during measurement. Default compilation/precision
policy, no kernel search on either side, 64 persistent workers. Each pair
asserts exact equality of the compiled plans.

The primary runs use 50 alternating AB/BA pairs after at least ten warmup pairs
and one second of warmup. Parameters are restored before every execution,
outside timing. CI latency includes input upload, completed execution and
output/loss readback. SmolVLA measures completed execution, excluding uploads
and readbacks, with the existing base action expert: 16 layers, action chunk
50, VLM sequence 16, deterministic synthetic inputs and weights. SGD learning
rates are 0.001 for CI and 1e-5 for SmolVLA.

Every sample checks full logits or scalar loss. Training additionally checks
all available parameter gradients and updated parameters, excluding allocation
padding, and requires an actual update. **All primary and diagnostic comparisons
had zero maximum absolute error.** The independent f64 oracle suite and its
tolerances remain unchanged. Full-state readback/comparison happens between
samples; these timings measure completed-step latency, not sustained throughput.

## Primary completed-work latency

Milliseconds; brackets contain the interquartile range. Ratios are the median
of paired persistent/ordinary times, not the ratio of independent medians.

| Case | Ordinary median [IQR] | Persistent median [IQR] | Paired ratio [IQR] |
|---|---:|---:|---:|
| CI inference | 0.05139 [0.04800–0.05217] | 0.05246 [0.04970–0.05357] | 1.021 [0.956–1.122] |
| CI forward/backward | 0.06506 [0.06401–0.06558] | 0.08040 [0.08021–0.08093] | 1.241 [1.228–1.263] |
| CI forward/backward/SGD | 0.06704 [0.06593–0.06775] | 0.08238 [0.08200–0.08279] | 1.231 [1.219–1.253] |
| SmolVLA forward | 4.44244 [3.88218–4.57366] | 4.88146 [4.14028–5.23476] | 1.085 [1.063–1.111] |
| SmolVLA forward/backward/SGD | 15.32991 [13.09990–15.64697] | 17.16084 [14.29606–17.49099] | 1.136 [0.984–1.159] |

SmolVLA training has substantial outliers: ordinary/persistent p95 values are
79.03/54.39 ms. Raw samples retain them. The evidence supports keeping ordinary
execution as the default; it does not justify a precise universal slowdown.

## Coverage, memory and construction

Graph launch counts exclude the queue-reset transfer and appended optimizer
dispatches. SGD remains an explicitly reported ordinary fallback. Memory is
Session-reported buffer residency, including scheduler allocations, rather than
total driver/process memory. Persistent sessions disable lifetime aliasing.

| Case | Persistent source operations | Graph launches ordinary → persistent | Persistent / fallback launches | Resident MiB ordinary → persistent |
|---|---:|---:|---:|---:|
| CI inference | 2 / 5 | 5 → 5 | 2 / 3 | 0.06995 → 0.08289 |
| CI forward/backward, with or without SGD | 5 / 18 | 18 → 18 | 5 / 13 | 0.13229 → 0.16976 |
| SmolVLA forward | 21 / 154 | 154 → 154 | 21 / 133 | 563.42 → 599.84 |
| SmolVLA forward/backward/SGD | 88 / 523 | 523 → 522 | 87 / 435 | 982.08 → 1027.61 |

Scheduler allocations are respectively 1,280, 4,608, 39,424 and 133,632 bytes.
Observed construction times (ordinary/persistent ms) are CI inference 1.08/1.65,
CI training 9.80/3.66, SmolVLA forward 60.91/52.62 and SmolVLA training
86.05/74.92. Both selections share one GPU context and ordinary sessions are
built first. Shader-cache reuse makes these unsuitable as a matched cold-start
comparison.

The already optimized production graphs offer little launch reduction in the
current handler scope. The strict synthetic fixtures separately demonstrate
ten independent operations in four launches and eight chained operations in
three launches. Those launch reductions are not production timing wins.

## Separate diagnostics

Ten alternating pairs, at least five warmup pairs and 500 ms warmup, with GPU
timestamps, native capture statistics and worker counters enabled. GPU values
are summed pass-start intervals through submission completion; separate input
uploads and readbacks are excluded. Do not substitute these instrumented values
for the primary wall times above.

| Case | Ordinary GPU median ms | Persistent GPU median ms | Workers claiming work per persistent region, min–max (mean) |
|---|---:|---:|---:|
| CI inference | 0.013536 | 0.016560 | 4–4 (4.0) |
| CI forward/backward | 0.028032 | 0.039808 | 1–4 (3.4) |
| CI forward/backward/SGD | 0.030768 | 0.043968 | 1–4 (3.4) |
| SmolVLA forward | 3.462912 | 3.762880 | 3–64 (42.1) |
| SmolVLA forward/backward/SGD | 14.401648 | 16.382880 | 3–64 (54.9) |

Each final-step histogram sums to its region's logical task count. Worker
activity is not SM occupancy or CU affinity. Driver-reported persistent shader
register counts are 16 for CI and 16–22 for SmolVLA; shared memory is 4 bytes
for CI and 4–1,028 bytes for SmolVLA. Raw executable statistics are retained;
the implausible reported per-thread local-memory value is not used to infer
spilling or occupancy. No Metal GPU performance claim is made.

## Reproduction and retained records

Use `systemd-run --user --scope --slice=ml.slice` before each command on the
qualification machine. Run commands sequentially, with no other GPU workload:

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
cargo run --release --example bench_ci_latency --offline -- --paired
cargo run --release --example bench_smolvla_train --features models --offline -- --paired --force
MEGANEURA_GPU_CAPTURE=1 cargo run --release --example bench_ci_latency --offline -- \
  --paired --diagnostics --warmup 5 --warmup-ms 500 --runs 10
MEGANEURA_GPU_CAPTURE=1 cargo run --release --example bench_smolvla_train --features models --offline -- \
  --paired --diagnostics --warmup 5 --warmup-ms 500 --runs 10 --force
```

Raw JSON and corresponding `.log` files remain under
`target/megakernel-results/`, outside Git per repository policy. The JSON
contains every sample, execution region, selected key, worker histogram and
available compiled-pipeline statistic. SHA-256 identities:

| Record | SHA-256 |
|---|---|
| `milestone1-ci-paired.json` | `aea514271c90bc1bd91475a509540b92a1b991a99dbf46c1b44ed0d755f9bbdb` |
| `milestone1-smolvla-paired.json` | `e9d85bc23991dcc7880b2b2b16848b44e8e758ad824a01dd3676f44a15aa1963` |
| `milestone1-ci-diagnostic.json` | `999928f73bce860fb53b3ee85f961bb4e933715d62b8f0e24162016c017441ab` |
| `milestone1-smolvla-diagnostic.json` | `7f827a9b09faf69bb4adb596e975962357cff57b60a7bd68759e115061325725` |

Ordinary and persistent suite qualification, strict coverage fixtures, lavapipe
checks and cross-compilation limits are recorded in the
[integration plan](megakernel_plan.md). These establish the Milestone 1 exit
gate and permit the Naga protocol investigation to start.
