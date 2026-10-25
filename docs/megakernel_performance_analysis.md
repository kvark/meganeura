# Persistent graph performance: NVIDIA counter diagnosis

This investigation measures the earlier **regional** backend. The newer
[single-dispatch graph results](megakernel_whole_graph.md#nvidia-counter-diagnosis)
initially showed a different regression: the lane adapter repeated 64-lane tensor
calculations across a 256-lane physical workgroup. Masking inactive subgroups'
products removed that approximately fourfold tensor work. The linked report
tracks subsequent instruction and local-memory costs and the remaining gap.

The principal SmolVLA regression comes from the fixed worker pool. At 64
workers, large tiled kernels lose parallelism; increasing the pool restores
their performance but adds overhead to kernels with few tiles. The dominant
regressing kernels use the portable ticket queue, without device-scope fences.
Changing the Naga atomic interface is therefore not the first performance fix
supported by these measurements.

This diagnosis uses an RTX 5080 (GB203), NVIDIA driver 595.91.07, and
[Nsight Perf SDK 2026.3](https://developer.nvidia.com/nsight-perf-sdk).
The production source is `9b0a0aaf21bf2e2cdfa8af48483bff56bba1e485`, with
the benchmark capture hook added in this change. The measured executable's
SHA-256 is
`c836c9296de86c491d0bbe3bee8b8378b905e380b3775e59a8d795b72ceea7d4`.
No production shader, arithmetic implementation, precision policy, or default
worker count changed during the investigation.

## Counter evidence

The SDK's standalone `Monitoring` executable sampled a warmed, continuously
replayed graph. Each target ran for ten seconds; monitoring started one second
after `CAPTURE_READY` and collected three seconds at 20 ms intervals. The table
uses the remaining 146 samples after discarding the first five. These are
whole-workload averages, not individual-kernel counters.

| Forward graph metric | Ordinary | 64 workers | 256 workers | 512 workers |
|---|---:|---:|---:|---:|
| Graphics engine active, % | 92.80 | 95.76 | 94.02 | 93.56 |
| Compute active warp slots, % of peak | 9.64 | 5.28 | 9.77 | 11.58 |
| Tensor pipe activity, % of peak | 2.68 | 1.40 | 2.04 | 2.06 |
| Instructions executed, % of peak | 10.84 | 5.98 | 9.01 | 9.45 |
| DRAM read throughput, % of peak | 11.26 | 5.87 | 8.52 | 8.59 |
| L2 texture-sector hit rate, % | 69.97 | 74.06 | 71.56 | 70.42 |
| GPC clock, GHz | 2.853 | 2.846 | 2.843 | 2.843 |
| Diagnostic resident replay median, ms | 3.726 | 7.112 | 4.884 | 4.881 |

For the forward/loss/backward graph, active warp slots similarly fall from
12.25% to 6.28% with 64 workers, then recover to 11.25% with 256 and 12.84%
with 512. Its diagnostic replay medians are 10.518, 19.017, 13.778, and
13.371 ms respectively. Optimizer updates are disabled for counter attribution.

The 64-worker case keeps the engine active but executes far less work per unit
time. Clocks are essentially unchanged, and aggregate DRAM throughput is low.
This supports insufficient parallelism rather than CPU submission gaps or
whole-workload memory-bandwidth saturation as the leading explanation. The
active-warp metric includes elapsed idle time; it is not a per-kernel achieved
occupancy measurement. These aggregates cannot exclude a local memory stall
inside an individual shader.

Every ordinary/64/256/512 capture used identical selected implementation keys.
Loss, all logical parameters, and all available parameter gradients matched
bitwise across backends and before/after capture. State SHA-256 values:

- Forward: `4e5d66e6ce5e95142b848651d316b4df64855a9d85669499cabc3a81089be629`.
- Forward/loss/backward: `f67022044ae8966ac48c5d62cd0907411fb42f4d1864c1ba0b3d555b17ce7be0`.

All eight targets and monitors exited successfully, with no SDK warnings or
dropped-data reports and no graph fallback. The capture path performs no input
uploads or state readbacks inside the resident loop. Artifact relocation ran
concurrently on the CPU during these diagnostic captures; it was paused for
the primary paired benchmark below.

## Dispatch attribution

The existing structured profiler captures five samples per physical execution
region and retains ordinary logical members, pipeline keys, workgroup geometry,
and driver executable statistics. Its additional compute-pass boundaries affect
timing, so these intervals explain costs rather than replace the paired
benchmark. In particular, the multi-window backward profile perturbs the
ordinary total substantially.

| Forward kernel | Logical tiles | Ordinary interval | 64-worker interval | 256-worker interval |
|---|---:|---:|---:|---:|
| `MatMul[50x4096x720]` | 256 | 67.58 µs | 296.99 µs | 69.60 µs |
| `FusedMatMulAdd[50x720x2048]` | 46 | 65.47 µs | 81.89 µs | 96.22 µs |

Values are medians across repeated regions. The large matrix appears sixteen
times. Its aggregate interval falls from 4.774 ms at 64 workers to 1.103 ms at
256, nearly matching ordinary dispatch's 1.088 ms. The small fused matrix has
sixteen ordinary instances and fifteen singleton persistent instances; the
remaining instance is composed with another matrix and excluded from this row.

For the large matrix, ordinary dispatch launches `[2,128,1]` workgroups, while
the persistent kernel launches `[64,1,1]` and processes the same 256 tiles in
waves. Raising workers to 256 changes neither its generated pipeline nor its
arithmetic. This is a direct worker-count ablation. The persistent shader uses
55 registers versus ordinary's 72, and 2192 versus 2176 bytes of shared memory;
increased register pressure does not explain this kernel's regression.

The small fused matrix has only 46 logical tiles. Its persistent shader retains
48 registers and increases shared memory only from 8448 to 8464 bytes. Yet it
gets slower as workers rise. The portable queue makes one atomic ticket claim
per logical tile plus one terminal claim per worker, along with workgroup
synchronization and dynamic tile dispatch. Thus this region requires 110
claims with 64 workers and 302 with 256, versus no scheduler claims in ordinary
dispatch. Excess workers and queue overhead are the supported explanation;
these captures do not separately quantify atomic latency, barrier stalls, or
the cost of dynamic indexing.

Backward attention shows the same pool sensitivity. The 32 ordinary dQ/dKV
dispatch intervals sum to 2.175 ms. The sixteen composed persistent regions
sum to 5.518 ms with 64 workers and 2.393 ms with 256. Composition increases
registers from 35 to 46 and shared memory from 512 to 1040 bytes, but changing
only worker count removes most of the penalty.

Only four forward regions use the native dependency protocol. Their combined
interval is 0.184 ms at 64 workers, compared with 0.127 ms for their ordinary
members. The other 132 regions use portable queues and account for 4.281 ms of
the 4.339 ms summed interval regression. That puts approximately 99% of this
profiled forward gap outside the device-fence protocol. The native regions may
still benefit from a more efficient queue and ordering protocol; this result
does not justify weakening synchronization.

The driver reports an implausible 64 GiB `Local Memory Size` for these pipelines.
That field is retained in raw artifacts but excluded from any spill diagnosis.

## Completed-step benchmark

Each worker count uses 50 alternating ordinary/selected pairs after at least
ten warmup pairs and one second of warmup. Parameters and inputs are restored
before each execution; timing includes `step()` and completion wait. The
training case includes SGD at learning rate `1e-5` and checks the loss, every
available parameter gradient, and every updated parameter after each sample.
Timestamps, SDK sampling, worker counters, concurrent builds, and artifact
copying are disabled during these measurements.

| Workers | Case | Ordinary median, ms | Selected median, ms | Median paired ratio | 95% bootstrap interval |
|---|---|---:|---:|---:|---:|
| 64 | Forward | 4.417 | 8.442 | 1.909 | 1.903–1.913 |
| 256 | Forward | 4.396 | 5.385 | 1.228 | 1.225–1.234 |
| 512 | Forward | 4.413 | 5.413 | 1.263 | 1.226–1.305 |
| 64 | Forward/backward/SGD | 15.213 | 24.488 | 1.611 | 1.595–1.613 |
| 256 | Forward/backward/SGD | 15.106 | 18.337 | 1.219 | 1.201–1.226 |
| 512 | Forward/backward/SGD | 15.103 | 17.892 | 1.170 | 1.159–1.191 |

The selected median improves by 36.2% forward and 25.1% with SGD when the pool
increases from 64 to 256. Ordinary dispatch remains faster. Intervals resample
each run's 50 paired ratios 10,000 times with seed `20261025`; they describe
sampling uncertainty within these runs, not variation across machines or days.
All comparisons have zero observed error. Arithmetic plans and implementation
keys match exactly, with no graph or appended-work fallback.
The 512-worker forward run has a wider ratio distribution; medians of paired
ratios need not equal ratios of marginal medians. These measurements do not
establish one globally optimal worker count.

## Next optimization

1. Choose pool size per region using logical tile count, workgroup shape, and
   device occupancy limits. Cap unnecessary workers for small independent
   regions; give large matrix and attention regions enough parallelism.
2. Avoid shared ticket claims for the initial independent wave, then compare
   chunked claims or another inexpensive distribution scheme for further tiles.
   Preserve the same selected kernels and exact state checks.
3. Re-run full completed-step benchmarks, including appended optimizer work,
   before changing defaults. Evaluate native dependency publication and fence
   costs separately after the larger portable-queue cost is addressed.

A global increase to 256 is a useful diagnostic, not a general sizing policy.
Increasing to 512 barely changes forward replay time and launches many more
empty workers. No production default is changed by this analysis.

## Reproduction

The SDK is installed locally at `/x/Tools/nsight-perf-sdk-2026.3/`. Its
`NvPerf/bin/x64/libnvperf_grfx_host.so` symlink resolves to
`../../lib/x64/libnvperf_grfx_host.so`. This was needed for the SDK executable's
own host-library lookup. Counter access was already enabled
(`RmProfilingAdminOnly: 0`). No driver settings or clock controls were changed.

Build once, then keep other GPU workloads and builds stopped during collection:

```sh
cargo build --release --features models --example bench_smolvla_train
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  python3 scripts/profile_smolvla_sdk.py \
  --sdk /x/Tools/nsight-perf-sdk-2026.3 \
  --output target/sdk-forward --phase forward --workers 64 256 512
```

Repeat with `--phase backward` and a new output directory for the
forward/loss/backward graph. The script requires Python 3.11+, generates the
single-pass metric configuration, runs the ordinary reference and each worker
count sequentially, rejects monitor warnings/failures, checks complete observed
state and selected implementation keys, and records provenance, logs, CSVs,
raw timing samples, and binary snapshots. The metric roster is qualified on
GB203; unsupported GPUs must use an appropriate supported single-pass roster.
It neither locks clocks nor injects a library into the target process.

For dispatch attribution:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
MEGANEURA_GPU_TIMING=1 MEGANEURA_GPU_CAPTURE=1 \
  target/release/examples/bench_smolvla_train --force --warmup 10 --runs 20 \
  --profile-json target/ordinary-profile
```

Repeat with `MEGANEURA_PERSISTENT_WORKERS=256`,
`MEGANEURA_PERSISTENT_STRICT=1`, and `MEGANEURA_DEVICE_SCHEDULING=1` for selected
execution. `--profile-json` alone captures both phases; adding
`--capture-phase forward|backward --capture-seconds 10` selects the resident
loop used by the SDK script. Both paths disable appended optimizer work and
validate state after capture. The existing paired benchmark remains the
completed-step measurement path.

For the primary completed-step comparison, omit all profiling environment
variables and run each count separately:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  target/release/examples/bench_smolvla_train --force --device-scheduling \
  --workers 256 --warmup 10 --runs 50 > target/paired-256.json
```

Raw investigation artifacts are local under
`target/megakernel-results/perf-sdk-20261025/`, including
`counter-summary.json`, SDK receipts/CSVs/logs, structured profiles, and state
snapshots. `paired-summary.json` summarizes all 300 paired comparisons;
`paired-benchmark-receipt.json` records the copy pause and benchmark commands.
Formatting and Clippy for the model benchmark pass, and the reusable SDK script
was exercised end to end for ordinary/64/256 forward captures. Initial Nsight
Graphics captures reported dropped periodic samples
and were rejected. A `MEGANEURA_SERIAL_DISPATCH=1` experiment changed the
compiled graph from 523 to 555 operations, so it was also rejected as evidence
about composition alone. Neither experiment contributes to the conclusions.
