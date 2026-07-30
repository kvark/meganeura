# Structured performance profiling

Meganeura has two complementary profiling outputs:

- Perfetto traces show CPU spans, submissions, waits, and GPU passes on a
  timeline.
- Structured session profiles retain repeated hardware timestamp samples and
  attach each dispatch to its selected pipeline, workgroup geometry, execution
  phase, and coarse kernel family.

The structured profile is the default tool for explaining a benchmark gap.
It is reproducible across Vulkan and Metal, can be collected by Inferena after
the ordinary benchmark without changing its reported latency, and produces
JSON that can be compared across revisions. RenderDoc and vendor-specific
capture tools remain useful for a single unexplained shader, but are not part
of the benchmark protocol.

## Quick check

Run the small self-contained example:

```sh
MEGANEURA_GPU_TIMING=1 \
  cargo run --release --example profile_session -- gap-profile.json
```

The example first measures the ordinary grouped-pass execution, then collects
five instrumented samples. The JSON records both times and their ratio.

For paper workloads, use Inferena's profile option so the profile and normal
measurement share the exact graph, inputs, precision policy, and revision:

```sh
INFERENA_MEGANEURA_PATH=../meganeura \
  ./run.sh --frameworks meganeura --model Whisper-tiny \
  --profile --profile-samples 5 --results-dir results/gap-study
```

Inferena writes one sidecar per execution mode under
`results/gap-study/profiles/`.

## Timing contract

Set `MEGANEURA_GPU_TIMING=1` before constructing the first GPU context. Blade
then allocates hardware timestamp queries. During structured capture,
Meganeura deliberately records one compute pass per plan dispatch. Blade's
two-command-buffer ring is advanced with two ordinary executions before each
sample is read back.

On Vulkan, a dispatch interval starts at the top-of-pipe timestamp for its
pass and ends at the next pass's top-of-pipe timestamp. It therefore includes
the dispatch and the inter-pass memory barrier before the following dispatch.
Metal uses pass boundary counter samples. This is appropriate for ranking
end-to-end dispatch costs, but it is not an instruction-level kernel metric.

One-pass-per-dispatch execution is more intrusive than normal execution,
which groups dispatches and uses inline barriers. Never substitute the
profiled wall time for benchmark latency. The artifact includes:

- the normal benchmark median supplied by the caller;
- every profiled wall-time and timestamped-GPU sample;
- the instrumentation wall-time ratio;
- per-dispatch median, quartiles, and raw samples;
- phase and family aggregates;
- pipeline variants and driver-reported executable statistics, when
  available;
- device, driver, plan, barrier, and memory metadata.

The family shares use the sum of each dispatch's median so they add to 100%.
The separate median of each family's per-run total is retained as well.

## Capture API

Call `meganeura::profiler::capture_session_profile` after the normal benchmark
and pass the normal median through `CaptureOptions::unprofiled_median_ms`.
The input-preparation closure runs before the retained execution and both ring
advance executions.

The collector describes dispatches in the compiled execution plan. Disable
optimizer, gradient-accumulation, and gradient-clipping passes before capture;
those passes are appended by the runtime and do not have plan metadata. The
collector rejects a timestamp-count mismatch instead of silently assigning
an auxiliary pass to the wrong shader.

Blade currently supports at most 1,000 timed passes per command encoder. The
collector reports an explicit error if a plan exceeds that limit.

## Escalating beyond the built-in profile

Start with the largest family and dispatch deltas across GPUs or revisions.
Use driver pipeline statistics to check register pressure and spilling, then
inspect the generated WGSL/SPIR-V for the small set of suspect pipelines.

Use RenderDoc only when resource bindings, barriers, generated shader state,
or a driver-specific anomaly needs visual inspection. For instruction
throughput, occupancy, cache behavior, or tensor-core utilization, use the
vendor tool for the affected platform (for example Nsight, Radeon GPU
Profiler, or Xcode GPU capture). Those captures are supporting forensic
evidence, not portable benchmark artifacts.
