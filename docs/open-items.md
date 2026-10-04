# Open items

Remaining work from the October 2026 audit. Completed fixes are listed in
[CHANGELOG.md](../CHANGELOG.md); GPU test setup is in [testing.md](testing.md).

## Training-step host cost

`optimizer_units()` rebuilds parameter segments each step, and `Pipelines::get`
performs a lookup per dispatch. Earlier instrumentation attributed about 7% of
host recording time to that lookup. This does not establish its contribution
to completed-step time or the benefit of caching either result.

Measure encoding, submission and completion separately on the target workload
before changing these paths. Time `step()` together with `wait()` when measuring
a completed step; keep input-upload timing explicit.

## CPU/GPU overlap

`Session::step` waits for the preceding submission before recording. Compare
completed-step throughput with and without encoder rotation using the same
graph, inputs, device and synchronization contract.

The earlier 1–5% overlap bound was based on submission-only timing and is
withdrawn. A corrected RTX 5070 run measured 0.0472 ms in the step call plus
0.0655 ms waiting for completion for the 4-block, width-64 case. The omitted
wait was substantial; subtracting encoding time still does not predict the
benefit of overlap. The temporary measurement tools remain in git history.

## Measurement cautions

Pin the adapter and share its context throughout a comparison. Periodic wall
clock stalls observed on the RTX 5070 also affected a trivial matmul, so a
single timing statistic cannot attribute them to an attention kernel. Use
repeated paired measurements and GPU timestamps where available.
