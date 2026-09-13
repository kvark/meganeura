# Final P3HPC cohort: evidence guide

The camera-ready manuscript uses the September 13 collection. Analysis did
not collect new timings, change a condition, or remove an outlier. The
companion report's original data remain separately versioned in paper/results;
they are not inputs to these tables.

## Identity and coverage

All nine archives share Inferena `efb1e5206f07e316c94e8e021514cad60c0639bf`,
Meganeura `75dfe901deb87ca0054c438437efd3aa388b7188`, Blade
`f6f2729e850cc0aefdc0bb18523da58a72765169`, Python 3.13.13, and
PyTorch 2.13.0 at `cf30153c4c131c8164ee7798e5022d810682e2cb`.
Vendor wheels/libraries differ. All manifests agree on the Cargo.lock and
shared checkpoint/config hashes (Windows path separators normalized).
[cohort.sha256](artifact/cohort.sha256) identifies the nine external inputs.

| Configuration | Valid / selected pairs | Conditions |
|---|---:|---|
| RTX 5070, Linux | 90 / 90 | Default without/with replay; searched with replay |
| H100 80GB, Linux | 90 / 90 | Same three conditions |
| RTX 3050, Windows 11 | 60 / 60 | Default without/with replay; searched omitted |
| RX 7900 XT | 30 / 30 | Default/no replay; searched omitted |
| Radeon 780M | 30 / 30 | Same subset, with recorded ROCm overrides |
| Arc B570 | 30 / 30 | Default/no replay; qualified embedding workaround |
| Apple M3, macOS 15.7.3 | 30 / 30 | Eager MPS |
| Intel RPL-U | 30 / 30 | Vulkan versus explicitly selected eager CPU |
| H100 360M/1.7B extension | 5 / 36 | Five strict pairs, then searched 1.7B capture fails |

Thus eight complete device campaigns contribute 390 pairs; the extension
adds five valid pairs and one failed attempt. Seven configurations have GPU
references; the RPL-U CPU reference is separate. RTX 5070 and B570 share one
host and were measured sequentially. B570 uses a secondary PCIe 3.0 x1 link.
An unrelated local Intel setup failure is not part of the supplied cohort or
the paper's availability evidence.

Every successful pair passes both the offline evidence audit and the frozen
Inferena checker. All eight complete replication reports are reproduced
exactly. Every pair individually meets the 5% gradient bounds: maxima are
0.603% sampled-output L2, 0.0863% scalar loss, 2.95% total-gradient norm, and
3.30% parameter-norm-vector L2. Full-element *PyTorch replay* qualification is
distinct from sampled/norm-based *cross-engine* validation.

## Final performance results

Ratios are Meganeura/PyTorch elapsed time. Primary light comparisons use
qualified CUDA replay, default/no-replay ROCm/XPU, and eager MPS.

| Contract / phase | Median ratio over 35 comparisons | Nominal native wins |
|---|---:|---:|
| Strict inference | 1.833 | 5 / 35 |
| Strict minimal shape | 1.430 | 8 / 35 |
| Strict F+L+B | 2.433 | 4 / 35 |
| Accelerated inference | 1.898 | 7 / 35 |
| Accelerated minimal shape | 1.574 | 10 / 35 |
| Accelerated F+L+B | 2.761 | 4 / 35 |

RTX 5070 strict ResNet inference is a nominal near-tie win (about 0.997),
not demonstrated statistical superiority. Radeon is the strongest surface;
H100 accelerated ResNet training is the largest deficit: 20.30x light,
18.92x searched. Windows has complete three-replicate results, not a partial
failure population. Arc's slow secondary link is disclosed, not assumed free.

Strict Pennycook workload means over the seven shared GPU-reference
configurations are 0.52/0.96 inference, 0.66/0.86 minimal, and 0.38/0.99
training (Meganeura/PyTorch). These conditional scores do not describe
universal support. RPL-U lacks a PyTorch GPU path; the separate MI300X
bring-up report lacks a validated Meganeura Vulkan path.

## Replay, preparation, and integrated tuning

Default replay reduces H100 135M token time 3.496 to 1.275 ms (2.74x)
and diffusion F+L+B 14.322 to 4.031 ms (3.55x). The token gains are 1.64x
on RTX 5070 and 1.48x on Windows RTX 3050.

Native search improves strict 135M prefill 13.265 to 9.744 ms on H100 (1.36x)
and 12.644 to 11.596 ms on RTX 5070 (1.09x). Recorded extra three-session
preparation amortizes after about 109 and 234 prefills. H100 native training
improves 1.23x. Production convolution candidates are now integrated:
strict ResNet inference improves 1.20x on H100 and 1.07x on 5070.
Searched PyTorch remains faster in all 60 matched CUDA phase comparisons.

Across 30 H100 searched pairs, PyTorch compiler/first-execution time totals
170.3 minutes (2.84 hours), versus 114.7 seconds native; default/replay
PyTorch totals 14.0 minutes. Full-tensor replay qualification adds 281.7 s
light / 292.5 s searched and is **not** kernel tuning. These are successful
record sums, not total wall duration or one model's startup.
Light native medians span 0.089–3.651 s strict and 0.115–3.633 s accelerated.

H100 strict ResNet PyTorch inference gains 1.28x with search, but compilation
grows 14.09 to 601.27 s: about 970,000 inference calls repay the additional
recorded preparation including capture, excluding research validation.
Private Inductor/Triton caches are fresh; persistent driver/library state is
not reset. These are not guaranteed cold-start or equal-deadline results.

## Scaling and failure interpretation

The H100 extension validates strict default 360M and 1.7B with and without
replay, plus searched 360M. Each larger point has one process, not three;
there are no accelerated extension results. Light/replay training ratios
narrow 5.28 → 5.00 → 3.82 from 135M to 360M to 1.7B. Prefill ratios
3.99 → 4.15 → 3.65 and token ratios 1.44 → 1.38 → 2.51 do not establish
uniform amortization. No scaling law is fitted.

The searched 1.7B record contains a completed Meganeura result and a PyTorch
error. Its traceback reaches a generated Triton reduction during training
forward capture, then reports `cudaErrorStreamCaptureInvalidated`.
It explicitly refers to a previous capture error without identifying that
initiating error. This final log does **not** contain the older cohort's
`CUBLAS_STATUS_EXECUTION_FAILED` diagnostic. Do not reuse that attribution.
H100 now records graphics driver 580.126.09; the older 570/CUDA-compatibility
provenance discussion no longer describes this allocation.

The requested policy fails operationally; no paired searched 1.7B time is
admitted, no numerical gate was relaxed, and unreached conditions are simply
unmeasured. This does not invalidate the complete main H100 campaign.

## Availability and limits that remain

- RPL-U is a working native GPU path versus PyTorch CPU, not evidence that
  Meganeura beats a PyTorch GPU on that device.
- ROCm search failures/timeouts and the 780M's architecture/SDMA overrides
  are documented bring-up findings, not inferred from omitted final arms.
  Arc's embedding-backward probe selects a qualified index-add equivalent.
- Windows search was omitted after costly/intermittently failing bring-up;
  the **final** light/replay campaign succeeds throughout.
- MI300X remains a separately supplied driver-engineering report, not a
  final-revision timing cell. Its original report is outside Git at
  ~/Downloads/p3hpc-v2/MI300x-story.md (SHA-256
  `e9be97e695140e8a36e09da0f0dd850113b0b22359182921dbc34b689d9776e8`).
  Stock RADV excludes graphics-less CDNA; compute-only driver experiments
  do not establish a working model path or a trivial fix.
- The final protocol still omits ROCm/XPU replay and MPS compilation,
  disables native-f32 cooperative tiles in strict mode, and pairs native
  search with PyTorch max-autotune instead of enabling it in every arm.
  These are explicit study limits, not backend impossibility or an
  equal-startup-budget experiment. [Deferred work](NEXT-COHORT.md).

## Profiling evidence and reproduction

The [same-pin RTX diagnostic](https://github.com/kvark/inferena/blob/experiment/nvidia-gap-2026-09-13/ANALYSIS-2026-09-13.md)
localizes ResNet training: ordinary cohort 42.162/13.410 ms; separate Systems
GPU span 39.736 ms versus CUDA kernel sum 13.312 ms. Structured profiling
adds 16% overhead and attributes 77.3% of summed dispatch medians to
convolution gradients. An ordinary tuner log visits 5/71 classes in 2.005 s
and misses the small-output weight-gradient hotspots. Neither wall-minus-
kernels nor a workgroup-barrier warp counter measures removable resource
barriers. Raw traces stay outside Git; source, controls, and recipes are on
the experiment branch.

Run from the repository root with Python 3.11+:

~~~sh
python3 paper/p3hpc/artifact/cohort.py "$HOME/Downloads/p3hpc" \
  --check paper/p3hpc/tables --output target/p3hpc-final-data
~~~

This audits external archives, regenerates nine LaTeX fragments, and exports
all per-condition medians/ranges to CSV without a GPU. Hashes identify exact
observations; rerunning source reproduces the procedure, not the timing noise.
Public hosting of the small measurement bundle remains an author decision.
No Intel server rental or additional cohort is required for this scoped paper.
