# Final P3HPC cohort: findings and interpretation

This is the camera-ready evidence guide. The original-submission numbers in
`paper/results/` remain historical evidence for the companion report; the
P3HPC manuscript now uses the September 12 cohort. No new measurements were
collected during this analysis.

## What was audited

All nine archives share Inferena `17d13a3a94cc5bdfaa63f0eb2cb7e8ee7593c053`,
Meganeura `fcdd76d1a4cd0e3d10507e56ea3f2412378a2ba7`, Python 3.13.13,
and PyTorch 2.13.0 at `cf30153c4c131c8164ee7798e5022d810682e2cb`.
Vendor wheel suffixes, drivers, and backend libraries differ and remain
recorded. Input archive/log identities are in [cohort.sha256](artifact/cohort.sha256).

| Campaign | Valid / selected process pairs | Interpretation |
|---|---:|---|
| RTX 5070, Linux | 90 / 90 | Complete light/no-graph, light/replay, searched/replay |
| H100 80GB, Linux | 90 / 90 | Same complete three-condition matrix |
| RX 7900 XT | 30 / 30 | Complete default/no-graph; searched condition unavailable |
| Radeon 780M | 30 / 30 | Same subset, with recorded ROCm overrides |
| Arc B570 | 30 / 30 | Complete default/no-graph; qualified embedding workaround |
| Apple M3 | 30 / 30 | Complete eager-MPS comparison |
| Intel RPL-U | 30 / 30 | Complete Vulkan-versus-eager-CPU support comparison |
| RTX 3050, Windows | 31 / 90 | First full repetition, then one pair, then capture failure |
| H100 larger models | 5 / 36 | Five strict pairs, then 1.7B searched capture failure |

There are 330 valid pairs in seven complete campaigns and 36 in interrupted
campaigns. Two attempted pairs fail; subsequent unreached conditions are
unmeasured. Both failed pairs contain a completed Meganeura record and a
PyTorch error record. Neither supplies a validated paired performance ratio.
MI300X is a separate bring-up report, not a tenth paired archive.

The analysis verifies checksums, source identity, raw/joined equality,
20-sample medians, numerical gates, replay qualification summaries, and
replication. All 366 completed pairs individually pass the original 5%
gradient bounds; the accelerated protocol's 10%-per-run/5%-median allowance
is not needed to admit any of them. Worst errors are 0.628% sampled-output
L2, 0.0861% loss, 2.94% total gradient norm, and 3.31% parameter-norm-vector
L2. The old 780M Whisper oracle dispute does not recur.

## The new headline

Ratios mean Meganeura time / PyTorch time. The primary light comparison uses
CUDA replay on NVIDIA, default compilation on ROCm/XPU, and eager MPS.
It covers six complete GPU-reference configurations; CPU and partial
campaigns do not enter these aggregates.

| Contract / phase | Median ratio | Nominal Meganeura wins |
|---|---:|---:|
| Strict inference | 1.659 | 5 / 30 |
| Strict minimal shape | 1.264 | 10 / 30 |
| Strict F+L+B | 2.407 | 4 / 30 |
| Accelerated inference | 1.903 | 5 / 30 |
| Accelerated minimal shape | 1.305 | 11 / 30 |
| Accelerated F+L+B | 2.486 | 4 / 30 |

One nominal strict inference win is RTX 5070 ResNet at 0.998: effectively
parity, not a statistically demonstrated advantage. The full per-workload
tables are generated under [tables](tables/). The six-system strict
Pennycook workload means are 0.55/0.96 inference, 0.68/0.83 minimal,
and 0.39/0.98 training (Meganeura/PyTorch). These are conditional shared-GPU
scores. Requiring support on every attempted GPU gives both stacks a known
hole: PyTorch on RPL-U, Meganeura on MI300X.

Radeon is the strongest performance surface. On RX 7900 XT, four strict
inference workloads are within 1.12× of PyTorch; SmolVLA is 0.72×.
Arc B570 wins the two small transformer shapes but loses full inference and
training. M3 loses full inference/training even against eager MPS.
On H100, accelerated ResNet training is 19.80× slower under light and 19.23×
under searched preparation. That gap belongs in the paper prominently.

## Replay and automatic search both matter

With default PyTorch compilation, whole-phase replay changes H100 135M
one-token latency from 4.244 to 1.276 ms (3.32×), and diffusion training
from 12.035 to 4.035 ms (2.98×). RTX 5070's token improvement is 1.65×.
The reviewer was right that omitting this control could reverse a result.

Meganeura's bounded search runs inside session construction and finds
useful alternatives: strict 135M prefill improves 1.38× on H100 and 1.09×
on RTX 5070. H100 training improves 1.24×. The measured prefill savings repay
the additional recorded preparation after about 111 and 238 calls,
respectively. Graph rewriting remains greedy in both arms.

PyTorch's searched CUDA policy wins all 60 phase comparisons against
searched Meganeura across the two complete NVIDIA machines and both
arithmetic modes. Its gains over light vary greatly: H100 strict ResNet
inference improves 1.28× but compilation grows 15.14 → 708.33 seconds,
requiring about 1.14 million inference calls to repay the additional
preparation. Other workload/phase combinations repay sooner or never.
The 68-SM Inductor gate declines GEMM template search on 5070/3050;
other compiler choices still improve some workloads. H100 runs the larger search.

These break-even estimates charge the complete recorded three-phase setup,
add CUDA graph preparation, and exclude research qualification. They are
differences of process medians, not confidence bounds. Checkpoint loading
and the rest of process startup are outside the reported compilation fields.
Private Inductor/Triton caches are fresh; persistent driver/vendor caches
remain as found. The light policy is not a globally cold-start guarantee.

## What the crashes do and do not invalidate

Windows fails at `r2/strict/SmolLM2-135M/max-autotune-graph1`;
H100 extension fails at `r1/strict/SmolLM2-1.7B/max-autotune-graph1`.
Both logs first report `CUBLAS_STATUS_EXECUTION_FAILED` in `cublasSgemm`
during the training **forward** call inside graph capture, followed by
`cudaErrorStreamCaptureInvalidated` while ending capture. This is not a
numerical gate rejecting Meganeura, a tuning timeout, or proven memory
exhaustion. The logs alone do not distinguish a library/driver problem
from a PyTorch or harness capture interaction.

The Windows condition succeeded in repetition one: do not call it
universally unsupported. Its 31 valid pairs demonstrate Windows execution,
but they do not satisfy the three-repetition campaign. Preserve every sample
and the crash log; do not invent remaining failures or pool the partial
records into a completed population. The complete 5070 and H100 five-model
campaigns are independent and usable.

The first Windows strict light/replay repetition has native/reference
inference ratios of 1.23–2.14 and training ratios of 2.19–2.77 across the
five workloads. Its small transformer shapes are close (1.03 and 1.08).
These are single-process observations, not replicated Windows speed claims.

H100 records graphics driver 570.195.03 with CUDA 13 wheels, but not the
loaded CUDA driver-library identity. CUDA 13 ordinarily requires driver
580 or later; NVIDIA's forward-compatibility package supports CUDA 13 on
570-series datacenter hosts. The archive cannot establish which compatibility
libraries were active. This is missing environment provenance, not proof
of an unsupported run or the cause of the crash; Windows shows the same
symptom on driver 591.86.
[CUDA driver requirements](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html),
[forward-compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).

## The larger models are useful, with a narrow claim

| H100 strict, default compilation + replay | 135M | 360M | 1.7B |
|---|---:|---:|---:|
| Process replicates | 3 | 1 | 1 |
| Prefill, Meganeura / PyTorch ms | 12.17 / 3.34 | 19.81 / 5.14 | 42.43 / 11.96 |
| Stateless token, ms | 1.52 / 1.28 | 2.21 / 1.86 | 6.99 / 3.27 |
| F+L+B, ms | 49.22 / 9.93 | 70.93 / 14.98 | 133.80 / 35.83 |
| Training ratio | 4.95 | 4.73 | 3.73 |

Training's gap narrows, prefill stays near 3.5–3.9×, and the token gap grows
at 1.7B. Thus larger weights amortizing fixed costs is a partial explanation,
not a general observed scaling law. Model architecture and matrix shapes
also change. Searched 360M completes (14.10/5.16 ms prefill; 55.11/14.42 ms
training). There are no paired searched 1.7B or accelerated extension results.
Batch one, 128 tokens, no optimizer, no KV cache, and no communication remain
limitations even on H100.

## Variability and attribution

H100 accelerated native ResNet inference has process medians
11.932/5.639/5.564 ms in the light/replay condition. The searched condition
has 13.296/5.524/5.480 ms. Some individual samples approach 19 ms, while
training stays near 50 ms. Keep the prescribed medians, flag their spread,
and make no claim that the table diagnoses the transient. Other visible
variation includes B570 training and CPU-reference latency. The generated
CSV reports minimum and maximum process medians, plus
`(max process median - min process median) / median` per phase.

The paper now includes the separate qualified RTX 5070 Nsight timeline
diagnostic: host recording, grouped GPU spans, and PyTorch kernel intervals
for 135M and resident 1.7B. It also explains the measured CPU downclock
effect and convolution-specialization experiments. These have different
source revisions and placement controls. They are explanations and tested
engineering directions, not replacements for final-cohort timings.
There is still no removable-barrier percentage or systematic per-kernel
decomposition across H100, AMD, and Intel. Wall-minus-kernels cannot supply it.

## Do we need an Intel server GPU?

Not to finish this paper. H100 supplies datacenter execution and preliminary
larger-model results; B570 supplies the missing native-XPU comparison.
Another full campaign is lower priority than reviewing the revised argument.

If access is already easy, Flex 170 has explicitly documented Vulkan 1.3
and 16 GB GDDR6, making it a plausible server deployment check. Its Xe-HPG
architecture would add less evidence about HPC scaling than a new large-memory
accelerator. [Intel Flex 170 specifications](https://www.intel.com/content/www/us/en/products/sku/230019/intel-data-center-gpu-flex-170/specifications.html)

Max/Ponte Vecchio would be a more distinct HPC target, but a PyTorch/XPU
path is not sufficient: verify a working supported Vulkan ICD on the actual
allocation before spending on collection. Intel's Max 1550 specification
lists oneAPI and 128 GB HBM2e, without promising a Vulkan version.
This absence is not itself proof that no experimental path can work.
[Intel Max 1550 specifications](https://www.intel.com/content/www/us/en/products/sku/232873/intel-data-center-gpu-max-1550/specifications.html)

MI300X is worth including precisely because it is an adverse result for our
substrate: the supplied report has working ROCm but no validated Vulkan
model execution. Mesa explicitly scopes RADV to graphics-capable hardware.
Neither trivial compute on a custom driver nor an invalid-gradient run
establishes a usable ML path. [RADV hardware scope](https://docs.mesa3d.org/drivers/radv.html)

## Reproduce the analysis

With the original archives and three supplied text reports in one directory:

```sh
python3 paper/p3hpc/artifact/cohort.py "$HOME/Downloads/p3hpc" \
  --check paper/p3hpc/tables --output target/p3hpc-final-data
```

Python 3.11+ and the standard library suffice. The command reads archives,
verifies their hashes and evidence, checks every generated table, and writes
the detailed CSV outside tracked source. It does not invoke either engine.
Keep the archives outside Git and preserve the measured branch; final hosting
and an immutable submission tag remain author decisions after review.
