# Final P3HPC evidence

The paper and canonical `presentation/` use the October 31–November 1, 2026
collection in `~/Downloads/p3hpc/`. `artifact/final_cohort.py` imports the
measured Inferena validators and regenerates the tables. All ratios are
**Meganeura time / PyTorch time**; below one favors Meganeura. No older result
fills a missing cell.

## Identity and audit

- Inferena `a0799e4210eafa25161ec50d475b186c4b19d3da`.
- Meganeura `567028b805425375148e88a7e1d464fafdd99777`.
- Blade `eae0c6f7aa3167c79d1135050cf8e17c17fabd0a`.
- Naga `a489554c96f87cbc5252ee00350730f6e93869f9`.
- PyTorch 2.13.0, source `cf30153c4c131c8164ee7798e5022d810682e2cb`;
  CUDA 13.0, ROCm 7.2, XPU and MPS wheels; Python 3.13.13.
- Paired protocol v33, Android v15; 30-second native search per session,
  120-second reference compile allowance. Desktop warmup ten seconds,
  Android two; twenty samples and three processes for performance collection.
- Sixteen archives, 272 native executions, 14.48 recorded device-hours.
  All eight shared lockfile/model/corpus hash sets agree. Archive hashes are
  in [final-cohort.sha256](artifact/final-cohort.sha256).
- Offline replay passes 262 desktop native receipt checks, 735 valid
  paired-phase receipt checks and ten Android full-tensor comparison receipt
  checks. Desktop before/after binary checks and completed-campaign final
  checks pass. Android archives omit raw NPZ tensors: this audit verifies
  hashed manifests and full comparison coverage, not the original arithmetic.

All 330 core native cells pass (11 GPUs × 5 models × 2 profiles × 3 phases).
Six GPUs supply 180 core native timing cells and 167 qualified ratios. Larger
models add 42 native cells and 34 ratios: **222 native performance cells,
201 paired ratios**, and 21 native times without a complete passing reference.

| Archive (canonical device key) | Native cells | Ratios | Collection |
|---|---:|---:|---|
| amd-780m | 30/30 | 0 | qualification only |
| amd-7900xt | 30/30 | 28 | three processes |
| amd-7900xt-large | 6/6 | 6 | three processes |
| amd-9600x | 30/30 | 0 | qualification only |
| amd-mi300 | 30/30 | 22 | three processes |
| amd-mi300-large | 12/12 | 9 | three processes |
| apple-m3 | 30/30 | 30 | three processes |
| intel-b570 | 30/30 | 30 | three processes |
| intel-b570-large | 6/6 | 4 | three processes |
| intel-rplu | 30/30 | 0 | qualification only |
| nvidia-3050 | 30/30 | 0 | qualification only |
| nvidia-5070 | 30/30 | 30 | three processes |
| nvidia-5070-large | 6/6 | 6 | three processes |
| nvidia-h100 | 30/30 | 27 | three processes |
| nvidia-h100-large | 12/12 | 9 | three processes |
| qualcomm-adreno | 30/30 | 0 | qualification only |

The raw MI300X filenames contain `amd-mi300x`; the analyzer normalizes their
device keys to `amd-mi300` for existing tables. RTX 3050 is now checked
against CPU f32, without performance collection. Its previous GPU timings
are not retained in the new tables. The other four correctness-only devices
are the three integrated GPUs and Quest 3S; they contribute no CPU/GPU ratio.

## Results and precision

| Contract | Inference median (wins/pairs) | Minimal shape | Training |
|---|---:|---:|---:|
| Strict | 1.46 (9/30) | 0.97 (15/28) | 1.59 (4/28) |
| Accelerated | 1.67 (5/27) | 1.01 (13/27) | 1.92 (5/27) |

These are ratios of engine median process times. The main talk's combined
inference/training median is 1.60 over 112 comparisons (58 strict, 54
accelerated), geometric mean 1.66; 71/112 are below 2. This is an aggregate,
not a bound. Minimal-shape duplicates are excluded from that combined figure.

Reduced precision gives native median speedups of **1.19 inference / 1.05
training**, against PyTorch's **1.46 / 1.64**, over each engine's own
qualified strict/accelerated pairs. Matrix kernels are selected on every
timed GPU. Of 42 accelerated core sessions per GPU, counts with matrix work
are 39 RTX 5070, 36 H100, 36 RX 7900 XT, 37 MI300X, 36 B570 and 37 M3.
BF16 occurs in three RTX 5070 and five B570 sessions. Selected descriptions
request reduced derivative products in 9, 9, 8, 3, 7 and 0 sessions in that
order. Private matrix-storage choices occur in three RX 7900 XT and four
B570 sessions. Dispatch counts are neither time shares nor utilization.

The native arithmetic summary in the archived harness is partly stale: it
still labels backward as f32 and describes private storage as Winograd-only.
The selected descriptions (`gradient_inputs`, `matrix_storage`) and pinned
compiler `src/train/search.rs` establish the actual candidates. F16/BF16
AT/BT derivative products and compatible private matrix storage are enabled;
NN derivatives remain protected. Public tensors, masters, loss, accumulation
and split partials stay f32. Necessary conversions execute in each timed call.
Do not copy the generic summary's blanket backward-f32 wording into the talk.

Native complete single-step gradient fidelity and PyTorch's matched SGD
trajectory checks still have different scopes. Passing either does not
establish equal future convergence or range. No threshold was changed for
this update. No tolerance-sensitivity, search-off or alternate-budget arm
was collected.

[gap_analysis.py](artifact/gap_analysis.py) holds both engines and both
profiles to identical qualified conditions and verifies the per-cell identity
`R_acc = R_strict × (S_PyTorch / S_Meganeura)`. Geometric means preserve it:

- 27 inference conditions: 1.24 → 1.66, gains 1.62 / 1.21, multiplier 1.34.
- 25 training conditions: 1.54 → 2.21, gains 1.53 / 1.07, multiplier 1.44.

These differ from the main-table medians and are not a causal time partition.
Reduced precision helps inference, but partial backward coverage and kernel
costs limit training gains. On RTX 5070 SmolLM2, strict/accelerated native
inference is 6.37/3.61 ms and training 23.08/15.64 ms; the accelerated ratios
remain 1.23 and 1.76. Whisper's accelerated training remains 6.17× PyTorch on
RTX 5070 and 12.02× on H100. B570 U-Net strict inference is 3.21 vs 1.09 ms
(2.95×), replacing the previous deck's 5.45× result.

## Missing references

All native cells above qualify independently. A missing reference is not a
native speed win, and later unattempted phases are not additional observed
backend failures.

- RX 7900 XT strict Whisper: training-output repeatability fails in processes
  1 and 3; their later latency phases are not attempted (two aggregate ratios
  unavailable).
- MI300X strict ResNet: gradient repeatability fails in process 2; its latency
  is not attempted (two ratios unavailable).
- MI300X accelerated U-Net: process 1 passes inference/training but fails
  minimal-shape forward fidelity (max error 0.02157 vs bound 0.02085);
  processes 2/3 fail inference repeatability. Whisper fails inference output
  repeatability. Neither condition has three qualified phase pairs (six ratios).
- H100 accelerated U-Net: forward fidelity fails (three ratios unavailable).
- H100 and MI300X accelerated 1.7B: worst-token KL fails while mean KL passes;
  later phases are not attempted (six ratios unavailable). These are contract
  rejections, not evidence of a driver defect.
- B570 strict 360M: process 2 fails uncaptured gradient repeatability for
  `model.layers.8.input_layernorm.weight`; max error 1.74046e-5 exceeds
  1.59646e-5, while RMS passes. Its later latency is not attempted, so two
  aggregate ratios are unavailable. All accelerated phases pass three pairs.

B570 360M no longer aborts with the SYCL graph-event wait error. The XPU
reference uses capture-compatible dense `index_add` embedding backward,
qualified before compilation. The remaining rejection is numerical, not
capture failure. RTX 5070 accelerated 360M now passes all three phases and
processes, including its real-text trajectory gate. Neither observation
retroactively qualifies the old cohort.

## Timing, search and preparation

Spread is `(max − min) / median` of three process medians. Core engine/phase
group medians and counts exceeding 10% are:

| GPU | Median spread | Above 10% / groups |
|---|---:|---:|
| RTX 5070 | 2.06% | 0/60 |
| H100 | 0.45% | 0/57 |
| RX 7900 XT | 0.78% | 4/58 |
| MI300X | 2.37% | 1/52 |
| B570 | 0.58% | 1/60 |
| M3 | 15.72% | 43/60 |

All repeats remain. M3 had sleep prevention enabled; no clock/power trace
establishes the cause. Avoid interpreting small M3 differences as stable wins.

The 504 core sessions measure 22,112 program trials; 248 (49%) select a
non-default graph, and 396 (79%) end at the deadline. There are 46 numerical
rejections and 15 scratch-limit trials. Search receipts sum to 186.6 minutes.
Strict preparation by workload, median across GPUs of complete three-process
groups, spans 47–67 s native and 8–50 s PyTorch compilation/first calls;
reference graph preparation is recorded separately. Non-search native work
has median 1.7 s and range 0.18–70.87 s across all campaigns, so search does
not dominate every process. The longest recorded reference compile is 100 s.

Core memo-state groups are 60 first-session misses and 120 reused decisions;
all campaigns total 124 misses and 148 reused decisions. Native preparation
medians differ by 1.6–3.6 s per GPU, without a randomized cache control.
Driver/page-cache coldness is not controlled. Selected programs are not
persisted by this campaign memo; graph/program search repeats.

The separate October 19 warmup study at Meganeura `c19cb1c` remains historical.
It is not rerun or relabeled as this cohort's revision. Its two processes per
GPU follow the same program through cumulative 5/15/30/60 s checkpoints.

## Footprint

At measured Meganeura `567028b8`, [footprint.py](artifact/footprint.py) counts
54,271 Rust lines and 7,025 WGSL lines in 113 shaders, excluding blank lines,
comments and Rust test modules. The measured RTX 5070/B570 native runner is
20,387,856 bytes (19.4 MiB), SHA-256
`e88a082fa50a686f35495e4c89f7a5dea7f247f70ae712d2b924043423ed8391`;
stripping a copy gives 15,586,168 bytes (14.9 MiB). It links libc, libm and
libgcc_s. The separately measured PyTorch 2.13 CUDA dependency closure is
4.42 GiB (29 distributions); that installation-size measurement is historical,
not a fresh package-size audit of every platform.

## MI300X driver provenance

Final archives record Mesa `git-8c7f5fdd60` and Linux `6.8.0-146-generic`.
The Mesa source is the head of draft
[Mesa MR 45239](https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/45239).
The AMDGPU patch for vendor source `amdgpu-6.19.14-2364437.24.04` is attached
there as `mi300-kernel-xcc-gang-20261008.tgz`. Its guarded patch matches the
earlier `mi300-driver-patches-20261008.tgz` evidence, identified by historical
[cohort.sha256](artifact/cohort.sha256), not the new sixteen-archive manifest.
Earlier provenance records `mi300_xcc_gang=1`,
`lockup_timeout=0,60000,0,0`, and the reference MIOpen override
`MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_XDLOPS_NHWC=0`. New manifests
record the Mesa build and host kernel, without independently recording all
those module settings. Strict 1.7B training takes 34.67 ms, about 38 TFLOP/s
using `6 × 1.71e9 × 128` FLOPs, above one XCC's roughly 20 TFLOP/s f32
matrix peak: [AMD specifies 163.4 TFLOP/s for the full GPU](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html),
divided by eight XCCs.
This supports multi-XCC execution, not balanced use of all eight dies.

## Separate single-operation evidence

The October 31 study in `presentation/data/op-bench.tsv` retains three-native-process
medians and one reference process per shape for a strict-f32 GEMM or batch-1 convolution on RX 7900 XT, RTX 5070
and B570. Native source is `5b232bd` plus the tuner scratch fix included in
`567028b8`; cooperative matrices are off. PyTorch 2.13 has TF32 off and
`cudnn.benchmark` on. Shapes, reference runner and collection driver are in
`presentation/data/op-bench/`; native runner is `examples/op_bench.rs`.

Operands differ between engines. This is a throughput probe, not a
cross-engine numerical comparison. Native GPU time sums pass timestamps; PyTorch times a graph of 30 serial ops
using device events and divides by 30. Its interval includes dispatch gaps
our sum excludes, favoring native most on small ops. This cannot allocate a
fraction of the cohort model time. Host-wall columns are not cross-engine
comparisons. Reference kernels use ordinary f32 units, so the controls locate
implementation losses without a reduced-precision explanation.

4096³ GEMM ratios are 1.65/2.11/2.19 on AMD/NVIDIA/Intel; small U-Net GEMMs
span 0.66–1.31 / 1.59–4.93 / 2.01–4.62. At 64×1024×256, cuBLAS split-K takes
6 µs vs native 28, and oneDNN 12 vs 55. B570 3×3 convolutions take 1.45–3.16×
PyTorch's time, RTX 5070 0.45–1.03×. RX 7900 XT geometric means are 1.11 for
GEMMs and 0.98 for convolutions. This identifies useful kernel targets, not
an unavoidable API penalty or a measured speedup from a proposed fix.

4096³ exceeds the tuner's 64 MiB scratch and uses the default tile. 2048³ is
covered after keeping encoder allowance outside that limit, restoring B570
3.58 ms from 5.31 and RX 7900 XT 1.08 from 1.30. Processes tune for two
seconds and can pick different tiles. Native convolution times include
per-call Winograd weight transforms. The earlier U-Net shared-memory query
at `99e3de2` uses 16.4 KiB, below Vulkan's 48 KiB; that specific kernel is not
limited by the ceiling. Timelines and matched-kernel inspection remain needed
for host/driver attribution. Minimal-shape parity does not bound host cost.

DinoVision is separately pinned and unchanged. Older compiler-stage, search,
attention and warmup diagnostics are not substituted into these cohort cells.
