---
marp: true
theme: meganeura
size: 16:9
paginate: true
title: 'Meganeura: Vulkan and Metal as a Performance-Portability Layer for GPU Training and Inference'
author: Dzmitry Malyshau
description: 'P3HPC / SC26 — working draft for a 20-minute talk'
footer: 'MEGANEURA · P3HPC / SC26'
---

<!-- _class: title -->
<!-- _paginate: false -->
<!-- _footer: Working draft · Final October 2026 cohort · audited October 19 -->

# Meganeura

## Vulkan and Metal as a Performance-Portability Layer for GPU Training and Inference

Dzmitry Malyshau · Independent Researcher

[github.com/kvark/meganeura](https://github.com/kvark/meganeura)

<!--
Time: 0:20. Target: about 19:15 of material in a 20-minute speaking slot.
Nineteen main slides; eleven backups outside the speaking slot.

The long-term motivation is continual reinforcement learning in an embodied
agent. If learning stays on the device, the machine must run both the policy
and its updates within the same hardware budget. This talk asks how far
consumer graphics APIs can carry that training and inference work.

All framework performance claims use the final paired-v33 / Android-v15
cohort: Inferena add19182, Meganeura c19cb1c, PyTorch 2.13.0 at cf30153c.
The paper and this deck use the same final data. DinoVision remains a
separately versioned application study. Source: data/summary.json and
../RESULTS.md. The eleven backup slides are outside the main talk.
-->

---

<!-- _header: Why graphics APIs? -->
<!-- _class: opening -->

# One GPU

GPUs grew up rendering games.

> Why does learning need a separate software stack?

The application already has a GPU, a driver, buffers and a queue.

**Can Vulkan + Metal carry inference and training too?**

<!--
Time: 0:45.
Open with the deployment problem. A game, robot or headset already uses a
graphics API. Adding learning often brings another runtime, compiler and
vendor-specific set of libraries. Those stacks offer powerful features and
tuned kernels; the question is how much of that work the existing graphics
stack can do, and what performance we give up.
The continual-RL motivation can be one spoken sentence here; its loop stays
in backup. This experiment measures runtime capability, not an RL algorithm.
-->


---

<!-- _header: 06 + 08 · native GPU access -->
<!-- _class: blade -->

# Blade

**A rendering library supplies the compute runtime.**

| Backend | Native platforms | GPU families |
|---|---|---|
| Vulkan | Linux · Windows · Android | NVIDIA · AMD · Intel · Qualcomm · others |
| Metal | macOS · iOS · tvOS | Apple devices |

Vulkan + Metal cover the major modern native platforms. Device features and driver support set the floor.

**WGSL → Naga → SPIR-V / MSL → driver**

One Blade context can own the ML buffers, compute pipelines and renderer.

<small>Naga is the shader compiler; wgpu's runtime is not used. Some matrix/BF16 features are carried in our Naga fork and are not all upstream at the measured revision.</small>

<!--
Time: 1:00.
Blade is a thin Rust GPU abstraction used for both rendering and compute.
It handles buffers, pipelines, command recording and synchronization.
Bindings follow named Rust shader-data fields; pass-level barriers avoid a
per-resource state tracker. Meganeura keeps the model graph and scheduling
policy above this layer. No CUDA/HIP/XPU runtime is linked into the native runner.

This is API/platform reach, not a claim that every device on those platforms
was tested. Modern native systems are broadly covered; very old GPUs and
drivers without the required compute features are outside the target.
Apple mobile platforms are Blade targets, not additional measured platforms
in this cohort. Experimental MI300X support required Mesa and kernel changes.

The measured pins are Blade 4845d6529c38b07718a624bf3c4f2ab4317c9bf4 and
Naga 13f582ddd2792befb46952498010248bd65818e6. The fork includes rectangular
cooperative matrices, BF16 and workgroup-scope matrices. Vulkan goes through
SPIR-V; Metal through MSL, including simdgroup matrices where available.
Sources:
https://github.com/kvark/blade/blob/4845d6529c38b07718a624bf3c4f2ab4317c9bf4/docs/README.md
https://github.com/kvark/wgpu/commit/13f582ddd2792befb46952498010248bd65818e6
https://github.com/kvark/inferena/blob/add19182e5479e3f7549894c524ac6691d3dc363/Cargo.lock
-->


---

<!-- _header: Mechanism · the route through the talk -->
<!-- _class: chart -->

# Inside Meganeura

![Eight checkpoints: describe, rewrite, differentiate, plan, specialize, compile, qualify and time, execute](figures/compiler-map.svg)

One compiler path for inference and training. **Measured search revisits the choices on the target GPU.**

<!--
Time: 1:35.
Follow the numbered track across the top, then right to left along the bottom.
The route connects graph semantics, GPU kernels and application execution.

01: Native Rust builders and ONNX/NNEF importers produce a typed static graph.
GGUF supplies an architecture and weights; its loader constructs the graph.
Primitive ops define semantics; composites have decompositions; private ops
represent compiler implementations. A model does not have to specify a GPU.

02–03: egglog holds equivalent forward expressions and matrix schedules.
Extract forward alternatives before differentiating each one; then explore
backward layouts. Each derivative must match its own forward graph. Inference
skips autodiff. Ordinary build can also optimize the combined training graph;
measured search preserves the alternatives it is comparing.

04–06: Lower and fuse ops, plan lifetimes, choose legal kernels for shapes,
capabilities and precision. WGSL templates/generators feed Naga; Blade builds
Vulkan or Metal pipelines. Matrix extensions expose different choices on each
device. This uses Naga, not the wgpu runtime.

07: Build initialized candidates, check full tensors against the original
strict-f32 build, tune kernels, then race whole programs. The loop also varies
submission count. Precision changes are gated candidates, not exact equalities.
The diagram groups these tasks; code generation and kernel tuning can rebuild
pipelines within a candidate. Search is optional.

08: Reuse the selected dispatch and memory plan. Each step still records,
submits and synchronizes; it does not rediscover the graph or allocate tensors.
An application can use the same Blade context and queue for rendering.

Two checks surround this route. Compiler tests use an independent CPU f64
interpreter, finite differences, full kernel outputs and randomized graphs.
Inferena qualifies the selected programs and the PyTorch baseline before
held-out timing. The numbered headers return to these checkpoints later.
Source: ../main.tex, How Meganeura Works; ../../../src/graph/composite.rs;
../../../docs/testing.md. Measured engine: c19cb1c.
-->

---

<!-- _header: 02–03 · graph and gradient choices -->

# Two compiler choices

<div class="columns examples">
<div>

### Pack the projections

`Q = XWq`, `K = XWk`, `V = XWv`

↕

`[Q K V] = X[Wq Wk Wv]`

Three products or one wider product.

</div>
<div>

### Split the reduction

`A[50,4096] × B[4096,720]`

↕

Eight partial products, each **K = 512**, then sum them.

More parallel work; an extra reduction.

</div>
</div>

Time both forms, including the reduction. Keep a faster candidate only after numerical checks.

<!--
Time: 1:05.

Give the concrete example before explaining the policy: three attention
projections can remain independent or become one packed matrix product.
For the second example, A is 50 by 4096 and B is 4096 by 720, one of the
paper's matrix-study shapes. A single output tile may leave too little
parallel work. Eight-way split-K computes eight partial 50 by 720 outputs,
each using 512 elements of the reduction dimension, then sums them.
This adds temporary storage and a reduction dispatch. Measure that complete
sequence, because a faster matrix kernel alone can still lose overall.

Bounded saturation keeps the search manageable. Small graphs are explored
whole; larger ones can use a verified repeated region such as a transformer
block. The ordinary extracted program is the incumbent. Search tunes
candidate kernels and compares whole programs with fresh command recording
and submission. Precision is a constraint, not an interchangeable rewrite.

The measured policy allows 16 graph/schedule forms and 64 whole programs
within a soft 30-second budget. The 30 seconds are per session, not total preparation for a workload. A
workload needs two or three sessions. The next slides show the available kernels and the cost of choosing them. An equally capable enumerator could find the same schedules;
the experiment does not prove an inherent advantage of e-graphs.

Source: ../main.tex, How Meganeura Works and Compile-Time Search.
-->

---

<!-- _header: 05–06 · what the API and compiler expose -->
<!-- _class: hardware -->

# Hardware access

| Hardware feature | Vulkan route¹ | Measured stack |
|---|---|---|
| Matrix multiply | KHR cooperative_matrix | Used on NVIDIA + AMD |
| BF16 matrices | KHR shader_bfloat16 | NVIDIA / Intel expose it; RADV here does not |
| Workgroup-wide tiles | NV cooperative_matrix2 | Used on NVIDIA |
| Tensor loads + matrix reductions | NV cooperative_matrix2 | RTX 5070 driver supports them; compiler does not |
| FP8 matrices | EXT shader_float8 | RTX 5070 driver supports them; compiler does not |

<small>¹ Extension names omit VK_. FP16 scalars, subgroup operations and packed int8 dots are also available where supported.</small>

<!--
Time: 1:15.
Separate four questions: can the hardware do it, does the driver expose it,
does our compiler support it, and did search choose it? They are not equivalent.
KHR cooperative matrices give Vulkan a portable abstraction for matrix units.
The measured compiler handles f16, f32 and BF16 tiles as advertised, including
Intel's rectangular 8x16x16 shape. Intel's selected-program result follows later.
NVIDIA workgroup-scope matrices and flexible dimensions are implemented and
selected. Blade enables those two matrix2 features, not tensor addressing,
block loads, matrix reductions or accumulator conversions. The October 15
read-only RTX 5070 driver query reports those features and FP8 support.
They are concrete compiler opportunities, not missing Vulkan extensions.
RADV on the measured RX 7900 XT and MI300X exposes no BF16 cooperative shape.

Do not use TF32 as an explanation for this cohort's gap. It is prohibited by
the strict policy and disabled for the accelerated baseline's remaining f32
operations. Faster lower-precision types are already used in accelerated.
Its possible role is an intermediate accuracy/range choice, discussed in backup.
No one-to-one hardware instruction match is established by these API queries.

Sources: data/hardware.json; Blade 4845d65 vulkan/init.rs:1373–1384;
https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_cooperative_matrix.html
https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_shader_bfloat16.html
https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_cooperative_matrix2.html
https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_shader_float8.html
-->


---

<!-- _header: 05 · the same silicon through two APIs -->
<!-- _class: limits -->

# API limits

### Shared memory per workgroup

| GPU | Vendor compute API | Vulkan |
|---|---:|---:|
| RTX 5070 | CUDA: **99 KiB** with opt-in | **48 KiB** |
| Arc B570 | Level Zero: **128 KiB** | **48 KiB** |

Both API pairs allow **1,024 threads** per workgroup.

**A CUDA kernel's tile and staging buffers may not fit through Vulkan.**

<small>NVIDIA 595.91.07 · Intel Mesa 26.0.3 / Level Zero 1.14.37020. Reported limits alone do not explain the timing gap.</small>

<!--
Time: 0:50.
The CUDA number is the opt-in maximum, not its default 48 KiB allocation.
101,376 bytes is 99 KiB; Vulkan reports 49,152 bytes. B570 reports 131,072
bytes through PyTorch XPU's Level Zero backend and 49,152 through Vulkan.
The native workgroup-matrix extension can reserve some shared memory for
compiler staging, so the reported limit is not all necessarily usable by
shader-declared arrays. Kernel feasibility also depends on registers,
occupancy, matrix shapes and workgroup size. More allowed shared memory is
an opportunity, not proof that a larger tile is always faster.
These values are device/driver observations, not permanent Vulkan API caps.
Sources: data/hardware.json. Vulkan queries October 15; CUDA/XPU queries
rechecked October 19 without a performance run.
https://docs.nvidia.com/cuda/archive/13.0.3/blackwell-tuning-guide/index.html
-->


---

<!-- _header: 07 · choosing a program -->

# Search on the GPU

<div class="search-steps">
<div><b>1</b><strong>Build</strong><span>Graph, layouts, pipelines<br>Initialize weights + inputs</span></div>
<div><b>2</b><strong>Check</strong><span>Full outputs + gradients<br>Original strict-f32 reference</span></div>
<div><b>3</b><strong>Tune</strong><span>Race kernel implementations<br>Reuse saved kernel decisions</span></div>
<div><b>4</b><strong>Compare</strong><span>Time the whole program<br>Keep a qualified improvement</span></div>
</div>

**30 s per session** · up to **16 graph forms** and **64 programs**.

Search varies layouts, fusion, matrix inputs and submission count. Each workload builds two or three sessions.

<small>Soft deadline; oracle preparation is outside it. Kernel decisions persist; whole-program search runs on each build.</small>

<!--
Time: 0:55.
Checkpoint 07 revisits the compiler choices. Candidate construction and
initialization precede full-output/gradient qualification. Legal kernel
variants are tuned inside the candidate before the complete-program race.
Additional checks follow tuning and selection; this is a schematic sequence.
Private kernels need 5% improvement; whole programs need 1% plus the measured
noise margin. Reject numerically invalid candidates before using their time.

Submission candidates range up to 64 chunks, under the same measurement
policy as graph layouts and fusion. These timings measure the ML graph alone;
the headset application later supplies a different scheduling objective.
Whisper builds two sessions, the other models three. Thirty seconds is a soft
search deadline per session, not an end-to-end setup bound. Initialization,
numerical checks and measurements consume time within that search; preparing
the original-f32 reference is outside it but inside native preparation.

Core receipts: 588 sessions, 23,669 trials, 13 numerical rejections and
34 scratch-limit outcomes. Kernel probes have a 1 GiB scratch cap. A scratch
limit is a resource outcome, not a numerical failure. Persisted kernel-class
decisions are scoped by campaign/model/profile; complete plans are searched
again. There is no search-off arm in this cohort, so no measured overall
search speedup is claimed. Source: ../tables/search.tex; data/summary.json;
../main.tex, Compile-Time Search.
-->

---

<!-- _header: Experiment -->

# The comparison

<div class="columns experiment"><div>

### Five workloads

- SmolLM2-135M
- SmolVLA action expert
- Scaled SD U-Net
- ResNet-50
- Whisper-tiny encoder

</div><div>

### Seven measured GPUs

- NVIDIA RTX 5070, H100, RTX 3050
- AMD RX 7900 XT, **MI300X**
- Intel Arc B570
- Apple M3

### Four correctness-only GPUs

Radeon 780M · Ryzen 9600X iGPU<br>
Intel RPL-U · Quest 3S / Adreno 740

</div></div>

**PyTorch 2.13:** default `torch.compile`; CUDA / HIP / XPU graph replay. MPS uses compiled calls. **Two profiles: strict and accelerated.**

<!--
Time: 0:55.
The baseline avoids per-operator Python dispatch where graph replay is available.
Max-autotune is off; compilation has a supervised 120-second limit. Native
search has 30 seconds per session, so this is not an equal-preparation-budget
comparison. The workloads and fixed shapes are identical within each pair.
Only SmolLM2 uses trained checkpoint weights; the others use deterministic
uniform parameters. Timed inputs are fixed; the separate AMP training gate
uses real text for all SmolLM2 sizes. RTX 5070 and B570 share a host, and the
Arc is on PCIe 3.0 x1. Eleven GPUs total, five vendors; seven timed GPUs,
four vendors. Source: ../main.tex, Methodology; ../tables/devices.tex.
-->

---

<!-- _header: 07 · correctness, then Inferena timing -->

# Check, then time

**Qualify → 10 s warmup → 20 synchronized samples → 3 process pairs**

- Meganeura: full outputs and gradients against its original **strict-f32 graph**.
- PyTorch AMP: f32 forward fidelity, repeatability, and **matched SGD trajectories**.
- Across engines: sampled outputs, loss, and gradient norms from f32 references.

**These checks have different scopes.** Wider gates could admit faster kernels and tilings in either engine.

<small>Timed training = forward + loss + backward. No optimizer update. Native selected programs do not undergo PyTorch's 256-step real-text gate.</small>

<!--
Time: 1:05.
Report the median of three process medians, alternating engine order. Warmup
is at least five calls and ten seconds; Android correctness-only runs use two
seconds and one sample, with no published timing. Upload/readback are untimed.
Native candidates are checked before selection and the selected program has
two final qualification executions. PyTorch uses full repeatability and
capture checks plus AMP fidelity: 256 matched real-text SGD updates for
SmolLM2, 32 repeated-input updates otherwise. All trainable parameters update
in that untimed gate; state is restored before timing. Native instead gates
single-step full gradients, so do not claim equivalent convergence evidence.
Accelerated cross-engine comparisons use preserved strict-f32 snapshots,
not the AMP candidate as ground truth. Rejected references are not wins.
Changing gates might change native search winners as well as Torch eligibility;
we have no tolerance-sensitivity experiment. Source: ../main.tex, Numerical gates.
-->

---

<!-- _header: Performance · strict f32 inference -->
<!-- _class: chart -->

# Inference by GPU

![Strict inference time ratios for seven GPUs and five workloads](figures/strict-inference.svg)

**Median 1.35×.** MI300X uses patched Vulkan support. B570 SD U-Net: **5.89 / 1.08 ms**; its native timing varies.

<!--
Time: 1:10.
Teal means a ratio below one, orange above one; colors are categorical,
not a continuous scale. RX 7900 XT SmolLM2 prefill is 5.56 versus 9.17 ms,
a 0.61 ratio. RTX 5070 SmolLM2 is near parity at 1.04. B570 loses all core
paired phases. Its rectangular matrix tiles are supported, but selected
B570 programs have no cooperative dispatches. SD U-Net strict inference
retains all three native process medians: 5.894, 6.128 and 4.883 ms; Torch
is 1.082, 1.082 and 1.084 ms. The 5.45 ratio is present in the raw receipts,
not a plotting error. Search timings and held-out timings differ substantially;
this is a performance issue to investigate, not a reason to replace the result.
An October 19 follow-up kept one selected strict SD U-Net program fixed:
inference took 4.438 / 5.472 / 5.786 / 5.766 ms after 5 / 15 / 30 / 60 seconds,
while sampled GT clocks fell from 1750 to 1400 to 1300 MHz. All checkpoints
use the same plan hash and one submission; no throttle flag was set near the
sample windows. This points to clock policy as a factor, not solely search
selection. It does not prove why clocks changed or explain the full historical
regression. One diagnostic process, core c19cb1c, study runner b110188; it
passes the original-f32 full-tensor gates and does not replace cohort values.
M3 has substantial variation in both engines, so a ratio near one is inconclusive.
Source: data/results.csv; data/summary.json.
-->

---

<!-- _header: Performance · strict f32 forward + loss + backward -->
<!-- _class: chart -->

# Training by GPU

![Strict training time ratios; two unqualified reference cells are marked with dashes](figures/strict-training.svg)

**Median 1.61×; 4 / 33 nominal native wins.** A numerical rejection supplies no speed ratio.

<!--
Time: 0:50.
There are four nominal native wins out of 33 comparisons. Missing strict
Whisper on RX 7900 XT and ResNet on MI300X are PyTorch repeatability-gate
failures. Native remains independently qualified. These failures do not
prove a vendor bug or unusable accuracy under another application's policy.
The higher-level training work is forward, loss and backward only: no
optimizer update in the timing. Host-wall times include recording,
submission, synchronization and kernels; these ratios do not separate them.
Source: data/results.csv; ../RESULTS.md.
-->

---

<!-- _header: 05 · arithmetic choices -->

# Accelerated arithmetic

| Core median time ratio | Strict f32 | Accelerated |
|---|---:|---:|
| Inference | **1.35×** · 35 pairs | **2.06×** · 32 pairs |
| Forward + loss + backward | **1.61×** · 33 pairs | **2.21×** · 32 pairs |

**PyTorch:** BF16 autocast on CUDA / HIP / XPU; FP16 on MPS. Backward included.
**Meganeura:** qualified f32 / f16 / BF16 matrix choices where supported; f32 storage and derivatives.

<small>TF32 is disabled. The profiles have different qualified subsets and arithmetic paths; their gaps do not isolate matrix-instruction efficiency.</small>

<!--
Time: 1:00.
Only two profiles. CUDA/HIP/XPU auto selects BF16 when hardware-supported,
otherwise FP16; MPS selects FP16. SmolLM2-135M one-token latency is a declared
f32 exception. Larger language models retain autocast. No dtype retry follows
numerical rejection. Remaining f32 matmuls have TF32 disabled in this cohort.
FP16 GradScaler housekeeping is timed, with no optimizer update. Meganeura
keeps f32 tensors and derivative regions; conversions happen inside each step.
On RTX 5070, SmolLM2 prefill is 6.28/6.07 ms native/Torch strict and 5.55/2.96
accelerated. Native improves but Torch improves more. Missing references mean
the aggregate strict/accelerated sets differ; do not present their ratio as
a matched precision speedup. Source: ../main.tex, Arithmetic contracts.
-->

---

<!-- _header: 05 + 07 · capability versus selection -->
<!-- _class: paths -->

# Selected paths

| GPU | Selected kernels | Matrix sessions¹ |
|---|---|---:|
| RTX 5070 / H100 / RTX 3050 | Scalar/vector + workgroup matrices | **24 / 21 / 24** |
| RX 7900 XT | Scalar/vector + f16 matrices | **22** |
| MI300X | Scalar/vector + matrix tiles | **30** |
| Arc B570 | Scalar/vector only | **0** |
| Apple M3 | Scalar/vector + simdgroup matrices | **30** |

NVIDIA selects f16 workgroup kernels. **B570's rectangular f16/BF16 path is available but never selected here.**

<small>¹ Out of 42 accelerated core sessions per GPU. BF16 selected in 3 sessions each on 5070 and 3050, zero elsewhere. Programs can mix kernel families.</small>

<!--
Time: 1:00.
Each device has 42 accelerated sessions: three repeats times three sessions
for four models and two for Whisper. The count means a selected program
contains at least one cooperative dispatch; it is not the fraction of FLOPs
or time on matrix units. The remaining programs use ordinary scalar/vector
and subgroup kernels. Within one selected program, both paths can coexist.

This table pairs advertised capability with selected-program evidence. It
does not assign every dispatch to a dtype without inspecting the program.
MI300X has f32 and f16 shapes. M3 reports f32/f16 and BF16 capabilities, but
these selected sessions have zero BF16 dispatches. B570's general f16 tile
field is zero because it describes square tiles; the compiler's rectangular
path is supported, yet selected cooperative dispatches are still zero.
NVIDIA selected kernel memo entries include f16 CooperativeWorkgroup choices
on 5070, H100 and 3050; none of that proves a per-feature speedup without an
ablation. BF16 dispatch counts are explicit in session receipts.
Sources: data/summary.json accelerated_selection; data/hardware.json;
final-cohort selected-session and kernel-memo receipts, pinned c19cb1c.
-->


---

<!-- _header: Performance · SmolLM2 -->
<!-- _class: chart -->

# Larger models

![Strict SmolLM2 prefill and training ratios on H100 and MI300X, including process ranges](figures/scaling.svg)

At 1.7B: native strict training **135.85 ms on H100**, **35.88 ms on MI300X**. Accelerated 1.7B has no qualified PyTorch ratio.

<!--
Time: 1:00.
The H100 strict training gap grows from 1.90 at 135M to 3.79 at 1.7B;
MI300X's narrows from 2.29 to 1.65. Each server's sizes share
recorded GPU identity, driver and host kernel. The chart's range
is the three paired process ratios, not a confidence interval. This is batch
one with 128-token prefill and one-token stateless forward, not production
language-model training or decoding.
Accelerated 1.7B fails PyTorch's worst-token KL gate on both servers although
mean KL passes. Native times remain qualified, but no paired ratios are
manufactured. RTX 5070 accelerated 360M training also lacks a ratio because
two of three real-text trajectories fail; keeping only the passing repeat
would violate the protocol. Source: ../tables/scaling.tex; ../RESULTS.md.
-->

---

<!-- _header: 07 · setup cost -->
<!-- _class: chart -->

# Preparation time

![Preparation by workload: Meganeura search and checks versus PyTorch compilation and first specializations](figures/preparation-models.svg)

Search uses most of Meganeura's setup time. **PyTorch's compilation has a separate 120 s limit.**

<!--
Time: 0:55.
These are seconds per workload process, across its requested phases.
Use strict f32 and the same fully completed three-process pairs for each
engine: seven GPUs for SmolLM2, SmolVLA and SD U-Net; six for ResNet and
Whisper, whose missing references stay excluded from both columns.
Compute a median of three process values on each GPU, then the median
across those GPUs. This is a description of the measured policies, with
their recorded cache states, not a controlled cold-cache compiler contest.

Meganeura preparation includes graph work, differentiation, pipeline creation,
initialization, the original-f32 oracle, full-tensor checks and search.
PyTorch compile_s covers torch.compile and first specializations of all
requested phases, including backward. It excludes subsequent graph capture
and qualification. Those separately recorded phase-preparation medians are
2.65 / 2.01 / 0.34 / 0.72 / 0.42 seconds respectively; they are not stacked
here because the scopes differ and medians cannot generally be added.
Both displayed counters exclude process startup, model loading and held-out
warmup. They are not full campaign duration or equal-work compiler timings.

Native preparation medians are 64.95 / 65.08 / 68.10 / 49.48 / 55.11 seconds;
PyTorch compilation: 54.21 / 26.48 / 29.25 / 13.59 / 12.49 seconds.
The corresponding native search medians are 60.01 / 62.02 / 67.15 / 47.68 /
54.60 seconds. Source: data/preparation.csv, generated directly from
the final archive records by ../artifact/final_cohort.py.
-->

---

<!-- _header: 08 · execution in an application -->
<!-- _class: deployment -->

# Sharing the GPU

![bg right:32% contain](../../figures/dinovision-quest3s.jpg)

**DinoVision · Quest 3S · Android / Adreno**


Two eye workers + renderer, **one context and queue**.

| Measurement | 1 chunk | 2 chunks |
|---|---:|---:|
| ML alone · ms/pass | 96.78 | 100.79 |
| Live · render submissions/s | 10.83 | 12.87 |
| Live · worker latency, ms | 149.51 | 119.25 |

Two chunks cost more alone. In the live run, worker latency was lower; **the cause was not traced**.

<small>Live test: 90 seconds per setting. Render submissions/s is the app's submission rate, not display FPS. Image: headset reconstruction.</small>

<!--
Time: 1:25.

Return to the motivating application. A DINOv3 encoder feeds a decoder
trained with Adam on a desktop GPU. The headset runs a joined encoder/decoder
graph, with two asynchronous eye sessions sharing Blade's Vulkan context
and queue with the OpenXR renderer. The decoder has 2.01 million parameters;
the deployed graph performs 2.359 billion multiply-accumulates.

This is a numerical deployment check as well as an integration story. With
one fixed public RGB frame and identical weights, preprocessed patches are
bit-exact. The encoder and reconstruction relative L2 differences are
0.001059 and 0.000457. The displayed image is the reconstruction, not an
input/output quality comparison.

In isolation, one submission takes 96.78 ms. Two chunks cost 4.1 percent.
Here isolation means the joined ML graph in the native benchmark, without
the renderer and the other eye worker sharing its queue. The unit is synchronized
milliseconds per inference pass. The one-chunk baseline is 96.78 ms; two
chunks take 100.79 ms.

The live test is a separate 90-second run per setting. Relative to its
one-chunk baseline, application render submissions rise from 10.83 to 12.87
per second and median worker latency falls from 149.51 to 119.25 ms. Worker
latency starts after the worker receives a frame and includes input upload,
step recording/submission, GPU wait, output readback and CPU planar-to-RGB
conversion. It excludes time before the worker receives the frame and
presentation afterward; it is not camera-to-display latency. All
percentage changes use the one-chunk value from the same test as denominator.
Inside a ninety-second live stereo session, two chunks improve render
submission count, slower-eye update rate (8.7 percent), and median worker
latency. Do not relabel render submissions as displayed frame rate. Larger
chunk counts help less and unevenly. The observed application result differs
from the isolated graph result.

Why can this matter with one shared queue? Sharing a context avoids context
switching but does not eliminate queueing or resource contention. The pinned
Meganeura step records chunk A, submits it, then records and submits B. Blade
holds its queue mutex only during each submit, so a renderer or another eye
can submit between them. Blade also emits conservative command-buffer-boundary
barriers. A renderer queued after a long compute submission can therefore wait
behind that work even though it shares the context.

This is an opportunity to change order, not a deliberate yield: there is no
wait between chunks and no guarantee another thread submits in between. If
all chunks are queued consecutively, splitting alone need not improve latency.
The archive has aggregate timing and device-state windows, not a queue/GPU
trace. It does not establish whether interleaving, CPU/GPU overlap, readback
queueing or another scheduling effect caused the 20.2% worker-latency change.
Do not turn that observation into a causal claim or a general recommendation. This is deployment inference, not headset training
or a comparison against PyTorch on Android.

Source: ../main.tex, Deployment Beyond the Matrix; exact absolute values
from ../../dinovision-section.tex, submission-chunking table;
../../figures/dinovision-quest3s.jpg. The paper links the pinned public
DinoVision source and evidence archive. Source inspection: DinoVision
dc35cdf/src/inference.rs:302–352; Meganeura d44803d/src/runtime.rs:3989–4049;
Blade 60c2eeb/blade-graphics/src/vulkan/mod.rs:676 and command.rs:355.
Pinned links:
https://github.com/kvark/dinovision/blob/dc35cdf1c7c910cdd93c5b5362846842ae469a21/src/inference.rs#L302
https://github.com/kvark/meganeura/blob/d44803d77e6f8bf777a6540c0128a314dfbf60a9/src/runtime.rs#L3989
https://github.com/kvark/blade/blob/60c2eeb8d6180ff2db5167e304d55b78a18dfee2/blade-graphics/src/vulkan/mod.rs#L676
-->

---

<!-- _header: Conclusions · where the remaining work sits -->
<!-- _class: explanation -->

# Why the gap?

<div class="factors">
<div><h3>API access</h3><p>Smaller workgroup storage on NVIDIA + Intel.</p><p>BF16 absent from the measured RADV matrix path.</p></div>
<div><h3>Our implementation</h3><p>Most tuning effort: AMD + NVIDIA, especially inference.</p><p>Intel arrived late; server access is scarce. Training has had less attention.</p></div>
<div><h3>Vendor software</h3><p>Compute libraries, shader compilers and drivers all affect the result.</p><p>MI300X needed driver patches. Exposed features can remain unused.</p></div>
</div>

**Next targets:** use the available matrix features, improve backward kernels, and profile the driver path.

<small>Development effort is author context. The cohort measures whole stacks; it does not apportion the gap among these factors.</small>

<!--
Time: 1:00.
Offer an explanation of the engineering problem, without pretending the
benchmark decomposes causes. API limits are measured facts. The distribution
of development effort is the author's account: AMD and NVIDIA consumer
hardware were available throughout; Intel came later, servers were intermittent,
and inference received more tuning than backward/optimizer work. This makes
untuned kernels a plausible factor, not an experimentally measured fraction.

The comparison includes Inductor and vendor libraries on one side, our
compiler and Vulkan/Metal drivers on the other. Driver code generation and
recording costs can matter, as can vendor libraries' kernel coverage. A slow
ratio alone cannot identify a driver fault. MI300X's patched Mesa/AMDGPU setup
establishes access; it does not isolate a driver performance contribution.
Similarly, B570's zero selected matrix sessions is a concrete implementation
target. NVIDIA tensor loads and matrix reductions are available but not used.
M3 timing variation and different numerical gates further limit attribution.
Sources: data/hardware.json, data/summary.json, cohort receipts;
author's development history; measured stack pins in README.md.
-->


---

<!-- _header: Conclusions · ML through Vulkan + Metal -->
<!-- _class: closing conclusions -->

# What works

**Mobile to server.** Five workloads qualify on **11 GPUs**: all **330 core cells** pass, including Quest 3S and MI300X.

**Small deployment.** One native executable + model weights. The measured Linux runner is **19.1 MiB**; it uses the system GPU driver.

**The full training cycle.** Forward → loss → backward → optimizer → save → load → infer. DinoVision trains with Adam on desktop Vulkan and deploys on the headset.

**Vulkan + Metal can carry the learning stack. Performance is the unfinished work.**

[github.com/kvark/meganeura](https://github.com/kvark/meganeura)

<!--
Time: 1:10.
Answer the opening question. A separate vendor compute stack is not required
for this demonstrated training and deployment workflow. One compiler handles
the graph, derivatives, memory plan and device-specific kernels. Blade keeps
the result inside a native application and its graphics stack.

Coverage means the stated numerical gates on eleven tested GPUs, not a proof
for every GPU or every model. MI300X used experimental driver/kernel support.
Seven GPUs have comparative timings; four have correctness evidence only.
Core strict median ratios are 1.35 inference and 1.61 forward-loss-backward;
accelerated ratios are 2.06 and 2.21 on their qualified subsets. Broad access
has been demonstrated; uniform speed parity has not.

The frozen Linux Inferena runner is 20,072,584 bytes (19.14 MiB), SHA-256
b8d25fac4992dd3cace578a81ec54e60bd543cb4ae1cec8b964ba9dff0fc258e.
This is a concrete executable footprint, not model size, process memory,
or an SDK installation comparison. It includes its native ML/compiler code;
system libc and a Vulkan loader/driver remain required. No Python or vendor
compute SDK is needed to deploy that executable. Other target builds vary.

DinoVision supplies full-cycle evidence: the same project trains its 2.01M
decoder through autodiff and Adam on an NVIDIA Vulkan host, serializes weights,
and loads it into the Quest encoder/decoder graph. This is not headset training.
The timed framework training comparison excludes optimizer updates; the
application study establishes functionality, not a training-throughput win.
Continual RL remains the motivation and a future application experiment.
Sources: data/summary.json, data/hardware.json;
https://github.com/kvark/dinovision/blob/dc35cdf1c7c910cdd93c5b5362846842ae469a21/REPORT.md
-->


---

<!-- _header: Backup · Performance · strict f32 -->
<!-- _class: headline -->

# Overall results

<div class="metrics">
<div><strong>1.35×</strong><span>Inference</span><small>10 / 35 nominal native wins</small></div>
<div><strong>1.03×</strong><span>Minimal shape</span><small>15 / 33 nominal native wins</small></div>
<div><strong>1.61×</strong><span>Forward + loss + backward</span><small>4 / 33 nominal native wins</small></div>
</div>

Median **Meganeura / PyTorch time** over qualified core comparisons. Lower is better; 1× is parity. Missing references stay missing.

<!--
Backup.
These are time ratios, not speedups, computed from unrounded medians. Each
cell has three independently qualified process pairs. The core matrix has
35 possible comparisons per phase/profile. Strict Whisper on 7900 XT and
ResNet on MI300X lack training and later minimal-shape references, leaving 33.
Minimal shape is not serving latency: SmolLM2 uses one token without KV cache.
Diffusion and Whisper repeat their full shapes. Nominal wins are descriptive,
not significance claims; M3 is especially variable. Across core plus large
models, 246 native measured cells qualify and 226 have paired ratios.
Source: data/summary.json, strict; data/results.csv.
-->

---

<!-- _header: Backup · Coverage -->

# Coverage

| Platform extension | Final native evidence | Comparison limit |
|---|---|---|
| MI300X | Core + 360M + 1.7B qualify | Experimental Mesa + AMDGPU |
| Radeon 780M, 9600X iGPU, RPL-U | **90 / 90** phase cells | CPU f32 correctness only |
| Quest 3S / Adreno 740 | **30 / 30** phase cells | CPU f64 full-tensor oracle |
| B570 / SmolLM2-360M | Internal execution checks only | XPU capture fails; incomplete |

**330 / 330 core native cells qualify.** No CPU/GPU speed ratios; no speed wins assigned to failed references.

<!--
Backup.
Seven measured GPUs plus four correctness-only GPUs. Across core and completed
larger models, 246 native measured cells qualify, with 226 paired ratios.
Twenty lack a fully qualified PyTorch comparison. Six requested B570 360M
cells are incomplete and excluded; its one native strict process has no
independent oracle and no three-process replication.
B570 fails below an Inductor-generated operator during XPU/SYCL graph capture:
wait cannot be used for an event associated with a command graph. This is an
execution compatibility issue; our harness also fails to classify it and
aborts. It does not diagnose a particular driver bug or all execution modes.
By contrast, the other missing references are numerical-contract failures
or subsequent phases not attempted. Do not conflate these failure classes.
Android archives retain comparison receipts and hashed manifests, not raw
NPZ tensors. Source: ../QUALIFICATION.md; ../RESULTS.md.
-->

---

<!-- _header: Backup · Measurement uncertainty -->
<!-- _class: chart -->

# Timing variability

![Between-process spread of qualified native and PyTorch times on the seven measured GPUs](figures/timing-spread.svg)

M3: **59 / 60** phase groups exceed 10% spread; median **33.84%**, despite sleep prevention. Every repeat remains in the results.

<!--
Backup.
Each point describes three process medians, not the 20 within-process samples.
Both engines move on M3. Sleep prevention was active; no clock/power trace
establishes the cause. Do not claim small M3 wins. B570 also has seven of
60 groups above 10 percent. H100 has one isolated native accelerated diffusion
minimal-shape outlier: 1.773, 5.310, 1.772 ms, despite the same graph/chunks.
Its reference already failed, so it cannot affect a paired speed ratio.
Longer warmup is not automatically a cure. The small two-GPU SmolLM2 study
in backup found no benefit from 60 versus 5 seconds, after search had already
heated the device. Source: data/results.csv; ../RESULTS.md.
-->

---

<!-- _header: Backup · numerical policy -->
<!-- _class: backup gates -->

# What the gates admit

| Check | Frozen bound / scope |
|---|---|
| Native strict fidelity | Full tensors: max and RMS ≤ 10⁻⁶ + 10⁻⁴ · reference scale |
| Native accelerated fidelity | Non-LM outputs: 2%; whole training gradient: 1% |
| Accelerated LM inference | Mean KL ≤ log(1.01); worst token ≤ log(1.10); mean loss within 2% + 10⁻⁶ |
| PyTorch accelerated training | 256 real-text SGD updates for SmolLM2; 32 repeated-input updates otherwise; 2% + 10⁻⁶ loss bounds |
| Cross-engine f32 snapshots | Sampled outputs/loss: 1%; gradient-norm vector: 5% strict, 10% per accelerated process and <5% median |

**Wider bounds could change both engines' results.** Native single-step fidelity and PyTorch trajectory checks are not interchangeable.

<!--
Backup. The stated percentages abbreviate scale-relative max/RMS bounds,
not a per-element relative tolerance. The native arithmetic oracle is prepared
from the original graph before precision conversion. PyTorch's forward
oracle is eager strict f32. Native search may retain an f32 program; PyTorch
has a predetermined dtype policy and no post-rejection dtype search.
Repeatability is separate: full outputs use 1e-6 + 1e-4 reference scale;
accelerated whole gradients use 1%. Neither engine's repeatability is an
independent accuracy proof. The cohort provides no identical native 256-step
trajectory check and no tolerance-sensitivity experiment. Offline relabeling
of failures cannot recover kernels that a new search could select.
Source: ../main.tex, Numerical gates; frozen Inferena qualification code.
-->

---

<!-- _header: Backup · pinned shader compiler -->
<!-- _class: backup precision -->

# SPIR-V precision

| Mechanism | Emitted by the measured Naga / Blade path? |
|---|---|
| RelaxedPrecision | **No** |
| NoContraction | **No** |
| FPFastMathMode / FPFastMathDefault | **No** |
| Rounding / denormal / special-value modes | **No** |
| f32, f16, BF16 types and conversions | **Yes**, according to the chosen kernel |

Ordinary f32 operations inherit Vulkan's rules. Explicit `fma` emits an FMA operation.

**“Strict” describes the f32 policy and numerical gates; it does not promise bitwise IEEE execution.**

<!--
Backup. Audited Naga 13f582d and Blade 4845d65, exactly the cohort pins.
No emission of the four control categories appears in the SPIR-V backend.
Blade's flags are FORCE_POINT_SIZE and optional DEBUG; it does not enable
float_controls2 or add precision decorations after Naga. f16 is a 16-bit
OpTypeFloat; BF16 is a 16-bit float with BFloat16KHR encoding and explicit
capabilities. Conversions use OpFConvert. WGSL fma maps to GLSL.std.450 Fma.

No RelaxedPrecision means we do not give the driver that particular permission
to evaluate an f32 result at relaxed precision. It does not prohibit the
other transformations Vulkan normally permits. With no NoContraction or
explicit float controls, contraction/reassociation and the default rounding,
denormal and special-value rules still apply as specified by Vulkan.
No explicit FastMath decoration does not mean strict IEEE math. The empirical
full-output/gradient checks and CPU f64 compiler tests are separate safeguards.

Sources:
https://github.com/kvark/blade/blob/4845d6529c38b07718a624bf3c4f2ab4317c9bf4/blade-graphics/src/vulkan/pipeline.rs#L55
https://github.com/kvark/wgpu/blob/13f582ddd2792befb46952498010248bd65818e6/naga/src/back/spv/block.rs#L1670
https://github.com/kvark/wgpu/blob/13f582ddd2792befb46952498010248bd65818e6/naga/src/back/spv/writer.rs#L2030
https://docs.vulkan.org/spec/latest/appendices/spirvenv.html
https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_shader_float_controls2.html
-->


---

<!-- _header: Backup · arithmetic choices -->
<!-- _class: backup -->

# Where TF32 fits

| Format | Exponent bits | Fraction bits | Role |
|---|---:|---:|---|
| FP32 | 8 | 23 | Strict input arithmetic |
| TF32 | 8 | 10 | FP32 range, more precision than BF16 |
| BF16 | 8 | 7 | Accelerated CUDA / HIP / XPU baseline |
| FP16 | 5 | 10 | Accelerated MPS baseline; native matrix kernels |

TF32 offers an accuracy/range tradeoff when FP16 or BF16 is unsuitable. It is not a strict-f32 substitute.

**TF32 is disabled in this cohort. Its absence does not explain these ratios.**

<!--
Backup. Nominal format bits, excluding sign and the implicit leading bit;
TF32 hardware may retain additional precision. It takes f32 data into tensor
operations with reduced input precision and f32 accumulation. BF16/FP16 have
higher peak matrix throughput on the relevant NVIDIA generations, but a
particular graph may need TF32's range or precision, or avoid materializing
conversions. That is an engineering hypothesis for a different contract,
not evidence that TF32 would improve our qualified accelerated programs.

Strict explicitly forbids reduced-precision inputs. Accelerated in this
cohort chooses BF16 autocast on CUDA/HIP/XPU and FP16 on MPS, while keeping
remaining f32 matmul/convolution TF32-disabled. The declared 135M one-token
exception also stays f32. Older TF32-only comparisons are not used here.
Sources:
https://github.com/kvark/inferena/blob/add19182e5479e3f7549894c524ac6691d3dc363/frameworks/pytorch/bench.py#L1339
https://github.com/kvark/inferena/blob/add19182e5479e3f7549894c524ac6691d3dc363/frameworks/pytorch/amp.py#L117
https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/accuracy-considerations.html
-->


---

<!-- _header: Backup · performance portability -->
<!-- _class: backup -->

# Portability scores

Pennycook metric: harmonic mean of relative efficiency; **zero if any platform lacks a qualified result**.

| Strict f32 · mean over workloads · seven GPUs | Meganeura | PyTorch |
|---|---:|---:|
| Inference | 0.62 | 0.87 |
| Minimal shape | 0.73 | 0.39 |
| Forward + loss + backward | 0.57 | 0.54 |

Missing Whisper and ResNet references strongly affect these rankings. A sole qualified engine scores efficiency one, **without implying a speedup**.

<!--
Backup. Efficiency is fastest qualified time divided by the engine's time.
Each workload/phase score is harmonic across seven measured GPUs, with a
zero for any missing qualified result; then take an arithmetic workload mean.
Strict Whisper on 7900 XT and ResNet on MI300X lack reference training and
minimal-shape results. Thus the apparent native aggregate advantage there
combines coverage and speed, and inherits the numerical policy. Do not use
it to claim faster execution than a rejected reference. Correctness-only
devices have no replicated performance and are not added to this metric.
Source: ../tables/portability.tex; data/summary.json.
-->

---

<!-- _header: Backup · warmup diagnostic -->
<!-- _class: chart backup -->

# Warmup

![SmolLM2-135M accelerated timings relative to each process's 60-second checkpoint, on RTX 5070 and B570](figures/warmup.svg)

Two processes per GPU. B570 stays within **0.91%**; RTX 5070 prefill is **0.67% / 3.04% slower** at 60 than at 5 seconds.

<!--
Backup. Separate study at Meganeura c19cb1c, Inferena study b110188 based on
9eb9928, not a new main-cohort condition. Same selected program throughout
5/15/30/60-second cumulative checkpoints, 20 samples per checkpoint, no memo.
Search already exercises the GPU, and earlier timing windows add heat.
Therefore this is not four independent cold starts, not all-model validation,
and not a direct measurement of the cohort's ten-second checkpoint.
RTX prefill clocks also decline; avoid claiming a controlled causal effect.
The study found no reason to extend warmup to 60 seconds for this workload.
Source: data/warmup-protocol.json and data/warmup.csv.
-->

---

<!-- _header: Backup · accelerated training -->
<!-- _class: chart backup -->

# Accelerated training

![Accelerated training time ratios for seven GPUs, with unqualified reference cells left empty](figures/accelerated-training.svg)

Median **2.21×** across **32 qualified pairs**. Numerical failures are visible gaps, not evidence that the rejected paths are slower.

<!--
Backup. CUDA/HIP/XPU use capability-selected BF16 here, MPS FP16. Native
retains f32 derivative regions. H100's SmolLM2 training is 17.49/4.77 ms;
B570's is 50.83/12.59 ms. These gaps coexist with native correctness and
broad coverage. Missing H100 diffusion and MI300X diffusion/Whisper are
contract failures. M3 retains all repeats and substantial uncertainty.
Source: data/results.csv; ../main.tex, Performance and Arithmetic contracts.
-->

---

<!-- _header: Backup · motivation -->

# Learning on the device

**Continual reinforcement learning:** keep updating a policy from experience during deployment.

> Can the learning loop fit on a consumer GPU?

**Observe → act → receive reward → update the policy → repeat**

An embodied agent has limited power and memory. Training must share the device with inference and perception.

This study measures the training and inference runtime needed for that goal.

<!--
Backup.

Continual RL is the motivation for wanting training after deployment. The
agent acts, sees what happened, receives a reward, and changes its policy.
Running a frozen policy only covers the inference part of that loop.

For a robot or wearable, local learning must fit the hardware it carries.
Cloud training is another design choice; the question here is what can stay
on the device. A desktop or mobile GPU is a more relevant deployment target
than assuming a datacenter accelerator is always available. Vulkan and Metal
give us access to that hardware and let ML share the application's GPU stack.

Meganeura addresses the runtime part. This paper does not demonstrate a
continual-RL algorithm, learning curves, on-device policy updates in a robot,
or power measurements. Its timing comparison is forward, loss, and backward;
optimizer updates are supported by the library but omitted from that timing.

Motivation: author's research direction. Measured scope: ../main.tex,
Introduction, Workloads and timing, and Deployment Beyond the Matrix.
-->

---

<!-- _header: Backup · saved kernel decisions -->
<!-- _class: chart backup -->

# Reusing kernel decisions

![Native preparation by GPU, with first-session memo misses and reused decisions separated](figures/preparation.svg)

Kernel choices persist. Graph and whole-program search run again on each build.

<!--
Backup. Ten first processes and twenty later processes per GPU, across five
models and both profiles. First-session memo miss describes kernel-decision
state, not cold driver caches. Later sessions in the first process can already
reuse decisions. Across all 287 native processes, there are 123 cold-first-
session and 164 warm-decision states. The modest separation is descriptive;
this is not a randomized cache ablation. These per-GPU bars include both
profiles and all core native runs; the main setup slide instead uses matched,
complete strict pairs grouped by model. Source: data/preparation-cache.csv.
-->
