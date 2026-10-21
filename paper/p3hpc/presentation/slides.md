---
marp: true
theme: meganeura
size: 16:9
paginate: true
title: 'Meganeura: Vulkan and Metal as a Performance-Portability Layer for GPU Training and Inference'
author: Dzmitry Malyshau
description: 'P3HPC / SC26 — 20-minute talk'
footer: 'MEGANEURA · P3HPC / SC26'
---

<!-- _class: title -->
<!-- _paginate: false -->
<!-- _footer: P3HPC workshop · SC26 · Chicago · November 2026 -->

# Meganeura

## Vulkan and Metal as a Performance-Portability Layer for GPU Training and Inference

Dzmitry Malyshau · Independent Researcher

[github.com/kvark/meganeura](https://github.com/kvark/meganeura)

<!--
Time: 0:20.

This talk asks how far the graphics stack can carry machine learning, training
as well as inference. My long-term interest is continual reinforcement
learning on a device, where the policy keeps learning after deployment.

If asked:
- The numbers come from the October 19 runs: Inferena add19182, Meganeura
  c19cb1c, PyTorch 2.13.0 at cf30153c. The published paper used the
  September 25 runs; the "Since the paper" slide compares the two.
-->

---

<!-- _header: Motivation -->
<!-- _class: opening -->

# One GPU, two stacks

<div class="game-intro"><div>

GPUs grew up rendering games. Games, simulators and headsets reach them through graphics APIs such as **Vulkan** and **Metal**.

Machine learning on the same GPU brings a vendor stack: **CUDA, ROCm, oneAPI or MPS**. None of them runs on a Quest headset.

> Can Vulkan and Metal carry inference and training too?

</div><figure>
<img src="assets/unreal-tournament.jpg" alt="Unreal Tournament gameplay: a first-person view with a rocket launcher, facing an opponent with a minigun">
<figcaption>Unreal Tournament (1999) · Epic Games / Digital Extremes<br>Screenshot from the GOG store listing</figcaption>
</figure></div>

<!--
Time: 0:45.

GPUs grew up rendering games. A game, a simulator, a robot or a headset
already has a GPU, a driver, buffers and a queue, through Vulkan or Metal.

Adding machine learning usually adds a second stack, one per vendor: CUDA on
NVIDIA, ROCm on AMD, oneAPI on Intel, MPS on Apple, usually with Python on top.
None of them runs on a Quest headset.

So the question is whether the graphics API the application already uses can
carry inference, and training as well. Meganeura is our attempt to find out.

If asked:
- The screenshot illustrates where graphics hardware came from. Source and
  attribution: assets/README.md.
-->

---

<!-- _header: Design · GPU access -->
<!-- _class: stack -->

# How do we reach every GPU?

**Blade is a Rust rendering library. We use its compute API.**

<div class="columns stack-columns"><div class="layers">
<div><b>Meganeura</b><span>Graph, kernels, memory and dispatch plan</span></div>
<div><b>Naga</b><span>WGSL → SPIR-V / Metal Shading Language</span></div>
<div><b>Blade</b><span>Device, buffers, pipelines, passes, timestamps</span></div>
<div><b>Driver</b><span>NVIDIA · RADV · ANV · Adreno · Apple</span></div>
</div><div>

- Binds shader data by field name
- Separates passes with global barriers
- Leaves checking to the API's validation layers
- Shares the renderer's context and queue

</div></div>

**Measured here:** Vulkan on Linux, Windows and Android · Metal on macOS

<!--
Time: 0:55.

Blade is a thin Rust layer over Vulkan and Metal that I wrote for rendering.
Meganeura uses its compute half. Shader data is bound by field name, so there
are no binding numbers to keep in sync. Passes are separated by global
barriers, with no per-resource state tracking, and correctness checking is
left to the native validation layers.

Kernels are written in WGSL. Naga, the shader translator from the wgpu
project, validates them and emits SPIR-V for Vulkan or Metal Shading Language.
We use Naga on its own, without the wgpu runtime.

Because Blade also renders, an application can draw and run its models on one
device and one queue. That is where this talk ends.

If asked:
- Blade also runs on iOS; this study measured Linux, Windows (the RTX 3050),
  Android and macOS.
- Pins: Blade 4845d65 and a Naga fork, kvark/wgpu 13f582d, which carries some
  matrix and BF16 features that were not upstream yet.

Sources: Blade blade-graphics/etc/motivation.md; ../main.tex, How Meganeura Works.
-->

---

<!-- _header: Design · compiler -->
<!-- _class: chart -->

# Inside Meganeura

![Eight steps from model description to execution: describe, rewrite, differentiate, plan, specialize, compile, check and time, execute](figures/compiler-map.svg)

One compiler for inference and training. **Search revisits these choices by timing them on the target GPU.**

<!--
Time: 1:00.

Read the top row left to right, then the bottom row right to left.

A model arrives from Rust builders, ONNX, NNEF or GGUF as a typed static
graph. An egglog e-graph holds equivalent forward forms and matrix schedules.
Each forward form gets its own backward pass; inference skips that step. The
planner lowers and fuses operations, orders the dispatches and reuses buffers
by lifetime. Kernels are specialized for shape, device and precision, written
out as WGSL and compiled through Naga and Blade into Vulkan or Metal pipelines.

The search builds candidate programs, checks them, tunes their kernels and
races whole programs on the device. The winner runs as a fixed plan: every
step records, submits and waits, with no graph work and no allocation.

If asked:
- Search is optional. Without it, the ordinary extracted program runs.
- The compiler tests and the benchmark checks get their own slides.

Sources: ../main.tex, How Meganeura Works; src/graph/composite.rs.
-->

---

<!-- _header: Design · models -->
<!-- _class: models -->

# How do we describe a model?

**A static graph of typed operations.** New models compose existing ops.

<div class="factors model-classes">
<div><h3>Primitive</h3><p>GPU lowering, CPU f64 semantics and a gradient rule.</p><small>MatMul · Add · Exp<br>Conv2d · Embedding</small></div>
<div><h3>Composite</h3><p>A named operation with a decomposition into primitives.</p><small>Softmax · RmsNorm<br>Gelu · CausalAttention</small></div>
<div><h3>Private</h3><p>Forms that optimization and differentiation introduce.</p><small>FusedMatMulAdd<br>WinogradConv2d<br>Conv2dGradWeight</small></div>
</div>

**Softmax(x) ⇄ exp(x − max x) / Σ exp(x − max x)**

<small>The decomposition defines and tests the op; the compiler recognizes the pattern and runs a fused kernel.<br>Rust builders, ONNX, NNEF and GGUF all produce this graph.</small>

<!--
Time: 0:50.

Every node has a static shape and a storage type. A primitive op carries three
things: a GPU lowering, CPU f64 semantics for testing, and a gradient rule.
A composite such as softmax or RMS norm is defined by its decomposition into
primitives. Tests go through the decomposition, and the compiler can still
recognize the pattern and run one fused kernel. Private ops are the forms that
compiler passes introduce, such as a matrix product fused with an add, or a
Winograd convolution. Models never build them directly.

If asked:
- Primitive semantics follow StableHLO where it applies.

Sources: src/graph/composite.rs, src/graph/decompose.rs and the importers.
-->

---

<!-- _header: Design · rewrites -->

# Which form is faster?

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

More parallel work and one extra reduction.

</div>
</div>

The e-graph keeps both forms. Search times each on the GPU, extra reduction included, and keeps a faster form only if it passes the numerical checks.

<!--
Time: 0:50.

Two examples of equivalent forms. In attention, the query, key and value
projections can stay three products or become one packed product. A 50 by 4096
by 720 product, one of the shapes in the paper's matrix study, can run as one
tiled product, or as eight partial products over 512-long slices of the
reduction followed by a sum. The split gives the GPU more parallel work but
adds temporary storage and a reduction dispatch, so we time the whole sequence.

Which form wins depends on the GPU, so the e-graph keeps both until they can
be measured. In the paper's matrix study, the search picked eight-way split-K
for this product on both GPUs it ran on.

If asked:
- An equally capable enumerator could find the same schedules. The e-graph's
  job is to keep the alternatives alive until they are timed.
- Reduced-precision matrix inputs are separate search options with their own
  checks.

Sources: ../main.tex, How Meganeura Works; camera-ready paper (tag
paper-p3hpc-work), Compile-Time Search.
-->

---

<!-- _header: Design · kernels -->
<!-- _class: kernels -->

# How do we write the kernels?

**WGSL source, specialized for shape, device and precision.**

<div class="columns kernel-columns"><div>

```wgsl
// Matrix-vector template, abridged
var<storage> matrix_b: $B_STORAGE;
const LANES: u32 = $LANESu;

// Filled by the generator:
$NORM_PROLOGUE
$WEIGHT_LOAD
$REDUCTION
```

</div><div>

### Three ways to compose

**Templates:** sections and substitutions for storage, loads and reductions.

**Pointwise chains:** one kernel, or a matrix product's epilogue.

**Matrix and attention generators:** code for the device's tile shapes.

</div></div>

**Quick to build:** everything outside the search takes a median **1.1 s** per workload.

<!--
Time: 0:55.

Every kernel is WGSL. Templates have sections and substitution points: this
one is the matrix-vector product used for decoding, and the generator fills in
the weight storage type, the vector width, how weights are loaded and how a
row is reduced, from the shapes, the device and the precision policy.
Pointwise operations need no template: a chain becomes one kernel, or folds
into a matrix product's epilogue. Matrix and attention kernels have
generators for each device's tile shapes.

Is generating source at build time slow? No. Everything outside the search,
including building the f32 reference program, takes a median 1.1 seconds per
workload. The search, which comes later, is what takes a minute.

If asked:
- 1.1 s is the median over 210 workload processes on the seven timed GPUs;
  the range is 0.18 to 10 s.
- A September 10 study on the RTX 5070, with fresh caches, broke this down:
  Naga takes about 0.15 ms per kernel and the driver's pipeline builds take
  most of the time. Backup "Compile stages".

Sources: src/shaders/matmul_gemv.wgsl; docs/shader-generation.md;
data/preparation.csv.
-->

---

<!-- _header: Design · testing -->
<!-- _class: correctness -->

# How do we know it is right?

**A CPU f64 interpreter defines the reference, separately from GPU lowering.**

| What can go wrong? | Check |
|---|---|
| Derivative rule | Symbolic gradients against finite differences, under a weighted loss |
| GPU kernel | Every output element against the interpreter, including tile edges |
| Compiler transformation | Composed and random graphs through rewrites, fusion and memory reuse |

Tests fill unwritten buffers with **NaN**, so a missing write shows up. Error bounds scale with **Σ|aᵢbᵢ|**, the size of each sum's terms.

<small>The benchmark runs check each selected program again before timing it.</small>

<!--
Time: 0:55.

When two engines disagree, comparing them cannot tell you which one is right.
So the reference is a plain CPU interpreter in f64, written separately from
the GPU lowering; only shapes and metadata are shared.

We keep three questions apart. Is the derivative rule right? Compare symbolic
gradients with finite differences, with each op under a weighted loss so that
a backward pass that assumes an all-ones upstream gradient fails. Does each
kernel compute its op? Compare every output element, tile edges included.
Does the compiler preserve meaning? Run composed and random graphs through
rewrites, fusion and memory planning.

These tests found dropped parameter gradients, loss reduction mistakes,
cancellation in norms, and a headset layout bug whose images still looked
plausible.

If asked:
- An element passes when |got − want| ≤ 2·10⁻⁴ · (sᵢ + floor · max s), where
  sᵢ comes from Σ|aᵢbᵢ|. That allows for cancellation.
- NaN fill: SessionOptions::poison, or MEGANEURA_POISON=1.
- These tests cover chosen cases and random graphs; they prove nothing about
  other inputs.

Sources: ../main.tex, Checking the Compiler and Its Kernels; docs/testing.md.
-->

---

<!-- _header: Design · matrix units -->
<!-- _class: hardware -->

# Can Vulkan reach the matrix units?

| GPU | Interface | Inputs | Tile M×N×K |
|---|---|---|---|
| RTX 5070 · H100 · RTX 3050 | `KHR_cooperative_matrix`, `NV_cooperative_matrix2` | f16 · BF16 | 16×16×16 |
| RX 7900 XT | `KHR_cooperative_matrix` | f16 | 16×16×16 |
| MI300X | `KHR_cooperative_matrix`, patched RADV | f16 · **f32** | 16×16×16 |
| Arc B570 | `KHR_cooperative_matrix` | f16 · BF16 | 8×16×16 |
| Apple M3 | Metal simdgroup matrices | f16 · **f32** · BF16 | 8×8×8 |

**One WGSL matrix type maps to all of them.** Accumulation is f32 everywhere.

<small>Out of reach from our shaders: TF32 inputs, Hopper's asynchronous warpgroup MMA, thread-block clusters.</small>

<!--
Time: 1:00.

Speed on modern GPUs lives in the matrix units. Vulkan reaches them through
the cooperative matrix extension, which the NVIDIA, AMD and Intel drivers
expose. On NVIDIA we also use the workgroup-scope extension, where several
subgroups share one larger tile. Blade enables these, and Naga translates
WGSL's cooperative matrix type to SPIR-V, or to Metal's simdgroup matrices.

The tiles differ by vendor, so the generators emit a kernel family per shape.
Only the MI300X, through our driver work, and the M3 take f32 inputs, so
those are the only two that use matrix units in strict f32.

Some hardware stays out of reach: there is no TF32 type, and Vulkan does not
expose Hopper's asynchronous warpgroup instructions or thread-block clusters.
The driver chooses the native instructions; we did not inspect the H100's.

If asked:
- The RTX 5070 driver also advertises tensor addressing and reductions; the
  compiler does not use them yet.
- The RX 7900 XT's RADV build in these runs offers no BF16 matrices.

Sources: data/hardware.json; run records (cooperative_*_tile fields);
https://docs.vulkan.org/features/latest/features/proposals/VK_NV_cooperative_matrix2.html
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions
-->

---

<!-- _header: Design · limits -->
<!-- _class: limits -->

# Same GPU, different limits

### Shared memory per workgroup

| GPU | Vendor compute API | Vulkan |
|---|---:|---:|
| RTX 5070 | CUDA: **99 KiB** with opt-in | **48 KiB** |
| Arc B570 | Level Zero: **128 KiB** | **48 KiB** |

Both API pairs allow **1,024 threads** per workgroup.

**A CUDA kernel's tiles and staging buffers may not fit through Vulkan.**

<small>NVIDIA 595.91.07 · Intel Mesa 26.0.3 · Level Zero 1.14.37020</small>

<!--
Time: 0:40.

The same silicon can look different through two APIs. On the RTX 5070, CUDA
lets a kernel opt into 99 KiB of shared memory; NVIDIA's Vulkan driver offers
48. On the Arc B570, Level Zero offers 128 KiB and Vulkan 48. A kernel
designed around the larger buffer has to be retiled for Vulkan.

If asked:
- Exact values: 101,376 bytes through CUDA opt-in (48 KiB by default),
  131,072 through Level Zero, 49,152 through Vulkan on both.
- The workgroup-scope matrix extension reserves some shared memory for the
  compiler, so not all of it is available to shader arrays.
- More shared memory allows larger tiles, and whether those are faster depends
  on the kernel. We have not measured how much of the gap the limits cause.

Sources: data/hardware.json (Vulkan queries October 15, CUDA and Level Zero
October 19); https://docs.nvidia.com/cuda/archive/13.0.3/blackwell-tuning-guide/index.html
-->

---

<!-- _header: Design · search -->

# How do we pick the fastest program?

<div class="search-steps">
<div><b>1</b><strong>Build</strong><span>Graph form, layouts, pipelines<br>Initialize weights + inputs</span></div>
<div><b>2</b><strong>Check</strong><span>Full outputs + gradients<br>against the original f32 build</span></div>
<div><b>3</b><strong>Tune</strong><span>Race kernel variants<br>Remember the winners</span></div>
<div><b>4</b><strong>Race</strong><span>Time the whole program<br>Keep it if it wins</span></div>
</div>

**30 s per session** · up to **16 graph forms** and **64 programs**.

Search varies graph forms, layouts, fusion, matrix input types and the number of submissions. A workload has two or three sessions: inference, minimal shape and training.

<!--
Time: 0:50.

For each candidate, the search builds a private session with initialized
weights and inputs and checks every output and gradient against the original
f32 build. It tunes the candidate's kernels, then races the whole program
against the best so far, recording and submitting fresh commands each time. A
kernel variant must win by 5 percent; a whole program by 1 percent plus the
measured noise. Kernel decisions are saved and reused; the program race runs
on every build.

In the October runs, 588 sessions tried 23,669 programs. Thirteen failed the
numerical checks and 34 hit the 1 GiB scratch limit for kernel probes.

If asked:
- 30 seconds is a soft deadline per session; preparing the f32 reference is
  outside it.
- The October runs have no search-off comparison. The paper's September
  ablation measured 1.18–1.55× from search on two GPUs.
- Whisper has two sessions because its minimal shape is its full shape.

Sources: ../main.tex, Compile-Time Search; ../tables/search.tex; data/summary.json.
-->

---

<!-- _header: Measurement · protocol -->
<!-- _class: measure -->

# How do we measure fairly?

<div class="columns experiment"><div>

### Five workloads

- SmolLM2-135M language model
- SmolVLA action expert
- Scaled Stable Diffusion U-Net
- ResNet-50
- Whisper-tiny encoder

</div><div>

### Seven timed GPUs

- NVIDIA RTX 5070, H100, RTX 3050
- AMD RX 7900 XT, MI300X
- Intel Arc B570 · Apple M3

### Four more, checked only

Radeon 780M · Ryzen 9600X iGPU · Intel RPL-U · Quest 3S

</div></div>

**Check → warm up 10 s → time 20 calls → repeat in 3 fresh process pairs**

Against PyTorch 2.13: `torch.compile`, with CUDA / HIP / XPU graph replay. **Ratio = Meganeura time ÷ PyTorch time.**

<!--
Time: 1:00.

Five workloads: a small language model, a robot action expert, a scaled
diffusion U-Net, ResNet-50 and Whisper's encoder. Each runs inference at its
full shape, inference at a minimal shape (one token, action or image), and a
training step: forward, loss and backward, without the optimizer update.

Seven GPUs from four vendors are timed. Four more, including the Quest
headset, are checked for correctness only.

Our harness, Inferena, runs both engines. Each must pass its numerical checks
before it is timed. Then it warms up for ten seconds and times 20 calls, each
waiting for the GPU to finish. That repeats in three fresh pairs of processes,
alternating which engine goes first, and we report the median of the three
process medians. Below one, we are faster.

If asked:
- The engines pass different checks. Meganeura compares full tensors with its
  f32 build. PyTorch must repeat exactly and, with reduced precision, track
  f32 over matched SGD trajectories. Across engines, sampled outputs, loss and
  gradient norms must agree. Backup "What the checks admit".
- max-autotune is off, and PyTorch compilation has a 120 s limit.
- Only SmolLM2 uses trained weights; the others use deterministic uniform
  parameters.

Sources: ../main.tex, Methodology; ../tables/devices.tex.
-->

---

<!-- _header: Measurement · MI300X -->
<!-- _class: mi300 -->

# MI300X needed a driver

<div class="columns mi300-columns"><div>

### A compute-only GPU

- No Vulkan driver ships for it; the paper could not run there.
- RADV, Mesa's Vulkan driver, now runs compute and matrix tiles on it, f32 tiles included.
- An AMDGPU patch spreads work across all eight compute dies.
- Draft upstream: Mesa MR 45239.

</div><div>

<div class="big-number"><strong>42 / 42</strong><span>native results pass every check, up to SmolLM2-1.7B</span></div>

Same Meganeura and Inferena revisions as every other GPU. Experimental drivers; no conformance claim.

</div></div>

<!--
Time: 0:45.

MI300X is a compute-only accelerator and no Vulkan driver ships for it, so the
paper said Meganeura could not run there. We extended RADV, Mesa's Vulkan
driver, building on Mesa's experimental CDNA work: buffer-based compute and
cooperative matrices, including f32 tiles, which none of the other timed
Vulkan GPUs offer. A patch to the AMDGPU kernel driver spreads the work across
all eight compute dies. The unchanged engine and harness then pass all 42
results, from the five core models up to the 1.7-billion-parameter language
model. The Mesa changes are posted as a draft merge request.

If asked:
- 42 = 30 core results (five models, two profiles, three phases) plus 12 for
  SmolLM2-360M and 1.7B.
- Device: AMD Instinct MI300X VF (RADV GFX940), 304 CUs, about 192 GiB.
- Mesa source 8c7f5fdd; the AMDGPU patch targets amdgpu-6.19.14 on Linux
  6.8.0-146 and is enabled with mi300_xcc_gang=1.
- PyTorch on this machine needs one MIOpen convolution solver excluded.

Sources: ../main.tex, Availability;
https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/45239 (draft,
opened October 15).
-->

---

<!-- _header: Results · strict f32 -->
<!-- _class: chart results -->

# Strict f32: same arithmetic

![Strict f32 time ratios for seven GPUs and five workloads, one panel each for inference, minimal shape and training](figures/strict-ratios.svg)

Medians: <span class="slower">1.35×</span> inference · <span class="slower">1.03×</span> minimal shape · <span class="slower">1.61×</span> training. Meganeura is faster in **29 of 101** comparisons, and in **11 of 13** on the RX 7900 XT.

<!--
Time: 1:00.

Strict f32 is the fair comparison: both engines compute in f32, with TF32 and
reduced-precision matrix inputs turned off. Each cell is our time divided by
PyTorch's; teal means we are faster.

At the minimal shape we are at parity overall, within 3 percent at the median,
and faster in 15 of 33 cells. Inference on the three NVIDIA GPUs is within 4
to 13 percent. On the RX 7900 XT we are faster in 11 of 13 comparisons.
Training trails at 1.61: our backward kernels have had the least tuning. The
Arc B570 and the MI300X are where we trail most.

If asked:
- Strict fixes the arithmetic width; the order of reductions still differs.
- Missing cells: PyTorch's Whisper training on the RX 7900 XT and ResNet
  training on the MI300X fail its repeatability check, and the minimal shape
  runs after training, so it was not attempted.
- Per-GPU inference medians: RTX 5070 1.04, H100 1.13, RTX 3050 1.12,
  RX 7900 XT 0.63, MI300X 2.25, B570 2.28, M3 1.35.
- The M3's spread between processes means its near-ties are ties.

Sources: data/results.csv; data/summary.json.
-->

---

<!-- _header: Results · since the paper -->
<!-- _class: since-paper -->

# Since the paper

**At the median, PyTorch's strict times moved less than 3% on five of the six GPUs in both. Ours fell 6.5% for inference and 15% for training.**

| Strict f32 medians · six GPUs | Paper · Sept 25 runs | Talk · Oct 19 runs |
|---|---:|---:|
| Inference | 1.27× | 1.13× |
| Minimal shape | 1.09× | 1.00× |
| Training | 1.92× | 1.55× |

**Also changed:** MI300X added · PyTorch's reduced precision is BF16 autocast instead of TF32 · search 30 s per session instead of 60 · warmup 10 s instead of 2.

<small>Without the noisy M3: 1.21× → 1.13×, 1.10× → 1.02×, 1.99× → 1.59×. The protocol also changed, so part of the difference may come from it.</small>

<!--
Time: 0:50.

If you read the paper, these are the numbers to compare. On the six GPUs both
sets of runs share, PyTorch's median strict times moved less than 3 percent,
leaving out the noisy M3. Ours fell by a median of 6.5 percent for inference
and 15 percent for training, and every one of those GPUs improved its strict
inference and training medians. That comes from kernel and search work since
September, which we did not ablate change by change.

Other things changed too: MI300X joined, PyTorch's reduced precision is now
BF16 autocast instead of TF32, the search budget halved to 30 seconds per
session, and warmup grew from 2 to 10 seconds.

If asked:
- All seven GPUs now: 1.35 / 1.03 / 1.61 strict, 2.06 / 1.49 / 2.21 with
  reduced precision. The paper reported 2.00 and 2.61 with reduced precision.
- One regression: U-Net inference on the B570 went from 2.97 to 5.89 ms. The
  October 21 Winograd work targets it.
- The benchmark runner grew from 13.5 to 19.1 MiB.

Sources: since_paper.py, which replays the camera-ready supplement
(../submission/meganeura-p3hpc-aux.zip) with the tagged artifact/cohort.py;
data/results.csv.
-->

---

<!-- _header: Results · reduced precision -->
<!-- _class: chart results -->

# With reduced precision

![Accelerated time ratios for seven GPUs and five workloads, one panel each for inference, minimal shape and training](figures/accelerated-ratios.svg)

Medians: <span class="slower">2.06×</span> inference · <span class="slower">1.49×</span> minimal shape · <span class="slower">2.21×</span> training. PyTorch uses BF16 autocast (FP16 on the M3); Meganeura may use f16 or BF16 matrix inputs in the forward pass.

<!--
Time: 0:50.

Now each engine may use reduced precision, which is how these models usually
run in practice. PyTorch uses BF16 autocast, or FP16 on Apple. Meganeura may
use f16 or BF16 inputs for forward matrix work, keeps f32 storage, and keeps
derivatives in f32. Both still have to pass their numerical checks, and TF32
stays off.

The gap widens: 2.06 for inference and 2.21 for training. The Apple M3 goes
the other way: there PyTorch's FP16 path is often slower than its f32 path, in
most training runs and in both transformer models.

If asked:
- Missing cells are PyTorch failures. On the H100, the U-Net's maximum error,
  0.02255, exceeds its 0.02085 bound. On the MI300X, the U-Net and Whisper
  fail repeatability.
- Meganeura is faster in 21 of 96 comparisons, 13 of them at the minimal
  shape.

Sources: data/results.csv; data/summary.json; ../NUMERICAL-POLICY.txt.
-->

---

<!-- _header: Results · reduced precision -->
<!-- _class: chart -->

# Who gains from reduced precision?

![Speedup from strict to accelerated for each engine on each GPU, for inference and training](figures/precision-gain.svg)

PyTorch moves forward and backward matrix work onto the matrix units. **Our NVIDIA programs use them in at most 19 dispatches, forward only; on the B570, in none.**

<!--
Time: 0:55.

Why does the gap widen? Compare each engine with itself. The bars show how
much faster each engine gets when reduced precision is allowed.

PyTorch gets 1.2 to 3.1 times faster on the NVIDIA GPUs, the RX 7900 XT and
the B570: autocast moves its matrix work, forward and backward, onto the matrix
units. Meganeura gains at most 11 percent. In its selected NVIDIA programs, the
matrix units run at most 19 dispatches, all in the forward pass, because
derivatives stay in f32 and the search seldom picked the f16 kernels. On the
B570 it picked none.

On the M3 and MI300X, Meganeura already uses f32 matrix tiles in strict f32,
and PyTorch gains little or slows down with reduced precision, so the ratio
improves there.

If asked:
- Across all cells, PyTorch's median gain is 1.50 for inference and 1.56 for
  training; ours is 1.03 and 1.01.
- In strict f32, the NVIDIA programs use no matrix dispatches; the M3 and
  MI300X use up to 513 and 543 per program in both profiles.
- Workgroup-scope products were selected in 6 of 42 reduced-precision
  sessions on each NVIDIA GPU.
- Another round of reduced-precision measurements is under way.

Sources: data/results.csv; run records (cooperative_dispatches per selected
session).
-->

---

<!-- _header: Results · larger models -->
<!-- _class: chart -->

# Larger models

![Strict SmolLM2 prefill and training ratios at 135M, 360M and 1.7B on H100 and MI300X, with process ranges](figures/scaling.svg)

Strict training ratio from 135M to 1.7B: H100 <span class="slower">1.90× → 3.79×</span>, MI300X <span class="slower">2.29× → 1.65×</span>.

<!--
Time: 0:35.

The same language model at three sizes, in strict f32, on the two servers. On
the H100 our training gap grows with model size, from 1.90 to 3.79. On the
MI300X it shrinks, from 2.29 to 1.65; there our kernels run on its f32 matrix
tiles, up to 579 matrix dispatches per program.

If asked:
- Batch one and 128-token prefill: this is not production training or
  decoding.
- At 1.7B our training step takes 135.85 ms on the H100 and 35.88 ms on the
  MI300X; PyTorch takes 35.84 and 21.73 ms.
- With reduced precision, PyTorch fails its worst-token KL check at 1.7B on
  both servers (its mean KL passes), so there is no reduced-precision ratio.

Sources: ../tables/scaling.tex; ../RESULTS.md.
-->

---

<!-- _header: Results · the gap -->
<!-- _class: explanation -->

# Why the gap?

<div class="factors">
<div><h3>Matrix units</h3><p>Reduced precision makes PyTorch 1.5× faster at the median, Meganeura 1.03×.</p><p>Our search seldom picks matrix kernels on NVIDIA and never did on the B570.</p></div>
<div><h3>Training kernels</h3><p>Strict training trails at 1.61×, against 1.35× for inference.</p><p>Backward kernels had the least tuning; Intel and server GPUs arrived late.</p></div>
<div><h3>API and drivers</h3><p>Less shared memory through Vulkan on NVIDIA and Intel; no clusters or warpgroup MMA.</p><p>MI300X needed driver patches to run at all.</p></div>
</div>

**Next:** matrix kernels for backward passes and convolutions, BF16 coverage, and a cheaper search.

<small>These runs cannot split the gap between these causes.</small>

<!--
Time: 0:45.

Three causes, from best measured to least. First, the matrix units: reduced
precision moves PyTorch's matrix work onto them, while our search seldom picks
matrix kernels on NVIDIA and never did on the B570. Second, training: even in
strict f32 it trails inference, and the backward kernels had the least
tuning. Development started on AMD and NVIDIA consumer GPUs; Intel came later,
and server access was intermittent. Third, the API and drivers: less shared
memory through Vulkan on NVIDIA and Intel, no thread-block clusters or Hopper
warpgroup instructions, and an MI300X that needed driver work to run at all.

If asked:
- This is a whole-stack comparison: PyTorch, Inductor and vendor libraries
  against our compiler plus the Vulkan or Metal driver.
- An October 21 follow-up already selects F16 Winograd matrix kernels for the
  U-Net on the B570.

Sources: data/summary.json; data/hardware.json; author's development history.
-->

---

<!-- _header: Results · setup cost -->
<!-- _class: chart -->

# What search costs

![Preparation by workload: Meganeura's preparation, mostly search, against PyTorch's compilation and first calls](figures/preparation-models.svg)

Meganeura: <span class="mega">49–68 s</span> per workload, almost all of it search. PyTorch compiles in <span class="torch">12–54 s</span>.

<!--
Time: 0:40.

Setup time per workload process in strict f32, as the median across GPUs.
Meganeura spends about a minute, almost all of it in the search; as we saw,
everything else takes about a second. PyTorch's compilation and first calls
take 12 to 54 seconds. The two bars cover different work: ours checks every
candidate, and PyTorch's excludes its graph capture and checks.

If asked:
- The search has 30 s per session, and a workload has two or three sessions.
- PyTorch's longest single compilation took 103 s, inside its 120 s limit.
- Kernel decisions saved from earlier processes shorten preparation only a
  little: backup "Reusing kernel decisions".
- PyTorch's separately recorded graph-capture medians are 2.65, 2.01, 0.34,
  0.72 and 0.42 s.

Source: data/preparation.csv, from ../artifact/final_cohort.py.
-->

---

<!-- _header: Deployment -->
<!-- _class: deployment -->

# On the headset

![bg right:36% contain](../../figures/dinovision-quest3s.jpg)

**DinoVision · Quest 3S · Android / Adreno 740**

On a desktop GPU: train a **2.01M-parameter decoder with Adam**, then save the weights.

On the headset: load the encoder and decoder, and reconstruct camera frames for both eyes.

**Inference and OpenXR rendering share one Blade context and queue.**

<small>2.36 billion multiply-accumulates per pass. Image: the headset's reconstruction.</small>

<!--
Time: 0:50.

Back to the opening: one GPU, one stack. DinoVision runs on a Quest 3S, an
Android headset with an Adreno GPU. A DINOv3 encoder feeds a small decoder that
reconstructs the camera image; the picture is its output. The decoder was
trained with Adam on a desktop Vulkan GPU, saved, and loaded on the headset.
Two eye workers share Blade's Vulkan context and queue with the OpenXR
renderer, and there is no second GPU runtime.

If asked:
- With the same frame and weights, preprocessing is bit-exact between desktop
  and headset; encoder and reconstruction relative L2 differences are
  0.001059 and 0.000457.
- Splitting a pass into two submissions changes the live schedule: backup
  "Queue measurements".
- Training runs on the desktop; the headset runs inference.

Sources: ../../dinovision-section.tex;
https://github.com/kvark/dinovision/blob/dc35cdf1c7c910cdd93c5b5362846842ae469a21/REPORT.md
-->

---

<!-- _header: Conclusion -->
<!-- _class: closing conclusions -->

# What works

Five models pass every check on **11 GPUs**, from a Quest 3S to an MI300X.

Deployment is **one 19 MiB executable** plus weights, on the system's GPU driver.

DinoVision trains on a desktop GPU and runs on the headset.

**Speed:** with the same arithmetic, near parity at the minimal shape and 1.35× PyTorch's time for inference. With reduced precision PyTorch pulls ahead, to 2.06×; that is our next target.

[github.com/kvark/meganeura](https://github.com/kvark/meganeura)

<!--
Time: 0:50.

Back to the question. Vulkan and Metal carried inference and training for five
models on eleven GPUs, from a headset to a datacenter accelerator, with one
compiler and no vendor compute stack. The runner is a 19 MiB executable that
needs only the system's GPU driver. DinoVision closes the loop: trained on a
desktop GPU, deployed on the headset, sharing the renderer's queue.

Speed is the unfinished part. With the same arithmetic we are close, and since
the paper we got faster while PyTorch stayed put. With reduced precision,
PyTorch's matrix-unit path pulls ahead, and that is where the work goes next.
Thank you.

If asked:
- Runner: Inferena add19182, 20,072,584 bytes, SHA-256 b8d25fac…. It links
  libc and loads the Vulkan driver; no Python or vendor SDK.
- MI300X uses experimental drivers; four GPUs are checked for correctness only.

Sources: data/summary.json; data/hardware.json; DinoVision REPORT.md.
-->

---

<!-- _header: Backup · matrix use -->
<!-- _class: paths -->

# Matrix dispatches by GPU

| GPU | Sessions with matrix work, of 42 | Matrix dispatches per program | Workgroup-scope sessions |
|---|---:|---:|---:|
| RTX 5070 · H100 · RTX 3050 | 24 · 21 · 24 | 0–19 | 6 each |
| RX 7900 XT | 22 | 0–31 | 0 |
| MI300X | 30 | 0–543 | 0 |
| Arc B570 | 0 | 0 | 0 |
| Apple M3 | 30 | 0–513 | 0 |

Reduced-precision programs of the five core models. In strict f32, only MI300X and M3 use matrix units, through f32 tiles, with similar counts.

<small>BF16 was selected in 3 sessions each on the RTX 5070 and RTX 3050.</small>

<!--
Backup, for "do you actually use the matrix units?"

Each GPU has 42 reduced-precision sessions: three repeats of three sessions for
four models and two for Whisper. A session counts once its selected program has
one matrix dispatch. On NVIDIA, the matrix dispatches come from
ResNet (11–19), Whisper (16), SmolVLA (0 or 16; 0 on the H100) and SmolLM2
(0 or 1); the U-Net has none. The workgroup-scope kernel is the single matrix
dispatch of six SmolLM2 sessions per NVIDIA GPU.

The B570's zero is a search outcome: the device offers 8×16×16 f16 and BF16
tiles, and the compiler at c19cb1c searches them. An October 21 follow-up
(PRs #273 and #274, revision a0cd01b) selects F16 Winograd matrix kernels for
U-Net inference and training on the B570. Those runs are not in these tables.

If asked:
- The counts show where matrix units run; they say nothing about their share
  of the time.
- At larger sizes the H100's reduced-precision training programs use 33 (360M)
  and 97 (1.7B) matrix dispatches.

Sources: data/summary.json accelerated_selection; run records;
/x/Code/inferena-results/pr273-274-review-20261021/review-summary.json.
-->

---

<!-- _header: Backup · reduced precision -->
<!-- _class: contracts -->

# Reduced-precision rules

| | PyTorch | Meganeura |
|---|---|---|
| Arithmetic | BF16 autocast on CUDA / HIP / XPU; FP16 on MPS | f16 or BF16 matrix inputs where selected; f32 storage and accumulation |
| Training | Autocast also covers backward | Derivatives stay in f32 |
| Checks | f32 forward fidelity, repeatability, matched SGD trajectories | Full outputs and gradients against its f32 build |

**TF32 is off in both profiles.**

<!--
Backup, for "what exactly does the accelerated profile allow?"

PyTorch uses autocast: BF16 on CUDA, HIP and XPU, FP16 with loss scaling on
MPS. Its matmul and cuDNN TF32 switches are off. Meganeura keeps f32 graph
storage and may use f16 or BF16 inputs with f32 accumulation in forward matrix,
convolution and attention work; derivative regions stay in f32. The two
policies therefore produce different arithmetic, especially in the backward
pass.

PyTorch's training check: 256 matched SGD updates on real text for SmolLM2,
32 repeated-input updates for the other workloads, each loss within 2% of
eager f32. Meganeura's: complete single-step gradients within 1% of its f32
build.

Source: ../NUMERICAL-POLICY.txt.
-->

---

<!-- _header: Backup · numerical checks -->
<!-- _class: backup gates -->

# What the checks admit

| Check | Bound |
|---|---|
| Meganeura, strict | Full tensors: max and RMS error ≤ 10⁻⁶ + 10⁻⁴ · reference scale |
| Meganeura, reduced precision | Outputs other than language models: 2%; whole training gradient: 1% |
| Language model, reduced precision | Mean KL ≤ log(1.01); worst token ≤ log(1.10); mean loss within 2% + 10⁻⁶ |
| PyTorch, reduced-precision training | 256 real-text SGD updates for SmolLM2, 32 repeated-input updates otherwise; loss within 2% + 10⁻⁶ |
| Across engines, f32 snapshots | Sampled outputs and loss: 1%; gradient norms: 5% strict, 10% per process and 5% median otherwise |

**Looser bounds could change which programs each engine runs.**

<!--
Backup, for "how loose are your checks?"

The percentages abbreviate bounds relative to the reference's scale, applied
to maximum and RMS error. Meganeura's reference is its original f32 graph,
prepared before any precision change; PyTorch's is eager strict f32.
Meganeura's search can keep an f32 program when reduced-precision candidates
fail; PyTorch's dtype recipe is fixed per model and backend.

Repeatability is checked separately: full outputs within 10⁻⁶ + 10⁻⁴ of the
reference scale, and whole reduced-precision gradients within 1%. We have no
tolerance-sweep experiment, so the effect of looser bounds is unmeasured.

Source: ../main.tex, Numerical gates; ../NUMERICAL-POLICY.txt.
-->

---

<!-- _header: Backup · coverage -->

# Coverage

| GPU | Result | Comparison |
|---|---|---|
| MI300X | Core models, 360M and 1.7B pass | Experimental Mesa and AMDGPU |
| Radeon 780M · 9600X iGPU · RPL-U | **90 / 90** results pass | Against CPU PyTorch f32; no timing |
| Quest 3S / Adreno 740 | **30 / 30** results pass | Against CPU f64 full tensors; no timing |
| B570 · SmolLM2-360M | Meganeura completes | PyTorch's XPU graph capture fails |

**All 330 combinations of GPU, model, profile and phase pass.**

<!--
Backup, for "where else does it run?"

Seven timed GPUs plus four correctness-only ones. Across the core and larger
models, 246 timed native results pass and 226 have a PyTorch comparison; the
other 20 lack a passing PyTorch result.

The B570's 360M PyTorch run fails inside an Inductor-generated operator during
XPU graph capture: "wait method cannot be used for an event associated with a
command graph". Our harness also fails to classify that message and stops, so
the B570 360M row is incomplete. That is an execution failure, separate from
the numerical failures elsewhere.

Android archives keep the comparison receipts and hashed manifests, without
the raw tensors.

Sources: ../QUALIFICATION.md; ../RESULTS.md.
-->

---

<!-- _header: Backup · timing variability -->
<!-- _class: chart -->

# Timing variability

![Between-process spread of native and PyTorch times on the seven timed GPUs](figures/timing-spread.svg)

Apple M3: **59 of 60** measurements vary by more than 10% between processes (median **34%**), with sleep prevention on. Every repeat stays in the results.

<!--
Backup, for "how noisy is this?"

Each point is the spread of three process medians. Both engines move on the M3, and we have no clock or power
trace that explains it, so treat small M3 differences as ties. The B570 has 7
of 60 groups above 10%. The H100 has one native outlier in a reduced-precision
U-Net minimal-shape run (1.773, 5.310 and 1.772 ms) whose PyTorch reference
had already failed, so it affects no ratio.

Source: data/results.csv; ../RESULTS.md.
-->

---

<!-- _header: Backup · warmup -->
<!-- _class: chart backup -->

# Warmup

![SmolLM2-135M reduced-precision timings relative to each process's 60-second checkpoint, on RTX 5070 and B570](figures/warmup.svg)

Two processes per GPU. The B570 stays within **0.91%**; RTX 5070 prefill is **0.67% / 3.04%** slower at 60 s than at 5 s.

<!--
Backup, for "is ten seconds of warmup enough?"

A separate study at Meganeura c19cb1c and Inferena b110188: the same selected
program measured at 5, 15, 30 and 60 cumulative seconds, 20 samples per
checkpoint. The search has already heated the GPU by then, so these are not
cold starts, and only one workload was tested. The study found no reason to
extend warmup to 60 seconds for it.

Source: data/warmup-protocol.json and data/warmup.csv.
-->

---

<!-- _header: Backup · compile stages, September 10 -->
<!-- _class: backup stages -->

# Compile stages

**September 10 study · RTX 5070 · fresh driver caches · no search**

| Per model | Naga parse, per kernel | All Naga stages | Vulkan pipelines | Cold Triton builds |
|---|---:|---:|---:|---:|
| SmolLM2-135M | 141 µs | 7.2 ms | 387 ms | 2,235 ms |
| ResNet-50 | 147 µs | 14.8 ms | 632 ms | 6,726 ms |
| Whisper-tiny | 172 µs | 20.2 ms | 851 ms | 6,599 ms |

Whole preparation with fresh caches: Meganeura **1.2–1.4 s**, PyTorch **11–24 s**.

<small>The engines build different kernel sets: 21, 66 and 56 native pipelines against 37, 181 and 96 Triton compilations.</small>

<!--
Backup, for "how fast is your compiler?"

An earlier study at the paper's development revision, with CPU spans added to
Meganeura and Blade on an experiment branch. Three fresh processes per engine,
model and cache state. Naga's column sums parsing, validation, specialization
and SPIR-V emission; the driver's pipeline creation dominates native
preparation. With the driver's disk cache reused, native pipeline totals fall
to 1.7, 5.5 and 3.9 ms.

The per-model columns compare build steps over different kernel sets. A matched
f32 GEMM study compiled the same product in both: a new kernel takes about
31 ms natively, almost all of it in the NVIDIA driver, against 100 to 167 ms
for Triton.

Source: ../../../docs/experiments.md, Compiler stages — September 10
(tags experiment/compiler-stages-2026-09-10).
-->

---

<!-- _header: Backup · saved kernel decisions -->
<!-- _class: chart backup -->

# Reusing kernel decisions

![Native preparation by GPU, with first-session memo misses and reused decisions separated](figures/preparation.svg)

Kernel choices persist across processes. Graph and whole-program search run again on each build.

<!--
Backup, for "can you cache the search?"

Ten first processes and twenty later processes per GPU, across five models
and both profiles. A first-session memo miss means no saved kernel decisions;
driver caches are a separate matter. Later sessions in the first process can
already reuse decisions. Across all 287 native processes there are 123
first-session and 164 reused-decision states. There was no randomized cache
experiment.

Source: data/preparation-cache.csv.
-->

---

<!-- _header: Backup · portability -->
<!-- _class: backup -->

# Portability scores

Pennycook's metric: the harmonic mean of each engine's efficiency across GPUs, **zero if any GPU lacks a passing result**.

| Strict f32 · mean over workloads · seven GPUs | Meganeura | PyTorch |
|---|---:|---:|
| Inference | 0.62 | 0.87 |
| Minimal shape | 0.73 | 0.39 |
| Training | 0.57 | 0.54 |

PyTorch's missing Whisper and ResNet training results score zero, which decides the last two rows.

<!--
Backup, for "what about the P3HPC portability metric?"

Efficiency is the fastest passing time divided by the engine's time. Each
workload and phase gets a harmonic mean across the seven timed GPUs, with a
zero for any missing result, then we average over workloads. Strict Whisper on
the RX 7900 XT and ResNet on the MI300X lack PyTorch training and minimal-shape
results, so Meganeura's higher scores there reflect coverage more than speed.
The correctness-only GPUs are not included.

Source: ../tables/portability.tex; data/summary.json.
-->

---

<!-- _header: Backup · arithmetic formats -->
<!-- _class: backup -->

# Where TF32 fits

| Format | Exponent bits | Fraction bits | Role here |
|---|---:|---:|---|
| FP32 | 8 | 23 | Strict f32 |
| TF32 | 8 | 10 | Off in both profiles |
| BF16 | 8 | 7 | PyTorch on CUDA / HIP / XPU; our matrix inputs |
| FP16 | 5 | 10 | PyTorch on MPS; our matrix inputs |

TF32 keeps f32's range with fewer fraction bits. With reduced precision, PyTorch uses BF16 instead.

<!--
Backup, for "why is TF32 off?"

Nominal bits, excluding the sign and the implicit leading bit. TF32 feeds f32
data into the matrix units at reduced input precision with f32 accumulation.
Strict f32 forbids reduced-precision inputs, so TF32 is off there. With reduced
precision allowed, PyTorch uses BF16 autocast, which is faster on these NVIDIA
GPUs, and its remaining f32 matrix products and convolutions also stay off
TF32. Vulkan has no TF32 type, so Meganeura could not use it anyway.

Sources:
https://github.com/kvark/inferena/blob/add19182e5479e3f7549894c524ac6691d3dc363/frameworks/pytorch/bench.py#L1339
https://github.com/kvark/inferena/blob/add19182e5479e3f7549894c524ac6691d3dc363/frameworks/pytorch/amp.py#L117
-->

---

<!-- _header: Backup · shader precision -->
<!-- _class: backup precision -->

# SPIR-V precision

| Mechanism | Emitted by the measured Naga / Blade path? |
|---|---|
| RelaxedPrecision | **No** |
| NoContraction | **No** |
| FPFastMathMode / FPFastMathDefault | **No** |
| Rounding / denormal / special-value modes | **No** |
| f32, f16, BF16 types and conversions | **Yes**, according to the chosen kernel |

Ordinary f32 operations follow Vulkan's default rules. Explicit `fma` emits an FMA.

**“Strict” means f32 arithmetic plus our numerical checks, under Vulkan's usual f32 rules.**

<!--
Backup, for "is strict really IEEE?"

Audited at exactly the measured pins, Naga 13f582d and Blade 4845d65. The
SPIR-V backend emits none of the four control categories, and Blade adds no
precision decorations. f16 is a 16-bit OpTypeFloat; BF16 uses the
BFloat16KHR encoding with its capabilities; conversions use OpFConvert; WGSL
fma maps to GLSL.std.450 Fma.

Without NoContraction or explicit float controls, the driver may still
contract and reassociate as Vulkan permits, under its default rounding,
denormal and special-value rules. The full-output checks and CPU f64 tests are
the safeguards.

Sources:
https://github.com/kvark/blade/blob/4845d6529c38b07718a624bf3c4f2ab4317c9bf4/blade-graphics/src/vulkan/pipeline.rs#L55
https://github.com/kvark/wgpu/blob/13f582ddd2792befb46952498010248bd65818e6/naga/src/back/spv/block.rs#L1670
https://github.com/kvark/wgpu/blob/13f582ddd2792befb46952498010248bd65818e6/naga/src/back/spv/writer.rs#L2030
https://docs.vulkan.org/spec/latest/appendices/spirvenv.html
-->

---

<!-- _header: Backup · model construction -->
<!-- _class: api-code -->

# The API

**Plain Rust builders. Shapes and storage types are checked as nodes are added.**

```rust
let mut g = Graph::new();
let x = g.input("x", &[32, 784]);
let labels = g.input_u32("labels", &[32]);
let fc1 = nn::Linear::new(&mut g, "fc1", 784, 128);
let h = fc1.forward(&mut g, x);
let h = g.relu(h);
let fc2 = nn::Linear::new(&mut g, "fc2", 128, 10);
let logits = fc2.forward(&mut g, h);
let loss = g.cross_entropy_loss_indices(logits, labels);
g.set_outputs(vec![loss]);
```

<small>Names survive compilation, profiling and readback.</small>

<!--
Backup, for "what does a model look like in code?"

Builders return node handles, and layers are thin wrappers over them. Shapes
are validated before any GPU work, and an error points at the model's own line.
Building a session then runs differentiation, optimization and pipeline
construction; a Trainer supplies the optimizer.

Sources: src/graph.rs, src/nn.rs, src/train.rs; presentation-story/api_demo.rs.
-->

---

<!-- _header: Backup · training -->

# Gradients

**Reverse-mode autodiff appends backward work to the same graph.**

<div class="pipeline">
<div><strong>Differentiate</strong><span>Model + loss → gradient ops</span></div><b>→</b>
<div><strong>Optimize</strong><span>Rewrites, kernels, buffer lifetimes</span></div><b>→</b>
<div><strong>Execute</strong><span>Forward + loss + backward</span></div>
</div>

- Composite gradients follow their primitive decompositions.
- Derivatives stay in f32 in both profiles.
- Packed parameter and gradient arenas let one optimizer dispatch update many tensors.

<small>The benchmark stops before the optimizer. DinoVision runs the full Adam training cycle.</small>

<!--
Backup, for "how does training work?"

Training uses the same compiler as inference, with a larger graph. Each
forward alternative is differentiated, and backward layouts are explored too.
Reduced precision in the forward pass does not carry over to derivatives. The
checks compare full gradients, which catch underflow that plausible outputs
can hide.

Sources: src/autodiff.rs, src/memplan.rs, src/train.rs.
-->

---

<!-- _header: Backup · headset scheduling -->
<!-- _class: deployment -->

# Queue measurements

![bg right:32% contain](../../figures/dinovision-quest3s.jpg)

**DinoVision · Quest 3S · Android / Adreno 740**

Two eye workers + renderer, **one context and queue**.

| Measurement | 1 chunk | 2 chunks |
|---|---:|---:|
| ML alone · ms/pass | 96.78 | 100.79 |
| Live · render submissions/s | 10.83 | 12.87 |
| Live · worker latency, ms | 149.51 | 119.25 |

**Splitting a pass lets the renderer submit between the halves.**

<small>Live test: 90 seconds per setting. Render submissions per second is the app's submission rate; display FPS was not measured.</small>

<!--
Backup, for "what does sharing the queue change?"

Alone, splitting one inference pass into two submissions costs 4.1 percent.
Live, next to the renderer and the other eye, the same split raises render
submissions from 10.83 to 12.87 per second and cuts the median worker latency
from 149.51 to 119.25 ms. Meganeura records and submits chunk A, then B, and
Blade holds its queue lock only during each submit, so the renderer can get in
between. An application that owns its whole GPU stack can make this call.

If asked:
- Worker latency runs from frame receipt through upload, recording,
  submission, GPU wait, readback and CPU colour conversion. It leaves out the
  camera and the display, so camera-to-display latency is longer.
- The archive has aggregate timings and no GPU trace, so the mechanism is
  inferred from the code.
- More chunks help less, and unevenly.

Sources: ../main.tex, Deployment Beyond the Matrix; ../../dinovision-section.tex;
DinoVision dc35cdf src/inference.rs:302–352; Blade 60c2eeb vulkan/mod.rs:676.
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
Backup, for "why do you care about training on the device?"

In continual RL the agent acts, sees what happened, receives a reward and
changes its policy. Running a frozen policy covers only the inference half of
that loop. For a robot or a wearable, local learning has to fit the hardware it
carries, and Vulkan and Metal reach that hardware while sharing the
application's GPU stack.

This work covers the runtime. It shows no continual-RL algorithm, learning
curves, on-robot updates or power measurements, and the timed training step
excludes the optimizer update.
-->
