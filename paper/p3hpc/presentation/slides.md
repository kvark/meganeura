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

<!-- header: 'Motivation' -->
<!-- _class: opening -->

# One GPU, two stacks

<div class="game-intro"><div>

Games and headsets draw through **Vulkan** or **Metal**.

ML adds a vendor stack: **CUDA, ROCm, oneAPI or MPS**.

> Can Vulkan and Metal carry training and inference too?

<div class="agenda">
<div><b>Productivity</b>Simple to ship?</div>
<div><b>Portability</b>Every GPU?</div>
<div><b>Performance</b>How fast?</div>
</div>

</div><figure>
<img src="assets/unreal-tournament.jpg" alt="Unreal Tournament gameplay: a first-person view with a rocket launcher, facing an opponent with a minigun">
<figcaption>Unreal Tournament (1999), Epic Games / Digital Extremes · GOG store screenshot</figcaption>
</figure></div>

<!--
Time: 1:00.

GPUs grew up rendering games. A game, a simulator, a robot or a headset
already has a GPU, a driver, buffers and a queue, through Vulkan or Metal.

Adding machine learning usually adds a second stack, one per vendor: CUDA on
NVIDIA, ROCm on AMD, oneAPI on Intel, MPS on Apple, usually with Python on top.
None of them runs on a Quest headset.

So can the graphics API the application already uses carry inference, and
training as well? The talk answers in the workshop's three parts.
Productivity: is it simple to build and ship? Portability: does one engine
reach every GPU? Performance: how fast is it next to PyTorch? The header shows
where we are.

If asked:
- The screenshot illustrates where graphics hardware came from. Source and
  attribution: assets/README.md.
- Why on the device: backup "Learning on the device".
-->

---

<!-- header: '**Productivity** · Portability · Performance · Conclusion' -->
<!-- _class: stack-compare -->

# One compact stack

<div class="stacks">
<div class="stack-mega">
<h3>Meganeura</h3>
<div class="layer"><b>Meganeura</b> compiler + runtime</div>
<div class="layer"><b>Naga</b> WGSL → SPIR-V · MSL</div>
<div class="layer"><b>Blade</b> Vulkan · Metal</div>
<div class="layer driver">Vulkan or Metal driver</div>
<p class="size"><strong>19 MiB</strong> executable</p>
</div>
<div class="stack-torch">
<h3>PyTorch</h3>
<div class="layer"><b>Python</b></div>
<div class="layer"><b>PyTorch</b> Inductor · Triton</div>
<div class="layer"><b>CUDA · ROCm · oneAPI · MPS</b></div>
<div class="layer driver">GPU driver</div>
<p class="size"><strong>4.42 GiB</strong> 29 packages</p>
</div>
</div>

**52k lines of Rust, 6.5k of WGSL.**

<!--
Time: 1:15.

On the left is everything Meganeura needs. Meganeura is the graph compiler
and runtime. Its kernels are written in WGSL; Naga, the shader translator from
the wgpu project, turns them into SPIR-V for Vulkan or Metal Shading Language.
Blade is a thin Rust layer over Vulkan and Metal that I wrote for rendering;
Meganeura uses its compute half. Below it is the driver the system already
has.

The benchmark runner built from this is one 19 MiB executable. It links only
libc, libm and libgcc_s, and loads the GPU driver at run time.

On the right is the stack we compare against. In the measured CUDA
environment, torch's dependency closure is 29 packages and 4.42 GiB.

The whole compiler is about 52 thousand lines of Rust and 6.5 thousand lines
of WGSL. And because Blade also renders, an application can draw and run its
models on one device and one queue; the portability part shows that on a
headset.

If asked:
- Runner: Inferena add19182, 20,072,584 bytes, 14.6 MiB stripped, rustc
  1.98.0. Torch and Triton alone are 1.69 GiB.
- Line counts exclude blank lines, comments and test modules: 52,175 Rust and
  6,481 WGSL lines in 106 shader files.
- Blade binds shader data by field name, separates passes with global
  barriers, and leaves checking to the API's validation layers.
- Pins: Blade 4845d65 and a Naga fork, kvark/wgpu 13f582d, which carries some
  matrix and BF16 features that were not upstream yet.
- Measured: Vulkan on Linux, Windows (the RTX 3050) and Android, Metal on
  macOS. Blade also runs on iOS.

Sources: ../RESULTS.md, Footprint; ../artifact/footprint.py;
data/hardware.json native_runner.
-->

---

<!-- _class: client-code -->

# How do we write a model?

<div class="client"><div>

```rust
use meganeura::{Graph, TrainConfig, Trainer, build_session, nn};

let mut g = Graph::new();
let x = g.input("x", &[32, 784]);
let labels = g.input("labels", &[32, 10]);
let fc1 = nn::Linear::new(&mut g, "fc1", 784, 128);
let h = fc1.forward(&mut g, x);
let h = g.relu(h);
let fc2 = nn::Linear::new(&mut g, "fc2", 128, 10);
let logits = fc2.forward(&mut g, h);
let loss = g.cross_entropy_loss(logits, labels);
g.set_outputs(vec![loss]);

let mut session = build_session(&g); // autodiff + compile
session.set_parameter("fc1.weight", &init);
let mut trainer = Trainer::new(session, TrainConfig::default());
trainer.train(&mut mnist.loader(32), 10);
```

</div><div class="formats">
<h3>Or import</h3>
<div><b>ONNX</b><span>graph + weights</span></div>
<div><b>NNEF</b><span>Khronos graph + weights</span></div>
<div><b>GGUF</b><span>quantized LLMs</span></div>
</div></div>

**Built or imported, it is the same graph**, for training or inference.

<!--
Time: 0:50.

This is a whole training program. Plain Rust builders make a small network:
two linear layers with a ReLU, and a cross-entropy loss. Shapes and types are
checked as each node is added, so a mismatch points at the line that caused
it. build_session differentiates the graph, optimizes it, generates the
kernels and creates the pipelines on the GPU it finds. The trainer then runs
forward, backward and the optimizer update for each batch. There is no Python
and no export step; the five benchmark models are written this way.

Models can also be imported. ONNX and Khronos's NNEF carry a graph and its
weights. GGUF, the llama.cpp format, carries weights and metadata but no
graph: Meganeura builds the transformer the file describes, keeps quantized
weights packed, and wraps the sessions and tokenizer in a text generator.
Either way the result is the same graph, so it can be trained or run.

If asked:
- Parameters start at zero; init holds the first layer's random initial
  weights, and mnist is an MnistDataset. Both are the application's data.
- The default optimizer is SGD; Optimizer also offers Adam.
- GGUF architectures: the Llama family (SmolLM2, Mistral, TinyLlama), Qwen2
  and Qwen3, Gemma 1 to 4. Weight types: f32, f16, BF16, Q4_0, Q4_1, Q8_0 and
  Q3_K to Q6_K.
- An ONNX or NNEF operator without an equivalent fails at import with its
  name. The importers are fuzzed so malformed files return errors.
- Outside the search, building a session takes a median 1.1 s per benchmark
  workload: backup "How do we write the kernels?".
- Loading and running an ONNX file: backup "Importing a model".

Sources: examples/mnist.rs; src/graph.rs, src/nn.rs, src/train.rs;
src/load/onnx.rs, src/load/nnef.rs, src/load/gguf/; src/models/;
tests/importer_fuzz.rs.
-->

---

<!-- _class: testing -->

# How do we know it is right?

**Reference:** a separate CPU f64 interpreter

<div class="checks">
<div class="what">Ops</div>
<div class="side">GPU result<small>every element</small></div>
<div class="vs">⇄</div>
<div class="side ref">f64 interpreter</div>
<div class="what">Derivative rules</div>
<div class="side">Symbolic gradient</div>
<div class="vs">⇄</div>
<div class="side ref">Finite differences</div>
<div class="what">Compiler passes</div>
<div class="side">Optimized graph<small>random graphs too</small></div>
<div class="vs">⇄</div>
<div class="side ref">Original graph</div>
</div>

Bounds scale with **Σ|aᵢbᵢ|** · unwritten buffers start as **NaN**

<div class="caught"><b>Caught:</b> dropped gradients · wrong loss reductions · cancellation in norms · a headset layout bug</div>

<!--
Time: 1:15.

When two engines disagree, comparing them cannot tell you which one is right.
So the reference is a plain CPU interpreter in f64, written separately from
the GPU lowering; only shapes and metadata are shared.

Each row is one question, with the thing under test on the left and its
reference on the right. Does each op compute the right thing on the GPU? Run
it through the whole GPU path and compare every output element with the
interpreter, tile edges included. Is its derivative rule right? Compare the
symbolic gradient with finite differences, with each op under a weighted
loss, so a backward pass that assumes an all-ones upstream gradient fails.
Does the compiler preserve meaning? Run composed and
random graphs through rewrites, fusion and memory planning, and compare with
the original graph.

Error bounds scale with the size of each sum's terms, which allows for
cancellation. Buffers start filled with NaN, so a missing write shows up.

These tests found dropped parameter gradients, loss reduction mistakes,
cancellation in norms, and a headset layout bug whose images still looked
plausible. They are what makes it safe to change the compiler quickly.

If asked:
- An element passes when |got − want| ≤ 2·10⁻⁴ · (sᵢ + floor · max s), where
  sᵢ comes from Σ|aᵢbᵢ|.
- NaN fill: SessionOptions::poison, or MEGANEURA_POISON=1.
- These tests cover chosen cases and random graphs; they prove nothing about
  other inputs.
- The benchmark runs check each selected program again before timing it:
  backup "What the checks admit".

Sources: ../main.tex, Checking the Compiler and Its Kernels; docs/testing.md.
-->

---

<!-- _class: chart search-loop -->

# How do we pick the fastest program?

![Search inputs (graph forms, layouts, fusion, matrix types, submissions) feed four steps for each candidate: build, validate, tune and measure. A candidate that fails validation is dropped; the winner is the fastest validated program.](figures/search.svg)

**30 s per session.** 52% of sessions pick a non-default program.

<!--
Time: 1:15.

The model code we wrote has nothing GPU-specific in it. Search is
what lets the same code run well on each GPU, with nobody tuning it by hand
for one. The compiler keeps equivalent forms of the graph in an e-graph, and
the search varies graph forms, layouts, fusion, matrix input types and the
number of submissions.

For each candidate it builds a private session with initialized weights and
inputs, and validates every output and gradient against the original f32
build. A failure drops the candidate. It then tunes the candidate's kernels
by timing their variants, and measures the whole program against the best so
far, recording and submitting fresh commands each time.
A kernel variant must win by 5 percent, a whole program by 1 percent plus the
measured noise. Kernel decisions are saved and reused; the program race runs
on every build.

In the October runs, 588 sessions measured 23,669 programs. In 306 of them,
52 percent, the winner differs from the ordinary extracted program, so the
alternatives do get used.

If asked:
- Up to 16 graph forms and 64 programs per session. In 428 sessions (73%)
  the deadline arrived before every program was measured.
- Thirteen programs failed the numerical checks and 34 trials hit the 1 GiB
  scratch limit for kernel probes.
- The search costs about a minute per workload: backup "What search costs".
- The October runs have no search-off comparison. The paper's September
  ablation measured 1.18–1.55× from search on two GPUs.
- Example forms: backup "Which form is faster?". Where candidates stop:
  backup "Where paths stop".

Sources: ../main.tex, What it explores; ../tables/search.tex; data/summary.json.
-->

---

<!-- header: 'Productivity · **Portability** · Performance · Conclusion' -->
<!-- _class: features -->

# Which hardware features do we reach?

<div class="reach"><div>

<div class="tier used"><h3>Used</h3><div class="chips">
<span>Matrix tiles</span><span>Workgroup matrices</span><span>f16 · BF16</span><span>Subgroup reductions</span><span>8-bit dot products</span><span>GPU timestamps</span>
</div></div>

<div class="tier unused"><h3>Exposed, not used yet</h3><div class="chips">
<span>FP8 matrices</span><span>Cooperative vectors</span><span>Tensor addressing</span>
</div></div>

</div><div>

<table class="tiles">
<tr><th>Matrix inputs</th><th>f16 tile shapes</th><th>f16</th><th>BF16</th><th>f32</th></tr>
<tr><td>RTX 5070 · H100 · RTX 3050</td><td class="tile">16×16×16<br>16×8×16 · 16×8×8</td><td class="dot">●</td><td class="dot">●</td><td class="no">○</td></tr>
<tr><td>RX 7900 XT</td><td class="tile">16×16×16</td><td class="dot">●</td><td class="no">○</td><td class="no">○</td></tr>
<tr><td>MI300X</td><td class="tile">16×16×16</td><td class="dot">●</td><td class="no">○</td><td class="dot f32">●</td></tr>
<tr><td>Arc B570</td><td class="tile">8×16×16</td><td class="dot">●</td><td class="dot">●</td><td class="no">○</td></tr>
<tr><td>Apple M3</td><td class="tile">8×8×8</td><td class="dot">●</td><td class="dot">●</td><td class="dot f32">●</td></tr>
</table>

</div></div>

<!--
Time: 1:20.

Vulkan and Metal reach more of the hardware than people expect. The matrix
units, through Vulkan's cooperative-matrix extension or Metal's simdgroup
matrices; on NVIDIA also the workgroup-scope extension, where several
subgroups share one larger tile. f16 and BF16 arithmetic and storage.
Subgroup reductions in the matrix-vector kernels, packed 8-bit dot products
for quantized weights, and GPU timestamps for profiling. Blade reports each
as a capability, and the compiler generates kernels only for what the device
reports.

The RTX 5070's driver offers more that we do not use yet: FP8 matrices,
cooperative vectors and tensor addressing.

On the right, the matrix input types each GPU offers, and the f16 tile
shapes. NVIDIA's driver offers several shapes; the others offer one.
Accumulation is f32 everywhere. Only the MI300X, through our driver work, and the M3 take f32
inputs, so those two are the only ones that use matrix units in strict f32.

If asked:
- The packed 8-bit dot products serve GGUF-quantized models; the benchmarks
  use f32 weights.
- The B570's 8×16×16 f16 and BF16 tiles are searched, but none of its
  selected programs used them in these runs: backup "Matrix dispatches by GPU".
- The RX 7900 XT's RADV build in these runs offers no BF16 matrices.
- Shapes queried on the RTX 5070 and B570 on October 30. NVIDIA: f16 at
  16×16×16, 16×8×16 and 16×8×8; BF16 and FP8 at one shape each; 8-bit
  integers at 16×16×32 and 16×8×32. B570: 8×16×16 for f16 and BF16. The
  selected NVIDIA programs use 16×16×16. H100 and RTX 3050 rows assume the
  same driver's shapes.
- What only the vendor APIs reach is on the next slide.

Sources: data/hardware.json; run records (cooperative_* fields); Blade
4845d65 blade-graphics/src/lib.rs, Capabilities; Meganeura c19cb1c
src/shaders/matmul_gemv.wgsl, src/codegen/workgroup.rs.
https://docs.vulkan.org/features/latest/features/proposals/VK_NV_cooperative_matrix2.html
-->

---

<!-- _class: limits-wide -->

# Same GPU, different limits

<table class="limits">
<tr><th>GPU</th><th>Shared memory</th><th>Only through the vendor API</th></tr>
<tr><td>RTX 5070 · RTX 3050</td><td><div class="bars"><span>CUDA</span><i class="vendor" style="width:34.9%"><b>99 KiB</b></i><span>Vulkan</span><i class="graphics" style="width:16.9%"><b>48 KiB</b></i></div></td><td>TF32 matrix inputs</td></tr>
<tr><td>H100</td><td><div class="bars"><span>CUDA</span><i class="vendor" style="width:80%"><b>227 KiB</b></i><span>Vulkan</span><i class="graphics" style="width:16.9%"><b>48 KiB</b></i></div></td><td>TF32 · warpgroup MMA · clusters</td></tr>
<tr><td>Arc B570</td><td><div class="bars"><span>Level Zero</span><i class="vendor" style="width:45.1%"><b>128 KiB</b></i><span>Vulkan</span><i class="graphics" style="width:16.9%"><b>48 KiB</b></i></div></td><td>—</td></tr>
<tr><td>RX 7900 XT · MI300X</td><td><div class="bars"><span>HIP</span><i class="vendor" style="width:22.6%"><b>64 KiB</b></i><span>Vulkan</span><i class="graphics" style="width:22.6%"><b>64 KiB</b></i></div></td><td>BF16 matrix inputs</td></tr>
<tr><td>Apple M3</td><td><div class="bars"><span>Metal</span><i class="graphics" style="width:11.3%"><b>32 KiB, both engines</b></i></div></td><td>—</td></tr>
</table>

**Tiles sized for CUDA or Level Zero may not fit through Vulkan.**

<!--
Time: 1:10.

The same silicon can look different through two APIs. The biggest difference
is shared memory per workgroup, which bounds tile sizes. On NVIDIA, CUDA lets
a kernel opt into 99 KiB on the consumer cards and 227 on the H100; NVIDIA's
Vulkan driver offers 48. On the Arc B570, Level Zero offers 128 and Vulkan 48.
On AMD, HIP and Vulkan both offer 64, and on Apple both engines run on Metal,
so the limit is the same.

Some instructions are also reachable only from the vendor API: TF32 on
NVIDIA, and on Hopper the asynchronous warpgroup matrix instructions and
thread-block clusters. AMD's GPUs have BF16 matrix instructions, which ROCm
PyTorch uses, but the RADV builds in these runs do not expose them. A kernel
designed around the larger limits or those instructions has to be redesigned
for Vulkan.

If asked:
- Queried values: 101,376 bytes through CUDA opt-in on the RTX 5070 (48 KiB
  by default), 131,072 through Level Zero, 49,152 through Vulkan on both
  (October 15 and 19).
- The H100, RTX 3050, AMD and Apple values come from vendor documentation and
  a public RADV report, not from queries on our hosts: data/limits.json.
- The workgroup-scope matrix extension reserves 8 KiB of the 48 for the
  compiler on the RTX 5070.
- The B570's Vulkan driver exposes the same f16 and BF16 matrix inputs as
  its hardware; we found nothing matrix-related missing there.
- For the Arc U-Net, the limit is not the cause: its largest kernel uses
  16 KiB (data/b570-unet-shared-memory.txt).

Sources: data/limits.json; data/hardware.json;
https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions
-->

---

<!-- _class: deployment -->

# One GPU, one stack

![bg right:40% contain](../../figures/dinovision-quest3s.jpg)

**DinoVision · Quest 3S**

Decoder trained with **Adam** on a desktop GPU

Headset: reconstructs both eyes' camera frames

**ML and OpenXR rendering share one queue**

<small>Image: the headset's reconstruction</small>

<!--
Time: 1:00.

At the small end is the device from the opening: one GPU, now with one
stack. DinoVision runs on a Quest 3S, an Android headset with an Adreno GPU. A DINOv3 encoder feeds a small decoder
that reconstructs the camera image; the picture is its output. The decoder
was trained with Adam on a desktop Vulkan GPU, saved, and loaded on the
headset. Two eye workers share Blade's Vulkan context and queue with the
OpenXR renderer, and there is no second GPU runtime.

If asked:
- 2.36 billion multiply-accumulates per pass.
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

<!-- _class: mi300 -->

# MI300X needed a driver

<div class="columns mi300-columns"><div>

- No Vulkan driver supports it
- RADV: compute + f16 / f32 matrix tiles
- AMDGPU patch: all eight dies
- Draft: Mesa MR 45239

</div><div>

<div class="big-number"><strong>42 / 42</strong><span>results pass, up to SmolLM2-1.7B</span></div>

</div></div>

<p class="mi300-dies"><b>All eight dies:</b> strict f32 training of SmolLM2-1.7B sustains about 37 TFLOP/s, nearly twice one die's peak.</p>

<!--
Time: 1:00.

At the other end is the MI300X, a compute-only accelerator. No Vulkan driver
supports it, so the paper said Meganeura could not run there. We extended RADV, Mesa's Vulkan
driver, building on Mesa's experimental CDNA work: buffer-based compute and
cooperative matrices, including f32 tiles. A patch to the AMDGPU kernel
driver spreads the work across all eight compute dies. The unchanged engine
and harness then pass all 42 results, from the five core models up to the
1.7-billion-parameter language model. The Mesa changes are posted as a draft
merge request, with the kernel patch attached.

Spreading across the dies matters: strict f32 training of the 1.7-billion
model sustains about 37 TFLOP/s, nearly twice the f32 peak of one die.

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

<!-- header: 'Productivity · Portability · **Performance** · Conclusion' -->
<!-- _class: measure-new -->

# How do we measure fairly?

<div class="measure-rows">
<div><b>Models</b><span class="kinds"><i><b>LLM</b>SmolLM2-135M</i><i><b>Robot policy</b>SmolVLA</i><i><b>Image diffusion</b>SD U-Net</i><i><b>Vision</b>ResNet-50</i><i><b>Speech</b>Whisper-tiny</i></span></div>
<div><b>GPUs</b><span class="gpu-groups"><i><b>NVIDIA</b>RTX 5070 · H100 · RTX 3050</i><i><b>AMD</b>RX 7900 XT · MI300X · <u>780M · 9600X</u></i><i><b>Intel</b>Arc B570 · <u>RPL-U</u></i><i><b>Apple</b>M3</i><i><b>Qualcomm</b><u>Quest 3S</u></i></span></div>
<div><b>Baseline</b><span>PyTorch 2.13 · <code>torch.compile</code> + graph replay</span></div>
<div><b>Each run</b><span class="flow">Check → warm up 10 s → time 20 calls → 3 fresh process pairs</span></div>
</div>

<div class="contracts-pair">
<div><h3>Strict f32</h3>f32 everywhere, TF32 off</div>
<div><h3>Reduced precision</h3>Each engine's own recipe</div>
</div>

<small>All checks pass on all 11 GPUs. Underlined: checked for correctness only, not timed.</small>

<!--
Time: 1:15.

Five models, one of each common kind: a small language model, SmolLM2; the
action expert of a robot policy, SmolVLA; a scaled image-diffusion U-Net;
ResNet-50 for image classification; and Whisper's speech encoder. We time
inference at the full shape and a training step: forward, loss and backward,
without the optimizer update.

The same engine passes every check on eleven GPUs from five vendors, from a
headset to a datacenter accelerator. Seven are timed against PyTorch 2.13
with torch.compile and, where available, graph replay; the three integrated
GPUs and the Quest are checked for correctness only. Our harness, Inferena, runs
both engines. Each must pass its numerical checks before it is timed. Then it
warms up for ten seconds and times 20 calls, each waiting for the GPU to
finish. That repeats in three fresh pairs of processes, alternating which
engine goes first, and we report the median of the three process medians.

Two contracts. Strict f32 means both engines do the same arithmetic. With
reduced precision, each engine uses its own recipe: PyTorch BF16 autocast,
FP16 on Apple; Meganeura f16 or BF16 inputs for forward matrix work.

If asked:
- The engines pass different checks. Meganeura compares full tensors with its
  f32 build. PyTorch must repeat exactly and, with reduced precision, track
  f32 over matched SGD trajectories. Across engines, sampled outputs, loss and
  gradient norms must agree. Backup "What the checks admit".
- max-autotune is off, and PyTorch compilation has a 120 s limit.
- Only SmolLM2 uses trained weights; the others use deterministic uniform
  parameters.
- The runs also time a minimal shape (one token, action or image): backup
  "Minimal shape".
- Correctness-only references: CPU PyTorch f32 for the integrated GPUs, CPU
  f64 full tensors for the Quest; 330 of 330 core results pass. Backup
  "Coverage".

Sources: ../main.tex, Methodology; ../tables/devices.tex.
-->

---

<!-- _class: chart results -->

# Strict f32: same arithmetic

![Strict f32 time ratios for seven GPUs and five workloads, one panel for inference and one for training](figures/strict-ratios.svg)

Medians: <span class="slower">1.35×</span> inference · <span class="slower">1.61×</span> training. Faster in **14 of 68**, and **7 of 9** on the RX 7900 XT.

<small>**Whisper (RTX 5070):** half our time is attention over 1,500 positions.</small>

<!--
Time: 1:20.

Strict f32 is the fair comparison: both engines compute in f32, with TF32 and
reduced-precision matrix inputs turned off. Each cell is our time divided by
PyTorch's; teal means we are faster.

Inference on the three NVIDIA GPUs is within 4 to 13 percent at the median.
On the RX 7900 XT we are faster in 7 of 9 comparisons. Training trails at
1.61: our backward kernels have had the least tuning. The Arc B570 and the
MI300X are where we trail most, and the B570's U-Net, at 5.45, is the case
study in two slides.

If asked:
- Per-GPU inference medians: RTX 5070 1.04, H100 1.13, RTX 3050 1.12,
  RX 7900 XT 0.63, MI300X 2.25, B570 2.28, M3 1.35. Training: 1.51, 1.90,
  1.56, 0.67, 2.31, 3.02, 1.23.
- Missing cells: PyTorch's Whisper training on the RX 7900 XT and ResNet
  training on the MI300X fail its repeatability check.
- Strict fixes the arithmetic width; the order of reductions still differs.
- The M3's spread between processes means its near-ties are ties.
- Larger models on the two servers: backup "Larger models".
- Whisper: a per-pass profile on the RTX 5070 (Meganeura 99e3de2, matrix
  units off, October 30) puts 2.98 of 5.77 ms in the four attention
  dispatches over 1,500 positions. Without Whisper the medians are 1.24 and
  1.55, so it moves them by less than 0.1; it is a kernel to fix, not an
  outlier to drop.

Sources: data/results.csv; data/summary.json.
-->

---

<!-- _class: chart results -->

# With reduced precision

![Reduced-precision time ratios for seven GPUs and five workloads, one panel for inference and one for training](figures/accelerated-ratios.svg)

**LLM (SmolLM2):** <span class="slower">1.87×</span> inference · <span class="slower">2.26×</span> training. PyTorch gains **1.46–1.48×**, we gain **1.04–1.08×**.

<small>**Whisper (RTX 5070):** our matrix-unit attention is no faster than our f32 attention.</small>

<!--
Time: 1:10.

Now each engine may use reduced precision, which is how these models usually
run in practice. PyTorch uses BF16 autocast, or FP16 on Apple. Meganeura may
use f16 or BF16 inputs for forward matrix work, keeps f32 storage, and keeps
derivatives in f32. Both still have to pass their numerical checks, and TF32
stays off.

Take the language model, SmolLM2: the median ratio is 1.87 for inference
and 2.26 for training, because reduced precision helps PyTorch by half again
and us by a few percent. Across all five models the medians are 2.06 and
2.21. Autocast moves PyTorch's matrix work, forward
and backward, onto the matrix units; our search seldom selects matrix kernels
on NVIDIA and never on the B570, and our derivatives stay in f32. The Apple M3
goes the other way: there PyTorch's FP16 path is often slower than its f32
path.

If asked:
- The gains are medians over cells where each engine qualified under both
  contracts. Per GPU, PyTorch gains 1.22–3.12 on NVIDIA, the RX 7900 XT and
  the B570; on the MI300X and M3, 0.76 and 1.08 for inference, 0.85 and 0.69
  for training. Backup "Who gains from reduced precision?".
- Missing cells are PyTorch failures. On the H100, the U-Net's maximum error,
  0.02255, exceeds its 0.02085 bound. On the MI300X, the U-Net and Whisper
  fail repeatability.
- Meganeura is faster in 8 of 64 comparisons.
- SmolLM2 numbers are medians over the seven GPUs; the gains are each
  engine's strict time over its reduced-precision time. All models: PyTorch
  1.50 and 1.56, Meganeura 1.03 and 1.01.
- Whisper on the RTX 5070: PyTorch drops from 3.58 to 1.06 ms, we from 5.59
  to 4.58. In the October 30 profile our matrix-unit attention takes 3.30 ms
  against 2.98 for the f32 kernel.
- Accounting on identical conditions: backup "Two gaps". Rules: backup
  "Reduced-precision rules".

Sources: data/results.csv; data/summary.json; ../NUMERICAL-POLICY.txt;
../main.tex, Results.
-->

---

<!-- _class: ops -->

# Where is the gap? One op at a time

<p class="case-context">Strict f32, one GEMM or convolution, our GPU time ÷ PyTorch's. The vendor libraries run on the same f32 units we use.</p>

<table class="ops">
<tr><th>Operation</th><th>Shapes</th><th>RX 7900 XT</th><th>RTX 5070</th><th>Arc B570</th></tr>
<tr><td>Large GEMM</td><td>4096³</td><td>1.7×</td><td>2.1×</td><td>2.2×</td></tr>
<tr><td>Transformer GEMMs</td><td>Speech, LLM</td><td>1.3×</td><td>1.5–1.7×</td><td>1.2–1.5×</td></tr>
<tr><td>Small GEMMs</td><td>Diffusion, M = 64–256</td><td class="near">0.65–1.5×</td><td class="far">1.6–4.9×</td><td class="far">2.0–4.6×</td></tr>
<tr><td>3×3 convolutions</td><td>Diffusion, Vision</td><td>0.8–2.0×</td><td class="near">0.45–1.0×</td><td class="far">1.45–3.2×</td></tr>
<tr><td>Other convolutions</td><td>1×1, stems</td><td class="near">0.6–1.2×</td><td>1.4–2.0×</td><td>1.4–1.75×</td></tr>
</table>

**Diagnosis:** our kernels, worst on the small matrices of the image-diffusion U-Net (5.45× on the Arc). **Next:** split-K and smaller tiles for small matrices.

<!--
Time: 1:30.

To find where the gap is, we timed single operations against PyTorch: one
GEMM or one convolution, strict f32, GPU time only, on three GPUs. In this
mode the vendor libraries use ordinary f32 kernels, CUTLASS on NVIDIA, oneDNN
on Intel and the ROCm libraries on AMD, so we compare kernels on the same
execution units.

Large GEMMs: the vendor libraries reach 1.7 to 2.2 times our throughput.
Transformer-sized GEMMs from Whisper and the LLM are closer. The small GEMMs
in the diffusion U-Net are where we lose most on NVIDIA and Intel: a 64-row
matrix product takes 28 microseconds for us and 6 for cuBLAS, and 55 against
12 on the Arc. There is too little work per dispatch for our tiles, and
cuBLAS splits the K dimension for these shapes.

Convolutions split by vendor. On the RTX 5070 our Winograd 3×3 convolutions
match or beat cuDNN; on the Arc, oneDNN's are 1.45 to 3.2 times faster. The
RX 7900 XT is our closest GPU, as in the model results: we win some small
GEMMs and convolutions there and lose others by up to 2×.

So the U-Net gap sits in our small-matrix and small-convolution kernels. The
fix is ours: split-K and smaller tiles.

If asked:
- Large GEMM: 13.3 against 22.3 TFLOP/s on the RX 7900 XT, 10.2 against
  21.6 on the RTX 5070, 3.2 against 6.9 on the B570. This shows our kernel
  does not yet reach the vendor libraries' f32 throughput; it does not show
  that Vulkan can.
- 4096³ needs more than the tuner's 64 MiB of scratch, so all three GPUs run
  our default tile. 2048³, which the tuner covers, is 2.0× on the RTX 5070
  and 1.9× on the Arc.
- Each process tunes for 2 s and may pick a different tile. On the RTX 5070
  and Arc we report the median of three processes; small Arc ops vary by up
  to 45% between them.
- Timing: our GPU time sums Vulkan pass timestamps; PyTorch's is a device
  interval around a graph of 30 serial ops, divided by 30, with the profiler
  off (profiling slows ROCm down). The PyTorch interval includes dispatch gaps
  between kernels, which matters most for the smallest ops, especially on the
  Arc.
- Shared memory is not the reason: the U-Net's largest generated kernel uses
  16 KiB of Vulkan's 48.
- Convolution times include our per-call Winograd weight transform.
- Earlier diagnostics on the U-Net: similar kernel counts (286 against 276),
  and F16 Winograd storage made the whole inference 9.3% faster.
- Current code (October 31: Meganeura 5b232bd on the RX 7900 XT, e40648f
  with the tuner scratch fix on the others), not the cohort's revision; do
  not combine with the cohort ratios. PyTorch 2.13.0 (CUDA 13.0, XPU, ROCm
  7.2), TF32 off, cudnn.benchmark on.

Sources: data/op-bench.tsv; data/op-bench/; examples/op_bench.rs;
data/b570-unet-shared-memory.txt; data/intel-gap.json; data/results.csv.
-->

---

<!-- _class: since-paper -->

# Since the paper

**Without the noisy M3, PyTorch's times moved under 3%; ours fell 6.5% (inference) and 15% (training).**

| Strict f32 medians · six GPUs in both | Paper · Sept 25 runs | Talk · Oct 19 runs |
|---|---:|---:|
| Inference | 1.27× | 1.13× |
| Training | 1.92× | 1.55× |

**Also changed:** MI300X added · BF16 autocast, not TF32 · 30 s search · 10 s warmup

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
session, and warmup grew from 2 to 10 seconds. So part of the difference may
come from the protocol.

If asked:
- Without the M3: 1.21× → 1.13× for inference, 1.99× → 1.59× for training.
  Minimal shape: 1.09× → 1.00×.
- The paper reported 2.00 and 2.61 with reduced precision; now 2.06 and 2.21
  on all seven GPUs.
- One regression: U-Net inference on the B570 went from 2.97 to 5.89 ms.
- The benchmark runner grew from 13.5 to 19.1 MiB.

Sources: since_paper.py, which replays the camera-ready supplement
(../submission/meganeura-p3hpc-aux.zip) with the tagged artifact/cohort.py;
data/results.csv.
-->

---

<!-- header: 'Productivity · Portability · Performance · **Conclusion**' -->
<!-- _class: verdict -->

# What does Meganeura show?

**One compact binary trains and runs models on every GPU we tried.**

<div class="verdicts">
<div><h3>Productivity</h3><div class="answer">19 MiB</div><p>One executable</p><p>No runtime dependencies</p></div>
<div><h3>Portability</h3><div class="answer">Mobile to server</div><p>11 GPUs, 5 vendors</p><p>Training passes on all</p></div>
<div><h3>Performance</h3><div class="answer">Within 2×</div><p>Median 1.70× PyTorch's time</p><p>f32 and reduced precision</p></div>
</div>

<!--
Time: 0:40.

First, what Meganeura itself shows. One 19 MiB executable trains and runs
these models, from a Quest headset to an MI300X.

Productivity: it is one executable with no runtime dependencies beyond the C
library and the GPU driver the system already has. No Python, no vendor SDK.
Portability: the same engine passes every check, training included, on
eleven GPUs from five vendors. Performance: across all 132 inference and
training comparisons, in both precisions, our median time is 1.70 times
PyTorch's. In strict f32 it is 1.35 for inference and 1.61 for training; with
reduced precision it is just over 2, because PyTorch gains much more from it.


If asked:
- Runner: Inferena add19182, 20,072,584 bytes, SHA-256 b8d25fac…. It links
  libc, libm and libgcc_s, and loads the GPU driver at run time.
- 1.70× is the median over the five core models' inference and training
  comparisons on the seven timed GPUs, 68 strict and 64 reduced-precision;
  the geometric mean is 1.81×. 76 of the 132 are under 2×.
- Training on the headset and the three integrated GPUs is checked for
  correctness only (forward, loss and backward); DinoVision's full Adam
  cycle ran on a desktop GPU.
- MI300X uses experimental drivers.

Sources: data/results.csv; data/summary.json; data/hardware.json;
../QUALIFICATION.md.
-->

---

<!-- _class: verdict -->

# Can Vulkan and Metal reach the vendor stacks?

**Some GPUs already match. Better kernels and drivers can close the rest.**

<div class="verdicts">
<div><h3>Reached</h3><div class="answer">Faster</div><p>RX 7900 XT, Apple M3: faster in 18 of 39</p><p>Strict f32 and reduced precision</p></div>
<div class="open"><h3>Kernels</h3><div class="answer">Ours to fix</div><p>Arc U-Net: convolution and matrix kernels</p><p>Whisper: attention over long sequences</p></div>
<div class="open"><h3>Drivers</h3><div class="answer">Expose more</div><p>TF32 · BF16 matrix inputs on AMD</p><p>MI300X needed a patch to run</p></div>
</div>

[github.com/kvark/meganeura](https://github.com/kvark/meganeura)

<!--
Time: 0:40.

Now the question in the title. Can a graphics API reach what the vendor
stacks deliver? On some configurations it already does. On the RX 7900 XT and
the Apple M3 we beat PyTorch in 18 of 39 comparisons, in strict f32 and with
reduced precision.

Where we are behind, the gap splits in two. Part of it is our own kernels: on
the Arc the U-Net loses in convolution and matrix kernels, and Whisper loses
in attention over long sequences. Part of it is what the drivers expose:
TF32 and AMD's BF16 matrix inputs are not reachable, and the MI300X needed a
driver patch to run at all. Both halves can be fixed without leaving the
graphics API. Thank you.

If asked:
- Wins: RX 7900 XT 7 of 9 strict and 3 of 10 reduced precision; Apple M3 3 of
  10 strict and 5 of 10 reduced precision. Five core models, inference and
  training.
- ROCm PyTorch on a consumer RDNA3 GPU may be less tuned than CUDA PyTorch,
  so part of the RX 7900 XT result is a weaker baseline.

Sources: data/results.csv; data/whisper-profile.txt;
data/intel-gap.json.
-->

---

<!-- header: 'Backup' -->
<!-- _class: backup backup-index -->

# Backup

<div class="columns backup-list"><div>

**Productivity**
[Importing a model](#19) · [Inside Meganeura](#20) · [How do we describe a model?](#21) · [Which form is faster?](#22) · [How do we write the kernels?](#23) · [Gradients](#24) · [Compile stages](#25)

**Portability**
[Matrix dispatches by GPU](#26) · [Where paths stop](#27) · [What search costs](#28) · [Reusing kernel decisions](#29) · [Coverage](#30)

</div><div>

**Performance**
[Minimal shape](#31) · [Who gains from reduced precision?](#32) · [Two gaps](#33) · [Larger models](#34) · [Change one path](#35) · [Reduced-precision rules](#36) · [What the checks admit](#37) · [Where TF32 fits](#38) · [SPIR-V precision](#39) · [Timing variability](#40) · [Warmup](#41) · [Portability scores](#42)

**Deployment**
[Queue measurements](#43) · [Learning on the device](#44)

</div></div>

<!--
Backup index. Each title links to its slide in the HTML version.
-->

---

<!-- _header: 'Backup · Productivity' -->
<!-- _class: api-code -->

# Importing a model

An ONNX file becomes the same graph, ready to compile and run.

```rust
use meganeura::{build_inference_session, load_onnx};
use std::path::Path;

let model = load_onnx(Path::new("resnet50.onnx"))?;
let mut session = build_inference_session(&model.graph);
for (name, data) in &model.weights {
    if session.has_parameter(name) {
        session.set_parameter(name, data);
    }
}
session.set_input("input", &image);
session.step();
session.wait();
let logits = session.read_output(1000);
```

<small>NNEF loads the same way with load_nnef. GGUF builds a text generator: load_gguf(path)?.generator(2048)?.</small>

<!--
Backup, for "can I bring an existing model?"

load_onnx translates the ONNX graph into Meganeura's graph and returns the
initializers as named weights. The has_parameter check skips any that the
compiled graph does not keep as parameters. The session then runs like any
other.

If asked:
- An operator without an equivalent fails at import with its name.
- GGUF has no graph; the loader builds the transformer the metadata
  describes and keeps quantized weights packed.

Sources: src/load/onnx.rs, src/load/nnef.rs, src/load/gguf/;
examples/onnx_compare.rs, examples/gguf_generate.rs.
-->

---

<!-- _header: 'Backup · Productivity' -->
<!-- _class: chart -->

# Inside Meganeura

![Eight steps from model description to execution: describe, rewrite, differentiate, plan, specialize, compile, check and time, execute](figures/compiler-map.svg)

One compiler for inference and training. **Search revisits these choices by timing them on the target GPU.**

<!--
Backup, for "what does the compiler do?"

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

Sources: ../main.tex, How Meganeura Works; src/graph/composite.rs.
-->

---

<!-- _header: 'Backup · Productivity' -->
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
Backup, for "how are models represented?"

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

<!-- _header: 'Backup · Productivity' -->

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
Backup, for "what does the e-graph actually hold?"

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

<!-- _header: 'Backup · Productivity' -->
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
Backup, for "how are kernels written?"

Every kernel is WGSL. Templates have sections and substitution points: this
one is the matrix-vector product used for decoding, and the generator fills in
the weight storage type, the vector width, how weights are loaded and how a
row is reduced, from the shapes, the device and the precision policy.
Pointwise operations need no template: a chain becomes one kernel, or folds
into a matrix product's epilogue. Matrix and attention kernels have
generators for each device's tile shapes.

Is generating source at build time slow? No. Everything outside the search,
including building the f32 reference program, takes a median 1.1 seconds per
workload. The search is what takes a minute.

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

<!-- _header: 'Backup · Productivity' -->

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
In the measured cohort, reduced precision in the forward pass does not
carry over to derivatives. The separate local attention study uses scaled
F16 operands with F32 accumulation without changing that cohort evidence. The
checks compare full gradients, which catch underflow that plausible outputs
can hide.

Sources: src/autodiff.rs, src/memplan.rs, src/train.rs.
-->

---

<!-- _header: 'Backup · Productivity · compile stages, September 10' -->
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

<!-- _header: 'Backup · Portability' -->
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

<!-- _header: 'Backup · Portability' -->
<!-- _class: chart gap-paths -->

# Where paths stop

![Four checkpoints connected left to right: expose, lower, qualify and measure. Each has two rejection branches; surviving candidates reach the selected program.](figures/path-admission.svg)

**An exposed matrix instruction is only the first checkpoint.**

<!--
Backup, for "why doesn't the B570 use its matrix units?"

This is the selection process, not a breakdown of the performance gap. Follow
the teal arrows to keep a candidate; each brown branch stops that candidate.
An instruction can be missing, a tile can exceed shared memory, the result can
fail qualification, or the complete program can lose to the incumbent. The
search deadline can also leave a candidate unmeasured.

The B570's zero selected matrix dispatches locates a failure to use exposed
capabilities in this collection. It is not evidence that Intel's matrix units
are fictitious.

If asked:
- The cohort provides selected dispatch counts, not GPU-time fractions or
  FLOP-weighted matrix utilization.
- The numerical gates differ between the engines. Wider gates might change
  selection; we did not measure a tolerance sweep.

Sources: data/summary.json; data/hardware.json; ../NUMERICAL-POLICY.txt.
-->

---

<!-- _header: 'Backup · Portability' -->
<!-- _class: chart -->

# What search costs

![Preparation by workload: Meganeura's preparation, mostly search, against PyTorch's compilation and first calls](figures/preparation-models.svg)

Meganeura: <span class="mega">49–68 s</span> per workload, almost all of it search. PyTorch compiles in <span class="torch">12–54 s</span>.

<!--
Backup, for "how long does the search take?"

Setup time per workload process in strict f32, as the median across GPUs.
Meganeura spends about a minute, almost all of it in the search; everything
else takes about a second. PyTorch's compilation and first calls take 12 to
54 seconds. The two bars cover different work: ours checks every candidate,
and PyTorch's excludes its graph capture and checks.

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

<!-- _header: 'Backup · Portability' -->
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

<!-- _header: 'Backup · Portability' -->

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

<!-- _header: 'Backup · Performance' -->
<!-- _class: chart results -->

# Minimal shape

![Minimal-shape time ratios for seven GPUs and five workloads, strict f32 and reduced precision](figures/minimal-shape-ratios.svg)

Medians: <span class="slower">1.03×</span> strict · <span class="slower">1.49×</span> reduced precision. Meganeura is faster in **15 of 33** and **13 of 32**.

<!--
Backup, for "what about latency?"

The minimal shape is one token, action or image: the work a real-time
application does per step. It runs after training in each process, so a
PyTorch training failure also removes its minimal-shape cell. In strict f32
we are at parity overall, within 3 percent at the median.

Sources: data/results.csv; data/summary.json.
-->

---

<!-- _header: 'Backup · Performance' -->
<!-- _class: chart -->

# Who gains from reduced precision?

![Per GPU, median speedup of each engine when reduced precision is allowed, for inference and training](figures/precision-gain.svg)

PyTorch gains **1.50×** for inference and **1.56×** for training; Meganeura **1.03×** and **1.01×**.

<!--
Backup, for "why does the gap widen?"

Strict time divided by reduced-precision time, for each engine, median over
the workloads where both engines passed in both profiles. On NVIDIA, the
RX 7900 XT and the B570, autocast moves PyTorch's matrix work in both
directions onto the matrix units, and it gains 1.22 to 3.12. Our selected
NVIDIA programs use matrix kernels in at most 19 dispatches, all in forward
regions; the B570's use none. On the MI300X and M3 PyTorch gains little or
slows down.

Sources: data/results.csv; ../main.tex, Results.
-->

---

<!-- _header: 'Backup · Performance' -->
<!-- _class: chart -->

# Two gaps

![Strict and reduced-precision Meganeura/PyTorch ratios on identical qualified model sets, by GPU and phase](figures/gap-accounting.svg)

**Existing f32 gap × unequal acceleration = reduced-precision gap.** On matched conditions, unequal acceleration multiplies the gap by **1.61× for inference, 1.56× for training**.

<!--
Backup, for "how much of the gap is precision?"

Separate the two questions. There is already a strict-F32 gap. Allowing reduced
precision then changes each engine's time by a different factor. For each
model, the accelerated ratio equals the strict ratio times PyTorch's speedup
divided by Meganeura's speedup. This is an exact accounting identity.

The chart holds the model set fixed for both engines and both profiles.
Geometric means preserve the identity: the matched inference ratio moves
from 1.28 to 2.06, and training from 1.57 to 2.46. The second number differs
from the earlier 2.21 training median because this is a geometric mean on 30
matched conditions, not a median on all 32 accelerated comparisons.

The arrow does not assign a fraction to Vulkan, the driver or the CPU. It
shows how much additional gap comes with the measured precision policies.

If asked:
- The matched engine gains are 1.69/1.05 for inference and 1.59/1.02 for
  training, PyTorch/Meganeura. Do not divide unrelated population medians.
- Strict coverage and accelerated coverage differ. Two accelerated training
  comparisons do not have a qualified strict counterpart and are excluded here.
- M3 process variability and the small MI300X denominator remain limitations.

Sources: ../artifact/gap_analysis.py; data/results.csv; ../tables/gap.json.
-->

---

<!-- _header: 'Backup · Performance' -->
<!-- _class: chart -->

# Larger models

![Strict SmolLM2 prefill and training ratios at 135M, 360M and 1.7B on H100 and MI300X, with process ranges](figures/scaling.svg)

Strict training ratio from 135M to 1.7B: H100 <span class="slower">1.90× → 3.79×</span>, MI300X <span class="slower">2.29× → 1.65×</span>.

<!--
Backup, for "does it scale?"

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

<!-- _header: 'Backup · Performance · separate local experiment' -->
<!-- _class: chart -->

# Change one path

![Whole training-step changes from replacing only attention backward with scaled F16, on RTX 5070 and B570](figures/attention-followup.svg)

**Same Vulkan API, same numerical bounds; different outcomes.** Range handling unlocks a path. Whole-model timing decides whether it wins.

<!--
Backup, for "can you close the gap without changing the API?"

This is a separate local experiment after the cohort, not a replacement for
any of the PyTorch ratios. The graph, forward kernels and submission count
stay fixed. Only attention backward changes, from F32 to scaled F16 operands
with F32 accumulation. Power-of-two scaling preserves tiny derivatives;
centering the keys keeps their common component from corrupting the scores.

Every comparison passes the unchanged full-model output and gradient checks.
Whisper gets 13.3 percent faster on the RTX 5070 but 11.4 percent slower on
the B570. Merely unlocking a matrix kernel is not enough. Search should keep
F32 where it is faster. BF16 is not required for this attention family.

If asked:
- Each condition has one process, ten seconds shared warmup, and 100
  interleaved samples per arm, with profiling off and ordinary recording.
  This does not characterize cross-process variability or learning convergence.
- A non-grid shared-key case exposed 3.35% dV error; score centering fixes it
  under the existing 1% F64 bound. Head widths 16/32/64/128 pass on both GPUs.
- Sharing the common F32 anchor dot over four lanes reduces B570 Whisper
  from 75.12 to 68.86 ms compared with a serial anchor, with unchanged
  operand types and no extra pre-pass. The auxiliary samples are retained.
- Reusing scaled score staging brings head-128 dQ/dKV from 50,816/51,648 to
  47,360/48,128 bytes: it fits the same 48 KiB limit without a new driver.
  That is checked admission, not a full-model head-128 speedup.

Sources: ../artifact/attention-followup.json; PR #279, shader revision 3d7cafa.
https://github.com/kvark/meganeura/commit/3d7cafaa777dd186a07745faaee6a4f81acc24c7
-->

---

<!-- _header: 'Backup · Performance' -->
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

<!-- _header: 'Backup · Performance' -->
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

<!-- _header: 'Backup · Performance' -->
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

<!-- _header: 'Backup · Performance' -->
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

<!-- _header: 'Backup · Performance' -->
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

<!-- _header: 'Backup · Performance' -->
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

<!-- _header: 'Backup · Performance' -->
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

<!-- _header: 'Backup · Deployment' -->
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

<!-- _header: 'Backup · Deployment' -->

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
