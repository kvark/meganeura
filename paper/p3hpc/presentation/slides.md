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
- The numbers come from the Oct 31–Nov 1 runs: Inferena a0799e42, Meganeura
  567028b8, PyTorch 2.13.0 at cf30153c. The published paper used the
  September 25 runs; the "Since the paper" slide compares the two.
-->

---

<!-- header: '**Introduction** · Productivity · Portability · Performance · Conclusion' -->
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

<!-- _class: apis -->

# Vulkan and Metal

<div class="api-cards">
<div><h3>Vulkan</h3><p class="who">Khronos open standard · since 2016</p><p>Windows · Linux · Android</p><p>Drivers from NVIDIA, AMD, Intel, Qualcomm, Arm and open-source Mesa</p><p class="lang">Shaders: <b>SPIR-V</b></p></div>
<div><h3>Metal</h3><p class="who">Apple · since 2014</p><p>macOS · iOS · visionOS</p><p>Every Apple GPU</p><p class="lang">Shaders: <b>Metal Shading Language</b></p></div>
</div>

<div class="api-compute"><p><b>Compute:</b> compute shaders · buffers · workgroup shared memory · subgroups · command queues</p><p><b>Optional, reported per device:</b> matrix tiles · 8-bit dot products · GPU timestamps</p></div>

**Low-level and explicit, built for games. No ML libraries: we write every kernel.**

<!--
Time: 0:45.

For those who do not live in graphics: Vulkan and Metal are the GPU APIs that
games and other applications draw with. Vulkan is the Khronos open standard,
released in 2016, with drivers from every major GPU vendor on Windows, Linux
and Android, including the open-source Mesa drivers. Metal is Apple's API,
since 2014, on every Apple device.

Both are low-level and explicit: the application allocates buffers, records
commands and submits them to a queue. For compute they offer compute shaders,
workgroup shared memory and subgroup operations. Optional features, which
each device reports, add matrix tiles, 8-bit dot products and timestamps.
Shaders are SPIR-V on Vulkan and the Metal Shading Language on Apple.

What they do not have is any ML library: no cuBLAS, no cuDNN. Everything
above the API, including every kernel, is ours.

If asked:
- Vulkan also reaches Apple GPUs through MoltenVK, a translation layer; we use
  Metal directly.
- WebGPU sits above both and has its own shading language: next slide.
- Matrix tiles: VK_KHR_cooperative_matrix on Vulkan, simdgroup matrices on
  Apple GPUs. Backup "Matrix paths selected".
-->

---

<!-- _class: wgsl -->

# WGSL: the WebGPU Shading Language

<div class="wgsl-grid"><div>

```rust
var<storage> indices: array<u32>;
var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.total { return; }
    let row = i / params.embed_dim;
    let col = i % params.embed_dim;
    var sum = 0.0;
    for (var s = 0u; s < params.seq_len; s++) {
        if indices[s] == row {
            sum += src[s * params.embed_dim + col];
        }
    }
    dst[i] = sum;
}
```
<p class="wgsl-caption">Embedding gradient, <code>src/shaders/scatter_add.wgsl</code></p>

</div><div class="wgsl-facts">
<div><b>Standard</b><span>W3C, specified with WebGPU</span></div>
<div><b>Portable</b><span>One source, translated to SPIR-V, MSL or HLSL</span></div>
<div><b>Native</b><span>Runs outside the browser: Naga in wgpu, Tint in Dawn</span></div>
<div><b>Compute</b><span>Storage buffers · workgroup memory · atomics · barriers · <code>f16</code> · subgroups</span></div>
</div></div>

<div class="naga-flow"><span>WGSL kernels</span><i>→</i><span><b>Naga</b> parse · validate</span><i>→</i><span>SPIR-V for Vulkan · MSL for Metal</span></div>

**All 113 kernels are WGSL. No browser involved.**

<!--
Time: 0:40.

Our kernels are written in WGSL, the WebGPU Shading Language. The W3C
specifies it together with WebGPU, the browser's GPU API, but nothing about
it requires a browser. It is a portable language: an implementation
translates one source into each platform's native shader format, SPIR-V for
Vulkan, MSL for Metal, HLSL for Direct3D. Two independent implementations
run natively: Naga, in the Rust wgpu project, and Tint, in Google's Dawn.

For compute it has what a kernel needs: storage buffers, workgroup shared
memory, atomics and barriers, with f16 and subgroup operations as
extensions. This is a real kernel, the embedding gradient, almost
unabridged: plain buffers, one thread per output element.

Meganeura uses Naga directly. It parses and validates each kernel and emits
SPIR-V for Vulkan or Metal Shading Language for Metal, inside our process.

If asked:
- Status: W3C Candidate Recommendation, from the GPU for the Web Working
  Group. Shipping in Chrome since 2023; Safari and Firefox since 2025.
- No @group or @binding attributes: Blade binds shader data by name.
- Matrix tiles use a Naga extension, wgpu_cooperative_matrix, from our pinned
  fork; it is not standard WGSL yet.
- Kernels are templates specialized for shape, device and precision before
  Naga sees them: backup "How do we write the kernels?".

Sources: src/shaders/scatter_add.wgsl; src/codegen.rs; Cargo.toml (naga).
-->

---

<!-- _class: stack-table -->

# Stack comparison

<table class="stack-table">
<tr><th></th><th>Size</th><th>Code</th><th>Kernels</th><th>Shader compiler</th><th>GPU API</th></tr>
<tr class="mega"><th>Meganeura</th><td><strong>19 MiB</strong><small>one executable</small></td><td>Rust<small>54k lines</small></td><td>WGSL<small>7k lines</small></td><td>Naga<small>WGSL → SPIR-V · MSL</small></td><td>Vulkan · Metal<small>through Blade</small></td></tr>
<tr class="torch"><th>PyTorch</th><td><strong>4.42 GiB</strong><small>29 packages</small></td><td>Python<small>C++ core</small></td><td>Triton<small>plus prebuilt vendor libraries</small></td><td>Triton · LLVM<small>Metal shaders on MPS</small></td><td>Vendor APIs<small>CUDA · ROCm · oneAPI · MPS</small></td></tr>
</table>

**Meganeura shares the application's graphics device and queue.**

<small>Size: Linux executable vs CUDA PyTorch installation, including dependencies. Some matrix features use a Naga fork.</small>

<!--
Time: 1:15.

Here are the two stacks side by side. The benchmark runner built from
Meganeura is one 19 MiB executable. It links only libc, libm and libgcc_s, and
loads the GPU driver at run time. A separate footprint measurement of this PyTorch CUDA build found a
dependency closure of 29 packages and 4.42 GiB.

Meganeura is about 54 thousand lines of Rust, the graph compiler and runtime,
and 7 thousand lines of WGSL kernels. Naga, the shader translator from the
wgpu project, turns WGSL into SPIR-V for Vulkan or Metal Shading Language.
Blade is a thin Rust layer over Vulkan and Metal that I wrote for rendering;
Meganeura uses its compute half. PyTorch's Inductor generates Triton kernels
for CUDA, ROCm and XPU, and Metal shaders for MPS, alongside calls to vendor
libraries. Because Blade also renders, an application can draw and run its
models on one device and one queue; the portability part shows that on a
headset.

If asked:
- Runner: Inferena a0799e42, 20,387,856 bytes, 14.9 MiB stripped, rustc
  1.98.0. Torch and Triton alone are 1.69 GiB.
- Line counts exclude blank lines, comments and test modules: 54,271 Rust and
  7,025 WGSL lines in 113 shader files.
- Blade binds shader data by field name, separates passes with global
  barriers, and leaves checking to the API's validation layers.
- Pins: Blade eae0c6f and a Naga fork, kvark/wgpu a489554, which carries some
  matrix and BF16 features that were not upstream yet.
- Measured: Vulkan on Linux, Windows (the RTX 3050) and Android, Metal on
  macOS. Blade also runs on iOS.
- On Apple GPUs, torch.compile generates Metal shaders instead of Triton.

Sources: ../RESULTS.md, Footprint; ../artifact/footprint.py;
data/hardware.json native_runner.
-->

---

<!-- header: 'Introduction · **Productivity** · Portability · Performance · Conclusion' -->
<!-- _class: client-code -->

# Writing a model

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
<div><b>GGUF</b><span>quantized LLM inference</span></div>
</div></div>

**One graph compiler for native models and imports.**

<!--
Time: 0:50.

This is the model and training loop; initialization and data loading are abbreviated. Plain Rust builders make a small network:
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
All imports feed the graph compiler. Training requires differentiable ops
and trainable parameters; the packed GGUF path shown here is for inference.

If asked:
- Parameters start at zero; init holds the first layer's random initial
  weights, and mnist is an MnistDataset. Both are the application's data.
- The default optimizer is SGD; Optimizer also offers Adam.
- GGUF architectures: the Llama family (SmolLM2, Mistral, TinyLlama), Qwen2
  and Qwen3, Gemma 1 to 4. Weight types: f32, f16, BF16, Q4_0, Q4_1, Q8_0 and
  Q3_K to Q6_K.
- An ONNX or NNEF operator without an equivalent fails at import with its
  name. The importers are fuzzed so malformed files return errors.
- Preparation outside search takes a median 1.7s per workload process: backup "How do we write the kernels?".
- Loading and running an ONNX file: backup "Importing a model".
- From PyTorch: today through torch.onnx.export and load_onnx; a direct
  exporter is planned. Backup "From PyTorch".

Sources: examples/mnist.rs; src/graph.rs, src/nn.rs, src/train.rs;
src/load/onnx.rs, src/load/nnef.rs, src/load/gguf/; src/models/;
tests/importer_fuzz.rs.
-->

---

<!-- _class: testing -->

# Correctness

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

<!-- _class: egraph -->

# Graph alternatives

<div class="egraph-layout"><div>

![An e-graph for Y = XW + b. The output e-class holds Add and FusedMatMulAdd e-nodes. Add points to the intermediate MatMul e-class and the bias e-class; MatMul and FusedMatMulAdd share the X and W input e-classes. The fused node also points to the same bias class.](figures/egraph.svg)

</div><div class="egraph-text">

<p><b>One output, two graph forms.</b> A rewrite adds an e-node to Y's e-class. The original stays.</p>
<p><b>Unfused lowering:</b> MatMul → temporary → Add.</p>
<p><b>Fused lowering:</b> one kernel; no intermediate matrix write or read.</p>
<p><b>Shared operands.</b> Both alternatives refer to the same X, W and b e-classes.</p>

</div></div>

**Y = XW + b.** Both forms remain available for extraction and GPU measurement.

<!--
Time: 0:45.

This is one of our actual rewrites, shown as an e-graph rather than as two
separate expression trees. Dashed boxes are e-classes; the solid boxes inside
are e-nodes. An e-node stores an operator and points to operand e-classes.

The output class Y contains two ways to compute XW plus b. Add reads P and
b, and P contains MatMul. The rewrite adds FusedMatMulAdd to that same output
class. Its edges point directly to the shared X, W and b classes.

Extracting through Add gives two operations and an intermediate matrix.
Extracting through the fused node avoids that intermediate's write and read.
The e-graph retains both; extraction proposes programs, and the next slide
shows how we check and measure them.

If asked:
- Later dispatch fusion can also merge the separate operators; these graph
  forms need not produce different final kernels. Timing happens after lowering.
- This schematic uses the actual fusion equality in src/optimize.rs. Numerical
  checks still gate floating-point implementations; algebraic equivalence
  does not promise bitwise equality.
- The same e-graph also carries matrix schedules, including split-K; the
  "Which form is faster?" backup keeps the packed-projection and split-K examples.
- Small graphs are explored whole; large graphs through a verified repeated
  region. Up to 16 forms are passed to program search.
- The diagram omits shape/type metadata, not operand dependencies.

Sources: src/optimize.rs, Add(MatMul) / FusedMatMulAdd rewrites;
src/optimize/search.rs, keeps_fused_and_unfused_implementations_before_kernel_tuning;
https://github.com/egraphs-good/egg/blob/main/src/tutorials/_01_background.rs
-->

---

<!-- _class: chart search-loop -->

# Program search

![Candidate programs, the e-graph forms combined with layouts, fusion, matrix types and submissions, go through four steps each: build, validate, tune and measure. An invalid candidate is dropped; the winner is the fastest validated program.](figures/search.svg)

**30s per session.** 49% of sessions pick a non-default program.

**No code for a particular GPU or platform:** choices follow reported capabilities and measured time.

<!--
Time: 0:55.

Search adapts the same compiler to each GPU. We still have to write and
improve the kernel families it chooses from. Besides the e-graph's forms, it varies layouts, fusion,
matrix input types and the number of submissions.

For each candidate it builds a private session with initialized weights and
inputs, and validates every output and gradient against the original f32
build. A failure drops the candidate. It then tunes the candidate's kernels
by timing their variants, and measures the whole program against the best so
far, recording and submitting fresh commands each time.
A kernel variant must win by 5 percent, a whole program by 1 percent plus the
measured noise. Kernel decisions are saved and reused; the program race runs
on every build.

In the final runs, 504 sessions measured 22,112 programs. In 248 of them,
49 percent, the winner differs from the ordinary extracted program, so the
alternatives do get used.

If asked:
- Up to 16 graph forms and 64 programs per session. In 396 sessions (79%)
  the deadline arrived before every program was measured.
- Forty-six programs failed the numerical checks and 15 trials hit the 1 GiB
  scratch limit for kernel probes.
- The search costs about a minute per workload: backup "What search costs".
- The final runs have no search-off comparison. The paper's September
  ablation measured 1.18–1.55× from search on two GPUs.
- Where candidates stop: backup "Where paths stop".

Sources: ../main.tex, What it explores; ../tables/search.tex; data/summary.json.
-->

---

<!-- header: 'Introduction · Productivity · **Portability** · Performance · Conclusion' -->
<!-- _class: features -->

# Hardware access

<div class="reach"><div>

<div class="tier used"><h3>Used</h3><div class="chips">
<span>Matrix tiles</span><span>Workgroup matrices</span><span>f16 · BF16</span><span>Subgroup reductions</span><span>8-bit dot products</span><span>GPU timestamps</span>
</div></div>

<div class="tier selected"><h3>Selected in accelerated runs</h3>
<table class="selected-paths">
<tr><th>GPU</th><th>Sessions with matrices</th></tr>
<tr><td>RTX 5070 · H100</td><td>39/42 · 36/42</td></tr>
<tr><td>RX 7900 XT · MI300X</td><td>36/42 · 37/42</td></tr>
<tr><td>Arc B570 · Apple M3</td><td>36/42 · 37/42</td></tr>
</table>
<small>At least one matrix dispatch; counts do not measure utilization.</small>
</div>

</div><div>

<div class="tier"><h3>Exposed by the driver</h3>
<table class="tiles">
<tr><th>GPU</th><th>f16 tile shapes</th><th>f16</th><th>BF16</th><th>f32</th></tr>
<tr><td>RTX 5070 · H100 · RTX 3050</td><td class="tile">16×16×16<br>16×8×16 · 16×8×8</td><td class="dot">●</td><td class="dot">●</td><td class="no">○</td></tr>
<tr><td>RX 7900 XT</td><td class="tile">16×16×16</td><td class="dot">●</td><td class="no">○</td><td class="no">○</td></tr>
<tr><td>MI300X</td><td class="tile">16×16×16</td><td class="dot">●</td><td class="no">○</td><td class="dot f32">●</td></tr>
<tr><td>Arc B570</td><td class="tile">8×16×16</td><td class="dot">●</td><td class="dot">●</td><td class="no">○</td></tr>
<tr><td>Apple M3</td><td class="tile">8×8×8</td><td class="dot">●</td><td class="dot">●</td><td class="dot f32">●</td></tr>
</table>
<small>● exposed · ○ absent<br>f32 tiles also serve the strict contract.</small>
</div>

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

The counts on the left show that matrix instructions are selected on every
timed GPU, including Intel. A count is not a time share or utilization measure.

On the right, the matrix input types each GPU offers, and the f16 tile
shapes. NVIDIA's driver offers several shapes; the others offer one.
Accumulation is f32 everywhere. Only the MI300X, through our driver work, and the M3 take f32
inputs, so those two are the only ones that use matrix units in strict f32.

If asked:
- FP8 matrices, cooperative vectors and tensor addressing are exposed by
  the RTX 5070 driver but unused by this compiler.
- The packed 8-bit dot products serve GGUF-quantized models; the benchmarks
  use f32 weights.
- The B570's 8×16×16 f16 and BF16 tiles are used in 36/42 accelerated
  sessions: backup "Matrix paths selected".
- The RX 7900 XT's RADV build in these runs offers no BF16 matrices.
- Shapes queried on the RTX 5070 and B570 on October 30. NVIDIA: f16 at
  16×16×16, 16×8×16 and 16×8×8; BF16 and FP8 at one shape each; 8-bit
  integers at 16×16×32 and 16×8×32. B570: 8×16×16 for f16 and BF16. The
  selected NVIDIA programs use 16×16×16. H100 and RTX 3050 rows assume the
  same driver's shapes.
- What only the vendor APIs reach is on the next slide.

Sources: data/hardware.json; run records (cooperative_* fields); Blade
eae0c6f blade-graphics/src/lib.rs, Capabilities; Meganeura 567028b8
src/shaders/matmul_gemv.wgsl, src/codegen/workgroup.rs.
https://docs.vulkan.org/features/latest/features/proposals/VK_NV_cooperative_matrix2.html
-->

---

<!-- _class: limits-wide -->

# GPU limits

<table class="limits">
<tr><th>GPU</th><th>Shared memory</th><th>Missing from this graphics path</th></tr>
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

# DinoVision

<div class="dino"><div>

<p class="dino-lead">A headset app: <b>ML and OpenXR rendering share one Vulkan queue</b></p>

<div class="dino-flow">
<div><h3>Train</h3><b>RTX 3050 · Windows</b><span>Vulkan · decoder trained with Adam</span></div>
<div class="dino-arrow">weights<span>→</span></div>
<div><h3>Deploy</h3><b>Quest 3S · Android</b><span>Vulkan · Adreno · both eyes</span></div>
</div>

<p class="dino-note">Same engine on both. No second GPU runtime on the headset.</p>

</div><figure>
<img src="../../figures/dinovision-quest3s.jpg" alt="The headset's reconstruction of its camera frame: a blurry room scene">
<figcaption>The headset's reconstruction of its camera frame</figcaption>
</figure></div>

<!--
Time: 1:00.

At the small end is the device from the opening: one GPU, now with one
stack. DinoVision runs on a Quest 3S, an Android headset with an Adreno GPU.
A DINOv3 encoder feeds a small decoder that reconstructs the camera image; the
picture is its output. The decoder was trained with Adam on an RTX 3050 under
Windows, through Vulkan, then its weights were saved and loaded on the
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

# MI300X driver

<p class="mi300-before"><b>Before our patches, Vulkan cannot use the MI300X at all:</b> no Vulkan driver supports this compute-only accelerator.</p>

<div class="columns mi300-columns"><div>

- RADV: compute + f16 / f32 matrix tiles
- AMDGPU patch targets all eight dies
- Draft: Mesa MR 45239

</div><div>

<div class="big-number"><strong>42 / 42</strong><span>results pass, up to SmolLM2-1.7B</span></div>

</div></div>

<p class="mi300-dies"><b>Multi-die execution:</b> strict f32 training sustains about 38 TFLOP/s, above one die's peak.</p>

<!--
Time: 1:00.

At the other end is the MI300X, a compute-only accelerator. Our stock Vulkan
path could not run it, so the submitted paper listed it as inaccessible. We extended RADV, Mesa's Vulkan
driver, building on Mesa's experimental CDNA work: buffer-based compute and
cooperative matrices, including f32 tiles. A patch to the AMDGPU kernel
driver spreads the work across all eight compute dies. The unchanged engine
and harness then pass all 42 results, from the five core models up to the
1.7-billion-parameter language model. The Mesa changes are posted as a draft
merge request, with the kernel patch attached.

Spreading across the dies matters: strict f32 training of the 1.7-billion
model sustains about 38 TFLOP/s, above the f32 peak of one die. That supports
multi-die execution; timing alone does not establish balanced use of all eight.

If asked:
- 42 = 30 core results (five models, two profiles, three phases) plus 12 for
  SmolLM2-360M and 1.7B.
- Device: AMD Instinct MI300X VF (RADV GFX940), 304 CUs, about 192 GiB.
- Mesa source 8c7f5fdd; the AMDGPU patch targets amdgpu-6.19.14 on Linux
  6.8.0-146 and is enabled with mi300_xcc_gang=1.
- Earlier provenance records the all-XCC module setting and one excluded
  MIOpen convolution solver for PyTorch. The final archives record the Mesa
  build and kernel, without independently recording all module parameters.
- Full-GPU FP32 matrix peak is 163.4 TFLOP/s, about 20.4 per XCC: AMD
  MI300X product specifications, accessed November 1, 2026.

Sources: ../main.tex, Availability;
https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/45239 (draft,
opened October 15).
https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
-->

---

<!-- header: 'Introduction · Productivity · Portability · **Performance** · Conclusion' -->
<!-- _class: measure-new -->

# The comparison

<div class="measure-rows">
<div><b>Models</b><span class="kinds"><i><b>LLM</b>SmolLM2-135M</i><i><b>Robot policy</b>SmolVLA</i><i><b>Image diffusion</b>SD U-Net</i><i><b>Vision</b>ResNet-50</i><i><b>Speech</b>Whisper-tiny</i></span></div>
<div><b>GPUs</b><span class="gpu-groups"><i><b>NVIDIA</b>RTX 5070 · H100 · <u>RTX 3050</u> <span class="os">⊞ Windows</span></i><i><b>AMD</b>RX 7900 XT · MI300X · <u>780M · 9600X</u></i><i><b>Intel</b>Arc B570 · <u>RPL-U</u></i><i><b>Apple</b>M3</i><i><b>Qualcomm</b><u>Quest 3S</u></i><em>Native: 30/30 core cells on each GPU. <u>Underlined</u>: checked against CPU references, no PyTorch GPU timing: Windows (RTX 3050), ROCm failed on the Radeon iGPUs, no XPU device on RPL-U, Android on Quest.</em></span></div>
<div><b>Baseline</b><span>PyTorch 2.13 · <code>torch.compile</code> + graph replay where available</span></div>
<div><b>Each run</b><span class="flow">Check → warm up 10s → time 20 calls → 3 fresh process pairs</span></div>
</div>

<div class="contracts-pair">
<div><h3>Strict f32</h3>f32 everywhere, TF32 off</div>
<div><h3>Reduced precision</h3>PyTorch autocast; Meganeura F16/BF16 search</div>
</div>

<small>Submodels; synthetic weights except SmolLM2. Training timing: forward + loss + backward, without the optimizer.<br>Gates differ: native full-gradient fidelity; PyTorch also checks SGD trajectories.<br>M3 timings vary between processes: median spread 16%.</small>

<!--
Time: 1:15.

Five models, one of each common kind: a small language model, SmolLM2; the
action expert of a robot policy, SmolVLA; a scaled image-diffusion U-Net;
ResNet-50 for image classification; and Whisper's speech encoder. We time
inference at the full shape and a training step: forward, loss and backward,
without the optimizer update.

The same engine passes every check on eleven GPUs from five vendors, from a
headset to a datacenter accelerator. Six are timed against PyTorch 2.13
with torch.compile and, where available, graph replay; the three integrated
GPUs, the RTX 3050 and the Quest are checked for correctness only. Our harness, Inferena, runs
both engines. Each must pass its numerical checks before it is timed. Then it
warms up for ten seconds and times 20 calls, each waiting for the GPU to
finish. That repeats in three fresh pairs of processes, alternating which
engine goes first, and we report the median of the three process medians.

Two contracts. Strict f32 fixes arithmetic width in both engines, not reduction order. With
reduced precision, each engine uses its own recipe: PyTorch BF16 autocast,
FP16 on Apple; Meganeura f16 or BF16 matrix inputs, private storage, and
selected backward products.

If asked:
- The engines pass different checks. Meganeura compares full tensors with its
  f32 build. PyTorch must repeat within fixed bounds and, with reduced precision, track
  f32 over matched SGD trajectories. Across engines, sampled outputs, loss and
  gradient norms must agree. Backup "What the checks admit".
- MPS compiles each phase but has no whole-phase graph replay.
- max-autotune is off, and PyTorch compilation has a 120s limit.
- Only SmolLM2 uses trained weights; the others use deterministic uniform
  parameters.
- The runs also time a minimal shape (one token, action or image): backup
  "Minimal shape".
- Correctness-only references: CPU PyTorch f32 for the RTX 3050 and integrated GPUs, CPU
  f64 full tensors for the Quest; 330 of 330 core results pass. Backup
  "Coverage".

Sources: ../main.tex, Methodology; ../tables/devices.tex.
-->

---

<!-- _class: chart results -->

# Strict f32

<div class="ratio-legend">
<div class="ratio-metric">Meganeura time ÷ PyTorch time</div>
<div class="ratio-keys"><span><i class="ours"></i>Meganeura faster</span><span><i class="theirs"></i>PyTorch faster</span><span><i class="missing">—</i>No ratio</span></div>
</div>

![Strict f32 time ratios for six GPUs and five workloads, inference and training](figures/strict-ratios.svg)

Medians: <span class="slower">1.46×</span> inference · <span class="slower">1.59×</span> training. Faster in **13 of 58** comparisons.

<!--
Time: 1:20.

Strict f32 fixes arithmetic width, with TF32 and reduced matrix inputs off.
Each cell is Meganeura time divided by PyTorch time; below one is faster.
The RX 7900 XT wins most comparisons. Intel and the server GPUs retain
substantial gaps. The Arc U-Net is now 2.95 times PyTorch's time.

If asked:
- Inference medians: RTX 5070 1.05, H100 1.11, RX 7900 XT 0.63,
  MI300X 2.19, B570 2.18, M3 1.34. Training: 1.47, 1.86, 0.61,
  2.19, 3.00, 1.24.
- Missing reference cells: RX 7900 XT Whisper and MI300X ResNet training
  fail repeatability; their later minimal-shape phases are not all attempted.
- Arithmetic width is fixed, reduction order is not.
- M3 process spread is a median 16%; near-parity results are uncertain.
- 30 inference comparisons, 28 training; 9 and 4 nominal wins respectively.
- The server scaling comparison follows the kernel study.

Sources: data/results.csv; data/summary.json.
-->

---

<!-- _class: chart results -->

# Reduced precision

<div class="ratio-legend">
<div class="ratio-metric">Meganeura time ÷ PyTorch time</div>
<div class="ratio-keys"><span><i class="ours"></i>Meganeura faster</span><span><i class="theirs"></i>PyTorch faster</span><span><i class="missing">—</i>No ratio</span></div>
</div>

![Reduced-precision time ratios for six GPUs and five workloads, inference and training](figures/accelerated-ratios.svg)

Median ratio: <span class="slower">1.67× inference · 1.92× training</span>. Native gains over f32: **1.19× · 1.05×**.

<!--
Time: 1:10.

Reduced precision now speeds up native inference substantially. The small
language model is a good example: on the 5070, inference falls from 6.37 to
3.61 milliseconds and training from 23.08 to 15.64. The gain is real, but
PyTorch accelerates too, so the ratios are still 1.23 and 1.76.

Across the models, native training gains much less than inference. We select
matrix paths on every timed GPU, including the Arc. The remaining issues are
how efficiently those kernels run, how much backward work they cover, and
the data movement around them. Whisper remains especially costly: its 5070
training step is 20.52 milliseconds against PyTorch's 3.33.

If asked:
- Gains use each engine's own qualified strict/accelerated pairs: native
  1.19 and 1.05; PyTorch 1.46 and 1.64. Matched-population geometric means
  are in “Two gaps”; do not divide these medians to obtain that accounting.
- Meganeura is faster in 10/54 inference/training comparisons, 5/27 each.
- Public tensors, masters and accumulation stay f32. Private reduced storage
  and selected transposed backward products are searched; NN derivatives
  remain protected. This is partial graph coverage, not all-f16 training.
- H100 U-Net fails PyTorch forward fidelity; MI300X U-Net and Whisper do not
  retain three passing reference processes. No older run fills a missing cell.
- SmolLM2 median accelerated ratios across the six GPUs: 1.28 inference,
  1.82 training. BF16 autocast on CUDA/HIP/XPU, FP16 on Apple; TF32 off.

Sources: data/results.csv; data/summary.json; ../NUMERICAL-POLICY.txt;
../main.tex, Results.
-->

---

<!-- _class: ops -->

# Kernel gaps

<p class="case-context">Strict f32, one GEMM or convolution, our GPU time ÷ PyTorch's. The vendor libraries run on the same f32 units we use.</p>

<table class="ops">
<tr><th>Operation</th><th>Shapes</th><th>RX 7900 XT</th><th>RTX 5070</th><th>Arc B570</th></tr>
<tr><td>Large GEMM</td><td>4096³</td><td>1.6×</td><td>2.1×</td><td>2.2×</td></tr>
<tr><td>Transformer GEMMs</td><td>Speech, LLM</td><td>1.2–1.3×</td><td>1.5–1.7×</td><td>1.2–1.5×</td></tr>
<tr><td>Small GEMMs</td><td>Diffusion, M = 64–256</td><td class="near">0.66–1.3×</td><td class="far">1.6–4.9×</td><td class="far">2.0–4.6×</td></tr>
<tr><td>3×3 convolutions</td><td>Diffusion, Vision</td><td>0.76–2.2×</td><td class="near">0.45–1.0×</td><td class="far">1.45–3.2×</td></tr>
<tr><td>Other convolutions</td><td>1×1, stems</td><td class="near">0.55–1.1×</td><td>1.4–2.0×</td><td>1.4–1.75×</td></tr>
</table>

**Small GEMMs and Intel convolutions are concrete targets.** Test split-K and smaller tiles.

<small>Separate GPU timing probe: native sums pass times; PyTorch's interval includes dispatch gaps. These ratios do not partition model time.</small>

<!--
Time: 1:30.

To find where the gap is, we timed single operations against PyTorch: one
GEMM or one convolution, strict f32, GPU time only, on three GPUs. In this
mode the vendor libraries use ordinary f32 kernels, CUTLASS on NVIDIA, oneDNN
on Intel and the ROCm libraries on AMD, so we compare kernels on the same
execution units.

Large GEMMs: the vendor libraries reach 1.6 to 2.2 times our throughput.
Transformer-sized GEMMs from Whisper and the LLM are closer. The small GEMMs
in the diffusion U-Net are where we lose most on NVIDIA and Intel: a 64-row
matrix product takes 28 microseconds for us and 6 for cuBLAS, and 55 against
12 on the Arc. There is too little work per dispatch for our tiles, and
cuBLAS splits the K dimension for these shapes.

Convolutions split by vendor. On the RTX 5070 our Winograd 3×3 convolutions
match or beat cuDNN; on the Arc, oneDNN's are 1.45 to 3.2 times faster. The
RX 7900 XT is our closest GPU, as in the model results: we win some small
GEMMs and convolutions there and lose others by up to 2.2×. Its geometric
means are 1.11× for GEMMs and 0.98× for convolutions.

These controls identify small-matrix and small-convolution kernels as
contributors to the U-Net gap. Split-K and smaller tiles are candidates to
test; this study does not measure how much model time they would recover.

If asked:
- Large GEMM: 13.5 against 22.3 TFLOP/s on the RX 7900 XT, 10.2 against
  21.6 on the RTX 5070, 3.2 against 6.9 on the B570. This shows our kernel
  does not yet reach the vendor libraries' f32 throughput; it does not show
  that Vulkan can.
- 4096³ needs more than the tuner's 64 MiB of scratch, so all three GPUs run
  our default tile. 2048³, which the tuner covers, is 1.2× on the RX 7900 XT,
  2.0× on the RTX 5070 and 1.9× on the Arc.
- Native uses three processes per shape, with 2s tuning in each. PyTorch
  uses one process per shape, thirty graph-event samples. Small Arc native
  ops vary by up to 45%. Operands differ; this is a throughput probe, not
  a cross-engine numerical comparison.
- Timing: our GPU time sums Vulkan pass timestamps; PyTorch's is a device
  interval around a graph of 30 serial ops, divided by 30, with the profiler
  off (profiling slows ROCm down). The PyTorch interval includes dispatch gaps
  between kernels, which matters most for the smallest ops, especially on the
  Arc.
- Shared memory is not the reason: the U-Net's largest generated kernel uses
  16 KiB of Vulkan's 48.
- Convolution times include our per-call Winograd weight transform.
- Current code (October 31, Meganeura 5b232bd plus the tuner scratch fix,
  567028b on main), with the tuner fix also present in the cohort. This
  remains a separate single-op study, not a decomposition of model time. PyTorch 2.13.0 (CUDA 13.0, XPU, ROCm 7.2), TF32 off,
  cudnn.benchmark on.

Sources: data/op-bench.tsv; data/op-bench/; examples/op_bench.rs;
data/b570-unet-shared-memory.txt; data/results.csv.
-->

---


<!-- _class: chart -->

# Larger models

![Strict SmolLM2 prefill and training ratios at 135M, 360M and 1.7B, H100 and MI300X](figures/scaling.svg)

Strict training, 135M → 1.7B: H100 <span class="slower">1.86× → 3.77×</span>, MI300X <span class="slower">2.18× → 1.64×</span>.

<!--
Time: 0:50.

The same language model at three sizes, batch one and 128-token prefill.
The server trends diverge: H100's strict training gap grows while MI300X's
shrinks. That rules out a single fixed per-model overhead as the whole story,
although per-dispatch cost may grow with the graph.

If asked:
- Native 1.7B training: 135.04 ms H100, 34.67 ms MI300X; PyTorch 35.85
  and 21.18 ms. MI300X reaches f32 matrix tiles under strict.
- Accelerated 1.7B PyTorch fails worst-token KL on both; no ratio is shown.
- B570 accelerated 360M is fully paired: prefill 14.63 vs 9.29 ms (1.57×),
  training 147.14 vs 27.47 ms (5.36×). This is an important remaining loss.
- RTX 5070 360M accelerated training is also fully paired, ratio 2.76×.
- Larger-model evidence: all 42 native cells pass; 34 have ratios.

Sources: ../tables/scaling.tex; ../RESULTS.md.
-->

---

<!-- _class: since-paper -->

# Since the paper

| | Paper · Sept 25 | Talk · Oct 31–Nov 1 |
|---|---:|---:|
| Strict inference, five shared GPUs | 1.33× | **1.12×** |
| Strict training, same qualified conditions | 1.94× | **1.47×** |
| Code: Rust · WGSL lines | 42k · 4.7k | 54k · 7k |
| Search budget per session | 60s | 30s |
| Preparation per workload | about 2 min | 47–67s |

**Also changed:** MI300X added · BF16 autocast, not TF32 · 10s warmup

<!--
Time: 0:50.

These are the submitted paper's September measurements against the final
collection. Hold the five shared timed GPUs and qualified conditions fixed;
the RTX 3050 is correctness-only now and MI300X is new. Strict inference
and training ratios improve to 1.12 and 1.47.

Across the four shared GPUs other than the noisy M3, our strict times fall
by a median 7.7 percent for inference and 16.4 percent for training.
PyTorch strict times move by less than one percent at the median. The
protocol also changed, so this is not a controlled compiler ablation.

If asked:
- Without M3, matched strict ratios: inference 1.27 → 1.12; training
  2.05 → 1.62. The five-GPU aggregate has different platform coverage from
  the six-GPU main table.
- Native runner: 13.5 MiB in the paper, 19.4 MiB now (14.9 stripped).
- B570 U-Net strict inference: 2.97 ms in September, 3.21 now. The old
  October slide's 5.89 ms is not the final result.
- Source counts exclude comments, blank lines and cfg(test) modules.

Sources: since_paper.py and the tagged paper-p3hpc-work analyzer;
data/results.csv; data/preparation.csv; ../artifact/footprint.py.
-->

---

<!-- header: 'Introduction · Productivity · Portability · Performance · **Conclusion**' -->
<!-- _class: verdict -->

# What remains?

**Broad hardware coverage. Performance depends on the paths we implement.**

<div class="verdicts">
<div><h3>Reached</h3><div class="answer">Competitive</div><p>RX 7900 XT, M3: 20/39 nominal wins</p><p>M3 timings vary between processes</p></div>
<div class="open"><h3>Kernels</h3><div class="answer">Main targets</div><p>Small GEMMs and Intel convolutions</p><p>Long attention and backward coverage</p></div>
<div class="open"><h3>Driver access</h3><div class="answer">Constraints</div><p>Shared-memory limits · Hopper instructions</p><p>MI300X requires experimental patches</p></div>
</div>

<!--
Time: 0:40.

On the RX 7900 XT and M3 we are faster in 20 of 39 inference/training
comparisons, though M3 timings are noisy. Elsewhere we have concrete kernel
targets: small products and Intel convolutions, long-sequence attention and
incomplete backward acceleration.

API limits and driver quality can matter too. We cannot explicitly request
Hopper's warpgroup instructions or distributed shared memory through this
stack, and MI300X needs experimental patches. But those facts do not assign
an exact fraction of the measured gap to the API. More kernel engineering
has gone into consumer AMD and NVIDIA than Intel or server hardware.

If asked:
- RX 7900 XT: 7/9 strict and 5/10 accelerated wins. M3: 3/10 and 5/10.
- Matrix paths are selected on all six timed GPUs; using them is not enough.
- The single-op study fixes f32 arithmetic and identifies implementation
  losses. Driver attribution still needs matched kernels and timelines.

Sources: data/results.csv; data/op-bench.tsv; data/limits.json.
-->

---

<!-- _class: verdict -->

# What works?

**One compact binary trains and runs models on every GPU we tried.**

<div class="verdicts">
<div><h3>Productivity</h3><div class="answer">19 MiB</div><p>One executable</p><p>C runtime + system GPU driver</p></div>
<div><h3>Portability</h3><div class="answer">Mobile to server</div><p>11 GPUs, 5 vendors</p><p>Training checks pass on all</p></div>
<div><h3>Performance</h3><div class="answer">1.60× median</div><p>112 inference/training comparisons</p><p>Both profiles, 6 timed GPUs</p></div>
</div>

[github.com/kvark/meganeura](https://github.com/kvark/meganeura)

<!--
Time: 0:40.

One executable compiles, trains and runs these models from a Quest headset
to an MI300X. It needs the C runtime and the system GPU driver, without
Python or a vendor SDK. All 330 core checks pass, including backward.
DinoVision demonstrates the full train-on-desktop, deploy-in-headset cycle.

Across 112 qualified core inference and training comparisons, both profiles,
our median time is 1.60 times PyTorch's. That is an aggregate, not an upper
bound: the table includes much larger losses. Reduced precision helps
inference more than training, which is where the next work belongs.

If asked:
- Runner: Inferena a0799e42, 20,387,856 bytes, SHA-256 e88a082f…;
  19.4 MiB as built, 14.9 MiB stripped. libc, libm and libgcc_s only.
- 58 strict comparisons, 54 accelerated; geometric mean 1.66×. 71/112
  are below 2×. No minimal-shape duplicates in this aggregate.
- Correctness-only: RTX 3050, three integrated GPUs and Quest. Full Adam
  cycle evidence is the separately pinned desktop DinoVision study.
- MI300X uses experimental drivers; this is not a stock-support claim.

Sources: data/results.csv; data/summary.json; data/hardware.json;
../QUALIFICATION.md.
-->

---

<!-- header: 'Backup' -->
<!-- _class: backup backup-index -->

# Backup

<div class="columns backup-list"><div>

**Productivity**
[Importing a model](#23) · [From PyTorch](#24) · [Inside Meganeura](#25) · [How do we describe a model?](#26) · [Which form is faster?](#27) · [How do we write the kernels?](#28) · [Gradients](#29) · [Compile stages](#30)

**Portability**
[Matrix paths selected](#31) · [Where paths stop](#32) · [What search costs](#33) · [Reusing kernel decisions](#34) · [Coverage](#35)

</div><div>

**Performance**
[Minimal shape](#36) · [Precision gains](#37) · [Two gaps](#38) · [Precision rules](#39) · [What the checks admit](#40) · [Where TF32 fits](#41) · [SPIR-V precision](#42) · [Timing variability](#43) · [Warmup](#44) · [Portability scores](#45)

**Deployment**
[Queue measurements](#46) · [Learning on the device](#47)

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

<small>NNEF loads the same way with load_nnef. GGUF builds a text generator: load_gguf(path)?.generator(2048)?. From PyTorch today: torch.onnx.export, then load_onnx (scripts/export_onnx.py).</small>

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

<!-- _header: 'Backup · Productivity · planned' -->
<!-- _class: from-torch -->

# From PyTorch

<p class="lead">Today: <code>torch.onnx.export</code>, then <code>load_onnx</code>. Planned: one call, no ONNX in between.</p>

<div class="torch-flow">
<div><h3>1 Capture</h3><b>torch.export</b><span>whole graph as ATen ops, no Python left</span></div>
<div><h3>2 Lower</h3><b>Core ATen</b><span>keep attention, norms and convolutions; decompose the rest</span></div>
<div><h3>3 Emit</h3><b>graph + safetensors</b><span>state_dict names; trainable from requires_grad</span></div>
<div><h3>4 Check</h3><b>f64 interpreter</b><span>against PyTorch on the CPU, before the file ships</span></div>
</div>

```python
meganeura.export(model, (example,), "model.mgn")  # pass a loss module to train
```

<p class="torch-note">Rust loads the file into the same Graph; training uses our autodiff and optimizers. Unsupported ops are listed together, with their source lines.</p>

<!--
Backup, for "can I bring a PyTorch model?"

Today the route is ONNX: torch.onnx.export writes the graph, load_onnx reads
it, and rewrite rules re-fuse the patterns PyTorch's exporter decomposes. It
works for standard models, but ONNX is an inference format and an extra
dependency, and its decompositions hide intent.

The planned exporter is a small, pure-Python package. torch.export captures
the whole model as an ATen graph with no Python left. We lower to the core
ATen operator set, but keep the operators we have fused kernels for, such as
scaled-dot-product attention, layer and group norm and convolution, so the
compiler sees intent instead of fragments. Each node maps to a Graph op; the
file holds the graph and safetensors weights under their state_dict names,
with requires_grad deciding parameter versus constant. Before writing, the
exporter runs our CPU f64 interpreter on the example input and compares it
with PyTorch. For training, the user exports the model with its loss; our
autodiff builds the backward pass and Meganeura's optimizers update the same
named weights, so checkpoints map back to the state_dict.

If asked:
- Static shapes: each exported shape is its own graph, as in Meganeura today.
- A torch.compile backend could reuse the same op table for trying it from
  Python; deployment still uses the file.
- Effort: the op table (a few hundred core ATen ops, most mapping to existing
  primitives), the Rust loader and per-op round-trip tests.
- Not implemented yet; nothing in the results uses it.

Sources: scripts/export_onnx.py; src/load/onnx.rs; src/optimize.rs, ONNX
decomposed op recognition; src/reference/.
-->

---

<!-- _header: 'Backup · Productivity' -->
<!-- _class: chart -->

# Inside Meganeura

![Eight steps from model description to execution: describe, rewrite, differentiate, plan, specialize, compile, check and time, execute](figures/compiler-map.svg)

One compiler for inference and training. **Search revisits these choices by timing them on the target GPU.**

<!--
Backup, for “what does the compiler do?”

Read the top row left to right, then the bottom row right to left.

A model arrives from Rust builders, ONNX, NNEF or GGUF as a typed static
graph. An egglog e-graph holds equivalent forward forms and matrix schedules.
Each forward form gets its own backward pass; inference skips that step. The
planner lowers and fuses operations, orders the dispatches and reuses buffers
by lifetime. Kernels are specialized for shape, device and precision, written
out as WGSL and compiled through Naga and Blade into Vulkan or Metal pipelines.

The search builds candidate programs, checks them, tunes their kernels and
races whole programs on the device. The winner runs as a fixed plan: every
step records, submits and waits, without graph rewriting or tensor allocation.

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

**Outside search:** median **1.7s** per process; up to **70.9s** in the full collection.

<!--
Backup, for "how are kernels written?"

Every kernel is WGSL. Templates have sections and substitution points: this
one is the matrix-vector product used for decoding, and the generator fills in
the weight storage type, the vector width, how weights are loaded and how a
row is reduced, from the shapes, the device and the precision policy.
Pointwise operations need no template: a chain becomes one kernel, or folds
into a matrix product's epilogue. Matrix and attention kernels have
generators for each device's tile shapes.

Native preparation outside the recorded search, including the f32 reference,
takes a median 1.7 seconds per process across the collection. The long tail
is 70.9 seconds, so this is not a bound on build time.

If asked:
- 1.7s is the median across all 272 native processes, including large
  models and correctness-only runs; the range is 0.18–70.87s.
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
- Selected backward products may use F16/BF16; accumulation stays f32.
- Packed parameter and gradient arenas let one optimizer dispatch update many tensors.

<small>The benchmark stops before the optimizer. DinoVision runs the full Adam training cycle.</small>

<!--
Backup, for "how does training work?"

Training uses the same compiler as inference, with a larger graph. Each
forward alternative is differentiated, and backward layouts are explored too.
Accelerated search can use F16/BF16 for selected transposed backward matrix
products. Other derivatives remain protected, and accumulation and public
gradients stay f32. The checks compare full gradients, which catch underflow
that plausible forward outputs can hide.

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

Whole preparation with fresh caches: Meganeura **1.2–1.4s**, PyTorch **11–24s**.

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

# Matrix paths selected

| GPU | Sessions with matrices, of 42 | Most matrix dispatches | Reused workgroup choices |
|---|---:|---:|---:|
| RTX 5070 | 39 | 482 | 20 |
| H100 | 36 | 482 | 14 |
| RX 7900 XT | 36 | 482 | 0 |
| MI300X | 37 | 543 | 0 |
| Arc B570 | 36 | 482 | 0 |
| Apple M3 | 37 | 513 | 0 |

<small>Accelerated core sessions. BF16 selected in 3 RTX 5070 sessions and 5 B570 sessions.</small>

<!--
Backup, for “do you actually use the matrix units?”

Each GPU has 42 accelerated sessions across three processes: three sessions
for four models, two for Whisper. A selected session counts if its program
contains at least one matrix dispatch. Counts do not measure time share,
throughput or the fraction of FLOPs on matrix units.

The workgroup column counts selected programs with reused workgroup tuning
choices, not every possible use of workgroup kernels. RTX 3050 is outside
this timing cohort. Under strict, only MI300X and M3 use f32 matrix tiles.

Selected descriptions also request reduced transposed backward products in
9 RTX 5070, 9 H100, 8 RX 7900 XT, 3 MI300X and 7 B570 sessions. Private
matrix-storage choices occur in 3 RX 7900 XT and 4 B570 sessions.

Sources: data/summary.json, accelerated_selection; pinned compiler search;
selected-program descriptions in the original receipts.
-->

---

<!-- _header: 'Backup · Portability' -->
<!-- _class: chart gap-paths -->

# Where paths stop

![Four checkpoints connected left to right: expose, lower, qualify and measure. Each has two rejection branches; surviving candidates reach the selected program.](figures/path-admission.svg)

**An exposed matrix instruction is only the first checkpoint.**

<!--
Backup, for "why can a matrix path be rejected?"

This is the selection process, not a breakdown of the performance gap. Follow
the teal arrows to keep a candidate; each brown branch stops that candidate.
An instruction can be missing, a tile can exceed shared memory, the result can
fail qualification, or the complete program can lose to the incumbent. The
search deadline can also leave a candidate unmeasured.

The B570 selects matrix work in 36/42 accelerated sessions. A different
candidate may still stop at any checkpoint. Neither successful selection nor
rejection measures how efficiently the surviving kernel uses the hardware.

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

![Preparation by workload: native preparation against PyTorch compilation and first calls](figures/preparation-models.svg)

Meganeura: <span class="mega">47–67s</span> per workload. PyTorch compiles in <span class="torch">8–50s</span>.

<!--
Backup, for “how long does the search take?”

Strict-f32 setup per workload, median across GPUs of complete three-process
medians. Native preparation includes graph construction, full-tensor checks,
reference construction and search. PyTorch compilation includes first calls;
its separate capture and replay checks are outside the orange bars.

Native work outside the recorded search is typically small, but not always:
a median 1.7 seconds per process across all campaigns, range 0.18–70.87.
The search's 30-second allowance is per session, not per model or process.
It is a soft deadline; reference preparation is outside it.

If asked:
- Six complete GPUs for the first three workloads, five for ResNet and
  Whisper. A partially successful reference process is not a complete group.
- Longest reference compile: 100 seconds under the 120-second allowance.
- Core search receipts total 186.6 minutes; no 15-second or search-off arm
  exists in this cohort, so it does not establish an optimal budget.

Sources: data/preparation.csv; ../tables/search.tex.
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
already reuse decisions. Across all 272 native processes there are 124
first-session and 148 reused-decision states. There was no randomized cache
experiment.

Source: data/preparation-cache.csv.
-->

---

<!-- _header: 'Backup · Portability' -->

# Coverage

| GPU | Result | Comparison |
|---|---|---|
| MI300X | Core models, 360M and 1.7B pass | Experimental Mesa and AMDGPU |
| RTX 3050 · 780M · 9600X iGPU · RPL-U | **120 / 120** cells pass | CPU PyTorch f32; no performance collection |
| Quest 3S / Adreno 740 | **30 / 30** cells pass | CPU f64 full tensors; no performance collection |
| B570 · SmolLM2-360M | **6 / 6** native cells pass | 4 qualified ratios; all accelerated phases paired |

**All 330 core combinations of GPU, model, profile and phase pass.**

<!--
Backup, for “where else does it run?”

Six timed GPUs plus five correctness-only devices. Core and larger models
supply 222 native timing cells and 201 qualified ratios; 21 lack a complete
passing reference. No historical result replaces a missing final-cohort cell.

B570 360M now completes. Captured accelerated inference, latency and training
pass all three processes. Strict training fails a gradient-repeatability
bound in process 2; that also leaves too few subsequent latency processes.
This is a numerical gate failure, not the earlier command-graph exception.
The XPU path uses a capture-compatible index_add embedding backward.

Android archives retain hashed manifests and complete comparison receipts,
without raw tensors; offline replay audits the receipts.

Sources: ../QUALIFICATION.md; ../RESULTS.md.
-->

---

<!-- _header: 'Backup · Performance' -->
<!-- _class: chart results -->

# Minimal shape

<div class="ratio-legend">
<div class="ratio-metric">Meganeura time ÷ PyTorch time</div>
<div class="ratio-keys"><span><i class="ours"></i>Meganeura faster</span><span><i class="theirs"></i>PyTorch faster</span><span><i class="missing">—</i>No ratio</span></div>
</div>

![Minimal-shape ratios on six GPUs and five workloads, strict and reduced precision](figures/minimal-shape-ratios.svg)

Medians: <span class="mega">0.97×</span> strict · <span class="slower">1.01×</span> accelerated. Faster in **15 of 28** and **13 of 27**.

<!--
Backup, for “what about latency?”

One token, action or image, run after training. SmolLM2 has no KV cache, so
this is not decode throughput. U-Net and Whisper use the same shape as their
full inference. A reference training failure may leave this phase unattempted.
Near parity does not bound CPU or submission overhead: the workload and the
reference denominator differ from full-shape inference.

Sources: data/results.csv; data/summary.json.
-->

---

<!-- _header: 'Backup · Performance' -->
<!-- _class: chart -->

# Precision gains

![Per-GPU median strict-over-accelerated speedup for both engines, inference and training](figures/precision-gain.svg)

Native gains **1.19× inference, 1.05× training**; PyTorch **1.46× and 1.64×**.

<!--
Backup, for “does reduced precision help?”

Yes, particularly inference. Bars use identical qualified workloads for both
engines and profiles on each GPU. The headline medians use each engine's own
qualified pairs; native matched medians are 1.19 and 1.04, on 27 inference
and 25 training conditions.

Matrix hardware is used, including on Intel, but backward coverage is partial.
Conversions and staging can offset arithmetic gains. M3 already has f32
matrix tiles, and its reference FP16 training often slows down. The chart
records selected implementations; it is not a feature-off ablation.

Sources: data/results.csv; ../main.tex, Results.
-->

---

<!-- _header: 'Backup · Performance' -->
<!-- _class: chart -->

# Two gaps

![Strict and accelerated ratios on identical qualified model sets, by GPU and phase](figures/gap-accounting.svg)

**Existing f32 gap × unequal acceleration = accelerated gap.** The multiplier is **1.34× for inference, 1.44× for training**.

<!--
Backup, for “how much of the gap is precision?”

For each condition, the accelerated ratio equals the strict ratio times
PyTorch's speedup divided by native speedup. Hold both engines and both
profiles to the same qualified model set; geometric means preserve this
identity and medians generally do not.

On 27 inference conditions the geometric-mean ratio is 1.24 → 1.66, with
engine gains 1.62 PyTorch and 1.21 native. On 25 training conditions it is
1.54 → 2.21, with gains 1.53 and 1.07. The 2.21 geometric mean on matched
conditions is distinct from the main table's 1.92 median over 27 accelerated
training comparisons.

This is accounting, not a causal allocation to CPU, GPU, API or driver.
M3 variability and small per-device denominators remain limitations.

Sources: ../artifact/gap_analysis.py; data/results.csv; ../tables/gap.json.
-->

---

<!-- _header: 'Backup · Performance' -->
<!-- _class: contracts -->

# Precision rules

| | PyTorch | Meganeura |
|---|---|---|
| Arithmetic | BF16 autocast on CUDA / HIP / XPU; FP16 on MPS | F16/BF16 matrix inputs; private reduced storage; f32 interfaces and accumulation |
| Training | Autocast also covers backward | Selected transposed backward products; other derivatives remain protected |
| Checks | f32 forward fidelity, repeatability, matched SGD trajectories | Full outputs and single-step gradients against its f32 build |

**TF32 is off in both profiles.**

<!--
Backup, for “what does the accelerated profile allow?”

Native parameters, public inputs/outputs, loss, matrix accumulators and
split partials stay f32. Search may select reduced storage for compatible
private readers, F16/BF16 products and Winograd transforms. It may select
reduced AT/BT backward products; NN derivatives remain protected. Required
conversions execute within each timed call.

Generic precision metadata in the archive still contains older blanket
backward-f32 wording. Selected descriptions and the pinned compiler expose
the actual derivative and private-storage choices; do not infer them from
that older summary alone.

PyTorch checks 256 real-text SGD updates for SmolLM2, 32 repeated-input
updates for the other models, loss within 2% of eager f32. Native checks
complete single-step gradients within 1% of its f32 build. Neither proves
that future training trajectories have the same accuracy or range.

Sources: ../NUMERICAL-POLICY.txt; Meganeura 567028b8 src/train/search.rs;
selected-program receipts.
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
https://github.com/kvark/inferena/blob/a0799e4210eafa25161ec50d475b186c4b19d3da/frameworks/pytorch/bench.py#L1339
https://github.com/kvark/inferena/blob/a0799e4210eafa25161ec50d475b186c4b19d3da/frameworks/pytorch/amp.py#L117
-->

---

<!-- _header: 'Backup · Performance' -->
<!-- _class: backup precision -->

# SPIR-V precision

| Mechanism | Measured Naga / Blade path |
|---|---|
| RelaxedPrecision / NoContraction | **No** |
| FPFastMathMode / FPFastMathDefault | **No** |
| F16 denormal preservation | **Yes**, when the device supports independent preservation |
| Other rounding / denormal / special-value modes | **No explicit request** |
| f32, f16, BF16 types and conversions | **Yes**, according to the selected kernel |

**“Strict” means f32 arithmetic plus numerical checks, under Vulkan's usual f32 rules.**

<!--
Backup, for “is strict really IEEE?”

Audited at measured Naga a489554 and Blade eae0c6f. Blade queries
shaderDenormPreserveFloat16 and denormBehaviorIndependence == ALL. When both
allow it, it sets Naga's PRESERVE_F16_DENORMS. For a module using Float16,
Naga then emits the DenormPreserve capability and DenormPreserve 16 execution
mode, adding SPV_KHR_float_controls below SPIR-V 1.4. This is automatic; it
is not an application-visible Blade capability or a guarantee for all devices.

There are no explicit RelaxedPrecision, NoContraction or FPFastMath
requests. Ordinary f32 operations follow Vulkan defaults; contraction and
other allowed transformations remain possible. Explicit WGSL fma emits an
FMA. Full numerical checks and CPU-f64 compiler tests remain necessary.

Sources:
https://github.com/kvark/blade/blob/eae0c6f7aa3167c79d1135050cf8e17c17fabd0a/blade-graphics/src/vulkan/init.rs
https://github.com/kvark/wgpu/blob/a489554c96f87cbc5252ee00350730f6e93869f9/naga/src/back/spv/writer.rs
https://github.com/kvark/wgpu/blob/a489554c96f87cbc5252ee00350730f6e93869f9/naga/src/back/spv/mod.rs
-->

---

<!-- _header: 'Backup · Performance' -->
<!-- _class: chart -->

# Timing variability

![Between-process spread of native and reference times on six GPUs](figures/timing-spread.svg)

Apple M3: **43 of 60** groups exceed 10% spread; median **16%**, despite sleep prevention. Every repeat stays.

<!--
Backup, for “how noisy is this?”

Each point is (max minus min) divided by the median of three process medians.
Both engines move on M3; no clock/power trace identifies the cause, so do
not claim small M3 wins. Other GPUs' median spreads are at most 2.4%.
Four RX 7900 XT groups exceed 10%, one B570 and one MI300X group; none on
RTX 5070 or H100. No outlier is removed.

Sources: data/results.csv; ../RESULTS.md.
-->

---

<!-- _header: 'Backup · Performance' -->
<!-- _class: chart backup -->

# Warmup

![SmolLM2-135M reduced-precision timings relative to each process's 60-second checkpoint, on RTX 5070 and B570](figures/warmup.svg)

Two processes per GPU. The B570 stays within **0.91%**; RTX 5070 prefill is **0.67% / 3.04%** slower at 60s than at 5s.

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

| Strict f32 · mean over workloads · six GPUs | Meganeura | PyTorch |
|---|---:|---:|
| Inference | 0.64 | 0.85 |
| Minimal shape | 0.76 | 0.37 |
| Training | 0.58 | 0.53 |

Missing Whisper and ResNet references contribute zero. The overall training lead depends on that coverage advantage.

<!--
Backup, for "what about the P3HPC portability metric?"

Efficiency is the fastest passing time divided by the engine's time. Each
workload and phase gets a harmonic mean across the six timed GPUs, with a
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

<div class="dino queue"><div>

**DinoVision · Quest 3S · Android / Adreno 740**

Two eye workers + renderer, **one context and queue**.

| Measurement | 1 chunk | 2 chunks |
|---|---:|---:|
| ML alone · ms/pass | 96.78 | 100.79 |
| Live · render submissions/s | 10.83 | 12.87 |
| Live · worker latency, ms | 149.51 | 119.25 |

**Two submissions improve the live schedule.** The shared queue can accept rendering between them.

<small>Live test: 90 seconds per setting. Render submissions per second is the app's submission rate; display FPS was not measured.</small>

</div><figure>
<img src="../../figures/dinovision-quest3s.jpg" alt="The headset's reconstruction of its camera frame: a blurry room scene">
<figcaption>The headset's reconstruction of its camera frame</figcaption>
</figure></div>

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
