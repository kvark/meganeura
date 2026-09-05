# Alternatives: compare the right layer

Primary project documentation checked September 5, 2026. These are
architectural comparisons, not Meganeura performance measurements. Only
PyTorch has a matched frozen comparison in this repository. Backend support
is version- and operator-dependent; a backend appearing in a project table
does not establish equal training coverage or performance.

## The map

| Layer | Representative alternatives | The useful comparison |
|---|---|---|
| End-to-end tensor/training framework | PyTorch, JAX, Burn, tinygrad, MLX, Candle | Graph/autodiff model, supported workloads, optimization, debugging, deployment. |
| ML graph compiler/runtime | TVM, IREE, Luminal | IR, search space, scheduling, compilation boundary and execution closure. |
| Kernel language/library | Triton, CubeCL, CUTLASS, vendor BLAS/DNN | How one operation is expressed and tuned; not complete framework parity. |
| Inference deployment | ONNX Runtime, TensorRT, ExecuTorch, WebLLM | Import/export, delegation, runtime footprint and supported targets. |
| LLM execution/serving | llama.cpp, vLLM | Quantization, real KV-cache decode, batching, memory and serving throughput. |
| General HPC portability | Kokkos, RAJA, SYCL; Halide as a DSL precedent | Abstraction boundary, schedule separation, specialization and productivity. |

Meganeura occupies several rows in a small implementation. That integration
is useful, but does not make other systems' individual mechanisms absent or
obsolete.

## Closest comparisons

### PyTorch and TorchInductor

PyTorch supplies a broad eager/autograd system plus graph compilation and
specialized backends. Its documented compiler modes include overhead-focused
CUDA graph execution and measured matmul/convolution selection. The default
mode is only one tradeoff. Our paper tested default Linux compilation and
eager MPS/CPU, not every automatic configuration.
[Compiler modes](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)

Meganeura's potential advantages are explicit static execution, a smaller
native closure and direct graphics integration. Its disadvantages include
operator breadth, dynamic-program support, distributed training, mature
vendor kernels and ecosystem tooling. A "Rust versus Python" explanation is
inadequate when both execute compiled GPU kernels. Use the frozen ratios for
observed performance and test stronger automatic baselines before attributing
small-shape wins to an inherent runtime advantage.

For development, eager tensor inspection, retained gradients, hooks and
anomaly/profiler tools are genuine PyTorch strengths. Meganeura's named
materialized debug sessions and plan/shader inspection help, but do not give
identical interactivity or coverage. See the source-backed
[observability comparison and debugging workflow](observability.md), including
the tools available in compiled PyTorch rather than only eager mode.

### Burn and CubeCL

Burn is the closest broad Rust-framework comparison: one API for training
and inference, dynamic graphs with JIT fusion, and backend decorators for
autodiff and other functionality. Current CubeCL paths include CUDA, ROCm,
Metal, Vulkan, WebGPU and CPU; the current README marks LibTorch deprecated
from 0.22.0. Its backend model is not just a set of handwritten eager kernels.
[Burn architecture and backend matrix](https://github.com/tracel-ai/burn)

Meganeura instead fixes a narrow native graphics path and explicit static
graph, while owning the compiler/runtime coupling. The comparison to run is
shared Vulkan/Metal hardware, matched graphs/gradients, cold and warm compile,
memory and deployment closure. Do not claim portable Rust autodiff or one-code
training-to-deployment is unique. Burn is also a source of reusable kernel-
abstraction ideas; adopting its whole stack would be a separate architectural
decision, not a prerequisite to learning from CubeCL.

### tinygrad

tinygrad is another direct challenge to the "minimal and general" premise.
It builds a low-level UOp graph, schedules kernels, performs BEAM search in
lowering, and executes through runtimes including lower-level hardware-queue
paths. Its frontend/autodiff-to-kernel pipeline makes it a more useful
conceptual comparison than an inference-only operator library.
[tinygrad's pipeline](https://docs.tinygrad.org/developer/developer/)

Meganeura's distinction is Rust-native embedding and the common Blade/Naga
graphics boundary, not invention of minimal tensor compilation or search.
Study how tinygrad represents transformations, caches work and validates
rewrites. Compare actual deployment prerequisites and supported devices;
userspace driver paths have a different maintenance/support tradeoff from
using installed graphics drivers. No speed ranking is established here.

### Luminal

Luminal's project describes a Rust inference compiler built from 15 primitive
operations and broad search, with CUDA/Metal targets. It claims to discover
complex fused algorithms from primitives. This is especially close to the
user's desired search-over-manual-specialization philosophy; those discovery
and performance claims were not independently tested in this audit.
[Luminal design](https://github.com/luminal-ai/luminal)

The key design question is search tractability versus retaining high-level
algorithms such as online attention. Meganeura currently keeps more semantic
archetypes and an integrated derivative/optimizer path. A useful experiment
would compare generated schedules for a few shared contractions/reductions,
including search cost and numerical qualification, not count primitive ops
and infer a winner.

### TVM / TensorIR / MetaSchedule

TVM is a strong precedent for separating an operator's computation from its
schedule and searching implementations. MetaSchedule explicitly separates
candidate generation, building, running, databases, cost models and search
budget allocation. Autotuning and persisted winners are established compiler
mechanisms, not a prospective Meganeura novelty.
[MetaSchedule tutorial](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)

Meganeura can borrow the separation in a much smaller Rust-owned tuner over
its existing variants. TVM's broader compiler machinery brings a larger
integration and learning surface; that does not establish a larger *deployed*
runtime for a matched artifact. Compare compilation and deployment separately.
Our added value would be the constrained, reliable integration and evidence
on the chosen graphics path, not a new general theory of autotuning.

### IREE

IREE compiles computation together with scheduling/execution logic using
MLIR, with AOT deployment across CPU architectures and GPU paths including
Vulkan, ROCm, CUDA and Metal. Its documented scope includes dynamic shapes,
control flow and streaming; experimental targets are labeled separately.
[IREE architecture and support matrix](https://iree.dev/)

It is therefore incorrect to imply Meganeura invented compiled Vulkan model
execution or that all alternatives require a heavyweight Python deployment.
Meganeura integrates graph construction and reverse autodiff directly into
its Rust library; IREE typically consumes programs exported from frontends.
"IREE is inference-only" is too broad: an exported differentiated computation
can be a compiler input even without a Meganeura-style training frontend.
Compare matched compiled programs, runtime/compiler footprints and graphics
interop rather than a bare runtime against an entire development toolchain.

### JAX / OpenXLA

JAX combines function transformations such as differentiation with traced,
compiled array programs; static specialization and compilation have
recompilation/control-flow tradeoffs. OpenXLA also supports persisted
autotuning results.
[JAX JIT](https://docs.jax.dev/en/latest/jit-compilation.html),
[XLA persisted autotuning](https://openxla.org/xla/persisted_autotuning)

JAX is a particularly useful conceptual comparison for treating training as
compilation rather than maintaining unrelated forward/backward engines.
Meganeura prioritizes a small native embedding interface and graphics-device
reach; it does not reproduce JAX's transformation ecosystem, distributed
execution or full compiler infrastructure. A future training comparison must
include optimizer/state semantics and compile amortization, not only matmul.

### MLX and Candle

MLX uses lazy arrays and shared-memory execution on Apple silicon, with
automatic differentiation and function transformations. It is a much more
relevant Apple-native alternative than comparing only to eager MPS.
[MLX](https://github.com/ml-explore/mlx)

MLX-LM already documents LoRA and QLoRA fine-tuning. The proposed Meganeura
quantized fine-tuning path could extend a common implementation across
Vulkan/Metal devices, but cannot be advertised as the first such capability
on Apple.
[MLX-LM fine-tuning](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)

Candle is a minimalist Rust tensor framework with CPU, CUDA and Metal
execution, pretrained-model examples and training support. Calling it
"inference only" or "no autodiff" is inaccurate.
[Candle](https://github.com/huggingface/candle)

Against Candle, the important distinction is Meganeura's explicit static
compiler and Vulkan reach versus Candle's tensor API and existing model
ecosystem. Against MLX, it is cross-vendor graphics portability versus a
system tailored to Apple. Neither comparison has matched Meganeura timings
here; source descriptions alone do not establish a performance ranking.

## Kernel-building systems are collaborators as well as alternatives

Triton supplies a GPU kernel language and configurable measured search; it
does not by itself supply a full model/autodiff/deployment framework. Its
autotuning supports keyed configurations, pruning and state-reset/restore
hooks. This is a useful example of bounding search and acknowledging state.
[Triton autotune API](https://triton-lang.org/main/python-api/generated/triton.autotune.html)

CUTLASS supplies composable CUDA linear-algebra implementation machinery,
including template and DSL interfaces. Generic specialization is not the
same as a model-specific handwritten kernel. Meganeura could aim for similarly
reusable tile concepts over its native graphics path, while recognizing that
NVIDIA-specific implementations can exploit facilities its common abstraction
does not expose.
[CUTLASS](https://github.com/NVIDIA/cutlass)

Vendor BLAS/DNN libraries are operation implementations, not competitors at
every framework layer. A PyTorch timing may involve generated code, a vendor
library, or both. To explain a gap, inspect the selected algorithm, layout,
precision and launch structure. The cuDNN autotuning setting is itself an
automatic baseline choice.
[PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

## Deployment and serving comparisons

ONNX Runtime delegates supported subgraphs to execution providers, with
provider ordering and fallback. This accommodates heterogeneous vendor paths,
but can introduce partition boundaries. Training offerings exist separately;
do not imply every inference provider supports backward execution.
[Execution-provider design](https://onnxruntime.ai/docs/execution-providers/)

TensorRT is an NVIDIA inference compiler/runtime that chooses tactics and
supports timing-cache reuse. It is a useful strong inference baseline, not an
optimizer-backed training replacement. A fair comparison must include engine
build cost, precision, dynamic-shape profile and deployment closure.
[TensorRT architecture](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html)

ExecuTorch starts from exported PyTorch programs and uses AOT transformation,
memory planning and backend delegation for edge deployment. This directly
challenges any claim that native, memory-planned edge execution is new. The
Meganeura distinction is keeping its own graph/autodiff/training loop in the
same library rather than making export the main workflow.
[ExecuTorch workflow](https://docs.pytorch.org/executorch/stable/intro-how-it-works.html)

llama.cpp is a C/C++ LLM execution stack with quantized models and multiple
backends including Vulkan and Metal. It is a necessary future comparator for
real cached decoding, not for arbitrary differentiation of all five paper
models.
[llama.cpp](https://github.com/ggml-org/llama.cpp)

vLLM's serving architecture includes frontend, engine and execution workers.
Service throughput/latency involves scheduling and cache management beyond
one static model invocation. Meganeura does not currently provide a comparable
full serving system; isolated one-token timing cannot establish superiority.
[vLLM architecture](https://docs.vllm.ai/en/latest/design/arch_overview/)

WebLLM brings MLC/TVM-based LLM inference to browser WebGPU. Browser sandbox,
shader features, download/compilation and cache behavior differ from native
Vulkan/Metal. Meganeura's native cooperative shader path is not automatically
a browser-compatible implementation.
[WebLLM](https://webllm.mlc.ai/)

## What an HPC audience will compare

Kokkos separates execution and memory concerns through portable abstractions;
RAJA supplies portable parallel-loop building blocks. They solve general HPC
kernel portability rather than providing an ML graph, autodiff and optimizer.
Meganeura makes domain-specific decisions above its graphics API layer.
[Kokkos programming model](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/ProgrammingModel.html),
[RAJA](https://github.com/LLNL/RAJA)

SYCL standardizes a C++ heterogeneous execution/memory model with backend
interoperability. It is a language/runtime portability boundary, not a neural
network compiler. A hypothetical SYCL backend would still need Meganeura's
graph, autodiff and algorithm implementation work; the experiment would ask
which devices and optimization facilities that boundary exposes.
[SYCL 2020 specification](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html)

Halide is an important precedent for separating the algorithm from its
schedule. That principle is compatible with a general system generating
specialized kernels. "One source" has never required one universal launch
geometry or one implementation for all machines.
[Halide](https://halide-lang.org/)

For productivity, a smaller closure and codebase are operational advantages,
not direct evidence of reduced developer effort. Programming-model studies
can instead measure semantic divergence between implementations; Meganeura
has not collected a comparable maintenance/effort dataset.
[Lin, Deakin and McIntosh-Smith](https://hpc.tomdeakin.com/bibliography/p3hpc24.html)

## The defensible positioning

The contribution is a compact, inspectable native training/inference stack
with a shared graphics execution boundary, plus a frozen, correctness-gated
cross-machine study and an embedded deployment example. Autodiff, graph
fusion, static memory planning, graphics-API inference, Rust ML and autotuning
all have precedents. The most valuable next comparisons are Burn/tinygrad for
minimal general training, MLX on Apple, stronger automatic PyTorch modes,
IREE/TVM for compilation tradeoffs, and llama.cpp for real decode.

An unmeasured alternative is not a loss and not a win. Before adding a new
benchmark, write down which layer and question it tests and which parts of
the two systems are included in the cost.
