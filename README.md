# meganeura

[![CI](https://github.com/kvark/meganeura/actions/workflows/ci.yml/badge.svg)](https://github.com/kvark/meganeura/actions/workflows/ci.yml)
[![Docs](https://docs.rs/meganeura/badge.svg)](https://docs.rs/meganeura)
[![Crates.io](https://img.shields.io/crates/v/meganeura.svg?label=meganeura)](https://crates.io/crates/meganeura)
[![arXiv](https://img.shields.io/badge/arXiv-2608.01563-b31b1b.svg)](https://arxiv.org/abs/2608.01563)

**Portable neural-network training and inference in Rust.** Meganeura uses
Vulkan on Linux, Windows, and Android and Metal on Apple platforms. It does
not require CUDA, ROCm, or Python at runtime.

[![logo](https://github.com/kvark/meganeura/raw/main/etc/logo.png)](/kvark/meganeura/blob/main/etc/logo.png)

> **Status:** actively developed; APIs and benchmark methodology are still in
> motion. Current workloads include SmolLM2, a SmolVLA action expert,
> ResNet-50, the Whisper-tiny encoder, and a scaled, timestep- and
> text-conditioned latent-diffusion U-Net. Issues and pull requests are
> welcome.

Define a graph, call `build_session`, train. Meganeura handles autodiff,
graph rewrites, WGSL specialization, Naga parsing and validation, and GPU
dispatch automatically. The rewrite engine supports a fast deterministic
greedy mode and experimental equality-saturation modes.

```rust
use meganeura::{Graph, Trainer, TrainConfig, build_session};

let mut g = Graph::new();
let x = g.input("x", &[32, 784]);
let labels = g.input("labels", &[32, 10]);

let w1 = g.parameter("w1", &[784, 128]);
let h = g.relu(g.matmul(x, w1));
let w2 = g.parameter("w2", &[128, 10]);
let logits = g.matmul(h, w2);

let loss = g.cross_entropy_loss(logits, labels);
g.set_outputs(vec![loss]);

// autodiff + graph rewrite + compile + GPU init
let session = build_session(&g);
let mut trainer = Trainer::new(session, TrainConfig::default());
trainer.train(&mut data, /* epochs = */ 10); // data loader: see examples/mnist.rs
```

A two-layer MLP, trained end to end on the GPU, in one screen.

## Why Meganeura

**Fast.** Meganeura is competitive with vendor-native ML stacks on selected
workloads while retaining one Vulkan/Metal implementation. Results vary
substantially by model, device, precision policy, and driver. The historical
[Inferena](https://inferena.tech) tables are useful exploratory data, but they
predate the audited paper protocol and should not be treated as
publication-grade comparisons. The new protocol reports raw samples, uses
matched workloads and full forward-plus-backward timing, and separates strict
f32 from reduced-input accelerated modes.

PyTorch CUDA currently leads several RTX 5080 workloads, especially training.
Meganeura's strongest result is therefore not universal speed superiority; it
is how much of that performance can be reached through a portable execution
stack that also runs on AMD, Intel, and Apple GPUs.

**Portable.** GPU access is provided by
[blade-graphics](https://github.com/kvark/blade/tree/main/blade-graphics):
Vulkan on Linux, Windows, and Android, and Metal on Apple platforms. Mesa's
Lavapipe provides a software Vulkan target for headless CI. The compute stack
does not require CUDA or ROCm, although performance and feature availability
still depend on each vendor's driver.

**Composable.** A small set of
[kernel archetypes](https://github.com/kvark/meganeura/blob/main/docs/kernel-archetypes.md)
— pointwise, reduction, matmul, convolution, and attention — compose into
specialized GPU shaders at compile time. The rewrite set recognizes
equivalent fused forms (for example, `x * sigmoid(x)` → SiLU and
`SiLU(gate) * up` → SwiGLU).
It can run either as a deterministic greedy pass or through equality
saturation with a traffic-aware extraction cost. Current ablations find the
same selected graph for the benchmark rewrite set, so equality saturation is
research infrastructure rather than a claimed source of runtime speedup.
Consolidating the remaining hand-written WGSL variants into parameterized
generators is active work.

## How it compares

|                                                 |GPU backends                        |Training      |Approach                                |
|-------------------------------------------------|------------------------------------|--------------|----------------------------------------|
|**Meganeura**                                    |blade-graphics (Vulkan, Metal)      |yes           |graph IR + rewrites + specialized WGSL         |
|[Candle](https://github.com/huggingface/candle)  |CUDA, Metal, CPU                    |limited       |eager tensors, hand-written kernels     |
|[Burn](https://github.com/tracel-ai/burn)        |CUDA, wgpu, NDArray, LibTorch       |yes           |modular multi-backend                   |
|[tch-rs](https://github.com/LaurentMazare/tch-rs)|CUDA, CPU (via libtorch)            |yes           |PyTorch FFI bindings                    |

Meganeura's wedge is a uniform graph, autodiff, compiler, and runtime stack for
both training and inference across desktop and edge-class Vulkan/Metal
devices.

## Install

```
cargo add meganeura
```

Worked examples live in [`examples/`](https://github.com/kvark/meganeura/tree/main/examples):

- [`mnist.rs`](https://github.com/kvark/meganeura/blob/main/examples/mnist.rs) — MNIST training end to end.
- [`train_deploy.rs`](https://github.com/kvark/meganeura/blob/main/examples/train_deploy.rs) — optimizer-backed training, checkpoint save, and reload into a fresh inference session.
- [`smollm2.rs`](https://github.com/kvark/meganeura/blob/main/examples/smollm2.rs) — LLM inference with HuggingFace weights.

Pretrained models can be loaded from ONNX or NNEF via `meganeura::load_onnx(...)` / `meganeura::load_nnef(...)`. Both lower through Meganeura’s IR, so the same graph rewrites apply to imported graphs and hand-built ones.

## System requirements

Meganeura runs best when the selected driver exposes hardware-accelerated
cooperative matrix operations:

- **Vulkan** —
  [`VK_KHR_cooperative_matrix`](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_cooperative_matrix.html).
- **Metal** — simdgroup matrix support.

Falls back to scalar matmul on older hardware. Headless Lavapipe works for CI.

When several adapters are present, select one with its backend-reported numeric
device ID (on Vulkan this is normally the PCI device ID, not an adapter ordinal):

```sh
MEGANEURA_DEVICE_ID=0x744c cargo run --release --example mnist
```

Decimal IDs are accepted too.

## Environment variables

The library core never reads the environment: `compile`, `runtime`,
`codegen`, and `optimize` accept strongly typed options only. Env-driven
behavior is an explicit client opt-in through the `from_env` constructors
in `meganeura::config` — `SessionConfig::from_env()` is the one-liner
that resolves everything below (the repo's own examples, benches, and
tests use it; embedders that want a hermetic library simply never call
it). Every `MEGANEURA_*` variable is declared in
`meganeura::config::REGISTRY` (a test pins this table against it).
Semantics are uniform: boolean variables treat unset as their default,
`0` as off, and anything else as on; `from_env` logs the active
overrides, and unrecognized `MEGANEURA_*` names produce a warning
instead of silently doing nothing. Fields assigned after `from_env`
win, so explicit code always has the last word.

| Variable | Effect |
|---|---|
| `MEGANEURA_DISABLE_COOP` | Force the portable scalar matmul path (regression diagnosis). |
| `MEGANEURA_COOP_F16` | Opt in to f16-input cooperative tiles when no f32 tile is advertised. |
| `MEGANEURA_FLASH_FWD_COOP=0` | Disable only cooperative flash-attention forward. |
| `MEGANEURA_FLASH_BWD_COOP` | Enable the experimental reduced-precision flash backward. |
| `MEGANEURA_NO_ALIAS` | Disable buffer lifetime aliasing (every value gets its own allocation). |
| `MEGANEURA_NO_DEVICE_LOCAL` | Keep all buffers host-visible. |
| `MEGANEURA_SERIAL_DISPATCH` | One compute pass per dispatch — serial execution for bisection. |
| `MEGANEURA_PIN_BUFS=3,25-40` | Force-pin logical buffers to bisect aliasing corruption. |
| `MEGANEURA_DUMP_PLAN` | Dump dispatch order, provenance, and the alias map at build. |
| `MEGANEURA_DUMP_WGSL=<dir>` | Write every generated shader into `<dir>`. |
| `MEGANEURA_OPTIMIZER` | Rewrite mode: `off` \| `greedy` \| `egglog-windowed` \| `egglog-outlined` \| `egglog-whole`. |
| `MEGANEURA_EGRAPH_COST` | Extraction objective: `ast-size` \| `tensor-traffic`. |
| `MEGANEURA_EGRAPH_CUTOFF=<n>` | Saturation segment-size ceiling (default 300). |
| `MEGANEURA_TUNE` | Measure coop vs scalar per kernel family at session build; keep the faster (`SessionConfig { tune: true }` equivalent). |
| `MEGANEURA_FLASH_EPT_CAP=<n>` | Flash forward elements-per-thread cap (power of two ≥ 2). |
| `MEGANEURA_FLASH_GRAD_Q_EPT_CAP=<n>` | EPT cap for flash dQ backward. |
| `MEGANEURA_FLASH_GRAD_KV_EPT_CAP=<n>` | EPT cap for fused flash dK/dV backward. |
| `MEGANEURA_FLASH_BWD_EPT_CAP=<n>` | Shared fallback cap for both flash backward kernels. |
| `MEGANEURA_DEVICE_ID=0x744c` | Adapter selection by numeric device id. |
| `MEGANEURA_GPU_TIMING` | Enable hardware timestamp pools (set before context creation). |

## Debugging

Three levels, cheapest first:

- **Provenance everywhere.** Name values while building
  (`let h = g.matmul(x, w); let h = g.named(h, "blk3.qkv");` — `nn` layers
  name their outputs automatically) and the name follows the value through
  autodiff, rewrites, and fusion into dispatch labels, profiler rows,
  `MEGANEURA_DUMP_PLAN`, and NaN reports. Shape panics report the
  model-builder line that created the bad node.
- **Debug sessions.** `build(&g, SessionConfig::debug())` disables buffer
  aliasing and keeps everything host-visible: `session.read_node_by_name("blk3.qkv")`
  returns any value after a step, and `session.step_debug()` attributes the
  first NaN/Inf to a named dispatch. Fused-away or aliased values return a
  structured error instead of garbage.
- **Eager evaluation.** `meganeura::eager::Eager` runs the graph you are
  *still building*, one `eval(&g, node)` at a time, on the same kernels the
  compiled path uses — the PyTorch-style inspect-as-you-go loop. The same
  graph then compiles unchanged via `build_session` for training speed.

Hosted CI executes Linux Vulkan (Lavapipe) and macOS Metal tests and compile-checks
Windows. Generated Vulkan shaders, including the f16 cooperative-matrix path used
on supported NVIDIA and AMD adapters, are also validated and translated to SPIR-V
offline. Real AMD and NVIDIA hardware testing remains part of release
qualification because software drivers cannot reproduce vendor-driver behavior.

## Profiling

For a repeatable per-dispatch JSON profile:

```sh
MEGANEURA_GPU_TIMING=1 \
  cargo run --release --example profile_session -- gap-profile.json
```

The report retains raw hardware-timestamp samples, selected pipeline variants,
workgroup geometry, forward/backward and kernel-family aggregates, device and
memory metadata, and the instrumentation overhead relative to normal grouped
execution. See [structured performance profiling](docs/performance-profiling.md)
for the Inferena harness and interpretation rules.

For a CPU/GPU timeline:

```
MEGANEURA_TRACE=trace.pftrace cargo run --example mnist
```

Open the trace in [Perfetto](https://ui.perfetto.dev):

[![perfetto trace](https://github.com/kvark/meganeura/raw/main/etc/example-trace.png)](/kvark/meganeura/blob/main/etc/example-trace.png)

## Citation

Machine-readable author and project metadata is available in
[`CITATION.cff`](https://github.com/kvark/meganeura/blob/main/CITATION.cff).
The paper citation and archival identifier will be added after publication.

## Contributing

Early project, small API surface, small community — a good time to show up. Open an issue before starting anything sizeable so we can align on direction.
