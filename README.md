[![CI](https://github.com/kvark/meganeura/actions/workflows/ci.yml/badge.svg)](https://github.com/kvark/meganeura/actions/workflows/ci.yml)
[![Docs](https://docs.rs/meganeura/badge.svg)](https://docs.rs/meganeura)
[![Crates.io](https://img.shields.io/crates/v/meganeura.svg?label=meganeura)](https://crates.io/crates/meganeura)
[![arXiv](https://img.shields.io/badge/arXiv-2608.01563-b31b1b.svg)](https://arxiv.org/abs/2608.01563)

# meganeura

**Portable neural-network training and inference in Rust.** Look, ma, no CUDA!

[![logo](https://github.com/kvark/meganeura/raw/main/etc/logo.png)](/kvark/meganeura/blob/main/etc/logo.png)

**Warning:** project is actively developed. Mostly optimization work, expanding the ops coverage, but with occasional fixes in correctness.

## Example

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

For local iteration see [testing and coverage](docs/testing.md) and [debugging the stack](#debugging).

## Why Meganeura

**Portable.** GPU access is provided by
[blade-graphics](https://github.com/kvark/blade/tree/main/blade-graphics):
Vulkan on Linux, Windows, and Android, and Metal on Apple platforms. Mesa's
Lavapipe provides a software Vulkan target for headless CI. Doesn't need any runtime.

**Lean.** Around 50K LOC of Rust+WGSL code in this repository.
Automatic shader composition based on kernel archetypes: pointwise, reduction, matmul, convolution, attention.
May produce a single 12Mb self-contained binary for deployment.

**Fast.** Meganeura is pretty fast.
It tries to be competitive with vendor-native ML stacks but lands at around 0.5x of their performance today.
Your results may very by model, device, precision policy, and the driver.
See [Inferena](https://inferena.tech) tables to get an idea.

## How it compares

|                                                 |GPU backends                        |Training      |Approach                                |
|-------------------------------------------------|------------------------------------|--------------|----------------------------------------|
|**Meganeura**                                    |blade-graphics (Vulkan, Metal)      |yes           |graph IR + rewrites + specialized WGSL         |
|[Candle](https://github.com/huggingface/candle)  |CUDA, Metal, CPU                    |yes           |tensor API, native kernels              |
|[Burn](https://github.com/tracel-ai/burn)        |CubeCL: CUDA, ROCm, Metal, Vulkan, WebGPU; CPU paths |yes |modular backends, JIT fusion      |
|[tch-rs](https://github.com/LaurentMazare/tch-rs)|CUDA, CPU (via libtorch)            |yes           |PyTorch FFI bindings                    |

Meganeura's strong sides are uniform graph, autodiff, compiler, and runtime stack for
both training and inference across desktop and edge-class Vulkan/Metal
devices.

## Install

```
cargo add meganeura
```

See the [changelog](CHANGELOG.md) for API and feature changes from 0.2.
CI assembles the packaged crate; full registry verification waits on a Blade
release that includes this timing API.

Features:
- "hf-hub" to enable HuggingFace downloads
- "models" for built-in models: SmolLM2, SmolVLA, SD_Unet, ResNet, Whisper
- "gguf" for [GGUF](https://huggingface.co/docs/hub/en/gguf) format loading of weights and graphs

Worked examples live in [`examples/`](https://github.com/kvark/meganeura/tree/main/examples):

- [`mnist.rs`](https://github.com/kvark/meganeura/blob/main/examples/mnist.rs) — MNIST training end to end.
- [`train_deploy.rs`](https://github.com/kvark/meganeura/blob/main/examples/train_deploy.rs) — optimizer-backed training, checkpoint save, and reload into a fresh inference session.
- [`smollm2.rs`](https://github.com/kvark/meganeura/blob/main/examples/smollm2.rs) — LLM inference with HuggingFace weights.

Current checkpoints store logical tensors without device padding and preflight
the restore before mutation. Adam/LaProp moments are allocated only when
requested; SGD and forward/backward-only sessions avoid that unused storage.
See the [checkpoint implementation](src/runtime/checkpoint.rs) for format
compatibility and restore checks. Resident buffer counts do not measure
driver peak memory.

Pretrained models can be loaded from ONNX or NNEF via `meganeura::load_onnx(...)` / `meganeura::load_nnef(...)`.
Both lower through Meganeura’s IR, so the same graph rewrites apply to imported graphs and hand-built ones.

A GGUF file needs nothing alongside it. It carries no graph, but it carries a
description — an architecture name and a set of dimensions — and
`load::gguf` reads that into a graph, fills it from the file's own tensors,
and uses the tokenizer the file embeds:

```rust
use meganeura::load::gguf::{load_gguf, GenerationOptions};

let model = load_gguf(std::path::Path::new("model.gguf"))?;
let mut generator = model.generator(2048)?;
println!("{}", generator.generate("The meaning of life is", &GenerationOptions::default())?);
```

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

## Environment variables

All of the environment variables are resolved in `SessionConfig::from_env()` and never visible to the core modules directly.

| Variable | Effect |
|---|---|
| `MEGANEURA_DISABLE_COOP` | Force the portable scalar matmul path (regression diagnosis). |
| `MEGANEURA_COOP_F16` | Allow f16-input cooperative tiles even for precision-sensitive derivative work; requires numerical validation. Default `Auto` protects derivatives and can already use f16 tiles for forward work. |
| `MEGANEURA_FLASH_FWD_COOP=0` | Disable only cooperative flash-attention forward. |
| `MEGANEURA_FLASH_BWD_COOP` | Enable the experimental reduced-precision flash backward. |
| `MEGANEURA_NO_ALIAS` | Disable buffer lifetime aliasing (every value gets its own allocation). |
| `MEGANEURA_NO_DEVICE_LOCAL` | Keep all buffers host-visible. |
| `MEGANEURA_SERIAL_DISPATCH` | One compute pass per dispatch — serial execution for bisection. |
| `MEGANEURA_NO_WINOGRAD` | Skip the Conv2d-to-Winograd rewrite; its selection heuristic weighs channel counts only, so this measures which side of it a workload belongs on. |
| `MEGANEURA_PIN_BUFS=3,25-40` | Force-pin logical buffers to bisect aliasing corruption. |
| `MEGANEURA_DUMP_PLAN` | Dump dispatch order, provenance, and the alias map at build. |
| `MEGANEURA_DUMP_WGSL=<dir>` | Write every generated shader into `<dir>`. |
| `MEGANEURA_OPTIMIZER` | Rewrite mode: `off` \| `greedy` \| `egglog-windowed` \| `egglog-outlined` \| `egglog-whole`. |
| `MEGANEURA_EGRAPH_COST` | Extraction objective: `ast-size` \| `tensor-traffic`. |
| `MEGANEURA_EGRAPH_CUTOFF=<n>` | Saturation segment-size ceiling (default 300). |
| `MEGANEURA_GREEDY_PACK_SWIGLU=0` | Skip packing consecutive SwiGLU ops into one parameter buffer during the greedy sweep. |
| `MEGANEURA_DEVICE_PARAMETERS` | Experimental placement of unaliased parameter buffers on the device: `1` → device-transient, `device-buddy` → device. Default is host-visible. |
| `MEGANEURA_REUSE_UPLOAD` | Reuse one staging buffer across `set_parameter` uploads instead of restaging per parameter. |
| `MEGANEURA_TUNE` | Opt-in bounded f32 matmul/convolution search at build (`SessionConfig { tune: true }`), using private scratch. |
| `MEGANEURA_FLASH_EPT_CAP=<n>` | Flash forward elements-per-thread cap (power of two ≥ 2). |
| `MEGANEURA_FLASH_GRAD_Q_EPT_CAP=<n>` | EPT cap for flash dQ backward. |
| `MEGANEURA_FLASH_GRAD_KV_EPT_CAP=<n>` | EPT cap for fused flash dK/dV backward. |
| `MEGANEURA_FLASH_BWD_EPT_CAP=<n>` | Shared fallback cap for both flash backward kernels. |
| `MEGANEURA_MATMUL_K_STAGE=<n>` | Scalar tiled matmul K staging depth: 8 \| 16 \| 32 (default 32). |
| `MEGANEURA_INTERLEAVE_COLUMNS` | Stagger scalar-matmul B loads across columns (16 lanes apart) instead of through consecutive ones. |
| `MEGANEURA_DEVICE_ID=0x744c` | Adapter selection by numeric device id. |
| `MEGANEURA_GPU_TIMING` | Enable hardware timestamp pools (set before context creation). |
| `MEGANEURA_GPU_CAPTURE` | Enable Blade's native-tool labels and shader debug information before context creation; independent of GPU timing. |

`Session::tune_with(TuneOptions)` enables auto-tuning across multiple dimensions. See the [tuning API](src/tune.rs) and
[whole-step experiment](examples/tune_session.rs).

## Debugging

See [testing and coverage](docs/testing.md) for the failure-investigation
workflow and [performance profiling](docs/performance-profiling.md) for
timing and capture tools.

Three levels, cheapest first:

- **Provenance everywhere.** Name values while building
  (`let h = g.matmul(x, w); let h = g.named(h, "blk3.qkv");` — `nn` layers
  name their outputs automatically) and the name follows the value through
  autodiff, rewrites, and fusion into dispatch labels, profiler rows,
  `MEGANEURA_DUMP_PLAN`, and NaN reports. Shape panics report the
  model-builder line that created the bad node.
- **Debug sessions.** `build(&g, SessionConfig::debug())` disables buffer
  aliasing and keeps everything host-visible: `session.read_node_by_name("blk3.qkv")`
  reads materialized values after `step()` and `wait()`. Graph rewrites and
  precision policy are separate controls. `session.step_debug()` scans primary
  output prefixes after execution; its first reported NaN/Inf is not a complete
  root-cause guarantee. Fused-away or aliased values return structured errors.
- **Eager evaluation.** `meganeura::eager::Eager` runs the graph you are
  *still building*, one `eval(&g, node)` at a time, on the same kernels the
  compiled path uses — the PyTorch-style inspect-as-you-go loop. The same
  graph then compiles unchanged via `build_session` for training speed.

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
MEGANEURA_TRACE=trace.pftrace cargo run --example mnist --features profiler
```

Open the trace in [Perfetto](https://ui.perfetto.dev):

[![perfetto trace](https://github.com/kvark/meganeura/raw/main/etc/example-trace.png)](/kvark/meganeura/blob/main/etc/example-trace.png)

## Citation

Machine-readable author and project metadata is available in
[`CITATION.cff`](https://github.com/kvark/meganeura/blob/main/CITATION.cff).
The paper citation and archival identifier will be added after publication.
