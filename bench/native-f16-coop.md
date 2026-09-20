# Native F16 cooperative-matrix experiment

Source-only follow-up to [submission tuning](submission-autotune.md), based on
Meganeura `741b35aaa36aa5029b24c36779b84e3dd280f469`. No new kernel is installed
in the runtime. The small runtime edit only adapts the square-tile capability
query to this branch's experimental Blade API. The submitted paper, Inferena
cohort and production PRs are unchanged. Keep raw outputs and binaries outside
Git.

Corrected pass attribution identifies dense F16-weight matmuls as the largest
prefill target: about 69% of instrumented pass intervals on the RTX 5070 and
77% on the B570. These are not uninstrumented shader costs or a decomposition
of the gap to llama.cpp. The timestamp diagnostic is preserved separately at
Meganeura `484792e21259ece81aafeeff6b6fca79ea6cd807`; its Blade fix is
[PR #398](https://github.com/kvark/blade/pull/398).

## What this tests

The manifest pins Blade `8f80abb794c89b1299ec2a0e13386ee5296aad0b`, based on
the timestamp fix. Its experimental capability API exposes the advertised
subgroup-scoped `[M,N,K]` shapes with dimensions representable by Naga (8 or 16),
instead of discarding non-square modes. This machine reports:

| GPU | Default subgroup width | F16-input, F32-accumulator shapes |
| --- | ---: | --- |
| RTX 5070 | 32 | 16x8x8, 16x8x16, 16x16x16 |
| Arc B570 | 16 | 8x16x16 |

`native_f16_coop.rs` compares existing scalar F16-weight tiles, a scalar control
using the new staging layout, and several cooperative layouts. On square-capable
devices it also compares the existing cooperative generator using F32 weight
storage against a minimal adaptation using native F16 storage. These controls
use the same exact weights and F16-input arithmetic. They do not require the
new generic shader to win.

The generic shader stages F32 activations and native F16 weights through F16
workgroup memory, with F32 accumulators. Workgroup blocks and one, two or four
subgroups distribute independent output tiles. It handles incomplete input and
output tiles. Aligned outputs use direct cooperative stores; aligned residual
adds seed the accumulator from the residual. Other outputs use shared staging.
Residual seeding changes addition order, which needs qualification on real data.

WGSL only spells square cooperative types. The diagnostic changes both the
matrix types and cooperative-load dimensions in Naga IR to the advertised
dimensions. It retains Naga validation. It uses the device's default subgroup
width with ordinary pipeline flags, not varying subgroup sizes. This is a
Vulkan diagnostic; the Blade change preserves Metal's existing square mode,
but does not add or test a Metal kernel.

## Qualification and measurement

Shapes `[M,N,K]` are `[17,35,41]`, `[128,576,576]`, `[128,3072,576]` and
`[128,576,1536]`, each with and without a residual add. All inputs are
deterministic random multiples of 1/128, exactly representable as F16. A CPU
F64 dot product supplies the reference. These inputs isolate indexing and
layout errors; they do **not** establish the accuracy of rounding arbitrary
F32 activations to F16. Any runtime integration must respect the caller's
precision policy and qualify representative data. Native F16 weight storage
alone does not authorize reduced-precision activation arithmetic.

Before every sample, the entire output and 256 trailing guard words are reset
to a NaN sentinel. Every logical output must be finite and within 0.0001 of
the CPU reference; all guards must remain bit-identical. Rejected candidates
have no retained timing samples. The current cases pass with zero maximum
absolute error: 204 candidate/shape combinations on NVIDIA and 80 on Intel.

Each candidate gets ten samples, with three discarded as warmup. A sample
freshly records 32 dependent dispatches, then reads the output. Reset and CPU
checking are outside timing. `gpu_us` contains corrected pass intervals;
`wall_us` includes recording, submission, waiting and one amortized readback.
`compile_ms` includes shader creation and native pipeline creation, with
existing driver caches. `MEGANEURA_GPU_CAPTURE=1` additionally requests native
compiler statistics; those runs are separate from timing comparisons.

GPU runs are sequential and never overlap builds. The host is the i5-12400F,
with affinity `0,2,4,6,8,10`, NVIDIA driver 595.91.07 and Mesa 26.0.3. Clocks
are not fixed. Candidate order is fixed within a process; these are exploratory
microbenchmarks, not held-out autotuner or whole-model results. Larger layouts
lose on some shapes. A production tuner must measure the available choices;
there is no device table selecting a presumed winner.

## Compiler findings

Naga 30.0.1 panics while generating SPIR-V for a cooperative store whose pointer
or stride needs expressions that have not yet been emitted. The included
three-line `naga-coop-store.patch` flushes those expressions before appending
the store statement. The example's single CPU regression fails without this
patch and passes with it. The published crate identifies wgpu revision
`40f4a34ebaf56f9a046231f54125ad046239d3f3`. The original registry crate was not
modified. Upstream issue creation was denied, so the patch and reproducer are
preserved here; no upstream issue or merge is claimed.

An earlier generic shader stored `num_subgroups` in workgroup memory, then
returned early if it differed from the expected count. That version failed
numerically on NVIDIA while the scalar-staging control passed. Removing this
guard made every NVIDIA case pass without changing arithmetic, output checks
or tolerances. Intel passed both versions. This isolates a triggering shader
pattern, not the responsible compiler layer; it is not yet an upstream driver
bug diagnosis. Do not reuse timings from the rejected version.

Intel pipeline statistics also show that large accumulator sets can spill.
For the earlier direct-output 32x32x16 block, one subgroup generated 33 spills
and 34 fills, while two generated none. A 32x64x32 block with two subgroups
generated 67 spills and 74 fills. Those statistics explain a concrete cost in
these layouts, not all of Intel's performance gap. The driver's estimated
cycle count is not a hardware measurement.

## Reproduce

The diagnostic needs the included Naga patch in a private copy, not an edited
Cargo registry or a vendored compiler in this repository:

```sh
naga_coop_dir=$(mktemp -d)
curl -L https://crates.io/api/v1/crates/naga/30.0.1/download \
  | tar -xz -C "$naga_coop_dir" --strip-components=1
patch -d "$naga_coop_dir" -p1 < bench/naga-coop-store.patch

cargo test --features gguf --lib --example native_f16_coop -j2 \
  --config "patch.crates-io.naga.path=\"$naga_coop_dir\""
cargo build --release --example native_f16_coop -j2 \
  --config "patch.crates-io.naga.path=\"$naga_coop_dir\""

env -u LD_PRELOAD VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  taskset -c 0,2,4,6,8,10 target/release/examples/native_f16_coop \
  > /tmp/native-f16-5070.json 2> /tmp/native-f16-5070.log
```

Use `intel_icd.json` for B570, after the NVIDIA process exits. A debug build
enables Vulkan validation. On this memory-constrained machine commands were
wrapped with Inferena's `scripts/limited.py --memory-mib 4096 --seconds 300 --`;
choose an appropriate process/cgroup bound on another machine. The CPU test
invocation excludes opt-in GPU tests and does not run the example's GPU main.

Before production: fold the useful choices into the existing generator and
qualified tuning pass, check real model data under the declared precision
policy, then repeat the whole-model comparison. Microbenchmark improvements
alone are not grounds for a new paper cohort.
