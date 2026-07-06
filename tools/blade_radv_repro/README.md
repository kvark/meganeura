# blade-radv-repro

Standalone Blade compute-pipeline repro for a memory-consistency bug
on **RADV STRIX1** (AMD Radeon 890M, Phoenix iGPU, Mesa).

## Symptom

A compute pipeline made of several dispatches that read each other's
outputs produces **different** results depending on whether the
dispatches share one submit (with pipeline barriers between them) or
are split into separate submits (with `vkWaitForFences` between them).

Sequential submits agree with the CPU-sequential interpretation;
single-submit gives a deterministic but wrong answer. The same code runs
correctly on Mesa drivers for non-Phoenix AMD GPUs and on other vendors
in our testing.

## Minimal trigger

5 dispatches, single 6 M-element f32 buffer (24 MiB):

```
scale → relu → heavy → heavy → add(heavy_out, scale_out)
```

Each shader is trivial except `heavy`, which scatter-reads
`src_a[(i*17 + k) % n]` for k in 0..64. Buffers are
`gpu::Memory::Device` (RAM-resident, not host-visible).

```
$ BLOCKS=1 START_N=6000000 cargo run --release
Device: AMD Radeon 890M Graphics (RADV STRIX1)
blocks=1 buffers=6 dispatches=5 max_n=6000000
--- results ---
  mode=split     last_buf=5 n=6000000 nz=6000000 sum=3.0700e6 ...
  mode=multipass last_buf=5 n=6000000 nz=6000000 sum=3.1014e6 ...
  mode=single    last_buf=5 n=6000000 nz=6000000 sum=3.1014e6 ...
DIVERGENCE: mode=multipass sum=3.101376e6 vs split reference=3.070011e6
DIVERGENCE: mode=single sum=3.101376e6 vs split reference=3.070011e6
```

Exit code 2 = bug reproduced; 0 = all modes agree.

## Modes

| mode        | submits   | barriers between dispatches            |
|-------------|-----------|----------------------------------------|
| `split`     | N         | `vkWaitForFences` (queue drain)        |
| `multipass` | 1         | per-pass `vkCmdPipelineBarrier`        |
| `single`    | 1         | per-group `vkCmdPipelineBarrier`       |

All three should produce identical output. They don't on RADV STRIX1.

Blade's pipeline barrier (`blade-graphics/src/vulkan/command.rs:369`):

```
src_stage = ALL_COMMANDS
dst_stage = ALL_COMMANDS
src_access = MEMORY_WRITE | TRANSFER_WRITE
dst_access = MEMORY_READ | MEMORY_WRITE | TRANSFER_WRITE | TRANSFER_READ
```

By the Vulkan spec this is a maximal memory barrier — every prior write
is visible to every subsequent read. The Khronos sync validation layer
does not flag the dispatch sequence.

## Hypothesis

The Phoenix GFX11.5 IP either:

1. requires a finer-grained L2/scalar-cache flush that the broad
   `MEMORY_WRITE → MEMORY_READ` access pair doesn't actually emit, or
2. has a queue-internal cache that only drains on submit boundary.

## Repro environment

- Mesa: `vulkaninfo | grep driverInfo` (check Mesa version)
- GPU: AMD Radeon 890M (Phoenix / RDNA 3.5)
- API: Vulkan 1.3
- Blade rev: `72c30c1`

## Usage

```
cargo run --release                        # 15 blocks, 1 MiB start
BLOCKS=N START_N=M cargo run --release     # custom size
cargo run --release -- split               # single mode only
```

## Original context

Found while porting Google's `magenta-realtime` SpectroStream decoder
to [meganeura](https://github.com/kvark/meganeura). Decoder output
went all-zeros past block 5 with a single submit; splitting submits
restored correct output. This repro extracts the same pattern from
pure Blade — no meganeura involvement.
