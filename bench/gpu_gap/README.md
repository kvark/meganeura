# GPU gap probes, September 21

Research sources only. This directory is kept on the experiment branch, not in
the production PR. No timings, traces, models or compiled binaries are committed.
The concise conclusions are in `docs/gpu-gap-2026-09.md` on the review branch.

These are Linux diagnostics for the RTX 5070 and Arc B570, not the distributed
P3HPC collection protocol. Use one GPU at a time, with no concurrent builds or
CPU timing. Keep the ordinary validation policy. Do not reset the device, change
clocks, or disable validation to get a measurement through.

## Layout and revisions

The scripts expect a scratch directory containing three worktrees named
`source` (Meganeura), `blade`, and `inferena`, with the scripts copied beside
them. Create it outside every checkout. From the Meganeura checkout, after
fetching the experiment branches in the three repositories:

```sh
GAP_ROOT=$(mktemp -d)
git worktree add --detach "$GAP_ROOT/source" origin/experiment/gpu-gap-2026-09-21
git -C ../blade worktree add --detach "$GAP_ROOT/blade" d6f578b761f09e5cfb67cdab7203133fa7176e76
git -C ../inferena worktree add --detach "$GAP_ROOT/inferena" origin/experiment/gpu-gap-2026-09-21
cp "$GAP_ROOT/source/bench/gpu_gap/"* "$GAP_ROOT/"
cd "$GAP_ROOT"
```

The tested PyTorch wheels are 2.13.0+cu130 and 2.13.0+xpu, both at source
`cf30153c4c131c8164ee7798e5022d810682e2cb`, with Python 3.13.13. Select the
intended existing Inferena venv explicitly for paired runs. Do not install one
wheel over the other. SmolLM2 needs its pinned local model files; prepare them
using Inferena's normal model preparation command before setting offline mode.

Key source checkpoints:

| Question | Meganeura | Blade | Inferena |
| --- | --- | --- | --- |
| Original scalar-attention layout | `ab062b8` | `b885af1` | `8e29678` |
| Measured attention layouts | `f3ec333` | `d6f578b` | `95a9cce` |
| Wide-head gradients and fused split-K | `7bc6312` | `d6f578b` | `95a9cce` |
| Repeated-extraction fix | `f74c53b` | `d6f578b` | `a4fe580` |
| Balanced axes and split K-stage search | `212b496` | `d6f578b` | `a4fe580` |

Later commits may add docs, lint fixes or dependency pins. Record the actual
revisions and environment with each run. Historical checkpoints need their
matching Blade API, not a mixture of old square-tile callers and new shape lists.
The early Inferena revision includes a readback experiment that was subsequently
reverted; its inference shader ablations do not establish a preparation win.

## Build once, then freeze the executable

```sh
cd inferena
INFERENA_MEGANEURA_PATH="$GAP_ROOT/source" INFERENA_DRY_RUN=1 \
  CARGO_BUILD_JOBS=1 bash frameworks/meganeura/run.sh SmolVLA
cd ..
cp inferena/target/release/inferena-meganeura native-control
python3 compile_diagnostic.py attention_check.rs attention-check
python3 compile_diagnostic.py rectangular.rs rectangular
python3 compile_diagnostic.py capabilities.rs capabilities
```

For a local Blade source override, use Cargo's
`patch."https://github.com/kvark/blade".blade-graphics.path` configuration.
The current engine pin already identifies the matching tested Blade revision.
`compile_diagnostic.py` follows the runner's Cargo fingerprints rather than
guessing an rlib hash. `CARGO_TARGET_DIR` can point it at a shared build cache.
The runner wrapper itself expects its executable under `inferena/target`.

Bound expensive commands with the existing Inferena helper, for example:

```sh
python3 inferena/scripts/limited.py --memory-mib 4096 --seconds 600 -- \
  python3 probe.py intel-rectangular rectangular intel --nn
```

It uses a user systemd scope on this host. The model runs used 6144 MiB and soft
construction budgets of 60 seconds per phase. A compiler call or qualifier can
finish after that soft budget; the outer scope is the hard process bound.

## Numerical kernel checks and sweeps

```sh
./capabilities
python3 probe.py nvidia-attention attention-check nvidia 128 16
python3 probe.py intel-attention attention-check intel 256 8
python3 probe.py nvidia-wide attention-check nvidia 256 16 --wide
python3 probe.py intel-wide attention-check intel 256 8 --wide
python3 probe.py intel-rectangular rectangular intel --nn
python3 probe.py nvidia-rectangular rectangular nvidia --nn
python3 probe.py intel-splits rectangular intel --split-nn
python3 probe.py nvidia-splits rectangular nvidia --split-nn
python3 matrix_summary.py intel-rectangular nvidia-rectangular intel-splits nvidia-splits
```

The driver labels select Ubuntu's `intel_icd.json` or `nvidia_icd.json`. Adjust
that path on distributions with different ICD filenames. The attention probe
compares every output and Q/K/V gradient against an independent CPU-f64 oracle,
with grouped heads, causal/cross/windowed attention, ragged lengths and several
head dimensions. `--wide` selects head dimension 1024.

The rectangular probe uses exactly f16-representable random inputs, full-output
CPU-f64 checks and guarded output tails. GPU intervals include the final
reduction for split candidates. `wall_us` also includes readback and must not
be compared as pure kernel time. A single ordered sweep is not a paired speedup
confidence interval. `--scalar-nn` explores a separate scalar prototype whose
codegen differs in more than geometry. `--guard` deliberately reproduces Naga's
rejection of subgroup-guarded cooperative loads; it is not a usable candidate.

## Whole models

```sh
MEGANEURA_GPU_CAPTURE=0 python3 native.py nv-control Whisper-tiny 12036 32 native-control --tune
MEGANEURA_GPU_CAPTURE=0 python3 native.py intel-control Whisper-tiny 57868 32 native-control --tune
python3 summarize.py nv-control intel-control
python3 search_summary.py nv-control intel-control
```

Device IDs are specific to these two cards. Omitting `--tune` produces an
untuned physical-layout ablation, not a best-engine comparison. EPT is the
fourth positional argument; `MEGANEURA_FLASH_THREADS`, `MEGANEURA_FLASH_KEYS`
and `MEGANEURA_FLASH_INTERLEAVE` can fix the other layout axes. `--accelerated`
selects Inferena's accelerated-f32 contract. Every destination must be new.

For a full PyTorch/native pair, invoke `launch.py` with the selected venv's
Python, for example `launch.py pair SmolVLA cuda strict --local`. The backend
argument is `cuda` or `xpu`. Do not edit engine or harness sources while this
command runs: it builds the native runner both before and after PyTorch.
Check the outer comparison as well as each process exit code. The PyTorch
SmolVLA self-attention fix is in Inferena `6ff038f`; it is required for matching
semantics. The published cohort branch is not changed by these scratch runs.

## Native profiling and CPU probes

Set `GAP_NSYS` to the installed Nsight Systems executable and add `--nsys` to
`launch.py` for an NVTX-windowed inference trace. Keep profiling separate from
the held-out untraced model samples.

Set `GAP_NGFX` to Nsight Graphics' `ngfx` executable, then run
`graphics.py trace Whisper-tiny /absolute/path/to/native-control 32`.
The Inferena experiment adapter triggers capture after warmup. The capture is
bounded to three submissions and 200 ms, with clocks unchanged. Hardware
counters and PC samples are not latency fractions. Nsight may need its normal
driver permissions; do not change system policy inside the experiment.

`graph_cost.rs` times graph construction phases without a GPU. Pin it to the
same physical CPU for both arms and run it separately from GPU work.
`egglog_template.rs` compares rebuilding and cloning initialized rule databases;
`optimizer_idempotence.rs` checks repeated fused-matmul extraction.
`egglog_parallel.rs` reproduces egglog 2.0's shared table-notification state
between concurrent clones. The default run fails; `--independent` initializes
one template per worker and passes 800 searches. Production caches are per
thread for this reason, not one process-wide rule database.
`qualification_cost.rs` is an unshipped CPU reduction experiment that preserves
finite checks and maximum errors; it does not change the harness contract.

Revisions reproduce the procedure, not the exact original timing noise. Keep
raw outputs outside Git and retain both negative and positive conclusions.
