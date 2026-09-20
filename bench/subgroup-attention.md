# Cooperative attention layout study

Source-only experiment, 20 September 2026. No change to the submitted paper's
benchmark cohort or production defaults. See `egglog_search.md` for model,
CPU-reference, machine, and resource-limit details.

The workload is Inferena's deterministic Whisper-tiny encoder (3000 mel frames),
not full audio transcription. RTX 5070, NVIDIA 595.91.07, accelerated F32 policy:
F16 cooperative QK, F32 accumulation, scalar softmax and PV. Each sample freshly
records, submits and waits. Five warmups, 30 samples per process, three process
pairs with the middle pair reversed. Clocks are not locked. Timed samples exclude
validation/readback; every run checks the complete output against the same CPU
reference. Builds have separate target directories to avoid stale worktree
artifacts. Per-pass profiles are separate runs, not headline timings.

## Independent query tiles win

Baseline source: `5b4c97377b5d24378322bd8d7f42ee6c34da920f`.
Candidate: `17cebb9338b23d73a7fdb68461f5d4cf1ddeb050`.

The original 64-thread workgroup processes 16 query rows. The candidate assigns
16 different query rows to each of its two 32-lane subgroups, for 32 rows per
workgroup. Both subgroups share K/V staging. PV remains scalar. There is no
change to model math, precision, or backward kernels.

| Process pair | Baseline median (ms) | Independent queries (ms) |
| --- | ---: | ---: |
| 1 | 4.9233 | 3.9765 |
| 2 | 4.9308 | 3.9989 |
| 3 | 4.9602 | 3.9969 |

This is about 19% less whole-model time. The four attention passes total
2.360 ms in the baseline profile and 1.413 ms in the candidate profile (40% less).
Matrix passes stay at 1.437 ms. Full-output relative L2 error is unchanged at
8.4823e-5. Driver-reported registers rise from 122 to 146 and workgroup storage
from 7168 to 10240 bytes; fewer registers alone would not predict this result.

The candidate requires a guaranteed 32-lane subgroup, not a guessed/default
width or a vendor-name check. It is not yet a portable default. A production
implementation must check the subgroup contract and measure legal layouts.
The full-F64 oracle covers ragged lengths, mixed query/KV heads, full, causal,
and windowed masks. It caught a causal bound that was still hard-coded to 16
rows; the measured revision fixes it. Existing Q/K/V gradient checks also pass.

## Negative results matter

At `650f9337d80e3709a6f05000e2e441bdc403d4b8`, a 32-thread layout with a
32-key tile computes both QK and PV with cooperative matrices. Converting and
staging probabilities for PV does not pay off here: three candidate medians
are 6.8932 / 6.8778 / 6.8748 ms, versus 4.9684 / 4.9812 / 4.9618 ms for the
control. Attention time rises from 2.360 to 4.276 ms; matrix time is unchanged.
Full-output relative L2 is 8.6848e-5. The unchanged validation gate passes.
This version was reverted on the experiment branch.

## Measured layout selection

Revision `ec7929f94cb4be4ec67f10046a9680a7ff745829` restores the original default
and makes 1/2/4 independent query tiles explicit candidates in the existing
whole-program selector. It uses Blade
`2b328f8b643798813d8c9319b807030215d33b98`, whose new capability reports a fixed
compute subgroup width only when Vulkan's minimum and maximum agree. Metal,
GLES, and unknown/variable-width devices conservatively report no guarantee.
There is no vendor-name gate or assumed default width. Candidate workgroup
storage is capped at Vulkan's minimum 16-KiB guarantee. One broader existing
GPU test checks all legal layouts against F64 across masks, mixed heads, and
ragged sizes; it passes on NVIDIA and takes the existing fallback on Intel.

```sh
target/release/examples/egglog_model_search \
  Whisper-tiny /path/to/whisper-cpu.f32 fast baseline \
  --static --attention-tiles --confirm
# Repeat with --reverse; this reverses challengers, not the control.
```

`--static` disables the inner matmul tuner for this layout-only ablation; it
does not disable whole-program measurement. Three independent processes:

| Search order | Selected query tiles | Control (ms) | Selected (ms) | Paired reduction |
| --- | ---: | ---: | ---: | ---: |
| Forward | 2 | 4.9612 | 4.0291 | 18.8% |
| Reverse | 4 | 4.9665 | 3.9818 | 19.8% |
| Forward | 2 | 4.9544 | 4.0198 | 18.8% |

Each selected layout wins all 40 independent confirmation pairs. The 2% noise
guard does not consistently distinguish two from four tiles, which is fine:
neither is a universal hard-coded choice. Search, construction and qualification
together take 3.367 / 2.050 / 2.000 seconds; warm caches and four candidates,
not a cold compiler comparison. Full-output error remains 8.4823e-5. B570
reports variable subgroup width, excludes the independent-query candidates,
and passes its full-output reference check through the existing implementation.

Another ablation (`79822865052ec23be6c56d2339af62e7eaf01f07`) removes only the
minimum-workgroup estimate for F16 cooperative matrix products. SmolVLA slows
from 4.3895 / 4.3924 / 4.3995 ms to 6.7445 / 6.7408 / 6.7034 ms. The full-output
gate and all 26 existing skinny-matrix GPU checks pass. Simply forcing every
small product onto the currently available cooperative implementation is not
the solution. Keep family alternatives for measurement; do not replace this
estimate with a blanket cooperative preference.

## SmolVLA: split-K and its final reduction

Revision `58e4f2c2637de66c60e00f18a1836d64944f4630` also tests cooperative
split-K products. One fixed 32-lane subgroup computes each partial tile,
then the existing SumRows combines partials. The generator reuses cooperative
staging and the masked epilogue store; unmasked ragged stores would overwrite
the next partial. Precision and subgroup guards remain explicit.

All 27 whole-program candidates qualify in both search orders, but neither
search selects the cooperative family. Scalar split-K gives held-out medians
3.9261 / 3.9248 ms, against greedy controls of 4.4546 / 4.4434 ms.
Cooperative challengers take about 4.06–4.22 ms during selection. Full CPU
relative L2 is 1.96e-4 for the cooperative plan and 5.55e-6 for scalar split-K.
The broad full-F64 check covers normal and transposed products, ragged output
dimensions, and uneven K partitions. This is a qualified negative result,
not a reason to force cooperative execution.

Separate fixed-plan Nsight Systems captures complete successfully for the
greedy, scalar-split and cooperative-split implementations. Companion GPU
pass timings inside those instrumented captures put matrix work at
3.637 / 2.096 / 2.405 ms respectively. Reduction and normalization passes
rise from 0.132 to 0.749 / 0.759 ms. These are diagnostic instrumented
intervals, not clean benchmark latencies or measurements of barrier cost.

Revision `ad00f74db4afe9bf6694dc0c983e636d2f0c3832` addresses that reduction
work with another generic candidate: one lane serially sums the rows for
one output column. Adjacent lanes read adjacent columns; no workgroup
storage or barriers are needed. Workgroup sizes 64, 128 and 256 are measured
as alternatives, without a shape or device threshold. The existing broad
split-K oracle covers every variant on both GPUs.

Run `SmolVLA REFERENCE fast --program=3 --serial-sums --static --confirm`,
then repeat with `--reverse`; use `strict` on B570. Here the confirmation
control is the selected scalar split-K program, not the original greedy
graph. The only change is the row reduction implementation:

| GPU / order | Split-K control | Serial reduction | Median paired reduction |
| --- | ---: | ---: | ---: |
| RTX 5070 / forward | 3.9154 ms | 3.6748 ms | 6.2% |
| RTX 5070 / reverse | 3.8931 ms | 3.6086 ms | 7.5% |
| B570 / forward | 6.5871 ms | 6.0923 ms | 7.6% |
| B570 / reverse | 6.6028 ms | 6.0880 ms | 7.9% |

Held-out wins are 40/40, 36/40, 40/40 and 40/40; full CPU errors remain
below 5.6e-6. Forward order chooses width 64, reverse order 256: the 2% guard
does not establish a material difference between the serial widths. The
four-program searches take 2.8–3.2 s on NVIDIA and 6.0 s on Intel, including
construction and qualification, with inner matrix tuning disabled for this
ablation. Combining these candidates with graph search is the next step;
do not add the separate percentage improvements as if they were independent.

## Reproduce

Check out the indicated source revision, using a separate target directory for
each worktree. Generate `whisper-cpu.f32` with `bench/egglog_reference.py` as
described in `egglog_search.md`, then build and run:

```sh
CARGO_BUILD_JOBS=1 cargo build --release --features models --example egglog_model_search
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  target/release/examples/egglog_model_search \
  Whisper-tiny /path/to/whisper-cpu.f32 fast baseline --static
```

Use `--profile` only for a separate attribution run. The resource envelope used
here is 4 GiB host memory, zero swap, a 120-second process limit, and six physical
CPU cores. Raw JSON, traces, and frozen binaries remain outside Git. These are
diagnostic model studies, not replacements for the frozen P3HPC campaign.
