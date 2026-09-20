# Cooperative attention layout study

Source-only experiment, 20 September 2026. No change to the submitted paper's
benchmark cohort or production defaults. See `egglog-measured.md` for model,
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

Another ablation (`79822865052ec23be6c56d2339af62e7eaf01f07`) removes only the
minimum-workgroup estimate for F16 cooperative matrix products. SmolVLA slows
from 4.3895 / 4.3924 / 4.3995 ms to 6.7445 / 6.7408 / 6.7034 ms. The full-output
gate and all 26 existing skinny-matrix GPU checks pass. Simply forcing every
small product onto the currently available cooperative implementation is not
the solution. Keep family alternatives for measurement; do not replace this
estimate with a blanket cooperative preference.

## Reproduce

Check out the indicated source revision, using a separate target directory for
each worktree. Generate `whisper-cpu.f32` with `bench/egglog_reference.py` as
described in `egglog-measured.md`, then build and run:

```sh
CARGO_BUILD_JOBS=1 cargo build --release --example egglog_model_search
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  target/release/examples/egglog_model_search \
  Whisper-tiny /path/to/whisper-cpu.f32 fast baseline --static
```

Use `--profile` only for a separate attribution run. The resource envelope used
here is 4 GiB host memory, zero swap, a 120-second process limit, and six physical
CPU cores. Raw JSON, traces, and frozen binaries remain outside Git. These are
diagnostic model studies, not replacements for the frozen P3HPC campaign.
