# Kernel gap studies

Source-only experiments, 20 September 2026; no production defaults or paper
protocol changed. [egglog_search.md](egglog_search.md) records model/reference
pins, machine details, validation and the joint graph-search results.

## What profiling identifies

Validated NVIDIA captures of the unchanged Inferena models show different
bottlenecks. SmolVLA's greedy action expert spends about 91% of its instrumented
pass intervals in matrix products. Attention is below 4%. The PyTorch capture
uses many split-K CUTLASS GEMMs and combine kernels. This motivates keeping
unfused/split implementations in the search, not just changing an attention
kernel or forcing every product onto cooperative matrices.

Whisper's encoder spends about 53% in attention and 33% in matrix products;
convolution is only about 4%. The useful first target is its attention layout.
These are instrumented attribution percentages, not an additive explanation
of wall latency or barrier cost.

Source leads:

- `vla.cpp` at `57a21c01383ae4322d9f7f81cf17a54cd1da31f7` hoists fixed
  context K/V across denoising and uses F16 K/V in some paths. Its complete VLA
  policy is not the same workload as Inferena's action expert.
- `whisper.cpp` at `5670d5c0bbcb148feabef84400a07cfca9aa3b30` uses
  Conv1d im2col plus matrix products and tiled Vulkan attention. Its native
  cooperative-matrix paths are not equivalent to Naga's available interface.
  No end-to-end vla.cpp/whisper.cpp performance comparison is claimed here.

## Whisper: independent query tiles

At `17cebb9338b23d73a7fdb68461f5d4cf1ddeb050`, a 64-thread workgroup gives
each fixed 32-lane subgroup 16 different query rows, instead of processing
the same 16 rows. Both share K/V staging. QK uses F16 cooperative inputs and
F32 accumulation; softmax and PV remain scalar. The control is
`5b4c97377b5d24378322bd8d7f42ee6c34da920f`.

Three fresh-process pairs, reversing the middle pair, give Whisper encoder
medians 4.923/4.931/4.960 ms before and 3.977/3.999/3.997 ms after:
about 19% less whole-model time. Five warmups and 30 samples per process,
fresh recording/submission/waiting, complete CPU checks before and after.
Relative L2 is unchanged at 8.4823e-5.

Separate attention pass intervals fall from 2.360 to 1.413 ms; matrix intervals
stay at 1.437 ms. Registers rise from 122 to 146, and shared storage from
7168 to 10240 bytes. Register count alone would predict this poorly.
The driver's implausible per-thread local-memory statistic is not evidence
of spills and is not used.

Revision `ec7929f94cb4be4ec67f10046a9680a7ff745829` restores the old default
and exposes 1/2/4 query tiles as whole-program candidates. Blade
`2b328f8b643798813d8c9319b807030215d33b98` reports a guaranteed fixed
compute subgroup width only when Vulkan minimum and maximum agree.
Unknown/variable widths and other backends conservatively report no guarantee.
Candidate shared storage stays within Vulkan's minimum 16-KiB guarantee.
This is a capability check, not a vendor-name rule or an assumed default width.

```sh
target/release/examples/egglog_model_search \
  Whisper-tiny /path/to/whisper-cpu.f32 fast baseline \
  --static --attention-tiles --confirm
# Repeat with --reverse. --static disables inner matmul tuning only.
```

| Search order | Query tiles selected | Control | Selected | Paired reduction |
| --- | ---: | ---: | ---: | ---: |
| Forward | 2 | 4.897 ms | 3.991 ms | 18.6% |
| Reverse | 4 | 4.892 ms | 3.944 ms | 19.4% |
| Forward | 2 | 4.886 ms | 3.985 ms | 18.5% |

These final repeats use `95d55651f411ae68e2b19a59d1082562d5c4a2bf`, including
the store-ownership correction below in both the control and candidates.
All three win 39/40 held-out pairs. Search costs 2.7–3.9 seconds with warm
caches and four candidates, not a cold compiler comparison. The 2% guard
does not reliably distinguish two from four tiles; neither becomes a
hard-coded universal choice.

The existing broad F64 oracle covers ragged Q/KV lengths, different query/KV
head counts, full, causal and windowed masks; existing Q/K/V gradient checks
also pass. It caught an initial causal bound still assuming 16 rows.
B570 excludes these fixed-width candidates and passes its existing fallback.

## SmolVLA: matrix split-K and reduction cost

The generic serial SumRows candidate at
`ad00f74db4afe9bf6694dc0c983e636d2f0c3832` assigns one lane to a column and
serially sums its rows. Adjacent lanes read adjacent columns; no workgroup
storage or barriers are needed. Widths 64/128/256 are measured alternatives,
not shape/device thresholds.

In the earlier isolated split-K ablation, it reduces whole-model time by
6.2–7.5% on NVIDIA and 7.6–7.9% on Intel, with full CPU errors below 5.6e-6.
These are not percentages to add to the graph-search gains. The combined
45-plan study in `egglog_search.md` measures their interaction directly.

Fresh Nsight Systems 2026.4.1 captures at
`52a9f25545cdd0ba0aedf9c74a34af22d62dfcd1` compare the greedy program,
the original-graph 99-product scalar split-K program, and the same program
with serial reductions. All complete and pass full-output validation.
These captures predate the store-ownership correction described below;
they identify the kernel costs, not final production performance.

Despite requesting individual Vulkan workload tracing, this driver reports
GPU work at `vkQueueSubmit` granularity. Match each workload's correlation ID
to its host submit and preceding command-buffer begin. Identify the 44 full
steps by their pipeline-bind count (154 or 303): two qualification, five warmup,
30 ordinary timed, one output check, five pass-profile samples and a final check.
Use only ordinary steps 7–36 for this table.

| Plan | Dispatches | GPU submission span | Host begin-to-submit interval | Submit API call |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 154 | 3.429 ms | 1.467 ms | 0.050 ms |
| Scalar split-K | 303 | 2.349 ms | 1.763 ms | 0.041 ms |
| Split-K + serial reduction | 303 | 2.112 ms | 1.473 ms | 0.036 ms |

The selected program reduces the observed GPU span substantially, but still
has substantial host recording work. Host intervals are not CPU-busy time.
They and GPU spans are diagnostic, not independently additive causal costs.

Separate pass-instrumented samples put matrix work at 3.647/2.121/2.123 ms
and normalization/reduction at 0.131/0.754/0.561 ms. This isolates the reduction
change: matrix and attention intervals stay almost unchanged between the two
split plans. Instrumentation raises the selected GPU interval total to 3.054 ms,
versus the ordinary submission span of 2.112 ms. Do not subtract these quantities
or interpret wall time minus a sum of pass times as barrier overhead.
CPU stack sampling is unavailable under the current perf policy; no system
settings were changed.

For these exact candidate IDs at the recorded revision:

```sh
# Replace PROGRAM with 0, 11 or 12. Run one process at a time.
env -u LD_PRELOAD VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  nsys profile --trace=vulkan,nvtx --sample=none --cpuctxsw=none \
  --wait=primary --vulkan-gpu-workload=individual --output=UNIQUE_PREFIX \
  target/release/examples/egglog_model_search \
  SmolVLA /path/to/smolvla-cpu.f32 fast \
  --serial-sums --program=PROGRAM --static --profile
```

## Trials not promoted to production

These are qualified negative or small-effect results, not missing benchmark
rows. Source branches preserve them without committing raw data or binaries.

| Trial / revision | Result |
| --- | --- |
| Cooperative QK and PV, 32 keys (`650f9337d80e3709a6f05000e2e441bdc403d4b8`) | Whisper slows from ~4.97 to ~6.88 ms; probability conversion/staging does not pay off. |
| Remove the minimum-workgroup estimate (`79822865052ec23be6c56d2339af62e7eaf01f07`) | SmolVLA slows from ~4.39 to ~6.73 ms. Blanket cooperative routing is not a solution. |
| Cooperative split-K (`58e4f2c2637de66c60e00f18a1836d64944f4630`) | All 27 plans qualify; both search orders retain scalar split-K, ~3.93 ms versus cooperative challengers ~4.06–4.22 ms. |
| Cooperative K stages 16/32/64 (`94a65c1bfb205ade8b84b86832a703486eaa47e8`) | All 45 plans qualify; neither order selects the cooperative family. Deeper staging is slower here. |
| One 32-thread matrix subgroup (`bb16789dbc2fad074a530489d597c0abbae75456`) | Three pairs: Whisper ~4.97→5.45 ms, SmolVLA ~4.45→5.15 ms. |
| Two independent matrix row tiles (`6174fbadf77a17463acb27409705dbdf7ee12459`) | Three pairs: only ~1–3% less model time; full CPU errors unchanged. |

The last two use fixed-32 ablation branches, not capability-independent defaults.
The cooperative split-K oracle checks full F64 products across normal/AT/BT
directions, ragged edges and uneven K partitions. The independent-row trial
also passes the 26 existing skinny-matrix checks with F16 explicitly enabled.
Naga 30 emits subgroup-scoped cooperative matrices, while the
old 64-thread generator addresses the same output tiles from each subgroup.
Passing numerical checks is not proof that overlapping stores are race-free.
The [Vulkan memory model](https://docs.vulkan.org/spec/latest/appendices/memorymodel.html#memory-model-data-race)
does not exempt equal-value writes, and the
[cooperative-matrix scope contract](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_cooperative_matrix.html)
is per scope instance. This was found from source and the specification, not
from an observed output divergence in these runs.

The follow-up at `95d55651f411ae68e2b19a59d1082562d5c4a2bf` guards only the
stores of duplicated tiles with `subgroup_id == 0`. Staging, matrix arithmetic
and workgroup barriers stay unchanged. Independent-query candidates keep
their separate output tiles. This does not assume a 32-lane subgroup or add
another tuning parameter. Unlike subgroup-conditional matrix arithmetic,
the store-only guard passes Naga validation without disabling checks.

The production form is `789bcb60e82d8a2993a440acb500bf53b797cbeb` in PR #200:
matrix, convolution and attention forward/gradient generators, including
horizontal matrix wrappers. All 31 existing code-generation checks and
all-target/all-feature Clippy pass. NVIDIA passes the existing 26 skinny-matrix,
four convolution, full attention oracle, 11 convolution-derivative and Q/K/V
gradient checks. Intel fallback checks pass; its two explicitly cooperative-only
convolution tests stop at their existing capability assertion, not a numerical
failure. No new test executable or relaxed tolerance is needed.
[Production CI](https://github.com/kvark/meganeura/actions/runs/35515431629)
passes all six jobs, including Metal. Rust host line coverage is 84.07%; WGSL
execution is not instrumented and the existing CI backprop/bit-exact skips apply.

Three before/after process pairs leave complete SmolVLA and Whisper CPU errors
unchanged. Stable process medians suggest a small SmolVLA cost (~1%) and a small
Whisper reduction (~2%), with other pairs showing substantial timing drift.
This is a correctness fix, not a claimed speedup. The main model-search
comparison is repeated with corrected stores in both control and candidates.

All studies use sequential GPUs/builds, bounded host memory and unchanged
numerical gates. Raw traces, JSON and frozen binaries stay outside Git.
