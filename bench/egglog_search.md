# Measured extraction experiment

This branch keeps graph alternatives until after measuring their kernel
configurations. It reuses Meganeura's egglog rules, extractor and graph stamper.
It is not a production replacement for the post-plan tuner yet.

## What changed

Egglog's `extract_variants` varies only the root constructor and chooses the
cheapest children. The experiment asks the existing extractor for more
representatives by excluding selected e-nodes. This exposes nested choices
without implementing another rewriting engine. A bounded queue controls the
work; `truncated` means alternatives remain unexplored. This is neither a
globally optimal search nor a k-best guarantee.

Several escaping roots are extracted together, so a candidate contains their
shared computation. Small regions can have opaque dependencies outside them;
the existing outliner identifies repeated model regions. Estimated traffic
only orders exploration. It is not a substitute for timing the complete
lowered candidate, including its barriers, temporary storage and kernels.

## Joint graph and kernel probe

```sh
cargo test --lib optimize::search::
cargo build --release --example egglog_search
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json \
  target/release/examples/egglog_search 128 576 1536 forward
# Repeat with reverse, and with nvidia_icd.json.
```

The graph is `Add(MatMul(A,B), C)`. Both fused/unfused representations cross
K stages 8/16/32 and both column layouts: 12 complete candidates, all compiled
before timing. Later graph optimization and dispatch fusion are disabled so
they cannot collapse the alternatives. Precision is native f32; neither test
GPU exposes a native-f32 cooperative tile through this stack.

Each candidate is checked against an independent f64 matrix product over all
output elements. Inputs are bounded, nonzero synthetic dyadic values, not
trained model weights. This example is not comprehensive numerical
qualification. Timing uses fresh ordinary step/submit/wait execution, 20
repetitions per sample, three discarded and nine retained samples. Candidate
order rotates and reverses within a run. No command recording is replayed.

At implementation revision `d508b1a`, B570 results for M=128, K=576, N=1536
(milliseconds, two fresh processes with opposite initial order):

| Representation | Default K=32, plain columns | Best measured configuration |
| --- | ---: | ---: |
| Unfused, forward order | 0.21529 | 0.20415 |
| Fused, forward order | 0.22038 | 0.20041 |
| Unfused, reverse order | 0.21414 | 0.20366 |
| Fused, reverse order | 0.22047 | 0.19999 |

The unfused default wins, but the tuned fused form wins in both orders. Its
final margin is small, about 2%, not evidence of a large whole-model gain.
Both winners use K=16; the best fused column layout differs between runs.
Choosing a graph family from its default timing would discard that winner.
Extraction took 2.1–2.2 ms; all 12 builds together took 135–137 ms with warm
driver caches. These are not cold-start compiler comparisons.

The same probe also ran 50×512×512, 50×720×960 and 1×576×1536 on B570 and RTX
5070. The one-row case lowers to GEMV, so changing the matrix K-stage does not
create distinct kernels there and timing differences are noise, not a win.
Hardware: i5-12400F, affinity 0,2,4,6,8,10, unlocked clocks; NVIDIA 595.91.07,
Intel Mesa 26.0.3 with B570 on the secondary x1 link. Raw artifacts stay outside
Git; the branch retains the code and commands needed to repeat the experiment.

## Model-region survey and remaining work

```sh
cargo build --release --features models --example egglog_model_search
target/release/examples/egglog_model_search SmolVLA
target/release/examples/egglog_model_search Whisper-tiny
```

The CPU-only survey found eight bounded alternatives in a 45-node SmolVLA
region (seven repeated instances) in 6.9 ms. A 32-node Whisper region (three
instances) produced one representative in 4.0 ms. The latter is important:
many useful choices, including generated epilogues, are currently introduced
by lowering and are absent from the logical rewrite rules.

Before production integration:

- Retain legal lowering/fusion families as well as logical rewrites. Tune each
  complete candidate before selecting a representation.
- Preserve the region boundary's actual placement, precision and observable
  outputs. Turning every cut dependency into a host-visible input would time
  a different memory contract.
- Select before final allocation and scheduling. This avoids growing live
  buffers and repairing barrier groups for each operation-specific tuner.
- Reuse qualification, scratch limits, paired sampling and noise guards;
  record exclusions and unfinished searches. Do not replace them with a sum
  of independently measured kernel times.

The small-shape probe alone does not establish a whole-model improvement.
The following model experiment tests that separately. Production search is
still unchanged.

## Split-K as a structural candidate

`egglog_search 50 720 960 forward split` also crosses 32/64 output tiles with
1/2/4/8 K partitions, before final scheduling and allocation. Only the plain
matrix product currently supports partitioning. Its candidate includes SumRows
and the original add, so a fused one-dispatch form competes with complete
multi-dispatch forms, not with a partial kernel time. The implementation reuses
the scalar generator and reduction; it does not duplicate a matmul shader.

This is motivated by a validated Nsight Systems capture of the actual SmolVLA
action-expert workload: PyTorch uses many split-K GEMMs and combine kernels.
The prototype changes neither the benchmark protocol nor production defaults.
The source branch includes an opt-in broad test of normal/AT/BT products,
uneven K partitions, ragged output edges and rejection without plan mutation.
It passes on both RTX 5070 and B570 against full independent f64 products.

## Whole-model search

At `0c0e2fa`, `egglog_model_search` retains the optimized greedy control and
up to eight alternatives for one repeated region. Each crosses dispatch
fusion and plain/split-K lowering. All plans are lowered before allocation;
the selector then holds at most an incumbent and one challenger session.
Each candidate receives up to two seconds of ordinary kernel tuning before
paired whole-step selection. The total search has a soft 180-second limit,
a 32-program limit and a 3-GiB declared-plan-byte limit. The latter does not
bound driver allocations, pipeline objects or staging. Runs also used a
4-GiB process cgroup with no swap and a 300-second external timeout.

This uses the same action expert (50 action tokens, 16 context tokens) and
Whisper encoder (3000 mel frames) as Inferena, not the complete VLA policy or
Whisper decoder. Inputs and named weights use Inferena's fixed deterministic
initialization, not pretrained weights. `egglog_reference.py` imports the
unchanged PyTorch model code from Inferena
`fa5a04e1c1b38405cfa371a27c5dcef1319835d5`. References use CPU f32 and PyTorch
2.13.0, git revision `cf30153c4c131c8164ee7798e5022d810682e2cb`.

```sh
python bench/egglog_reference.py ../inferena SmolVLA /tmp/smolvla.f32
cargo build --release --features models --example egglog_model_search
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  target/release/examples/egglog_model_search \
  SmolVLA /tmp/smolvla.f32 fast --confirm
# Repeat with --reverse. For B570 use intel_icd.json and strict.
```

Every candidate is checked over the full model output before tuning, after
tuning and after selection samples, using the fixed Inferena forward gates:
relative L2 and relative squared-norm error both below 1%. This is inference
qualification, not a training or language-quality result. SmolVLA's measured
errors were below 5.6e-6 relative L2; no candidate was rejected numerically.

`--confirm` builds and tunes a fresh greedy control after selection, warms
both sessions for 30 pairs, then measures 40 alternating-order pairs. These
confirmation samples are not used to pick a plan. RTX 5070 allows reduced
input precision; B570 uses native f32:

| GPU / search order | Greedy control | Selected plan | Median paired reduction |
| --- | ---: | ---: | ---: |
| RTX 5070 / forward | 4.445 ms | 3.903 ms | 12.1% |
| RTX 5070 / reverse | 4.477 ms | 3.935 ms | 12.2% |
| B570 / forward | 9.825 ms | 6.580 ms | 33.5% |
| B570 / reverse | 9.858 ms | 6.604 ms | 33.0% |

The selected plan won all 40 confirmation pairs in each run. Different
orders on NVIDIA selected different members of the same unfused-plus-split-K
family: 97 versus 89 split products. Intel selected 97 in both orders.
This is bounded exploration, not proof of a
unique optimum. All 27 lowered candidates completed, but logical extraction
was truncated at eight representatives. Search took about 74 seconds per
NVIDIA run and 135 seconds per Intel run, including construction and qualification,
but excluding earlier graph extraction/lowering and the final confirmation.
Graph extraction itself took 7–9 ms. Do not label session-construction time
as shader compilation time.

Optimized SmolVLA required an outliner correction: topological sorting hoists
independent K/V projections before the repeated blocks. Those dependencies
must stay opaque and retain their actual bindings. Internal and inter-block
chain edges must still match; declaration-only regions are not useful search
targets. Earlier runs that selected those regions tested only physical
split-K alternatives and are not evidence of model-level logical search.

## Why these candidates

Separate validated Nsight Systems captures at Meganeura `e374a18` showed
many PyTorch split-K GEMMs and combine kernels in SmolVLA. Portable per-pass
profiling attributed about 91% of native inference intervals to matrix
products and less than 4% to attention. The split-K experiment targets that
measured bottleneck. In Whisper, cooperative attention was about 53% and
matrix products 33%; convolutions accounted for only 4%. The same structural
search retained Whisper's control, so this is not a universal split-K policy.

[vla.cpp](https://github.com/VinRobotics/vla.cpp/tree/57a21c01383ae4322d9f7f81cf17a54cd1da31f7)
also hoists fixed context K/V across denoising steps. That is a different
workload/lifetime contract from one Inferena action-expert call and is not
counted as a measured speedup here. Its attention paths can use f16 K/V even
with f32 accumulation.
[whisper.cpp](https://github.com/ggml-org/whisper.cpp/tree/5670d5c0bbcb148feabef84400a07cfca9aa3b30)
provides a second lead: its Vulkan attention tiles both QK and PV, whereas
Meganeura's current cooperative forward kernel accelerates only QK.

Nsight Vulkan intervals in these captures are grouped submissions, not
individual shaders. Portable pass timings came from separate instrumented
runs. Neither host wait time nor wall-minus-summed-GPU-time is a measurement
of CPU busy time or removable barrier cost. No P3HPC protocol or cohort was
changed, and no vla.cpp/whisper.cpp end-to-end speed comparison is claimed.
