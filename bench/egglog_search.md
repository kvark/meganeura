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

The initial whole-model helper starts from the greedy graph and recovers both
sides of the matmul/add equality. This does not recover every alternative an
earlier rewrite might discard. The extraction API itself accepts the original
graph; the final section tests that path separately. A production search should
retain the original equivalence space, with the greedy plan as a cheap incumbent,
not depend on reconstructing lost choices through an ever-growing reverse-rule
catalog.

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

At `5b4c97377b5d24378322bd8d7f42ee6c34da920f`, `egglog_model_search` retains the optimized greedy control and
up to eight alternatives for one repeated region. Each crosses dispatch
fusion and plain/split-K lowering. All plans are lowered before allocation;
the selector then holds at most an incumbent and one challenger session.
Identical dispatch-fusion alternatives within the same graph are skipped.
The helper reuses deterministic host weights, but each session has private
GPU allocations and receives a full upload. It skips parameter zeroing only
because initialization fills every named parameter. Each candidate receives
up to two seconds of ordinary kernel tuning before
paired whole-step selection. The total search has a soft 180-second limit,
a 32-program limit and a 3-GiB declared-plan-byte limit. The latter does not
bound driver allocations, pipeline objects or staging. Runs also used a
4-GiB process cgroup with no swap and a 300-second external timeout. Construction,
initialization, qualification and kernel tuning have separate report fields.

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
# --seconds=20 tests a shorter soft budget; --profile is a separate diagnostic.
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
| RTX 5070 / forward | 4.450 ms | 3.918 ms | 12.0% |
| RTX 5070 / reverse | 4.470 ms | 3.925 ms | 12.1% |
| B570 / forward | 9.857 ms | 6.597 ms | 32.9% |
| B570 / reverse | 9.856 ms | 6.603 ms | 33.2% |

The selected plan won all 40 confirmation pairs in each run. Different
orders on NVIDIA selected different members of the same unfused-plus-split-K
family: 97 versus 89 split products. Intel selected 97 in both orders.
This is bounded exploration, not proof of a
unique optimum. All 18 distinct lowered candidates completed, but logical extraction
was truncated at eight representatives. Search took 32.5–33.6 seconds per
NVIDIA run and 58.2–58.3 seconds per Intel run, including construction and qualification,
but excluding earlier graph extraction/lowering and the final confirmation.
Graph extraction itself took 7–9 ms. Do not label session-construction time
as shader compilation time.

With a 20-second soft budget, search stops with explicit truncation:

| GPU / search order | Candidates visited | Median paired reduction |
| --- | ---: | ---: |
| RTX 5070 / forward | 11 | 12.2% |
| RTX 5070 / reverse | 11 | 8.8% |
| B570 / forward | 7 | 33.2% |
| B570 / reverse | 6 | 28.2% |

Every selected plan again wins all 40 held-out pairs and passes the full CPU
reference. The reverse searches retain an 81-product split family instead of
reaching the better alternatives. Actual selector duration is 20.02–20.19 s;
this is a soft deadline, not preemption of in-flight driver/validation work.

Earlier `0c0e2fa` runs found the same approximate 12%/33% gains but rebuilt
27 candidates and regenerated weights each time (74/135 s). The table above
uses a fresh isolated Cargo build and the deduplicated selector. Independent
worktrees must not share Meganeura build metadata. Clocks remain unlocked;
both arms can drift within a run, so retain paired samples, not just minima.
These budgets are not a like-for-like compiler comparison with PyTorch.

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

## Extraction before greedy rewriting

Revision `d8d5876079ded4ed3585283e7c35f367bb3078ae` adds `--original-graph`.
Only the control is greedily optimized; the other representations are extracted
from a repeated region in the original model graph. The source API already
supports this. No final greedy pass is allowed to collapse the alternatives.

With `SmolVLA REFERENCE fast --original-graph --seconds=60 --confirm`, and then
`--reverse`, the NVIDIA controls are 4.4433 / 4.4637 ms and the selected plans
4.1797 / 4.1785 ms. Median paired reductions are 5.9% / 6.5%, winning 39/40
held-out pairs each. B570 (`strict`) gives 9.8401 / 9.7891 ms controls and
7.6115 / 7.6160 ms selected, 22.6% reductions and 40/40 wins. Full-output errors
remain below 5.6e-6. All 18 lowered plans complete, in 36–38 s on NVIDIA and
59.4 s on Intel; logical enumeration is still truncated at eight forms.

This bounded search finds less than the earlier optimized-region experiment:
it reaches 71/78 split products, not 97. Starting earlier is necessary to retain
the general equivalence space, but is not enough to enumerate the best useful
forms under a small bound. Region coverage and exploration order need work.
Keep the greedy incumbent, and consider both logical and lowered alternatives
under one budget. Do not replace a working bounded tuner with an unbounded
whole-model saturation or claim that enumerating eight forms solves extraction.

Revision `df3bce41c87ccd2091aada50df5f742ae4f31c5a` improves that exploration
order. Before excluding individual selected fusion sites, it excludes all
selected sites of the same constructor. This reaches an entirely unfused
family early, instead of spending the eight-form bound on small variations
of the same fused family. It still uses the same egglog rules and extractor.
The existing broad extraction test checks this ordering on two independent
products, including a bound of two representatives.

With the original graph and a 20-second budget:

| GPU / search order | Plans visited | Greedy control | Selected plan | Median paired reduction |
| --- | ---: | ---: | ---: | ---: |
| RTX 5070 / forward | 10 | 4.398 ms | 3.858 ms | 12.4% |
| RTX 5070 / reverse | 10 | 4.426 ms | 4.031 ms | 8.9% |
| B570 / forward | 7 | 9.627 ms | 6.422 ms | 33.4% |
| B570 / reverse | 6 | 9.519 ms | 7.817 ms | 18.8% |

Every selection wins all 40 held-out pairs and passes the same full CPU
reference, with relative L2 below 5.6e-6. Actual selector duration is
20.02–20.03 seconds. Forward order reaches 99 split products; reversed order
deliberately postpones that family and reaches only 78 before the deadline.
The bounded search is therefore still order-sensitive. The useful result is
that the original, pre-greedy graph now reaches the earlier 12%/33% gains
within 20 seconds, not that the global extraction problem is solved.

## Reusing measurements within compilation

Revision `43deac764ac6d225ae08899b62ef66a78ed0b72d` reuses completed
private-scratch kernel searches within one whole-program selection. The key
includes shape, direction, storage precision, binding sizes and placement,
code-generation knobs, and the initial/challenger sequence. The memo belongs
to one device and policy; it is neither a disk cache nor a global cache.
Incomplete, failed, or unqualified searches do not populate it. Reports list
reused choices separately rather than inventing new timing samples. Every
whole program is still initialized, checked and measured independently.

The opt-in broad GPU check covers two equivalent programs, full output and
parameter checks, reuse after a completed search, and no reuse after a
zero-budget search. It passes on both GPUs; Clippy and the existing CPU
extraction checks also pass.

The same original-graph helper, 18 plans, two-second inner tuning budget,
and warm driver caches give these forward/reverse results:

| GPU | Without reuse | With reuse | Kernel comparisons, without / with |
| --- | ---: | ---: | ---: |
| RTX 5070 | 35.8 / 35.5 s | 13.8 / 12.3 s | 921 / 935 versus 93 / 91 |
| B570 | 56.7 / 57.1 s | 32.0 / 32.8 s | 307 / 308 versus 126 / 128 |

The no-reuse control is `df3bce41c87ccd2091aada50df5f742ae4f31c5a`.
NVIDIA completes all 18 plans even with a 20-second bound. Intel visits eight
under that bound; a complete search still needs about 32 seconds. Full CPU
errors stay below 5.6e-6. Selected plans win all 40 held-out confirmation
pairs: about 12.4% / 8.8% reduction on NVIDIA and 33.1% / 32.3% on Intel
for the complete memoized search. Reversing a bounded Intel search gives
23.0% instead of 32.6%. A full non-memoized Intel run varies as far as 37.3%;
unlocked clocks and different tuning decisions remain relevant.

Even the complete NVIDIA search can retain a weaker graph family: one paired
comparison drifts from roughly 3 to 4 ms and does not clear the existing noise
guard. Reuse saves search work; it does not make a short, noisy comparison
conclusive. Longer warmup and joint graph/reduction candidates are tested
separately next, without weakening that guard.
