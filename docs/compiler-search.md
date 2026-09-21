# Compiler search: direction and alternatives

Architecture discussion for PR #200, updated with the PR #206 review on
2026-09-22.
The submitted P3HPC results use their recorded revisions and are unaffected by
this design work.

## Decision

Use egglog to retain equivalent structures and a bounded measurement loop to
choose their implementations. Keep the Rust/WGSL/Blade backend. Replace the
competing selection paths rather than adding another optimizer above them.

The immediate goal is to search the implementations we can already generate.
Generating fundamentally new kernels from lower-level primitives is a separate
project. Neither a larger search space nor equality saturation guarantees a
faster program under a finite compilation budget.

## Construction-time search

For a matrix product followed by an epilogue, keep fused, unfused and split-K
forms available together. Each form has a legal domain of scalar/cooperative
implementations, tile sizes and reduction choices. Do not first select a graph
by estimated memory traffic and only then tune that graph's kernels.

1. Retain equivalent forms using one set of rewrite rules. Shapes, precision
   constraints and effects delimit legal transformations.
2. Extract bounded, diverse implementation families. Leave numerical schedule
   parameters symbolic until needed; do not expand their whole Cartesian product.
3. Lower candidates through the normal compiler, scheduler and allocator.
4. Initialize private representative inputs/state, qualify, then measure.
5. Return the qualified incumbent when the budget ends. Application state must
   not have advanced during search.

These are distinct responsibilities, not competing optimizers. Cheap and
thorough construction should differ in budget, not duplicate transformation
implementations. Identical lowered programs and repeated implementation classes
can reuse work. Semantic equivalence alone is not performance equivalence.

Structural changes belong before allocation. A live session should not need
buffer growth, alias-map edits, dispatch insertion or profiling-index remapping
to install a tuned structure. Rebuilding explicitly is an acceptable tradeoff.
Existing layout-preserving kernel probes remain useful shared infrastructure.

Measured construction needs initialized inputs. An ordinary build must not
silently execute an uninitialized model. Calibrated construction makes its input
and state contract explicit; ordinary construction still uses the same rewrite
and lowering machinery. Synthetic isolated-kernel probes are not a substitute
for whole-program qualification on representative inputs.

## Bounds, correctness and observability

- Budget compilation, allocation, initialization, qualification and measurements,
  not just GPU kernel time. Deadlines are soft around in-flight driver calls.
- Preserve a legal incumbent. An incomplete or invalid comparison cannot win.
  Measurement noise and unrepresentative inputs still limit performance claims.
- Retain different physical interfaces, such as layouts, until their consumers
  and conversion costs have been considered. One cheapest logical expression is
  not necessarily the cheapest complete execution.
- Treat outputs, gradients and persistent updates as observable. Reset private
  state between trials; effects cannot be commuted as if they were pure values.
- Numerical tests complement transformation legality. A close result on a sample
  is not a universal equivalence proof, and real-arithmetic identities need not
  preserve a floating-point policy.
- Use compact choice/measurement reports, including skipped and unexplored work.
  Do not require serialization of runtime dispatch objects for search itself.
- Confirm selections end to end. Isolated timings omit interactions with memory,
  synchronization and CPU submission. Do not infer barrier cost by subtraction.

Start with dense projection families and cached attention, including shared
subgraphs and stateful validation on NVIDIA and Intel. Assess construction time,
memory, execution time and deleted machinery. Broad invariant/regression tests
and inspectable plans are preferable to multiplying timing-sensitive tests.

## API and current boundary

`train::build_measured(graph, config, options, initialize, qualify)` is the
calibrated entry point. `BuildSearchOptions` bounds graph forms, complete plans,
time and declared plan/state bytes. Its `tuning` field configures the existing
private kernel probes and paired noise guard. `BuildSearchReport` records the
extracted expressions, trial descriptions, phase times, rejections, selected
trial and unfinished work. The ordinary `build` API remains input-free.

The initializer supplies representative inputs and weights to a new private
session. The qualifier reads all observable outputs, gradients and state updates
after exactly one step and checks the application's numerical contract. The
runner resets persistent writes between steps and before returning the winner.
Configure runtime optimizers and external/shared writable bindings afterward.
An invalid challenger is discarded; an invalid incumbent aborts the search.
An initializer may share compatible immutable weights from the idle incumbent.

Implemented choices include fused and unfused graph forms, scalar matmul
tile, K-stage, unrolling and split-K equalities lowered by the ordinary compiler,
dispatch fusion,
forward-attention layouts, cached-attention splits, low-occupancy convolution
weight-gradient splits, and fresh submission chunk counts. An extracted matmul
schedule is locked, so a later kernel probe cannot replace its tile. Unlocked
dispatches still use those probes before whole-plan comparison. Attention
splits and convolution weight splits use the existing compiler, with no
session-buffer patching. There is no reusable command recording or separate
live structural tuner.

The collection harness gives private probes the remaining shared session
deadline, not an independent two-second slice. The shorter slices repeatedly
interrupted expensive qualification on B570 SmolLM2 and left later, repeated
matrix classes untuned. Completing the ordinary candidate's kernel search
first recovered a faster incumbent without adding another optimizer. This is
budget allocation, not a monotonic-performance guarantee: an exhausted total
budget or a change in execution state can still defeat that expectation.
Held-out inference and minimal-shape measurements must check the selected plan;
training improvements do not compensate for inference regressions.
The next-cohort policy is not yet released: B570 qualification exposed a
post-tuning full-gradient rejection on the first StableDiffusion training
program. Private kernel checks are not a whole-model guarantee. Preserving
the pre-tuning implementation and revalidating it is needed before that
policy can be qualified across platforms.

This is a bounded first implementation, not exhaustive graph scheduling. Small
graphs are searched together. Larger pure graphs can expose all operators with
rewrite rules, with other operators as opaque cut edges, within the same node
bound. This includes independent projections outside a repeated body. If that
region is too large or crosses precision domains, search falls back to one
verified repeated region. Sparse search does not cross mutations. Attention
layouts and submission chunks are still uniform across a plan. A matmul tile
is whatever enode extraction kept for that site. The report must be read with
those limits.
The matrix catalog covers ordinary and transposed products, with or without an
addend, and the wide projections inside packed SwiGLU/GeGLU. A scheduled product
passes its schedule to the matrix products generated by differentiation; lowering
checks their new dimensions. These are coupled forward/backward candidates, not
independently optimized gradient schedules. An automatic implementation in the
catalog retains ordinary lowering and private kernel probes on alternative graph
forms too, including native-f32 cooperative choices when legal. The explicit
matrix schedules currently describe scalar kernels; they do not jointly search
cooperative tile dimensions. A bounded search does not retain every schedule the
backend can generate.

### Cost and enumeration

`TensorTraffic` counts logical tensor bytes read and written. It is not an HBM
model: it omits caches, arithmetic, occupancy, register pressure and launch cost.
`AstSize` is available as an alternative ordering for ablations. Neither metric
selects a measured winner.

A tiled product is charged the same tensor bytes as its untiled equivalent.
Assigning tiles a token cost while charging other operations bytes biases the
logical search, even if the retained programs are timed later. Implementation
alternatives win only a tie in the logical estimate. Family exclusions retain
unfused alternatives before walking individual tile choices. Structural and
schedule exclusions alternate: many fusion sites must not fill the frontier with
one tile. Tile variants count toward `max_graphs` (sixteen by default, including the
ordinary form). Split tile widths and K staging are visited early; ordinary
construction already probes single-pass kernels. This is coverage ordering, not
a claimed prediction of latency. Graph rank and physical rank are traversed
diagonally, so a larger frontier cannot postpone every submission alternative
until after all graph baselines. Time and program bounds are unchanged.

The final decision uses qualified, paired whole-step wall times, including fresh
recording and submission. Per-pass GPU times are diagnostic, not substitutes for
that deployment objective. Warmup requires both the requested pair count and a
minimum duration (250 ms by default), within the existing total deadline. Two
pairs were not representative of sustained SmolVLA execution: increasing warmup
to 64 pairs changed the NVIDIA inference selection and held-out time from 3.64
to 2.60 ms with identical candidates and decision thresholds. This identifies a
measurement sensitivity, not its exact hardware cause. The time floor avoids
charging 64 pairs to every expensive workload. It can reduce the number of plans
visited; the report records the policy and truncated work.

A learned empirical extraction cost is not implemented yet. A useful next step
would reuse isolated measurements by implementation class
to order the frontier, retaining the whole-program comparison as the final check.
Adding uncalibrated FLOP or bandwidth coefficients would not establish that model.

Broader coverage has a CPU cost. On the i5-12400F, four-form warm extraction at
`1f8d58f` takes 6.66 / 69.04 / 63.46 ms for SmolLM2-135M / SmolVLA / Whisper-tiny
with tensor traffic, versus 6.54 / 67.93 / 56.67 ms with AST size. These are medians
of five calls after one discarded call, pinned to CPU 0, without GPU work.
SmolLM2 exceeds the sparse-region bound and uses outlining. The other two cover
more operators than the earlier 6–8 ms repeated-region search. Sixteen-form
SmolVLA preparation, including graph lowering, takes about 0.2–0.3 s in the native
runner. Session construction, initialization and full qualification dominate its
remaining preparation time.

The PR #206 review caught coverage losses: fusion sites crowded out tile
alternatives, packed projections hid their matrix schedule, outlining excluded
independent projections, and differentiation dropped split-K choices.
It also fixed a lost scalar epilogue and a split-K pipeline key that omitted
weight format. Counted and unrolled loops are both
candidates; unrolling is not a new unconditional default.

### PR #206 model check

SmolVLA strict-f32 checks compare main `da30316` with `1f8d58f`. The native
adapter is Inferena `a4fe580`, with its capability-metadata fields updated for
the newer Blade API. Each session gets 60 seconds and 64 complete-plan trials.
All model outputs and parameter gradients pass the unchanged construction-time
qualification. These runs do not refresh the paired PyTorch measurements.

The GPUs run serially, with no builds or clock changes during measurements.
The held-out medians use 20 samples after five warmup steps and at least two
seconds of warmup. GPU profiles are collected separately. Preparation includes
all three sessions, not just shader compilation. These are individual engineering
runs, not a new cohort or confidence intervals.

| GPU / revision | Inference (ms) | Minimum (ms) | Training (ms) | Preparation (s) |
| --- | ---: | ---: | ---: | ---: |
| RTX 5070 / main | 2.810 | 1.357 | 9.640 | 107.9 |
| RTX 5070 / reviewed | 2.606 | 1.148 | 8.635 | 143.5 |
| Arc B570 / main | 6.532 | 2.612 | 26.356 | 188.0 |
| Arc B570 / reviewed | 5.355 | 2.758 | 18.494 | 183.5 |

This is not a uniform win: Intel's minimum workload regresses by 5.6%. Its
training search finishes only six trials, versus 21 on NVIDIA; inference and
minimum finish 27 and 29. The bounded search still needs better coverage and
selection across workloads before claiming monotonic improvement.

With AST-size ordering on the same reviewed NVIDIA binary, inference / minimum /
training are 2.603 / 1.185 / 8.668 ms, with 143.8 s preparation. Inference and
training select the same complete-plan descriptions as tensor traffic. One model
does not validate either estimate as a latency predictor, and these differences
do not justify a new cost formula. Calibrated kernel-class timings are a more
useful next experiment than adding guessed bandwidth or FLOP coefficients.

The opt-in `search_study` example checks all outputs against an f64 reference.
Run it with `MEGANEURA_EGRAPH_COST=ast-size` or `tensor-traffic` to compare ordering.
These are kernel diagnostics, not held-out model results or a publication cohort.
There is no persistent measured-plan cache yet. Rectangular cooperative kernels
remain experiments. The [September 21 investigation](gpu-gap-2026-09.md) records
the earlier search limits, costing fixes and measured kernel changes. The broader
alternatives below remain research directions, not additional dependencies or
hidden search paths.

## Refactor checkpoint

Relative to PR #200 at `789bcb6`, this removes roughly 370 lines of non-test
compiler/runtime/shader source. The separate greedy rewrite implementation and
the live attention/submission tuner modules are gone. Kernel generation and
pipeline preparation remain shared with ordinary execution.

This is not a large total-line-count reduction: the documentation, benchmark
adapter and broad invariant tests more than offset that saving. The main benefit
is removing live allocation/alias/dispatch repair, not reducing every subsystem
to fewer lines. Rebuilding candidate sessions also costs more than private
isolated probes; construction time must be reported alongside execution time.

Precision domains are opaque cut edges for rewriting. Preserving them also keeps
forward/backward shared-output associations intact, including cross-entropy's
logits gradient. The existing end-to-end gradient tests caught this integration
issue when egglog became the default; no numerical tolerance was relaxed.

## Why not retain greedy rewriting indefinitely?

The historical small, mostly locally profitable rule set did not demonstrate an
execution advantage for equality saturation. The recorded SmolLM ablation used
0.089 ms for greedy rewriting, 2.94 ms for outlined egglog and 56.2 ms for
whole-graph inference saturation; whole differentiated-graph saturation took
7.43 s. These are historical observations, not measurements of the proposed
joint search. See the ablation in `paper/main.tex`.

Equality saturation preserves represented alternatives, but resource limits can
prevent discovering a derivation. Extraction optimizes a supplied cost model,
not actual execution time automatically. Tree extraction can also overcharge
shared work in a DAG. Keeping two separately implemented rule sets adds drift
without resolving any of these problems.

## CPU overhead and review follow-up

The September 20 review found avoidable work in our integration, not just in
egglog. Ordinary extraction rebuilt the same cost table for every escaping
root. It now computes one extractor and shares one term DAG per saturated
segment. Node bindings use direct e-graph lookup rather than evaluating new
expressions. The segment report is passed as one object instead of eleven
independent arguments.

Alternative extraction uses an explicit postorder stack, term-ID memoization
and directly interned integer literals. It no longer creates an AST to recover
literal values. Exclusion lookup borrows function names and argument slices,
without allocating an edge on every cost query. The queue, visited set and
extractor share immutable exclusion sets. Candidates are deduplicated by term
ID in a shared DAG; expression strings are generated only for retained reports.
Egglog's public terms and function lookup still use constructor names; replacing
those with a second local operator registry would add another mapping to maintain.

CPU-only release measurements on zork's i5-12400F, pinned to physical CPU 0,
compare `dd743f4` with this follow-up. Each cell is the median of five warm
invocations after one discarded invocation, in a fresh process per model/arm.
No GPU context, shader compilation or GPU tuning is involved. Training totals
include automatic differentiation and the ablation harness's graph copies;
they are not complete deployment preparation times.

| Workload | Inference before / after (ms) | Training before / after (ms) |
|---|---:|---:|
| SmolLM2-135M | 4.92 / 4.31 | 734.19 / 91.85 |
| SmolVLA | 6.61 / 5.98 | 387.24 / 51.54 |
| StableDiffusion | 28.05 / 17.65 | 521.42 / 88.39 |
| ResNet-50 | 11.61 / 10.27 | 541.29 / 450.10 |
| Whisper-tiny | 6.01 / 5.49 | 175.81 / 125.51 |

The bounded four-form repeated-region search separately takes 4.19 / 3.57 ms
for SmolLM2, 4.24 / 3.59 ms for SmolVLA, and 2.54 / 2.27 ms for Whisper.
Whisper has one represented candidate here; the other two hit the four-form
bound. This measures enumeration, not the subsequent physical-program search.
The same small timing harness was applied to the baseline. Ordinary graph
node/fusion counts, e-graph sizes and extraction-failure counts match in all
ten cases. Peak process RSS does not increase (8.5--325.3 MiB after the change).

Reproduce with `cargo run --release --features models --example
optimizer_ablation -- --model SmolVLA --phase training --repeats 6`, discarding
the first returned sample. The opt-in CPU search probe is `cargo test --release
--features models --lib cpu_search_overhead -- --ignored --nocapture`.
Neither adds a default CI timing test or another test executable.

The remaining ResNet/Whisper preparation cost is mostly outside the reported
egglog/stamping intervals, which total about 40/22 ms in the final training
samples. Differentiation and graph-copy costs need separate attribution before
changing egglog again. Linux sampling was unavailable (`perf_event_paranoid=4`);
these are elapsed-time experiments, not a claim about sampled CPU hotspots.

## September 20 integration gate (historical)

Do not start a distributed cohort merely by merging this PR. Inferena at
`fa5a04e1` still calls ordinary `build` and then `tune_with`, before model inputs
and weights are initialized. A pin update alone would miss joint structural
search. Its `hub` feature and direct `Dispatch.use_coop` field access also need
the current API spellings.

First adapt the runner to initialized, qualified `build_measured` sessions;
retain graph/plan coverage and skipped-region receipts, explicit total budgets,
and peak-memory bounds for two candidate sessions. Then run all five models in
both contracts on the local NVIDIA and Intel devices, check outputs/gradients,
and compare held-out latency and preparation costs before freezing one pin.
The previous GGUF check still has a small NVIDIA regression, variable Intel
decode timing and higher preparation cost; these CPU improvements do not
establish a GPU speedup or resolve calibration stability.

Platforms without a usable PyTorch GPU path belong in the separate
[qualification workflow](../paper/p3hpc/QUALIFICATION.md), never a CPU/GPU speed
comparison. RPL-U has retained qualification evidence; Mendocino is pending.

## Alternatives for a later session

The judgments below concern fit for Meganeura, not a ranking of published
speedups across different workloads, devices, precision policies and budgets.

### Halide

Separates computation from scheduling: tiling, producer placement, storage,
recomputation and parallelism. Its GPU autoscheduler uses hierarchical sampling
and memoization. This is a strong route toward composing kernels instead of
writing another fused template. It does not alone supply all alternative
algebraic formulations. Adoption would require translating our operations and
integrating another compiler with our context/buffer ownership.

Borrow scheduling concepts if we broaden code generation. Do not adopt a new
compiler solely to simplify selection among existing WGSL kernels. Adams2019
is CPU-only; Anderson2021 is the relevant full-GPU autoscheduler.

- [GPU autoscheduler paper](https://arxiv.org/abs/2012.07145)
- [Official autoscheduler integration](https://halide-lang.org/docs/HalideCMakePackage.html)

### Ansor

Generates coarse implementation sketches, samples complete configurations and
improves them with evolutionary search and a learned cost model. It allocates
tuning effort across subgraphs. This helps separate structural and numerical
choices without a complete hand-written template per operator.

The original pipeline partitions graphs before scheduling; broader joint graph
optimization was left as future work. Learned models and evolutionary search
also have cold-start costs, and hardware intrinsics still need derivation rules.
Borrow sketch generation and budget allocation, not necessarily the TVM stack.

- [Ansor, OSDI 2020](https://www.usenix.org/system/files/osdi20-zheng.pdf)

### MetaSchedule

TVM separates schedule rules, candidate generation, search strategies,
builders/runners and a tuning database. This is a useful engineering reference
for reusable legality and measurement, and for compact reproducible decisions.
It cannot recover graph alternatives discarded before workload extraction.

Direct adoption needs TVM integration; recreating all its extension points in
Rust could itself become bloat. Borrow the separation of responsibilities and
decision records. Initially use one concrete policy, not a plugin framework.

- [MetaSchedule architecture](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)

### Cascades

Maintains memoized groups of equivalent expressions and interleaves logical and
physical optimization on demand under required physical properties. This maps
naturally to implementation families and layout-sensitive consumers: retain the
best alternative per required property rather than one universal winner.

It is the strongest alternative to egglog for our immediate scope. However,
implementing matching, memoization and search ourselves risks rebuilding similar
infrastructure. GPU costs are not cleanly compositional, and adding context for
sharing, allocation and neighboring work can enlarge the search state. Revisit
if bounded egglog matching/extraction remains the dominant problem.

- [Cascades framework, 1995](https://15721.courses.cs.cmu.edu/spring2019/papers/22-optimizer1/graefe-ieee1995.pdf)

### Mirage

Represents programs across kernels, thread blocks and threads, searching both
algebraic and scheduling transformations, including new kernel structures. Its
restricted-domain probabilistic equivalence checking is supplemented with
floating-point tests. This can expose implementations absent from our templates.

Adopting its scope means substantially more compiler and verification work.
The published NVIDIA evaluation does not establish transfer to our Vulkan/Metal
backend. Its mathematical guarantees do not cover arbitrary stateful training or
all floating-point behavior. Use it as an offline research reference first.

- [Mirage, OSDI 2025](https://www.usenix.org/system/files/osdi25-wu-mengdi.pdf)

### EquiForge

This September 11, 2026 preprint puts tensor expressions, tiled computation,
reductions and storage alternatives into one e-graph. It extracts implementation
families, prunes redundant partial candidates and progressively tunes schedules.
It is the closest research blueprint to retaining structure through measurement.

Its evaluation allows four hours per configuration, uses Triton and tests NVIDIA
A100/RTX 5090. Algebraic rules assume real arithmetic; numerical validation is
separate. Neither short startup nor our precision policy is solved by adopting
the design. Borrow representation and deduplication ideas selectively.

- [EquiForge preprint](https://arxiv.org/html/2609.12330v1)

### Supporting references

- [Egglog ruleset scheduling](https://egraphs-good.github.io/egglog-tutorial/04-scheduling.html)
- [Egglog extraction and cost models](https://egraphs-good.github.io/egglog-tutorial/05-cost-model-and-extraction.html)
- [TENSAT: shared-DAG extraction](https://proceedings.mlsys.org/paper_files/paper/2021/file/cc427d934a7f6c0663e5923f49eba531-Paper.pdf)
- [Guided equality saturation](https://arxiv.org/abs/2111.13040)
