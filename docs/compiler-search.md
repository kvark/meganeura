# Compiler search: direction and alternatives

Architecture discussion and implementation notes, 2026-09-20, for PR #200.
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

Implemented choices include fused/unfused graph forms, dispatch fusion,
cached-attention splits and fresh submission chunk counts. Existing
layout-preserving kernel choices are tuned before
whole-plan comparison; completed comparisons can be reused within that build.
Attention splits use the existing compiler, with no session-buffer patching.
There is no reusable command recording or separate live structural tuner.

This is a bounded first implementation, not exhaustive graph scheduling. Small
graphs are searched together; large graphs currently explore one verified
repeated region. Stateful or mixed-precision regions remain opaque. Physical
settings are currently applied uniformly to eligible operations within a plan,
not independently to every site. The report must be read with those limits.
There is no persistent measured-plan cache yet. Dense split-K and alternate row
reductions remain prototypes on the experiment branch, not additions to this
refactoring. The broader alternatives below
remain research directions, not additional dependencies or hidden search paths.

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
