# Related-work notes for the Meganeura paper

Status: literature map for drafting, not a finished related-work section.
Prefer the peer-reviewed version of each work in the bibliography when one is
available.

## The closest current comparison

The most directly adjacent recent work is **Llamas on the Web** (Levine et
al., 2026). It implements a WebGPU backend for `llama.cpp`, uses static memory
planning and tunable templated kernels, and evaluates ten language models on
16 devices from eight vendors. It reports competitive or better performance
than vendor-specific `llama.cpp` backends on some devices.

This paper makes it especially important not to pitch Meganeura merely as
"fast ML through a graphics API." The clean distinction is:

- LlamaWeb studies memory-efficient, multi-precision LLM **inference in the
  browser**;
- Meganeura studies a shared graph, autodiff, compiler, kernel, memory, and
  runtime stack for **both training and inference**, across several model
  families, through native Vulkan and Metal;
- Meganeura's primary comparison uses an explicitly strict f32 contract and
  checks backward results, while its accelerated paths are reported
  separately.

Source:
[Llamas on the Web, arXiv:2605.20706](https://arxiv.org/abs/2605.20706).

Two other graphics-API papers sharpen the evaluation:

- **WebLLM** is a compact seven-page systems preprint that puts a graphical
  architecture overview early, states one memorable outcome (“up to 80%” of
  native performance), and keeps the system story focused on three browser
  constraints. It is useful as a presentation model even though its
  evaluation is much narrower than Meganeura's.
- **Characterizing WebGPU Dispatch Overhead** separates API dispatch cost,
  Python/framework overhead, and shader efficiency, reports dtype-matched and
  mismatched comparisons explicitly, and retains inconclusive experiments in
  its analysis. Its primary `cs.LG` category with `cs.DC` and `cs.PF`
  cross-lists is a strong precedent for Meganeura's arXiv classification.

Sources:
[WebLLM, arXiv:2412.15803](https://arxiv.org/abs/2412.15803);
[WebGPU dispatch overhead, arXiv:2604.02344](https://arxiv.org/abs/2604.02344).

## Portable ML compilers and runtimes

### TVM

TVM established the modern performance-portability motivation for ML
compilers: graph- and operator-level optimization, mapping to heterogeneous
hardware primitives, memory-latency hiding, and learned cost models. Its
evaluation includes low-power CPUs, mobile GPUs, server GPUs, and an FPGA
accelerator.

Meganeura should not claim graph lowering, fusion, static compilation, or
performance portability as new in themselves. The useful contrast is the
deliberately narrower execution substrate—consumer graphics APIs—and the
full-training correctness/performance study using the same implementation as
inference.

Source:
[TVM, OSDI 2018](https://www.usenix.org/conference/osdi18/presentation/chen).

### IREE and TinyIREE

IREE uses MLIR to provide a unified compiler/runtime that scales from larger
targets down to mobile, embedded, and bare-metal deployment. TinyIREE makes
the edge and small-footprint motivation particularly relevant to Meganeura's
uniform-stack story.

The distinction to preserve is empirical rather than absolute: Meganeura asks
how far a compact Vulkan/Metal implementation can approach a mature
vendor-native stack for matched inference **and backward execution**. The
TinyIREE paper centers compilation-to-deployment options and footprint on
embedded targets.

Source:
[TinyIREE, IEEE Micro 2022](https://research.google/pubs/tinyiree-an-ml-execution-environment-for-embedded-systems-from-compilation-to-deployment/).

### Glow and MLIR

Glow demonstrates a pragmatic heterogeneous ML compiler based on a
strongly-typed high-level graph, a lower address-only IR, lowering to a small
primitive set, static memory planning, and target-specific code generation.
MLIR generalizes the case for reusable multi-level compiler infrastructure
across abstraction levels and heterogeneous targets.

These systems are useful architectural context. Meganeura's direct-Naga
retrospective contributes a smaller representation lesson: a validated
backend/interchange IR need not be a good human authoring and debugging
boundary.

Sources:
[Glow, arXiv:1805.00907](https://arxiv.org/abs/1805.00907);
[MLIR, CGO 2021](https://research.google/pubs/mlir-scaling-compiler-infrastructure-for-domain-specific-computation/).

### Triton

Triton argues that tile-level programming and compilation can make custom
deep-learning kernels approach hand-tuned vendor libraries without requiring
every author to write low-level GPU code. It is the right comparison for
Meganeura's kernel archetypes and specialization, not for its whole-system
portability claim.

Meganeura operates at a different boundary: it owns graph construction,
autodiff, graph rewrites, memory planning, dispatch, and execution, and lowers
through graphics APIs rather than exposing a kernel language to the user.

Source:
[Triton, MAPL 2019](https://research.ibm.com/publications/triton-an-intermediate-language-and-compiler-for-tiled-neural-network-computations).

## Tensor-graph search and equality saturation

### TASO

TASO automatically generates and formally verifies tensor-graph
substitutions, then uses cost-based backtracking search to jointly optimize
graph substitutions and layouts. It is prior art for search-based graph
optimization and for treating rewrite correctness as a first-class concern.

Source:
[TASO, SOSP 2019](https://www.cs.cmu.edu/~zhihaoj2/papers/sosp19.pdf).

### egg and egglog

`egg` contributes rebuilding and e-class analyses as fast, extensible
mechanisms for equality saturation. `egglog` unifies Datalog-style fixpoint
reasoning with equality saturation and extraction. Meganeura uses egglog; it
does not contribute a new e-graph data structure or equality-saturation
algorithm.

Sources:
[egg, POPL 2021](https://arxiv.org/abs/2004.03082);
[egglog, PLDI 2023](https://arxiv.org/abs/2304.04332).

### TENSAT

TENSAT is the closest published equality-saturation tensor optimizer. It
addresses phase ordering by representing many equivalent tensor graphs in an
e-graph and uses an ILP-based extraction strategy. Its evaluation reports
better runtime graphs and much lower search time than TASO.

Meganeura's ablation is a useful counterpoint, not a competing algorithmic
claim: with Meganeura's current small, mostly locally beneficial rewrite set,
a deterministic greedy pass selects essentially the same executable graphs
at far lower compile cost. The result says that equality saturation pays only
when the available alternatives and cost model create consequential global
choices.

Source:
[TENSAT, MLSys 2021](https://proceedings.mlsys.org/paper_files/paper/2021/file/cc427d934a7f6c0663e5923f49eba531-Paper.pdf).

### Glenside

Glenside introduces a pure, access-pattern-based tensor IR that makes
low-level, layout-aware, hardware-centric rewrites expressible. It shows
rewriting used to discover mappings such as `im2col` and accelerator
invocations.

Meganeura should cite Glenside when discussing representation and hardware
mapping, while making clear that Meganeura's contribution is an executable
training/inference system and empirical evaluation rather than a new pure
tensor IR.

Source:
[Glenside, MAPS 2021](https://arxiv.org/abs/2105.09377).

## Performance portability

Pennycook, Sewall, and Lee define performance portability over an explicit
set of platforms and propose a harmonic-mean-style metric that becomes zero
when an application cannot run on a platform in the set. This is a useful
discipline for the paper:

- declare the device set before computing an aggregate;
- report per-device normalized performance as well as any aggregate;
- define the reference implementation available on each device;
- do not let one strong GPU hide an unsupported or weak platform;
- keep strict and accelerated arithmetic as separate problems.

The final paper can report the metric, but it should not replace the raw
per-device results or gap analysis.

Source:
[A Metric for Performance Portability, PMBS 2016](https://arxiv.org/abs/1611.07409).

## Defensible novelty boundary

The paper should claim the combination and evidence, not ownership of each
ingredient:

1. a compact, executable Vulkan/Metal stack whose same graph/compiler/runtime
   path supports inference and automatically differentiated training;
2. a controlled study spanning transformer, robotics, convolutional,
   diffusion-style, vision, and speech workloads, with strict arithmetic and
   backward correctness gates;
3. cross-vendor and edge-class evidence from one frozen source revision;
4. measured explanations of the remaining vendor-native gaps;
5. negative engineering evidence about direct backend-IR authoring,
   equality saturation with an insufficiently rich rewrite space, and
   independently selected cooperative-kernel geometry and epilogues.

Do not claim the novelty of equality saturation, graph fusion, autodiff,
static memory planning, WGSL/WebGPU ML, portable ML compilation, or
cooperative matrices individually.

## Draft related-work organization

Use three compact paragraphs in the manuscript:

1. heterogeneous ML compilers and portable runtimes: TVM, Glow, MLIR/IREE,
   Triton, and LlamaWeb;
2. graph optimization and equality saturation: TASO, egg/egglog, TENSAT, and
   Glenside;
3. performance-portability methodology: Pennycook et al., followed by the
   exact Meganeura distinction.

Unpublished neighboring projects do not need extended comparison unless
reviewers or public claims make them directly relevant.
