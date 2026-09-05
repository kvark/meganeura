# Meganeura: study and workshop preparation

Prepared 2026-09-05 against development base `bd6be08`, with the audit and
scalar-tuning follow-up on `codex/audit-observability-tuning`. Performance
statements refer to the separately frozen
paper revision, not today's code. No new GPU execution or benchmarking was
performed for this audit.

## The one-minute explanation

Meganeura is a native Rust compiler and runtime for static neural-network
graphs. The same graph, reverse-mode autodiff, shader generators, memory
planner, and runtime serve inference and training. Blade supplies Vulkan and
Metal access; Naga translates the shader representation. Deployment does not
need Python, CUDA, or ROCm.

The research question is not whether Rust is faster than Python. It is how
much useful ML performance a compact, shared graphics-API implementation can
recover across consumer GPUs, including devices poorly served by vendor ML
stacks. The frozen study compares five workloads on five machines under
explicit numerical gates. It wins 12/20 GPU-referenced strict-f32
minimal-shape latency comparisons, but has a 1.78× median valid training-time
ratio against those GPU references. It is credible and uneven, not a universal
PyTorch replacement.

The next engineering opportunity is to replace performance thresholds with
small, reusable, measured searches over legal kernel implementations. That
means specializing to tensor structure and hardware, not recognizing model
names. The first implementation now searches two scalar-f32 matmul tile sizes
in isolated scratch; cooperative/fused search and whole-step confirmation
remain future work. It has CPU validation, not new GPU performance evidence.

## Reading paths

For a 30-minute orientation, read this page, the opening and summary tables
of [results](results.md), and the short answers in
[workshop questions](p3hpc-questions.md).

For a two-hour technical pass, read in this order:

1. [Architecture](architecture.md): follow one graph all the way to execution.
2. [Observability and debugging](observability.md): inspect values and plans;
   understand the eager-PyTorch tradeoff and probe limitations.
3. [Design decisions](design-decisions.md): understand why the design changed.
4. [Results and evidence](results.md): know exactly what each number means.
5. [Alternatives](alternatives.md): compare layers and scope, not slogans.
6. [Performance plan](performance-plan.md): the implemented first slice and
   the minimal, general route forward.
7. [Questions and exercises](p3hpc-questions.md): practice defending the claims.

For the engineering backlog, use the [September audit](../audit-2026-09.md).
For camera-ready work, use the [revision plan](../../paper/p3hpc/REVISION.md)
and [paper source](../../paper/p3hpc/main.tex).

## Keep these distinctions straight

| Phrase | What it actually means here |
|---|---|
| Portable | A shared Vulkan/Metal implementation with scalar fallbacks; not identical speed or universal operator coverage. |
| General | Shape/operator-driven compilation within the supported static IR; not arbitrary PyTorch program compatibility. |
| Minimal | A compact owned stack and deployment closure, helped by substantial dependencies; not the fewest possible operations or source lines. |
| Automatic | Graph differentiation, rewriting, specialization, scheduling, and memory planning; only limited, opt-in measured kernel selection today. |
| Training result | Frozen matrix measures forward + loss + backward, without an optimizer update. The library separately supports optimizer-backed training. |
| Minimal latency | Small, stateless shapes; the LLM cell is one token without KV cache, not production autoregressive decoding. |
| Valid | Passes sampled-output/loss and gradient-norm gates; not proof of elementwise gradients or convergence. |
| Strict f32 | A controlled arithmetic permission contract, not bitwise identity across engines. |
| Current code | Development base plus audit changes; no newly established performance numbers. |

## Claims to avoid

Do not say that all Apple/AMD/Intel training is faster, that the paper tests
an RTX 5080 (it tests a 5070), that equality saturation provides the measured
speedups, that all kernel choices are autotuned, or that compensated f16
preserves the exponent range of f32. Do not describe the validation as full
gradient comparison, the Intel CPU ratios as GPU-to-GPU speedups, or the
Quest demonstration as on-device training. LoRA/QLoRA on Apple is not unique
to Meganeura. Compiler specialization and automatic tuning are established
ideas; the particular implementation and validity-gated portability evidence
are the contribution.

The documentation marks measured evidence, source-level observations, and
proposals separately. If a workshop question goes beyond that evidence,
"we have not measured that yet" is the technically correct answer.
