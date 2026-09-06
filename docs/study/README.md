# Meganeura: study and workshop preparation

Prepared 2026-09-05 against development base `bd6be08`, subsequently rebased
onto `8069cf3` with checkpoint/memory work, `a455409` for training holdouts,
then `230cab0` for controlled crossover confirmation
on `codex/audit-observability-tuning`.
Measured sources are retained as `evidence/tuning-2026-09-05`,
`evidence/holdouts-2026-09-06`, `evidence/crossover-2026-09-06` and
`evidence/readback-2026-09-06`, `evidence/allocation-profile-2026-09-06` and
`evidence/staging-reuse-2026-09-06` and `evidence/training-profile-2026-09-06`;
rebasing does not relabel those measurements as results of newer code. Paper
performance statements refer to the separately frozen revision. The initial audit was
CPU-only; subsequent GPU qualification is described separately in the
[performance plan](performance-plan.md), after the device was released.

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
names. The implementation searches scalar-f32 and legal native-f32 cooperative
matmul tiles in isolated scratch. Scalar GPU qualification passed; native-f32
hardware coverage is still due. F16-input/complex-fusion search and persistent
winners remain future work. The whole-step experiment harness is separate
from the frozen publication evidence. Its [first five-process experiment](../experiments/tuning-2026-09-05/README.md)
found repeatable 1.15×/1.13× gains on two synthetic dense chains; two smaller
cases retained their initial tiles. This is not a new PyTorch comparison.

The [broader five-process holdouts](../experiments/holdouts-2026-09-06/README.md)
now cover nonlinear inference and optimizer-backed trajectories: full control-
session gradients/moments and update counts pass, but none of the 30 case runs
clears the whole-step gain/regression guard. This is why tuning remains opt-in
and why controlled confirmation and wider kernel-family coverage matter.

The [six-process crossover](../experiments/crossover-2026-09-06/README.md)
now confirms a median 1.177× dense-chain gain with the selected plan on either
session. MLP+Adam has no confirmed gain; two untuned/untuned controls are noisy.
ResNet is unchanged, and its new phase timers put roughly 98% of search time
in qualification. Read this as a lesson in separating numerical correctness,
kernel selection, search cost and whole-step acceptance—not as a new paper result.

The [staging follow-up](../experiments/readback-2026-09-06/README.md), on published
Blade 0.9, separates CPU readback copies from numerical checks and transfer/wait.
Across six paired processes, ResNet's median total search falls from 606 to
39 ms: CPU readback allocation/copy falls from 582 to 2 ms, while the same
validation takes about 6 ms. All 108 comparisons qualify and state remains
bit-exact through Adam step 178. Private tuning staging now defaults to
Download, with Shared still available; tuning itself remains opt-in. Added
preparation/transfer costs and the first slower dense run are retained. This
is cheaper search on one device, not faster ResNet execution or a Blade-version
comparison. Development now requires Rust 1.92.

The [allocation/reuse follow-up](../experiments/staging-reuse-2026-09-06/README.md)
then locates the remaining preparation cost in staging allocation. One exact-size
buffer now survives between comparisons within a tuning call, never after return.
Across six paired processes this lowers median dense search 44.01→31.84 ms and
MLP+Adam 64.27→45.46 ms. ResNet has no reuse opportunity and no guarded change.
All validation and scratch byte bounds remain intact. This is another search-cost
result, not a reversal of the earlier inconclusive MLP whole-step result.

The [whole-step localization follow-up](../experiments/training-profile-2026-09-06/README.md)
prioritizes backward convolution in ResNet and backward attention in SmolLM2.
All 45 profiled full states match ordinary execution, but short-model timing
drift and instrumented pass overhead remain visible. These are F+L+B profiles,
not optimizer-backed or cross-engine speedups. Reviewing convolution then found
and repaired a non-same-padding dX indexing bug using an independent f64 oracle;
read [the design lesson](design-decisions.md#9-a-shared-baseline-is-not-an-independent-oracle).

The memory/restore follow-up removes unused Adam allocations, reports actual
resident tensor-buffer requests, and restores logical checkpoints only after
whole-file preflight. Read the [checkpoint and memory chapter](checkpoints-and-memory.md)
for the failure boundary, resume limitations and real-device qualification.

## Reading paths

For a 30-minute orientation, read this page, the opening and summary tables
of [results](results.md), and the short answers in
[workshop questions](p3hpc-questions.md).

For a two-hour technical pass, read in this order:

1. [Architecture](architecture.md): follow one graph all the way to execution.
2. [Observability and debugging](observability.md): inspect values and plans;
   understand the eager-PyTorch tradeoff and probe limitations.
3. [Checkpoints and memory](checkpoints-and-memory.md): understand state,
   portable serialization, lazy optimizer allocation and honest memory metrics.
4. [Design decisions](design-decisions.md): understand why the design changed.
5. [Results and evidence](results.md): know exactly what each number means.
6. [Alternatives](alternatives.md): compare layers and scope, not slogans.
7. [Performance plan](performance-plan.md): the implemented first slice and
   the minimal, general route forward.
8. [Questions and exercises](p3hpc-questions.md): practice defending the claims.

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
| Current code | Rebased audit/tuning/checkpoint/memory changes; qualification and tuning experiments must be distinguished from the frozen model results. |

## Claims to avoid

Do not say that all Apple/AMD/Intel training is faster, that the paper tests
an RTX 5080 (it tests a 5070), that equality saturation provides the measured
speedups, that all kernel choices are autotuned, or that compensated f16
preserves the exponent range of f32. Do not describe the frozen matrix's
validation as full gradient comparison (the newer control-session holdouts
are separate), the Intel CPU ratios as GPU-to-GPU speedups, or the
Quest demonstration as on-device training. LoRA/QLoRA on Apple is not unique
to Meganeura. Compiler specialization and automatic tuning are established
ideas; the particular implementation and validity-gated portability evidence
are the contribution.

The documentation marks measured evidence, source-level observations, and
proposals separately. If a workshop question goes beyond that evidence,
"we have not measured that yet" is the technically correct answer.
