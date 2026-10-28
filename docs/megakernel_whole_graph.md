# One dispatch for the entire Session graph

`PersistentOptions { whole_graph: true, device_scheduling: true, ... }` runs a
nonempty graph in one compute dispatch. A fixed physical worker pool repeatedly
claims logical workgroup tiles and schedules dependent operations on the GPU.
The step's optimizer, global/adaptive clipping and temporal accumulation are
nodes in the same dispatch. The transfer reset and compute dispatch share one
queue submission. Initial parameter/moment allocation and explicit optimizer
calls remain separate operations outside that step boundary.

The runtime preserves `e4e70c5`'s guarded [inactive storage reads](#inactive-storage-reads-and-worker-participation),
including read-only views that may alias writable bindings. It retains the
preceding scratch-read repair and packing optimizations. Passing numerical
tests alone did not reveal these extra inactive accesses.

The latest [advisory ready-word summary](#advisory-ready-word-summary), retained
in `c3ad104`, skips empty words while keeping the original full scan before an
idle epoch. Against qualified `3f95fa4`, 1,000 completed-step comparisons improve
persistent training **1.39% original / 1.61% prewarmed**, with both pooled 95%
intervals below no change. Matched SDK training replay improves **2.24%**;
ordinary training is stable. Forward completed-step gain remains unestablished.
That study's paired persistent/ordinary medians are **1.289× forward / 1.052× training**,
or **1.300× / 1.139×** prewarmed. All requested workers remain enabled; baseline
parity remains unmet.

Two subsequent [summary-maintenance controls](#advisory-summary-maintenance-controls)
establish no gain. Moving hint publication earlier while removing empty-word
clears leaves GPU diagnostics flat. Guarding redundant RMWs reduces estimated
global atomic sectors by 5.56%, but L2 atomic traffic and SDK training replay
remain nearly unchanged. Both probes are reverted; `c3ad104` remains qualified.

A subsequent [single-word summary scan](#rotated-single-word-summary-scan) passes
full qualification and 1,000 completed-step comparisons. Original training
improves 0.71%, but prewarmed training remains inconclusive. The fixed criterion
requires an absolute training gain under both protocols, so this candidate is
also reverted without adding repeats. Baseline parity remains open.

The following [targeted two-tile claims](#targeted-transposed-a-two-tile-claims)
also regress. After correcting an admission guard that excluded subgroup votes,
looped and fixed two-call forms increase training GPU diagnostics by 1.04% and
1.15%, with registers rising from 84 to 103/96. Both are reverted.

The subsequent [private tile reservation](#private-second-tile-reservation)
preserves that single handler call but also establishes no gain. SDK global/L2
atomic estimates fall 3.51%/3.73%, while training replay stays flat. Disabling
reservations with identical shaders and moving the check outside idle polling
also produce no retained optimization. All probes are reverted; `c3ad104`
remains the qualified runtime.

The following [final-stage barrier control](#final-scalar-at-stage-barrier) also
stays flat: training GPU diagnostics change −0.11%, while SDK replay changes
+0.15% persistent / +0.04% ordinary. It preserves arithmetic, tile geometry,
registers and shared memory. The probe is reverted.

The subsequent [individual AT tile clocks](#individual-at-tile-clocks) separate
tile intervals from gaps between claims. The two largest sampled shapes average
only 53–54 overlapping tile intervals despite 168 workers participating in the
whole training graph. Their between-tile gaps include other work and cannot be
counted as idle time. The calibrated probe adds 1.25% training GPU time and is
removed. Subsequent [optimizer claim clocks](#optimizer-claims-alongside-at-tiles)
find no recorded optimizer body overlapping either largest sampled shape, and
none inside same-worker gaps between sampled AT claims. Subsequent
[all-claim traces](#all-graph-claim-attribution) identify concurrent BT work:
AT720×4096×50 finishes before its paired BT node in all 32 observations.
Its long individual span therefore does not establish a whole-step loss.
The subsequent [two-tile SwiGLU claims](#two-tile-swiglu-claims) reduce diagnostic
GPU time by about 0.65%, but all four model absolute intervals include no change
in 1,000 completed-step comparisons. The candidate is reverted without extra
repeats. Qualified runtime remains `c3ad104`.
The following [four-tile metadata control](#four-tile-swiglu-metadata-control)
preserves the two-tile shader bytes and improves forward GPU diagnostics 1.68%
against qualified. In 1,000 normal comparisons, original forward and prewarmed
training improve, but neither phase passes both protocols. It is also reverted.
Subsequent [family-isolation controls](#swiglu-family-isolation-and-forward-only-batching)
select forward-only batching. Its production candidate passes full correctness
qualification and 1,000 normal comparisons, but neither model phase passes both
protocols. It is reverted; the temporary metadata probe is removed.
The following [three-tile forward control](#three-tile-forward-claim-control)
preserves all shader bytes and native statistics against four tiles, but is
0.24% slower in forward GPU diagnostics and 0.20% slower in training. It fails
the predeclared advancement gate and is reverted without normal timing or SDK
captures. Qualified runtime remains `c3ad104`.
The subsequent [complete forward claim trace](#complete-forward-claim-attribution)
finds a 137-node dependency chain with 477–484 µs between recorded predecessor
completion and first handler entry. Only about 21 µs overlaps other recorded
handler bodies. This identifies publication-to-entry latency as an attribution
target; the remaining gaps include synchronization, queue work, parameter setup
and instrumentation, and are not established recoverable time. The probe is
removed and the qualified runtime remains unchanged.
The subsequent [forward stage trace](#forward-publication-and-entry-stages)
separates completion, publication bounds, the zero-ticket claim, ticket return
and handler entry. Delay is distributed across these stages; the extra checks
also perturb the gaps. No fence or runtime change is retained from the trace.
The subsequent [successor release-fence control](#successor-release-fence-control)
passes the source-level ordering and emitted SPIR-V audit, but fails its fixed
diagnostic gate: forward improves 0.60% while training regresses 0.67%, exceeding
the 0.5% guardrail. The one-fence deletion is reverted without extra timing or
SDK captures. Qualified runtime remains `c3ad104`.

The following [parameter staging controls](#parameter-staging-controls) also
establish no gain. Caching copies leaves model diagnostics flat; caching the
broadcast value increases forward registers and time. A direct shared-array
read fails existing packing admission tests before GPU execution. Removing only
the emitted post-read barrier improves training by 0.46% against qualified and
0.33% against a matched loader control, below the required 0.5% against both.
All four probes are removed; no Naga API change follows from these results.

The subsequent [native cooperative B prefetch](#native-cooperative-b-prefetch)
also fails its diagnostic gate: forward/training GPU time rises 1.40%/1.50%
while ordinary execution stays roughly flat. Model registers remain 86 forward
and fall 84→83 in training, so register count alone does not explain the result.
The probe is reverted and the qualified runtime remains `c3ad104`.

The following [active-lane payload-fence control](#active-lane-payload-fences)
passes full correctness qualification and improves forward GPU diagnostics
1.56%. Its fixed 1,000 completed-step comparisons improve original forward
2.84%, but prewarmed forward and both training intervals remain inconclusive.
Neither model wins under both protocols, so it is reverted without extra runs
or SDK captures. The retained memory contract remains unchanged.

The subsequent [disjoint ready-word scans](#disjoint-ready-word-scans) avoid
examining a word again in the fallback after the advisory pass has already
checked it. Correctness and source audits pass, but forward/training GPU
diagnostics change +0.21%/−0.30%, below the required gain. The candidate is
reverted without extra repeats; qualified runtime remains `c3ad104`.

The following [scalar GEMV scratch pooling](#scalar-gemv-scratch-pooling) removes
4,112 shared bytes from each model shader. Forward registers also fall 86→77,
but participating workers rise 168→252 and forward GPU time worsens 21.38%.
Training improves only 0.51%. The candidate fails the model-regression guardrail
and is reverted. Smaller resource counts alone do not predict a useful gain.

The subsequent [bounded idle-poll backoff](#bounded-idle-poll-backoff) tests
64/256 clock ticks against both qualified execution and an identical zero-delay
probe. Neither delay improves a model phase by 0.5% against both controls.
The 256-tick training result improves 0.71% against qualified but regresses
0.09% against zero delay. The experiment is restored without extra runs or a
Naga API change; it does not establish how much polling traffic changed.

The following [counter-spacing control](#node-counter-spacing) puts independent
node counters 128 bytes apart, with identical shaders and storage in a dense
control. Forward/training GPU time regresses 0.82%/1.73% against dense and
1.12%/1.68% against qualified execution. The fixed gate fails; the layout is
restored without extra runs or fresh SDK captures.

The preceding [cooperative prologue factor caching](#cooperative-prologue-factor-caching)
keeps each lane's two per-row factors in registers across the K loop, after
the existing shared-cache initialization and barrier. Against qualified
`cf8940f`, 1,000 completed-step comparisons improve persistent forward by
**5.8% original / 6.6% prewarmed**. SDK forward replay improves **6.9% persistent /
2.3% ordinary**. CI and training shader bytes are unchanged; no new gain is
attributed to those cases. That qualification's paired persistent/ordinary
medians are
**1.310× forward / 1.069× training**, or **1.282× / 1.165×** with prewarming.
All requested workers remain enabled. Baseline parity remains unmet.

Three subsequent [column-factor reuse probes](#cooperative-column-factor-reuse)
reduce duplicated factor loads but regress persistent GPU time. All are reverted;
the then-qualified runtime was `3f95fa4`.

The [refreshed sparse clocks](#refreshed-handler-clock-attribution) now identify
short-K transposed-A weight gradients as a stronger remaining target. The
instrumentation is removed after calibration; this is attribution evidence,
not another retained speedup.

Three subsequent [short-K tile probes](#short-k-transposed-a-tile-geometry)
replace selected 64×64 tiles with 32×32, 32×64 or 64×32. All pass correctness
checks but regress persistent training GPU diagnostics by 9.5%, 2.5% and 1.3%.
They are reverted. Subsequent [optimizer-overlap controls](#optimizer-overlap-controls)
preserve all shader bytes: globally deferring updates regresses training about
5%, while serializing independent updates in prerequisite-count order yields
no useful gain. All three dependency probes are reverted; the then-qualified runtime
was `3f95fa4`.

Two [cooperative-attention probes](#cooperative-attention-ownership-and-row-softmax)
also establish no gain. Restricting score computation to its publishing subgroup
leaves model GPU diagnostics within 0.2% of control. Sharing row softmax results
adds a barrier and shared storage; forward SDK replay regresses 0.57% despite
slightly lower estimated instruction counts. Both are reverted.

The subsequent [subgroup-sharing probes](#cooperative-attention-subgroup-sharing)
avoid additional barriers and scratch. Sharing one producer's softmax or
distributing keys across lanes also fails to improve GPU diagnostics, despite
preserving the original reduction order. Both probes and their temporary Blade
capability are reverted; the qualified runtime is unchanged.

Two [attention scratch-reset controls](#cooperative-attention-scratch-initialization)
omit redundant converter clears, with and without the converter's entry barrier.
Both pass correctness and exact source audits but leave model GPU diagnostics
within 0.2% of control. Both are reverted; no new speedup is established.

The following [cooperative-matrix reset controls](#cooperative-matrix-scratch-initialization)
also stay within 0.2% of control and are reverted. An expanded strict matrix
sweep identifies seven admission rejections shared by the baseline and both
candidates, while all forty-four admitted cases pass.

Two [ready-width specialization probes](#immutable-ready-bitmap-width) pass full
qualification and 3,000 completed-step comparisons, but no absolute model gain
survives reversed-order confirmation. Refined training SDK GPU time improves
1.02% and normalized prewarmed timing improves in both sequences; that evidence
does not meet the declared absolute retention criterion. Both are reverted.
Fresh control paired persistent/ordinary medians are **1.309× forward / 1.093×
training**, or **1.275× / 1.164×** prewarmed. That study retained `3f95fa4`.

The [current-footprint epoch control](#completion-epoch-control-on-the-current-runtime)
then tests continuous ready-bitmap scanning against the latest optimized handlers.
It passes full qualification and 1,000 completed-step comparisons. Short GPU
intervals improve 1.18% forward / 0.81% training, with unchanged model worker
participation, but all absolute completed-step model intervals include no change.
The control is reverted; the retained epoch policy stays. This completes the
previously deferred direct forward comparison.

The preceding [fused-matrix tile selection](#fused-matrix-tile-selection) adds
small-tile routing for fused transposed-B products and extends the existing
32×16 fused-plain refinement to K≥256. Against qualified `da86b45`, 1,000
completed-step comparisons improve persistent forward/training by **4.8%/4.1%**
under the original protocol. Matched prewarming measures 9.1%/4.2%, although
ordinary forward also shifts substantially. SDK replay confirms smaller
forward/training gains of **3.5%/4.4%**. Ordinary training improves about 1.8%.
That qualification's paired persistent/ordinary medians were **1.321× forward /
1.080× training**, or **1.368× / 1.166×** with prewarming. CI shaders were unchanged.

The preceding [completion-epoch policy](#completion-gated-idle-scans) waits on
one counter between empty scans of a multiword bitmap. Its separate comparison
improves training 7.0%/5.5%, with a 1.05% original-protocol forward regression.
The [current-footprint control](#completion-epoch-control-on-the-current-runtime)
now repeats that comparison without establishing a completed-step benefit from
removing epoch waiting.

The earlier storage repair's 500 primary comparisons measure a 5.58% absolute training regression
from the storage repair. Paired persistent/ordinary medians are 1.355× forward
and 1.219× training. A separate worker-count control supports a cost from the
increased number of participating workers. The repair is retained for
correctness; requested worker counts and the 256-worker primary gate remain
unchanged. Baseline performance parity remains unmet.

A subsequent [attention-reduction experiment](#serial-attention-reduction-tail)
removes two barriers while preserving the exact addition tree. Its 500 new
comparisons establish no useful gain, so the patch is reverted. Fresh retained
control ratios are 1.357× forward and 1.231× training.

The [subgroup-shuffle experiment](#subgroup-attention-reduction) also preserves
the exact tree, but has an unacceptable tradeoff: persistent training improves
1.66% while ordinary training regresses 8.12%. It is reverted with its dependency
changes. Its retained-control ratios are 1.346× forward and 1.217× training.

A [static-unrolling follow-up](#static-subgroup-reduction) fixes the ordinary
regression but does not establish a persistent completed-step gain in its
confirmation run. It is also reverted. Confirmation retained-control ratios
are 1.424× forward and 1.209× training; forward timings vary substantially.

A [packed-activity specialization](#fully-occupied-packed-claims) changes only
one matrix handler in each model phase. It reduces instruction work but does
not establish an absolute model speedup in confirmation, and is reverted.
Same-binary ordinary training visits range from 11.79 to 14.72 ms. A subsequent
[stability investigation](#gpu-idle-effects-and-benchmark-stability) localizes
the large spikes to GPU execution and suppresses them with untimed prewarming.
The warmed diagnostic still shows a substantial persistent GPU-time gap.
The [current clock study](#warmed-handler-clock-attribution) identifies expensive
narrow matrix stages and overlapping SGD work. Its smaller probe preserves
training's register count and worker participation. Neither probe is retained.
The subsequent [capacity control](#worker-capacity-and-concurrency-control)
reproduces a training improvement without clock instrumentation, while keeping
the requested 256 workers and all handler arithmetic.
A [single-broadcast claim prototype](#single-broadcast-claim-snapshot) then passes
50 diagnostic comparisons without a material model improvement. It is reverted;
the qualified runtime and both forks remain unchanged.

For model benchmarks, use `--whole-graph --workers 256`. The
[current worker sweep](#worker-counts-and-deeper-scalar-staging) finds lower CI
backward/SGD latency with an explicit 64-worker pool; no universal replacement
count is established. For a regular Session configured from the environment:

```sh
MEGANEURA_PERSISTENT_WORKERS=256 MEGANEURA_DEVICE_SCHEDULING=1 \
MEGANEURA_WHOLE_GRAPH=1 your_application
```

Ordinary execution remains the default. The older region implementation remains
available by leaving `whole_graph` off. Its earlier counter results do not
describe the costs of this combined shader.

On this Linux profiling host, launch GPU tests and measurements with
`python3 scripts/run_bounded_job.py LABEL --gpu -- COMMAND ...`. Each job gets a
separate verified cgroup (16 GiB soft memory limit, 20 GiB hard limit, 1 GiB swap).
The GPU flag holds a shared file lock across command execution, so marked jobs
cannot overlap. Library tests also use the GPU and need this flag. Builds use
the same launcher without the GPU flag; finish them before primary measurements.
The launcher now records memory peaks, limit events and memory/CPU/I/O pressure
after each job, preserving the target's exit status and forwarding termination.

## Advisory ready-word summary

The retained change starts from `5cfea07` and compares against qualified runtime
`3f95fa4`. For multiword graphs it adds one advisory summary bit per ready-bitmap
word. A leader visits hinted words in cyclic order from its existing start word.
Each visit re-reads the authoritative bitmap, and only the original tile-counter
RMW grants ownership. If the hinted pass fails, the leader performs the original
full bitmap scan before caching an empty completion epoch. That scan also
repairs missing hints. Single-word graphs keep their original shader and layout.

Publishers set hints after their ready bits. Claimers clear a hint when removing
the final bit from a ready word, or when a hinted word is empty. These operations
can race and leave missing or stale hints; both states are permitted. A missing
hint can postpone a ready word while other hinted work succeeds, but finite DAG
work and the full-scan fallback preserve progress. Cyclic preference is preserved
among hinted words; concurrent selection need not match the control interleaving.
Summary state never authorizes ownership, completion or an idle epoch by itself.

The existing per-lane payload, completion fan-in, dependency publication and
epoch fences remain intact. No Naga interface or compiler pin changes. Nodes
are published once. Same-invocation ready-bit clearing precedes a subsequent
load through the same queue reference; the reasoning in the
[memory contract](megakernel_memory_contract.md#progress-completion-and-reuse)
preserves the original per-worker overclaim bound even across the two searches.

The existing two configuration padding words hold the summary offset/count.
The queue adds one word per 1,024 graph nodes to each live/reset buffer: **eight
additional scheduler/resident bytes** in each model phase. Payload allocations
and logical buffer sizes do not change. Host completion validation still checks
the full ready bitmap, every node counter and worker task totals. Advisory bits
may remain stale at completion, so their index domain is checked separately.

The host checker exhausts **30,948 ready-set/hint/worker states**, verifies
**7,295 segmented scan boundary cases**, and enumerates twenty publication/clear
interleavings. It observes both missing and stale hints. This is not a complete
Vulkan memory-model checker; its progress argument allows arbitrary hint
staleness and relies on the unchanged authoritative scan and publication rules.
The existing GPU boundary test expands to **32/33/65/1024/1025 nodes**, independent
chains and fan-in, all intermediate outputs, poisoned storage, three replays and
**1/3/256/4096 workers**.

Qualification passes **676 library tests (eight ignored), 168 strict whole-graph
oracle cases, 51 matrix regressions, fourteen expanded smoke tests and 52 codegen
checks**, plus strict all-target/all-feature Clippy, formatting, release builds
and Metal-target compilation. Rebuilt executables match the frozen candidate
binaries exactly. Metal GPU execution and additional hardware remain unqualified.

Generated-source audits recover each whole graph from the original by replacing
its claim block and adding the one summary-publication RMW. Removing hint
maintenance from the helper recovers every token in the original tile-claim loop,
with its successful break represented as a helper return. All handler bodies,
**nineteen CI sources and forty ordinary model sources** remain byte-identical.
All epoch and payload fences remain. The two model graphs change only their
scheduler functions and add `graph_claim_word`.

Fifty full-state diagnostics pass every selection, work, one-dispatch and
no-fallback check. Maximum model error remains 4.66e-9. Persistent forward GPU
time changes **4.095744→4.074752 ms (−0.51%)**, and training
**13.039840→12.724256 ms (−2.42%)**. Ordinary training changes +0.03%.
Model registers remain **86/84**, shared bytes **18,928/29,344**, participating
workers **168**, requested workers **256**, and logical tiles **22,042/83,312**.
Code grows 99,584→102,400 forward and 238,720→241,664 training bytes.

The normal gate completes **1,000 comparisons**: 500 original and 500 with
matched 100 ms prewarming, control/candidate/candidate/control visits, 25
interleaved pairs per case/visit, at least ten warmups and one warmup second.
All five cases pass full-state budgets and execution checks; all samples remain.

| Persistent completed step | Control, ms | Candidate, ms | Candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| Original forward | 4.705179 | 4.636049 | 0.9853 [0.9101, 1.0028] |
| Original training + SGD | 15.560865 | 15.345248 | 0.9861 [0.9678, 0.9977] |
| Prewarmed forward | 4.370782 | 4.543098 | 1.0394 [0.9383, 1.0737] |
| Prewarmed training + SGD | 13.459792 | 13.243651 | 0.9839 [0.9633, 0.9991] |

Training improves **1.39% original / 1.61% prewarmed**, with both absolute
intervals below one. Original ordinary training moves +2.43%, inflating the
normalized improvement; prewarmed ordinary training moves +0.11%. Paired
normalized training improves 3.77% and 2.22%, respectively. Forward absolute
intervals include no change; no forward completed-step gain is claimed.
Original CI backward changes +0.88% persistent / +1.01% ordinary on unchanged
shaders; the normalized interval includes no change. All CI shifts remain in
the record. Pooled bootstrap intervals omit some between-visit variation.

Current paired persistent/ordinary medians are **1.289× forward / 1.052× training**,
or **1.300× / 1.139×** prewarmed. These are medians of paired ratios, not ratios
of aggregate medians. Baseline parity remains unmet.

Eight matched SDK backend captures contain **32 metric groups**. Every group
has at least 100 post-warmup samples, valid target/device identity and successful
exits. Persistent forward replay changes **4.217166→4.203696 ms (−0.32%)**,
versus ordinary 3.274125→3.276010 ms (+0.06%). Persistent training replay improves
**13.347622→13.049200 ms (−2.24%)**, versus ordinary
11.719042→11.721722 ms (+0.02%).

Estimated persistent training warp instructions fall **0.83%**, integer-pipe
work changes −0.03%, FP32 −0.01%, global-load sectors −0.15% and shared wavefronts
−0.25%. Local traffic remains zero. Estimated global atomic sectors rise
**24.78%**, and L2 atomic sectors rise **22.95%**. Barrier waiting falls
**46.80→45.45% of active warp cycles**, while active compute warps stay near
32.6% of peak. Faster execution accompanies increased atomic traffic and reduced
waiting, not a large reduction in total integer work. These counters do not
isolate the exact source of the gain.

Forward estimated global/L2 atomic sectors rise 20.47%/40.27%, with a much
smaller replay change. Counter rates multiplied by duration are estimates;
barrier shares are not elapsed-time fractions. Maximum clock/instruction-rate
spreads are **0.013%/1.980%**. Unchanged ordinary forward instruction estimates
move about 1.44%, illustrating their uncertainty. Zero ordinary atomic counts
have no defined candidate/control ratio.

All eight actual exported states match bit-for-bit between control and candidate
within each backend. The four forward states are also identical across backends:
**147,034,513 floats / 588,138,052 bytes**, SHA-256
`4e5d66e6ce5e95142b848651d316b4df64855a9d85669499cabc3a81089be629`.
Each training state contains **246,883,105 floats / 987,532,420 bytes**. Ordinary
SHA-256 is `f67022044ae8966ac48c5d62cd0907411fb42f4d1864c1ba0b3d555b17ce7be0`;
persistent SHA-256 is `1e359875fa92965d4b0aa9026c050702ae3ac165aa59c0a4cc74a6381589c3b7`.
Cross-backend training passes tolerance but is not bit-identical. SDK training
uses zero-rate SGD for stable replay, separately from normal nonzero-rate timing.

Runtime **`c3ad104` is retained**. Evidence remains local in
`target/ready-word-summary/`. Manifest
`target/megakernel-results/ready-word-summary/summary.json` independently verifies
**542 artifact hashes and 43 completed study cgroups**, SHA-256
`cfb49fce887414cff3cbc913bb1676ba9f1707a4427226df080ed4e66b3bb1a9`.
Peak recorded job memory is **4,715,782,144 bytes**, with zero memory-limit events.
All heavy jobs use verified 16/20/1 GiB memory-high/hard/swap limits; GPU jobs are
serialized, and builds/copies/heavy analysis stay outside measurements. All
launched jobs finish. Both compiler forks remain clean and unchanged.

The subsequent [maintenance controls](#advisory-summary-maintenance-controls)
test that extra atomic traffic directly, without establishing a useful gain.

## Advisory summary maintenance controls

Starting from `1548824`, both controls use qualified runtime `c3ad104` and its
frozen `target/ready-word-summary/final-*` binaries. The first moves successor
hint publication ahead of the existing ready-bit release fence and removes
empty-word hint clears. The unique last-ready-bit clear and fallback repairs
remain. The second restores the original publication order, then guards
empty-word clears, fallback repairs and successor hint sets with an atomic load
of the hint bit. It leaves the unique last-ready-bit clear unconditional.

Hints remain advisory, including when a guard races and skips an update. Every
authoritative bitmap, tile-counter, dependency and epoch operation retains its
relative order, as do payload fences. The full scan remains mandatory before
caching an empty epoch. No worker is retired or capped, and no compiler-fork
change is involved.

Each host checker passes 30,948 ready/hint/worker states and 7,295 segmented
scan-boundary cases. The first also checks forty publication/clear schedules
across the two policies: the simplified model sees more missing-hint outcomes
with early publication, eight versus two. The second enumerates twelve guarded
set/clear interleavings, including skipped updates. These are not complete Vulkan
memory-model proofs and establish no stronger hint consistency guarantee.

Both candidates pass the release benchmark build, formatting, **52 codegen
checks and fourteen whole-graph smoke tests**, including 1,024/1,025-node summary
boundaries and 1/3/256/4096-worker replays. Each has fifty diagnostic comparisons
over all five CI/model cases, with 256 requested workers and matched 100 ms
prewarming. All state, selection, completion, single-dispatch and no-fallback
checks pass; maximum model error is 4.66e-9.

Exact source inverses cover both changed model shaders. Only `graph_claim_word`
and `graph_main` change; all handlers, nineteen CI shaders and forty ordinary
model shaders stay exact. Queue and payload allocations do not change. Both
model phases retain 86/84 registers, 18,928/29,344 shared bytes, 168 participating
workers and 22,042/83,312 logical tiles. Code size changes −128/−256 bytes for
early publication, and +256/+256 bytes for guarded RMWs.

| Five-pair GPU diagnostic median | Qualified control | Candidate | Change |
|---|---:|---:|---:|
| Early publication: forward | 4.081536 ms | 4.084448 ms | +0.07% |
| Early publication: training | 12.736928 ms | 12.729952 ms | −0.05% |
| Guarded RMWs: forward | 4.073088 ms | 4.109792 ms | +0.90% |
| Guarded RMWs: training | 12.736192 ms | 12.757504 ms | +0.17% |

Ordinary GPU changes remain within 0.19%. Neither candidate passes this initial
timing screen. No normal completed-step series or full library/oracle/Metal
qualification is claimed for either rejected candidate.

Four matched training SDK backend captures collect sixteen metric groups for
the guarded candidate. Replay changes **13.043040→13.074000 ms (+0.24%)**
persistent, versus **11.694901→11.720792 ms (+0.22%)** ordinary. Estimated global
atomic sectors fall **259,564→245,135 (−5.56%)**, while L2 atomic sectors change
**243,661→243,109 (−0.23%)**. This does not establish summary-maintenance traffic
as a useful timing target. It also does not isolate the cost of every remaining
queue operation.

Estimated warp/integer/FP32 work and global loads all shift about −0.5%, which
should not be interpreted as a selective arithmetic reduction. Barrier waiting
changes **45.31→45.48% of active warp cycles**; active compute warps stay near
32.6% of peak, with no local traffic. Counter rates multiplied by replay duration
are estimates, and barrier share is not an elapsed-time fraction. Maximum
between-group clock/instruction-rate spreads are 0.44%/0.71%; all groups collect
at least 100 post-warmup samples and pass adapter/exit validation.

All four actual exported training states contain **246,883,105 floats** each.
Control and candidate match bit-for-bit within each backend: ordinary SHA-256
`f67022044ae8966ac48c5d62cd0907411fb42f4d1864c1ba0b3d555b17ce7be0`, persistent
`1e359875fa92965d4b0aa9026c050702ae3ac165aa59c0a4cc74a6381589c3b7`.
Cross-backend states pass the numerical budget but are not bit-identical. SDK
training uses zero-rate SGD for stable replay; no forward SDK capture is claimed.

Both probes are reverted. All **twenty source/dependency hashes**, both qualified
benchmark executable hashes and fourteen restored smoke tests match the retained
runtime. Evidence stays local in `target/ready-summary-maintenance/` and
`target/ready-summary-guarded-rmw/`. Manifest
`target/megakernel-results/ready-summary-maintenance/summary.json` independently
verifies **761 artifact hashes and 31 completed study cgroups**, SHA-256
`31509e91a8fe9d43fb40ffdb6f9fbe744252f58b289e78291aa0369bf498ca3a`.
Peak recorded memory is **4,715,696,128 bytes**, with zero memory-limit events.
All heavy jobs use the verified 16/20/1 GiB cgroup limits, GPU runs are serialized,
and all launched jobs finish. Both compiler forks remain clean and unchanged.

The following [single-word scan](#rotated-single-word-summary-scan) tests that
remaining traversal cost. The short-K transposed-A gap also remains unresolved.

## Rotated single-word summary scan

This candidate starts from `c0dbc51`, comparing against qualified runtime
`c3ad104` and its archived `target/ready-word-summary/final-*` executables. For
graphs with 2–32 ready words (33–1,024 nodes), it loads the advisory summary once,
rotates it by the worker's starting word, then visits set bits using trailing-bit
extraction. Mapping back modulo 32 preserves cyclic preference, including a
partial final ready word. An explicit width guard still excludes invalid bits.
The two model graphs have five/seventeen ready words and both use this path.

The general segmented scan is extracted unchanged and selected above 1,024
nodes; the original single-ready-word path remains below 33 nodes. Source
reconstruction checks recover the entire original general template exactly.
Only `graph_claim` changes in either model shader. The authoritative word helper,
fallback scan, payload/dependency/epoch operations, summary maintenance, handlers
and allocations remain exact. All nineteen CI shaders and forty ordinary model
shaders remain byte-identical. One snapshot can miss a concurrently published
hint that the old wrapped scan would observe, but hints already permit arbitrary
staleness and cannot authorize waiting without the full fallback.

The host checker passes **645,696 rotation cases**, covering widths 1–32, all
single/two-set-bit masks, invalid advisory bits and varied worker indices. It also
passes 30,948 ready/hint/worker states, 7,295 segmented scan-boundary cases and the
twenty existing publication/clear interleavings. This is not a complete Vulkan
memory-model proof. GPU smoke retains 32/33/65/1024/1025-node boundary coverage,
poisoned storage, independent chains/fan-in and repeated 1/3/256/4096-worker runs.

Full qualification passes **676 library tests (eight ignored), 168 strict
whole-graph oracle cases, 51 matrix regressions, fourteen smoke tests and 52
codegen checks**, plus strict all-target/all-feature Clippy, formatting, release
builds and Metal-target compilation. The qualified rebuild matches the frozen
candidate binaries exactly; Metal GPU execution and additional hardware are not
qualified.

Fifty diagnostics preserve all selection, state, completion, one-dispatch and
no-fallback checks. Forward GPU time changes **4.074432→4.069120 ms (−0.13%)**;
training changes **12.748768→12.671168 ms (−0.61%)**. Ordinary GPU changes stay
within 0.11%. Model registers remain 86/84, shared bytes 18,928/29,344,
participating workers 168, and logical tiles 22,042/83,312. Code shrinks by
640 forward / 768 training bytes. All 256 requested workers remain enabled.

The primary protocol is fixed before collection: **1,000 completed-step
comparisons**, five cases, control/candidate/candidate/control visits, 25 pairs
per case/visit, at least ten warmups and one warmup second. Half use the original
protocol and half add matched 100 ms prewarming. Retention requires both absolute
training intervals below one, no established material regression, complete
qualification and corroborating SDK evidence. No samples are discarded.

| Persistent completed step | Qualified control | Candidate | Candidate/control, pooled bootstrap 95% interval |
|---|---:|---:|---:|
| Original forward | 4.667234 ms | 4.659364 ms | 0.9983 [0.9914, 1.0402] |
| Original training + SGD | 15.388254 ms | 15.278244 ms | 0.9929 [0.9771, 0.9970] |
| Prewarmed forward | 4.482318 ms | 4.393332 ms | 0.9801 [0.9313, 1.0694] |
| Prewarmed training + SGD | 13.158245 ms | 13.097540 ms | 0.9954 [0.9865, 1.0074] |

All numerical/execution checks pass, with maximum model error 4.66e-9. Original
training improves **0.71%**, but the **−0.46%** prewarmed change remains
inconclusive. Normalized training intervals include one under both protocols;
ordinary training shifts +0.73%/+0.37%. Original normalized forward worsens
**4.78% [1.91%, 9.01%]**, while absolute persistent forward is flat and ordinary
forward shifts −2.24%. Prewarmed ordinary forward shifts −6.81% with a broad
interval; all these samples remain included. Every CI persistent absolute and
normalized interval includes one. Pooled intervals omit some between-visit
variation.

The candidate **fails the declared two-protocol criterion and is reverted**.
No timing confirmation or fresh SDK capture is run. Fresh qualified-control
paired persistent/ordinary medians are **1.253× forward / 1.078× training**, or
**1.300× / 1.139×** prewarmed. These are medians of paired ratios, not ratios of
aggregate medians; baseline parity remains unmet.

All twenty source/dependency hashes, both qualified executable hashes and
fourteen restored smoke tests are verified, and both temporary shader files are
removed. Evidence remains in `target/ready-summary-single-scan/`; manifest
`target/megakernel-results/ready-summary-single-scan/summary.json` independently
verifies **414 artifact hashes and 39 completed study cgroups**, SHA-256
`d1b329f9e1a1eacfda16e027485f151c1ee8b2f0b8876961bba16a6cc09ea84d`.
Peak memory is **3,718,868,992 bytes**, with zero memory-limit events. Every heavy
job uses the verified 16/20/1 GiB limits; GPU runs are serialized and measurements
do not overlap builds, copies or heavy analysis. All launched jobs finish, and
both compiler forks remain clean and unchanged.

The following [targeted claims](#targeted-transposed-a-two-tile-claims) test the
remaining short-K transposed-A target, separately from the earlier all-node
four-tile batching experiment.

## Targeted transposed-A two-tile claims

These probes start from `ed2505b`, against qualified runtime `c3ad104` and the
frozen `target/ready-word-summary/final-*` binaries. Admission requires plain
F32 `MatMulAT` with `Kernel::Default`, K in 1–64, at least 256 logical tiles,
no horizontal packing, prologue, epilogue or BF16 storage, and a full physical
scalar workgroup. Per-node metadata keeps small and long-K grids at one tile,
including when they share the batched handler. Arithmetic selection and logical
tile dimensions are unchanged.

The initial guard also rejects `body.subgroups`. The model's scalar output-vote
guard uses subgroups, so that attempt **admits no model batching**. All nineteen
CI and forty-two model shader sources stay identical. Preserve its fifty
diagnostics as an admission control, not evidence for or against model batching.
The revision allows subgroup votes while continuing to exclude cooperative
kernels and smaller logical workgroups.

The captured host plan admits **33 training nodes**: sixteen 2048×720×50,
sixteen 720×4096×50, and one 1440×720×1 product. Their **18,708 logical tiles**
correspond to **9,354 valid two-tile claims**. This count is derived from the
selected host plan and admission predicate, not a hardware claim counter.

The first admitted form loops over the existing tile call. A uniform workgroup
barrier separates iterations before shared scratch is reused. The original
tile initialization, arithmetic and internal barriers remain. The final payload
fence/barrier, clamped completion count, dependency publication, worker totals
and overflow checks retain the existing batch protocol. No new shared storage
or compiler-fork change is introduced.

The second admitted form emits the first call directly and conditionally emits
the second, with the same inter-tile barrier. The condition is uniform and false
for a one-tile claim or odd final tail. It removes the loop while keeping the
same admission, inputs and arithmetic. It also changes compiler code duplication
and register allocation, so this is not a pure measurement of loop overhead.

Each attempt passes release builds, formatting and **52 codegen checks**. The
initial attempt passes the fourteen original whole-graph smoke cases plus a
separately added focused case; each revision passes all fifteen together. The
focused case covers 255/256-tile admission boundaries, an odd 289-tile grid,
K=65 exclusion, dependent consumers and live outputs, poisoned scratch, two
input replays and **1/3/256/4096 workers**.

The host model verifies **8,194 claim partitions and 135,000 logical coordinates**.
Its two-lane scratch model admits twenty-four schedules without an inter-tile
rendezvous and finds eight reuse hazards; all sixteen schedules with the
rendezvous are safe. This finite model is not a complete GPU memory-model proof.

All **150 diagnostic comparisons** pass complete-state, matching-selection,
logical-completion, one-dispatch and no-fallback checks. Maximum model error
remains 4.66e-9. Each attempt uses fresh control/candidate visits, five pairs per
case, ten warmups, at least one warmup second and matched 100 ms prewarming.
All 256 requested workers remain enabled.

For both admitted variants, exactly one case in `graph_main` changes: case 20.
All handler functions, other cases, scheduler/publication functions, nineteen CI
shaders, forty ordinary model shaders and the whole forward shader remain exact.
Reversing the loop/index/activity changes recovers the original case tokens;
the fixed form reconstructs exactly from the audited loop. Replacing the changed
case recovers the entire control source byte-for-byte. A full physical workgroup
allows the case's active-group bound to become one without disabling any lane.

| Admitted form | Training GPU control → candidate | Change | Registers | Training code bytes |
|---|---:|---:|---:|---:|
| Two-tile loop | 12.870688→13.003936 ms | +1.04% | 84→103 | 241,664→248,320 |
| Fixed two calls | 12.764064→12.911488 ms | +1.15% | 84→96 | 241,664→256,000 |

Ordinary training changes +0.11%/+0.16%. Forward GPU changes stay within 0.21%
on identical shaders. Model shared bytes remain 18,928/29,344, participating
workers 168, and logical tiles 22,042/83,312. Register/code growth accompanies
the training regressions but does not isolate their cause. Fewer valid claims
do not establish a performance benefit; batching can also reduce parallelism.

**All three attempts are reverted.** No full library/oracle/Metal qualification,
normal completed-step series or fresh SDK capture is claimed for these rejected
screens. Twenty-one source/dependency hashes, both qualified executable hashes
and fourteen restored smoke tests match the retained runtime. The temporary
focused test is archived with the candidates.

Evidence stays in `target/at-two-tile-{claims,vote-claims,unrolled}/`. Manifest
`target/megakernel-results/at-two-tile-claims/summary.json` independently verifies
**1,020 artifact hashes and 42 completed study cgroups**, SHA-256
`7c7ab7dfcba6e560766c592175e4a575455b12b41cfe99e1ed2d80b0ecba0dc3`.
Peak memory is **3,717,894,144 bytes**, with zero memory-limit events. All heavy
jobs use the verified 16/20/1 GiB limits, GPU work is serialized, and every
launched job finishes. Both compiler forks remain clean and unchanged.

The following private-reservation control preserves the existing single handler
call while testing the same claim reduction. Baseline parity remains unmet.

## Private second-tile reservation

Starting from `4993113`, these probes compare against qualified `c3ad104` and
the frozen `target/ready-word-summary/final-*` binaries. They keep the preceding
admission predicate: **33 short-K scalar AT nodes, 18,708 logical tiles and
9,354 valid two-tile range claims**, derived from the captured host plan.

The leader claims two adjacent tiles and holds the second in two private words.
Each tile executes through the original outer iteration, single handler call,
scratch rendezvous, payload fences and per-tile completion. A metadata flag
selects claim width two while the broadcast execution width remains one. The
overflow bound uses the actual claim width. No shared storage is added and no
compiler-fork API changes.

The leader consumes a reservation before inspecting readiness or waiting on an
idle epoch. Reserved work remains included in the node's unfinished count, so
clearing its ready bit cannot prematurely complete the node or graph. The host
model checks **5,862 states and 14,790 transitions** across 27 one-node cases,
plus sixteen metadata masks. Ownership never overlaps completed work, reserved
tiles survive readiness exhaustion, and all modeled nonterminal states can
advance. This finite model is not a complete Vulkan memory-model proof.

The initial form checks the private slot inside the idle-poll loop. A second
build disables the runtime metadata flag while retaining identical shaders,
separating the policy from the added code. A third form checks the slot once at
function entry, outside idle polling. This is valid because a failed scan cannot
create a reservation and a successful claim immediately exits the scan. The
single-word claim path already checks once at entry.

Each build passes formatting, release builds, **52 codegen checks and fifteen
whole-graph smoke tests**. The focused reservation case covers 255/256-tile
admission, a 289-tile odd tail, K=65 exclusion, dependent consumers, poisoned
scratch and two input replays at **1/3/256/4096 workers**. It is expanded from
eight to forty-one nodes to exercise both single-word and summary claim paths;
the initial candidate passes that expansion separately. The disabled control
does not exercise consumption of reserved tiles.

All **175 diagnostic comparisons** pass state, selection, logical completion,
one-dispatch and no-fallback checks. Maximum model error remains 4.66e-9.
Only training's `graph_claim_word` and `graph_claim` change, with one added
`graph_take_reserved` helper. Exact source inverses preserve `graph_main`, every
case, all handlers, completion and publication. All nineteen CI shaders, forty
ordinary model shaders and the forward graph remain exact. Enabled and disabled
reservations have all **61 shader sources** in common.

| Diagnostic training visit | Ordinary GPU ms | Persistent GPU ms |
|---|---:|---:|
| Initial control → reservation | 10.753056→10.748736 | 12.705536→12.917888 (+1.67%) |
| Fresh control / disabled / enabled | 10.757536 / 10.757440 / 10.765408 | 12.872736 / 12.786048 / 12.893376 |
| Entry-check control → candidate | 10.765504→10.780352 | 12.751104→12.983776 (+1.82%) |

Enabled reservations are 0.84% slower than disabled metadata in the middle
comparison. These short sequential visits do not isolate a precise causal
effect; the fresh enabled/control difference is only 0.16%. Moving the check
out of idle polling does not recover performance. All three builds use training
registers **84→86** and code bytes **241,664→243,072**. Model shared bytes,
logical tiles and 168 participating workers remain unchanged, with all 256
requested workers enabled. Forward GPU changes stay within 0.24% on identical
shaders.

Four matched SDK backend captures collect **sixteen metric groups** for the
initial active reservation. Persistent training replay changes
**13.079270→13.111235 ms (+0.24%)**, versus ordinary
**11.679766→11.719162 ms (+0.34%)**. Estimated global/L2 atomic sectors fall
**3.51%/3.73%**, but warp instructions rise **1.18%**. Integer instructions and
shared traffic stay within 0.14%. Barrier waiting stays at **45.32% of active
warp cycles**, active compute warps move 32.60→32.73%, and persistent local
traffic remains zero. Clock/instruction-rate group spreads stay below 0.42%.
These counters establish no useful replay gain or exact bottleneck attribution;
rates times duration are estimates and barrier shares are not elapsed time.

All four actual exported states contain **246,883,105 floats** and match
bit-for-bit between control and candidate within each backend. Cross-backend
training passes tolerance but is not bit-identical. SDK training uses zero-rate
SGD for stable replay, separately from the normal nonzero-rate diagnostics.

**All three builds are reverted.** Twenty-one source/dependency hashes, both
qualified executable hashes and fourteen restored smoke tests match the retained
runtime. No full library/oracle/Metal qualification or normal completed-step
series is claimed for these rejected screens. SDK evidence covers only the
initial active form. Both compiler forks remain clean and unchanged.

Evidence stays in `target/at-private-tile-reservation/` and
`target/at-private-reservation-{disabled,entry}/`. Manifest
`target/megakernel-results/at-private-reservation/summary.json` independently
verifies **1,228 artifact hashes and 46 completed study cgroups**, SHA-256
`bdec68fe9e02f0d3b9f3f8841b68112a0ef610a1b53e84912fe63a07487b2ce7`.
Peak study memory is **4,716,912,640 bytes**, with zero memory-limit events.
Every heavy job uses a separate verified 16/20/1 GiB cgroup, GPU jobs are
serialized, and artifacts remain on local `/x` storage. All launched jobs finish.

The following control isolates a final-stage synchronization opportunity without
changing claim policy. Keep the existing completed-step gates and qualified
runtime as the control.

## Final scalar AT stage barrier

This probe starts from `542c69c`, against qualified `c3ad104` and its frozen
benchmark binaries. It omits the scratch-reuse barrier after the final K stage
of plain F32, 64×64 scalar AT products without addends, epilogues or split-K.
Both ordinary and persistent execution use the same change. Existing horizontal
batches remain eligible: their uniform entry selects exactly one of two bodies.
The captured plan identifies **82 affected nodes**, including sixteen horizontal
nodes, totaling **25,972 logical tiles**. These are host-derived barrier
opportunities, not measured native barrier instructions.

The original pre-reduction barrier orders staged writes before reads. The
uniform condition `end_k - t > K_TILE` retains the reuse barrier whenever another
K stage will write scratch. After the final shared read, only register/output
work remains. Ordinary execution ends; persistent execution retains its outer
workgroup rendezvous before scratch can be reused. This ordering uses the
[WGSL workgroup memory semantics](https://www.w3.org/TR/WGSL/#memory-semantics).
Converter initialization, payload fences, completion, publication and queue
policy remain unchanged. No compiler-fork change is introduced.

A finite two-/three-lane scratch model checks **7,878 states and 17,016
transitions** across 32 ordinary/persistent cases with up to four K stages and
three successive tiles. All retain correct staged values. Negative controls
that remove an intermediate reuse barrier or the outer rendezvous produce
four/two hazards. Another **804 bound cases** verify the last-stage predicate,
including empty K and unsigned boundary arithmetic. This is not a complete
Vulkan memory-model proof or authorization for allocations beyond runtime limits.

Release builds, formatting, **52 codegen checks and fifteen whole-graph smoke
tests** pass. The focused test uses K=1/31/32/33/50/64/65/97, output edges,
dependent consumers, live intermediate outputs, poisoned scratch and two exact
dyadic-input replays. It tests sixteen-/forty-nine-node graphs under ordinary
execution and **1/3/256/4096 workers**. An initial passing implementation also
allowed input-free epilogues; the scope was tightened before measurement and
all checks rerun.

All **fifty diagnostics** pass selection, complete-state, logical-work,
one-dispatch and no-fallback gates. Two ordinary shader sources change; reversing
the barrier condition restores each exactly. Training changes only handler 20
and the two internal horizontal bodies of handler 29. Removing the condition
and alpha-renaming Naga's expression temporaries recovers every staging,
arithmetic, initialization and store statement. All other functions, cases,
graph entry, nineteen CI shaders and the forward graph remain byte-exact.
The first auditor incorrectly expected only one changed function; its script
and failed log remain preserved. The corrected audit covers all three without
rerunning GPU work.

Training GPU medians are **12.746464→12.732512 ms (−0.11%)**, versus ordinary
**10.748576→10.751072 ms (+0.02%)**. Forward changes
**4.073280→4.079136 ms (+0.14%)** on an unchanged shader. Training retains
**84 registers and 29,344 shared bytes**; code grows 241,664→241,792 bytes.
Both phases retain 168 participating workers, all 256 requested workers, and
22,042/83,312 logical tiles. Buffer allocations and scheduler sizes are unchanged.
These short sequential visits establish no useful gain.

Four matched SDK backend captures collect **sixteen metric groups**. Persistent
training replay changes **13.023640→13.043179 ms (+0.15%)**, versus ordinary
**11.675162→11.679551 ms (+0.04%)**. Persistent estimated warp instructions,
integer instructions, global loads and shared traffic each move less than 0.12%.
Barrier waiting changes **45.20→45.26% of active warp cycles**, active compute
warps stay near 32.6%, and persistent local traffic remains zero. Ordinary
barrier share falls 6.91→6.65% without a replay gain. Clock/instruction-rate group
spreads stay below 0.43%/0.54%. Counts are rate-times-duration estimates; barrier
shares do not isolate elapsed-time costs or prove native instruction counts.

All four actual exported states contain **246,883,105 floats** and match
bit-for-bit between control and candidate within each backend. Cross-backend
training passes tolerance but is not bit-identical. SDK training uses zero-rate
SGD for replay stability, separately from normal nonzero-rate diagnostics.

**The probe is reverted.** Twenty-one source/dependency hashes, both qualified
binary hashes and fourteen restored smoke tests match the retained runtime.
No full library/oracle/Metal qualification or normal completed-step series is
claimed. Both compiler forks remain clean and unchanged.

Evidence is under `target/at-final-stage-barrier/`. Manifest
`target/megakernel-results/at-final-stage-barrier/summary.json` independently
verifies **448 artifact hashes and 26 completed study cgroups**, including the
preserved auditor failure, SHA-256
`e86974a534b7f72a4a3885a11745b192f98def9dff9837aa3b0a78d92321d388`.
Peak study memory is **4,698,873,856 bytes**, with zero memory-limit events.
Every heavy job uses a separate verified 16/20/1 GiB cgroup; GPU jobs serialize,
and artifacts remain on local `/x`. All launched jobs finish.

Next, sample individual AT tile entry/exit times, storing each timestamp
immediately so no clock value stays live across the handler. Calibrate overhead
and resource changes against fresh uninstrumented controls. The existing node
spans include gaps between claims; individual tile intervals can distinguish
that effect from slow tile execution. Keep every numerical and completed-step
gate, and do not sum overlapping intervals into purported recoverable time.

## Individual AT tile clocks

This attribution study starts from `09d8014`, using qualified `c3ad104` and its
frozen binaries as controls. It samples the first, middle and last node of each
eligible scalar AT shape, or all nodes when fewer than three exist. It records
**every tile of twenty selected nodes: 5,466 intervals per trace**, including
horizontal batches and K=1 tails. Sampling complete nodes avoids choosing only
particular column or edge coordinates. These are a subset of the 82 eligible
nodes and 25,972 tiles, not the entire training workload.

The leader writes entry and exit timestamps directly to unique queue slots;
no timestamp remains live across the handler. Entry follows parameter broadcast
and the existing payload acquire. Exit follows the original payload fence and
workgroup barrier, before completion/publication. Thus a recorded interval
includes handler initialization, arithmetic and those final synchronizations,
plus instrumentation overhead. Gaps between a worker's successive tiles also
include scheduling, parameter broadcast, publication and potentially other
nodes. They are not measurements of idle polling alone.

A temporary Blade hook replaces one marker call in the timestamp helper with
device-scope `OpReadClockKHR` after Naga validation. Both SPIR-V versions pass
`spirv-val --target-env vulkan1.3`; reversing the transformation recovers the
entire original module byte-for-byte. Naga source and API remain unchanged.
Reversing four clock call sites and the helper definitions similarly recovers
the complete training WGSL. Every handler, case, claim, completion and publication
statement remains exact. All nineteen CI sources, forty ordinary model sources
and the forward graph remain byte-identical.

The [shader-clock specification](https://raw.githubusercontent.com/KhronosGroup/SPIRV-Registry/main/extensions/KHR/SPV_KHR_shader_clock.asciidoc)
provides a shared source clock at device scope, but guarantees neither a fixed
rate nor code-motion barriers. Raw ticks are preserved. The microseconds below
are approximate, scaled by each final step's GPU timestamp interval divided by
its worker clock envelope. The two traces measure 12.893088/12.893184 ms, with
envelope ratios 0.999804/0.999725 ticks per timestamp nanosecond. Neither this
close agreement nor unchanged register counts makes the probe free of effects.

**Two hundred paired diagnostics** use control/probe/probe/control visits,
ten pairs per case/visit, ten warmups and at least one warmup second, with
100 ms prewarming and 256 requested workers. All full-state, selection,
logical-work, one-dispatch and no-fallback gates pass. Eight preflight pairs,
five graph-codegen checks and fourteen smoke tests also pass. The two final
training steps supply **10,932 tile intervals**; the trace file is overwritten
per process, so these are not traces of every timing sample.

Training GPU medians change **12.762896→12.922816 ms (+1.25%)**, while ordinary
training is effectively unchanged at 10.769344→10.769280 ms. Forward GPU time
changes 4.077728→4.080608 ms (+0.07%) on an unchanged shader. Training retains
**84 registers, 29,344 shared bytes and 168 participating workers**. The probe
adds 2,560 native code bytes and 231,048 scheduler/resident bytes. All requested
workers remain enabled; graph work stays 22,042/83,312 tiles for forward/training,
with 97,522 additional optimizer tiles in training.
This is instrumentation calibration, not a completed-step optimization gate.

The table uses medians over selected node observations in both final traces.
“Overlap” is the sum of tile intervals divided by their node envelope; it is
average concurrent recorded intervals, not occupancy or resident workgroups.
“Body share” divides recorded intervals by the sum of each participating
worker's first-to-last span for that node. A single-tile worker has no internal
gap and contributes a body share of one.

| AT shape | Tiles/node | Ordinary dispatch µs | Persistent node span µs | Tile median µs | Overlap | Body share |
|---|---:|---:|---:|---:|---:|---:|
| 2048×720×50 | 384 | 22.53 | 51.44 | 6.49 | 53.1 | 80.7% |
| 720×4096×50 | 768 | 36.83 | 98.12 | 6.35 | 54.3 | 72.6% |
| 960×720×50 | 180 | 14.37 | 22.02 | 7.07 | 62.1 | 85.6% |
| 720×960×50 | 180 | 14.34 | 23.22 | 7.00 | 55.8 | 83.2% |
| 320×320×16, two products | 50 | 6.14 | 7.67 | 4.39 | 32.7 | 100% |
| 720×320×50, two products | 120 | 10.24 | 10.83 | 7.14 | 89.4 | 100% |
| 720×720×1 | 144 | 8.19 | 9.30 | 4.55 | 71.5 | 100% |
| 1440×720×1 | 276 | 12.32 | 16.37 | 4.18 | 73.2 | 76.6% |

The fresh ordinary split-pass profile preserves complete state and matching
implementations. Its timestamp boundaries and overhead differ from persistent
clocks; ordinary backward also excludes SGD. These columns locate the gap but
do not establish an exact same-instrumentation kernel ratio. Ordinary AT uses
92 registers, or 94 for horizontal batches, and 16,768 shared bytes.

For 720×4096×50, the median sampled node uses only 78 distinct workers and has
about 54 overlapping tile intervals on average. Its longest worker's accumulated
tile time covers about 74% of the node envelope. This supports investigating
work distribution and concurrent work, but does not show that the remaining
workers are idle or that the gaps are recoverable. In particular, neither
overlapping node spans nor worker intervals may be summed into elapsed savings.

**The probe is removed.** Twenty-one main source/dependency hashes, three Blade
source hashes, both qualified binary hashes and fourteen restored smoke tests
match. Both compiler forks are clean. No full library/oracle/Metal qualification,
normal completed-step series or fresh SDK counters are claimed for this study.

Evidence is under `target/clock-at-tiles/`. Manifest
`target/megakernel-results/clock-at-tiles/summary.json` independently verifies
**371 artifact hashes and 27 completed study cgroups**, SHA-256
`1ee03f8a5b7f32c6d3f5b90c8199ecef009ed817de6ca07ff79ec6246f2d8b5d`.
Peak job memory is **4,247,101,440 bytes**, with zero memory-limit events. Every
heavy job runs in a separate verified 16/20/1 GiB cgroup, GPU jobs serialize,
and artifacts stay on local `/x`. All launched jobs finish.

Next, extend the same metadata-selected recording to optimizer claims to
identify which work overlaps the long AT spans and intervening gaps. Account
for existing four-tile optimizer claims and calibrate the extra recording cost
before drawing conclusions. Keep handler arithmetic, requested workers and all
completed-step gates. Do not repeat global optimizer deferral or alter queue
priority solely from these partial traces.

## Optimizer claims alongside AT tiles

This study starts from `3f3aa9c`, against qualified `c3ad104`. It extends the
previous clock metadata to the four SGD nodes, without changing the timestamp
helper or any handler, case, scheduling, completion or publication statement.
The optimizer's **97,522 logical tiles form 24,382 claims of up to four tiles**.
The existing 83,312 training graph tiles are additional; the benchmark reports
graph and appended-command worker counters separately.

Two instrumented modes reserve identical timestamp storage. AT-only recording
leaves the optimizer table pointers zero; combined recording enables those
pointers. Each claim writes at its first logical tile's slot, leaving other
slots zero. The decoder verifies every unused slot, all claim owners against
the original per-node worker histograms, and both partial final SGD claims.
It also verifies nonoverlapping intervals across all recorded nodes on each
worker. Both modes preserve all 256 requested workers.

All **61 shader sources match between recording modes** and the previous AT
probe. Reversing the four clock call sites and helper definitions restores the
complete original training WGSL. CI and forward remain uninstrumented. The two
captured SPIR-V modules differ only in one `OpString` debug source-file path;
every other instruction is exact. Both modules and their pre-clock originals
validate, and inverse clock transformation restores each original byte-for-byte.
An auxiliary audit first assumed an `OpName` difference; its script and failed
log remain preserved alongside the corrected audit. No GPU work was rerun.

**Three hundred diagnostics** use qualified/AT-only/combined/combined/AT-only/
qualified visits, ten pairs per case/visit, ten warmups and at least one warmup
second, 100 ms prewarming and 256 workers. All complete-state, matching-selection,
logical-work, one-dispatch and no-fallback gates pass. Twelve preflight pairs,
five graph-codegen checks and fourteen smoke tests also pass.

| Training GPU diagnostic | Qualified | AT-only | Combined |
|---|---:|---:|---:|
| Persistent | 12.810240 ms | 12.930128 ms | 12.946784 ms |
| Ordinary | 10.779280 ms | 10.773040 ms | 10.772608 ms |

Combined recording costs **0.13% over AT-only / 1.07% over qualified** in these
pooled GPU medians. All training modes retain **84 registers, 29,344 shared
bytes and 168 participating graph workers**. Both instrumented modes use
244,224 code bytes and 5,302,712 scheduler bytes, respectively 2,560 and
4,131,928 above control. Resident bytes rise by the same metadata delta.
This calibrates the probe; it establishes
no runtime improvement or normal completed-step result.

Two final steps per instrumented mode yield **70,628 recorded intervals**:
10,932 AT-only intervals and 59,696 combined intervals. Each combined step
covers all 24,382 SGD claims and all 5,466 tiles of the twenty selected AT nodes.
These are final-step traces, not traces of every timing sample. Raw ticks,
worker ownership, unused slots and all per-node intervals remain archived.

No recorded SGD body overlaps **any of the six observations per shape**
for AT2048×720×50 or AT720×4096×50. Their combined-mode median spans remain
51.60/98.78 µs, with approximately 54.6/53.6 simultaneous recorded AT intervals.
The larger shape still uses a median of 78 distinct workers. SGD therefore
does not directly occupy the workers during these sampled long AT spans.

Some other shapes do overlap SGD: four of six AT960×720×50 observations,
two of six AT720×960×50 observations, two of six horizontal AT720×320×50
observations, and both observations of each K=1 tail. Even there, **zero recorded
SGD-body ticks occur inside any same-worker gap between sampled AT claims**.
This separates simultaneous work on other workers from work interposed between
a given worker's AT claims. It does not identify unrecorded graph handlers or
the scheduler work outside recorded bodies.

The four SGD nodes have these observed ranges across the two combined traces:

| Node | Claims | Node envelope µs | Median four-tile claim µs | Mean overlapping claims |
|---|---:|---:|---:|---:|
| 523 | 8,180 | 670.74–671.54 | 11.30–11.33 | 136.74–136.93 |
| 524 | 7,562 | 1,442.57–1,443.82 | 10.95 | 57.44–57.49 |
| 525 | 7,920 | 1,229.07–1,248.02 | 10.02–10.05 | 61.18–62.15 |
| 526 | 720 | 38.06–39.75 | 3.91 | 74.08–77.29 |

The same [clock limitations](#individual-at-tile-clocks) apply: entry/exit
intervals include handler initialization, the original payload fence/barrier
and recording overhead; clock rate and code-motion barriers are not guaranteed.
Scaled microseconds use each step's GPU interval and worker envelope. Small
pooled overhead does not eliminate per-node perturbation: the final K=1 node's
median envelope differs 11.6% between modes. Final AT-only steps measure
12.812480/13.173120 ms, versus combined 12.813408/12.843488 ms. All are preserved.
Overlapping intervals cannot be summed into recoverable elapsed time.

**The probe is removed.** Twenty-one main source/dependency hashes, three Blade
source hashes, both qualified binary hashes and fourteen restored smoke tests
match. Both compiler forks are clean. No full library/oracle/Metal qualification,
normal completed-step series or fresh SDK counters are claimed.

Evidence is under `target/clock-at-optimizer/`. Manifest
`target/megakernel-results/clock-at-optimizer/summary.json` independently verifies
**504 artifact hashes and 35 completed study cgroups**, including the preserved
auxiliary audit failure, SHA-256
`691a7e9b96bc87989d83235ff268a56b30c2de04c93ca0cc93cbbe50353f6b6d`.
Peak job memory is **3,720,683,520 bytes**, with zero memory-limit events. Every
heavy job uses a separate verified 16/20/1 GiB cgroup; GPU jobs serialize,
artifacts remain on local `/x`, and all launched jobs finish.

Next, extend metadata-selected claim recording to the remaining graph handlers,
including packed-axis claims and partial completion. Compare identical shaders
and reserved storage with recording disabled for those nodes, then identify
which handlers overlap the long AT spans and same-worker gaps. Preserve all
arithmetic, fences, requested workers and completed-step gates. Direct optimizer
competition is unsupported for the largest sampled AT shapes, so these traces
provide no reason to repeat global SGD deferral or alter its eligibility.

## All graph claim attribution

This study starts from `137161c`, against qualified `c3ad104`. It extends clock
metadata to every nonempty node of the training graph, including packed-axis
claims and partial completion. The shader, timestamp helper, arithmetic,
scheduling and synchronization remain identical to the preceding probe.
Sparse recording selects the same twenty AT nodes and all four SGD nodes;
full recording enables every node. Both modes reserve identical storage,
including unused slots for packed-axis padding. CI and standalone forward
remain uninstrumented.

A finite decoder audit checks **3,648 geometry cases, 94,336 claims and 162,000
logical coordinates**, including 19,134 partial claims and empty grids. It
independently enumerates coordinates and requires exact coverage without
duplicates. Actual traces verify padded limits, logical completion counts,
every claim owner against the original per-node worker histograms, all unused
slots and cross-node interval ordering on each worker. Full recording covers
**527 nodes and 94,238 claims per step**: 83,312 graph tiles and 97,522 optimizer
tiles, with 183,218 allocated claim slots including padding.

All **61 shader sources match between recording modes** and the preceding
probe. Reversing clock call sites and helpers recovers the complete original
training source. Both captured SPIR-V variants validate alongside their
pre-clock originals, and inverse transformation restores each original exactly.
Removing debug instructions makes the two variants byte-identical. No Naga
source or API change is introduced.

**Three hundred diagnostics** use qualified/sparse/full/full/sparse/qualified
visits, ten pairs per case/visit, ten warmups and at least one warmup second,
100 ms prewarming and 256 requested workers. All complete-state, selection,
logical-work, one-dispatch and no-fallback gates pass. Twelve preflight pairs,
five graph-codegen checks and fourteen smoke tests also pass.

| Training GPU diagnostic | Qualified | Sparse | Full |
|---|---:|---:|---:|
| Persistent | 12.809296 ms | 13.040096 ms | 13.066480 ms |
| Ordinary | 10.782416 ms | 10.776304 ms | 10.775248 ms |

Full recording costs **0.20% over sparse / 2.01% over qualified** in pooled
GPU medians. The short preflight instead measures +7.01% over sparse; all
samples are retained without extra repeats. Training retains **84 registers,
29,344 shared bytes and 168 participating graph workers**. Both instrumented
modes use 244,224 code bytes and 8,511,912 scheduler bytes: respectively 2,560
and 7,341,128 above control. Resident bytes rise by the same metadata delta.
All requested workers remain enabled. This is instrumentation calibration,
not a normal completed-step improvement.

Two final steps per recording mode provide **248,172 intervals**: 188,476 full
and 59,696 sparse. Full-step traces measure **14.859520 / 12.895904 ms**, versus
sparse **12.848320 / 12.935520 ms**. The longer first full trace is preserved.
Low pooled overhead does not imply unperturbed individual traces. Raw ticks
and all worker intervals remain available; approximate microseconds use each
trace's timestamp interval and worker envelope. The existing shader-clock
rate and code-motion limitations still apply.

The missing concurrency is predominantly the preceding transposed-B gradient:

| Sampled AT shape | Mean simultaneous AT bodies | Other bodies | All bodies | Other-body share of same-worker AT gaps |
|---|---:|---:|---:|---:|
| 2048×720×50 | 52.4 | 78.0 | 132.6 | 0% |
| 720×4096×50 | 54.0 | 90.0 | 143.9 | 1.68% |

These are medians over six observations per shape; separately aggregated
columns need not add exactly. During AT720×4096×50, the preceding
BT50×720×4096 accounts for nearly all concurrent bodies on other workers.
The 1.68% of internal AT gap time occupied by another body comes from one
horizontal AT node in one trace. Most gaps contain no other recorded body:
they include claim handling, parameter broadcast, completion/publication,
waiting and instrumentation. Body concurrency is not hardware occupancy, and
unrecorded worker lifetime is not synonymous with GPU idle time.

A fresh ordinary profile preserves complete state and matching implementation
keys. Adjacent AT/BT nodes with matching matrix dimensions give this comparison:

| AT shape | Matched pairs | Ordinary BT+AT interval sum µs | Persistent pair envelope µs | AT finishes before BT |
|---|---:|---:|---:|---:|
| 2048×720×50 | 13 | 65.54 | 58.80 | 0/26 observations |
| 720×4096×50 | 16 | 186.35 | 159.17 | 32/32 |
| 960×720×50 | 13 | 51.23 | 35.34 | 26/26 |
| 720×960×50 | 9 | 55.10 | 44.70 | 18/18 |

The 720×4096×50 AT node spans about 98.20 µs, yet always finishes before the
paired BT node. Its apparent per-node regression does not establish a loss
for the combined work. Ordinary split-pass timestamps and persistent clock
envelopes have different overhead; these are attribution comparisons, not an
exact speedup claim or a full graph critical-path proof. Pair envelopes also
overlap other work and cannot be summed into recoverable whole-step time.

The trace identifies a separate scheduling-cost candidate: **sixteen explicit
SwiGLUConcat and sixteen SwiGLUConcatGrad nodes each execute 400 single-tile
claims**. Median recorded bodies are approximately 0.70/0.99 µs; median node
spans are 7.39/12.34 µs, versus ordinary intervals of 4.10/6.30 µs. Their F32
source uses independent element writes and no workgroup scratch or barriers.
Earlier [pointwise-batching probes](#scratch-clearing-and-pointwise-batching-probes)
admitted only generated pointwise bodies, whose largest model grid was 141
tiles; neither their 512 nor 1,024 threshold selected model work. Those controls
do not test these explicit 400-tile handlers.

**The probe is removed.** Twenty-one main source/dependency hashes, three Blade
source hashes, both qualified binary hashes and fourteen restored smoke tests
match. Both compiler forks are clean. No full library/oracle/Metal qualification,
normal completed-step series or fresh SDK counters are claimed.

Evidence is under `target/clock-all-claims/`. Manifest
`target/megakernel-results/clock-all-claims/summary.json` independently verifies
**515 artifact hashes and 37 completed study cgroups**, SHA-256
`55a3aaa0b8cf24bdaa250f13ecbdff5011a44c2a795d0f38c6262ed5f197b605`.
Peak job memory is **4,270,448,640 bytes**, with zero memory-limit events.
Every heavy job uses a separate verified 16/20/1 GiB cgroup; GPU jobs serialize,
artifacts remain on local `/x`, and all launched jobs finish.

Next, test two-tile sequential claims for large explicit F32 SwiGLU forward/
gradient grids, preserving the selected per-element arithmetic, independent
writes, clamped completion and small-grid behavior. Verify admission on actual
model nodes and partial claims, then check native resources and all five cases
before normal completed-step qualification. Keep SGD batching unchanged. Do not
prioritize AT based only on its longer span while useful BT work overlaps it.

## Two-tile SwiGLU claims

Starting from `f0de2f2` and qualified runtime `c3ad104`, this candidate admits
only explicit, unfused F32 `SwiGLUConcat` and `SwiGLUConcatGrad` handlers. Grids
with at least 256 logical tiles claim two sequential tiles; smaller grids keep
one, including those sharing a cached handler with a larger grid. Existing
four-tile SGD batching remains unchanged. No Naga/Blade source, arithmetic,
payload fence, shared GPU state or queue allocation changes.

The source audit verifies 19 CI and 40 ordinary model sources byte-for-byte.
Every callable arithmetic handler remains exact. Only one forward and two
training dispatch cases acquire the existing sequential-repeat wrapper;
reversing those cases restores both complete original shaders exactly. All
sixteen forward and sixteen gradient nodes have 400 logical tiles. Admission
therefore saves 3,200 forward / 6,400 training claims per step by host-plan
calculation; these are not hardware atomic counts.

The new replay test combines nine grid sizes: **1, 3, 255, 256, 257, 400, 401,
1,024 and 1,025 tiles**, with element tails, forward/gradient consumers and
small/large nodes sharing cached handlers. Two changed-input replays produce
bit-exact ordinary/persistent outputs with **1/3/256/4,096 workers**. A finite
coverage/completion audit enumerates **2,050 tile counts, 1,066,881 claims and
2,100,225 logical coordinates**; 21 gradient layouts verify 39,908 independent
writes. This supplements the existing publication contract, not a complete
Vulkan memory-model proof.

Qualification passes **677 library tests (eight ignored), 168 strict oracle
cases, 51 matrix regressions, 53 codegen checks and fifteen whole-graph smoke
tests**, plus Clippy with warnings denied, Metal-target compilation, formatting
and release builds. Clippy initially rejected an implicit reference pattern;
the corrected spelling produces byte-identical frozen release binaries and
passes fifteen post-fix smoke tests. The first source-plan auditor also failed
because repeated forward plans lacked a matching phase banner. The corrected
parser uses actual dispatch-zero plan starts and requires four complete,
matching plans. Both failures and original scripts/sources remain preserved.
The initial manifest auditor also incorrectly required a GPU lock for CPU analysis; its
corrected job classification and original failure are preserved. No GPU
measurements were rerun for these corrections.

All **50 diagnostics** preserve complete-state budgets, selected implementations,
logical work, requested workers, one dispatch and no fallback. Payload,
scheduler and resident byte counts are unchanged. Both model phases retain
168 participating graph workers, with all 256 requested workers enabled.

| Model diagnostic | Qualified | Candidate | Change |
|---|---:|---:|---:|
| Forward persistent GPU | 4.070432 ms | 4.044128 ms | −0.65% |
| Training persistent GPU | 12.727744 ms | 12.644640 ms | −0.65% |
| Forward registers / shared / code bytes | 86 / 18,928 / 102,400 | 86 / 18,928 / 102,528 | +128 code bytes |
| Training registers / shared / code bytes | 84 / 29,344 / 241,664 | 83 / 29,344 / 241,920 | −1 register, +256 code bytes |

The predeclared normal gate uses **1,000 completed-step comparisons**:
control/candidate/candidate/control, 25 pairs per case/visit, ten warmups and
at least one warmup second. Half use the original protocol; half add matched
100 ms prewarming. Every state, selection and completion check passes.

| Persistent completed step | Qualified | Candidate | Candidate/control, pooled bootstrap 95% interval |
|---|---:|---:|---:|
| Original forward | 4.663694 ms | 4.639423 ms | 0.9948 [0.9876, 1.0575] |
| Original training + SGD | 15.372354 ms | 15.239828 ms | 0.9914 [0.9747, 1.0019] |
| Prewarmed forward | 4.365727 ms | 4.331816 ms | 0.9922 [0.9300, 1.0644] |
| Prewarmed training + SGD | 13.205901 ms | 13.162524 ms | 0.9967 [0.9832, 1.0133] |

**All four model absolute intervals include no change.** Neither phase meets
the requirement for an absolute completed-step gain under both protocols.
There is no significant persistent regression in another model or CI case.
Original ordinary training moves +2.48%, and prewarmed ordinary training
+1.47%; this makes the normalized training ratios look more favorable than
the absolute persistent changes. Unchanged CI shaders also show timing
variation. Pooled intervals omit some between-visit variation.

**The candidate is reverted.** No extra primary repeats or fresh SDK counters
are collected after the failed normal gate. Twenty-two source/dependency
hashes, both qualified binary hashes and fourteen restored smoke tests match;
both compiler forks remain clean. Qualified runtime remains `c3ad104`, and
baseline parity remains unmet.

Evidence is under `target/swiglu-two-tile/`. Manifest
`target/megakernel-results/swiglu-two-tile/summary.json` independently verifies
**443 artifact hashes and 45 completed study cgroups**, SHA-256
`f1fdddd47caf2976456ee20738887d6de32fae336a92cf8fdc6bcae58cad8c23`.
Peak job memory is **3,714,166,784 bytes**, with zero memory-limit events.
Every heavy job uses a separate verified 16/20/1 GiB cgroup; GPU jobs serialize,
builds and heavy analysis stay outside timing, and all artifacts remain on `/x`.
All launched jobs finish.

A next control can vary the admitted claim width to four while requiring the
same generated shader and native resources as this two-tile candidate. That
reduces available claims from 200 to 100 per 400-tile node, so any saved queue
work must be weighed against reduced parallelism. Verify partial completion,
changed-input replay and all five diagnostic cases before a new fixed normal
gate. Keep the acceptance criterion and requested worker count unchanged.

## Four-tile SwiGLU metadata control

Starting from `778f5be`, this study compares qualified `c3ad104`, the archived
two-tile SwiGLU candidate and a four-tile candidate. The only production-source
change from two to four is the admitted claim width. Admission remains limited
to explicit unfused F32 SwiGLUConcat/Grad grids with at least 256 logical tiles;
smaller nodes and existing SGD batching retain their previous behavior.

All **61 generated shader sources match byte-for-byte between two and four**,
and every reported native pipeline statistic matches across both visits.
Only claim-width metadata changes between these runtime policies: each
admitted 400-tile node offers 100 four-tile claims instead of 200 two-tile claims. Against the
qualified single-tile path, host-plan calculation saves **4,800 forward /
9,600 training claims per step**. These are not hardware atomic counts.
Reversing the existing repeat wrappers still recovers the complete qualified
shaders, including all arithmetic and synchronization. Allocations do not grow.

Replay coverage expands to **thirteen grid sizes**: 1, 3, 255–259, 400–403,
1,024 and 1,025 tiles. This includes every partial four-tile remainder, element
tails, and small/large grids sharing cached handlers. All 52 graph operations
complete in one dispatch; two changed-input replays match ordinary outputs
bit-for-bit with **1/3/256/4,096 workers**. The finite audit checks **2,050 tile
counts, 550,209 claims, 2,100,225 logical coordinates and 39,908 independent
gradient writes**. Existing overflow guards use the actual per-node width.
This is not a complete Vulkan memory-model proof.

The diagnostic protocol is fixed before measurement: **150 comparisons**,
qualified/two/four/four/two/qualified, five pairs per case/visit, ten warmups,
at least one warmup second and matched 100 ms prewarming. All complete-state,
selection, logical-work, one-dispatch and no-fallback checks pass. Every
requested worker remains enabled; 168 participate in each model graph.

| Persistent GPU diagnostic | Qualified | Two tiles | Four tiles |
|---|---:|---:|---:|
| Forward | 4.082928 ms | 4.058848 ms | 4.014400 ms |
| Training + SGD | 12.747248 ms | 12.823504 ms | 12.790784 ms |

Four tiles improve forward **1.68% versus qualified / 1.10% versus two**; both
visits show the forward improvement. Training changes **+0.34% versus qualified
/ −0.26% versus two**. The fresh two-tile training control is +0.60% versus
qualified, unlike its earlier −0.65% diagnostic result. Both studies and all
visits remain preserved. Forward meets the predeclared advancement gate;
diagnostics do not establish a completed-step gain.

Both batched variants use **86 registers / 18,928 shared bytes / 102,528 code
bytes** forward, and **83 / 29,344 / 241,920** training. Relative to qualified,
forward adds 128 code bytes; training removes one register and adds 256 code
bytes. Payload, scheduler and resident bytes remain unchanged.

Full qualification passes **677 library tests (eight ignored), 168 strict
oracle cases, 51 matrix regressions, 53 codegen checks and fifteen whole-graph
smoke tests**, plus strict Clippy, Metal-target compilation, formatting and
release builds. Both rebuilt binaries match the frozen candidate exactly.

The subsequent **1,000 completed-step comparisons** use the unchanged
control/candidate/candidate/control sequence, 25 interleaved pairs per case/
visit, ten warmups and at least one warmup second. Half use the original
protocol; half use matched 100 ms prewarming. Every numerical and execution
gate passes. A separate audit covers all **1,150 diagnostic/normal comparisons
and 70 case visits**: optimizer implementation keys, grids and logical task
counts match ordinary and persistent execution in every mode. Training retains
all four SGD chunks and **97,522 optimizer tiles** per step.

| Persistent completed step | Qualified | Four tiles | Candidate/control, pooled bootstrap 95% interval |
|---|---:|---:|---:|
| Original forward | 4.671153 ms | 4.595394 ms | 0.9838 [0.9235, 0.9973] |
| Original training + SGD | 15.345569 ms | 15.163832 ms | 0.9882 [0.9746, 1.0017] |
| Prewarmed forward | 4.363912 ms | 4.592538 ms | 1.0524 [0.9803, 1.0592] |
| Prewarmed training + SGD | 13.242206 ms | 13.054930 ms | 0.9859 [0.9753, 0.9990] |

**Original forward improves 1.62%; prewarmed training improves 1.41%.** Their
respective pooled intervals lie below one. However, prewarmed forward and
original training remain inconclusive. Neither model phase satisfies the
fixed requirement for an absolute gain under both protocols. No model or CI
case shows a significant persistent regression.

Prewarmed forward's persistent median instead rises 5.24%, while its unchanged
ordinary path rises 6.40%. Its normalized ratio can therefore appear better
without an absolute completed-step gain. Unchanged CI shaders also show timing
variation. Pooled bootstrap intervals omit some between-visit variation.

**The candidate is reverted without extra primary repeats or SDK captures.**
Twenty-two source/dependency hashes, both qualified benchmark binaries and
fourteen restored smoke tests match. Both compiler forks are clean. Qualified
runtime remains `c3ad104`; baseline parity remains unmet.

Evidence is under `target/swiglu-four-tile/`. Manifest
`target/megakernel-results/swiglu-four-tile/summary.json` independently verifies
**806 artifact hashes and 52 completed study cgroups**, SHA-256
`fe65edc72c745fea25d0c43908162692e4fb561519f1ac698654b9569f3cb2a9`.
Peak job memory is **3,717,292,032 bytes**, with zero memory-limit events.
Every heavy job uses a separate verified 16/20/1 GiB cgroup, GPU jobs serialize,
and builds/copies/heavy analysis stay outside measurements. All artifacts
remain on local `/x`, and all launched jobs finish.

Next, isolate forward and gradient claim widths through metadata while
requiring the same repeated shaders and native resources. Compare disabled
batching, forward-only and gradient-only batching before combining policies.
The forward-only graph's diagnostic gain does not establish that widening
gradient claims helps training. Preserve partial completion, exact arithmetic,
requested workers and the fixed absolute timing criterion under both protocols.

## SwiGLU family isolation and forward-only batching

Starting from `2116962`, a temporary host metadata probe isolates forward and
gradient widths using one frozen binary. It always builds the existing repeated
handlers, then chooses widths at graph construction: disabled (1/1), forward
only (4/1), gradient only (1/4), or both (4/4). These settings remain fixed for
the graph's lifetime. Small grids retain width one, and optimizer nodes retain
their original metadata. Relative to the archived four-tile candidate, the
probe adds no GPU code or allocation.

All **61 shader sources match** across probe modes and the archived four-tile
candidate; reported native pipeline statistics match across probe modes.
Twenty actual immutable frame
records verify that **only the intended claim-width bits differ**: every other
node word and every dependency edge is exact. Training keeps four SGD chunks,
24,382 optimizer claims and 97,522 optimizer tiles. Recorded metadata gives:

| Mode | Forward scheduled claims | Training scheduled claims |
|---|---:|---:|
| Disabled | 15,898 | 94,238 |
| Forward only | 11,098 | 89,438 |
| Gradient only | 15,898 | 89,438 |
| Both | 11,098 | 84,638 |

These counts derive from captured node limits and widths, not hardware atomic
traffic. They exclude failed claims. A finite audit covers **16,400 mask/family/
tile-count cases, 10,601,736 claims and 16,801,800 logical coordinates**. The
probe passes 53 codegen checks, fifteen whole-graph smoke tests, the changed-input
replay test in each additional mode, formatting, release builds and Clippy.

**Two hundred fifty diagnostics** use qualified/disabled/forward/gradient/both
and the reverse order, five pairs per case/visit, ten warmups, at least one
warmup second, and matched 100 ms prewarming. Every full-state, selection,
logical-work, optimizer, one-dispatch and no-fallback check passes.

| Persistent GPU diagnostic | Qualified | Disabled | Forward only | Gradient only | Both |
|---|---:|---:|---:|---:|---:|
| Forward | 4.083808 ms | 4.078144 ms | 4.021296 ms | 4.090400 ms | 4.020240 ms |
| Training + SGD | 12.769808 ms | 12.785840 ms | 12.731264 ms | 12.832928 ms | 12.650864 ms |

Forward-only batching improves forward **1.53% versus qualified** and training
0.30%. Gradient-only batching shows no improvement in its affected phase.
Combined batching improves both model GPU medians, but its unchanged CI
inference shader measures +3.80%, exceeding the predeclared 3% diagnostic guard.
This is measured variation, not evidence that the metadata policy changes CI
work. Forward-only is the sole policy that passes every advancement condition;
these diagnostics do not qualify a retained optimization.

The production candidate **removes every probe field, environment switch and
metadata dump**. It admits only explicit unfused F32 SwiGLUConcat grids of at
least 256 tiles. Gradient dispatch cases return to their qualified form. A
source reconstruction verifies that single admission change against the
archived four-tile candidate; its other source files remain exact. The forward
shader matches the probe, while training changes only its gradient dispatch case. The existing
arithmetic, synchronization and SGD batching remain unchanged.

Because removing an inactive gradient repeat wrapper changes the training
shader, a separate **50-pair diagnostic** checks the production build:

| Persistent GPU diagnostic | Qualified | Production candidate | Change |
|---|---:|---:|---:|
| Forward | 4.089824 ms | 4.015680 ms | −1.81% |
| Training + SGD | 12.949664 ms | 12.675456 ms | −2.12% |

Forward uses **86 registers / 18,928 shared bytes / 102,528 code bytes**;
training uses **83 / 29,344 / 241,792**. Against qualified, each phase adds 128
code bytes and training removes one register. Payload, scheduler and resident
bytes remain exact. Both phases retain 168 participating graph workers, with
all 256 requested workers enabled. Every diagnostic state/execution check
passes. Native differences and diagnostic GPU savings require independent
completed-step confirmation.

Full production qualification passes **677 library tests (eight ignored),
168 strict oracle cases, 51 matrix regressions, 53 codegen checks and fifteen
whole-graph smoke tests**, plus strict Clippy, Metal-target compilation,
formatting and release builds. Thirteen-grid replay covers every partial
four-tile remainder with 1/3/256/4,096 workers and exact changed-input outputs.
Both rebuilt benchmark binaries match the frozen candidate.

The fixed **1,000 completed-step comparisons** use control/candidate/candidate/
control visits, 25 interleaved pairs per case/visit, ten warmups and at least one
warmup second. Half use the original protocol; half add matched 100 ms
prewarming. All numerical and execution gates pass. Optimizer accounting is
verified separately across all 1,050 production diagnostic/normal comparisons,
in addition to the 250 probe comparisons. The observation handle was lost
mid-run; the original driver and cgroup were confirmed live and monitored to
completion. No benchmark process was restarted.

| Persistent completed step | Qualified | Production candidate | Candidate/control, pooled bootstrap 95% interval |
|---|---:|---:|---:|
| Original forward | 4.670894 ms | 4.603128 ms | 0.98549 [0.91889, 1.00017] |
| Original training + SGD | 15.366479 ms | 15.128452 ms | 0.98451 [0.97046, 0.99943] |
| Prewarmed forward | 4.358717 ms | 4.286056 ms | 0.98333 [0.97639, 0.99879] |
| Prewarmed training + SGD | 13.209936 ms | 13.121010 ms | 0.99327 [0.98442, 1.00661] |

Prewarmed forward improves **1.67%**, and original training improves **1.55%**,
with their respective intervals below one. Original forward's upper endpoint
is **1.0001700494**, and prewarmed training remains inconclusive. **Neither
model phase passes both required protocols.** No model or CI case shows a
significant persistent regression. The near miss does not change the criterion
or justify extra samples. Pooled intervals omit some between-visit variation.

**The production candidate is reverted; the probe is removed.** No extra
primary repeats or SDK captures follow the failed gate. Twenty-two source/
dependency hashes, both qualified binaries and fourteen restored smoke tests
match. Both compiler forks remain clean. Qualified runtime stays `c3ad104`;
baseline parity remains unmet.

Evidence remains under `target/swiglu-family-widths/` and
`target/swiglu-forward-four/`. The combined manifest
`target/megakernel-results/swiglu-forward-four/summary.json` independently verifies
**1,546 artifact hashes and 77 completed study cgroups**, SHA-256
`505ab5c443f3eac7fb29164416f9b41e26c0203c7d33a8a8e769148494a9c058`.
Peak job memory is **3,716,567,040 bytes**, with zero memory-limit events.
All heavy jobs use separate verified 16/20/1 GiB cgroups; GPU jobs serialize,
heavy analysis stays outside measurements, and artifacts remain on local `/x`.
Every launched job finishes.

The following control compares three versus four tiles for forward-only claims with identical
shaders and native resources. A 400-tile node offers 134 three-tile claims
instead of 100 four-tile claims; both fit below the 168 participating workers
observed in model graphs. This tests more parallel claims and less work per
claim against extra queue traffic. It does not assume simultaneous residency
or a gain from the counts alone. Preserve partial completion, small-grid and
SGD behavior, and the fixed criterion under both timing protocols.

## Three-tile forward claim control

Starting from `9d7441c`, this control changes the archived forward-only candidate's
sequential claim width from four to three. Production admission is otherwise
identical: only explicit unfused F32 `SwiGLUConcat` nodes with at least 256 tiles
qualify. Small grids, gradient handlers, four-tile SGD claims, arithmetic,
publication fences and all requested workers retain their prior behavior.

Each admitted 400-tile node offers **134 claims instead of 100**. Across sixteen
forward nodes, this saves **4,256 successful claims** versus qualified execution
in either model phase, compared with 4,800 saved by four-tile batching. These
metadata-derived counts exclude failed claims and do not establish hardware
atomic traffic, simultaneous residency or a performance gain.

The candidate passes **53 codegen checks and fifteen whole-graph smoke tests**,
formatting and release builds. The thirteen-grid fixture covers all three-tile
remainders, the 256-tile threshold, element tails and mixed cached handlers.
Changed-input replay is bitwise exact against ordinary execution at
**1/3/256/4,096 workers**. A finite audit covers **2,050 tile counts, 722,433
claims and 2,100,225 logical coordinates**.

The predeclared diagnostic series uses qualified/four/three/three/four/qualified
visits, five interleaved pairs per case/visit, ten warmups, at least one warmup
second and matched 100 ms prewarming: **150 comparisons** total. Advancement
requires an affected model GPU median at least 0.5% below qualified and below
four tiles, neither model phase more than 0.5% above qualified, and no CI case
more than 3% above qualified. This screens candidates; it does not replace the
completed-step retention criterion.

All **61 shader sources and reported native pipeline statistics match** between
three and four tiles. Source reconstruction confirms that the only production
change against four tiles is the admitted width. Against qualified, all
19 CI sources and 40 ordinary model sources remain exact; each model's graph
changes only the forward dispatch case. Removing that repeat wrapper reconstructs
the entire qualified source byte for byte, and every handler body is unchanged.

| Persistent GPU diagnostic | Qualified | Four tiles | Three tiles | Three versus four |
|---|---:|---:|---:|---:|
| Forward | 4.079488 ms | 4.012048 ms | 4.021536 ms | +0.24% |
| Training + SGD | 12.761264 ms | 12.675456 ms | 12.700736 ms | +0.20% |

Three tiles improve forward **1.42% versus qualified** and training **0.47%**,
but neither phase beats four tiles. **The advancement gate fails.** These small
differences are diagnostic medians, not established completed-step regressions.
No CI case exceeds the guard; its shaders remain unchanged.

Forward uses **86 registers / 18,928 shared bytes / 102,528 code bytes** and
training **83 / 29,344 / 241,792**, identical to four tiles. Both phases retain
168 participating graph workers with all 256 requested workers enabled.
Every state, selection, single-dispatch, no-fallback and logical-work gate passes.
Payload, scheduler and resident allocations remain exact. All 150 comparisons
preserve optimizer accounting, including four model SGD chunks and 97,522
logical optimizer tiles.

**The candidate is reverted.** No additional diagnostics, full qualification,
normal timing or SDK captures follow the failed gate. Twenty-two source/
dependency hashes, both qualified binaries and fourteen restored smoke tests
match. Both compiler forks remain clean. Qualified runtime stays `c3ad104`, and
baseline parity remains unmet.

Evidence remains under `target/swiglu-forward-three/`. The independently checked
manifest `target/megakernel-results/swiglu-forward-three/summary.json` verifies
**735 artifact hashes and 26 completed study cgroups**, SHA-256
`123da1eb03c8bf875c27706310a46abfed46a89a3050b7fb94b92939b22be563`.
Peak study memory is **3,717,373,952 bytes**, with zero memory-limit or OOM events.
All heavy jobs use separate verified 16/20/1 GiB cgroups; GPU jobs serialize,
heavy analysis stays outside measurements, and artifacts remain on local `/x`.
Every launched job finishes, including the separate manifest and verification
jobs.

Further claim-width sweeps are not supported by this evidence. The following
study returns to the qualified runtime and extends calibrated complete-claim timing to the forward
graph, whose remaining relative gap is larger. Compare dependency-aware node
envelopes with matched ordinary dispatches, including overlapping work, before
choosing another runtime change. Keep instrumentation overhead, source/native
resource changes and full-state checks separate from performance attribution;
do not sum overlapping spans or treat participating workers as occupancy.

## Complete forward claim attribution

Starting from `b387702`, a temporary diagnostic extends the existing clock
instrumentation to the **154-node forward fixture**. One frozen binary compares
worker timestamps alone with all claim timestamps. Both modes reserve identical
storage and use identical shaders; only the per-node recording offsets change.
The host also preserves the exact immutable node metadata and physical
dependency edges. Training and CI graphs remain uninstrumented.

The timestamp helper is unchanged from the earlier AT study. Each timestamp is
stored immediately, so no clock value remains live across a handler. A local
Blade hook substitutes Device-scope `OpReadClockKHR` for the validated marker;
Naga is unchanged. Both original and transformed SPIR-V validate, and reversing
the substitution reproduces each original module byte for byte. The two captured
modules differ only in debug records. All **61 WGSL sources match between
recording modes**. Against qualified, all 19 CI sources and 41 model sources
remain exact; removing the clock hooks reconstructs the entire forward graph
source. Handler arithmetic, geometry, claims and synchronization are unchanged.

The probe passes **five focused graph codegen tests and fourteen whole-graph
smoke tests**. A finite geometry audit covers **3,648 cases, 94,336 claims,
162,000 logical coordinates and 19,134 partial claims**, including empty grids.
Every decoded full trace contains **15,898 claims covering 22,042 logical tiles**,
with exact per-worker ownership and completion counts. Disabled recording leaves
every reserved claim record zero. Metadata and dependency edges match across all
recording modes and visits.

Twelve preflight comparisons precede a fixed **300-comparison diagnostic series**:
qualified/disabled/all/all/disabled/qualified, ten interleaved pairs per case/
visit, ten warmups, at least one warmup second and matched 100 ms prewarming.
Every numerical, implementation-selection, one-dispatch, no-fallback and
logical-work gate passes. No timing run is repeated.

| Persistent GPU diagnostic | Qualified | Recording disabled | All claims |
|---|---:|---:|---:|
| Forward | 4.091376 ms | 4.120768 ms | 4.131344 ms |
| Training + SGD, unchanged shader | 12.792032 ms | 12.765296 ms | 12.784640 ms |

Full recording adds **0.98%** forward GPU time versus qualified and **0.26%**
versus disabled recording. Preflight overhead is 1.04%. This stays below the
predeclared 10% threshold for treating durations as highly perturbed, but does
not eliminate individual interval perturbation. Forward retains **86 registers
and 18,928 shared bytes**. Code grows from 102,400 to 104,576 bytes; initial/live
queue storage adds **891,104 bytes**. Both recording modes have identical native
statistics and allocations. All 256 requested workers remain enabled, with
168 participating in graph work.

Two final full traces preserve **31,796 claim intervals**. Every one of the
161 physical dependency edges has its producer endpoint before its consumer
entry in both traces: **322 ordering checks**. Following the last-finishing
predecessor back from the final node yields a **137-node chain** in each trace.
Its nonoverlapping node spans and predecessor-to-entry gaps partition the
observed final-node envelope; this is an observed execution path, not a causal
prediction of a changed schedule.

| Scaled interval | First full trace | Second full trace |
|---|---:|---:|
| GPU dispatch | 4.106144 ms | 4.090816 ms |
| Chain node spans | 3,620.16 µs | 3,612.05 µs |
| Chain predecessor-to-first-entry gaps | 484.32 µs | 477.07 µs |
| Gap time overlapping another recorded claim body | 20.63 µs | 20.69 µs |
| Gap time outside all recorded claim bodies | 463.69 µs | 456.38 µs |
| Final node endpoint to worker-envelope exit | 1.67 µs | 1.70 µs |

Most of the roughly **0.48 ms** chain gap therefore lies outside recorded
handler bodies. These intervals still contain completion fan-in, dependency
publication, ready scanning, claim acquisition, broadcasts, parameter loading,
fences and probe work. They are not proof of GPU idleness or 0.46 ms of removable
overhead. The full-pass mean is about 70 overlapping claim bodies; this is not
an occupancy estimate.

A separate qualified ordinary profile preserves complete forward/backward
state and selected implementations. Its pass-start timestamps include dispatch
boundaries; forward profiling adds **16.8% wall time**. Consequently the following
matched node medians identify candidates, rather than exact kernel speed ratios.
The listed spans also must not be summed across overlapping nodes.

| Shape | Ordinary dispatch median | Persistent node-span median, scaled | Median claim body, scaled |
|---|---:|---:|---:|
| FusedMatMulAdd 50×720×2048 | 63.55 µs | 69.04 µs | 52.63 µs |
| MatMul 50×4096×720 | 49.15 µs | 55.86 µs | 55.42 µs |
| FusedMatMulAdd 50×720×960 | 32.74 µs | 33.16 µs | 25.30 µs |
| FlashAttentionCoop, sequence 50 | 9.15 µs | 21.62 µs | 14.77 µs |
| SwiGLUConcat, 102,400 elements | 4.10 µs | 7.24 µs | 0.70 µs |
| RmsNorm 50×720 | 4.11 µs | 2.16 µs | 1.65 µs |

The attention span repeats an already investigated discrepancy; it does not
justify repeating rejected ownership, softmax-sharing or scratch-clear probes.
The dependency-chain gaps instead motivate the following narrower measurement of
publication-to-entry stages. It records last-tile completion, bounds around ready
publication, the zero-ticket claim and handler entry, with identical disabled
metadata/storage controls. It brackets publication because a consumer may claim
before the publishing worker records its post-publication timestamp. Calibrate
the new instrumentation before attributing the gaps to any single stage or
changing fences, atomics, worker eligibility or descriptor semantics.

The [shader-clock specification](https://raw.githubusercontent.com/KhronosGroup/SPIRV-Registry/main/extensions/KHR/SPV_KHR_shader_clock.asciidoc)
does not guarantee a fixed tick rate or code-motion barriers. Raw ticks remain
available; scaled microseconds use each dispatch's GPU interval and worker
envelope. Two final traces per mode do not capture every measured step, and
small aggregate overhead does not remove per-node scheduling perturbation.

**The probe is removed.** Twenty-one main source/dependency hashes, three Blade
source hashes, both qualified binary hashes and fourteen restored smoke tests
match. Both forks are clean. No runtime optimization, normal completed-step
qualification, full oracle/Metal qualification or fresh SDK capture is claimed.
Qualified runtime remains `c3ad104`; baseline parity remains unmet.

Evidence remains under `target/clock-forward-claims/`. Manifest
`target/megakernel-results/clock-forward-claims/summary.json` independently verifies
**515 artifact hashes and 35 completed study cgroups**, SHA-256
`b8bed92e5f23d4ba774ecc1fab07ac477477e297f0d27f75a8e2fe5769c03519`.
It preserves an early verification failure: verification began before the first
manifest job finished. That original job completed; only artifact verification
was repeated, with no GPU rerun. Peak job memory is **4,246,446,080 bytes**, with
zero memory-limit or OOM events. Heavy work uses separate verified 16/20/1 GiB
cgroups, GPU jobs serialize, artifacts remain on local `/x`, and every launched
job finishes.

## Forward publication and entry stages

Starting from `6379d88`, a temporary probe adds up to five unique events per forward
node: the completion-winning RMW observation, bounds around ready-bit and
summary-bit publication, the zero-ticket claim observation, and return from the
ticket broadcast. Existing per-claim body timestamps provide handler entry and
exit. Immediate stores avoid keeping a timestamp live across a handler. No new
shared GPU variables are introduced, and no synchronization operation changes.

One frozen binary compares body recording with body-plus-stage recording. Both
modes reserve identical queue storage; **only one stage-recording flag differs**.
Both record all **15,898 claims and 22,042 logical tiles**. The disabled stage
table remains zero. The host preserves exact node metadata and physical edges.
Only the 154-node forward fixture is instrumented; training and CI stay exact.

The probe passes **five focused graph codegen tests and fourteen whole-graph
smoke tests**. The inherited finite claim audit covers 3,648 cases, 94,336 claims,
162,000 logical coordinates and 19,134 partial claims. Twelve preflight
comparisons precede **300 fixed diagnostics** in qualified/body/stages/stages/
body/qualified order, ten pairs per five-case visit, ten warmups, at least one
warmup second and matched 100 ms prewarming. All state, implementation-selection,
logical-work, single-dispatch and no-fallback gates pass; no capture is repeated.

| Persistent GPU diagnostic | Qualified | Body recording | Body and stages |
|---|---:|---:|---:|
| Forward | 4.091360 ms | 4.229872 ms | 4.249744 ms |
| Training + SGD, unchanged shader | 12.848192 ms | 12.781216 ms | 12.795888 ms |

Stage recording adds **3.87%** forward GPU time versus qualified and **0.47%**
versus the identical shader with stage recording disabled. Preflight overhead
is 3.99%. All 61 shader sources and reported native statistics match between
recording modes. Forward remains **86 registers / 18,928 shared bytes**, while
code grows from 102,400 to 108,544 bytes. Initial/live diagnostic queues add
**909,592 bytes**; payload allocation is unchanged. All 256 workers remain
enabled, with 168 participating in graph work.

Removing every clock hook reconstructs the entire qualified forward shader.
All handler bodies, geometry, claims and publication operations remain exact;
the 19 CI sources and 41 other model sources are unchanged. Both original and
transformed SPIR-V validate, reversing the clock substitution restores each
original module byte for byte, and the two captured modules differ only in
debug records. Naga is unchanged; the temporary Blade hook is later removed.

Four final traces contain **63,592 claim intervals**, with exact ownership and
completion counts. All 161 physical edges obey producer-endpoint-before-consumer-
entry ordering in every trace: **644 checks**. Each trace has a 137-node
last-finishing-predecessor chain. The two stage-enabled traces also validate
**1,500 unique stage timestamps**: completion, claim and ticket-return events
for all 154 nodes, plus publication bounds for 144 dependent nodes.

The following totals cover the chain's **136 dependent nodes**. Publication's
actual point lies inside the recorded bounds; the lower/upper claim latency
uses those bounds. The zero-ticket owner may enter after another worker, so its
entry lag is subtracted to recover the earliest-entry gap.

| Scaled interval, summed on the observed chain | First stage trace | Second stage trace |
|---|---:|---:|
| Last predecessor body to latest predecessor completion marker | 69.33 µs | 69.31 µs |
| Latest completion marker to publication-before marker | 142.57 µs | 141.76 µs |
| Publication to zero-ticket claim, bounded | 153.33–181.16 µs | 155.85–183.78 µs |
| Zero-ticket claim marker to ticket return | 61.23 µs | 61.17 µs |
| Ticket return to zero-ticket handler entry | 123.77 µs | 124.21 µs |
| Zero-ticket entry lag behind earliest handler entry | 6.92 µs | 6.95 µs |

Using the upper publication-to-claim latency, the first five rows total **578.06/580.23 µs**
to zero-ticket entry. Subtracting its lag yields **571.14/573.28 µs** to first
entry. Adding the root's 9.64/9.54 µs entry gap reproduces the full chain gaps
of **580.78/582.83 µs**. The final dispatches measure 4.214912/4.218784 ms.
No consumer timestamp falls inside its publication bracket in these samples;
the protocol still requires bounds because that ordering is not guaranteed.

This does **not** identify one dominant removable delay. Completion-to-publication
includes fences, dependency atomics and immutable-edge loads. Publication-to-claim
includes polling and acquisition. Ticket-return-to-entry includes node broadcasts,
parameter loading and the payload acquire. Every interval also contains probe
work. Even body-only controls show 556–560 µs chain gaps; the preceding, smaller
probe showed 477–484 µs. Aggregate overhead below the predeclared 10% threshold
does not make per-stage measurements unperturbed or directly recoverable.
The [clock-rate and code-motion limitations](#complete-forward-claim-attribution)
continue to apply.

The next narrowly scoped candidate is to audit the release fence repeated before
each successor dependency decrement. The last-tile acquire/release fence already
dominates the successor loop. Establish whether it can serve the required release
for every decrement using the [Vulkan fence-to-fence and inter-thread ordering rules](https://docs.vulkan.org/spec/latest/appendices/memorymodel.html#memory-model-synchronizes-with),
the actual emitted SPIR-V and the absence of intervening payload writes. Only if
that proof holds, test omitting the per-edge fence while retaining all-lane payload
fences, last-tile acquisition, final-dependency acquisition/publication and epoch
ordering. This is a candidate, not a revised memory contract. It needs full
correctness, native-resource, completed-step and SDK qualification; the trace
does not justify a new Naga API or establish a speedup.

**The probe is removed.** Twenty-one main source/dependency hashes, three Blade
source hashes, both qualified binaries and fourteen restored smoke tests match.
Both forks are clean. No runtime optimization, normal completed-step series,
full oracle/Metal qualification or fresh SDK capture is claimed. Qualified
runtime remains `c3ad104`, and baseline parity remains unmet.

Evidence remains under `target/clock-forward-stages/`. Manifest
`target/megakernel-results/clock-forward-stages/summary.json` independently verifies
**506 artifact hashes and 32 completed study cgroups**, SHA-256
`3cfa41441f8c688c690d752f6f5e2f6fcafccf0cda50ad3e218b73927505385b`.
Peak job memory is **3,720,974,336 bytes**, with zero memory-limit or OOM events.
All heavy work uses separate verified 16/20/1 GiB cgroups; GPU jobs serialize,
artifacts stay on local `/x`, and every launched job finishes.

## Successor release-fence control

Starting from `8b25a00` with qualified runtime `c3ad104`, this candidate deletes
only the `storageFence()` immediately before each successor dependency decrement
in `graph_worker.wgsl`. The preceding last-tile acquire/release fence remains.
There is no Naga API, dependency, arithmetic, geometry, queue-layout or worker
admission change.

The source-level argument uses
[Vulkan fence synchronization and inter-thread ordering](https://docs.vulkan.org/spec/latest/appendices/memorymodel.html#memory-model-synchronizes-with).
Let C be the last-tile fence and P the final predecessor's fence before ready
publication. All payload lanes still fence before the workgroup rendezvous;
the leader fence and tile-completion RMW sequence still collect their accesses
at C. C also supplies release ordering before every successor decrement. Each
counter's contiguous relaxed RMW sequence connects earlier predecessors' C to
the final predecessor's P. For that final predecessor itself, C's acquire side
also orders P. These relations compose within UniformMemory inter-thread
happens-before; this does not assume unrestricted transitivity of plain
happens-before.

Every writer still makes its own payload writes available directly to the queue
family, and every consumer lane still performs its visibility operation. No
availability/visibility chains are needed. Between C and the decrements, the
fan-out loop accesses immutable metadata and writes only queue atomics and
function-local variables. Earlier successor publications create no new payload
writes requiring a later release. The final-dependency publication fence and
post-loop completion-epoch fence remain. The argument also preserves read-before-
write alias dependencies, zero-successor nodes and partial/packed completion.

A temporary output-only Blade hook captures the actual SPIR-V supplied to Vulkan
for control and candidate. All **ten modules validate**. In all five graph pairs,
C dominates the successor loop and dependency decrement, and the loop contains
no payload writes or calls. Surviving storage fences retain QueueFamily scope
5 and semantics `0x6048`. Module headers match; excluding debug records, each
candidate differs by **exactly one removed OpMemoryBarrier**. The hook and local
dependency override are removed before production builds. This concrete audit
and ordering argument supplement tests; they are not a complete formal model
of all Vulkan executions.

Production passes **52 codegen tests and fourteen whole-graph smoke tests**.
Twenty audit-build comparisons and fifty production diagnostics pass state,
implementation-selection, allocation, optimizer, logical-work, one-dispatch
and no-fallback checks. Each production visit uses five interleaved pairs,
ten warmups, at least one warmup second and 100 ms prewarming. The audit builds
bridge to production through all **61 exact shader sources** per mode, exact
whole-graph native statistics and full ordinary-statistic multisets. All 56
ordinary sources remain exact between control and candidate; the five graph
sources differ only by the fence deletion.

| Persistent GPU diagnostic, ms | Qualified | One fence removed | Change |
|---|---:|---:|---:|
| CI inference | 0.030752 | 0.031040 | +0.94% |
| CI forward/backward | 0.066432 | 0.067968 | +2.31% |
| CI forward/backward/SGD | 0.072224 | 0.072480 | +0.35% |
| SmolVLA forward | 4.076992 | 4.052448 | −0.60% |
| SmolVLA forward/backward/SGD | 12.745984 | 12.830944 | +0.67% |

Ordinary model GPU medians remain 2.926656→2.923296 ms forward and
10.751168→10.748992 ms training. Every graph's native code shrinks 128 bytes.
Forward retains **86 registers / 18,928 shared bytes**; training retains
**84 registers / 29,344 shared bytes**. Both have 168 participating workers out
of 256 enabled. Payload, scheduler and resident allocations are unchanged.

The policy recorded before measurement requires at least one model GPU median
to improve 0.5%, neither model to regress over 0.5%, and no CI regression over
3%. **Training fails the guardrail.** These small diagnostic samples do not
establish completed-step gains or regressions, but they do determine that this
candidate does not advance. No additional repeats, full library/oracle/Metal
qualification, normal completed-step series or SDK captures are run.

The initial bridge audit incorrectly compared the pipeline-cache iteration order.
Entry-point names are nonunique. Its script and failed log are preserved; the
corrected audit compares complete-record multisets, retaining duplicates and
every statistic. The unique whole-graph pipeline is also compared directly.
Only analysis is rerun; no GPU capture is repeated.

**The candidate is reverted.** All 22 archived source/dependency hashes, the
Blade source hash, both qualified binaries and fourteen restored smoke tests
match. Both forks are clean. The retained memory contract is unchanged.
The following parameter study audits repeated immutable parameter staging on
consecutive claims for the same node, preserving transitions, all lanes and
existing shared state.

Evidence remains under `target/successor-release-fence/`. Manifest
`target/megakernel-results/successor-release-fence/summary.json` independently
verifies **667 artifacts and 26 completed study scopes**, SHA-256
`3a2acf4364d4e9bd6c87f9e453a99b83a02f70d6e3de03cb5ddd8a5cbd93c22c`.
One inspection scope's receipt is transcribed from observed terminal tool
output; all other study scope receipts are raw logs. Peak job memory is
**3,721,666,560 bytes**, with zero memory-limit or OOM events. Heavy work uses
separate verified 16/20/1 GiB cgroups, GPU jobs serialize, artifacts stay on
local `/x`, and every launched job finishes. Baseline parity remains unmet.

## Parameter staging controls

Four controls start from `9fad2a9`, retaining qualified runtime `c3ad104` as
their baseline. They investigate the immutable per-node parameter copy and
uniform broadcast before handler entry. None advances beyond its fixed
diagnostic gate, and all source and dependency changes are restored.

First, complete archived claim traces estimate opportunities for consecutive
same-node reuse. Two forward traces each contain 15,898 claims; 3,741/3,744
claims (23.53%/23.55%) could reuse the previous node's parameters. Two training
traces each contain 94,238 graph/optimizer claims; 47,834/46,080
(50.76%/48.90%) could reuse them. Nonconsecutive node revisits remain misses.
These are historical instrumented claim orders, not measured candidate cache
hit rates, traffic reductions or recoverable time.

**Copy-only cache.** A function-local node ID guards the strided copy from
immutable storage parameters into the existing shared parameter array. The
`workgroupUniformLoad` remains unconditional. All physical lanes see the same
broadcast ticket; the first valid claim and every node transition refresh the
array. Idle tickets leave the cache untouched, and each dispatch initializes
it again. No shared state, handler arithmetic, fence or selected geometry
changes. The control passes 52 codegen and fourteen whole-graph smoke tests.

**Complete value cache.** A second control keeps both the node ID and the
returned parameter array in function-local variables across claims, skipping
the copy and uniform broadcast on a hit. All lanes start with the same ID and
update it to the same broadcast node ID, preserving uniform control flow by
induction. Existing Naga validation flags and checks remain unchanged. Naga's
specific `WorkGroupUniformLoad` control-flow validation is incomplete, so
passing tests alone is not the uniformity argument. This control also passes
52 codegen and fourteen whole-graph smoke tests.

Each cache control runs fifty production diagnostics: five interleaved pairs
per case/mode, ten warmups, at least one warmup second and 100 ms prewarming.
All 56 ordinary shader sources remain exact; the five graph sources differ
only by the audited cache edit. Native ordinary-statistic multisets, selected
implementations, allocation, optimizer work, state budgets and logical work
match. Every case keeps one dispatch, no fallback and all 256 enabled workers.

| Persistent GPU diagnostic, ms | Copy control | Copy cache | Value control | Value cache |
|---|---:|---:|---:|---:|
| CI inference | 0.031296 | 0.031104 | 0.031104 | 0.031264 |
| CI forward/backward | 0.067552 | 0.067712 | 0.067584 | 0.067968 |
| CI forward/backward/SGD | 0.073952 | 0.072064 | 0.072736 | 0.073760 |
| SmolVLA forward | 4.072736 | 4.072320 | 4.079712 | 4.161568 |
| SmolVLA forward/backward/SGD | 12.809696 | 12.837408 | 12.741952 | 12.768320 |

Copy caching changes forward/training by **−0.01%/+0.22%**; value caching by
**+2.01%/+0.21%**. Neither meets the required 0.5% model improvement, and value
caching also exceeds the 0.5% model regression guardrail. Forward/training
registers change from 86/84 to **88/84** for copy caching and **100/92** for
value caching. Shared memory remains 18,928/29,344 bytes. Native model code
grows from 102,400/241,664 bytes to 102,528/241,920 and 109,952/242,944 bytes,
respectively. These results do not establish a completed-step change or prove
that register growth alone causes the regression.

**Direct shared-array read.** A third source control replaces the uniform load
with a workgroup barrier followed by a direct array read. Fifty codegen tests
pass, but two existing packing admission tests fail: the mixed-geometry matrix
batch becomes 1 instead of 4, and scalar attention loses its required packing
axis. The direct read loses the uniform-value classification needed by the
scope-aware cooperative operand check. The probe stops before GPU execution;
no validation or assertions are weakened. Previously captured control files
are explicitly marked as reused provenance, not new measurements or a
completed candidate SPIR-V audit. Its unused Blade dump hook is removed.

**Emitted post-read barrier.** A final application-specific counterfactual
preserves every WGSL source and Naga validation result. A temporary Blade loader
captures emitted SPIR-V and, only for the graph entry point, removes the
`OpControlBarrier` immediately after the parameter-array load. A matched loader
control leaves the original bytes untouched. Both modes use the same frozen
binary and debug emission options; the archived qualified binary supplies a
third control. This is a diagnostic transform, not a general compiler rule or
a new WGSL memory contract.

Before Vulkan sees each module, the transform verifies that the parameter load
is bracketed by matching workgroup execution/memory barriers with semantics
`0x108`, that the shared parameter array has only the expected write site,
and that every control-flow path from the removed barrier to another write
crosses a retained workgroup barrier or terminates. The preceding barrier
still orders all parameter writes; intervening handler scratch is disjoint.
All storage fences remain. This application-specific ordering audit supplements
the [uniform-load synchronization contract](https://gpuweb.github.io/gpuweb/wgsl/#workgroupUniformLoad-builtin);
it does not change that contract or constitute a general optimization proof.

Fourteen GPU smoke tests pass with elision. All **44 original/output module
pairs** validate before pipeline creation. Independent checks verify byte-exact
controls and recover every elided original by reinserting only that instruction.
All five benchmark graph sources have matching original non-debug SPIR-V
between loader modes. Seventy-five fixed diagnostics compare qualified,
loader-control and elided modes, using the same warmup and pairing protocol.
All 61 shader sources match across modes. Qualified/control native statistics
match exactly; ordinary native-statistic multisets match across all three.
State, optimizer, selection, allocation, one-dispatch and no-fallback checks pass.

| Persistent GPU diagnostic, ms | Qualified | Loader control | Barrier elided |
|---|---:|---:|---:|
| CI inference | 0.031040 | 0.031072 | 0.030304 |
| CI forward/backward | 0.067520 | 0.067392 | 0.067808 |
| CI forward/backward/SGD | 0.072256 | 0.072320 | 0.072320 |
| SmolVLA forward | 4.080512 | 4.074720 | 4.080896 |
| SmolVLA forward/backward/SGD | 12.760128 | 12.743936 | 12.701280 |

Elision changes forward by **+0.01%/+0.15%** against qualified/loader control,
and training by **−0.46%/−0.33%**. The predeclared gate requires the same model
phase to improve at least 0.5% against both controls, neither model to regress
over 0.5% against either, and no CI regression over 3%. **It does not advance.**
Model registers and shared memory remain 86/84 and 18,928/29,344 bytes;
forward code stays 102,400 bytes and training grows 128 bytes to 241,792.
Both model phases have 168 participating workers out of 256 enabled, which
does not by itself establish occupancy.

The initial freeze rejected the local Blade override's `Cargo.lock` change.
Its script and failed log are preserved. The corrected freeze admits exactly
the single removed git-source line for the overridden Blade package, hashes
the probe lockfile and rejects every other change. No measurement is repeated.

**All four controls are reverted.** Every study verifies 22 restored main
source/dependency hashes, both qualified binary hashes and fourteen restored
smoke tests. The direct-read study also reruns all 52 restored codegen tests.
Temporary Blade hooks are removed and both forks are clean. The combined
study contains **175 diagnostic comparisons**, no normal completed-step series,
no fresh SDK captures and no extra repeats after a failed gate. Parameter
caching and post-read barrier elision are closed for this runtime; they do not
justify a new Naga API. The following native cooperative operand prefetch study
preserves arithmetic, all existing barriers, bounds and packing admission.

Evidence remains under `target/parameter-copy-cache/`,
`target/parameter-value-cache/`, `target/parameter-read-barrier/` and
`target/parameter-post-barrier-spv/`. Combined manifest
`target/megakernel-results/parameter-staging/summary.json` independently verifies
**1,491 artifact hashes and 49 completed study scopes**, SHA-256
`69cc88023adad8a9c2e38f40fdff0d2ab164af69242200b6d681e3ebb7db142f`.
Peak study job memory is **3,716,214,784 bytes**, with zero memory-limit or OOM
events. Every heavy job has a separate verified 16/20/1 GiB cgroup; GPU jobs
serialize and artifacts stay on local `/x`. Qualified runtime remains `c3ad104`;
baseline parity remains unmet.

## Native cooperative B prefetch

Starting from `eeb95f8`, this control changes plain 16×16 f16-input cooperative
matrices without typed layouts, compensation or multiple copies. The separate
8×8 implementation and scalar matrix templates remain unchanged. It applies
identically to ordinary and persistent execution and retains selected geometry.

The template's `B_STAGE_0/1` loads the mathematical right-hand B operand into
`shared_a0/1`; `A_STAGE_0/1` loads the left-hand operand into `shared_b0/1`.
Normal/AT B loads run along N; BT loads run along K and scatter transposed.
The probe redirects each B stage's four assignment targets into a private
vector, retaining every original branch, bounds check, fallback address, zero
and f16 conversion. The original staging site writes those four values into
the original shared cells. Narrow prologue variants retain only their first
B tile. All 64 logical lanes keep their original ownership.

Private values initialize only for a nonempty K range. After the original
staging barrier, `k - t > 16` guards loading the next stage using `t + 16`.
Eligible kernels have no split range; original tail bounds remain. The trailing
reuse barrier still completes all current shared reads before the next shared
writes. Cooperative loads, MMA order, accumulators, A prologues and epilogues
are unchanged. Inputs remain read-only for the handler, whose outputs are
written after its K loop. Graph alias dependencies and inactive-read guards
still cover the entire handler. No shared allocation, barrier, fence or queue
operation is added or removed.

The probe passes **52 codegen tests and fourteen whole-graph smoke tests**.
Fifty fixed diagnostics use five interleaved pairs per case/mode, ten warmups,
at least one warmup second and 100 ms prewarming. Every selection, allocation,
state-budget, optimizer, logical-work, one-dispatch and no-fallback check passes.
Maximum error remains 4.66e-9. All 256 requested workers remain enabled.

All **nineteen CI shaders** and **38 ordinary model shaders** are byte-identical.
The two changed ordinary model sources have exact inverses: redirecting the
private components back to their original shared targets and restoring the
staging positions recovers every original source byte. One handler changes in
each model graph. Every other function, including the scheduler, remains exact;
changed signatures and barrier counts match. All storage reads in the changed
packed handlers retain explicit inactive-lane guards. That graph audit is not
a byte-exact inverse of Naga's normalized changed handlers.

| Persistent GPU diagnostic, ms | Qualified | B prefetch | Change |
|---|---:|---:|---:|
| CI inference | 0.030528 | 0.030880 | +1.15% |
| CI forward/backward | 0.068128 | 0.067744 | −0.56% |
| CI forward/backward/SGD | 0.073856 | 0.072256 | −2.17% |
| SmolVLA forward | 4.077376 | 4.134464 | +1.40% |
| SmolVLA forward/backward/SGD | 12.785120 | 12.977408 | +1.50% |

Ordinary model GPU medians change 2.922720→2.920512 ms forward and
10.737760→10.754784 ms training. Native model registers are **86→86 / 84→83**,
shared memory remains **18,928 / 29,344 bytes**, and code grows
**102,400→104,960 / 241,664→247,040 bytes**. Both model phases retain 168
participating workers and 22,042/83,312 logical graph tiles; training also
executes all 97,522 optimizer tiles across four appended phases. These resource
observations do not identify an instruction-level cause or establish occupancy.
CI shaders and native resources are unchanged despite small timing differences.

Both model phases exceed the predeclared **0.5% regression guardrail** and
neither improves 0.5%, so the probe does not advance. There are no additional
repeats, full library/oracle/Metal qualification, normal completed-step series
or fresh SDK captures. The small diagnostic samples determine this decision;
they do not establish completed-step performance changes.

The initial source audit incorrectly required an inactive-lane guard on every
handler, including unchanged full-width handlers without inactive lanes. Its
script and failed log are preserved. The corrected audit compares unchanged
functions byte-for-byte and checks every storage read in changed packed
handlers. Only analysis is rerun; no GPU measurement is repeated.

**The probe is reverted.** All 22 source/dependency hashes, both qualified
binary hashes and fourteen restored smoke tests match. Both compiler forks
remain clean. The result closes this native B lookahead control; it does not
justify another fence API, removal of existing barriers, or an instruction-cache
diagnosis from code size alone. Further candidates must pass the same fixed
correctness, resource, completed-step and matched SDK gates.

Evidence remains under `target/coop-b-prefetch/`. Manifest
`target/megakernel-results/coop-b-prefetch/summary.json` independently verifies
**359 artifacts and seventeen completed study scopes**, SHA-256
`05f92a599a0980086ef2fa400113df9d778f41ea25e7c6de4ec81c526b2ef39b`.
Peak study job memory is **3,713,859,584 bytes**, with zero memory-limit or OOM
events. All heavy work runs in separate verified 16/20/1 GiB cgroups, GPU jobs
serialize, and artifacts remain on local `/x`. Qualified runtime remains
`c3ad104`; baseline parity remains unmet.

## Active-lane payload fences

Starting from `24d22f7`, this probe moves the two lane-level payload fences
from the worker entry point into each handler case, guarded by that case's
existing `graph_active_lane` value. The first fence precedes the complete
handler/repeat sequence and the second follows it. Handler bodies, masks,
arithmetic, geometry, parameters, scratch, descriptors and all queue operations
remain unchanged. Every workgroup control barrier remains unconditional.

The ordering argument relies on the retained inactive-access repair: smaller
and packed handlers perform no payload storage reads or writes in inactive
lanes, including through read-only aliases. Full-width handlers have no inactive
lanes. Every lane that accesses payload still performs its own queue-family
availability/visibility operations through the canonical descriptor reference.
The successful-claim leader fence and uniform broadcasts still precede its
acquire. Its release still precedes the same workgroup rendezvous and leader
completion fence. Last-tile, successor-publication and epoch fences remain.
Earlier active claims have already completed their own publication protocol;
later active claims still acquire. Empty nodes perform no payload access.

This preserves the relevant
[Vulkan fence and memory-operation rules](https://docs.vulkan.org/spec/latest/appendices/memorymodel.html#memory-model-synchronizes-with)
without relying on availability/visibility chains. It is a source-level
argument supplemented by structural and emitted-code audits, not a complete
formal model of Vulkan executions. It does not change the production contract.

Exact inverse substitutions reconstruct all five original graph sources.
All **56 ordinary shaders** and their complete native-statistic multisets remain
exact. A fresh audit built against pinned Naga `bc7b70bc` validates the emitted
WGSL and checks storage reads, shared-read protection and shared/storage writes.
All **102 adapted storage-read sites** remain guarded in both variants.

A temporary output-only Blade hook captures the actual five candidate SPIR-V
modules supplied to Vulkan, without changing emission flags or instructions.
Every module validates. Across **71 handler cases**, both payload fences use
the same SSA activity predicate, QueueFamily scope 5 and semantics `0x6048`.
The acquire guard dominates the handler call; the release guard postdominates
it. Their conditional blocks contain only the intended fence, and the remaining
entry-point fences retain their scopes and semantics. Five audit comparisons
bridge all 61 shader sources, full native-statistic multisets, state, allocation
and execution records to the production candidate.

The initial SPIR-V audit assumed the activity predicate would have an `OpName`.
It did not. Its script and failed log remain archived. The corrected audit
checks the defining `OpULessThan` and its original activity-declaration source
line, the shared SSA predicate and both control-flow properties. Only analysis
is rerun; no shader, test or GPU measurement changes. The hook and local Cargo
override are removed, and both rebuilt production candidate binaries match
their frozen hashes before normal timing.

Qualification passes **676 library tests (eight ignored), 168 strict
whole-graph oracle cases, 51 matrix regressions, 52 codegen tests and fourteen
whole-graph smoke tests**, plus strict all-target/all-feature Clippy, formatting,
release builds and Metal-target compilation. Metal GPU execution and additional
hardware remain unqualified. Every state budget, selected implementation,
allocation, optimizer phase, logical task, one-dispatch and no-fallback check
passes. Maximum error remains 4.66e-9; all 256 workers remain enabled.

Fifty production diagnostics use five interleaved pairs per case/mode, ten
warmups, at least one warmup second and 100 ms prewarming:

| Persistent GPU diagnostic, ms | Qualified | Guarded payload fences | Change |
|---|---:|---:|---:|
| CI inference | 0.030752 | 0.031136 | +1.25% |
| CI forward/backward | 0.067584 | 0.067936 | +0.52% |
| CI forward/backward/SGD | 0.072224 | 0.072224 | 0.00% |
| SmolVLA forward | 4.077376 | 4.013792 | −1.56% |
| SmolVLA forward/backward/SGD | 12.860416 | 12.855040 | −0.04% |

This passes the fixed advancement gate. Ordinary model GPU medians remain
2.922496→2.922496 ms forward and 10.733440→10.731808 ms training. Model
registers remain **86/84**, shared bytes **18,928/29,344**, and participating
workers **168**. Code grows **102,400→107,264 / 241,664→253,056 bytes**.
These resources do not establish occupancy or isolate the effect of skipping
inactive fences from moving them into cases.

The fixed **1,000 completed-step comparisons** use qualified/candidate/candidate/
qualified visits, 25 interleaved pairs per case/visit, all five cases and both
original and 100 ms prewarmed protocols. Every sample and visit is retained.

| Persistent completed step | Qualified, ms | Candidate, ms | Candidate/control [pooled 95% interval] |
|---|---:|---:|---:|
| Original forward | 4.681407 | 4.548516 | 0.9716 [0.9138, 0.9849] |
| Prewarmed forward | 4.373801 | 4.336035 | 0.9914 [0.9274, 1.0603] |
| Original training + SGD | 15.322953 | 15.151188 | 0.9888 [0.9665, 1.0161] |
| Prewarmed training + SGD | 13.115031 | 13.260397 | 1.0111 [0.9901, 1.0138] |

Original forward improves **2.84%**, but neither model has an absolute gain
under both protocols. No raw persistent interval establishes a regression.
Unchanged ordinary inference shifts −7.51% original / −8.91% prewarmed;
original inference's normalized ratio worsens 2.22% with an interval above one.
Ordinary forward shifts −1.89% original / +8.68% prewarmed, with both intervals
including no change. Original forward's normalized ratio is also inconclusive.
These ordinary shifts and broad intervals remain in the record; pooled
bootstrap intervals do not capture all between-visit variation. There are no
additional repeats or fresh SDK captures after the failed retention gate.

**The candidate is reverted.** All 23 source/dependency hashes, the original
Blade source hash, both qualified benchmark binaries and fourteen restored
smoke tests match. Both forks are clean. Qualified runtime `c3ad104` and its
all-lane payload-fence contract remain unchanged. The result does not establish
that inactive fences are a recoverable whole-step cost; future candidates need
a distinct mechanism and the same correctness, timing and matched SDK gates.

Evidence remains under `target/active-payload-fences/`. Manifest
`target/megakernel-results/active-payload-fences/summary.json` independently
verifies **611 artifacts and 54 completed study scopes**, SHA-256
`03b0e5a21385f6262af91eaec9b57f698198b11e04d733acfcff58920a689d0c`.
Peak job memory is **3,719,712,768 bytes**, with zero memory-limit or OOM events.
Heavy work uses separate verified 16/20/1 GiB cgroups, GPU jobs serialize,
builds/copies/heavy analysis stay outside timing, and artifacts stay on local
`/x`. Baseline parity remains unmet.

## Disjoint ready-word scans

Starting from `8777d02`, this candidate records which authoritative ready-word
helpers the advisory scan has already called. The fallback visits only the
other words. A separate template applies to 2–32 ready words; single-word
graphs and graphs above 1,024 nodes retain their original templates. Both model
graphs use the candidate path, with five/seventeen ready words.

The local visited mask starts empty on each scan attempt. A word is marked only
when its valid hint leads to the unchanged authoritative helper. Failure of
that helper means it examined the word and could not allocate any observed
ready bit. Before caching the sampled epoch as idle, the two passes together
therefore still examine every word. A publication after a word's visit either
belongs to the acquired completion epoch or causes a subsequent epoch change,
as in the original scan's progress argument. Skipped fallback repairs may leave
missing hints, which remain advisory. No mask survives the attempt, and removing
repeated helper calls cannot increase the existing overclaim bound.

Host checks pass **94,454 static ready/hint/worker cases** and thirty small
publication schedules. The latter use sequential consistency and do not model
Vulkan weak-memory behavior; they supplement the source-level argument, not a
formal memory-model proof. Ownership RMWs, the authoritative helper, queue
layout, every fence and barrier, handlers and all requested workers stay intact.
No new shared storage, descriptor, Naga capability or arithmetic is introduced.

The candidate passes formatting, release builds, **52 codegen checks and fourteen
whole-graph smoke tests**, including 1,024/1,025-node boundaries and repeated
1/3/256/4096-worker execution. Fifty diagnostic comparisons use five pairs per
case/mode, ten warmups, at least one warmup second and matched 100 ms prewarming.
All selection, allocation, full-state, optimizer, completion, one-dispatch and
no-fallback checks pass; maximum model error remains 4.66e-9.

Exact inverse substitutions recover both original model graph sources and the
code-generation branch. All nineteen CI shaders, forty ordinary model shaders
and their native-statistic multisets remain exact. Model registers remain
86/84, shared bytes 18,928/29,344, participating workers 168 and logical graph
tiles 22,042/83,312. All 256 requested workers remain enabled. The four SGD
phases preserve another 97,522 logical tiles. Code grows by 128 forward / 256
training bytes; queue and payload allocations do not change.

| Five-pair persistent GPU median | Qualified control | Candidate | Change |
|---|---:|---:|---:|
| CI inference | 0.030688 ms | 0.030560 ms | −0.42% |
| CI backward | 0.068128 ms | 0.067456 ms | −0.99% |
| CI backward + SGD | 0.072416 ms | 0.073920 ms | +2.08% |
| Model forward | 4.071136 ms | 4.079776 ms | +0.21% |
| Model training + SGD | 12.893696 ms | 12.854656 ms | −0.30% |

Ordinary model GPU changes remain within 0.06%. Neither model reaches the
predeclared 0.5% improvement threshold. The candidate is reverted without extra
repeats, normal completed-step timing, full library/oracle/matrix qualification
or fresh SDK captures. These diagnostics do not qualify a performance gain.

All **23 source/dependency hashes**, both qualified executable hashes and
fourteen restored smoke tests match; the temporary template is removed and
both compiler forks remain clean. Local evidence is under
`target/disjoint-ready-scans/`. Manifest
`target/megakernel-results/disjoint-ready-scans/summary.json` independently
verifies **363 artifacts and seventeen completed study scopes**, SHA-256
`ea7dd6231c8f538f9211b5a4eacb1b8b8fa352b31a808fa2b7e2b34df7fedee2`.
Peak recorded memory is **3,714,416,640 bytes**, with zero memory-limit or OOM
events. Heavy jobs use separate verified 16/20/1 GiB cgroups. GPU work is
serialized, and builds/copies/heavy analysis stay outside measurements. All
launched jobs finish; artifacts remain on local `/x` storage. Qualified runtime
`c3ad104` and its memory contract remain unchanged. Baseline parity is unmet.

## Scalar GEMV scratch pooling

Starting from `3689ef1`, this candidate represents forward tree GEMV's vector
scratch as four adjacent f32 components. The ordinary and persistent generators
share the change. In a combined graph, the existing scalar bank is large enough
to hold those elements, so the separate vector bank and its sentinel disappear.
Subgroup reductions, transposed GEMV and integer-dot templates stay unchanged.

Element `i`, component `c` maps bijectively to `4*i+c`. Reads reconstruct the
original vector before its original vector addition; stores split the result
without conversion. Each reduction step writes the first half and reads the
disjoint second half, retaining column grouping and the original barrier. The
last pair is still added by lane zero. The existing allocator separates arrays
live within a handler and packed tile groups; the tile reset and handler-boundary
barriers protect scratch reuse. Products, tree order, fused addends, storage
guards, output coordinates, fences and queue operations remain unchanged.

Qualification for the diagnostic screen passes formatting, release builds,
**52 codegen checks, fourteen whole-graph smoke tests and four GEMV parity tests
in each ordinary/strict whole-graph mode**. The GEMV suite covers reduction
widths, grouped columns, row/K tails, fused addends and normalization. Fifty
diagnostic comparisons preserve selected implementations, allocation, full
state, optimizer work, completion, one dispatch and no fallback. Maximum model
error remains 4.66e-9. Runs use five pairs per case/mode, ten warmups, at least
one warmup second and matched 100 ms prewarming.

The changed ordinary GEMV source has an exact inverse. All nineteen CI sources
and the other thirty-nine ordinary model sources remain exact. Every ordinary
native-statistic multiset also remains exact, including GEMV; this does not
establish native instruction identity. The graph audit reconstructs both entire
original sources outside the changed GEMV function, accounting for bank renaming
and the removed sentinel initialization. Within that function it checks all
sixteen reconstructed vector loads, 64 scalar loads, 32 scalar stores, eight
reduction strides, nine barriers, reset coverage, unchanged products and output
guards. This source audit supplements the layout argument, not a formal compiler
or Vulkan memory-model proof.

| Persistent model resource or GPU diagnostic | Qualified forward | Candidate forward | Qualified training | Candidate training |
|---|---:|---:|---:|---:|
| GPU time | 4.079008 ms | 4.951104 ms | 12.903296 ms | 12.837504 ms |
| Registers | 86 | 77 | 84 | 84 |
| Shared bytes | 18,928 | 14,816 | 29,344 | 25,232 |
| Code bytes | 102,400 | 102,784 | 241,664 | 242,304 |
| Participating workers | 168 | 252 | 168 | 168 |

Forward GPU time regresses **21.38%**, while training improves **0.51%**.
Ordinary GPU changes are +0.002% forward / −0.26% training. CI GPU changes are
+2.11%/+0.47%/+0.62% on unchanged shaders. All 256 requested workers remain
enabled, with 22,042/83,312 logical graph tiles and 97,522 SGD tiles in four
unchanged phases. Queue and payload allocations are unchanged.

The forward result contradicts using lower registers or shared capacity as a
speedup proxy. Increased participation is consistent with concurrency effects,
but does not measure occupancy or isolate them from the changed GEMV body,
scratch addresses and native compilation. No new counter attribution is claimed.
The candidate fails the fixed model-regression guardrail and is reverted without
extra repeats, normal timing, full library/oracle/matrix qualification or fresh
SDK captures. This result does not justify a worker cap or weakened gates.

All **24 source/dependency hashes**, both qualified binaries and fourteen
restored smoke tests match; both compiler forks remain clean. Evidence stays in
local `target/gemv-scalar-scratch/`. Manifest
`target/megakernel-results/gemv-scalar-scratch/summary.json` independently verifies
**367 artifacts and nineteen completed study scopes**, SHA-256
`d44f14b8114080911997ebaa421c2747f6882f1fa56de415cc890669d6b74034`.
Peak recorded memory is **3,715,629,056 bytes**, with zero memory-limit or OOM
events. Heavy jobs use separate verified 16/20/1 GiB cgroups; GPU work serializes
and measurements do not overlap builds, copies or heavy analysis. All launched
jobs finish. Qualified runtime `c3ad104` and its memory contract remain unchanged;
baseline parity is unmet.

## Bounded idle-poll backoff

Starting from `8e7070a`, this diagnostic adds a bounded clock loop only after
a worker leader observes the same idle epoch again. Successful claims and
new-epoch scans retain their original path. The loop compares elapsed low-word
ticks against 64 or 256, with at most sixteen iterations. A zero-delay control
returns immediately. All three probe modes use the same frozen binary and shader;
only an otherwise unused uniform word changes. Renaming `Config.nodes` to
`poll_ticks` preserves its offset, the 32-byte structure and all allocations.
The real node count used for construction and completion remains unchanged.

The loop contains no collective or shared write, and no clock-dependent value
reaches a handler or uniform broadcast. Its finite iteration bound preserves
progress independently of clock wrapping or rate. All original ownership,
publication, epoch-acquisition and payload fences retain their order. Every
requested worker remains live; there is no worker cap or global barrier.
The [shader-clock specification](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_shader_clock.html)
does not guarantee a fixed rate or provide a code-motion barrier. These values
are ticks, not nanoseconds, and the study does not measure achieved polling
intervals or atomic traffic.

A temporary Blade hook checks/enables the device-clock feature and replaces
two explicit marker calls with device-scope `OpReadClockKHR`. It validates
emitted SPIR-V before pipeline creation. An independent audit validates all
**21 input/output pairs** and reconstructs every original module byte-for-byte
by reversing only the two calls and added scope/capability/extension records.
The model instruction streams match across the three modes after removing
debug records. All 61 generated sources match across probe modes; all 56
ordinary sources match qualified execution. Exact inverses recover all five
qualified graph sources, preserving every handler, guard, fence and barrier.
No Naga source or API change is introduced.

Formatting, release builds, **52 codegen checks and fourteen whole-graph smoke
tests in each of the three delay modes** pass. The fixed **100 diagnostic
comparisons** use five pairs per case/mode, ten warmups, at least one warmup
second and 100 ms prewarming. Complete state, selection, allocation, optimizer,
completion, one-dispatch and no-fallback checks pass; maximum model error is
4.66e-9. All ordinary native-statistic records stay exact. Probe-mode records
also match each other exactly. CI records match qualified execution except
for the source-hash pipeline label changed by the unused member rename.
The initial audit incorrectly required that label to match; its failure is
preserved, and the corrected audit normalizes only that label without repeating
any GPU run.

| Persistent GPU diagnostic | Qualified | Zero delay | 64 ticks | 256 ticks |
|---|---:|---:|---:|---:|
| Model forward | 4.078880 ms | 4.079168 ms | 4.076576 ms | 4.080544 ms |
| Model training | 12.848736 ms | 12.746208 ms | 12.874432 ms | 12.757344 ms |

The 64-tick delay changes forward/training **−0.06%/+0.20% against qualified**
and **−0.06%/+1.01% against zero delay**, failing both the improvement criterion
and training guardrail. The 256-tick delay changes **+0.04%/−0.71% against
qualified**, but **+0.03%/+0.09% against zero delay**. It has no qualifying gain
against both controls. CI remains within its 3% guardrail.

All modes retain forward/training registers **86/84**, shared bytes
**18,928/29,344** and **168 participating graph workers** out of 256 requested.
The probe grows code from 102,400/241,664 to 104,576/244,096 bytes. Graph work
remains 22,042/83,312 tiles, with 97,522 SGD tiles in four phases. The failed
fixed gate ends the experiment: no extra repeats, normal timing, full
library/oracle/matrix qualification or fresh SDK captures follow. These results
do not prove polling cheap or identify a different delay policy as useful.

All **24 main source/dependency hashes**, three modified Blade sources, both
qualified binaries and fourteen restored smoke tests match. The temporary
Blade helper is removed; both compiler forks are clean. Evidence stays in
local `target/idle-poll-backoff/`. Manifest
`target/megakernel-results/idle-poll-backoff/summary.json` independently verifies
**670 artifacts and 24 completed study scopes**, SHA-256
`35d44081151e276b2b9285078d2b13f9075b6f69c96dc86a5f7b0feceacce424`.
Peak job memory is **3,716,980,736 bytes**, with zero memory-limit or OOM events.
Heavy jobs use separate verified 16/20/1 GiB cgroups. GPU jobs serialize, and
measurements do not overlap builds, copies or heavy analysis. All launched
jobs finish. Qualified runtime `c3ad104`, its memory contract and dependency
pins remain unchanged; baseline parity is unmet.

## Node-counter spacing

Starting from `b9fe9fe`, this diagnostic changes only the physical placement of
the three atomic counters per node: allocation, completion and dependency.
Dense counters remain four bytes apart; spaced counters are 128 bytes apart.
Both probe modes use the same frozen binary, generated shaders, 32-byte Config,
counter-region alignment, queue capacity, reset-copy size and histogram offset.
Only the uniform counter stride and initial counter-value positions differ.
The dense control reserves the same unused space as the spaced variant. A
separate qualified control accounts for the added address calculation, alignment
and reserved capacity. The spacing is a diagnostic choice, not a claimed
universal cache-line or atomic granularity.

Logical counter `(node, role)` maps to `base + (3*node + role)*stride`, for
stride 1 or 32 words. The base rounds past the header, ready bitmap and summary
to a multiple of 32 words. Reserving `3*N*32` words makes the mapping injective
and disjoint from the following worker histogram. Checked host arithmetic
rejects overflow before allocation; every shader intermediate is bounded by
that checked total. Initialization and completion validation use the same
mapping. Completion additionally checks that every unused padding word remains
zero, including the aligned gap and the dense control's unused tail.

Each logical counter retains its original modification order, atomic operands,
ownership rules and fence/RMW chains. No atomic operation, fence, shader branch,
eligibility rule or worker count changes. The unused Config node-count word
becomes the stride at the same offset; actual node construction and completion
counts remain intact. All immutable node metadata, dependencies, parameters,
descriptors, handler arithmetic and payload allocation keep their meaning.
Naga and Blade remain unchanged.

Formatting, release builds, **52 codegen checks and fourteen whole-graph smoke
tests in each layout** pass. The fixed **75 diagnostic comparisons** use
qualified/dense/spaced visits, five pairs per case/visit, ten warmups, at least
one warmup second and 100 ms prewarming. All full-state, selected-implementation,
optimizer, completion, one-dispatch and no-fallback checks pass. Maximum model
error remains 4.66e-9; padding checks pass throughout.

All **61 probe sources** match between layouts. All **56 ordinary sources** and
their native-statistic multisets match qualified execution. Inverting only the
uniform-member rename and counter-address expressions recovers all five entire
qualified graph sources, including every handler, guard, fence and atomic.
Complete native-statistic multisets also match between probe layouts. The audit
checks the actual uniform stride, counter base, histogram offset and allocation
delta for every case. These are source/layout checks, not a formal compiler or
Vulkan memory-model proof.

| Persistent GPU diagnostic | Qualified | Dense probe | Spaced probe |
|---|---:|---:|---:|
| Model forward | 4.073120 ms | 4.085088 ms | 4.118784 ms |
| Model training | 12.738688 ms | 12.732992 ms | 12.953120 ms |

Spacing regresses forward/training **0.82%/1.73% against dense** and
**1.12%/1.68% against qualified**. Both model guardrails fail, and neither
phase meets the improvement criterion. Ordinary model GPU changes stay within 0.11%
against qualified; CI remains within its 3% regression guardrail. All samples
are preserved. The failed screen does not justify an extra spacing sweep,
normal timing, full library/oracle/matrix qualification or fresh SDK captures.

Forward/training registers remain **86/84**, shared bytes **18,928/29,344**, and
participating graph workers **168** out of 256 requested. The probe adds 128
code bytes to each model pipeline. Initial and live queue padding together add
**114,768/392,184 scheduler bytes**, exactly matched between probe layouts;
payload storage is unchanged. Work remains 22,042/83,312 graph tiles and 97,522
SGD tiles in four phases. This test changes counter placement without changing
the operations, but does not identify a cache or atomic-traffic cause for the
observed slowdown. Lower contention was a hypothesis, not an established fact.

All **24 source/dependency hashes**, both qualified binaries and fourteen
restored smoke tests match. Both forks are clean. Local
`target/counter-spacing/` preserves the proof, policy, patches and captures.
Manifest `target/megakernel-results/counter-spacing/summary.json` independently
verifies **479 artifacts and seventeen completed study scopes**, SHA-256
`8248db4d8b628387e0f84adceae2aba7193ead4e022a18a879a612bd7b4f2d1f`.
Peak job memory is **3,716,063,232 bytes**, with zero memory-limit or OOM events.
Heavy work uses separate verified 16/20/1 GiB cgroups, GPU jobs serialize,
and no build, copy or heavy analysis overlaps measurement. All launched jobs
finish. Qualified runtime `c3ad104`, its memory contract and dependency pins
remain unchanged; baseline parity is unmet.

## Completion-epoch control on the current runtime

This counterfactual starts from `f28a00f`, with qualified runtime `3f95fa4`.
It selects the original continuous bitmap scanner for every graph while keeping
the latest matrix, packing, inactive-read and prologue optimizations. Single-word
CI shaders stay exact. Multiword graphs remove only two private epoch counters,
the epoch-aware claim function and the extra fence between successor publication
and the remaining-node decrement.

Successful claims retain their acquire fence. Tile payload fences, completion
fan-in, dependency RMW fences and release before ready-bit publication remain.
Continuous rescanning does not cache an empty scan by completion epoch, so it
does not need the additional epoch-observation ordering. Empty scans retry;
workers terminate only at global completion. No worker is retired or capped.
Queue layout, buffers, selected arithmetic and logical work stay unchanged.

Generated-source audits recover the entire candidate from those exact changes.
Every handler body, all nineteen CI sources and all forty ordinary model sources
remain byte-identical. The first audit's regular-expression extractor stopped
at a nested column-zero closing brace in the entry point. Its failed log and
script are preserved; a brace-balanced extractor verifies the full source
inverse without repeating GPU work.

The control passes **676 library tests (eight ignored), 168 strict whole-graph
oracle cases, 51 matrix regressions, fourteen whole-graph smoke tests and 52
codegen checks**, plus strict all-target/all-feature Clippy, formatting, release
builds and Metal-target compilation. The qualified rebuild matches both frozen
candidate binaries exactly. No Metal GPU execution or additional hardware is
qualified; neither compiler fork changes.

Fifty diagnostic comparisons use five pairs per case/mode and matched 100 ms
prewarming. Full state, selection, one dispatch, no fallback and complete work
accounting pass; maximum model error remains 4.66e-9.

| Model phase | Persistent GPU interval, control → continuous | Change | Ordinary GPU change |
|---|---:|---:|---:|
| Forward | 4.098400 → 4.049888 ms | −1.18% | +0.05% |
| Training + SGD | 13.065216 → 12.959072 ms | −0.81% | +0.06% |

Forward/training registers fall **86/84→84/82**, code shrinks
99,584/238,720→99,072/238,336 bytes, and shared memory stays 18,928/29,344 bytes.
Both modes have **168 participating model workers**, all 256 requested workers
enabled, and 22,042/83,312 logical tiles. The earlier epoch qualification changed
training participation from 252 to 168. That earlier resource effect does not
recur here. This counterfactual still changes generated code and registers, so
it does not isolate the cost of polling from compiler/resource effects.

The normal gate completes **1,000 comparisons**, 500 original-protocol and 500
with matched 100 ms prewarming. Each uses control/candidate/candidate/control
visits, 25 interleaved pairs per case/visit, at least ten warmups and one warmup
second. Every state/execution gate passes and all samples are preserved.

| Persistent completed step | Epoch control, ms | Continuous scanning, ms | Candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| Original forward | 4.619804 | 4.636788 | 1.0037 [0.9943, 1.0666] |
| Original training + SGD | 15.309254 | 15.260173 | 0.9968 [0.9769, 1.0173] |
| Prewarmed forward | 4.382197 | 4.329527 | 0.9880 [0.9267, 1.0103] |
| Prewarmed training + SGD | 13.583753 | 13.377017 | 0.9848 [0.9746, 1.0042] |

All four absolute model intervals include one. Prewarmed normalized training
improves 0.48%, ratio 0.9952 [0.9917, 0.9988], but original normalized intervals
include no change. Pooled bootstrap intervals omit some between-visit variation.
CI timing moves on identical shaders; no shader gain is attributed to those
cases. The direct forward comparison is complete, and **the epoch policy remains
retained** because its removal establishes no absolute completed-step gain.
No fixed confirmation or fresh SDK capture is run; prepared SDK scripts are
archived but were not executed.

Fresh retained-control paired persistent/ordinary medians are **1.271× forward /
1.093× training**, or **1.279× / 1.164×** prewarmed. They are medians of paired
ratios, not ratios of aggregate medians. Baseline parity remains unmet.
Nineteen source/dependency hashes and both rebuilt benchmark binaries match the
qualified control exactly; fourteen restored smoke tests pass. Both forks remain
clean. The implementation is fully reverted.

Evidence remains local in `target/completion-epoch-revisit/`. Manifest
`target/megakernel-results/completion-epoch-revisit/summary.json` independently
verifies **396 artifact hashes and 38 completed study cgroups**, SHA-256
`6fa2e4a4abd16b198411500a771e2dd9ba00ef3fcc1df6d94c1ca9ecbff42644`.
Peak recorded job memory is **3,722,444,800 bytes**, with zero memory-limit events.
All heavy jobs use verified 16/20/1 GiB memory-high/hard/swap limits, GPU jobs are
serialized, and builds/copies/heavy analysis stay outside measurements. All jobs
finish; no shared-drive artifacts are created.

The subsequent [advisory summary](#advisory-ready-word-summary) skips empty
words while retaining a full-scan fallback before epoch waiting. Its argument
allows stale hints and preserves authoritative ownership and progress checks.

## Immutable ready-bitmap width

Two probes start from `1b0b2d3`, against qualified runtime `3f95fa4`. They replace
the ready-bitmap scan bound and modulo divisor with the width already known
when generating the graph. The first specializes every width; the refinement
keeps single-word graphs byte-identical and specializes only the multiword
completion-epoch scheduler. It preserves the exact u32 expressions, overflow
behavior, scan order, atomics, fences, completion policy and requested workers.
It does not introduce the previously rejected last-successful-word cursor.

`Graph::record` derives width from the final work length, including optimizer
nodes, and supplies the same value to queue allocation, configuration and source
generation. Source participates in the pipeline cache key. Frame reuse requires
matching graph nodes, work length, handlers, grids and bindings; parameter-only
updates preserve width. External recordings own immutable frames through
submission retirement. Empty graphs return before queue construction.

Source audits reverse exactly two replacements within `graph_claim`; every
other byte is unchanged. The first changes five graphs, with widths 1/1/1/5/17,
while preserving sixteen ordinary CI and forty ordinary model sources. The
refinement preserves **all nineteen CI sources**, including their three graph
shaders, and changes only the two model graphs. All forty-two model sources
match the first candidate exactly. No handler, arithmetic selection, allocation,
scratch size or logical work changes. Model resources remain **86/84 registers,
18,928/29,344 shared bytes and 168 participating workers**, with all 256 requested
workers enabled. Code shrinks 99,584→99,200 forward and 238,720→238,464 training
bytes. Logical work remains 22,042/83,312 tiles.

Each candidate passes **676 library tests (eight ignored), 168 strict whole-graph
oracle cases, 51 ordinary matrix regressions, fourteen whole-graph smoke tests
and 52 codegen checks**, plus strict all-target/all-feature Clippy, formatting,
release builds and Metal-target compilation. Existing smoke coverage includes
32/33/65-node bitmap boundaries, poisoned storage, parameter/optimizer changes,
external recording snapshots, repeated execution and 1/3/256/4096 workers.
No Metal GPU execution or additional hardware is qualified. Qualified rebuilds
match both frozen candidate binaries exactly; both compiler forks are unchanged.

Each probe completes 1,000 normal completed-step comparisons: 500 original and
500 with matched 100 ms prewarming, control/candidate/candidate/control visits,
25 interleaved pairs per case/visit, at least ten warmups and one warmup second.
All five cases pass matching implementation, complete-state, logical-work,
one-dispatch and no-fallback checks. Fifty separate diagnostic comparisons per
candidate also pass. Maximum absolute model error remains 4.66e-9.

| Probe | Protocol / model phase | Control, ms | Candidate, ms | Candidate/control, bootstrap 95% interval |
|---|---|---:|---:|---:|
| All widths | Original forward | 4.695474 | 4.634854 | 0.9871 [0.9796, 0.9982] |
| All widths | Original training + SGD | 15.490774 | 15.536485 | 1.0030 [0.9975, 1.0189] |
| All widths | Prewarmed forward | 4.386577 | 4.684283 | 1.0679 [0.9528, 1.0764] |
| All widths | Prewarmed training + SGD | 13.446528 | 13.340097 | 0.9921 [0.9739, 0.9950] |
| Multiword only | Original forward | 4.710884 | 4.659223 | 0.9890 [0.9830, 0.9974] |
| Multiword only | Original training + SGD | 15.580531 | 15.542225 | 0.9975 [0.9810, 1.0138] |
| Multiword only | Prewarmed forward | 4.415392 | 4.548398 | 1.0301 [0.9366, 1.0699] |
| Multiword only | Prewarmed training + SGD | 13.448068 | 13.379692 | 0.9949 [0.9763, 1.0152] |

The first candidate's CI SGD medians regress 3.3% original / 3.5% prewarmed;
their intervals include no change. The refinement restores the exact CI
shaders. Remaining CI timing shifts therefore cannot establish a shader gain.
Refined original forward's apparent 1.10% absolute gain tracks ordinary's
1.15% improvement. Refined prewarmed training improves the paired persistent/
ordinary ratio 0.74% [0.42%, 0.99%], but its absolute interval includes no change.
Pooled bootstrap intervals omit some between-visit variation.

Both diagnostic visits measure about **0.9% lower persistent training GPU
time**; forward remains within 0.2%. Four matched SDK backend captures, sixteen
metric groups, compare only refined training plus SGD. Persistent replay
improves **13.396912→13.260011 ms (1.02%)**, versus ordinary
11.715802→11.735737 ms (+0.17%). Estimated persistent warp instructions fall
1.04%; integer-pipe work changes −0.10%, FP32 −0.13%, global-load sectors +0.01%
and shared wavefronts −0.09%. Local traffic stays zero. Barrier waiting changes
47.03→46.51% of active warp cycles; this is not an elapsed-time fraction.
Active compute warps stay near 32.6% of peak. Counter rates multiplied by replay
duration are estimates: unchanged ordinary shader work also shifts by about
1.2% in these estimates. Maximum clock/instruction-rate group spreads are
0.50%/1.19%. These counters support a small GPU-time change without isolating
its cause or establishing an end-to-end win.

All four actual SDK training states contain **246,883,105 floats / 987,532,420
bytes**. Control and candidate match bit-for-bit within each backend; ordinary
and persistent states pass the numerical budget but are not bit-identical.
Ordinary SHA-256 is `f67022044ae8966ac48c5d62cd0907411fb42f4d1864c1ba0b3d555b17ce7be0`;
persistent SHA-256 is `1e359875fa92965d4b0aa9026c050702ae3ac165aa59c0a4cc74a6381589c3b7`.
SDK replay uses zero-rate SGD for stable state, separately from normal training.
There are no fresh forward or first-candidate SDK captures.

Because refined absolute training intervals included no change, a predeclared
confirmation requires an absolute model completed-step benefit before retention.
It reverses visit order to candidate/control/control/candidate and completes
another **1,000 comparisons**, with the same five cases, protocols and checks.
All pass. This brings the study to **3,000 completed-step comparisons**, plus
100 diagnostics, without removing any sample or repeating until success.

| Confirmation, persistent completed step | Control, ms | Candidate, ms | Candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| Original forward | 4.683378 | 4.678629 | 0.9990 [0.9421, 1.0190] |
| Original training + SGD | 15.363919 | 15.368989 | 1.0003 [0.9724, 1.0165] |
| Prewarmed forward | 4.383697 | 4.384576 | 1.0002 [0.9638, 1.0666] |
| Prewarmed training + SGD | 13.458263 | 13.351132 | 0.9920 [0.9708, 1.0105] |

Prewarmed normalized training improves again, by **0.87%**, with ratio
0.9913 [0.9882, 0.9949]. However, every absolute model interval includes one;
the prewarmed absolute training estimate is −0.80% [−2.92%, +1.05%]. Thus a
small steady-state GPU/relative benefit repeats, but no absolute completed-step
benefit survives the confirmation criterion. **Both probes are reverted**;
this is inconclusive absolute timing evidence, not proof of zero GPU benefit.
No further confirmation is run. Fresh qualified-control paired persistent/
ordinary medians are **1.309× forward / 1.093× training**, or **1.275× / 1.164×**
prewarmed. These are medians of paired ratios, not ratios of aggregate medians.

All nineteen source/dependency hashes and both rebuilt benchmark binaries match
the qualified control exactly; fourteen restored smoke tests pass. Both compiler
forks remain clean. Runtime stays `3f95fa4`; baseline parity remains unmet.

Artifacts remain local in `target/ready-width-constant/` and
`target/ready-width-multiword/`. Manifest
`target/megakernel-results/ready-width-specialization/summary.json` independently
verifies **908 artifact hashes and 98 completed bounded study cgroups**,
SHA-256 `f4f7a6576cce8a9ab0bd2b6cd5fb10f349080f28ddea7d4aa132a5c71a10599b`.
Peak recorded job memory is **4,717,015,040 bytes**, with zero memory-limit events.
Heavy jobs use verified 16/20/1 GiB memory-high/hard/swap limits, GPU work is
serialized, and builds/copies/heavy analysis stay outside measurements. All
launched jobs finish. The subsequent
[epoch control](#completion-epoch-control-on-the-current-runtime) performs the
deferred direct forward comparison against the latest qualified runtime.

## Cooperative matrix scratch initialization

Two controls start from `d0caa54`, retaining runtime `3f95fa4`. They omit converter
clears for selected non-compensated 16×16 f16-input matrix, prologue and epilogue
generators. Typed, horizontal, convolution, compensated, 8×8 and f32-input
generators retain resets. The first keeps the converter entry barrier; the
second removes it. All original shader and worker barriers, fences, inactive
sentinels and guards remain. Ordinary shaders and arithmetic selection do not
change. Each model phase selects sixteen affected projection nodes.

Direct, transposed and scalar staging each cover all 256 operand cells, with
explicit padding writes before reads. Prologue caches initialize all 32 rows
before their barrier and the retained per-lane factor loads. A zero-K path
never reads operand scratch; initialized accumulators still supply results.
Epilogue cooperative stores cover all 1,024 result cells before scatter.
Index checks enumerate 3,564 parameter/tile cases, including 324 zero-K cases,
and result ownership for subgroup widths 4–128 without equating subgroup lanes
with local invocation IDs. Full-result coverage relies on the original
cooperative intrinsic semantics. Packed groups retain disjoint scratch; worker
barriers order prior reuse even when the converter entry barrier is omitted.

Each probe passes **52 codegen checks, 51 ordinary matrix regressions, fourteen
whole-graph smoke tests and fifty full-state diagnostics**. The expanded strict
matrix sweep is **44 passing / seven rejected** for both candidates and an
independently rebuilt, unmodified baseline. Two rectangular cases require their
physical cooperative workgroup size; five typed cases encounter unsupported
graph scalar types. Test names and admission errors match exactly. These are
existing admission limits, not a wholly passing strict suite; no numerical
failure occurs in the admitted cases. Smoke coverage includes poisoned scratch,
ragged K/rows, changed inputs/factors and 1/3/64/4096 workers.

Five diagnostic pairs per case/mode use matched 100 ms prewarming. Selection,
full-state budgets, completion, one dispatch and no fallback all pass; maximum
absolute error is 4.66e-9. All 256 requested workers remain enabled, with 168
participating model workers and 22,042/83,312 logical tiles.

| Probe | Phase | Persistent GPU interval, control / candidate | Change | Ordinary GPU change |
|---|---|---:|---:|---:|
| Omit clears | Forward | 4.099456 / 4.092960 ms | −0.16% | +0.02% |
| Omit clears | Training + SGD | 13.046176 / 13.061536 ms | +0.12% | +0.04% |
| Omit clears and entry barrier | Forward | 4.100032 / 4.096992 ms | −0.07% | −0.05% |
| Omit clears and entry barrier | Training + SGD | 13.058848 / 13.063136 ms | +0.03% | +0.22% |

Both retain 18,928/29,344 shared bytes and 86 forward registers; training
registers fall 84→82. Code size falls from 99,584/238,720 to 98,944/238,080 bytes
for clears alone and 98,816/237,952 with the entry barrier removed. Neither
establishes a useful speedup. Wall timings vary more than GPU intervals; no
completed-step gain, normal retention gate or fresh SDK capture is claimed.

All nineteen CI and forty model sources remain byte-identical, including all
fifty-six ordinary shaders. Each model graph changes one matrix handler. Source
audits isolate four reset loops (800 forward / 1,024 training cells), their
cursor declarations and, in the second probe, one entry barrier. Every remaining
handler token matches after one-to-one temporary renaming; all source outside
the handler is exact. No scratch capacity, handler arithmetic or selection
changes. A formatting failure is corrected before building and preserved.

Both probes and wrappers are reverted. Nineteen original source/dependency
hashes and both qualified binaries match exactly; fourteen restored smoke tests
pass. Both compiler forks remain clean. Evidence is local in
`target/coop-matrix-clear/` and `target/coop-matrix-clear-barrier/`. Manifest
`target/megakernel-results/coop-matrix-clearing/summary.json` independently
verifies **660 artifact hashes and 37 completed bounded cgroups**, SHA-256
`19d44db268f658048323a1501c9c48d97be4616456c14ce2943186dd9da88509`.
Peak recorded job memory is **3,716,595,712 bytes**, with zero memory-limit events.
All heavy work uses separate verified `ml.slice` scopes, GPU jobs are serialized,
and artifacts stay on local `/x`. All jobs finish. Baseline parity remains unmet.

The subsequent [ready-width study](#immutable-ready-bitmap-width) specializes
the scan bound and modulo divisor while preserving word order, claims,
publication and workers. It verifies frame reuse and pipeline width agreement.

## Cooperative attention scratch initialization

Two independent controls start from source revision `701602e` and qualified
runtime `3f95fa4`. They omit converter-generated scratch clears only for
whole-graph `FlashAttentionCoop` and `FlashAttentionCoopF32` handlers. Ordinary
shaders, regional execution and appended commands retain their initialization.
The first preserves the converter entry barrier; the second also omits that
now-empty barrier. Both preserve every original attention staging/reuse barrier,
worker rendezvous, payload fence, inactive sentinel and access guard. Neither
changes a shader template, dependency, compiler fork or selected implementation.

The storage audit establishes complete writes before every scratch read:
Q staging covers all 16×head-width cells, including explicit zeros for padded
queries. Each full key tile fills transposed K and V before its barrier. The
publishing subgroup stores the entire 16×16 score matrix before its barrier.
PV consumes those initialized cells; scalar tails use initialized Q and global
K/V. Paths without full key tiles never read K/V/score scratch. Worker barriers
order prior claims before scratch reuse, and first use requires no prior values.
Index-domain checks cover head widths 16–512 and 3,913 sequence/mask/window
cases, including 3,364 full key tiles and 15,898 scalar tail keys. Full-score
coverage relies on the cooperative store semantics already required by the
original shader; these checks complement GPU tests rather than proving arbitrary
generated shaders.

Each probe passes **52 codegen checks, 22 attention reference tests in each
execution mode, fourteen whole-graph smoke tests and fifty full-state diagnostic
comparisons**. The strict reference run uses 64 workers; diagnostics request 256.
Five pairs per case/mode use matched 100 ms prewarming and the existing minimum
warmup count/time. Selection, numerical budgets, completion, one dispatch and
absence of fallback pass. Maximum absolute error remains 4.66e-9.

| Probe | Phase | Persistent GPU interval, control / candidate | Change | Ordinary GPU change |
|---|---|---:|---:|---:|
| Omit clears | Forward | 4.095392 / 4.089824 ms | −0.14% | −0.04% |
| Omit clears | Training + SGD | 13.041344 / 13.062976 ms | +0.17% | +0.01% |
| Omit clears and entry barrier | Forward | 4.089312 / 4.088832 ms | −0.01% | +0.05% |
| Omit clears and entry barrier | Training + SGD | 13.064448 / 13.040768 ms | −0.18% | +0.23% |

Both retain **86/84 forward/training registers**, **18,928/29,344 shared bytes**,
168 participating model workers and all 256 requested workers. Logical work
remains 22,042/83,312 tiles. Compiled code shrinks from 99,584/238,720 to
98,816/238,080 bytes in both probes. Neither establishes a useful gain. Completed
wall times vary more than GPU intervals, including slower candidate forward
visits; no wall-time improvement is claimed. No normal retention gate, full
library/oracle/matrix qualification or fresh SDK capture is claimed.

All nineteen CI and forty model sources remain byte-identical, including all
fifty-six ordinary shaders. Each model graph changes exactly one attention
handler. Source audits remove four reset loops covering 3,328 cells and their
cursor declarations; the second additionally removes the entry barrier. Every
remaining handler token matches after one-to-one generated-temporary renaming,
and all source outside the handler matches exactly. Application identifiers and
field names remain intact. The first audit initially missed temporary names
such as `_e637` and `_e275_1`; the corrected audit passes, with the original script
and failed log preserved. No GPU work is rerun. Temporary wrappers produce a
dead-code warning outside tests; no Clippy qualification is claimed for them.

Both probes and their wrappers are removed. All nineteen original source and
dependency hashes and both qualified benchmark hashes match exactly; fourteen
restored smoke tests pass. Both compiler forks are clean and unchanged. Local
evidence is in `target/coop-attention-clear/` and
`target/coop-attention-clear-barrier/`. The independently verified manifest
`target/megakernel-results/coop-attention-clearing/summary.json` records
**659 artifact hashes and 32 completed bounded cgroups**, SHA-256
`498d0a1ec49a38910c73767dc509cde14089c6f40b0a187c1824d53784cb2ab9`.
Peak recorded job memory is **3,719,450,624 bytes**, with zero memory-limit events.
GPU jobs are serialized, heavy work uses separate verified `ml.slice` scopes,
and artifacts stay on local `/x`. All launched jobs finish. Runtime remains
`3f95fa4`; baseline parity remains unmet.

The subsequent [matrix controls](#cooperative-matrix-scratch-initialization)
audit their different staging, prologue and result storage, but also establish
no gain. These results do not identify a unique cause of the remaining overhead.

## Cooperative attention subgroup sharing

Two independent follow-ups to the shared-memory softmax probe keep all original
workgroup barriers, shared arrays and per-output arithmetic. Both select the
same shader in ordinary and persistent execution, gated by a temporary Blade
capability reporting compute-stage `SHUFFLE | BALLOT` support. Unsupported
devices retain the original generator and template. Naga remains unchanged.

Neither probe assumes that local invocation IDs match subgroup lane IDs.
They use ballots to find active lanes and shuffles to check their row IDs before
sharing. This respects the active-source requirement for
[WGSL subgroup shuffles](https://www.w3.org/TR/WGSL/).

The **producer-sharing probe** proposes the first lane in each native four-lane
cluster. It falls back to the caller's own lane if that source is inactive or
owns another query row. Every selected source computes its own maximum and
probability. All lanes retain the original correction, serial sum and PV order.
A symbolic audit checks 6,624 active-lane/row assignments, including exhaustive
four-lane cases and larger groups with holes and arbitrary row assignments.

The **distributed-key probe** requires all active lanes in a subgroup to belong
to complete, equal-row four-lane clusters. Otherwise it uses scalar computation.
Each fast-path lane loads/scales four keys and computes four probabilities.
Shuffles gather keys **0 through 15 in the original order** for the maximum,
serial sum and PV accumulation; even the maximum sequence is not reassociated.
A second ballot makes the fast-path condition subgroup-uniform. A symbolic
audit checks 6,630 assignments and the exact gathered key sequence. Speculative
masked score reads remain within initialized scratch and retain inactive guards.

Each probe passes **53 codegen checks, 22 attention reference tests in each
execution mode, fourteen whole-graph smoke tests and fifty full-state diagnostic
comparisons**. The added codegen test validates subgroup control flow and exact
shared-storage accounting for both operand precisions and head widths 16–512.
The five-pair visits use matched 100 ms prewarming and the existing minimum
warmup count/time. Selection, numerical budgets, completion, one dispatch and
absence of fallback all pass; maximum error remains 4.66e-9.

| Probe | Phase | Persistent GPU interval, control / candidate | Change | Ordinary GPU change |
|---|---|---:|---:|---:|
| Producer sharing | Forward | 4.097664 / 4.118048 ms | +0.50% | +1.06% |
| Producer sharing | Training + SGD | 13.042080 / 13.093248 ms | +0.39% | +0.36% |
| Distributed keys | Forward | 4.095328 / 4.129984 ms | +0.85% | +1.10% |
| Distributed keys | Training + SGD | 13.070464 / 13.107744 ms | +0.29% | +0.20% |

Both retain **18,928/29,344 shared bytes**, 168 participating model workers and
all 256 requested workers. Graph work remains 22,042/83,312 tiles. Producer
sharing retains 86/84 forward/training registers and adds 384 code bytes to each
graph. Distributed keys uses 85/84 registers and code sizes 101,120/243,968 bytes,
versus 99,584/238,720 control. Neither establishes a gain. Removing additional
workgroup synchronization alone does not rescue these sharing schemes; the
measurements do not identify one hardware cost as the cause.

All nineteen CI and thirty-nine model sources remain byte-identical. Exact
ordinary-source inverses isolate the changed row/probability blocks and lane
argument. Each graph has one changed attention handler; all source outside it
matches after removing the added lane argument at its call site. Barrier counts,
scratch declarations, score calculation, tails, dependencies and fences remain.
No normal retention gate, full library/oracle/matrix qualification or fresh SDK
capture is claimed for either rejected probe.

Both shader probes, their codegen test, the temporary Blade capability and local
dependency override are removed. All original source/dependency hashes and both
qualified benchmark hashes match exactly, and fourteen restored smoke tests
pass. Both compiler forks are clean. No test, capture or audit fails or is rerun.

Local evidence is in `target/coop-attention-subgroup-softmax/` and
`target/coop-attention-distributed-softmax/`. The independently verified manifest
`target/megakernel-results/coop-attention-subgroup-sharing/summary.json` records
**675 artifact hashes and 30 completed bounded cgroups**, SHA-256
`81dd95e9110af58d4e152de8199305a70d89ab397976bbc6249c4b3220e8178c`.
Peak recorded job memory is **3,718,881,280 bytes**, with zero memory-limit events.
All heavy jobs use separate verified `ml.slice` scopes, GPU jobs are serialized,
and artifacts stay on local `/x`. All launched jobs finish. Runtime remains
`3f95fa4`; baseline parity remains unmet.

The subsequent [scratch-initialization controls](#cooperative-attention-scratch-initialization)
prove complete writes across forward-attention tails, masks and reuse, but
omitting converter clears and their entry barrier establishes no gain. Both are
reverted, separately from the rejected F32 scalar scratch-clear probe.

## Cooperative attention ownership and row softmax

Two independent probes start from qualified runtime `3f95fa4`, using its archived
`validated-*` binaries as controls. Neither changes worker counts, selected
implementations, graph dependencies, queue atomics, fences or submission scope.
The same forward shader serves ordinary and persistent execution.

The **score-owner probe** places cooperative Q/K loads and accumulation inside
the existing `sg == 0u` publication condition. The subgroup that writes the
score tile performs the same reduction in the same order; other subgroups'
discarded private results are unused. The predicate is subgroup-uniform, and
all workgroup barriers remain unconditional. This does not eliminate inactive
logical groups or assume a mapping between local invocation and subgroup IDs.

The **row-softmax probe** starts independently from the qualified shader. One
of each query row's four output chunks computes its maximum, probabilities and
serial sum, then publishes the results to the other chunks through shared
memory and one additional workgroup barrier. The score array becomes probability
storage after the complete row-maximum pass. Two new f32[16] arrays hold the
maximum and sum; every row, including padding, is initialized. The original
correction, scalar PV order, scalar tails, masking and all existing barriers
remain. Shared-memory admission accounts for the additional **128 bytes** in
both operand-precision specializations. Graph scratch adds **160 bytes** after
inactive-sentinel padding, and both new statistic reads retain their guards.

Each probe passes **52 codegen checks, 22 attention reference tests in ordinary
execution, the same 22 tests in strict whole-graph execution, fourteen whole-graph
smoke tests and fifty diagnostic comparisons**. The diagnostic protocol uses
five pairs per case/mode, 100 ms matched prewarming and at least ten warmups and
one warmup second. All complete-state budgets, selection, completion, one-dispatch
and no-fallback checks pass; maximum error remains 4.66e-9. This is a preflight,
not the normal completed-step retention gate or full library/oracle/matrix suite.

| Probe | Model phase | Persistent GPU interval, control / candidate | Change | Ordinary GPU change |
|---|---|---:|---:|---:|
| Score owner | Forward | 4.091200 / 4.094368 ms | +0.08% | +0.00% |
| Score owner | Training + SGD | 13.067264 / 13.051168 ms | −0.12% | −0.09% |
| Shared row softmax | Forward | 4.102880 / 4.119936 ms | +0.42% | +0.24% |
| Shared row softmax | Training + SGD | 13.045152 / 13.109824 ms | +0.50% | +0.12% |

All 256 requested workers remain enabled, with 168 participating model workers.
Graph work stays 22,042 forward / 83,312 training tiles. Score ownership leaves
forward/training register counts at 86/84, shared storage at 18,928/29,344 bytes
and code sizes at 99,584/238,720 bytes. Shared softmax raises registers to **94/88**,
shared storage to **19,088/29,504 bytes**, and code sizes to **103,040/242,048 bytes**.
No gain is established; source-level removal of duplicated work alone is
insufficient evidence for retention.

All nineteen CI sources and thirty-nine model sources are byte-identical in
each probe. Exact ordinary-source inverses isolate the intended score block or
softmax block and declarations. Each model graph has one changed attention
handler. For shared softmax, the audit first reverses the two new scratch arrays,
their inactive initialization and the corresponding renumbering of existing
f16 scratch slots; remaining differences outside attention are blank lines.
Other handler arithmetic, scratch ownership and guards remain unchanged.

Four matched **row-softmax forward SDK backend captures / sixteen metric groups**
measure ordinary replay **3.263045→3.266990 ms (+0.12%)** and persistent replay
**4.203976→4.228145 ms (+0.57%)**. Persistent estimated warp instructions fall
0.91%, integer-pipe work 1.07%, FP32 work 1.70%, global-load sectors 1.06% and
shared wavefronts 0.48%. Local loads/stores remain zero. Barrier waiting rises
**55.29→55.57% of active warp cycles**, with active compute warps near 32.3% of
peak. These small estimates do not separate added synchronization, register
lifetime, scratch layout and code composition as causes; waiting shares are
not elapsed-time fractions. No new training or score-owner SDK result is claimed.

All four actual exported states are bitwise identical across both binaries and
backends: **147,034,513 floats / 588,138,052 bytes** each, SHA-256
`4e5d66e6ce5e95142b848651d316b4df64855a9d85669499cabc3a81089be629`.
Maximum within-capture clock/instruction-rate spreads are **0.142%/1.183%**.
The instruction-rate spread also limits interpretation of small counter changes.
No GPU test or SDK capture fails or needs a repeat.

Both probes are reverted. Source/dependency hashes and rebuilt benchmark binaries
match the qualified controls exactly; fourteen restored smoke tests pass. Both
compiler forks remain clean and unchanged. Three non-GPU audit failures are
preserved and corrected: an ordinary-shader filename prefix, an archived-source
directory name, and the initial assumption that adding scratch would leave all
other handler text unchanged. No GPU work is rerun for these corrections.

Evidence is local in `target/coop-attention-score-owner/` and
`target/coop-attention-row-softmax/`. The independently verified manifest
`target/megakernel-results/coop-attention-ownership/summary.json` records **723
artifact hashes and 34 completed bounded cgroups**, SHA-256
`f480e9ed17113cacc59404b27c066ac631f66b353bbac39e244819ac94fd4ba2`.
Peak recorded job memory is **3,717,472,256 bytes**, with zero memory-limit events.
All heavy work uses separate verified `ml.slice` scopes; GPU work is serialized,
and builds/copies/heavy analysis stay outside measurements. Artifacts remain on
local `/x`, and all launched jobs finish. Runtime remains `3f95fa4`; baseline
parity remains unmet.

The subsequent [subgroup-sharing probes](#cooperative-attention-subgroup-sharing)
avoid the extra workgroup barrier and scratch arrays, with explicit ownership
checks and unchanged arithmetic order. They also establish no gain. Fewer
active-lane operations do not establish a speedup.

## Optimizer overlap controls

Three host-side dependency probes test optimizer overlap against qualified
runtime `3f95fa4`. They preserve every original dependency and add no GPU nodes.
The first links original graph sinks to appended-command roots, so all optimizer
work waits for the graph. The second links appended commands in recording
order. The third, limited to originally independent appended commands, chains
them by increasing original ancestor count, breaking ties by recording index.
Ancestor count is a readiness heuristic, not a measured completion time.

All **61 shader sources and complete pipeline-resource multisets remain
identical** in every probe. Handler selection, logical grids, descriptors,
arithmetic, packing, scratch, queue layout, fences and atomic instructions are
unchanged. Independent audits of fifteen metadata dumps verify original-edge
inclusion, exact added edges, acyclicity, transitive prerequisites, initial
pending counts and ready bits. Removing each temporary host block recovers the
original runtime source exactly. The original graph metadata also matches
between the three controls.

Model training has 523 graph nodes and four SGD nodes. Global deferral adds
129 edges, changing 698→827 edges and scheduler/resident memory by 516 bytes.
It adds one edge and four bytes to CI SGD. Each chain adds three model edges
and twelve bytes, leaving CI dependencies unchanged. No other allocation
changes. All 256 requested workers remain enabled; model participation stays
168. Forward retains 86 registers / 18,928 shared bytes / 99,584 code bytes;
training retains 84 / 29,344 / 238,720. Graph tasks remain 22,042 forward and
83,312 training, with 97,522 additional SGD tiles in training.

Each probe passes **52 codegen checks, fourteen whole-graph smoke tests and
fifty diagnostic comparisons**. Each control/candidate visit uses five
interleaved ordinary/persistent pairs per case, at least ten warmups and one
warmup second, and 100 ms matched prewarming. Every full-state, selection,
logical-completion, one-dispatch and no-fallback check passes; maximum model
error remains 4.66e-9.

| Dependency control | Persistent training GPU interval, control / candidate, ms | Change | Ordinary training GPU change |
|---|---:|---:|---:|
| Defer all optimizer work | 13.056032 / 13.694144 | +4.89% | +0.02% |
| Chain in recording order | 13.058112 / 13.724640 | +5.10% | −0.07% |
| Chain by prerequisite count | 13.048288 / 13.064224 | +0.12% | +0.11% |

The recording-order chain also makes its last three updates wait for the full
graph, so it does not isolate inter-optimizer concurrency. The exact numbers
of original graph ancestors, in SGD recording order, are:

| Policy | SGD 523 | SGD 524 | SGD 525 | SGD 526 |
|---|---:|---:|---:|---:|
| Original | 511 | 470 | 380 | 165 |
| Global deferral | 523 | 523 | 523 | 523 |
| Recording-order chain | 511 | 523 | 523 | 523 |
| Prerequisite-count chain | 523 | 470 | 380 | 165 |

The last chain runs 526→525→524→523. Unlike the recording-order chain, it retains
the earlier graph prerequisites for the first three updates it executes. Both
chains have identical edge counts, allocations and shader bytes, yet substantially
different timings. The prerequisite-count chain's ratio of GPU medians changes
only +0.009% after ordinary normalization; no useful gain is established.
The unchanged CI backward control shifts −2.5% in one visit, and ordinary CI
inference shifts −4.9% in another, reinforcing the limits of these small samples.

Four fresh **global-deferral training SDK captures / sixteen metric groups**
confirm persistent replay **13.304061→13.973526 ms (+5.03%)**, versus ordinary
**11.670792→11.682692 ms (+0.10%)**. Estimated persistent warp instructions change
+0.43%, integer work −0.07%, FP32 work −0.10%, global-load sectors +0.27% and
shared wavefronts −0.06%. Local loads/stores remain zero. Barrier waiting rises
**47.06→48.71% of active warp cycles**, while active compute warps stay near
32.6% of peak. The extra elapsed time is not accompanied by a comparable
increase in estimated instruction or traffic totals. This supports scheduling
and overlap as contributors without identifying one stall mechanism. Counter
totals are estimated from rates and replay duration; barrier shares describe
active warp cycles, not elapsed-time fractions.

All four exported states contain **246,883,105 floats / 987,532,420 bytes** and
are independently hashed. States match bitwise across control/candidate within
each backend. Ordinary/persistent states use the unchanged numerical budget,
with maximum error 4.66e-9; they are not bitwise equal. SDK replay uses zero
learning rate for stable state. Maximum within-capture clock/instruction-rate
spreads are 0.448%/0.282%; no capture fails or requires a repeat. No fresh SDK
result is claimed for either chain or forward.

All three probes are reverted. Runtime and dependency hashes are restored,
both forks are clean, rebuilt benchmark binaries match the qualified controls
exactly, and fourteen final restored smoke tests pass. A missing-receipt error
in the first manifest attempt exposed stale study labels in a copied restoration
script. Source/binaries were already restored; an independent hash audit writes
the omitted receipts and verifies that earlier-study receipts remain unchanged.
The original scripts and failed manifest are preserved; no GPU work is rerun.

Local evidence is in `target/optimizer-deferred/`, `target/optimizer-serial/`
and `target/optimizer-ready-order/`. Manifest
`target/megakernel-results/optimizer-overlap/summary.json` contains **1,023
independently verified artifact hashes and 42 completed bounded cgroups**,
SHA-256 `d32ee878e1cd96b64371789d186b537f76ee9a3b1815be7a13b5ffc0171025a8`.
Peak recorded job memory is **4,720,439,296 bytes**, with zero memory-limit
events. Heavy jobs use separate verified 16/20/1 GiB memory-high/hard/swap scopes;
GPU jobs are serialized, timing excludes builds/copies/heavy analysis, and all
artifacts stay on local `/x`. Every launched job finishes. No normal retention
gate or full library/oracle/matrix qualification is claimed for these reverted
controls. Baseline parity remains unmet.

Next, inspect the forward cooperative-attention handler and its shared-memory
staging/synchronization. The refreshed trace spans about 20.5 µs per node versus
9.2 µs in ordinary profiling. Preserve selected arithmetic, inactive guards,
scratch ownership and required barriers; different profiling contracts still
prevent treating that span difference as directly recoverable time.

## Short-K transposed-A tile geometry

Three selector-only probes against qualified runtime `3f95fa4` test the large
transposed-A weight gradients identified by the refreshed clocks. Each admits
the existing scalar small-tile path for unpinned F32-storage products with
0<K≤64, at least 128 original 64×64 workgroups and a portable 32×32 grid.
Cooperative preference, required precision and locked schedules remain
authoritative; transposed-A addends remain excluded. Two follow-ups refine
eligible non-epilogue products to existing 32×64 and 64×32 scalar shapes.
K-stage width, accumulation order, storage types, per-output arithmetic, barriers,
inactive guards, queue operations and dependency pins are unchanged.

Exactly **66 training nodes** change selection in each probe: sixteen instances
each of 2048×720×50, 720×4096×50, 960×720×50 and 720×960×50, plus 720×720×1
and 1440×720×1. Independent per-node grid checks recover the exact total task
increase. Both backends select the same implementations. All nineteen CI shader
sources and forty model sources remain identical, including the forward graph.
The unused ordinary 64×64 AT module disappears; the square probe reuses an
existing small-tile module, while each rectangle adds one ordinary AT module.
Each probe changes the training graph module.

Each completes **52 codegen checks, nine selector tests, 51 matrix regressions,
fourteen whole-graph smoke tests and fifty diagnostic comparisons**. Diagnostics
use one control and one candidate visit, five interleaved ordinary/persistent
pairs per case, at least ten warmups and one warmup second, and 100 ms matched
prewarming. All full-state budgets, matching-selection, logical-completion,
one-dispatch and no-fallback checks pass. Maximum model error remains 4.66e-9.

| Candidate tile | Persistent training GPU interval, control / candidate, ms | Change | Ordinary training GPU change | Graph tasks |
|---|---:|---:|---:|---:|
| 32×32 | 13.059392 / 14.299648 | +9.50% | −0.25% | 83,312→153,000 |
| 32×64 | 13.057664 / 13.390336 | +2.55% | −0.11% | 83,312→106,636 |
| 64×32 | 13.054304 / 13.219520 | +1.27% | −1.18% | 83,312→107,137 |

Training measurements include SGD; the task totals above count graph work only.
All 256 requested workers remain enabled, and all model captures record 168
participating workers. Training shared memory stays 29,344 bytes. Registers
remain 84 for the first two probes and fall to 82 for 64×32. Code changes from
238,720 bytes to 224,000 / 235,392 / 234,624 bytes. Forward retains 86 registers,
18,928 shared bytes, 99,584 code bytes and 22,042 graph tasks; its GPU medians
change by at most 0.11% on identical shader sources.

Reducing per-thread accumulator storage does not produce a persistent speedup
in these diagnostics. Tile counts, repeated staging and shader composition
change together, so the results do not isolate queue claims as the cause.
These are small diagnostic samples, not a full completed-step retention gate.
No new SDK capture, full library/oracle qualification or retained optimization
is claimed. All three probes are reverted.

Two non-GPU audit errors are preserved with their original scripts and logs:
the square source audit expected one too many common modules, overlooking the
retired 64×64 AT module; the first rectangular task estimate omitted parentheses
around integer ceil factors. Corrected audits verify the removed source and
each changed node's exact grid. No GPU work is rerun for either correction.
Runtime source and dependency hashes are restored exactly, both forks are clean,
rebuilt benchmark binaries match the qualified controls byte-for-byte, and
fourteen restored smoke tests pass.

Evidence remains local in `target/short-k-at-small/`, `target/short-k-at-32x64/`
and `target/short-k-at-64x32/`. Manifest
`target/megakernel-results/short-k-at-geometry/summary.json` contains **923
independently verified artifact hashes and 43 completed bounded cgroups**,
SHA-256 `2134d9f5946db274360079ff9d96bfcba9c86752e8f03cde9b30d1313a670b8b`.
Peak recorded job memory is **3,716,517,888 bytes**, with zero memory-limit
events. All heavy jobs use separate verified 16/20/1 GiB memory-high/hard/swap
limits; GPU jobs are serialized and builds/copies/heavy analysis stay outside
timing. No jobs survive the completed investigation. Baseline parity remains open.

The subsequent [optimizer-overlap controls](#optimizer-overlap-controls) test
deferral and two serialization orders while preserving handler bytes, logical
grids, workers, fences, original dependency edges and one dispatch/submission.
None establishes an improvement; any future retained change still requires the
full completed-step and matched-counter qualification.

## Refreshed handler clock attribution

The sparse probe is refreshed against runtime `3f95fa4`, after the matrix-selector
and prologue-row changes. It records each node/worker's first and last timestamp,
consuming clock values immediately. The temporary Blade hook replaces two explicit
marker calls after Naga validation. All **37 original/transformed SPIR-V pairs**
validate, and independent reversal recovers every original binary exactly.
Removing the instrumentation also recovers the worker template exactly. All
callable handler text in the five benchmark graph modules, and all **56 ordinary
shader sources**, remain byte-identical. Naga and atomic semantics are unchanged.

Eight preflight comparisons precede **200 paired diagnostics** across all five
cases: control/probe/probe/control visits, ten pairs per case/visit, at least ten
warmups and one warmup second, and 100 ms prewarming. Every full-state budget,
selected-implementation, logical-work, one-dispatch and no-fallback check passes.
Maximum model error remains 4.66e-9. All 256 requested workers remain enabled.

| Case | Control / probe GPU interval, ms | Probe change |
|---|---:|---:|
| CI inference | 0.031056 / 0.033456 | +7.7% |
| CI backward | 0.067744 / 0.074496 | +10.0% |
| CI SGD | 0.073744 / 0.079600 | +7.9% |
| Model forward | 4.102992 / 4.181968 | +1.9% |
| Model training + SGD | 13.075872 / 13.247376 | +1.3% |

Training retains **84 registers / 29,344 shared bytes**. Forward changes from
86 to 84 registers with 18,928 shared bytes. Both model phases retain **168
participating workers**. Code grows 99,584→100,608 / 238,720→239,872 bytes.
The four timestamp words per node/worker increase forward/training scheduler
storage by 1,261,568 / 4,317,184 bytes, exactly matching the resident-memory
increase. Ordinary model GPU medians move only +0.15%/+0.03%; ordinary forward
wall time shifts −8.8%, so wall changes are not used to calibrate the probe.

Fresh ordinary profiles preserve exact complete state and match the benchmark's
selected implementations. Representative comparisons across both trace visits:

| Phase / shape | Nodes | Ordinary median, µs | Persistent median span, µs |
|---|---:|---:|---:|
| Forward `MatMul[50x4096x720]` | 16 | 49.09 | 54.47 |
| Forward `FusedMatMulAdd[50x720x2048]` | 16 | 63.49 | 68.51 |
| Forward `FlashAttentionCoop[50]` | 16 | 9.20 | 20.50 |
| Training `MatMulAT[720x4096x50]` | 16 | 36.90 | 107.99 |
| Training `MatMulAT[2048x720x50]` | 16 | 22.80 | 54.55 |
| Training `MatMulBT[50x720x4096]` | 16 | 136.22 | 160.52 |
| Training cross-attention dKV, `q=50,kv=16,h=15/5` | 8 | 106.50 | 144.61 |

The first transposed-A shape stays near 108 µs in both visits. It uses 768
logical tiles per node; the smaller shape uses 384. Their ordinary kernels use
92 registers and 16,768 shared bytes. These short-K weight-gradient products
are the next target for a geometry/claim-distribution audit, including contention
with optimizer updates. The forward prologue's remaining span difference is
substantially smaller than in the earlier clock study.

Both final training steps remain in the record: **14.902624 / 13.137280 ms**,
despite prewarming. SGD envelope unions cover 4.192 / 2.958 ms; their tails after
the last non-SGD handler cover 1.779 / 1.143 ms. The corresponding ordinary SGD
passes are both about 1.351 ms. Summing SGD spans would double-count overlap,
and the final tails are not estimates of savings from removing updates.

Ordinary profiles split dispatches into timed passes and exclude SGD from the
backward profile. Persistent node spans include gaps between claims and overlap
other work. Each trace is the final measured step of its visit; its timestamp
interval and clock envelope provide the time scale. Clock rate and compiler
code-motion barriers are not guaranteed. These observations select targets;
they do not establish additive recoverable costs or a new performance result.

Five focused graph-codegen tests and fourteen whole-graph smoke tests pass.
The initial preflight audit incorrectly required resident bytes to stay equal
despite the added timestamp queue. Its script/log are preserved; the corrected
audit verifies the exact queue/resident delta, without rerunning GPU work.
Runtime, Blade source and dependency lock are restored exactly, both forks are
clean, rebuilt benchmark binaries match the qualified controls byte-for-byte,
and fourteen restored smoke tests pass. No full retained-change qualification
or fresh SDK counter capture is claimed.

Evidence is local in `target/clock-refreshed/`. Manifest
`target/megakernel-results/clock-refreshed/summary.json` contains **431 independently
verified artifact hashes and 26 completed bounded cgroups**, SHA-256
`0170ed293fd5ddd121b551a4a5646e065509fff79ea80cbf2b4ffbdce40cca61`.
Peak recorded job memory is **4,282,998,784 bytes**, with zero memory-limit events.
Every heavy job uses verified 16/20/1 GiB memory-high/hard/swap limits; GPU work
is serialized and builds/copies/heavy analysis stay outside measurements.
All jobs finish. The runtime remains `3f95fa4`; baseline parity remains open.

## Cooperative column-factor reuse

Three probes against qualified `3f95fa4` reuse per-K-column factors across the
cooperative prologue's two A staging row bands. None establishes a useful GPU-time
gain, and all are reverted. Each retains the qualified per-row factor caching,
factor multiplication order, conversion points, selected geometry, packing,
scratch capacity, original barriers, payload fences and queue operations.

The first loads four factors into a private vector per K stage, guarded separately
by each column's K bound and the first row's validity. Both row bands index that
vector for complete and partial groups of four columns. The second uses one
complete-vector predicate and direct `.x/.y/.z/.w` references, retaining original
scalar factor reads for partial columns. The third moves the complete-vector
loads inside the first row band's existing vector branch, after its original A
loads, eliminating the additional setup branch.

A valid second-band row implies a valid first-band row, and both bands use the
same K indices. No factor mutation or output write separates the original two
loads. The new values therefore replace the same factor values without adding
global reads, extending lifetime across K stages, or introducing element-zero
fallback accesses. Each inverse audit recovers the complete ordinary control
shader exactly. In each graph, only `graph_handler_10_entry` changes; source
outside it is byte-identical. All **forty other model and nineteen CI sources**
are identical, including every training source. Separate generated-source audits
verify the row/K bounds and activity guards on all factor storage loads.

Each probe passes **52 codegen checks, two prologue regressions and fourteen
whole-graph GPU smoke tests**, including padded rows, ragged K/N tails, a single
K stage, inactive final packs, poisoned storage, changed inputs/parameters and
1/3/64/4096 workers. Each also completes fifty diagnostic comparisons across the
five cases, using five pairs per case/mode, ten warmups, at least one warmup
second and 100 ms prewarming. All state budgets, matching implementations,
logical-work totals, one-dispatch and no-fallback checks pass; maximum model
error remains 4.66e-9.

| Forward GPU diagnostic, ms | Ordinary, control → probe | Persistent, control → probe |
|---|---:|---:|
| Per-component guarded factors | 2.924992 → 2.906848 | 4.095488 → 4.334112 |
| Complete-vector setup | 2.923520 → 2.912704 | 4.093632 → 4.215872 |
| Loads inside existing vector branch | 2.924864 → 2.922720 | 4.094912 → 4.213568 |

Persistent forward GPU time regresses **5.8%, 3.0% and 2.9%**, respectively.
The first retains 86 registers and shrinks code from 99,584 to 99,072 bytes.
The second and third use **84 registers**, with 99,968 and 100,480 code bytes.
All keep 18,928 shared bytes, 168 participating workers and 22,042 forward
logical tiles. Training retains 84 registers, 29,344 shared bytes, 238,720 code
bytes, 168 participating workers and 83,312 tiles. All 256 requested workers
remain enabled. Short completed-step medians vary between visits and do not
constitute the normal performance gate.

Four matched SDK backend captures, sixteen metric groups, compare the
complete-vector setup probe with the qualified control. They confirm persistent
forward replay **4.204356→4.325156 ms (+2.87%)**, versus ordinary
**3.264280→3.247570 ms (−0.51%)**. No SDK capture is claimed for the other two
variants or training.

Persistent estimated global-load sectors fall **3.32%**, and shared wavefronts
fall **5.62%**, but estimated warp instructions rise **3.37%** and integer-pipe
work rises **2.25%**. FP32 work changes +0.03%; local loads/stores remain zero.
Average active compute warps change 32.24→32.45% of peak. Barrier waiting changes
55.51→54.52% of active warp cycles, which is not an elapsed-time fraction.
The traffic reduction does not yield a GPU-time improvement. The counters do
not isolate the added control flow, code composition or register lifetime from
changes in scheduling and polling.

Ordinary estimated global-load sectors fall 3.52% and integer work falls 4.71%.
Local load/store sectors fall from about 120,965 to 55,326, without shader or
native-instruction attribution. All four actual exported forward states are
bitwise identical: **147,034,513 floats / 588,138,052 bytes** each, SHA-256
`4e5d66e6ce5e95142b848651d316b4df64855a9d85669499cabc3a81089be629`.
Maximum within-capture clock/instruction-rate spreads are **0.429%/0.790%**;
no test or capture fails or requires a repeat.

All three probes are rejected without a normal completed-step retention gate
or full library/oracle/matrix qualification. Source and dependency hashes are
restored, both compiler forks remain unchanged and clean, rebuilt release
benchmark binaries match the qualified controls byte-for-byte, and fourteen
restored smoke tests pass.

Artifacts remain local in `target/coop-prologue-column-{factors,vector,inline}/`.
Manifest `target/megakernel-results/coop-prologue-column-factors/summary.json`
contains **1,017 independently verified artifact hashes and 48 completed bounded
cgroups**, SHA-256 `84f971d85673712991140a20491a09178cbac0c096e1e6ff3d7f66a335132892`.
Peak recorded job memory is **3,718,488,064 bytes**, with zero memory-limit events.
All heavy jobs use verified 16/20/1 GiB memory-high/hard/swap limits; GPU work is
serialized and builds/copies/heavy analysis stay outside measurements. Every
launched job finishes. The next investigation refreshes sparse per-handler
timing against the latest runtime, checking instrumentation resources and
worker participation before using it to choose another optimization target.

## Cooperative prologue factor caching

The retained change caches loop-invariant per-row prologue factors in private
scalars for the normal-layout, tile-16, uncompensated, F32-weight vector staging
path. Each of its 64 logical lanes owns four adjacent columns in each of two
16-row bands. `v4_row = lid.x >> 2` stays in 0..15 throughout the K loop, so the
lane's factor indices stay `v4_row` and `16 + v4_row`. The existing cache writes
initialize all 32 entries, including zero padding, and the original workgroup
barrier precedes the new private loads. There are no subsequent cache writes
within the handler. This also bounds the extra reads for padded rows or K=0.

Factor multiplication order, per-K-column global loads, conversion points,
staging, geometry and selected arithmetic remain unchanged. Packed inactive
groups retain the sentinel guards on both new shared loads. The change adds no
global read or cross-operation immutability assumption. Queue operations,
fences, scratch capacity, dependency pins and both compiler forks are unchanged.

The source audit finds exactly two changed modules: the ordinary forward
prologue and its whole-graph shader. The other **forty model and nineteen CI
sources are byte-identical**, including every training source. Removing the two
private declarations and reversing their uses exactly recovers the ordinary
control source. Only `graph_handler_10_entry` changes in the graph; everything
outside that function is byte-identical. The audit verifies cache writes,
original barrier, private-load ordering and inactive guards explicitly.

Forward registers rise **84→86**, code shrinks **102,400→99,584 bytes**, shared
memory stays **18,928 bytes**, and participating workers stay **168**. Training
keeps 84 registers, 29,344 shared bytes, 238,720 code bytes and 168 participating
workers. All 256 requested workers remain enabled. Logical work remains
**22,042 forward / 83,312 training tiles**. CI resources and selected
implementations are unchanged.

The **1,000 completed-step comparisons** comprise 500 original-protocol and
500 matched `--prewarm-ms 100` comparisons, with control/candidate/candidate/control
visits, 25 interleaved pairs per case/visit, at least ten warmups and one warmup
second. Every state budget, matching-selection, logical-work, single-dispatch
and no-fallback check passes. Maximum model error remains 4.66e-9.

| Persistent completed step, ms | Control | Candidate | Candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| Original forward | 4.991556 | 4.703744 | 0.9423 [0.9377, 0.9534] |
| Prewarmed forward | 4.993896 | 4.662704 | 0.9337 [0.8765, 0.9980] |
| Original training + SGD | 15.328789 | 15.274573 | 0.9965 [0.9766, 1.0198] |
| Prewarmed training + SGD | 13.710784 | 13.442557 | 0.9804 [0.9770, 1.0168] |

Ordinary forward improves 2.48% under the original protocol, with ratio
0.9752 [0.9296, 0.9983]. Its prewarmed ratio, 0.9926 [0.9346, 1.0776], is
inconclusive. Forward's normalized persistent/ordinary ratio is inconclusive
under the original protocol, 0.9799 [0.9354, 1.0231], but improves with prewarming,
0.9486 [0.9245, 0.9756]. Paired persistent/ordinary medians become
**1.310/1.069 forward/training original**, and **1.282/1.165 prewarmed**. These
use medians of paired ratios, not ratios of aggregate medians.

Negative controls remain in the record: original CI inference shifts −7.27%
persistent and −8.91% ordinary, with an inconclusive normalized interval, despite
identical shaders/resources. Prewarmed ordinary training shifts −0.28% with its
pooled interval below one, also on identical shader bytes. All other raw CI
persistent intervals and both training normalized intervals include no change.
No new CI or training gain is attributed to this patch. All samples and visits
remain archived; pooled bootstrap intervals omit some between-visit variation.
The forward gain has not been directly remeasured against the pre-epoch control.

Four fresh forward SDK backend captures, sixteen metric groups, confirm:

| SDK forward replay, ms | Control | Candidate | Change |
|---|---:|---:|---:|
| Ordinary | 3.350070 | 3.274660 | −2.25% |
| Persistent | 4.536408 | 4.223386 | −6.90% |

Persistent estimated warp instructions fall **14.63%**, integer-pipe work
**10.54%**, shared wavefronts **4.49%**, FP32 work 0.47% and global-load sectors
0.54%. Local load/store traffic remains zero. Barrier waiting rises
**53.25→55.33% of active warp cycles** while active compute warps stay near
32.3% of peak. These are rate-times-duration estimates and active-warp waiting
shares, not elapsed-time attribution. The results support reduced repeated
factor reads and integer work as contributors, alongside changed code composition
and register lifetime; they do not isolate a single instruction-level cause.

Ordinary estimated warp/integer work falls 2.96%/3.26%, and shared wavefronts
fall 9.39%. Ordinary local load/store sectors rise from about 55,320 to 120,937 /
120,946, without shader or native-instruction attribution. No training SDK
capture is claimed for this change. Maximum within-capture clock/instruction-rate
spreads are **0.323%/0.627%**; no capture fails or needs a repeat. All four actual
exported states are bitwise identical, **147,034,513 floats / 588,138,052 bytes**
each, SHA-256 `4e5d66e6ce5e95142b848651d316b4df64855a9d85669499cabc3a81089be629`.

Qualification passes **676 library tests (eight ignored), 168 strict whole-graph
oracle cases, 51 matrix regressions, fourteen whole-graph GPU smoke tests and
52 codegen checks**, plus strict all-target/all-feature Clippy, formatting,
release builds and Metal-target compilation. The expanded packed-matrix fixture
covers padded rows and a single K stage, alongside its existing ragged K/N tails,
inactive final packs, poisoned storage, changed inputs/parameters and
1/3/64/4096 workers. The new ordinary regression asserts cooperative prologue
selection at 17×719×4096 and 32×16×4096, then compares two input/weight updates
against unfused cooperative and scalar GPU execution. Metal GPU execution and
additional hardware remain unqualified.

There are **75 diagnostic comparisons**, including a 25-comparison final-binary
check. All 61 generated sources, complete pipeline-resource multisets, selections,
execution and tensor allocations match the preflight candidate exactly. An initial
source-verifier prefix error and two Clippy pattern-style failures are preserved
and corrected; neither changes generated shader behavior. No GPU test or
measurement fails or is rerun for those corrections.

Evidence stays local in `target/coop-prologue-register-factors/`. The manifest
`target/megakernel-results/coop-prologue-register-factors/summary.json` contains
**616 independently verified artifact hashes and 49 completed bounded cgroups**,
SHA-256 `9820d36d64ae8d925e199dc0b643b2425e73b972190f4f405535296005ffbc75`.
Peak recorded job memory is **3,720,613,888 bytes**, with zero memory-limit events.
Every heavy job uses verified 16/20/1 GiB memory-high/hard/swap limits; GPU jobs
serialize, and builds/copies/heavy analysis stay outside measurements. All jobs
finish. Use the archived `validated-*` binaries as the next qualified control.

## Scalar B register staging

Two probes against runtime `cf8940f` change F32 scalar matrices with at most
32 output rows per tile. Both are reverted. Selection, tile counts, K stages,
split boundaries, zero padding, arithmetic order, epilogues and shared-memory
layouts remain unchanged in both ordinary and persistent execution. Every
original workgroup barrier, payload fence, inactive-storage guard and queue
operation remains. These probes do not add the second shared stage used by the
earlier double-buffer experiment.

The lookahead probe holds one B stage in named per-lane scalars. It loads the
first stage only for a nonempty split range, writes those values into the
original shared cells, then fetches the next B stage before current arithmetic.
The next fetch is guarded by `end_k - t > K_TILE`, preventing reads beyond the
split range. The existing reuse barrier still precedes the next shared writes.
The second probe loads only the current B stage before staging A, then writes
B into shared memory before the original staging barrier. Its values need not
survive arithmetic or the reuse barrier. Both also explicitly unroll B's
per-lane load/store ownership; the comparison does not isolate scheduling from
that code-generation change.

Each passes **52 codegen checks, fourteen whole-graph GPU smoke tests and fifty
diagnostic comparisons**. The five cases use five pairs per mode, ten warmups
and at least one warmup second, 100 ms prewarming, and worker/resource
instrumentation. Every state budget, matching implementation, task-total,
single-dispatch and no-fallback check passes; maximum error remains 4.66e-9.
The lookahead's initial build fails for a missing `std::fmt::Write` import; its
source, log and exit are preserved. The import is fixed before all passing
tests and measurements.

| Diagnostic GPU time, ms | Ordinary, control → probe | Persistent, control → probe |
|---|---:|---:|
| Lookahead forward | 2.986848 → 3.177792 | 4.411840 → 4.846368 |
| Lookahead training + SGD | 10.731808 → 11.144480 | 13.062048 → 13.634016 |
| B-first forward | 2.986208 → 2.974720 | 4.410208 → 4.751904 |
| B-first training + SGD | 10.739104 → 10.427424 | 13.043680 → 13.392704 |

Lookahead increases forward/training registers from 84/84 to **92/86** and code
from 102,400/238,720 to **105,856/247,168 bytes**. B-first uses **86/82 registers**
and **102,272/238,336 code bytes**. Both keep 18,928/29,344 shared bytes,
168 participating workers, all 256 requested workers and 22,042/83,312 logical
tiles. CI registers remain 48 and shared bytes remain 9,616/9,648. In each probe,
three ordinary CI modules and seven ordinary model modules change, along with
their three/two graph modules. Thirteen CI and thirty-three model sources are
byte-identical; every changed candidate module contains the new staging locals.

The persistent slowdown survives a separate **B-first forward SDK comparison**:
ordinary replay changes **3.333440→3.317160 ms (−0.49%)**, while persistent replay
changes **4.528698→4.883405 ms (+7.83%)**. Four backend captures and sixteen groups
pass without repeats. Actual exported state is bitwise identical across both
backends and both binaries: **147,034,513 floats / 588,138,052 bytes** each,
SHA-256 `4e5d66e6ce5e95142b848651d316b4df64855a9d85669499cabc3a81089be629`.
Clock rates across all groups range from 2.832 to 2.843 GHz; maximum within-capture
clock/instruction-rate spreads are 0.371%/0.926%.

Persistent estimated warp instructions change −0.42%, integer-pipe instructions
−0.07%, FP32 work −0.07%, global-load sectors −0.11% and shared wavefronts −0.07%.
Local loads/stores remain zero. Barrier waiting changes **53.52→51.27% of active
warp cycles**; this is not an elapsed-time fraction or evidence of a latency
saving. Active compute warps remain about 32.5% of peak. Thus the slower replay
has nearly unchanged estimated work and worker participation. It supports a
latency/issue-scheduling investigation, without identifying a particular
instruction-level cause. Ordinary estimated integer work rises 22.4% and local
load/store sectors rise from about 55,312 to 127,545 per replay; these counters
do not attribute that traffic to a particular shader. Counter-rate products
remain estimates. No lookahead or training SDK capture is claimed.

Both probes are rejected without a normal completed-step retention gate or
full library/oracle/matrix qualification. Original sources, dependency hashes
and clean compiler forks are restored. Rebuilt release benchmark binaries
match the qualified controls exactly, and fourteen restored smoke tests pass.
Artifacts are under `target/scalar-b-{register-prefetch,first-staging}/`.
The manifest `target/megakernel-results/scalar-b-prefetch/summary.json` contains
**693 independently verified artifact hashes and 29 completed cgroup records**,
SHA-256 `f92e340d7a729027fb85ccfb1542260e35ca00c5d66a48ead7f5650a73b70fc6`.
Peak recorded memory is **3,717,189,632 bytes**, with zero memory-limit events.
Heavy jobs use separate verified 16/20/1 GiB cgroups and local NVMe; GPU jobs
are serialized and builds/copies/analysis do not overlap measurements.
The following [row-factor qualification](#cooperative-prologue-factor-caching)
retains loop-invariant shared factor values in registers while preserving
initialization, factor order and inactive guards. Baseline parity remains unmet.

## Single-word claim controls

Three scheduler probes against qualified `cf8940f` target graphs with at most
32 nodes. All are reverted. The retained runtime, dependency pins and both
compiler forks are unchanged; baseline parity remains unmet.

1. Load a node's allocation counter before attempting its fetch-add, rejecting
   exhausted nodes without an RMW. Only the original RMW grants ownership.
2. Cache exhausted nodes in a private bitmap used by the worker's leader.
   Only the final valid allocator clears the global ready bit; failed claimers
   remember exhaustion and skip redundant clears and later claims. Nodes are
   published once per dispatch. The cache prevents repeated overclaims while
   the final valid allocator's clear is pending, preserving the host's
   `claim_limit + batch * workers + 1` overflow bound.
3. Read the graph's remaining-node counter only after a scan fails to claim
   work. A valid allocated tile keeps its node incomplete until execution, so
   a successful claim does not need that termination check. Ready-bit clearing,
   including failed-claim helping, remains unchanged.

Every probe preserves payload fences, successor publication, completion,
queue layout, logical tasks and all 256 requested workers. The multiword
completion-epoch source is untouched. Proof sketches and exact candidate
patches are archived alongside the measurements.

Each candidate passes **52 codegen checks, fourteen whole-graph smoke tests and
fifty diagnostic comparisons** across the three CI and two model cases.
Diagnostics use five pairs per case/mode, at least ten warmups and one second
of warmup, with 100 ms prewarming and worker/resource instrumentation. Every
full-state budget, matching implementation, task-total, single-dispatch and
no-fallback check passes; maximum error is 4.66e-9.

| Diagnostic persistent GPU time, ms | Counter precheck, control → probe | Exhaustion cache, control → probe | Delayed completion load, control → probe |
|---|---:|---:|---:|
| CI inference | 0.030912 → 0.030784 | 0.030944 → 0.030624 | 0.030656 → 0.030784 |
| CI backward | 0.066496 → 0.069888 | 0.066336 → 0.068096 | 0.067616 → 0.066144 |
| CI backward + SGD | 0.073760 → 0.074176 | 0.072096 → 0.073888 | 0.073920 → 0.072384 |

The first two probes show no useful improvement. The third's approximately
2% backward/SGD timestamp reduction motivates **600 completed-step CI
comparisons** without diagnostics: control/candidate/candidate/control visits,
25 interleaved pairs per case/visit, both original and 100 ms prewarmed protocols.
All state and execution checks pass, but wall time does not confirm the gain:

| Persistent completed step | Control, ms | Probe, ms | Probe/control, bootstrap 95% interval |
|---|---:|---:|---:|
| Original inference | 0.066945 | 0.069226 | 1.0341 [0.9895, 1.1183] |
| Original backward | 0.093201 | 0.093981 | 1.0084 [0.9991, 1.0169] |
| Original backward + SGD | 0.096296 | 0.095056 | 0.9871 [0.9684, 1.0015] |
| Prewarmed inference | 0.065166 | 0.063846 | 0.9797 [0.9391, 1.0285] |
| Prewarmed backward | 0.092716 | 0.093390 | 1.0073 [0.9980, 1.0210] |
| Prewarmed backward + SGD | 0.094516 | 0.094711 | 1.0021 [0.9930, 1.0108] |

All six raw persistent intervals include one. Prewarmed backward's normalized
persistent/ordinary ratio regresses **1.45% [0.15%, 3.06%]**. The original SGD
ordinary median falls 0.57%, with its ratio interval below one despite identical
ordinary shader sources. Preserve these shifts and all visits; pooled bootstrap
intervals do not capture all between-visit variation. The confirmation covers
CI only, rather than the complete five-case retention gate.

For each probe, only the three CI whole-graph sources change. The other sixteen
CI sources and all forty-two model sources are byte-identical to control.
Independent inverse substitutions recover all nine changed CI sources exactly.
CI register counts stay 48 and shared memory stays 9,616/9,648 bytes. Executable
sizes for inference/backward/SGD change from 18,944/61,440/65,536 bytes to
19,072/62,720/64,896 for the precheck, 19,584/61,312/66,816 for the cache, and
18,944/62,336/65,536 for the delayed load. Model registers remain 84, shared
memory remains 18,928/29,344 bytes and code remains 102,400/238,720 bytes.
All model diagnostics still record 168 participating workers and
22,042/83,312 logical tiles. No model speedup or new SDK counter result is claimed.

All original source and dependency hashes are restored. Rebuilt release
benchmark binaries match the qualified controls byte-for-byte, and all
fourteen restored smoke tests pass. The rejected candidates do not receive
full library/oracle/matrix qualification. Local evidence is under
`target/single-word-{claim-precheck,exhausted-mask,completion-load}/`.
The manifest `target/megakernel-results/single-word-claims/summary.json` contains
**932 independently verified artifact hashes and 44 completed cgroup records**,
SHA-256 `2e410b83da3835d5e0089355e03b723239bd4e2091be30ad29df78a88e6e2ee7`.
Peak recorded memory is **3,714,752,512 bytes**, with zero memory-limit events.
All heavy jobs use verified separate 16/20/1 GiB cgroups, GPU work is serialized,
and builds/copies/analysis do not overlap timing. All artifacts remain on local
NVMe under `/x`. Continue investigating the remaining forward payload costs;
these queue-operation reductions do not justify a runtime change.

## Fused matrix tile selection

Two changes reuse the existing scalar generator through the shared selector:

- Fused transposed-B products can use the 32×32 small kernel when the original
  64×64 grid has fewer than sixteen workgroups. The small-module dispatcher now
  routes this addend form to the generator instead of its large-tile fallback.
  The broader narrow-transposed policy remains restricted to products without
  an addend. Transposed-A addends keep their original treatment.
- Remaining fused-plain SmallTile products with M≤64, N≤1024 and K≥256 can use
  32×16 geometry, extending the previous K≥1024 policy. Transposed-B products
  without an addend retain their K≥1024 refinement. K stage, interleaving,
  storage constraints and split-K precedence are preserved.

The matrix problem now also treats `schedule_locked` as pinned. This prevents
automatic selection from replacing an extracted default or small kernel.
Cooperative preference for unlocked problems, epilogue handling, arithmetic,
dependency pins, queue protocol, fences and inactive-access guards are unchanged.
No shader template or compiler-fork change is required.

Exactly sixteen `FusedMatMulAdd[50x720x960]` nodes per model phase change from
46 to 90 logical tiles. Training also changes sixteen fused transposed-B nodes
with K=32/320/960 from 12 to 46 tiles. Total forward/training work becomes
**22,042 / 83,312** tiles, up from 21,338 / 82,064. Both backends select the same
implementations, with unchanged nodes, tensor allocations, storage formats and
queue bytes within each instrumentation mode. Each valid output keeps its K
accumulation order. The complete step remains one dispatch and one submission,
with all 256 workers enabled.

The new selections remove duplicate or larger handlers. Forward registers fall
**86→84**, training registers rise **82→84**, and code shrinks
**112,384→102,400 / 256,512→238,720 bytes**. Shared memory stays
18,928 / 29,344 bytes; both phases still record 168 participating workers.
All nineteen CI sources/resources are identical. The model replaces both graph
sources, replaces one ordinary module and removes another; 39 sources are
identical. Geometry, staging, padding, register allocation and code composition
change together; per-node simultaneous residency is not measured.

The transposed-addend-only probe passes fifty diagnostic comparisons and changes
training GPU time 13.666→13.224 ms. A separate combined preflight passes fifty
more, measuring forward 4.571→4.424 ms and training 13.688→13.073 ms. These are
short diagnostics, not primary performance qualifications. Before full
qualification, the addend policy is restricted to its original small-grid floor
and the schedule-lock repair is added. A twenty-five-comparison final bridge
verifies identical measured shader sets, full resource multisets, selection,
execution and tensor allocations. Final qualified binaries drive the gates below.

**500 original-protocol and 500 prewarmed comparisons** use
control/candidate/candidate/control visits, 25 interleaved pairs per case/visit,
at least ten warmups and one warmup second. Every full-state budget, backend
selection, task-count, one-dispatch and no-fallback check passes. Maximum error
remains 4.66e-9. The supplemental 100 ms prewarming remains off by default.

| Persistent completed step | Control, ms | Candidate, ms | Candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| Original CI inference | 0.074216 | 0.069526 | 0.9368 [0.9285, 1.0252] |
| Original CI backward | 0.093391 | 0.092736 | 0.9930 [0.9764, 1.0003] |
| Original CI SGD | 0.095111 | 0.094881 | 0.9976 [0.9657, 1.0305] |
| Original forward | 5.185632 | 4.935020 | 0.9517 [0.9381, 0.9650] |
| Original training + SGD | 16.159464 | 15.500535 | 0.9592 [0.9411, 0.9780] |
| Prewarmed CI inference | 0.066636 | 0.067780 | 1.0172 [0.9846, 1.0894] |
| Prewarmed CI backward | 0.092525 | 0.093165 | 1.0069 [0.9922, 1.0171] |
| Prewarmed CI SGD | 0.094610 | 0.095701 | 1.0115 [0.9971, 1.0271] |
| Prewarmed forward | 5.156147 | 4.687214 | 0.9091 [0.9003, 0.9680] |
| Prewarmed training + SGD | 14.333292 | 13.732029 | 0.9581 [0.9381, 0.9676] |

Normalized training ratios improve **0.9747 [0.9540, 0.9958]** original and
**0.9747 [0.9552, 0.9881]** prewarmed. Ordinary training also improves:
14.581034→14.312707 ms (−1.84%) and 11.780522→11.565481 ms (−1.83%), with both
ratio intervals below one. Original forward's normalized ratio improves
**0.9307 [0.9153, 0.9766]**. Prewarmed forward's normalized ratio is
**0.9907 [0.9648, 1.0136]**: its large persistent wall shift accompanies an
ordinary median shift of **3.740338→3.428646 ms (−8.33%)**, whose interval includes
one. Do not attribute the full 9.1% wall improvement to this policy.

Normalized values are medians of paired ratios, not ratios of aggregate medians.
Original CI inference's normalized ratio rises **2.74% [0.46%, 9.48%]**, while
its raw persistent interval includes one and ordinary inference falls 8.52%.
All CI shifts occur on identical shader/resource identities and are not
attributed to the selector. All samples and visits remain archived; pooled
bootstrap intervals omit some between-visit variation. The new forward gain
does not directly remeasure the earlier pre-epoch control.

Eight SDK backend captures collect thirty-two sequential metric groups, with
no failed capture or repeat. Forward replay is graph-only; complete training
replay uses zero-rate SGD, separately from primary nonzero-rate training.

| SDK replay | Control, ms | Candidate, ms | Change |
|---|---:|---:|---:|
| Ordinary forward | 3.348321 | 3.342171 | −0.18% |
| Persistent forward | 4.702349 | 4.536708 | −3.52% |
| Ordinary training + SGD | 11.951973 | 11.728481 | −1.87% |
| Persistent training + SGD | 13.968511 | 13.350892 | −4.42% |

Each actual forward state has **147,034,513 floats**, and each training state
has **246,883,105 floats**. Independent hashing verifies bitwise equality
between control and candidate within each backend/phase. Forward also matches
bitwise across backends; training passes the unchanged numerical budget.
Maximum clock/instruction-rate group spreads are 0.024%/0.632%.

Estimated persistent warp instructions rise **3.24% forward / 0.80% training**,
integer-pipe work **7.80% / 3.16%**, global-load sectors **1.97% / 1.50%**, and
shared wavefronts **5.27% / 1.28%**. FP32 instruction work changes +0.11% / −0.29%.
Persistent local loads/stores remain zero. Active compute warps remain near
32.5% / 32.7% of peak. Barrier waiting falls **55.56→53.15% / 49.40→47.04% of
active warp cycles**, not elapsed time. Ordinary local-load/store sectors fall
from about 79,000 to 55,000 each in forward and rise from 618,000 to 693,000 in
training; the counters do not identify individual shaders. Faster replay with
more instruction/load work supports useful tile parallelism as a contributor,
without isolating it from padding, staging or code composition.

Qualification passes **676 library tests (eight ignored), 168 strict whole-graph
oracle cases, 50 matrix regressions, fourteen whole-graph GPU smoke tests and
all 52 codegen checks**, plus strict all-target/all-feature Clippy, formatting,
release builds and Metal-target compilation. New checks cover selector boundaries,
storage, locked default/small schedules and preserved long-K partial buffers and
SumRows parameters. Ragged fused products at 35×65×263 and 35×65×1031 compare
every output against signed dyadic CPU arithmetic with poisoned storage, changed
inputs and ordinary/1/3/256 workers. Metal GPU execution remains unqualified.

The archive preserves the initial setup failure on ignored `Cargo.lock`, a new
test's moved-value compilation error and an old selector expectation. Setup is
recovered from verified originals; both tests are corrected before the passing
full qualification. No GPU measurement fails or is rerun for these corrections.

Local evidence is in `target/narrow-bt-addend/` and
`target/narrow-fused-products/`. Manifest
`target/megakernel-results/narrow-fused-products/summary.json` contains **975
independently verified artifact hashes and 59 recorded bounded cgroups**, SHA-256
`d56c5cf57f8506240d3b8d96446ec6433f53ef58c0c61e1657ca4d3bb6a0841e`.
All recorded memory-limit events are zero; peak job memory is 4,709,384,192 bytes.
SDK-export pressure totals remain in the logs. Every heavy job uses a separate
verified 16/20/1 GiB memory-high/hard/swap cgroup on local `/x`. GPU runs serialize;
builds, copies and heavy analysis stay outside measurements. Every job finishes,
and both compiler forks and dependency pins remain unchanged.

Next, audit exhausted-tile claims in the single-word small-graph scheduler and
the remaining forward payload costs. Preserve all requested workers, publication
fences and logical work. The unchanged CI path and the remaining model gaps
still prevent baseline parity.

## Narrow transposed-B columns

On qualified control `2db140d`, the shared selector now also admits transposed-B
products to the existing 32×16 scalar refinement. It applies only to remaining
SmallTile choices after split-K selection, with M≤64, N≤1024 and K≥1024.
Reduced storage, pinned schedules and epilogues retain their previous treatment.
The requested K stage and column interleaving are preserved; illegal geometry
falls back to the original choice. Fused plain products keep their existing
policy, and transposed addends remain unchanged. No shader template, dependency,
queue, fence, inactive-access guard or compiler-fork changes are required.

Exactly sixteen training `MatMulBT[50x720x4096]` nodes change from **46 to 90
logical tiles**. The complete step increases from 81,360 to **82,064** tiles.
Each valid output retains its K accumulation order. Both backends select the
same implementations, with unchanged node counts, tensor allocations, storage
formats and queue bytes within each instrumentation mode. The step still uses
one dispatch and one submission, with all 256 workers enabled.

Training remains at **82 registers, 29,344 shared bytes and 168 participating
workers**. A new handler increases compiled code from 249,088 to 256,512 bytes.
Forward remains at 86 registers, 18,928 shared bytes and 112,384 code bytes.
All nineteen CI sources and the forward sources/resources are identical to the
control. The model adds one ordinary module and replaces the training graph
source; 41 sources remain identical. Geometry, staging work, padding and code
composition change together. Per-node simultaneous residency is not measured.

The primary gate uses frozen, fully qualified binaries with
control/candidate/candidate/control visits, 25 interleaved pairs per case/visit,
at least ten warmups and one warmup second. **500 original-protocol and 500
matched-prewarming comparisons** pass all state budgets, matching backend
selection, task accounting, one-dispatch and no-fallback checks. Maximum error
remains 4.66e-9. The supplemental prewarming is 100 ms and stays off by default.

| Persistent completed step | Control, ms | Candidate, ms | Candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| Original CI inference | 0.069001 | 0.068051 | 0.9862 [0.9414, 1.0198] |
| Original CI backward | 0.093046 | 0.092900 | 0.9984 [0.9913, 1.0102] |
| Original CI SGD | 0.094525 | 0.094686 | 1.0017 [0.9740, 1.0056] |
| Original forward | 5.208277 | 5.155987 | 0.9900 [0.9327, 1.0114] |
| Original training + SGD | 16.537376 | 15.943783 | 0.9641 [0.9589, 0.9920] |
| Prewarmed CI inference | 0.067681 | 0.068521 | 1.0124 [0.9690, 1.0598] |
| Prewarmed CI backward | 0.092671 | 0.093031 | 1.0039 [0.9925, 1.0173] |
| Prewarmed CI SGD | 0.094746 | 0.095400 | 1.0069 [0.9613, 1.0161] |
| Prewarmed forward | 5.169596 | 5.168677 | 0.9998 [0.9664, 1.0578] |
| Prewarmed training + SGD | 14.597705 | 14.324968 | 0.9813 [0.9575, 0.9920] |

Normalized training ratios improve to **0.9794 [0.9565, 0.9984]** original and
**0.9667 [0.9621, 0.9748]** prewarmed. Ordinary training changes
14.899791→14.860695 ms under the original protocol, with its interval including
one. Prewarmed ordinary training regresses **11.721597→11.790887 ms (+0.59%)**,
ratio 1.0059 [1.0013, 1.0199]. The absolute persistent gain is established in both
protocols; the normalized gain also includes this ordinary-execution cost.
Normalized values use medians of paired ratios, not ratios of aggregate medians.
Prewarmed CI backward's normalized ratio rises **1.48% [0.35%, 3.36%]**, although
its raw persistent interval includes one. CI and forward timing changes occur
on identical shader/resource identities and are not attributed to the selector.
All samples and visits remain archived; pooled bootstrap intervals omit some
between-visit variation. The earlier forward tradeoff remains unresolved.

Four complete-step SDK backend captures with sixteen sequential metric groups
measure persistent replay **14.353883→13.969311 ms (−2.68%)** and ordinary replay
**11.904653→11.955133 ms (+0.42%)**. They use zero-rate SGD for stable replay,
separately from the primary nonzero-rate training protocol. All **246,883,105
floats** in each actual state file are independently verified bitwise identical
between control and candidate within each backend. Cross-backend values pass
the unchanged budget. Maximum clock/instruction-rate group spreads are
0.012%/0.274%; every capture succeeds without a repeat.

Persistent estimated warp instructions increase **3.60%**, integer-pipe work
**7.58%**, global-load sectors **5.27%** and shared wavefronts **4.93%**. FP32
instruction work falls 0.20%; local loads/stores remain zero. Active compute
warps remain near 32.7% of peak. Barrier waiting falls **52.77→49.55% of active
warp cycles**, not elapsed time. Ordinary local-load/store sectors fall from
about 669,000 to 617,000 each, despite its slower replay; these totals do not
identify a responsible shader. Faster persistent replay with more instruction
and load work supports useful tile parallelism as a contributor, without
isolating it from geometry, padding, staging or code composition.

Qualification passes **674 library tests (eight ignored), 168 strict whole-graph
oracle cases, 50 matrix regressions, thirteen whole-graph GPU smoke tests and
all 52 codegen checks**, plus strict all-target/all-feature Clippy, formatting,
release builds and Metal-target compilation. The new 35×65×1031 fixture checks
every result against signed dyadic CPU arithmetic with ragged row/column/K tails,
poisoned storage, changed inputs and ordinary/1/3/256/4096 workers. Selection
tests cover the shape/storage boundaries, K-stage legality, interleaving, pinned
schedules and preserved long reductions. Metal GPU execution remains unqualified.

There are **75 diagnostic comparisons**. Final binaries match the preflight
candidate's generated shader sets, full resource multisets, selection, execution
and tensor allocations. Two postprocessing assertions were corrected: diagnostic
counter allocations differ from timing allocations by `2 * nodes * workers * 4`
bytes, and the corrected CPU analysis log must not be classified as a GPU job.
The original scripts, failed logs and corrected results are retained. No GPU
measurement or qualification test failed or was rerun for these corrections.

Local evidence: `target/narrow-bt-columns/`. Manifest
`target/megakernel-results/narrow-bt-columns/summary.json` contains **600
independently verified artifact hashes and 46 recorded bounded cgroups**, SHA-256
`19b66b88f1941ed3dc5577f07882679790539a7bf736d22a1544e2d912a9ad6b`.
All recorded memory-limit events are zero; peak job memory is 4,720,279,552 bytes.
SDK-export pressure totals remain in the logs. All heavy jobs run on local `/x`
with verified 16/20/1 GiB memory-high/hard/swap limits. GPU jobs serialize, and
builds, copies and heavy analyses stay outside measurements. Both forks and
dependency pins remain unchanged. Every launched job reaches a terminal state.

The [fused-matrix follow-up above](#fused-matrix-tile-selection) investigates the
remaining fused transposed-B and forward matrix stages while preserving the
shared selection, arithmetic contract and all five acceptance cases.

## Narrow transposed-B tile selection

The shared matrix family now admits its existing 32×32 scalar kernel for
unpinned F32 transposed-B products with at most 64 rows, K≥256 and fewer than
128 original 64×64 workgroups. The existing small-grid rule remains in place;
cooperative preference, reduced storage and explicit schedules keep their
precedence. Transposed products with an addend retain their existing kernel.
No shader template, arithmetic mode, queue protocol, fence, dependency, access
guard or compiler fork changes.

In SmolVLA training, exactly sixteen `MatMulBT[50x2048x720]` nodes change from
32 to 128 logical tiles. Each valid output retains its K accumulation order.
Long reductions retain their original split count, partial storage and SumRows
parameters; the selected tile geometry can change within those partitions.
The complete step now counts 81,360 logical tiles, up from 79,824, while keeping
the same nodes, buffer allocation, queue bytes, one dispatch and one submission.
All 256 workers remain requested. Both backends select the same implementations.

The training pipeline reuses an existing small-tile handler and drops the
unused large-tile variant. Registers remain **82**, shared storage remains
**29,344 bytes**, participating workers remain **168**, and code shrinks
264,704→249,088 bytes. Forward remains 86 registers and 18,928 shared bytes.
All nineteen CI shader sources and all model sources except the training graph
and its now-unused ordinary large-tile module are byte-identical. Geometry,
padded arithmetic and compiled-handler composition change together; the study
does not isolate their individual contributions or measure per-node residency.

The primary gate compares qualified `6fd43c8` binaries with the fully tested
candidate on source base `89a49d5`. Control/candidate/candidate/control visits
retain 25 interleaved pairs per case, at least ten warmups/one second and the
256-worker gate. **500 original-protocol and 500 prewarmed comparisons** pass
all full-state budgets, matching backend selection, task accounting, one-dispatch
and no-fallback checks. Maximum model error remains 4.66e-9. Prewarming is the
same supplemental 100 ms control and remains off by default.

| Completed-step measurement | Control, ms | Candidate, ms | Candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| Original forward | 5.203167 | 5.227358 | 1.0046 [0.9969, 1.0137] |
| Original training + SGD | 17.042129 | 16.412600 | 0.9631 [0.9490, 0.9855] |
| Prewarmed forward | 5.176952 | 5.133936 | 0.9917 [0.9370, 1.0012] |
| Prewarmed training + SGD | 15.006357 | 14.719415 | 0.9809 [0.9650, 0.9836] |

Training's normalized ratios also improve: **0.9777 [0.9570, 0.9912]** original
and **0.9775 [0.9717, 0.9990]** prewarmed. Ordinary training medians change
15.025→14.906 ms and 12.133→11.714 ms respectively. Normalized results use
medians of paired ratios, not ratios of aggregate medians. Pooled bootstrap
intervals do not capture all between-visit variation; all visits and samples
remain archived. The original CI-backward persistent median rises **0.88%**,
with its raw interval above one; ordinary rises 0.75%, while the normalized
interval includes one. CI and forward shifts occur on unchanged shader/resource
identities and are not attributed to this selection change.

Four complete-step SDK backend captures, sixteen metric groups and zero-rate
SGD measure persistent replay **14.884→14.354 ms (−3.56%)** and ordinary replay
12.004→11.913 ms (−0.75%). All 246,883,105 captured floats are independently
verified bitwise identical between control and candidate within each backend.
Cross-backend state remains within the existing budget and is not bitwise equal.
Maximum clock/instruction-rate group spreads are 0.017%/0.494%.

Persistent estimated warp instructions increase **0.54%**, integer-pipe work
**2.84%** and global-load sectors **2.89%**, while FP32 activity falls **0.63%**.
Shared wavefronts rise 0.17%; persistent local traffic remains zero. Average
active compute warps stay near 32.7% of peak. Barrier waiting changes
54.74→52.79% of active warp cycles; it is not an elapsed-time fraction.
Ordinary local sectors increase from about 402,000 to 666,000 per replay;
these counters do not identify the responsible shader. Faster replay with
more estimated instruction and load work supports investigating useful tile
parallelism, while code footprint and padding remain joint contributors.

Qualification passes **673 library tests (eight ignored), 168 strict whole-graph
oracle cases, 50 matrix regressions and twelve whole-graph GPU smoke tests**,
plus all 52 codegen checks, strict all-target/all-feature Clippy, formatting,
release example builds and Metal-target compilation. New checks cover selection
boundaries, storage/schedule constraints and preserved long-K partitions. A
35×1025×263 GPU fixture checks every output against signed dyadic CPU results,
ragged row/column/K tails, poisoned storage, changed inputs and
ordinary/1/3/256/4096-worker execution. Metal GPU execution remains unqualified.

There are **75 diagnostic comparisons**, including a post-qualification bridge.
Test additions change executable hashes; primary and SDK gates use the final
qualified binaries. Their full generated shader sets, resource multisets,
selection, execution and buffers exactly match the preflight candidate.
Dependency pins and both fork checkouts remain unchanged.

Local evidence is under `target/narrow-bt-selection/`. Manifest
`target/megakernel-results/narrow-bt-selection/summary.json` contains **588
independently verified artifact hashes and 44 recorded job cgroups**, SHA-256
`afc185b0f1db71993b119b8381835a180d24f2da5bee48fff2246c2910d8c88b`.
Every heavy job has verified 16/20/1 GiB memory-high/hard/swap limits. GPU jobs
serialize, with builds and heavy analysis outside measurement intervals; all
artifacts remain on local `/x`. Peak job memory is 4,709,826,560 bytes and
recorded memory-limit events are zero. Small SDK-export pressure totals remain
in the logs. All jobs finish. Baseline parity and the forward gap remain open.

The [column refinement above](#narrow-transposed-b-columns) follows up on the
remaining 46-tile `50x720x4096` stages while preserving selected precision,
K order, requested workers and all five completed-step cases.

## Completion-gated idle scans

An idle worker repeatedly scanning every ready-bitmap word cannot discover new
work until a producer publishes a successor. For graphs with more than 32 nodes,
the retained policy lets lane zero wait for the remaining-node count to change
before rescanning. Other lanes wait at the existing uniform claim broadcast.
Successful workers can keep consuming already-ready tiles in the same epoch.
The one-word path remains byte-identical to its control. Selection uses the full
step's bitmap width, including optimizer nodes, and source identity distinguishes
the pipeline cache entries.

A release fence after successor publication and before the remaining-node RMW
pairs with an acquire before a scan of a newly observed epoch. This prevents a
worker from caching the new count after missing its preceding publication.
The [memory contract](megakernel_memory_contract.md#progress-completion-and-reuse)
records the ordering and progress argument. Existing payload fences, relaxed
atomics, arithmetic, descriptor identity, inactive access guards and logical
completion accounting remain unchanged. No worker is capped or retired early;
the original 256-worker benchmark gate is retained. Neither fork changes.

The earlier completion-polling experiment used a different shader footprint.
Repeating it on `e4e70c5` passes 50 diagnostic comparisons and improves model
training GPU time 15.531→14.553 ms, while slowing small CI graphs by about
2–7 µs. Restricting it to multiword bitmaps preserves all 19 CI shader sources
and their resources exactly. Another 50 diagnostic comparisons confirm that
both model shaders match the unrestricted prototype byte-for-byte; reversing
only its epoch changes recovers their entire control source exactly.

Control/candidate/candidate/control sequences then pass **500 original-protocol
and 500 prewarmed completed-step comparisons**, with 25 pairs per case/visit,
ten minimum warmups and at least one warmup second. The supplemental sequence
prewarms each sample for 100 ms. All five cases preserve arithmetic selection,
full-state budgets, one dispatch, no fallback and complete task accounting.
Maximum model error remains 4.66e-9; all raw samples are retained.

| Completed-step measurement | Control, ms | Epoch policy, ms | Candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| Original forward | 5.126816 | 5.180661 | 1.0105 [1.0031, 1.0384] |
| Original training + SGD | 18.003525 | 16.747768 | 0.9302 [0.8580, 0.9480] |
| Prewarmed forward | 5.132527 | 5.146547 | 1.0027 [0.9457, 1.0418] |
| Prewarmed training + SGD | 16.163839 | 15.267334 | 0.9445 [0.9331, 0.9525] |

Ordinary training changes 15.010→15.052 ms and 11.805→11.835 ms respectively.
Training's normalized intervals also exclude no change. The **1.05% original
forward regression is a retained tradeoff**, not a speedup in all cases.
Pooled intervals do not capture all between-visit variation. CI differences,
including a lower prewarmed SGD median, occur on identical shaders/resources
and are not attributed to this policy.

Forward/training registers change **84/80→86/82**, while shared bytes stay
18,928/29,344. Executable sizes grow by 384 bytes each. Forward participation
stays 168; training participation changes **252→168**. The added private state
changes allocation, so the timing result does not isolate polling savings from
resource/concurrency effects. The policy adds no diagnostic scratch padding.

Fresh SDK captures preserve bitwise-identical complete state within each
backend, including all 246,883,105 captured training floats. Persistent replay
improves **15.971→14.867 ms (6.9%)**, while ordinary replay is 11.996→11.972 ms.
Average active compute warps fall **49.10%→32.75% of peak**, with SM-active time
near 98.2%. Estimated warp instructions fall 16.25%, integer-pipe activity 0.85%,
FP32 work 0.36%, global-load sectors 3.19% and shared wavefronts 5.44%. Persistent
local load/store traffic remains zero. Barrier waiting falls 63.92%→55.00% of
active warp cycles. These rate-times-duration estimates and waiting shares do
not assign elapsed-time savings to individual instructions.

The initial candidate ordinary capture has 2.28% instruction-rate spread across
counter groups, exceeding the 2% audit limit. It remains archived; only that
ordinary capture is repeated, preserving exact full state and selection. The
final comparisons have maximum clock/instruction-rate spreads of 0.32%/0.73%.
There are five backend captures and twenty metric groups including the repeat.

A further 50-pair diagnostic preflight compiles the known bitmap width into
the shader. It reduces executable sizes but shows no useful timing improvement
and does not resolve the forward cost. Reversing its two constant substitutions
recovers all five epoch-control shaders exactly. It is removed; the qualified
multiword policy's source and rebuilt benchmark binaries are restored exactly.

Qualification passes **671 library tests (eight ignored), 168 strict whole-graph
oracle tests, 50 matrix regressions, 52 codegen tests and eleven whole-graph GPU
smoke tests**, plus strict all-target/all-feature Clippy, formatting, release
examples and Metal-target compilation. The new GPU test covers 32/33/65-node
boundaries, independent chains, three replays and 1/3/256/4096 workers, checking
every intermediate output and logical-task total. All eleven restored smoke
tests pass. Metal GPU execution and additional hardware remain unqualified.

The combined local record spans `target/completion-epoch-{current,multiword,constant-bitmap}/`.
Manifest `target/megakernel-results/completion-epoch-policy/summary.json` contains
1,000 completed-step comparisons, 150 diagnostic comparisons, **1,089 independently
verified artifact hashes and 69 recorded bounded cgroups**, SHA-256
`6383db3e024e3b83c92b473b6853e17e3e8bc744399636f746a5f61ad6ba42c1`.
Peak job memory is 4,709,658,624 bytes; recorded memory-limit events are zero.
Small SDK-export pressure totals remain in the logs. Every job completes on
local storage, with GPU captures serialized and builds/analysis outside timing.
The remaining forward regression and gaps in all five cases stay open.

## Epoch state and directional fence controls

Two diagnostic follow-ups use retained runtime `6fd43c8`. Neither establishes
a useful GPU-time improvement, so both are removed. The retained runtime,
dependency lock, fork checkouts and benchmark binaries are restored exactly.

The first packs the two private epoch words into one, using the high bit to
record an empty scan. Existing checked queue allocation bounds the node count
below that bit; the sentinel remains distinct. It changes no synchronization,
queue operation, arithmetic or requested worker count. Fifty diagnostic
comparisons pass, but model registers remain 86 forward / 82 training, shared
bytes remain 18,928 / 29,344, and executable sizes remain 112,384 / 264,704.
Model GPU medians change 4.568→4.578 ms forward and 14.525→14.579 ms training.
Reducing source-level private state does not reduce the compiled allocation.

The second changes only the two fences added by epoch polling. A temporary
Blade hook keeps Naga validation and then substitutes an acquire before the
new-epoch scan and a release after successor publication. It preserves
QueueFamily scope, UniformMemory, acquire visibility and release availability;
all existing payload fences remain acquire/release. This follows the existing
[Vulkan fence-to-fence synchronization rules](https://docs.vulkan.org/spec/latest/appendices/memorymodel.html#memory-model-synchronizes-with).
No Naga API, WGSL source, worker count, queue layout or instrumentation counter
changes. The same frozen diagnostic binary selects unchanged or directional
SPIR-V; both enable debug names so the hook can identify the intended operations.

Seven original/transformed module pairs validate before use and pass an
independent inverse audit: removing two constants and reversing two fence
operands recovers each entire original module byte-for-byte. One initial model
attempt stops before pipeline creation because debug line records intervene
between the fence and store. Its script, log and nonzero exit remain archived.
The corrected identification skips only debug records and retains all operation
and queue-address checks. The original and transformed modules then validate.

Seventy-five diagnostic comparisons cover the retained binary and both probe
modes, with five paired samples per case, 256 requested workers, at least ten
warmups/one second and 100 ms prewarming. All generated WGSL files are identical across
these modes, and registers, shared memory, executable sizes and model worker
participation match. Model GPU medians are:

| Case | Retained binary, ms | Probe control, ms | Directional fences, ms |
|---|---:|---:|---:|
| Forward | 4.564960 | 4.572256 | 4.567840 |
| Training + SGD | 14.579840 | 14.550592 | 14.589280 |

Across both studies, all **125 diagnostic comparisons** pass full-state budgets,
selected-implementation equality, logical task totals, one-dispatch and
no-fallback checks. Maximum model error remains 4.66e-9. Each probe passes
52 codegen checks and eleven whole-graph GPU smoke tests; all eleven restored
smoke tests also pass. These are preflight results, without a full performance
gate or additional SDK captures. Native instruction bytes remain unavailable;
the result neither proves identical driver lowering nor qualifies another GPU.
It provides no performance justification for extending the Naga fence API.

Evidence stays local in `target/completion-epoch-packed-state/` and
`target/epoch-directional-fence/`. Manifest
`target/megakernel-results/epoch-state-fences/summary.json` contains **854
independently verified artifact hashes and 31 recorded job cgroups**, SHA-256
`0cca15260635881c3b3beacc9521c7758ab62419100f878b159b12ced2d4c31b`.
Every heavy job has verified 16/20/1 GiB memory-high/hard/swap limits; GPU jobs
serialize and no build or heavy analysis overlaps measurements. Peak job memory
is 3,725,115,392 bytes, with no recorded memory-limit events. All jobs finish.
The forward regression and baseline parity remain open; the measured narrow
matrix stages remain the next optimization target.

## Single-broadcast claim snapshot

The earlier combined-broadcast experiment retained a separate ticket broadcast
and parallel parameter loads. This follow-up instead makes lane zero load node
metadata and all parameters before broadcasting one complete `GraphClaim`.
The queue, payload fences, handler arithmetic, inactive access guards and
completion/publication protocol remain unchanged. Shared-header accounting
includes the structure's 16-byte alignment; the five measured cases retain
their existing shared-memory sizes.

Five prewarmed diagnostic pairs per case in each control/candidate visit pass
all state budgets, matching arithmetic selection, complete worker accounting,
one-dispatch and no-fallback checks: **50 comparisons** in total. All five
generated whole-graph shaders reverse byte-for-byte by undoing only this
broadcast transformation. Ordinary and other shader sources/resources match.

| Diagnostic GPU median, ms | Control | Single broadcast |
|---|---:|---:|
| CI inference | 0.030816 | 0.030432 |
| CI forward/backward | 0.066208 | 0.065920 |
| CI forward/backward/SGD | 0.072288 | 0.069824 |
| SmolVLA forward | 4.534400 | 4.576864 |
| SmolVLA forward/backward/SGD | 15.622144 | 15.578432 |

Ordinary model GPU medians remain approximately 3.00/11.00 ms. Model registers
remain 84/80, shared bytes 18,928/29,344, and participating workers 168/252.
The requested pool remains 256 throughout. This short preflight shows no
material model gain; it is not a full performance qualification, a confidence
bound on small effects, or an SDK counter study. Archive the prototype without
running the full acceptance gate.

The initial packing check fails because the subgroup-geometry prelude still
references the replaced ticket variable. A diagnostic rerun reproduces that
failure. Updating both references restores all **52 codegen checks and ten
whole-graph GPU smoke tests**, without changing the test expectation or Naga
validation. Initial binaries are archived and never benchmarked. After removing
the prototype, rebuilt benchmark binaries match the qualified controls exactly,
and all ten restored GPU smoke tests pass. Runtime, dependencies and both forks
remain unchanged.

Records are under `target/claim-snapshot-control/`. Manifest
`target/megakernel-results/claim-snapshot-control/summary.json` contains **319
independently verified artifact hashes and 17 recorded bounded job scopes**,
SHA-256 `18d52aea99213939bace7f97f351763bc4a225f28ffbff65a58cf10400120ebb`.
It preserves both initial failures and the corrected results. Peak job memory
is 3,718,123,520 bytes; all recorded memory-limit event counts are zero. All
jobs complete, with artifacts on local storage.

## Worker capacity and concurrency control

The full clock probe changed both register allocation and worker participation.
This control instead appends **12,288 unused bytes** to the final scratch bank
when existing scratch exceeds 24 KiB, with matching admission accounting.
The threshold and amount are diagnostic choices, not a portable runtime policy.
The inactive-read sentinel, indices, initialization stores, selected geometry,
handler bodies, queue operations and fences remain unchanged.

Only the model training shader changes. Reversing its one array-capacity
declaration reproduces the entire control source exactly. All 19 CI sources and
42 other model sources match byte-for-byte, including forward and ordinary
execution. Cached pipeline/resource multisets also match except for that
training pipeline's identity and shared allocation. Training retains **80
registers and 264,320 code bytes**, while shared memory grows from **29,344 to
41,632 bytes** and participating workers change **252→168**. The dispatch still
requests 256 workers, and workers still exit only after graph completion.

The local CUDA driver reports 84 multiprocessors, 102,400 shared bytes and
65,536 registers per multiprocessor. Three padded workgroups cannot fit within
that shared-memory bound. These hardware limits and participation observations
do not by themselves measure simultaneous Vulkan residency; the SDK activity
measurements below support the concurrency change.

Control/padded/padded/control visits pass **500 original-protocol and 500
prewarmed completed-step comparisons**, with 25 pairs per case/visit, ten minimum
warmups, at least one warmup second and all five cases. Prewarming is 100 ms
per sample in the supplemental sequence; the primary gate remains unchanged.
All state budgets, implementation selections, completion checks, one-dispatch
and no-fallback checks pass. Maximum model error is 4.66e-9.

| Training measurement | Control, ms | Padded, ms | Padded/control, bootstrap 95% interval |
|---|---:|---:|---:|
| Original completed-step protocol | 17.773244 | 16.734058 | 0.9415 [0.9298, 0.9580] |
| Matched 100 ms prewarming | 16.038833 | 14.834796 | 0.9249 [0.9183, 0.9344] |

Ordinary training medians change by about 0.2% in both sequences. Paired
persistent/ordinary medians move 1.211→1.140 without prewarming and
1.356→1.255 with it. The original normalized ratio's interval includes no
change, while the prewarmed normalized interval is 0.9190–0.9339. Forward's
unchanged shader establishes no timing change. Small CI differences occur
despite identical source/resources; pooled sample intervals do not capture all
between-visit variation. All raw samples and visit results remain archived.

Four SDK backend captures collect 16 sequential single-pass metric groups and
preserve bitwise-identical actual state files within each backend, including
all 246,883,105 training state floats. Persistent zero-rate-SGD replay improves
**15.963→14.746 ms (7.6%)**; ordinary replay stays 12.001→11.992 ms. Average
active compute warps fall **49.19%→32.76% of peak**, while SM-active time stays
about 98.3%. Estimated per-step warp instructions fall 11.2%, global-load
sectors 3.8% and shared wavefronts 2.4%. Integer-pipe activity changes −0.27%
and FP32 work +0.02%. Barrier waiting falls 63.88%→53.73% of active warp cycles.
Maximum clock-group spread is 0.52%; instruction-rate spread stays below 0.92%.
These rate-times-replay estimates and waiting shares are not elapsed-time
attributions or proof of native instruction identity.

The result supports excess concurrent workers as a contributor to the training
gap at this footprint. Array capacity can also affect driver initialization,
bounds handling and allocation/cache decisions. The control does not establish
a universal worker-count or padding policy, and it does not close the remaining
gap. Archive the hard-coded allocation probe; investigate a general scheduler
change rather than applying this threshold to arbitrary graphs.

The probe passes 52 codegen checks and ten whole-graph GPU smoke tests. Its
source is restored exactly; rebuilt benchmark binaries match the retained
controls, and all ten restored smoke tests pass. Neither fork nor any dependency
pin changes. No full retained-change qualification is claimed. All heavy jobs
use separate bounded cgroups and local storage; all memory-limit event counts
are zero. SDK state export produces small pressure totals, preserved in the
logs. Peak job memory is 4,722,343,936 bytes.

Records are under `target/worker-capacity-control/`. The manifest
`target/megakernel-results/worker-capacity-control/summary.json` preserves
1,000 completed-step comparisons, ten diagnostics, four SDK backend captures,
36 recorded job scopes and **441 independently verified artifact hashes**,
SHA-256 `1b39e34fff15ca42638641be94297571d623817128742e2a00eed338af32f442`.
Two initial archival checks misclassified a CPU-analysis log and then attempted
verification before a manifest existed; corrected checks pass and both failed
logs remain. No benchmark or test failed in this capacity study.

## Warmed handler clock attribution

The retained Naga and Blade forks have no shader-clock API. The archived
diagnostic already provides a feature-checked Vulkan device clock, so this
study adapts that temporary probe to runtime `e4e70c5`. Blade replaces explicit
marker calls after Naga validation and validates the resulting SPIR-V before
pipeline creation. Independent reversal reproduces all **64 original module
binaries** exactly: 32 pairs for each probe. Naga, atomic semantics, handler
arithmetic, memory guards and the requested 256-worker pool remain unchanged.

The full probe records worker lifetimes, first/last node timestamps and summed
handler intervals. It passes 400 paired comparisons across all five cases,
with control/probe/probe/control visits at both zero and 100 ms prewarming.
However, model registers rise from 84/80 to 94/89 forward/training. Training
changes from 252 participating control workers to 168 long-lived instrumented
workers, with the other 88 entering near completion. Warmed training GPU time
falls from 15.553 to 14.486 ms while ordinary GPU time stays about 11.03 ms.
This **6.9% apparent improvement is a profiling perturbation**, not an accepted
optimization or proof that clock reads help. Register allocation, queue
storage, bookkeeping and work distribution change together.

The sparse probe keeps only first/last node/worker timestamps. Clock values
are consumed immediately; none survives across a handler body. It passes
another 200 paired comparisons with 100 ms prewarming, plus four preliminary
model comparisons. Training retains **80 registers, 29,344 shared bytes and
252 participating workers** in both visits. Forward uses 86 rather than 84
registers, unchanged shared capacity and 168 participating workers.

| Case | Control / sparse-probe persistent GPU interval, ms | Probe change |
|---|---:|---:|
| CI inference | 0.030928 / 0.033280 | +7.6% |
| CI backward | 0.067872 / 0.074544 | +9.8% |
| CI SGD | 0.072192 / 0.079488 | +10.1% |
| Model forward | 4.552896 / 4.639440 | +1.9% |
| Model training + SGD | 15.536304 / 15.839776 | +2.0% |

These include queue-reset timing and worker counters; they are diagnostics,
not the normal completed-step performance gate. Sparse-probe ordinary GPU
medians stay within 0.1% of control for both model phases. Every comparison
preserves selected implementations, full-state budgets, logical completion,
one compute dispatch and no fallback. Maximum model error is 4.66e-9.

Fresh ordinary dispatch profiles preserve bitwise exact complete state and
match the paired benchmark's implementation keys. Representative spans are:

| Phase / shape | Nodes | Ordinary median interval, µs | Sparse persistent median span, µs | Median participating workers |
|---|---:|---:|---:|---:|
| Forward `MatMul[50x4096x720]` | 16 | 53.06 | 74.91 | 128 |
| Forward `FusedMatMulAdd[50x720x2048]` | 16 | 63.55 | 68.80 | 90 |
| Training `MatMulBT[50x720x4096]` | 16 | 127.06 | 207.72 | 46 |
| Training `MatMulBT[50x2048x720]` | 16 | 43.01 | 94.49 | 32 |
| Training `FusedMatMulAdd[50x720x2048]` | 16 | 63.49 | 96.03 | 90 |
| Training cross-attention dKV, `q=50,kv=16,h=15/5` | 8 | 106.50 | 156.84 | 32 |

The four SGD nodes overlap backward work and each other. In the two sparse
training traces their envelope unions cover 1.848/1.853 ms, with only
0.652/0.635 ms after the last non-SGD handler. The ordinary SGD pass measures
1.355/1.348 ms. Summing the four persistent spans would double-count overlap;
their final tail does not explain the entire model gap. Removing SGD could
also change contention and scheduling, so that tail is not a savings estimate.

The table compares different instrumentation contracts: ordinary profiles
split dispatches into timed passes, while persistent spans include gaps between
a node's claims and can overlap other nodes. The ordinary backward profile
excludes SGD. Each sparse trace is the final measured step of its visit; paired
timing distributions retain every sample. Clock-to-time scaling uses that
trace's GPU interval and envelope. The
[SPIR-V clock extension](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_shader_clock.html)
guarantees neither a fixed rate nor a compiler code-motion barrier. These
measurements identify targets, not additive recoverable costs.

Both probes pass five focused graph-codegen tests and ten whole-graph GPU
smoke tests. Both forks, runtime source and lock are restored exactly; rebuilt
benchmark binaries match the frozen controls byte-for-byte, and all ten
restored smoke tests pass. No full retained-change suite or new SDK counter
capture is claimed. All jobs use separate bounded cgroups and local `/x`
storage. The 46 recorded job scopes have zero memory-limit events; peak job
memory is 4,254,867,456 bytes, below the 16 GiB soft limit.

Evidence is under `target/clock-{warmed,sparse}/`. The manifest
`target/megakernel-results/clock-warmed-sparse/summary.json` preserves 600 paired
diagnostic comparisons, four preflight comparisons, 30 traces, 60 resource
records and **366 independently verified artifact hashes**, SHA-256
`3d907706ef92bbc33c78183b2c580ffecdbcaf5918c12d323e0dcc0da4e53b32`.

Next, isolate the resource/concurrency effect without clock bookkeeping or a
smaller requested pool. A shared-capacity control can preserve all 256 workers
and handler arithmetic while testing a different residency limit. Verify the
actual compiled footprint and source differences before attributing a result
to residency. Then target narrow matrix stages with normal and matched
prewarmed completed-step gates. Baseline parity remains unmet.

## GPU idle effects and benchmark stability

This investigation keeps runtime and shader source exactly at `e4e70c5`. It
collects 600 diagnostic comparisons: a frozen control, host tracing, two fixed
four-visit prewarming controls, and GPU timestamp comparisons. Every sample
keeps the selected arithmetic, 256 requested workers, complete state checks,
logical completion and one compute dispatch without fallback. No CPU or GPU
policy is changed, and no samples are discarded.

The first frozen control peaks at 3.38 GiB of cgroup memory and records no
memory-limit events or memory pressure. The unexported-state prewarming visits
also record no memory pressure. State-file export visits peak around 5.37 GiB
and record 6–116 ms of aggregate memory pressure over their whole run, while
all limit-event counters remain zero. These are distinct measurements:
[`memory.high`](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory-interface-files)
can throttle a job, but a configured limit alone does not prove that happened.

Host tracing separates `Session::step` from `Session::wait`. Ordinary/persistent
training recording and submission take about 0.62/0.29 ms. Long samples spend
66–79 ms in the completion wait, with negligible measured CPU runqueue delay.
Sparse 100 ms NVML observations show low GPU clocks and P3/P5 states around
those samples. The long CPU state qualification between samples can leave the
GPU idle after the initial warmup.

A fixed sequence of 0, 100, 100, 0 ms of per-sample prewarming uses one frozen
binary and 25 pairs per model case per visit: 200 comparisons. Prewarming runs
the same completed workload outside timing, then restores parameters and inputs
before the measured execution. In the training samples, the unwarmed ordinary
and persistent paths reach about 78–79 ms; none of the prewarmed training samples
exceeds 40 ms. Prewarmed persistent visit medians are 16.28 and 16.08 ms,
compared with 17.92 ms in the first unwarmed visit. All state checks pass.

A second fixed 200-comparison sequence tests whether restoration after warming
undoes the benefit. Its alternative restores first, warms training with zero-rate
SGD, then sets the original nonzero learning rate before timing. Saved complete
training states—246,883,105 floats per backend—are bitwise identical across all
four visits within each backend. That alternative establishes no useful further
benefit, so its learning-rate manipulation is not retained.

GPU timestamps then distinguish shader execution from queue-start and host-wake
delays. These 100 comparisons retain the normal pass grouping and enable worker
counters, so they remain diagnostic. Decoded calibrated trace intervals match
the benchmark's GPU intervals for every sample. An approximately 78 ms host
sample contains roughly 77 ms between GPU timestamps; the spike is inside the
GPU interval. Submission-to-GPU-start and GPU-end-to-host-return delays are
each below 0.6 ms in these captures.

| GPU interval | Unwarmed ordinary / persistent median ms | Prewarmed ordinary / persistent median ms |
|---|---:|---:|
| Forward | 3.476 / 4.927 | 2.999 / 4.528 |
| Training + SGD | 13.919 / 17.248 | 11.034 / 15.584 |

Prewarmed ordinary training GPU intervals span 11.001–11.054 ms and persistent
intervals 15.397–15.773 ms. Unwarmed maxima reach 76.919/78.772 ms. Prewarming
holds observed clocks near 2.85 GHz with memory at 15 GHz and suppresses the
large spikes. It changes cache state as well as power state; this does not
isolate clock changes alone or identify a specific driver mechanism. Small host
wake delays still affect completed-step measurements. The remaining persistent
GPU-time gap is substantial even in the warmed control.

The common paired benchmark now accepts `--prewarm-ms`, **off by default**.
It validates completion during untimed prewarming, restores matched state
again, and records warm-step counts and elapsed time for each sample. Callers'
`prepare` callback must restore every state item their execution modifies.
Both current benchmarks use inference/backward or stateless SGD. The original
five-case completed-step gate and its default warmup remain unchanged; add this
as a supplemental matched control, for example:

```sh
python3 scripts/run_bounded_job.py model-prewarmed --gpu -- \
  target/release/examples/bench_smolvla_train --whole-graph --workers 256 \
  --runs 25 --warmup 10 --warmup-ms 1000 --prewarm-ms 100 --force
```

The retained option passes 50 additional full-state comparisons covering all
five cases with prewarming off and on. Clippy across all targets/features,
formatting and release example builds pass. Launcher checks verify ordinary
failure status, signal status, termination forwarding without an orphaned test
child, resource summaries and serialization of two competing GPU jobs. Temporary
host/state/timestamp tracing and zero-rate prewarming remain archived; only the
common optional control and launcher reporting are retained. Runtime, shaders,
dependency pins and both fork worktrees remain unchanged.

Evidence is local under `target/benchmark-stability/`, including every sample,
GPU clock/resource observations, complete states, calibrated traces, prototype
patches and frozen binaries. The manifest is
`target/megakernel-results/benchmark-stability/summary.json`.
It records 228 artifact hashes and 48 bounded cgroups; SHA-256
`b24aa8737fe2c9fad652756fbb5bf2fe0be3a17e2a5ebe65af28cb8a5f8a17f4`.

The subsequent [clock study](#warmed-handler-clock-attribution) attributes
residual GPU time inside the combined dispatch and quantifies its own
instrumentation overhead. Keep the unchanged primary gate alongside matched
prewarmed controls.
Baseline parity remains unmet.

## Fully occupied packed claims

This experiment proves when every physical group in a packed worker has a real
logical tile. Only parallel packing of smaller logical workgroups qualifies:
the batch must fill the physical worker, and the packed axis must divide evenly
by that batch. Every known grid sharing a handler participates. Empty grids
skip the handler, but unknown or all-empty initial grids cannot establish the
proof. Original generic packing admission and subgroup geometry checks remain.

Optimizer handlers may later receive different grids. The candidate rechecks
every newly prepared frame and restores dynamic activity before shader source
and pipeline cache lookup if any grid is partial. Existing frames keep immutable
pipelines. The idle-frame shortcut already requires identical handlers, grids
and bindings, so it reuses an existing proof. Tests independently enumerate
claim/group coverage and check full-to-partial reuse, unknown and mixed grids.

All 673 library tests (eight ignored), 54 codegen tests, 168 strict whole-graph
oracle tests, 50 matrix regressions, ten whole-graph smoke tests, 22 attention
references in each mode, strict Clippy, formatting and Metal compilation pass.
Metal GPU execution remains unqualified. One redundant test launch used a
nonexistent standalone `megakernel` target; those tests are part of the passing
`smoke` suite. The launch error is preserved separately.

Ten diagnostic comparisons preserve selection, geometry, buffers, scheduler,
completion and one-dispatch execution without fallback. All 57 ordinary shader
sources and all three CI persistent shaders are byte-identical to control.
Only handler 10 changes in each model shader: four logical 64-lane groups fill
the physical 256-lane worker, allowing `graph_active_lane = true`. Every other
source byte is identical, including the previously audited 102 storage-read
guards, scratch protection and stores. Partial attention handlers stay guarded.

Forward code size falls 112,000→108,928 bytes, while training falls
264,320→263,040 bytes. Registers remain 84/80 and shared allocations remain
18,928/29,344 bytes for forward/training. Diagnostic participation stays
168/252 workers, with 256 requested and 21,338/79,824 logical tasks completed.
These statistics do not establish an occupancy change.

The primary sequence is control, candidate, candidate, control, with 25 pairs
per case per visit: 500 completed-step comparisons at 256 workers. Forward
persistent time improves 5.1260→4.7867 ms, ratio 0.9338 [0.8711, 0.9975], but
training changes 17.8367→17.7141 ms, ratio 0.9931 [0.9836, 1.0041]. Unchanged
ordinary CI inference also moves −8.00% and persistent CI SGD −1.72%.

One fixed 500-comparison confirmation uses the same frozen binaries, all five
cases and unchanged sample acceptance, without rebuilding. Its model results:

| Case | Control persistent ms | Candidate persistent ms | Candidate/control, 95% pooled bootstrap interval |
|---|---:|---:|---:|
| Forward | 4.819140 | 4.658913 | 0.9668 [0.8859, 1.0145] |
| Training + SGD | 17.838643 | 18.000229 | 1.0091 [0.9962, 1.0176] |

Neither absolute model interval establishes improvement. Confirmation paired
persistent/ordinary medians change 1.4261→1.3258 forward and 1.2659→1.4954
training, while pooled ordinary training changes 14.5976→11.9985 ms. Across
visits, the same control binary's ordinary training medians range
11.7893–14.7227 ms, and the same candidate's range 11.7905–14.7248 ms. Samples
also include large completion-time spikes. All visits and samples are preserved;
pooled bootstrap intervals do not capture all between-visit variation. These
ordinary changes cannot be attributed to altered ordinary shader source.

Four complete-step SDK backend captures, using sixteen metric groups, measure
ordinary replay 11.9882→11.9860 ms and persistent replay 15.9345→15.8890 ms
(−0.29%). Estimated persistent warp instructions fall 0.81% and integer-pipe
thread instructions 2.01%; shared wavefronts change +0.18% and global loads
+0.04%, with no local traffic. Barrier waiting remains 64.12→64.19% of active
warp cycles. Counter rates times replay duration are estimates; barrier share
is not an elapsed-time stall cost. Maximum within-capture clock and repeated
instruction-rate spreads are 0.43% and 0.68%.

All 246,883,105 captured state floats are bitwise identical between control and
candidate within each backend, verified from the actual files. Cross-backend
maximum error remains at most 4.66e-9 under the unchanged budget; the backends
are not bitwise identical. All 1,000 normal comparisons and ten diagnostics
pass numerical, selection, completion and single-dispatch checks.

The candidate is archived and reverted. Both rebuilt release benchmark binaries
match retained `e4e70c5` byte for byte. Blade and Naga are unchanged and clean;
this experiment introduces no dependency or atomic-semantics changes.

Evidence is local under `target/packed-full-activity/`, including the proof,
patch, frozen binaries, source audit, raw visits, fixed confirmation design and
SDK states. Manifest `target/megakernel-results/packed-full-activity/summary.json`
records 495 artifact hashes and 50 bounded cgroups; SHA-256
`b79afc2a756bfde3e66d5b4bbb1fb13f44526b499f016c0bbc1dd94edaecc8f2`.
Every heavy job used the verified memory limits and GPU jobs were serialized.

Before another small optimization, run frozen identical-binary controls and
attribute the visit variation using CPU recording/submission/wait timing, GPU
clocks and cgroup memory/CPU pressure. Keep telemetry separate from primary
qualification, preserve all samples and retain the existing benchmark boundary
and 256-worker gate. No CPU or GPU policy was changed in this experiment.

## Static subgroup reduction

This follow-up keeps the preceding shuffle candidate's capability reporting,
runtime geometry guard, exact addition order, original-tree fallback and head
packing. Its only new shader change expands the five shuffle iterations into
five explicit stages. The Blade and Naga patches match the preceding qualified
experiment exactly; their test record, including the five verified pre-existing
wgpu GPU failures, is reused rather than represented as a new green suite.

The ordered-tree audit verifies both arrays in both helpers: 64 leaves,
63 additions, two helper barriers, and active shuffle sources at every stage.
All 673 library tests (eight ignored), 52 codegen tests, 168 strict graph oracle
tests, 50 matrix regressions, ten whole-graph smoke tests, attention references
in both modes, strict Clippy, formatting and Metal compilation pass. Metal GPU
execution remains unqualified.

Fifteen diagnostic comparisons preserve selected implementations, geometry,
buffers, scheduler, work accounting and one-dispatch execution without fallback.
The fresh looped-control binaries, shader bytes and compiler statistics match
the preceding experiment. Unrolled source changes only the expected attention
helpers and associated local names. The original fallback, other handlers,
globals and all 102 guarded storage-read sites remain intact.

Ordinary attention shared allocations return from the looped candidate's
1,536/2,048 bytes to 512 bytes each. Registers instead rise from 40 each to
44/46. The persistent training shader reports the same resources as the looped
version: 79 registers, 29,344 shared bytes and 262,912 code bytes. These counts
do not prove identical native instructions or an occupancy change. Diagnostic
participation remains 252 workers in training and 168 in forward, out of 256
requested; every logical task completes.

The primary sequence is tree, looped, unrolled, unrolled, looped, tree, with
25 paired samples per case per visit: **750 comparisons**. Training results are:

| Version | Ordinary ms | Persistent ms | Paired persistent/ordinary |
|---|---:|---:|---:|
| Retained tree | 14.651669 | 17.907924 | 1.2287× |
| Looped shuffles | 15.590231 | 17.564922 | 1.1418× |
| Unrolled shuffles | 14.104727 | 17.715782 | 1.2466× |

Unrolling improves ordinary training versus the looped version by 9.53%, ratio
0.9047 [0.8870, 0.9266], while persistent timing between them includes no change,
1.0086 [0.9933, 1.0223]. Against the retained tree, unrolled persistent training
improves 1.07%, ratio 0.9893 [0.9768, 0.9996], but ordinary uncertainty spans
0.9585–1.1885 despite its lower median. The same sequence records 0.60% CI-backward
and 7.44% forward persistent regressions, even though their shader bytes and
resources are unchanged. Keep these results and all visits; do not attribute
that movement to the reduction.

The unresolved variation triggers one fixed **500-comparison confirmation**,
using the same frozen tree/unrolled binaries in control, candidate, candidate,
control order. There is no rebuild or sample filtering:

| Case | Control persistent ms | Unrolled ms | Candidate/control, 95% interval |
|---|---:|---:|---:|
| CI inference | 0.069696 | 0.069491 | 0.9971 (0.9592–1.0392) |
| CI backward | 0.093186 | 0.093356 | 1.0018 (0.9936–1.0085) |
| CI SGD | 0.095005 | 0.094981 | 0.9997 (0.9921–1.0110) |
| Model forward | 5.137887 | 5.167241 | 1.0057 (0.9945–1.0711) |
| Model training + SGD | 17.872130 | 17.785119 | 0.9951 (0.9784–1.0058) |

All absolute persistent intervals now include no change. Ordinary training
improves **14.686630→14.150677 ms**, ratio **0.9635 [0.9427, 0.9699]**. The
paired persistent/ordinary training ratio grows **1.2094→1.2545**, relative ratio
**1.0373 [1.0140, 1.0565]**. Pooled bootstrap intervals do not capture all
between-visit variation. The experiment does not establish progress toward
persistent baseline parity; preserve the ordinary optimization separately.

Four complete-step SDK backend captures and sixteen metric groups measure
ordinary **11.9853→11.3701 ms (−5.13%)** and persistent
**15.9301→15.6469 ms (−1.78%)**. Estimated persistent warp instructions fall
2.86% and shared wavefronts 2.09%, with zero local traffic. Ordinary warp
instructions fall 2.00% and shared wavefronts 6.33%. Persistent barrier waiting
stays nearly unchanged, 63.97→63.95% of active warp cycles; ordinary changes
5.41→5.80%. These are not elapsed stall costs. Maximum clock/instruction-rate
spreads across metric groups are 0.361%/0.886%.

Independent hashing verifies bitwise-identical control/candidate states within
each backend across all 246,883,105 floats, with the same hashes as the preceding
experiment. Cross-backend maximum error stays 4.66e-9 under the unchanged budget.
All 1,250 normal comparisons and fifteen diagnostics pass state, selection,
completion and one-dispatch checks.

Revert the experiment and its dependency patches. Both rebuilt benchmark
binaries match the retained `e4e70c5` control byte for byte, the lock is restored,
and both fork worktrees are clean. Source, binaries, three-way samples,
confirmation design/results, captures and SDK states remain under local
`target/attention-shuffle-unrolled/`; the manifest is
`target/megakernel-results/attention-shuffle-unrolled/summary.json`.
It contains 663 independently verified artifact hashes and 53 recorded bounded
cgroups; SHA-256
`e63e24ead7641824d41ab1ce7fddcfa88a4c4eb8e44eca82c6c173b0214cddaa`.
All heavy jobs use separate verified 16/20/1 GiB cgroups, GPU jobs serialize,
and builds or heavy analyses do not overlap timing or SDK capture.

Next, audit inactive-lane guard costs on packed claims whose physical groups
are all proved occupied. Any specialization must cover every grid sharing the
handler and retain the original guards for partial claims, tails and unsupported
geometries. This differs from the earlier full-width logical-tile mask probe.

## Subgroup attention reduction

This rejected experiment replaces the common rowwise dQ/dKV reduction with a
shuffle helper on devices reporting compute shuffle support and a nominal
subgroup size of 32. The helper checks its runtime geometry, reads the first
pair of operands from shared memory, and finishes the same ordered tree with
subgroup shuffles. Other helper geometries use the original tree. Seven helper
barriers become two; the caller's reuse barrier remains. Both ordinary and
persistent paths receive the same module.

The experiment adds a temporary Blade shuffle capability and corrects Naga's
compute subgroup-size/count uniformity classification, preserving fragment
subgroup-size nonuniformity. Partial head packing remains enabled, with the
existing dense/full-subgroup admission checks. Logical lane numbering follows
subgroup coordinates without assuming a relationship to physical local IDs.
The symbolic source audit preserves all 64 leaves and 63 additions and checks
active shuffle sources. Emitted-source auditing preserves the fallback exactly,
all other handlers, globals, scheduler and completion accounting, and verifies
the new lane mapping. All 102 storage-read sites remain guarded.

An initial diagnostic capture caught a routing error: the rowwise scalar
variant bypassed the new device-aware generator. Its shader remained unchanged,
so those captures cannot measure this optimization. Fixing the actual route and
adding a runtime-selection regression precede all timings below. The initial
captures and qualification failures remain in the artifact record. A later
text-audit failure is corrected by preserving field names during one-to-one
local-variable renaming; the remaining statements compare exactly.

The integrated candidate passes 673 library tests (eight ignored), 52 codegen
tests, 168 strict whole-graph oracle tests, 50 matrix regressions, ten graph
smoke tests, 22 attention references in each execution mode, Clippy, formatting
and Metal compilation. Metal GPU execution remains unqualified. The mixed-head
fixture preserves exact outputs with poisoned scratch, partial dimensions,
three replays and 1/3/64/4096 workers.

Naga passes 144 unit, 270 integration and six doctests (24 ignored), plus strict
Clippy. The required full wgpu run has 1,145 passes, eight failures and six skips.
Three missing-DXC failures pass after installing the compiler locally. The
other five depth/stencil failures reproduce with identical error messages on
the unchanged `bc7b70b` fork revision. The full wgpu suite is therefore not
green. Tool-dependent SPIRV-Cross snapshot noise is archived and restored after
checking that its SPIR-V inputs are unchanged.

The primary sequence runs control, candidate, candidate, control, with 25 paired
samples per case per visit: 500 comparisons. The control is retained runtime
`e4e70c5`; requested workers remain 256. Every comparison passes the unchanged
state budget and uses one persistent dispatch without fallback.

| Case | Control persistent ms | Candidate ms | Candidate/control, 95% interval |
|---|---:|---:|---:|
| CI inference | 0.069025 | 0.069456 | 1.0062 (0.9823–1.0635) |
| CI backward | 0.093316 | 0.093186 | 0.9986 (0.9924–1.0068) |
| CI SGD | 0.096266 | 0.094736 | 0.9841 (0.9477–1.0003) |
| Model forward | 5.163906 | 5.142052 | 0.9958 (0.9882–1.0056) |
| Model training + SGD | 17.972040 | 17.674273 | 0.9834 (0.9692–0.9938) |

Ordinary training regresses **14.695235→15.888007 ms**, ratio
**1.0812 [1.0473, 1.0861]**. The apparent paired persistent/ordinary improvement,
1.2170→1.1225, largely reflects a slower baseline. This is not an acceptable
way to close the gap. Pooled bootstrap intervals do not capture all between-visit
variation. CI and forward shaders/resources are unchanged, so their movement
is not attributed to the patch.

Training workers use 79 rather than 80 registers, retain 29,344 shared bytes,
and shrink from 264,320 to 262,912 code bytes. Diagnostic participation remains
252 workers in training and 168 in forward, with 256 requested. The two ordinary
attention shaders instead use 40 registers each, previously 30/35; their shared
allocations grow from 512 bytes each to 1,536/2,048 bytes. These compiler
statistics describe an additional cost, but do not establish its elapsed-time
contribution or an occupancy change. Native instructions are unavailable.

Four complete-step SDK backend captures and sixteen metric groups confirm
persistent replay **15.9159→15.6716 ms (−1.54%)**, versus ordinary
**11.9885→12.8687 ms (+7.34%)**. Persistent estimated warp instructions fall
2.43% and shared wavefronts 1.62%, while integer thread instructions rise 0.44%.
Ordinary shared wavefronts fall 4.67%, but warp and integer thread instructions
rise 2.54% and 4.97%. Fewer shared accesses and source barriers do not establish
a net win. Barrier waiting changes 64.13→63.82% of active warp cycles in
persistent execution and 5.39→8.13% in ordinary execution; these are not elapsed
stall costs. Maximum clock and instruction-rate spreads across groups are
0.401% and 0.411%.

Independent hashing confirms bitwise-identical control/candidate state within
each backend across all 246,883,105 floats. The backend hashes remain those
recorded in the preceding experiment; cross-backend maximum error remains
4.66e-9 under the unchanged budget. Cross-backend state is not bitwise identical.

Revert the runtime, shader, Blade and Naga changes. Both rebuilt release
benchmark binaries match the retained control byte for byte; the dependency
lock is restored and both fork worktrees are clean. Preserve the candidate
source, patches, binaries, raw samples, generated shaders and SDK states under
local `target/attention-shuffle/` and `target/attention-shuffle-integrated/`.
The manifest `target/megakernel-results/attention-shuffle/summary.json` contains
846 independently verified artifact hashes and 74 recorded bounded cgroups;
SHA-256 `353988b31267f56938642a0e2ba432f08e47a1cf980f4f2df40d1f5130668c23`.
Every heavy job uses a separate verified 16/20/1 GiB cgroup; GPU jobs serialize,
and builds or heavy analyses do not overlap measurements.

Next, separate runtime subgroup-geometry overhead from the looped shuffle
helper. A statically unrolled exact-order helper is a distinct candidate, not
an established fix. Preserve the runtime fallback and all existing gates, and
require ordinary execution to avoid regressions as well.

## Serial attention reduction tail

The rowwise dQ and dKV helpers use seven workgroup barriers per dual reduction,
plus a caller barrier before reusing scratch. This probe lets lane zero finish
the last three levels after the eight-lane stage, reducing the helper to five
barriers. It preserves the original addition tree, operand order, masks and
caller synchronization. Both ordinary and persistent paths use the same
candidate shaders. No subgroup operation, capability, packing rule, scratch
allocation or worker-count policy changes.

An independent interpreter evaluates the actual original and candidate helper
source as ordered symbolic trees: all 64 leaves and 63 additions match for
both arrays in each helper. It checks cross-lane reads/writes between barriers
and verifies that callers consume only element zero and reinitialize every
lane before the next reduction. This source audit supplements GPU execution
checks; it does not describe native instructions.

Qualification passes 671 library tests (eight ignored), 52 code-generation
tests, 168 strict whole-graph oracle tests, 50 matrix regressions, ten
whole-graph smoke tests, 22 attention references in each mode, strict Clippy,
formatting and Metal compilation. Metal GPU execution remains unqualified.

Ten diagnostic comparisons preserve selection, geometry, buffers, scheduler
metadata, logical work and the one-dispatch boundary. Generated source differs
only in the two reduction helpers and harmless local-variable renaming in
the dKV caller. The initial text audit rejected that rename; the corrected
audit verifies a one-to-one local renaming and checks the remaining source.
Both audit versions and the initial failure are preserved. The independent
Naga access audit still finds 102 guarded storage-read sites and valid shared
read/write protection.

All CI and forward shaders/resources remain unchanged. The training worker
keeps 80 registers and 29,344 shared bytes; code size falls 264,320→263,552
bytes. Diagnostic participation remains 168 workers forward and 252 training,
with 256 requested. Native instruction bytes are unavailable.

The primary run has 500 completed-step comparisons, ordered control, candidate,
candidate, control, with 25 paired samples per case per visit. The control is
the retained storage repair, `e4e70c5`. Full-state budgets remain unchanged.

| Case | Control persistent ms | Candidate ms | Candidate/control, 95% interval | Candidate paired persistent/ordinary |
|---|---:|---:|---:|---:|
| CI inference | 0.069441 | 0.069506 | 1.0009 (0.9619–1.0169) | 1.291× |
| CI backward | 0.092991 | 0.093335 | 1.0037 (0.9977–1.0156) | 1.434× |
| CI SGD | 0.095166 | 0.096525 | 1.0143 (0.9942–1.0499) | 1.435× |
| Model forward | 5.186537 | 5.144061 | 0.9918 (0.9866–1.0043) | 1.368× |
| Model training + SGD | 17.727893 | 17.877664 | 1.0084 (0.9909–1.0463) | 1.237× |

Every absolute and normalized persistent timing interval includes no change.
Pooled bootstrap intervals do not capture all between-visit variation. Timing
movement in unchanged CI/forward shaders is not attributed to this patch.

Four full-step SDK backend captures / sixteen sequential metric groups preserve
all 246,883,105 state floats bitwise within each backend. Hashing the actual
state files independently verifies this; cross-backend error remains at most
4.66e-9 under the existing budget. The backends are not bitwise identical.
Persistent diagnostic replay changes 15.9458→15.8365 ms (−0.69%), while ordinary
changes 11.9655→11.8055 ms (−1.34%). Estimated persistent warp instructions and
shared wavefronts each fall 1.05%; integer instructions change −0.07%, FP32
instructions +0.03%, and global-load sectors +0.03%. Persistent local traffic
stays zero. Barrier waiting changes 64.03→63.92% of active warp cycles.
Maximum sequential clock/instruction-rate spreads are 0.411%/0.575%.

The counter reductions are small and do not establish a completed-step gain.
Barrier shares are fractions of active warp cycles, not elapsed-time
attribution. This experiment does not show that other barriers are removable
or that all synchronization costs are negligible. The patch is reverted,
and ordinary release examples are rebuilt from the restored runtime.

Evidence is local under `target/attention-tree-tail8/`; the candidate patch,
frozen binaries, generated shaders, states and failed audit remain available.
Manifest: `target/megakernel-results/attention-tree-tail8/summary.json`.
It contains 451 independently verified artifact hashes and 39 recorded bounded
cgroups; SHA-256
`a2b20605778269ef2fdf1c0befa8356a8bd9d17bb4d47f9749c3837a354e12b2`.
All heavy jobs use separate verified 16/20/1 GiB cgroups, with serialized GPU
work and no builds, copies or heavy analyses overlapping measurements.
No dependency pin or Naga/Blade source changes.

The current capability API exposes subgroup voting but not shuffle support.
Any subsequent exact-order shuffle probe must expose and check that capability,
retain the ordinary tree fallback, and preserve head packing and uniform
barrier admission. Do not infer its benefit from this smaller serial-tail
probe. The five completed-step gates and performance-parity goal remain open.

## Inactive storage reads and worker participation

The [storage-access contract](megakernel_memory_contract.md#inactive-storage-accesses)
now guards every storage load in an adapted smaller handler. This covers both
in-place updates and read-only bindings that alias a writable view. Active
addresses, operands and memory-access order remain intact. The change adds no
buffers, descriptors or fences, and preserves the existing shared-read guards,
workgroup barriers, queue, metadata and single-dispatch boundary.

A new structural test fails on `137bfdd` and passes after the repair. It uses
an in-place write and two views of one canonical descriptor. An independent
Naga audit inspects all captured adapted handlers: their 102 storage-read sites
change from unguarded to guarded, while shared-load protection and write guards
remain valid. All 671 library tests (eight ignored), 52 code-generation tests,
168 strict graph oracle tests, 50 matrix regressions, ten whole-graph tests,
22 attention references in each mode, strict Clippy, formatting and Metal
compilation pass. Metal compilation does not qualify Metal GPU execution.

All ordinary shaders/resources and all CI worker shaders/resources remain
unchanged. Model forward registers fall 86→84, with 18,928 shared bytes and
code size 107,904→112,000 bytes. Training registers fall 91→80, with 29,344
shared bytes and code size 270,848→264,320 bytes. In separate diagnostic steps,
workers receiving tasks remain 168 forward but rise 168→252 for training.
These counts describe participation, not a direct residency measurement.

The primary run uses 500 completed-step comparisons: 25 paired samples per
case in two control and two candidate visits, with 256 workers and unchanged
full-state budgets. The control is the scratch-only repair, `137bfdd`.

| Case | Control persistent ms | Storage-guarded ms | Guarded/control, 95% interval | Paired persistent/ordinary |
|---|---:|---:|---:|---:|
| CI inference | 0.070175 | 0.069131 | 0.9851 (0.8756–1.0279) | 1.269× |
| CI backward | 0.093246 | 0.093625 | 1.0041 (0.9972–1.0106) | 1.436× |
| CI SGD | 0.095896 | 0.094590 | 0.9864 (0.9530–0.9966) | 1.415× |
| Model forward | 4.962865 | 5.126381 | 1.0329 (0.9899–1.1077) | 1.355× |
| Model training + SGD | 16.911478 | 17.855279 | 1.0558 (1.0431–1.0756) | 1.219× |

Training regresses in absolute time; its normalized interval includes no
change. Forward intervals also include no change. CI timing moves despite
identical shaders/resources, so its apparent SGD improvement is not attributed
to the repair. Pooled intervals do not capture all between-visit variation.

Four full-step SDK backend captures / sixteen metric groups preserve all
246,883,105 captured floats bitwise within each backend. Actual state hashes
match the preceding runtime. Cross-backend maximum error remains at most
4.66e-9 under the unchanged budget; the backends are not bitwise identical.
Persistent diagnostic replay rises 14.7897→15.9556 ms (+7.88%), while ordinary
changes +0.18%. Estimated integer instructions fall 10.54%, but warp instructions
rise 10.59%, shared wavefronts rise 3.21%, and global-load sectors change +0.28%.
FP32 instructions are effectively unchanged and local traffic remains zero.
Barrier waiting rises 53.08→64.03% of active warp cycles. Maximum sequential
clock/instruction-rate spreads are 0.407%/0.896%. These are sampled rate-times-
duration estimates; barrier shares do not attribute elapsed-time cost.

A leader-only polling probe then runs 200 comparisons against the guarded
runtime. Other lanes wait at the existing uniform broadcast while the leader
rescans; no fence or worker-count change is introduced. All 671 library tests,
52 code-generation tests and ten whole-graph tests pass. Handler bodies and
worker main bodies are byte-identical to the guarded control. Model register,
shared-memory and executable-size statistics are unchanged. Forward changes
4.846→4.977 ms and training 17.749→17.691 ms, with wide intervals including
no change. The probe is reverted; its patch and binaries are preserved.

A separate 200-comparison diagnostic uses the same corrected binary and
identical pipeline keys/resources at explicit worker counts of 168 and 256.
It does not change the primary gate or cap the implementation's requested
workers:

| Case | 256 workers ms | 168 workers ms | 168/256, 95% interval |
|---|---:|---:|---:|
| Model forward | 5.128862 | 5.132506 | 1.0007 (0.9518–1.0643) |
| Model training + SGD | 17.981100 | 16.824818 | 0.9357 (0.9241–0.9508) |

Training improves 6.43% in absolute time at 168 workers. Its normalized interval
includes no change. Forward still has 168 workers receiving tasks at either
count; training participation changes 252→168. This supports a cost from
additional worker concurrency in this sequence, without distinguishing queue
contention, synchronization and other residency effects. It does not establish
a universal worker-count policy or performance parity.

The complete record contains 900 normal comparisons, 24 diagnostics, and the
four SDK backend captures. Manifest:
`target/megakernel-results/inactive-storage-reads/summary.json`, with 810
independently verified artifact hashes and 65 recorded bounded cgroups;
SHA-256 `1c9259057a2f56395f1726c9f3231cd24d19c43c22d5364407a8b43a6eda01a8`.
Evidence is local under `target/inactive-storage-reads/` and
`target/inactive-storage-leader-poll/`. The reverted shader and restored source
match the qualified storage-guard runtime exactly. No dependency pin or
Naga/Blade source changed. All heavy jobs completed in separate verified
16/20/1 GiB cgroups; GPU jobs were serialized and measurements did not overlap
builds, copies or heavy analysis.

The next experiment should target productive synchronization in rowwise
attention gradients. Check shuffle capability and exact addition order before
replacing tree-reduction barriers, and preserve head packing when introducing
subgroup builtins. Fewer instructions or lower register counts alone remain
insufficient evidence for retention.

## Inactive scratch reads

The [memory contract](megakernel_memory_contract.md#inactive-scratch-accesses)
describes the source-level race and repair. Store masking alone left inactive
loads outside the mask. Ordinary shared loads now select a reserved address;
cooperative shared loads are skipped by inactive subgroups. Original active
indices, arithmetic, workgroup barriers and queue publication remain intact.
This also addresses smaller unpacked handlers, not just partial head batches.
Inactive storage reads and intra-node in-place/aliased bindings remain a
separate audit item.

Qualification passes 670 library tests (eight ignored), all 168 strict graph
oracle tests, 50 matrix regressions, ten whole-graph smoke tests and 22 attention
references in each mode. All 51 code-generation tests, strict Clippy, formatting
and Metal compilation pass. Metal compilation is not Metal GPU qualification.
The new structural test checks reserved addresses for scalar/vector loads,
helper calls, pooled-bank growth and an original helper argument whose name
resembles the generated activity argument. The mixed-geometry test also checks
that cooperative loads are inside the activity guard.

All ordinary shader files and ordinary pipeline resources match `2997a85`.
The queue claim function and worker main body match apart from the one-time
reserved-slot initialization. Selected implementations, buffers, metadata,
logical completion and the single-dispatch boundary stay unchanged. Model
forward redirects 29 ordinary shared loads; training redirects 81. Forward
registers fall 100→86, shared bytes rise 18,832→18,928, and code bytes rise
107,648→107,904. Training registers fall 98→91, shared bytes rise
29,232→29,344, and code bytes rise 269,824→270,848. These resource changes alone
do not establish a speedup.

The normal run contains 500 completed-step comparisons: 25 paired samples per
case in each of two control and two candidate visits, ordered control,
candidate, candidate, control. It uses 256 requested workers and the same
warmup, arithmetic selection and full-state budgets as prior measurements.
The control is historical `2997a85`, with the known inactive-read defect.

| Case | Control persistent ms | Repaired persistent ms | Repaired/control, 95% interval | Paired persistent/ordinary |
|---|---:|---:|---:|---:|
| CI inference | 0.069920 | 0.069920 | 1.0000 (0.9886–1.0642) | 1.306× |
| CI backward | 0.093381 | 0.093016 | 0.9961 (0.9834–1.0024) | 1.433× |
| CI SGD | 0.094941 | 0.095280 | 1.0036 (1.0003–1.0106) | 1.416× |
| Model forward | 4.941066 | 4.893955 | 0.9905 (0.9227–1.0794) | 1.338× |
| Model training + SGD | 16.928983 | 17.001523 | 1.0043 (0.9876–1.0161) | 1.158× |

The absolute CI SGD result is a small 0.36% regression; its normalized interval
includes no change. Both model absolute and normalized intervals include no
change, with substantial variation in forward timing. Pooled bootstrap
intervals do not account for all between-visit variation. The repair is a
correctness change, not a demonstrated performance optimization.

A final namespace-matching correction prevents an original helper argument
from being mistaken for the generated activity flag. After rebuilding, all
62 captured shader files and all five cases' pipeline resources match the
timed candidate exactly. Five additional diagnostic comparisons pass full
state checks. The correction affects construction outside the timed boundary;
native instruction bytes are unavailable. SDK captures use the corrected
binary, and both binary versions are preserved.

Four full-step SDK backend captures, with sixteen sequential metric groups and
zero-rate SGD for stable replay, preserve all 246,883,105 captured floats
bitwise within each backend. Independent hashing verifies the actual files.
Cross-backend states are not bitwise identical; their maximum error remains
at most 4.66e-9 under the unchanged budget. SDK replay changes
14.6935→14.7789 ms (+0.58%) for persistent execution and
11.9698→11.9862 ms (+0.14%) for ordinary execution.

Persistent estimated warp instructions change +0.98%, integer instructions
+1.63%, FP32 instructions −0.11%, global-load sectors −0.14%, and shared
wavefronts −1.48%. Local traffic remains zero. Barrier waiting is essentially
unchanged, 52.98→53.02% of active warp cycles. Maximum clock spread is 0.130%;
instruction-rate spread is 1.315% across the sequential groups. Rates multiplied
by diagnostic replay duration are estimates, not native instruction counts or
elapsed stall-time attribution. The corrected persistent capture still has
about 1.61× ordinary warp instructions and 2.11× integer instructions, with
53.02% barrier waiting versus ordinary's 5.34%. These remain profiling targets;
the scratch repair does not resolve the performance gap.

Rejected intermediate patches remain local. A separate scratch-bank prototype
first failed generated identifier resolution. Giving every physical group its
own writable scratch passed focused tests but exceeded the 48 KiB device limit
for wide attention. Conditional scalar-load variants changed exact outputs by
up to roughly 1e-7 and were rejected without changing tolerances; the cause of
those numerical differences is not established. Reserved load addresses avoid
introducing conditional floating-point values. The initial Clippy failure and
its correction are also preserved.

Manifest: `target/megakernel-results/inactive-scratch-reads/summary.json`, with
679 independently verified artifact hashes and 68 recorded bounded cgroups;
SHA-256 `afab156166ef0616cfb04303c1d50b75a2c97995b2716d429b108468a0c9753a`.
Raw evidence is local under `target/inactive-read-sentinel/`, including the
namespace-correction comparison. The three rejected prototype roots are
`target/inactive-scratch-isolation/`, `target/inactive-scratch-owned/`, and
`target/inactive-read-mask/`. All heavy jobs completed in verified separate
cgroups, with 16 GiB memory-high, 20 GiB memory-max and 1 GiB swap. GPU jobs
were serialized; builds, copies and heavy analyses did not overlap measurement.
No dependency pin or Naga/Blade source changed. Keep the PR as a draft; the next
step is the inactive-storage audit followed by profiling from this repaired
control.

## Resource and arithmetic contract

The same implementation selector and binding function supply ordinary and
whole-graph execution. A Blade visitor captures buffers and plain parameter
bytes, including the optimizer's existing segmented arena bindings. Kernels are
deduplicated by source, call and layout. Typed declarations alias one descriptor
array, with exactly one descriptor at offset zero per backing buffer. Node
parameters select its canonical array index and carry the original binding's
element offset and uniform bytes. This preserves descriptor identity across
types, slices and optimizer chunks, as required by shader-only synchronization.
Uniform word offsets are fixed by each handler's layout; descriptor indices and
element offsets are passed by value through its helpers.

One physical workgroup is at least 256 invocations and must be a multiple of
each selected logical workgroup. Eligible smaller tiles share the worker in
groups of up to four, each with its original logical coordinates and separate
scratch. The chosen grid axis minimizes queue claims. Subgroup/cooperative
handlers use divisor widths 4/2/1. Scalar handlers permit partial batches:
the claim space pads each packed axis, while completion counts only valid
logical tiles. Inactive scalar groups clamp logical coordinates to a valid
group. Ordinary shared loads select a reserved read-only element outside every
handler's scratch range; inactive cooperative subgroups skip shared matrix
loads. Claims cannot straddle the packed axis boundary. Inactive groups
participate in every workgroup barrier but do not write shared/storage memory
or execute cooperative products assigned to local accumulators. Other extra-lane
calculations are still repeated. Subgroup handlers use virtual subgroup coordinates. A shader guard
checks dense, linear subgroup mapping, full groups and compatible widths before
executing graph work. This mapping is a qualification condition, not a portable
Vulkan assumption: [subgroup and local invocation indices are not generally
related](https://docs.vulkan.org/refpages/latest/refpages/source/SubgroupLocalInvocationId.html).
Unsupported mapping sets a completion diagnostic and executes no tile in that
worker. Call `validate_persistent_completion()` during qualification.

Shared arrays reuse a pool by element type and slot; capacity is the maximum
across handlers, including each handler's packing width. Original per-tile
initialization remains in each callable body. Packing retries with smaller
widths if the combined scratch exceeds the device limit.
Each pooled bank reserves an additional 16 bytes (rounded to element stride)
for the inactive read address. Physical lane zero initializes that element once
before the worker loop, followed by a workgroup barrier. The allocation counts
toward admission limits. Handler stores never address the reserved element.
Uniform parameters are loaded through `workgroupUniformLoad` and passed by value
through tile helpers, preserving Naga's uniform-control-flow validation.
No validation flags related to barrier uniformity are disabled.

Packing requires uniform barrier control across the physical workgroup. An
admission-only module inserts workgroup-scope cooperative operations at barriers
to apply the fork's scope-aware dependency analysis through helper calls and
locals. These probes never appear in the runtime shader. Query-dependent causal
attention rows fail this check; independent heads at one row can pass. Native workgroup
broadcasts and switch-containing bodies also stay unpacked; the latter avoids
relying on incomplete post-switch exit analysis. Failure to qualify packing
retains the same handler inside the single graph dispatch.

SGD handlers first built for grids of at least 1,024 tiles also batch four
adjacent tiles sequentially when logical and physical workgroups match. This
amortizes queue claims, parameter broadcasts and publication across independent
updates, without changing the selected SGD body or element arithmetic. The
handler must have no workgroup scratch. A final partial batch executes and
counts only the remaining logical tiles; the cached handler can therefore also
serve later small chunks. Other optimizer handlers retain their existing claims.

Admission checks descriptor counts, storage ranges, shared memory (including
reserved workgroup cooperative storage), and counter sizes. Imported allocations
are rejected because distinct handles may alias without a provable relationship.
Workgroup cooperative tiles must already match the physical workgroup size;
atomic side effects currently require matching geometry as well. Unsupported
resources reject whole-graph selection even when regional fallback is enabled.
Storage arrays must be directly indexed and binding offsets must be divisible
by their element stride. This is an opt-in Vulkan path tested on the RTX 5080;
the [descriptor-reference audit](megakernel_memory_contract.md) supersedes earlier
qualification claims. It is not unrestricted support for arbitrary shaders or hardware.

## Dependencies and completion

The ready bitmap contains nodes, not individual tiles. Each node has a next-tile
counter, remaining-tile counter and predecessor count. A worker claims one or
more tiles from a ready node; the final claim clears its
ready bit. Completion subtracts that claim's logical tile count, and the final
completion releases every successor whose predecessor count reaches zero. Zero-tile
nodes use one control task to propagate dependencies and perform no arithmetic.

Edges point forward in recording order. Conflicting reads/writes of the same
backing allocation create RAW/WAR/WAW edges, covering arena slices and lifetime
reuse. Reads and unrecognized writes cover the entire allocation. Selected scalar
`MatMulAT` and `SumRows` outputs use proved byte ranges, allowing disjoint writes
without weakening physical read/write hazards. Rebinding rebuilds these ranges
from the actual buffer pieces. Transitive dependency edges are omitted without
changing reachability. Producer and consumer lanes use the existing
[native fence protocol](megakernel_memory_contract.md). Naga combines writes
across all declarations mapped to the same descriptor binding, preserving
NonPrivatePointer accesses through typed aliases. Tile-boundary fences provide
availability and visibility, with explicit SPIR-V `Aliased` decorations.
The ready load, tile-completion RMWs and predecessor RMWs carry the ordering
chain; every consumer lane performs its own storage visibility operation.

Idle workers keep looking for ready work until the global remaining-node count
is zero. No workgroup waits at a global barrier or requires an unscheduled
workgroup to become resident. Admission constructs a DAG, and finite executing
tiles publish their successors. Completion validation checks remaining nodes,
ready bits, remaining tiles, predecessor counters, diagnostics and optional
per-node worker claim totals.

`Session::record` creates separate immutable scheduler parameters and queue
storage for each outstanding recording. Recordings are associated with the
submission reported through `track_submission`; report submissions in recording
order before recording later unsubmitted work. Waiting for an older submission
does not permit a newer unsubmitted recording's metadata to be overwritten.
After completion, unchanged topology reuses descriptors, dependencies, queue
storage and pipeline; only uniform values are uploaded. The existing external
recording API still rejects temporal accumulation, while `step()` supports it.

`ExecutionReport::compute_dispatches` reports physical launches including the
optimizer. `appended` rows describe logical optimizer phases within the same
dispatch when `device_scheduled` is true. One queue reset is a transfer operation,
not a compute launch. A profiling interval covers the combined graph; it cannot
attribute time to its internal nodes.

## Qualification and measurements

The [cooperative prologue staging measurements](#contiguous-cooperative-prologue-staging)
are the current qualification record. Measurements before `ed81459` below are
historical hardware observations: they predate the descriptor-reference repair
and do not establish its memory-model contract.

Initial checks cover 1, 3, 64 and 4096 workers, repeated resets, empty nodes,
fan-in/fan-out, physical allocation reuse, external repeated recording, selected
matrix/attention kernels, mixed 128/256-lane reductions, and exact small-graph
parameters, gradients and optimizer state. Vulkan validation is enabled in the
GPU tests. All seven focused whole-graph tests pass. Packed matrix coverage
includes two- and four-tile claims, partial rows and repeated scratch reuse
with 1, 3, 64 and 4096 workers. The external-recording test
also waits for a submitted step while a later recording with a different
learning rate remains unsubmitted, then verifies all three parameter updates.
The optimizer fixture includes a 1,025-tile SGD chunk with a partial final
batch, followed by small chunks reusing the handler, across all four worker
counts and repeated optimizer/reset stages.

On RTX 5080, strict whole-graph selection passes all **168 oracle tests**,
including attention, convolution, reduced-precision storage, scatter, fusion and
backward cases. Queue completion is checked by the shared reference harness.
The earlier broad ordinary-path integration run passed **664 library, 168 oracle, 182
regression, 81 smoke and 22 GGUF tests** (8 library, 21 regression and 2 smoke
tests remain ignored).
Formatting, all-target Clippy, bare/all-feature checks and documentation pass.
The Metal library cross-check passes with `models,gguf`; all-feature cross-checking
is blocked by the Linux C compiler's unsupported Apple flags in `ring`.
No Metal GPU execution is claimed.

The concurrent-packing change passes **665 library tests** (eight ignored),
all **168 strict whole-graph oracle tests**, the seven focused graph tests,
formatting and strict all-target/all-feature Clippy. The Naga and Blade
dependency revisions are unchanged.

```sh
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
MEGANEURA_PERSISTENT_WORKERS=64 MEGANEURA_DEVICE_SCHEDULING=1 \
MEGANEURA_WHOLE_GRAPH=1 MEGANEURA_PERSISTENT_STRICT=1 \
cargo test --test oracle --all-features -- --test-threads=1
```

Logs remain on local storage under `target/whole-graph-*.log`.

## Completed-step benchmark

Clean implementation `96e57613f5e2feaba79755f8b30a9197c592d52e`, RTX 5080
(GB203), NVIDIA 595.91.07. Each row uses 50 alternating ordinary/selected pairs
after at least ten warmup pairs and one second of warmup. Parameters are restored
outside timing; each sample waits for completion and checks outputs, gradients
and updated parameters. SDK sampling, GPU timestamps, worker counters, builds
and artifact copies are disabled during primary measurements.
SmolVLA times `step()` plus completion wait; the CI boundary also includes input
upload and output/loss readback. Combined pipelines compile lazily on first use
and are warmed before timing. The harness's `construction_ms` field measures
Session construction and therefore excludes that first-use compilation; it is
not a complete cold-start metric for this backend.

| Workload | Workers | Ordinary median, ms | Whole-graph median, ms | Median paired ratio | 95% bootstrap interval |
|---|---:|---:|---:|---:|---:|
| SmolVLA forward | 64 | 4.436 | 13.241 | 2.989 | 2.972–3.016 |
| SmolVLA forward/backward/SGD | 64 | 12.302 | 45.257 | 3.495 | 3.000–3.509 |
| SmolVLA forward | 256 | 4.398 | 10.103 | 2.300 | 2.294–2.313 |
| SmolVLA forward/backward/SGD | 256 | 15.134 | 28.536 | 1.886 | 1.851–1.913 |
| CI inference | 256 | 0.0573 | 0.0757 | 1.323 | 1.318–1.332 |
| CI forward/backward | 256 | 0.0655 | 0.0994 | 1.523 | 1.508–1.533 |
| CI forward/backward/SGD | 256 | 0.0677 | 0.1054 | 1.570 | 1.551–1.582 |

All **350 comparisons** pass. SmolVLA forward and all CI cases are bitwise
equal; SmolVLA training's maximum absolute difference is `4.6566128730773926e-9`,
within the existing comparator tolerance. Both paths have identical compiled
plans and selected implementation keys. Every whole-graph case reports exactly
**one physical compute dispatch**, with no fallback. SmolVLA's ordinary path
has 154 forward dispatches and 527 forward/backward/SGD dispatches. The four SGD
rows in the whole-graph report are logical nodes in its single dispatch.

The uninstrumented scheduler allocations are 51,904 bytes for SmolVLA forward
and 98,668 bytes for training. These include metadata, immutable parameters and
both queue/reset buffers. Separate outstanding external recordings retain
additional snapshots until their submissions complete.

Intervals bootstrap each row's paired ratios 10,000 times using seed `20261025`
in table order. They describe within-run sampling uncertainty. The 64-worker
training distribution is broad, and ordinary training medians differ between
runs; a median paired ratio need not equal a ratio of marginal medians. These
data establish correctness and a regression, not a portable optimum or speedup.

## NVIDIA counter diagnosis

The Perf SDK 2026.3 forward captures use the same ten-second resident replay,
three-second monitor and 20 ms sampling interval as the
[regional investigation](megakernel_performance_analysis.md). Each capture has
146 samples after discarding the first five, successful target/monitor exits and
no SDK warnings. Ordinary, 64-worker and 256-worker captures have identical
implementation keys and bitwise-identical state, SHA-256
`4e5d66e6ce5e95142b848651d316b4df64855a9d85669499cabc3a81089be629`.

| Forward metric | Ordinary | Whole graph, 64 | Whole graph, 256 |
|---|---:|---:|---:|
| Diagnostic replay median, ms | 3.789 | 12.633 | 9.857 |
| SM active, % | 61.36 | 75.38 | 98.60 |
| Active compute warp slots, % of peak | 9.55 | 12.56 | 32.87 |
| Tensor pipe activity, % of peak | 2.66 | 3.19 | 4.14 |
| DRAM read throughput, % of peak | 11.12 | 3.35 | 4.30 |
| L2 texture-sector hit rate, % | 69.97 | 93.18 | 93.17 |
| Launched workgroups per second | 4,404,327 | 5,129 | 26,301 |
| GPC clock, GHz | 2.842 | 2.845 | 2.834 |

The lane adapter has a concrete avoidable cost: a logical 64-invocation tensor
tile executes in a physical 256-invocation workgroup. Extra subgroup lanes
repeat the tile's calculations and cooperative matrix instructions, while their
stores are masked. That preserves barrier participation and arithmetic but
performs four copies of this calculation. The product of tensor activity and
replay duration is **4.00× / 4.05×** ordinary for 64/256 workers. This is an
aggregate activity-time estimate, not an exact per-kernel instruction count;
its agreement with the explicit fourfold duplication is strong evidence for
this cost. More active warp slots do not imply more useful graph work.

Multiplying workgroup launch rate by replay duration estimates about 65 and 259
workgroups per step, close to the selected fixed pools of 64 and 256. This
independently supports the change in launch behavior. The ordinary estimate is
16,689, close to its 16,538 logical workgroups. Clocks are nearly unchanged and
DRAM throughput is far below peak, so clock throttling and aggregate bandwidth
saturation do not explain this regression.

Separate five-pair timestamp/statistics runs report **98 registers/thread and
26,112 shared bytes/workgroup** for the combined forward shader; training uses
**91 registers/thread and 26,128 shared bytes/workgroup**. Every operation pays
the combined pipeline's resource footprint. Ready-queue polling, dynamic
descriptor/parameter indexing, conservative allocation dependencies and coherent
accesses/fences add costs, but these captures do not isolate their individual
latencies. The driver's implausible approximately 64 GiB local-memory statistic
is excluded from spill claims. The timestamp training run contains a 90 ms
outlier; the table above uses the separate primary measurements.

## Optimization in progress

[Naga f4fe7e8](https://github.com/kvark/wgpu/commit/f4fe7e85790d44f46c7107675ca075a3471d6163)
checks cooperative control flow at the matrix's scope, including helper calls,
locals and loop-carried dependencies. Subgroup-uniform activity arguments can
therefore guard the extra subgroups' products. Workgroup operations retain their
existing validation; no workgroup barrier is placed inside the new condition.
The ordinary arithmetic and logical geometry are unchanged. The corresponding
[Blade pin](https://github.com/kvark/blade/commit/fd04c27dff43ab02894186d26078afeb833c9631)
uses the same Naga identity. Atomic syntax and fence semantics are unchanged.

A separate SDK capture after masking those products estimates **1.012× ordinary
tensor activity per step**, down from approximately 4×. This is the ratio of
tensor activity × replay duration, not an instruction count. Ordinary and
whole-graph replay medians are 3.793 ms and 9.277 ms; both captures have 146
counter samples and identical complete forward state. Removing duplicate tensor
work alone therefore does not explain or remove the whole slowdown. Captures are
in `target/megakernel-results/masked-products-sdk/` with commands, binary hashes
and source-status receipts. They precede the later indexing/fence changes.

Further changes cache descriptor indices as function arguments, reconstruct
uniforms with constant offsets, remove transitive dependency edges, and omit
per-access availability/visibility for descriptor globals that Naga proves are
written by this dispatch. Those globals retain NonPrivatePointer on every
access, so the existing tile-boundary fences cover communication. Read-only
aliases keep `@coherent` to prevent private-access elision. A generated-SPIR-V
check verifies both classes of table accesses, including mixed geometry and a
cooperative handler.

Qualification passes 664 library tests, all 168 strict whole-graph oracle tests,
and the six focused graph tests. Naga passes 144 unit, 268 integration and six
documentation tests (24 ignored), plus strict all-target/all-feature Clippy;
Blade passes 20 unit tests (one ignored). Meganeura's strict Clippy also passes.
Clean commit `58458d2bcd1fb4c518be714e2e11e27130b95fdd` was then measured with
the same completed-step boundaries, 50 alternating pairs per row and at least
ten warmup pairs / one second of warmup. No diagnostics or concurrent builds
were active during these measurements.

| Workload | Workers | Ordinary median, ms | Whole-graph median, ms | Median paired ratio | 95% bootstrap interval |
|---|---:|---:|---:|---:|---:|
| SmolVLA forward | 256 | 4.448 | 8.426 | 1.897 | 1.890–1.910 |
| SmolVLA forward/backward/SGD | 256 | 15.123 | 23.134 | 1.534 | 1.518–1.549 |
| CI inference | 256 | 0.0517 | 0.0665 | 1.278 | 1.252–1.423 |
| CI forward/backward | 256 | 0.0663 | 0.1003 | 1.513 | 1.487–1.522 |
| CI forward/backward/SGD | 256 | 0.0683 | 0.1037 | 1.515 | 1.477–1.537 |
| CI inference | 64 | 0.0522 | 0.0687 | 1.317 | 1.310–1.326 |
| CI forward/backward | 64 | 0.0665 | 0.0975 | 1.459 | 1.419–1.472 |
| CI forward/backward/SGD | 64 | 0.0686 | 0.1008 | 1.474 | 1.459–1.485 |
| CI inference | 16 | 0.0522 | 0.0689 | 1.316 | 1.305–1.328 |
| CI forward/backward | 16 | 0.0670 | 0.0992 | 1.475 | 1.432–1.493 |
| CI forward/backward/SGD | 16 | 0.0682 | 0.1020 | 1.495 | 1.485–1.505 |

All **550 comparisons** pass with identical plans and arithmetic selection,
one persistent dispatch and no fallback. Forward and CI state are exact;
SmolVLA training's maximum absolute difference remains
`4.6566128730773926e-9`. Bootstrap settings match the earlier table, with samples
drawn in sorted artifact-filename order and then case order. Results and SHA-256
identities are in `target/megakernel-results/whole-graph-opt-58458d2/summary.json`.
**Baseline parity has not been reached.** The SmolVLA paired ratios improve
from 2.300 / 1.886 to 1.897 / 1.534; the small CI graph still has substantial
GPU scheduling overhead. Separate diagnostic timestamps place CI inference at
about 13 / 31 microseconds of GPU time and training at 28 / 72 microseconds for
ordinary / persistent execution. These timestamps are not primary timings.

The same commit's separate SDK forward capture estimates **1.028× ordinary
tensor activity per step**, with replay medians 3.775 / 7.995 ms. Both captures
have 146 samples, exact state and successful target/monitor exits without SDK
warnings. Receipts are under the results directory's `sdk-forward/` subdirectory.
Thus the duplicate matrix work is removed, while the remaining slowdown needs
resource and scheduling improvements.

## Instruction and local-memory costs

A second SDK metric set measures instructions and memory transactions during
the same state-checked forward replay. The script accepts `--metrics` for
reproducible single-pass sets; exact metric names and commands are saved in
each receipt. The SDK's derived stall percentages require multiple passes on
this device and cannot be collected by its standalone live sampler. Individual
raw stall components can be sampled in sequential single-pass groups, as
described below; failed configuration probes are excluded from attribution.

At the `58458d2` implementation, ordinary / persistent replay medians are
3.785 / 7.988 ms. Counter rates multiplied by replay duration estimate the
following per-step work. These are whole-workload activity estimates, not
per-kernel counts or isolated timings.

| Metric | Ordinary | Persistent | Ratio |
|---|---:|---:|---:|
| Warp instructions, millions | 389.8 | 1137.6 | 2.919 |
| Predicated-on integer FMA-pipe thread instructions, millions | 701.0 | 2910.6 | 4.152 |
| Heavy FP32-pipe thread instructions, millions | 1770.1 | 1718.6 | 0.971 |
| Light FP32-pipe thread instructions, millions | 2649.1 | 2890.7 | 1.091 |
| Global-load sectors, millions | 51.11 | 136.78 | 2.676 |
| Global-store sectors, millions | 2.813 | 2.843 | 1.011 |
| Local-load sectors, millions | 0 | 14.139 | — |
| Local-store sectors, millions | 0 | 15.695 | — |
| Shared-memory wavefronts, millions | 128.96 | 208.88 | 1.620 |

The local-memory traffic is measured directly, independent of the driver's
unusable local-memory-size statistic. Explicit scalar output accumulators in
cooperative attention replace its dynamically indexed local array, using the
existing template expansion in both ordinary and persistent paths. The order
of each accumulator's operations, precision and selected kernel remain the same.
The follow-up SDK capture estimates **0.041 million local-load sectors and
0.690 million local-store sectors** per persistent step: reductions of 99.7%
and 95.6%. Replay medians are 3.797 / 7.553 ms. Both captures have 146 samples
and the same bitwise-exact state hash as the original forward capture. These
receipts are under `target/megakernel-results/whole-graph-memory-sdk/` and
`scalar-attention-sdk/`; the latter records the uncommitted candidate source.

The scalar change substantially reduces this memory cost but leaves roughly
2.88× the warp instructions, 4.29× integer-pipe instructions and 1.62× shared
wavefronts. These aggregates do not isolate the cost of queue polling,
descriptor addressing, extra lanes or scratch initialization.

The scalar-accumulator change passes 664 library tests (eight ignored), all
26 ordinary attention oracle tests, all 168 strict whole-graph oracle tests,
the six focused graph tests and strict all-target/all-feature Clippy. SDK
capture qualification also compares the complete forward state with the
unchanged arithmetic reference.

Experiments with directional payload fences, scalar parameter broadcasts,
unchanged-parameter upload elision and merged metadata broadcasts gave no
material improvement and were reverted. Per-node descriptor specialization
also gave no useful speedup and made training compilation exceed four minutes.
Claiming four tiles per queue operation regressed CI and SmolVLA and was
reverted. Saved patches and exploratory paired records remain under `target/`;
none of those experiments changes the published fence contract.

## Warp waiting and synchronization

The capture script also accepts repeated `--metric-group` arguments. Each group
must fit one hardware pass; groups run sequentially during one longer resident
replay. Every group retains its command, timestamps, raw CSV, 146 post-warmup
samples and individual counter means. The target must remain resident through
the last group and pass the same complete-state and arithmetic checks. These
groups are not simultaneous measurements.

On the `5fff909` implementation, 18 groups collect the raw active-warp rate and
17 issue states. The L1TEX component of long-scoreboard waiting and the non-RT
MIO component each fit one pass; their combined SDK metrics do not. Normalizing
each raw rate by the first group's `smsp__warps_active.avg.per_second` gives:

| Active warp-cycle state | Ordinary, % | Whole graph, 256 workers, % |
|---|---:|---:|
| Workgroup barrier | 3.623 | 50.447 |
| Long scoreboard, L1TEX | 42.303 | 21.740 |
| Fixed-latency wait | 12.913 | 9.749 |
| MIO throttle, non-RT | 17.380 | 4.707 |
| Short scoreboard | 4.032 | 4.460 |
| Selected to issue | 9.402 | 3.929 |
| No instruction | 0.588 | 0.950 |
| Memory barrier | 0.000 | 0.493 |

Clock and instruction-rate counters accompany every group. Their means vary by
at most 0.67% from the across-group mean, and the 17 states sum to 100.001% /
99.920%, supporting cross-window stability. Replay medians are 3.777 / 7.398 ms;
both modes have the same forward-state hash listed above. Target and monitor
exits succeed without SDK warnings. Receipts and the derived summary are in
`target/megakernel-results/whole-graph-stalls-sdk/`. The executable is from the
published implementation; source-status receipts disclose the modified capture
script and an unrelated untracked configuration directory.

These are **shares of active warp cycles, not shares of elapsed time**. NVIDIA
defines barrier waiting as a warp waiting for sibling warps to reach the
[workgroup barrier](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html).
Here that includes both handler barriers and inactive workers/extra subgroups
waiting for their leader or useful lanes. The data does not establish that
removing barriers would recover half the runtime. The low memory-barrier and
instruction-fetch shares argue against prioritizing another fence API change
or instruction-cache optimization. The smaller long-scoreboard percentage also
does not imply less absolute memory waiting: persistent execution has more
active warp cycles per step. A leader's atomic or fence latency can still delay
many waiting warps; a small aggregate memory-barrier share does not exclude
that critical-path cost.

A compact reproduction of the active, barrier and memory-barrier groups is:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python3 scripts/profile_smolvla_sdk.py --sdk /x/Tools/nsight-perf-sdk-2026.3 \
  --output target/new-stall-capture --phase forward --workers 256 --whole-graph \
  --metric-group smsp__warps_active.avg.per_second \
    gpc__cycles_elapsed.avg.per_second sm__inst_executed_realtime.sum.per_second \
  --metric-group smsp__warps_issue_stalled_barrier.avg.per_second \
    gpc__cycles_elapsed.avg.per_second sm__inst_executed_realtime.sum.per_second \
  --metric-group smsp__warps_issue_stalled_membar.avg.per_second \
    gpc__cycles_elapsed.avg.per_second sm__inst_executed_realtime.sum.per_second
```

Further exploratory changes moved parameters into a uniform buffer, removed the
native scratch-clear prefix, and resumed bitmap searches at the last successful
word. Each passed short paired state comparisons but gave no useful full-step
improvement and was reverted. The uniform-buffer SDK capture likewise showed no
replay improvement and increased estimated integer-pipe instructions. These
negative results narrow the remaining investigation without weakening the
memory contract or changing ordinary arithmetic selection.

An additional candidate guarded synchronization-free loops with the existing
lane-activity predicate. It preserved calls containing barriers and loops that
could return, then passed full Naga validation, 664 library tests, all 168 strict
whole-graph oracle tests and 250 completed-step paired comparisons. SDK captures
estimated 2.13× ordinary warp instructions and 1.81× global-load sectors, down
from 2.88× / 2.69×. Complete forward state remained bitwise identical. However,
the uninstrumented SmolVLA paired ratios were 1.798× forward (95% interval
1.778–1.845) and 1.509× training (1.472–1.618), overlapping the published results
below. CI ratios were 1.340× / 1.500× / 1.519×. The transformation was reverted
because it did not establish a useful timing improvement. Its patch, binary
hashes, raw samples and bootstrap summary are retained in
`target/megakernel-results/masked-loops-paired/`; separate counters are in
`target/megakernel-results/masked-loops-sdk/`.

This also bounds the conclusion from excess instructions: many can be removed
without recovering the elapsed-time gap. It motivated concurrent tile packing,
measured below, which puts spare subgroups to work on independent logical tiles
while preserving scratch isolation and the completion/fence contract. The
earlier rejected sequential four-tile batching did not provide this behavior.

## Scalar-accumulator completed-step measurement

Clean commit `36304a0` includes the scalar attention accumulators. The same
uninstrumented 50-pair protocol passes all **250 comparisons**, with exact
forward/CI state and maximum training difference `4.6566128730773926e-9`.
Every persistent case reports one dispatch and identical arithmetic selection.
All rows use 256 workers.

| Workload | Ordinary median, ms | Whole-graph median, ms | Median paired ratio | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| SmolVLA forward | 4.439 | 8.047 | 1.816 | 1.804–1.858 |
| SmolVLA forward/backward/SGD | 15.133 | 22.821 | 1.510 | 1.485–1.529 |
| CI inference | 0.0520 | 0.0691 | 1.324 | 1.316–1.332 |
| CI forward/backward | 0.0665 | 0.1002 | 1.496 | 1.485–1.513 |
| CI forward/backward/SGD | 0.0677 | 0.1032 | 1.505 | 1.485–1.524 |

These results improve SmolVLA over the initial implementation but remain below
ordinary performance. CI does not use the changed attention kernel; its gap
remains. Records and hashes are under
`target/megakernel-results/whole-graph-scalar-36304a0/summary.json`, with the
same bootstrap procedure in filename/case order. The isolated final run is
the only SmolVLA record used here. An invalid ICD-path attempt and two cancelled
overlapping runs are retained and explicitly excluded in that summary.

## Concurrent tile packing measurements

Commit `88d5e51` packs eligible logical tiles concurrently into the existing
worker workgroup. The matrix tiles use separate scratch ranges and preserve
their original subgroup coordinates and arithmetic. The grid axis is selected
from actual dispatch dimensions to reduce claims. Causal attention remains
unpacked because its query-dependent barrier loop fails the admission check.
The focused packing fixture covers two- and four-tile claims, partial rows,
repeated scratch reuse and worker pools from 1 to 4096. The regional selected
geometry/scratch test also passes after the shared subgroup-detection change.

All **250 completed-step comparisons** pass at 256 workers, with identical
plans and implementation keys, one physical dispatch and no fallback. Each row
uses 50 alternating pairs and at least ten warmup pairs / one second of warmup.
Forward and CI state are exact; the maximum training difference remains
`4.6566128730773926e-9`. Tracked source was clean. Receipts disclose the unrelated
untracked `configs/` directory, binary hashes, commands and start/end times;
SDK, CI and SmolVLA runs were sequential, with no profiling, builds or analysis
during the primary measurements.

| Workload | Ordinary median, ms | Whole-graph median, ms | Median paired ratio | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| SmolVLA forward | 4.421 | 6.483 | 1.468 | 1.461–1.514 |
| SmolVLA forward/backward/SGD | 15.149 | 21.620 | 1.426 | 1.410–1.553 |
| CI inference | 0.05054 | 0.06830 | 1.337 | 1.250–1.440 |
| CI forward/backward | 0.06647 | 0.10046 | 1.522 | 1.499–1.529 |
| CI forward/backward/SGD | 0.06810 | 0.10078 | 1.486 | 1.469–1.505 |

The forward median falls from 8.047 to 6.483 ms compared with the preceding
scalar-accumulator measurement, while the ordinary median is nearly unchanged.
The training interval is broader and overlaps the preceding result, so it does
not establish a training speedup. CI uses no packed matrix tiles and retains
its scheduling overhead. **Baseline parity remains unmet.** Raw records and the
10,000-resample bootstrap summary, using the same seed and filename/case order,
are in `target/megakernel-results/packed-tiles-88d5e51/`.

Separate SDK captures collect six sequential single-pass groups per mode, each
with 146 samples, successful exits and no warnings. Complete forward state
retains SHA-256 `4e5d66e6ce5e95142b848651d316b4df64855a9d85669499cabc3a81089be629`.
Ordinary / persistent replay medians are 3.775 / 6.013 ms. Estimated per-step
activity, calculated as counter rate times replay duration, is:

| Metric | Persistent / ordinary |
|---|---:|
| Warp instructions | 1.869× |
| Predicated-on integer FMA-pipe thread instructions | 2.537× |
| Global-load sectors | 1.115× |
| Shared-memory wavefronts | 1.185× |
| Tensor activity × duration | 1.016× |

The corresponding pre-packing estimates were 2.88× warp instructions, 4.29×
integer-pipe instructions, 2.69× global-load sectors and 1.62× shared wavefronts.
Packing substantially reduces redundant work while maintaining the tensor work
of the ordinary implementation. These remain aggregate activity estimates.

Barrier waiting is now 62.50% of persistent active warp cycles, versus 3.73%
ordinary; memory-barrier waiting is 0.49%. Clock/instruction-rate means vary by
at most 0.33% across their sampled groups. The increased barrier share alongside
lower runtime reinforces that it is not an elapsed-time fraction. The capture
does not separate scheduler waiting from handler barriers. Receipts, raw CSVs
and the derived summary are in `target/megakernel-results/packed-tiles-sdk/`.

## Reproduction and artifact identities

On this Linux host, run each build, test and benchmark in a separate user
cgroup. The parent session must remain outside it. For example:

```sh
systemd-run --user --scope --slice=ml.slice \
  -p MemoryHigh=16G -p MemoryMax=20G -p MemorySwapMax=1G \
  cargo build --release --features models --example bench_smolvla_train --example bench_ci_latency
```

The checked-in [launcher](../scripts/run_bounded_job.py) creates a unique scope,
verifies its membership and limits from inside the child, then executes the
command. For example:

```sh
python3 scripts/run_bounded_job.py release-build -- \
  cargo build --release --features models --example bench_smolvla_train --example bench_ci_latency
```

It requires a systemd user manager and cgroup v2. Verification fails closed if
membership or limits differ. The child inherits the current directory and
environment, and its exit code is propagated. Captures keep their target and
SDK monitor in that scope. GPU jobs remain serial: wait for the previous job's
exit before launching another. Primary measurements exclude builds and analysis.
Launcher checks verified scope membership, limits and propagation of exit code 7.
The earlier local launcher remains at `/x/Tools/meganeura-run-job.py`.

A diagnostic run exposed an unrelated host-memory failure before GPU execution:
the optimizer eagerly expanded a shared expression DAG for a debug-log argument,
even when that logging target was filtered out. The kernel killed the benchmark
at approximately 29 GiB resident memory. Commit `cbf5167` logs the root term and
DAG size instead. The same debug-enabled SmolVLA workload then completed all 20
paired comparisons with 3.54 GiB peak RSS; ordinary training construction took
88 ms. The failed run contributes no timing data. Logs are
`target/head-packing-oom.log`, `target/head-packing-bounded.log` (a bounded
reproduction stopped before its memory limit), and `target/head-packing-fixed.log`.

## Subsequent experiments

Worker counts 64, 128 and 512 each passed 20 short SmolVLA comparisons. None
clearly improved both phases over 256: 64 workers measured 5.94 ms forward but
33.89 ms training; 128 measured 6.21 / 22.80 ms and 512 measured 6.14 / 21.64 ms.
These short probes are not primary performance claims. Records are
`target/packed-workers-{64,128,512}.{json,log}`.

Retesting inactive-loop suppression after packing passed 100 comparisons with
the full 50-pair protocol. Forward measured 1.448× ordinary (95% interval
1.441–1.519), training 1.418× (1.392–1.453). Both overlap the published results,
so the transformation remains reverted. The patch, binary hash and summary are
under `target/megakernel-results/packed-masked-paired/`.

Another prototype separated workgroup coordinates into scalar arguments, so
barrier analysis could admit independent attention heads while still rejecting
query-axis packing with unequal loop bounds. Three-tile claims packed SmolVLA's
15 heads. Compiler checks and the focused selected-geometry GPU test passed;
100 full-step comparisons also passed with unchanged arithmetic selection.
However, shared storage grew from 29,184 to 41,216 bytes for forward execution.
Forward measured 6.534 ms / 1.475× ordinary (1.464–1.483), training 21.645 ms /
1.431× (1.416–1.610), giving no useful improvement. The prototype was reverted;
records and its patch are under `target/megakernel-results/head-packing-paired/`.
Sorting scratch slots by size did not reduce this allocation and was also
reverted; its short probe and patch are `target/head-ranked-smolvla.*` and
`target/head-packing-ranked.patch`.

Removing the final worker-loop barrier, relying on the next queue broadcast's
rendezvous, passed all seven focused graph tests but gave no clear CI or
SmolVLA improvement. A direct-successor experiment let the final producer reserve
the successor's first tile, fencing that reservation before publishing remaining
work. It also passed the seven focused tests, 150 CI comparisons and 20 short
SmolVLA comparisons without improving either workload. Both were reverted.
Patches and records are `target/joined-claim*` and
`target/successor-reservation*`. No rejected experiment changes the published
queue or memory contract.

Further resource diagnostics at `05ed893` report **113 / 111 registers per
thread**, **29,184 / 29,200 shared bytes**, and **113,280 / 273,664 executable
bytes** for forward / training combined shaders. These are separate instrumented
runs, not primary timing measurements. Ordinary forward pipelines use 16–120
registers and at most 16,768 shared bytes. The common footprint is a plausible
occupancy cost, but these statistics do not isolate its runtime contribution.
The driver's implausible local-memory statistic remains excluded from spill
claims. Pipeline statistics require `MEGANEURA_GPU_CAPTURE=1` as well as
`--diagnostics`; records are `target/resource-stats.{json,log}`.

Five follow-ups were reverted:

| Experiment | Finding |
|---|---|
| Struct-valued node parameters | Two codegen tests and both diagnostic comparisons pass; registers and shared storage remain unchanged. |
| Reload completion metadata after the handler | Both diagnostic comparisons pass; forward registers stay 113 and training falls only from 111 to 110. |
| Physical tensor-range dependencies | Seven focused tests and 20 short comparisons pass. Forward retains 161 edges; training falls only from 706 to 698. Persistent medians are 6.312 / 21.483 ms. |
| 128-lane cooperative forward attention in both paths | The selected-geometry test and 20 comparisons pass. The combined register/shared footprint is unchanged; persistent medians are 6.211 / 21.513 ms. |
| Leader-computed tile coordinates | Seven focused tests and 20 comparisons pass. Persistent medians are 6.177 / 21.581 ms. |

The three timing probes use ten pairs per phase and at least one second of
warmup. They do not establish an improvement over the published measurements.
Their forward comparisons are exact and maximum training error is 4.66e-9.
Patches and records use `target/parameter-struct*`, `metadata-lifetime*`,
`range-dependency*`, `wide-attention*` and `leader-coordinates*`; the summary and
patch hashes are in `target/megakernel-results/resource-followups/summary.json`.

A separate SDK capture of leader-computed coordinates collects two sequential
single-pass groups per mode, each with 146 samples, successful exits and no
warnings. Forward state retains the same hash as the published capture.
Ordinary / persistent replay medians are 3.774 / 6.044 ms. Activity-time estimates
are 1.889× ordinary warp instructions, 2.552× integer-pipe instructions, 1.132×
global-load sectors and 1.204× shared wavefronts, close to the published packing
results. Thus this experiment did not remove a substantial part of the measured
excess instruction work. Receipts are in
`target/megakernel-results/leader-coordinates-sdk/`. All follow-up jobs used
separate bounded cgroups. The published runtime and performance claims remain
unchanged; parity is still unmet.

## Worker clock diagnostics

A temporary Blade instrumentation patch inserts device-scope
`OpReadClockKHR` instructions at explicit marker calls, without changing Naga's
language or validation rules. Each transformed module passes `spirv-val` before
pipeline creation. The device clock and Vulkan memory-model device-scope
features are queried and enabled. These patches are diagnostic artifacts, not
part of the runtime dependency. Source and raw records are under
`target/clock-profile/`; the first capture is preserved in its `first/` directory.

In that capture, worker entry-to-exit envelopes are 6.421 ms forward and
21.035 ms training, within 0.04% of the corresponding dispatch timestamps.
Task counts agree with the logical grids and all recorded intervals are
contained in worker lifetimes. There are 168 long-lived workers; the other 88
enter near completion. Summed handler intervals occupy 27.47% / 45.39% of
summed worker lifetimes. These are worker-time fractions, **not elapsed-time
savings**: many workers wait while a narrow node has too few tiles to occupy
them. Handler intervals include the publication fence and workgroup rendezvous.
Instrumentation also raises register counts to 126 / 123, and clock reads do
not guarantee compiler code-motion barriers, so this is diagnostic attribution.

Scalar matrix handlers dominate the forward trace. Its 50 `MatMul` nodes have
3.671 ms of summed node spans and 32 `FusedMatMulAdd` nodes have 2.625 ms;
cooperative attention contributes 0.307 ms. Spans can overlap and must not be
added as elapsed time. Separate ordinary dispatch profiles retain bitwise exact
complete state and show materially shorter matrix intervals. For example,
`FusedMatMulAdd[50x720x2048]` is about 63 microseconds ordinary versus
115–118 microseconds between the first and last persistent worker timestamps.
The instrumentation differs, so this does not establish an exact kernel ratio.
Training also exposes roughly 95,000 small SGD tiles as a queue-batching target.
Ordinary profiles and comparisons are `target/clock-profile/ordinary.*` and
`compare-{forward,backward}.json`.

Three compiler-oriented experiments followed:

| Experiment | Finding |
|---|---|
| Subgroup-uniform parameter hints | Ten paired comparisons pass, but forward registers remain 126 and timing does not show a useful gain. No Naga uniformity change is retained. |
| Split read-only descriptor views and rely on existing tile fences for read visibility | All 20 uninstrumented comparisons pass. Forward improves to 5.228 ms, but training regresses to 27.519 ms. A separate diagnostic run reports 140 / 139 registers, up from 113 / 111. |
| Fence-based visibility with the original descriptor views | All 20 comparisons pass; forward / training medians are 6.317 / 21.649 ms, with large training outliers. No improvement is established. |

Read-only splitting therefore has a substantial resource tradeoff. None of
these experiments is retained, and no memory-semantics change is proposed from
the short correctness probes. Patches and receipts are
`target/clock-profile/{parameter-hint,readonly-clock,fence-visibility}*` and
`readonly-primary-meganeura.patch`; timing records are
`target/{parameter-hint-probe,readonly-primary-smolvla,fence-visibility-smolvla}.{json,log}`.
All jobs used separate bounded cgroups and local storage.

## Large SGD claim batches

The first four-tile SGD prototype batches every grid. All seven focused tests
and 250 paired comparisons pass, but small CI training regresses to 0.1108 ms
from the published 0.1008 ms. Restricting the transformation to large initial
grids removes that regression. Sixteen-tile claims offer little further benefit
in short SmolVLA probes, so the retained width is four. The selected SGD body,
element arithmetic and graph submission count remain unchanged.

At **`b23d422`**, all **250 final comparisons** pass with the same 50-pair,
256-worker protocol and isolated measurements. Forward and CI comparisons are
exact; maximum SmolVLA training error is 4.66e-9. Every selected case reports one
compute dispatch and no fallback, with identical plans and arithmetic selection.
All 665 library tests (eight ignored), 168 strict whole-graph oracle tests,
seven focused tests, formatting and strict all-target/all-feature Clippy pass.

| Workload | Ordinary median, ms | Persistent median, ms | Paired ratio | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| SmolVLA forward | 4.441 | 6.510 | 1.465 | 1.453–1.474 |
| SmolVLA forward/backward/SGD | 15.129 | 20.965 | 1.390 | 1.378–1.608 |
| CI inference | 0.05213 | 0.07008 | 1.343 | 1.338–1.351 |
| CI forward/backward | 0.06603 | 0.10120 | 1.533 | 1.519–1.538 |
| CI forward/backward/SGD | 0.06737 | 0.10095 | 1.498 | 1.483–1.519 |

A fresh, separate checkout of the prior runtime (`3628005`) passes another
100 SmolVLA comparisons under the same protocol. Its forward median is 6.497 ms
(1.475× ordinary, interval 1.463–1.507), and training is 21.552 ms
(1.424×, 1.400–1.453). Large-grid batching lowers the observed training median
by 2.7%; the paired-ratio intervals overlap, and these separate runs do not
establish a statistically isolated speedup. The small-grid penalty is avoided,
and the transformation reduces large SGD queues to roughly one claim per four
logical tiles. **Baseline parity remains unmet.**

The qualified executables, source/binary hashes, commands, cgroup identities,
successful exits, raw samples and summaries are under
`target/megakernel-results/sgd-large-paired/`; the contemporary comparison is in
`sgd-contemporary-baseline/`. Both use clean tracked source. Prototype records
are `sgd-batch4-paired/` and `target/sgd-batch{,16}-smolvla.{json,log}`. All
measurements exclude profiling, builds and concurrent analysis.

## Descriptor identity repair: current measurements

At **`ed81459`**, Naga `bc7b70b` and Blade `c09c89f` preserve one descriptor
reference per backing buffer across typed views and nonzero offsets. No
publication fence or selected arithmetic is removed. Isolated reruns against
the published pins pass **665 library tests** (eight ignored), **168 strict
whole-graph oracle tests**, **eight focused GPU tests**, and strict Clippy.
The [memory contract](megakernel_memory_contract.md) records compiler validation
and the eight workspace failures already present in the saved baseline.

All **250 primary comparisons** pass: 50 alternating AB/BA pairs for each case,
256 workers, at least ten warmup pairs and one second of warmup. Tracked source
is clean; receipts disclose unrelated untracked `configs/`. Every selected case
uses one compute dispatch and the same arithmetic, without fallback. Forward
and CI state are exact; maximum SmolVLA training error is **4.66e-9**. Identical
state is restored outside timing, and qualification readbacks are excluded.

| Workload | Ordinary median, ms | Persistent median, ms | Paired ratio | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| SmolVLA forward | 4.448 | 6.608 | 1.489 | 1.484–1.494 |
| SmolVLA forward/backward/SGD | 15.154 | 20.884 | 1.375 | 1.358–1.396 |
| CI inference | 0.05220 | 0.07068 | 1.353 | 1.345–1.374 |
| CI forward/backward | 0.06598 | 0.10280 | 1.557 | 1.547–1.565 |
| CI forward/backward/SGD | 0.06745 | 0.10130 | 1.510 | 1.489–1.523 |

Intervals bootstrap paired-ratio medians 10,000 times with seed 20261025 and
describe within-run uncertainty. The repair is retained for correctness;
these measurements do not show performance parity or an isolated speedup.

A separate NVIDIA Perf SDK forward capture collects six sequential groups per
mode, each with 146 post-warmup samples. All monitor/target processes exit zero,
without SDK warnings, and both modes produce identical state (SHA-256
`4e5d66e6ce5e95142b848651d316b4df64855a9d85669499cabc3a81089be629`). Diagnostic
replay medians are **3.772 ms ordinary / 6.103 ms persistent**. Cross-group clock
means stay within 0.25% and warp-instruction rates within 0.70% of their means.

| Estimated activity per forward replay | Persistent / ordinary |
|---|---:|
| Warp instructions | 1.924× |
| Integer-pipe thread instructions | 2.552× |
| Global-load sectors | 1.117× |
| Shared-memory wavefronts | 1.193× |
| Tensor activity | 1.017× |
| DRAM read activity | 1.000× |

Persistent execution also has about 39,854 local-load and 678,795 local-store
sectors per replay, versus zero ordinary. Workgroup-barrier waiting is **62.28%**
of active warp cycles, versus **3.75%** ordinary; memory-barrier waiting is
**0.47%** versus zero. These are active-warp shares, not elapsed-time fractions
or recoverable-runtime estimates. Counter rates times replay duration estimate
activity; the peak-normalized tensor and DRAM metrics support relative comparison.

A separate pipeline-statistics capture reports **112 registers** in both
combined shaders, with **29,200 / 29,232 bytes** of shared storage and
**114,560 / 272,000 bytes** of executable code for forward/training. The driver's
implausible approximately 64 GiB per-thread Local Memory Size field is excluded;
local traffic above comes from the SDK counters.

The measured gap still includes substantial extra integer/addressing and worker
coordination work, with waiting workers contributing to barrier stalls. These
counters do not isolate each handler's elapsed cost. The follow-ups below do
not improve both phases. Worker placement is now measured, while spill
attribution remains open. Retained changes must pass both phases' existing gates.

Primary binaries, commands, hashes, cgroup identities, receipts and raw samples
are in `target/megakernel-results/canonical-descriptors-paired/`; SDK records are
in `canonical-descriptors-sdk/`. All GPU jobs use separate bounded scopes and
the shared GPU lock. No builds, profiling or analysis overlap primary timing.
Library tests also execute GPU work: an earlier overlapping test attempt was
followed by isolated library, focused and oracle reruns before these measurements.

## Immutable allocations and occupancy follow-ups

An audit at `4469a03` includes every appended optimizer command and every slice
of a backing allocation. Forward has **147 immutable allocations out of 170**,
totaling **399,576,768 bytes**. Training has only **13 out of 205**, totaling
**187,056 bytes**, because optimizer updates make the weights mutable.

A temporary implementation separated immutable and mutable descriptor tables,
retaining one offset-zero descriptor per physical buffer and aliasing typed
views within each table. Distinct immutable-binding masks required 18 forward
handlers instead of 15, and 39 training handlers instead of 33. It preserved
the mutable publication protocol and allowed private reads only for storage
never written anywhere in the step.

A second version conservatively demoted all allocations sharing a handler
binding with mutable storage, propagating until no further demotions occurred.
That retained one handler per kernel and one descriptor per buffer. It left
137 immutable forward buffers (399,400,768 bytes) and three immutable training
buffers (11,056 bytes).

These are **exploratory** runs: 20 alternating pairs per phase, at least ten
warmup pairs and one second of warmup, with the same completed-step boundary
and full state comparison. They do not replace the 50-pair primary results.

| Experiment | Forward persistent ms / paired ratio | Training persistent ms / paired ratio | Diagnostic registers, forward / training |
|---|---:|---:|---:|
| Immutable handler variants | 5.941 / 1.334 | 25.163 / 1.625 | 143 / 140 |
| Immutable bindings shared by every invocation | 6.522 / 1.509 | 20.830 / 1.357 | 112 / 105 |
| Original shader, 84 workers | 5.564 / 1.410 | 24.633 / 1.663 | Not recaptured |
| Original shader, 256 workers, fresh control | 6.583 / 1.568 | 21.110 / 1.373 | Not recaptured |
| Uniform broadcast inside each handler | 6.512 / 1.488 | 20.800 / 1.388 | 123 / 118 |

All **200 comparisons** in these five completed trials pass; forward is exact
and maximum training error is 4.66e-9. Each persistent step uses one dispatch
and unchanged selected arithmetic. The variant trial's forward gain coincides
with higher register use and a training regression. The 84-worker control
has the same direction of tradeoff, making occupancy a plausible contributor;
these runs do not establish SM placement or isolate a read-only-load benefit.
Moving the uniform broadcast inside each selected handler increases registers
and shows no useful gain. None of these implementation changes is retained.

Two incomplete attempts are also recorded. The initial variant generator
renamed a storage struct still referenced by its typed table; shader validation
rejected it, and shared type declarations were preserved for the completed
trial. A separate 64 KiB shared-memory reservation intended to constrain
residency was rejected by the runtime's 48 KiB workgroup limit before dispatch.
No admission or validation check was disabled.

Patches, preserved binaries, samples, separate pipeline captures, cgroup logs
and a hashed summary are in `target/megakernel-results/immutable-variants/`.
GPU jobs were serialized in bounded scopes and their exits confirmed before
the next GPU job. Builds and analysis did not overlap measurements. The
published runtime and compiler pins remain the qualified canonical-descriptor
implementation above; the rejected probes did not proceed to its full
library/oracle/CI acceptance gate.

## Worker placement and compiler follow-ups

Temporary instrumentation based on
[`VK_NV_shader_sm_builtins`](https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_shader_sm_builtins.html)
records the leader's SM ID at entry and exit alongside per-node worker counts.
Both instrumented SPIR-V modules pass Vulkan 1.3 validation before pipeline
creation. In the 256-worker capture, **168 workers perform tasks on 84 SMs,
exactly two per SM**. Every leader's entry and exit IDs match. The first
46-tile matrix node uses only 23 SMs; sixteen `FusedMatMulAdd[50x720x2048]`
nodes use a median of 33 SMs. With 84 workers, every active worker has its own
SM. These are observations on this device/driver, not a portable placement rule.

A probe elects one worker per SM for nodes with at most 84 claims, allowing all
workers to handle larger nodes. It raises the narrow fused-matrix family's
median SM count from 33 to 46, but does not improve the matching instrumented
control. Both use 116 forward / 115 training registers. With 20 alternating
pairs per phase, control persistent medians are **7.118 / 22.511 ms** and owner
medians are **7.214 / 23.014 ms**. The SM instrumentation and capture settings
affect compilation; these timings do not replace the uninstrumented primary
results. The owner policy is reverted. Clustering alone does not establish the
cause of the elapsed-time gap.

The driver's pipeline-executable query returns **zero internal representations**
for both combined shaders, so this route supplies no ISA or spill locations.
Further compiler probes retain the selected arithmetic and memory protocol:

- An SPIR-V `Inline` hint leaves 112 registers and executable sizes unchanged.
- Actual inlining, after dead-branch elimination and return merging, produces
  one function with no calls in each module. Validation and state comparisons
  pass, but registers remain 112 and code grows to 115,968 / 274,688 bytes.
  Normal 20-pair medians are 6.571 / 21.069 ms, with ratios 1.492 / 1.411.
- A descriptor-uniformity probe removes 102 / 200 `NonUniform` decorations,
  restricted to accesses rooted at the canonical payload binding. The builder
  supplies every descriptor index through `workgroupUniformLoad`, and Vulkan
  permits [dynamically uniform descriptor indexing within a workgroup](https://docs.vulkan.org/guide/latest/extensions/VK_EXT_descriptor_indexing.html).
  The corresponding core feature/capability is enabled; alias decorations and
  memory-access semantics remain intact. Validation and state comparisons pass,
  but registers and executable sizes are unchanged. Normal 20-pair medians are
  6.564 / 20.948 ms, with ratios 1.490 / 1.413.

Neither compiler probe shows a useful gain, and neither is retained. The initial
inliner warned that early returns prevented complete inlining; return merging
then rejected unreachable blocks. Removing dead branches first resolved that
precondition. Failed/preliminary outputs are preserved separately.

A 50-pair sweep with the preserved canonical CI executable also fails to close
the gap, despite these graphs having only 1–16 logical tiles per node:

| Workers | Inference persistent ms / ratio | Backward persistent ms / ratio | SGD persistent ms / ratio |
|---:|---:|---:|---:|
| 1 | 0.09433 / 1.810 | 0.27334 / 4.197 | 0.32322 / 4.808 |
| 4 | 0.07043 / 1.352 | 0.11261 / 1.725 | 0.13058 / 1.942 |
| 16 | 0.07112 / 1.339 | 0.09831 / 1.496 | 0.09868 / 1.461 |
| 64 | 0.07419 / 1.292 | 0.09658 / 1.482 | 0.09623 / 1.444 |

All **760 comparisons** across the ownership/control, compiler and CI pool
trials pass. CI and forward state are exact; maximum training error is 4.66e-9.
Every persistent step has one compute dispatch and matched selected arithmetic.
All runs use separate bounded cgroups, serialize GPU work and keep builds and
analysis outside measurements. No runtime change from these trials is retained.

Finally, a separate timestamp capture disables worker histograms while retaining
GPU timing, so the reset image has its normal size. Twenty pairs per workload
pass. Median pass intervals for `reset whole-graph queue` are **2.432 µs** for
each CI case, **3.920 µs** for SmolVLA forward and **5.136 µs** for training.
These include the transfer pass boundary, not just DMA time. They are too small
to explain the gap, and no more complex reset protocol is added. The following
probes compare intra-workgroup scheduling and cross-workgroup coordination;
larger-model performance and all shared acceptance gates remain in scope.

Raw records are in `target/{sm-placement,shader-isa,inline-probe,spirv-opt-probe,
uniform-descriptor-probe,ci-worker-sweep,queue-reset-timing}/`. The combined
manifest, hashes and placement checks are in
`target/megakernel-results/placement-compiler-followups/summary.json`.

## Coordination follow-ups

A temporary specialization for an explicitly selected **one worker** walks
nodes in recording order, caches each node's parameters and executes its tiles
with workgroup storage/shared barriers. Every dependency points from an earlier
recorded node to a later one, so this order respects RAW/WAR/WAW dependencies.
All payload accesses are confined to one physical workgroup; the existing
canonical descriptors remain intact. Selected handler bodies, tile packing and
arithmetic are unchanged. Actual tile completion still updates diagnostics.

| One-worker workload | Original persistent ms | Specialized persistent ms | Specialized / ordinary paired ratio |
|---|---:|---:|---:|
| CI inference | 0.09433 | 0.06661 | 1.300 |
| CI forward/backward | 0.27334 | 0.15581 | 2.382 |
| CI forward/backward/SGD | 0.32322 | 0.17628 | 2.635 |
| SmolVLA forward | 263.379 | 234.231 | 60.787 |
| SmolVLA forward/backward/SGD | 1561.585 | 983.762 | 81.875 |

CI uses 50 pairs per case and compares with the earlier one-worker sweep.
SmolVLA uses a fresh original control followed by the specialization, each with
20 pairs per phase. All runs have at least ten warmups and one second of warmup.
These are separate runs, not directly interleaved original/specialized pairs;
each reported ratio compares its persistent candidate with its own interleaved
ordinary control. Ordinary forward latency also moves between the model runs.
Training improves substantially within the one-worker configuration, but
remains far slower than the qualified 256-worker path. Several coordination
operations change together, so this does not assign a precise cost to fences
alone. The specialization is reverted.

Two 256-worker probes keep node parameters while claiming further tiles from
the same node. They synchronize scratch between claims and defer payload
publication and completion accounting until the worker finishes that node's
claims. One applies this to all nodes; the other only repeats when the node has
more tiles than the pool can claim at once. Both retain cross-workgroup fences
at publication/acquisition and the original selected arithmetic.

| Probe | CI inference / backward / SGD paired ratios | SmolVLA forward persistent ms / ratio | SmolVLA training persistent ms / ratio |
|---|---|---:|---:|
| Reuse across all nodes | 1.377 / 1.575 / 1.638 | 6.281 / 1.586 | 21.042 / 1.397 |
| Reuse only on larger grids | 1.434 / 1.532 / 1.525 | 6.527 / 1.515 | 21.167 / 1.391 |

Each probe runs 20 pairs for all five cases. Neither demonstrates a useful
full-step gain, and both are reverted. Across these probes and the one-worker
controls, **430 normal comparisons** pass, plus three one-worker diagnostic
comparisons checking task counts. CI and model forward are exact; maximum
training error is 4.66e-9. Every candidate uses one dispatch and the same
selected implementation keys as its ordinary control. The initial reuse
attempt was rejected before execution because WGSL does not permit taking a
vector component's address; the completed probe broadcasts the whole vector.
An initial build omitted the model feature and was rerun with it enabled.

All jobs run in separate verified bounded cgroups, GPU jobs are serialized,
and builds/analysis do not overlap measurements. Patches, binaries, raw samples,
logs and hashes are local under `target/{single-worker-probe,node-cache-probe}/`
and `target/megakernel-results/coordination-followups/summary.json`. The runtime
and compiler pins are restored and release examples rebuilt. These diagnostic
experiments do not change the primary qualification or establish parity. The
following investigation tests ready-bitmap polling and contention, with local
publication proofs and the same completed-step comparisons.

## Ready-queue and polling results

An append-only queue reserves one slot per graph node. It preserves the existing
payload fences, marks a slot consumed after its final tile claim, and advances
a monotonic search head. Consumers scan past unpublished reservations so they
can execute other ready nodes. Slots are not recycled within a dispatch.
Completion checks retain exact node/tile counts and additionally require a
permutation of all nodes in consumed slots. The proof sketch is preserved with
the prototype; it does not replace full acceptance qualification.

Another probe keeps the bitmap and lets idle leaders wait for a change in the
remaining-node count before rescanning. A new release fence after publishing
successors and before decrementing that count pairs with an acquire before
the next scan. Without that ordering, a worker could cache the new count while
missing published bits. Sibling lanes wait at the uniform broadcast. Successful
claims can continue consuming already-ready tiles within the same completion
epoch. No new Naga interface is involved.

Two additional probes raise the physical worker width to 512 lanes, or make
the active-lane mask constant only when the original logical tile fills the
physical worker. Each preserves the selected arithmetic and reports its actual
worker geometry. Every probe runs 20 pairs in all five cases, with ten warmups
and at least one second of warmup.

| Probe | CI inference / backward / SGD paired ratios | SmolVLA forward ms / paired ratio | SmolVLA training ms / paired ratio |
|---|---|---:|---:|
| Ready-node queue | 1.385 / 1.564 / 1.569 | 6.298 / 1.631 | 21.364 / 1.411 |
| 512-lane workers | 1.486 / 1.725 / 1.810 | 8.598 / 1.951 | 33.801 / 2.232 |
| Constant full-tile active mask | 1.335 / 1.563 / 1.510 | 6.581 / 1.570 | 20.812 / 1.376 |
| Completion-gated leader polling | 1.349 / 1.631 / 1.692 | 6.613 / 1.529 | 21.064 / 1.384 |

All **400 comparisons** pass with one dispatch and matching implementation
keys. CI and forward are exact; maximum training error is 4.66e-9. None shows
a useful full-step gain, so all four runtime changes are reverted. Ordinary
control latency varies across runs; these exploratory ratios are not substitutes
for the primary 50-pair results.

Fresh Perf SDK captures compare graph-only forward replay, collecting six
sequential groups for the queue and three for completion-gated polling. Every
group has 146 post-warmup samples, successful target/monitor exits and no SDK
warnings. Device identity and complete state agree between modes. Rate times
replay duration gives the following activity estimates relative to ordinary:

| Implementation | Persistent replay ms | Warp instructions | Integer instructions | Global-load sectors | Shared wavefronts | Barrier share of active warp cycles |
|---|---:|---:|---:|---:|---:|---:|
| Qualified bitmap, earlier capture | 6.103 | 1.924× | 2.552× | 1.117× | 1.193× | 62.28% |
| Ready-node queue | 6.128 | 2.153× | 2.528× | 1.158× | 1.358× | 54.58% |
| Completion-gated leader polling | 6.151 | 1.603× | 2.544× | 1.140× | 1.097× | 65.60% |

Ordinary replay medians are 3.772, 3.765 and 3.774 ms respectively. Reducing
polling removes substantial warp instructions and shared traffic without
reducing replay latency; the measured integer activity remains almost unchanged.
The barrier percentages are not elapsed-time fractions. These results shift the
next experiment toward matrix addressing and the resource cost of combining
handlers, rather than claiming that idle instruction removal recovers the gap.

One initial queue capture selected RADV for the benchmark while NvPerf monitored
the RTX 5080. Both processes completed and numerical state matched, but NVIDIA
shader counters were zero. That capture is explicitly **invalid and excluded**.
The benchmark now emits adapter metadata before resident capture and includes
it in the JSON result. The helper requires the NVIDIA driver, one matching
NvPerf device, and nonzero shader activity when those metrics are collected.
Receipts include adapter identity, `counter_capture_valid` and an error on
failure. Matching real SDK logs pass; missing, non-NVIDIA, mismatched and
ambiguous adapter fixtures are rejected. Corrected captures pass these checks.
The earlier primary capture's NVIDIA ICD, target log and monitor identity were
also audited successfully; its measurements remain valid.

The capture guard is retained. Release examples rebuild, strict benchmark
Clippy and formatting pass. The runtime and dependency pins match the qualified
implementation. Raw samples, binaries, proofs, patches and SDK records are in
`target/{ready-queue-probe,wide-worker-probe,active-lane-probe,completion-poll-probe}/`;
the combined manifest is `target/megakernel-results/queue-followups/summary.json`.
All workloads used verified bounded cgroups, serialized GPU jobs and local
storage. Builds and analysis did not overlap timing. Parity remains unmet.

## Matrix resource isolation and claim priority

A temporary fixture repeats the selected `FusedMatMulAdd[50x720x2048]`
SmallTile operation 32 times, with its original `[23, 2, 1]` logical grid.
Every operation writes the same output, creating a strict WAW chain with
1,472 logical tasks. Both modes use identical operands, model-buffer sizes and
selected arithmetic; each comparison checks the full 36,000-element output.
The operands remain hot across repeated operations. This fixture diagnoses
matrix execution and does not replace the complete-model acceptance gate.

Three persistent variants compile only the active handler, or also compile
the original SmolVLA forward/training handlers without scheduling additional
nodes. The extra handlers affect code, shared scratch and parameter capacity.
Separate pipeline captures report the following resources:

| Shader | Registers | Shared bytes |
|---|---:|---:|
| Ordinary matrix | 48 | 8,448 |
| Isolated persistent matrix | 48 | 9,584 |
| With unused forward handlers | 112 | 29,200 |
| With unused training handlers | 105 | 29,232 |

Twenty paired completed-step samples per case give these persistent medians.
Ordinary medians range from 1.673 to 1.729 ms across the same runs.

| Workers | Isolated ms | With forward handlers ms | With training handlers ms |
|---|---:|---:|---:|
| 32 | 4.125 | 3.725 | 3.712 |
| 64 | 2.109 | 1.942 | 1.944 |
| 84 | 2.123 | 1.934 | 1.936 |
| 128 | 2.772 | 2.946 | 2.951 |
| 256 | 4.179 | 2.948 | 2.931 |

Adding unused handlers increases resource use yet improves the 256-worker
case. The best sampled pools are 64–84 workers; padded variants still take
about 1.15 times ordinary latency. Thus register count alone does not explain
the slowdown. The results are consistent with worker residency and contention
affecting this narrow workload; they do not isolate either as the sole cause.
The driver's implausible Local Memory Size statistic remains excluded.

Reading existing tile counters after separate diagnostic steps requires no
new shader instrumentation. At 256 workers, unsuccessful claims total 6,701
for the isolated shader and 3,885/3,884 for the padded shaders. At 64 workers,
all three total 576. Each count is one final-step diagnostic sample.

A temporary priority window obtains each worker's rank from an atomic start
counter. During the first four unsuccessful scans of a completion epoch,
workers whose rank exceeds a node's claim count skip that node. All workers
become eligible after those bounded scans; wide nodes remain available
throughout. The hint does not assume workgroup launch order or simultaneous
residency. Existing publication fences, failed-claim helping and ready-bit
clearing remain intact, and completion verifies every requested worker started.

The window reduces the 256-worker diagnostic failed-claim totals to
3,307/1,622/1,653. Normal isolated-matrix latency falls to 3.657 ms, but the
padded cases remain 2.954/2.935 ms. Registers are 48/116/108 respectively,
with unchanged shared storage. Full-step comparisons against fresh controls
show no useful overall gain:

| Case | Existing scheduler ms | Priority window ms | Existing / priority paired ratios |
|---|---:|---:|---:|
| CI inference | 0.07071 | 0.07469 | 1.350 / 1.344 |
| CI forward/backward | 0.10293 | 0.10341 | 1.545 / 1.597 |
| CI forward/backward/SGD | 0.10246 | 0.10989 | 1.507 / 1.574 |
| SmolVLA forward | 6.408 | 6.281 | 1.513 / 1.621 |
| SmolVLA forward/backward/SGD | 20.953 | 21.644 | 1.391 / 1.432 |

Each ratio compares persistent execution with its own paired ordinary control;
ordinary forward latency varies between runs. Reducing failed atomic claims
alone does not recover the performance gap. The next controlled comparison
holds the worker pool fixed while isolating descriptor/address calculations.

All **560 normal comparisons and twelve diagnostics** pass with one persistent
dispatch, identical selected arithmetic and no fallback. Matrix, CI and model
forward outputs are exact; maximum model-training error is 4.66e-9. The
temporary handler-padding API, claim dump and priority scheduler are reverted.
Runtime source matches `ed81459`, and the ordinary release benchmark examples
are rebuilt. No new Naga or Blade change is retained. All jobs use verified
bounded cgroups, serialized GPU work and local `/x` storage; builds and
analysis do not overlap measurements. Raw samples, source patches, binaries
and the priority proof remain in `target/{matrix-handler-probe,claim-priority-probe}/`.
The checked manifest and hashes are in
`target/megakernel-results/matrix-followups/summary.json`. Parity remains unmet.

## Matrix constants and inner-loop expansion

Further controls keep the same 32-node matrix chain, selected arithmetic,
operands and complete-output checks. A descriptor probe substitutes a constant
descriptor index and element offset only when every node using that handler
agrees. A dimensions probe substitutes the invariant integer uniform words while
leaving descriptor lookup dynamic. A combined probe also substitutes invariant
grid coordinates. Constants replace helper-argument uses directly, avoiding a
dependence on driver propagation through calls. Uniform-specialized diagnostics
assert that reused recordings do not change the embedded plain values.
Generated shader dumps verify every intended substitution.

Twenty normal pairs per case give these completed-step persistent medians at
64 workers. Ordinary medians range from 1.669 to 1.703 ms in these runs.

| Probe | Isolated ms | With unused forward handlers ms | With unused training handlers ms |
|---|---:|---:|---:|
| Fresh control | 2.104 | 1.936 | 1.940 |
| Constant descriptors and offsets | 2.049 | 1.937 | 1.943 |
| Constant dimensions | 1.843 | 1.929 | 1.929 |
| Constants plus grid coordinates | 1.998 | 1.928 | 1.917 |
| Explicit inner-loop expansion | 2.162 | 1.939 | 1.946 |

The isolated dimensions result improves, but the combined shaders retain their
gap. At 256 workers, the combined variants stay between 2.924 and 2.955 ms;
the isolated variants range from 4.084 to 4.342 ms. A fresh 46-worker control,
matching each node's logical tile count, takes 2.107/1.938/1.946 ms and does not
improve on the 64-worker control.

Four separate SDK comparisons collect three sequential single-pass groups per
mode at 64 workers. Every group has 146 samples, successful target/monitor exits,
matching NVIDIA adapter identity and no SDK warning. All eight replays preserve
exact output and matching arithmetic (state SHA-256
`cbf2e7ebdfb60979fce668c7879a168a1923982a6d37441a12c5399d0a1752fe`).
Repeated instruction rates deviate by at most 0.92% from their group mean;
clock rates deviate by at most 0.28%. Rate times replay duration gives:

| Capture | Ordinary / persistent replay ms | Warp activity ratio | Integer activity ratio | Global-load ratio | Shared-wavefront ratio | Persistent barrier share |
|---|---:|---:|---:|---:|---:|---:|
| Isolated control | 1.728 / 2.136 | 1.312 | 1.863 | 1.050 | 1.028 | 27.49% |
| Forward-handler control | 1.704 / 1.969 | 1.202 | 3.380 | 1.054 | 1.029 | 27.02% |
| Isolated, constant dimensions | 1.710 / 1.873 | 1.290 | 1.252 | 1.059 | 1.033 | 27.71% |
| Forward handlers, all constants | 1.710 / 1.954 | 1.148 | 2.856 | 1.052 | 1.026 | 27.22% |

The integer column uses the SDK's FMA-heavy integer-thread metric; it does not
enumerate every integer operation. Ratios use each capture's ordinary control.
These are activity estimates, not simultaneous exact instruction counts.
Ordinary barrier shares are about 2.6%; all barrier percentages describe active
warp cycles, not elapsed-time fractions.

Constant dimensions reduce the isolated shader's estimated integer activity
by about one-third and shorten replay. In the combined shader, fixing all three
sources of dynamic addresses removes about 16% of that activity but improves
replay by less than 1%. Thus descriptor indirection and these address calculations
alone do not recover the combined shader's gap. Compiling unused handlers still
changes instruction behavior even when the executed matrix work is fixed.

The inner-loop probe separately expands the selected scalar matrix template's
32 arithmetic iterations using the existing unroll generator. It preserves
every accumulator's operation order, all stage loads, scratch, stores and
barriers. Dumps contain 128 explicit accumulator updates and no `kk` loop in
the active handler. The isolated executable remains 17,408 bytes; forward and
training combinations grow to 121,216 and 303,104 bytes without a timing gain.
All probes retain 48/112/105 registers for isolated/forward/training variants
and the same shared-memory capacities. No missing-unroll diagnosis is established.

All **660 normal comparisons and twelve diagnostics** pass with exact output,
one persistent dispatch, identical selected arithmetic and no fallback. These
are matrix diagnostics; no whole-model speedup is claimed. All runtime and
temporary benchmark API changes are reverted, production release examples are
rebuilt and formatting passes. Runtime source and dependency pins remain at the
qualified implementation. Every job used a verified bounded cgroup, GPU jobs
were serialized, and builds/analysis stayed outside measurement intervals.
Source patches, proofs, binaries, raw samples and SDK receipts remain in
`target/{matrix-address-probe,matrix-unroll-probe}/`; their checked manifest is
`target/megakernel-results/matrix-address/summary.json`.

The following experiment varies serial versus independent-node dependencies
while holding shader code and resource layouts fixed. Full-model baseline
parity remains unmet.

## Independent nodes and terminal publication

A new matrix fixture gives each of the 32 selected matrix nodes its own output
allocation. The nodes share the same three input buffers and have no native
data hazards; lifetime aliasing is disabled. Every node retains the original
46 logical tiles, parameters and arithmetic. Every paired sample reads all
32 complete outputs outside timing, verifies that they agree exactly and are
nonzero, and compares them with ordinary execution.

The parallel mode uses the empty dependency graph. A serial control adds an
artificial edge from each node to the next in the persistent path only. Both
modes have identical physical pipeline keys, descriptor layouts, buffer sizes,
parameter words, metadata sizes and queue sizes. Separate sessions own distinct
allocations. The edge buffers both have 32 words; unused entries are unreachable.
Diagnostic receipts verify that only dependency metadata and initial readiness
differ. Neither the scheduler shader nor its publication protocol changes.

Twenty completed-step pairs per case give these medians:

| Workers | Compiled handlers | Ordinary independent ms, parallel / chain runs | Persistent parallel ms | Persistent artificial chain ms |
|---|---|---:|---:|---:|
| 64 | Matrix only | 0.733 / 0.710 | 1.565 | 2.135 |
| 64 | With unused forward handlers | 0.708 / 0.707 | 1.405 | 1.963 |
| 64 | With unused training handlers | 0.708 / 0.739 | 1.410 | 1.990 |
| 256 | Matrix only | 0.733 / 0.728 | 0.788 | 4.144 |
| 256 | With unused forward handlers | 0.709 / 0.726 | 0.798 | 2.970 |
| 256 | With unused training handlers | 0.708 / 0.732 | 0.799 | 2.972 |

Within each handler variant, driver executable statistics also match between
the dependency modes: 48/112/105 registers and 9,584/29,200/29,232 shared bytes
for isolated/forward/training variants. The implausible driver Local Memory Size
statistic remains excluded. The independent combined shader comes much closer
to ordinary execution despite retaining its resource footprint and dynamic
addresses. Adding a chain exposes a large scheduling cost in this narrow
workload. This does not establish how much overlap a real model permits, nor
does it prove that finer dependencies will recover the full-model gap.

The ordinary control executes the same independent plan in both modes. Thus
the artificial-chain ratios measure an imposed delay, not ordinary execution
of a naturally dependent graph. These distinct-output results also are not
directly comparable with earlier fixtures that repeatedly wrote one output.

A separate prototype removes publication fences and the first post-body
workgroup rendezvous only for nodes with no successors. All input acquisition,
nonterminal publication and the final workgroup barrier remain. Conservative
RAW/WAR/WAW reachability means terminal payloads have no later conflicting
access in the dispatch. Their external visibility still depends on real dispatch
completion **and existing API memory dependencies**, not the scheduler counter
or host fence alone. The proof and source patch are preserved with the probe.

Twenty paired comparisons per full-step case, followed by fresh controls, give:

| Case | Existing scheduler ms | Terminal elision ms | Existing / elision paired ratios |
|---|---:|---:|---:|
| CI inference | 0.07007 | 0.07082 | 1.359 / 1.358 |
| CI forward/backward | 0.10347 | 0.10166 | 1.547 / 1.544 |
| CI forward/backward/SGD | 0.10132 | 0.10162 | 1.502 / 1.510 |
| SmolVLA forward | 6.549 | 6.557 | 1.570 / 1.519 |
| SmolVLA forward/backward/SGD | 21.284 | 20.888 | 1.406 / 1.383 |

Ratios compare each persistent sample with its own ordinary control. Forward
latency is essentially unchanged. The small training difference does not
establish a benefit given substantial variation, including long outliers in
both modes. The terminal candidate is reverted without claiming full memory
protocol qualification or a speedup.

All **440 normal comparisons and six diagnostics** pass with one persistent
dispatch, identical selected arithmetic and no fallback. Matrix, CI and model
forward outputs are exact; maximum model-training error is 4.66e-9. All runtime
and temporary benchmark APIs are reverted, production release examples rebuilt,
and formatting checked. Runtime source still matches `ed81459`; Naga and Blade
pins are unchanged. No broad GPU suite is repeated for these discarded probes.

Every build, test and analysis job uses a verified separate bounded cgroup.
GPU jobs are serialized, builds/analysis stay outside timing intervals, and all
artifacts use local `/x` storage. The recovered terminal-probe logs confirm the
20 GiB memory caps and GPU lock; neither job remained running after recovery.
Raw results, patches, proofs and binaries are in
`target/{matrix-dependency-probe,terminal-publication-probe}/`; the checked
manifest and artifact hashes are in
`target/megakernel-results/matrix-dependencies/summary.json`.

The following controls measure intermediate dependency widths before attempting
finer dependencies in a real graph. Such a change needs explicit proofs for
access ranges, lifetime reuse, original tile coordinates and completion counts.

## Dependency width and handler isolation

The distinct-output fixture adds an edge `n-W -> n` for each node `n >= W`.
Widths 1/2/4/8/16 create that many independent chains; width 32 leaves all
nodes independent. The ordinary plan remains independent at every width.
All cases retain the same shader keys, resource layouts, parameter words and
buffer sizes within each compiled-handler variant. Eighteen metadata receipts
check the exact edges, ready bits, predecessor counters and unchanged fields.

Twenty normal pairs per case give these persistent medians at 256 workers:

| Ready chains | Matrix only ms | With unused forward handlers ms | With unused training handlers ms |
|---|---:|---:|---:|
| 1 | 4.141 | 2.972 | 2.979 |
| 2 | 2.168 | 1.518 | 1.534 |
| 4 | 1.230 | 0.983 | 1.019 |
| 8 | 0.911 | 0.801 | 0.840 |
| 16 | 0.767 | 0.802 | 0.829 |
| 32 | 0.796 | 0.801 | 0.798 |

Ordinary independent controls range from 0.708 to 0.760 ms. At 64 workers,
the combined variants plateau near 1.4 ms even with more independent chains.
The 256-worker combined shader needs about eight chains to approach its
independent-node timing. This is an imposed scheduling constraint, not a model
speedup. Splitting a real matrix by rows may enable earlier consumers without
increasing the total ready tile count enough to reproduce this benefit.

A separate scheduler probe replaces fetch-add claims with a load and retrying
compare-exchange, advancing the tile counter only while work remains. It retains
last-claim/helper ready-bit clearing, partial-batch clamping and every publication
fence. The forward-handler matrix chain slows from 2.972 to 3.106 ms. Full model
forward/training medians become 7.841/24.871 ms, substantially worse than the
qualified scheduler. All 120 matrix and 100 full-step pairs pass, but the probe
is reverted. Final counter overshoot is not a count of failed CAS attempts.

Eighteen further diagnostics compile each unique forward implementation family
into the independent matrix shader without scheduling its work:

| Compiled alongside the matrix | Registers | Shared bytes |
|---|---:|---:|
| Nothing | 48 | 9,584 |
| Horizontal scalar matmul | 128 | 16,880 |
| Cooperative matmul with prologue | 108 | 16,768 |
| Cooperative attention | 80 | 12,688 |
| GEMV | 80 | 15,728 |
| Full forward handler set | 112 | 29,200 |

The other individual forward families retain 48 registers. Composition remains
nonadditive: the horizontal-only combination has more registers than the full
shader. Instrumented timings from these diagnostics are not speedup evidence.
Inspecting that horizontal family reveals the geometry bug fixed below.

All **940 normal comparisons and 54 diagnostics** across the width, claim and
family probes pass with unchanged selected arithmetic, one persistent dispatch
and no fallback. The scheduler and temporary benchmark APIs are reverted.
One resource-only capture omitted metadata dumps; a separate complete capture
provides all 18 audited width receipts. Local proofs, patches, binaries and raw
results are indexed in `target/megakernel-results/dependency-width/summary.json`.

## Correct horizontal matmul geometry

Horizontal fusion admits `Kernel::SmallTile` and retains its 32×32 output-tile
grid. Its scalar runtime key previously omitted tile geometry, and the generator
always emitted the default 64×64 body. The small grid therefore overscheduled
the larger body and included its larger register tile in the combined shader.

Commit `12f7c5a` carries scalar geometry in the horizontal implementation key
and passes it, with the session's existing staging knobs, to the common
generator. Ordinary and persistent execution now both honor the selected
32×32 or 64×64 tile. The plan's logical grids, bindings and output buffers remain
the same. Queue scheduling, publication, descriptors and worker counts do not
change. In the default model, K-stage depth and column layout already matched
the previous defaults; the corrected body visits K in the same order.

The expanded GPU regression checks all three packed outputs against a CPU
reference with non-multiple M/N/K dimensions and repeated input updates. Both
geometries remain alive on one context to exercise separate cache entries, in
ordinary and whole-graph modes. The fixture asserts actual selection; an initial
too-small large-tile control was automatically reselected and is preserved as a
failed fixture attempt. Module tests check generated tile-column spacing as
well as validation, since overscheduling can still produce correct output.

Qualification passes **665 library tests (eight ignored), 168 strict whole-graph
oracle tests and nine focused GPU tests**: two horizontal regressions and seven
whole-graph geometry, scratch, replay, lifetime and optimizer tests. Formatting
and strict all-feature/all-target Clippy pass. Naga and Blade pins are unchanged;
their previously recorded checks are not rerun for this shared generator fix.

Fresh production binaries from clean tracked source `12f7c5a` run 50 alternating
AB/BA pairs per workload, with ten warmups and at least one second of warmup.
The existing completed-step boundaries, state restoration and full-state checks
remain unchanged. Every case reports one persistent dispatch, matching corrected
arithmetic and no fallback. CI and forward state are exact; maximum training
error is 4.66e-9.

| Workload, 256 workers | Ordinary ms | Persistent ms | Paired ratio | 95% interval |
|---|---:|---:|---:|---:|
| CI inference | 0.05527 | 0.07423 | 1.358 | 1.230–1.430 |
| CI forward/backward | 0.06613 | 0.10271 | 1.551 | 1.541–1.569 |
| CI forward/backward/SGD | 0.06743 | 0.10201 | 1.522 | 1.510–1.539 |
| SmolVLA forward | 4.235 | 6.061 | 1.427 | 1.413–1.439 |
| SmolVLA forward/backward/SGD | 14.947 | 20.716 | 1.380 | 1.360–1.401 |

Intervals bootstrap paired-ratio medians 10,000 times and describe within-run
uncertainty, not between-run variation. A separate 20-pair-per-case control uses
the preserved pre-fix binaries: persistent forward/training medians are
6.396/20.779 ms. The new forward median is about 5.2% lower; training is effectively
unchanged and CI shows no established benefit. Old-binary controls are historical
timing comparisons, not matching-arithmetic pairs against the corrected binary.
Their recorded worktree revision is not their binary build identity. An earlier
40-pair model exploration is also preserved. Baseline parity remains unmet.

Separate driver diagnostics report 116/117 registers for corrected forward/
training shaders, versus 112/112 before the fix. Forward shared storage falls
from 29,200 to 20,880 bytes; training remains 29,232 bytes. Executable sizes fall
from 114,560/272,000 to 101,248/259,968 bytes. The driver's implausible Local
Memory Size field remains excluded.

Fresh NVIDIA SDK captures use the previous six sequential single-pass metric
groups per mode, with 146 samples in every group, no SDK warnings, matching
NVIDIA adapter identity and successful target/monitor exits. Both corrected
paths retain the exact previous forward-state SHA-256
`4e5d66e6ce5e95142b848651d316b4df64855a9d85669499cabc3a81089be629`.
Repeated instruction rates deviate by at most 0.50% from their group mean;
clock rates deviate by at most 0.23%. Rate times replay duration gives:

| Metric | Corrected ordinary | Corrected persistent |
|---|---:|---:|
| Graph replay ms | 3.579 | 5.582 |
| Warp instructions, relative to ordinary | 1.000 | 1.934 |
| FMA-heavy integer-thread activity, relative | 1.000 | 2.625 |
| Global-load sectors, relative | 1.000 | 1.097 |
| Shared-memory wavefronts, relative | 1.000 | 1.155 |
| Barrier waiting, share of active warp cycles | 3.70% | 61.57% |
| Local-load/store sectors | 0 / 0 | 0 / 0 |

The earlier forward capture estimated about 39,854 local-load and 678,795
local-store sectors for persistent execution. The corrected capture reports
zero for both metrics. This is evidence for the measured forward replay, not
proof that every supported graph is spill-free. The integer metric does not
count all integer operations; rate-times-duration values are activity estimates,
and barrier share is not an elapsed-time fraction. Substantial extra instructions
and coordination costs remain after local traffic disappears.

All jobs use verified separate bounded cgroups; GPU jobs are serialized, and
builds/analysis/profiling do not overlap primary timing. Artifacts stay on `/x`.
Tests, source patch, pre-fix controls, production binaries and SDK receipts are
indexed in `target/megakernel-results/horizontal-geometry/summary.json`. The next
investigation targets the remaining cost on this corrected runtime, including
training counters and the cooperative-prologue handler's resource footprint.

## Scalar matmul output stores (October 26, 2026)

The corrected runtime's first backward SDK capture completes successfully on
both backends, but the wrapper's cross-backend bitwise check fails. That failure
is preserved. A separate audit compares all 246,883,105 state floats: loss and
parameters are exact; 54,475,347 gradient values differ, with maximum error
4.6566128730773926e-9. The largest fraction of the existing benchmark tolerance
is 0.000460. Each backend remains bitwise stable across its entire capture.
The SDK script's exact-state gate is not weakened.

This capture measures **forward plus backward with the optimizer disabled**,
not the complete SGD step. Ordinary/persistent replay takes 10.611/17.175 ms.
Rate times duration estimates about 12.08 million local-load and 206.12 million
local-store sectors for persistent execution, versus zero ordinary. The earlier
forward-only zero-traffic result therefore did not extend to backward work.

A cooperative-matmul probe moves output-address calculations after accumulation.
It saves one forward and two training registers, but 40 normal comparisons show
no useful timing gain. The probe is reverted; its source, binary and two resource
diagnostics remain in `target/coop-output-address-probe/`.

Commit `5dbabed` changes the common scalar matmul generator. It previously built
a nested array from completed scalar accumulators and dynamically indexed that
array in output loops. It now emits a bounds-checked store for each accumulator,
in the same row/column order. Both ordinary and persistent execution use this
change. K staging and accumulation, selected geometry, column mapping, split
offsets, fused addends and epilogues remain unchanged. The WGSL store fragment
stays in `matmul.wgsl`. Queue scheduling, fences, scratch isolation and submission
are unchanged; the shared convolution generator retains its existing output path.

Qualification passes **665 library tests (eight ignored), 168 strict whole-graph
oracle cases, 50 matrix regressions and seven whole-graph tests**, plus formatting
and strict all-feature/all-target Clippy. Matrix regressions cover transposes,
partial dimensions, packed outputs, rectangular candidates and split reductions.
Naga and Blade pins are unchanged. Clean tracked source `5dbabed` produces the
final binaries; the unrelated untracked `configs/` directory is disclosed.

All 250 primary comparisons pass with 256 workers, 50 alternating AB/BA pairs
per workload, ten warmups and at least one second of warmup. Every persistent
case has one dispatch, identical selected arithmetic and no fallback. CI and
forward are exact; maximum training error remains 4.66e-9.

| Workload | Ordinary ms | Persistent ms | Paired ratio | 95% interval |
|---|---:|---:|---:|---:|
| CI inference | 0.05728 | 0.07564 | 1.319 | 1.310–1.327 |
| CI forward/backward | 0.06613 | 0.10239 | 1.553 | 1.543–1.567 |
| CI forward/backward/SGD | 0.06714 | 0.10096 | 1.508 | 1.484–1.524 |
| SmolVLA forward | 4.207 | 6.056 | 1.436 | 1.427–1.482 |
| SmolVLA forward/backward/SGD | 14.945 | 19.403 | 1.310 | 1.276–1.471 |

Intervals bootstrap paired-ratio medians 10,000 times and describe within-run
uncertainty. Long outliers on both paths remain in every calculation and widen
the training interval; no samples are excluded. A subsequent 50-pair-per-phase
control uses the preserved geometry-corrected binary: forward/training medians
are 6.097/20.725 ms persistent and 4.217/14.940 ms ordinary. The observed training
median falls about **6.4%**, with forward effectively unchanged. The normalized
training intervals overlap, so the median improvement is not a guarantee across
runs. Independently bootstrapping the two runs' persistent latency medians gives
a new/old ratio of 0.936 (95% interval 0.924–0.952); ordinary is 1.000
(0.978–1.021). These intervals also exclude between-run drift. The source
revision observed by an old binary is not its build identity.
Earlier 20-pair-per-phase candidate/control runs are also preserved: there are
470 normal comparisons and four resource diagnostics across this investigation.
**Baseline parity remains unmet.**

The new backward SDK capture uses six sequential single-pass groups per mode,
146 samples each, matching NVIDIA adapters, successful process exits and no
warnings. Both modes preserve their exact respective pre-change state hashes;
the known cross-backend gradient difference remains. Repeated instruction rates
deviate by at most 0.22% from their group means and clocks by at most 0.25%.

| Backward graph metric | Ordinary | Persistent |
|---|---:|---:|
| Replay ms | 10.526 | 15.926 |
| Warp instructions, relative to ordinary | 1.000 | 1.833 |
| FMA-heavy integer-thread activity, relative | 1.000 | 2.923 |
| Global-load sectors, relative | 1.000 | 1.364 |
| Shared-memory wavefronts, relative | 1.000 | 1.186 |
| Barrier waiting, share of active warp cycles | 5.74% | 57.40% |
| Local-load/store sectors | 0 / 0 | 0 / 0 |

Removing the output arrays eliminates the measured local traffic and shortens
persistent backward replay by about 7.3%. This supports array materialization or
its resulting register pressure as a cause; the aggregate counters do not map
traffic to individual instructions. Counts estimate rate times duration, the
integer metric covers one pipe class, and barrier share is not elapsed-time share.
The capture excludes optimizer work and is not a full-step timing substitute.
It uses the preserved exploratory binary; the only subsequent shader change
before the final production build corrects a source comment.

Separate full-step resource diagnostics report unchanged forward resources:
116 registers, 20,880 shared bytes and 101,248 executable bytes. Training falls
from 117 to 114 registers and from 259,968 to 249,344 executable bytes, with
29,232 shared bytes unchanged. The driver's implausible Local Memory Size field
is excluded. Diagnostic instrumentation timings are not performance results.

All jobs use verified separate cgroups with 16 GiB MemoryHigh, 20 GiB MemoryMax
and 1 GiB MemorySwapMax. GPU jobs are serialized; builds, copies and heavy
analysis do not overlap primary timing. The agent remains outside those scopes
and all artifacts stay on `/x`. Captures, patches, binaries, test logs, raw
samples, uncertainty calculations and artifact hashes are indexed by
`target/megakernel-results/scalar-output-stores/summary.json`; the preserved
initial backward failure and numerical audit are in
`target/corrected-training-sdk/summary.json`. Remaining work is to isolate the
extra instruction and scheduling cost after local traffic disappears.

## Complete-step SDK replay and host attribution

The retained runtime remains `5dbabed`; these changes extend capture tooling.
`--capture-phase step` includes forward, backward and SGD, using a zero learning
rate so repeated captures preserve state. It selects the same graph and optimizer
pipeline keys and task counts as the normal nonzero-learning-rate benchmark:
ordinary execution has 527 compute dispatches, while persistent execution has
one. Four SGD nodes contribute 97,522 logical tiles. Zero-rate replay is a
diagnostic and does not replace the completed-step measurements above.

The SDK wrapper keeps exact cross-backend comparison as its default. Explicit
`--state-comparison benchmark` compares every captured float using the shared
benchmark budget `1e-5 + 1e-4 * max(abs(a), abs(b))`, records the policy and error,
and requires NumPy for chunked memory-mapped comparison. Both policies still
require bitwise-stable replay within each backend. The previously failed exact
backward capture remains preserved. Run a new complete-step capture with:

```sh
python3 scripts/run_bounded_job.py sdk-step --gpu -- \
  env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  python3 scripts/profile_smolvla_sdk.py --sdk /x/Tools/nsight-perf-sdk-2026.3 \
  --output target/new-step-capture --phase step --workers 256 --whole-graph \
  --state-comparison benchmark
```

Separate six-group counter and eighteen-group stall captures each collect 146
samples per group and mode. All target and monitor exits succeed, adapters match,
and no SDK warnings occur. Captured state hashes match each backend's previous
backward capture exactly. Comparing all 246,883,105 floats across backends gives
the existing maximum error of 4.66e-9; 54,475,347 float representations differ.
The maximum error/budget fraction is 0.0004594. Cross-backend state is numerically
qualified, not bitwise equal.

The six-group ordinary/persistent replay medians are 11.941/17.245 ms. Both modes
have zero local-load and local-store sectors. Rate-times-duration estimates are
1.844× warp instructions, 2.659× FMA-heavy integer-thread activity, 1.297× global
load sectors and 1.216× shared-memory wavefronts. The integer metric covers that
execution pipe rather than all integer instructions. Repeated instruction rates
vary by at most 0.33% from their within-mode group mean; clocks by at most 0.25%.

The eighteen-group replay medians are 11.982/17.287 ms. Its leading issue states
are below, expressed as percentages of active warp cycles, **not elapsed-time
percentages**. The SDK defines no-instruction waiting to include instruction
selection/fetch as well as cache misses; these measurements do not establish
instruction-cache thrashing.

| Issue state | Ordinary % | Persistent % |
|---|---:|---:|
| L1TEX memory dependency | 60.049 | 19.109 |
| Workgroup barrier | 3.035 | 51.770 |
| Memory barrier | 0.000 | 1.428 |
| No instruction | 0.340 | 0.939 |
| MIO dependency | 4.072 | 3.028 |
| MIO queue throttle | 10.270 | 6.660 |
| Fixed-latency dependency | 8.518 | 8.568 |

Across those eighteen groups, repeated instruction-rate deviation is at most
0.91% and clock deviation at most 0.38%. Counters describe the combined shader;
they do not attribute barrier cycles to individual scheduler or tile barriers.

A separate 300-comparison CI host probe splits input upload, recording/submission,
completion wait and readback. For forward/backward/SGD, median recording/submission
falls from 18.475 to 7.360 microseconds, while completion wait rises from 48.055
to 92.805 microseconds. Input and readback are unchanged at 0.22 and 0.54
microseconds. The other two CI cases show the same direction. These instrumented
host intervals include driver and wakeup costs and are not pure GPU timestamps
or primary timing results. The temporary instrumentation is reverted.

Retesting immutable descriptor variants after scalar output stores gives
6.114 ms forward and 23.354 ms training, with paired ratios 1.551 and 1.567 across
twenty pairs per case. All 100 CI/model comparisons preserve selected arithmetic,
state budgets and one dispatch, but training regresses against the retained
19.403 ms primary measurement. Training resources rise to 142 registers and
337,920 bytes of code, from 114 and 249,344. The variant is reverted.

Capture-tool qualification passes three execution admissions, seven rejection
fixtures and six state-comparison fixtures, plus formatting and strict all-feature,
all-target Clippy. Runtime sources are unchanged; the existing runtime test
qualification still applies. All jobs use the verified bounded cgroups and GPU
serialization described above. Captures, tooling source patches, binaries,
host/variant probes and 236 independently verified artifact hashes are indexed by
`target/megakernel-results/step-sdk/summary.json`. The independent hash-audit log is
`target/step-sdk-artifact-audit.log`.

## Flat packing probe

A subsequent prototype packs consecutive linear tile indices across grid-axis
boundaries, allowing a partial final claim. The existing lane masks suppress
inactive groups' writes/products, scratch remains separate, and inactive groups
clamp reads to the last valid tile. The generic queue already clamps final-batch
completion. Barrier admission still applies the scope-aware checker to every
original barrier, rejecting divergent control before trying existing axis packing.
No queue, fence or submission protocol changes are involved.

Both parser checks and all seven whole-graph GPU tests pass. The existing packed
matrix fixture includes 130- and 258-tile grids with partial four-tile batches,
1/3/64/4096 workers, changed inputs, exact outputs and checked completion counts.
Twenty alternating pairs per model phase, followed by twenty pairs of the
preserved current binary, give:

| Case | Flat ordinary / persistent ms | Current ordinary / persistent ms | Flat / current paired ratios |
|---|---:|---:|---:|
| SmolVLA forward | 4.248 / 6.378 | 3.980 / 6.023 | 1.542 / 1.448 |
| SmolVLA forward/backward/SGD | 14.948 / 19.544 | 14.893 / 19.367 | 1.317 / 1.302 |

All eighty normal comparisons preserve selected arithmetic, one persistent
dispatch, no fallback and the previous state-error bound. Two separate metadata
diagnostics reveal that all sixteen eligible cooperative nodes have grid
`[2, 128, 1]`. They already pack four tiles along the 128-wide axis in the retained
shader. Flat packing changes tile grouping without increasing the number of
useful subgroups per worker; the suspected divisibility limitation is absent in
this model. The experiment does not establish a timing improvement and is fully
reverted, including its temporary metadata dump. The restored benchmark binaries
are rebuilt from the retained source.

These are exploratory comparisons with all outliers retained; diagnostic timings
are excluded, and the rejected candidate does not receive the full oracle gate.
Source patch, proof, binaries, test logs, raw samples and metadata are preserved
in `target/flat-packing-probe/`, indexed by
`target/megakernel-results/flat-packing/summary.json`. Every build, test, timing and
analysis job uses a verified separate bounded cgroup, and GPU jobs run serially.
The runtime remains `5dbabed`, with the primary qualification and parity gap above.

## Scalar staging and combined broadcast probes

Three further prototypes are evaluated on source base `f3b38b6` and fully
reverted. The retained runtime and its primary qualification remain `5dbabed`.
Every candidate preserves selected arithmetic and one persistent compute
dispatch, and all normal state comparisons satisfy the existing budget.

The first two change the common scalar Small-tile generator used by both paths.
Alternating two scratch stages lets iteration i+1 load a different buffer while
slower lanes finish reading iteration i. The staging barrier in i+1 then orders
all those reads before iteration i+2 reuses the first buffer. K arithmetic,
geometry, tails, split offsets and output stores are unchanged. Larger tiles keep
their original scratch and reuse barrier. The first version uses dynamic stage
bases; the second emits two explicit stages with constant offsets and checks the
original end-K condition before each stage.

Twenty alternating pairs per model phase produce:

| Variant | Forward ordinary / persistent ms | Training ordinary / persistent ms |
|---|---:|---:|
| Dynamic scratch offsets | 4.679 / 6.032 | 15.999 / 19.130 |
| Subsequent retained-runtime control | 3.834 / 5.712 | 14.926 / 19.487 |
| Constant scratch offsets | 4.102 / 6.625 | 15.376 / 20.222 |

The dynamic variant's paired ratios fall to 1.289 forward and 1.194 training,
partly because ordinary execution becomes slower. This is not evidence of
approaching the original baseline. The explicit-stage variant also fails to
improve persistent timing. Dynamic/static forward resources are both 112
registers and 29,328 shared bytes, with 102,400/122,880 code bytes. Training uses
114/103 registers, 29,360 shared bytes and 250,880/283,264 code bytes. Reduced
register counts alone do not establish a speedup.

The dynamic candidate's library run passes 664 tests and fails the assertion
that expected one Small-tile scratch allocation. Updating that expectation for
two stages makes the focused check pass; the original failure is preserved.
The static candidate passes all 48 code-generation tests. Each version passes
eight strict matrix oracle cases and all seven whole-graph GPU tests. Both the
runtime changes and the temporary test-expectation change are reverted.

The third prototype groups shared node metadata and parameter words into one
struct. The claiming leader publishes the node before the unchanged ticket
broadcast; all lanes then copy parameters using that ticket, followed by one
uniform struct load. Successful claims use two uniform loads instead of four.
The empty/done paths, shared-memory bound, payload fences, claims, dependencies
and completion protocol are unchanged. Two compiler checks and all seven
whole-graph GPU tests pass.

Twenty pairs per case, followed by twenty fresh retained-runtime pairs, give:

| Case | Combined broadcast persistent ms | Control persistent ms | Candidate / control paired ratios |
|---|---:|---:|---:|
| CI inference | 0.07047 | 0.07008 | 1.357 / 1.358 |
| CI forward/backward | 0.10160 | 0.10226 | 1.532 / 1.551 |
| CI forward/backward/SGD | 0.10153 | 0.10070 | 1.507 / 1.499 |
| SmolVLA forward | 6.044 | 6.009 | 1.513 / 1.549 |
| SmolVLA forward/backward/SGD | 19.472 | 19.348 | 1.279 / 1.283 |

The combined broadcast retains 116 forward registers and uses 113 training
registers, versus 114 retained. Shared storage is unchanged. Code sizes are
101,120/249,344 bytes. There is no useful completed-step gain.

A matched SDK check collects two sequential single-pass groups per mode and
variant, with 146 samples per group. Both captures include zero-rate SGD,
matching adapters, successful target/monitor exits and no warnings. Each
backend preserves its exact previous state hash; all 246,883,105 cross-backend
floats satisfy the existing tolerance, with maximum error 4.66e-9. Persistent
candidate/control replay medians are 17.232/17.262 ms, with approximately
2.791/2.797 billion warp instructions per replay. Barrier waiting is
51.92%/51.81% of active warp cycles. Ordinary replay is 11.953/11.959 ms.
Repeated instruction-rate deviation within each two-group capture is at most
0.46%, and clock deviation at most 0.0022%. These counters show little change
from reducing the WGSL broadcasts. They do not establish the driver's exact
barrier instruction counts; active-warp shares are not elapsed-time shares.

All **320 normal comparisons and six resource diagnostics** retain their raw
samples and outliers. No full oracle qualification is claimed for the rejected
candidates. Timings from resource captures and zero-rate resident replay do not
replace the normal completed-step gate. The restored benchmark binaries are
rebuilt, and all runtime source changes are absent. Builds, tests, timing,
captures and analysis use verified bounded cgroups; GPU jobs remain serialized
and every artifact stays on `/x`.

Patches, proofs, binaries, test logs, source dumps, SDK records and 226 artifact
hashes are indexed by
`target/megakernel-results/synchronization-probes/summary.json`. The original
library assertion failure remains in `target/scalar-double-buffer-probe/lib.log`.
The following investigation tests handler function boundaries and register
lifetimes, with local-traffic and completed-step checks.

## Handler function-control probes

Two temporary Blade backend probes apply SPIR-V `DontInline` controls after
normal validated generation, on Meganeura source base `bca1536`. The production
Naga and Blade pins, application shaders, arithmetic, fences, descriptors,
scratch and scheduler protocol remain unchanged. Both probes are fully reverted;
the retained runtime remains `5dbabed` and its primary qualification above applies.

The first probe changes the control words of all direct handler callees of
`graph_main`, excluding `graph_claim`. A binary audit verifies that only those
words change in fifteen modules, all validated for Vulkan 1.3. Four exact smoke
tests pass and three fail: mixed-width reductions, optimizer updates, and the
attention branch of the selected-geometry test. Maximum differences are
1.19e-7, 2.33e-10 and 1.19e-7 respectively. **These exact-test failures reject
the candidate**, regardless of their numerical magnitude. No performance run
qualifies it. Removing the hint restores all seven passes.

The second probe selects only scalar matrix handlers, including their horizontal
wrappers. A shadow emission with debug names must become byte-for-byte identical
to the original module after removing debug instructions, including the header
and IDs. This identifies the intended handler functions without changing the
driver's input debug settings. Independent audits verify all twenty-three
original/modified module pairs: only twenty-five selected function-control words
change in total, and both versions pass Vulkan 1.3 validation. The classification
uses matrix-dimension arguments and scalar accumulators as an experimental
filter, not a proposed compiler API. `DontInline` is advisory; the SPIR-V hint
does not prove the driver's machine-code call structure.

All seven unchanged whole-graph smoke tests pass. Twenty alternating pairs per
model phase, followed by a fresh retained-runtime control, give:

| Case | Hint ordinary / persistent ms | Control ordinary / persistent ms | Hint / control paired ratios |
|---|---:|---:|---:|
| SmolVLA forward | 3.646 / 15.518 | 4.224 / 6.082 | 4.258 / 1.442 |
| SmolVLA forward/backward/SGD | 11.824 / 60.716 | 14.897 / 19.514 | 5.049 / 1.312 |

All eighty comparisons retain identical implementation keys, one persistent
compute dispatch, no fallback and the existing maximum training error of
4.66e-9. All samples and outliers remain. These are exploratory twenty-pair
runs; ordinary execution also varies between runs. The much slower persistent
candidate is rejected without claiming full oracle qualification.

Four separate resource diagnostics show forward registers rising from 116 to
127 and training from 114 to 128. Code sizes rise from 101,248/249,344 to
116,352/285,184 bytes; shared storage remains 20,880/29,232 bytes. Those diagnostic
timings and the driver's implausible Local Memory Size field are excluded from
performance conclusions.

Matched complete-step SDK captures include zero-rate SGD and one single-pass
metric group per mode/variant, with 146 samples each, matching adapters,
successful exits and no SDK warnings. Both variants preserve the exact prior
state hash within each backend. All 246,883,105 cross-backend floats pass the
existing comparison budget with maximum error 4.66e-9; they are not bitwise equal
across backends. Selected graph/optimizer pipelines and task counts match the
normal nonzero-rate training benchmark.

Persistent replay rises from **17.402 to 55.590 ms**, while ordinary replay stays
11.944/11.952 ms. Estimated persistent warp instructions rise from **2.830 to
8.595 billion per replay**. Local traffic changes from zero to approximately
**427.10 million load and 3.301 billion store sectors per replay**; ordinary
traffic remains zero. These rate-times-duration estimates support increased
local-memory and instruction costs from the compiler hint. They do not identify
particular stack/spill instructions or prove how much time each cost contributes.
The single-pass capture has no cross-group stability estimate, and zero-rate
replay does not replace the completed-step gate.

Both backend patches and the Cargo override are reverted. The retained examples
are rebuilt from the git-pinned dependencies, and all seven restored whole-graph
smoke tests pass. Every build, test, copy, capture and heavy analysis uses a
verified bounded cgroup; GPU work is serialized and artifacts stay on `/x`.
Proofs, patches, failed and successful logs, binaries, SPIR-V/source dumps,
measurements and **324 independently verified artifact hashes** are indexed by
`target/megakernel-results/noinline-probes/summary.json`. The independent audit
log is `target/noinline-probes-artifact-audit.log`.

The next source-level experiment tested a shared scalar body for horizontal
matrix slices, selecting buffer views uniformly before entering the K loop.
Its result and the subsequent tile selection are recorded below. Baseline
parity remains open.

## Narrow scalar tiles

Runtime `bed8c9196c7821aa2989199f71db5473901dac62` selects existing 32×16 scalar
tiles for unpinned F32 fused matmul/add products with M≤64, N≤1024 and K≥1024,
when the remaining choice is SmallTile without an epilogue. Selection runs
**after split-K planning**, so the existing split kernels and reduction nodes
remain authoritative. It preserves the configured K stage and column mapping;
an illegal shape retains SmallTile. Explicit schedules are unchanged. Both
ordinary and persistent execution use this common selection and retain each
output's original K accumulation order and addend.

Exactly sixteen nodes in each model phase change from 46 to 90 logical tiles,
adding 704 graph tiles. Other implementation keys and appended optimizer tasks
are unchanged. Worker count, queue protocol, canonical descriptors, payload
fences and the single dispatch/submission boundary are unchanged. The production
Naga and Blade pins are unchanged; there are no local dependency overrides.

### Candidate comparison

The preceding horizontal-body experiment uniformly selected B/C buffer views
and offsets before calling one shared scalar body. Source audits confirm one
such body in forward and two in training. Two focused GPU tests cover three
siblings, both 32/64 geometries, tails and changed inputs; all seven whole-graph
smoke tests pass. Eighty candidate/control comparisons show no timing benefit:
persistent forward/training medians are 6.072/19.457 ms versus 6.088/19.245 ms
control. Code size falls to 96,768/233,728 bytes, but that does not justify
retaining the rewrite. It is reverted; no full oracle or SDK qualification is
claimed for it.

Three scalar shapes were then tested with the same eligible fused products:

| Rows × columns | Tiles per changed node | Persistent forward / training ms | Observation |
|---|---:|---:|---|
| 16×32 | 92 | 5.904 / 19.257 | Repeat improves persistent timing, but ordinary forward is 2.3% slower than its fresh control |
| 16×16 | 180 | 6.010 / 19.511 | Twenty-pair exploration does not justify a repeat |
| 32×16 | 90 | 5.943 / 18.915 | Fifty-pair repeat supports lower training latency without an established ordinary regression |

The 32×16 repeat's fresh persistent control is 5.982/19.546 ms. Its candidate /
control median ratio is 0.993 forward (95% bootstrap interval 0.969–1.019) and
0.968 training (0.936–0.987). These are exploratory results, separate from the
clean-runtime primary measurements below. Temporary selectors hard-coded the
default tuning knobs; integration fixes that limitation and preserves the
original split-K policy. Every temporary source rewrite is reverted.

All tile candidates pass their focused exact-output fixture and the existing
whole-graph checks. Each has separate resource diagnostics. The first 16×16
diagnostic failed before execution because its ICD path was mistyped; that log
and empty result remain alongside the successful corrected capture. Separate
single-group complete-step SDK checks for 16×32 and 32×16 preserve the prior
state hash within each backend and measure zero local traffic. Instrumented
timings are excluded from the normal timing gate.

### Qualification and primary measurements

The integrated source passes **666 library tests** (eight ignored), **168 strict
whole-graph oracle tests**, **50 matrix regression tests**, and **eight whole-graph
tests**, plus formatting and strict all-target/all-feature Clippy. The new
selection test covers K-stage 8 fallback, stage 16 with interleaving, stage 32,
an explicit SmallTile schedule and K=8192's original split-K/reduction sequence.
The new GPU fixture compares exact signed dyadic CPU results for 35×65×1031
with an addend, changed inputs, ordinary execution and 1/3/256 workers. It checks
selected geometry, complete worker counts and one persistent compute dispatch.

Clean-runtime binaries then run fifty alternating pairs per case with 256
workers, ten warmups and one second of additional warmup. All **250 primary
comparisons** pass the unchanged full-state budget, matching ordinary/persistent
implementation keys and one persistent compute dispatch without fallback.
The maximum training difference remains 4.66e-9. All samples and outliers are
retained. Times are completed-step wall medians; ratios are medians of paired
ratios and therefore need not equal the ratio of the time columns.

| Case | Ordinary ms | Persistent ms | Paired ratio | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| CI inference | 0.052125 | 0.071101 | 1.363 | 1.355–1.379 |
| CI forward/backward | 0.066411 | 0.102471 | 1.537 | 1.526–1.557 |
| CI forward/backward/SGD | 0.067991 | 0.100801 | 1.490 | 1.478–1.508 |
| SmolVLA forward | 4.134971 | 5.853895 | 1.471 | 1.395–1.475 |
| SmolVLA forward/backward/SGD | 15.088333 | 19.131577 | 1.276 | 1.254–1.388 |

A fresh old-runtime binary control adds fifty pairs per model phase. Persistent
control medians are 6.096263/19.459284 ms, giving candidate / control ratios of
0.960 forward (0.930–1.002) and **0.983 training (0.971–0.993)**. Ordinary control
medians are 4.256071/15.085087 ms; corresponding ratios are 0.972 (0.911–0.985)
and 1.000 (0.971–1.096). Thus training's absolute improvement is about 1.7%,
while forward and normalized paired comparisons remain inconclusive. The
candidate / control ratios of paired medians are 1.030 (0.974–1.045) forward
and 0.995 (0.929–1.078) training. Bootstrap intervals use 10,000 resamples and
describe within-run sampling uncertainty, not between-run drift or other GPUs.
**Baseline parity remains unmet; ordinary execution remains the default.**

### Complete-step counter evidence

Final resource diagnostics report 116/114 registers, 20,880/29,232 bytes of
shared storage and 108,544/256,512 bytes of compiled code for forward/training.
The six-group NVIDIA SDK capture uses complete graph plus zero-rate SGD replay,
146 samples per group per backend, the matching RTX 5080 adapter, successful
target/monitor exits and no SDK warnings. Receipts identify the clean runtime
and preserved binaries. Repeated instruction rates vary by at most 0.245%
across the five groups collecting that metric; elapsed-cycle rates vary by at
most 0.357% across all six groups.

The actual state files match the prior qualified SHA-256 within each backend:
`f67022044ae8966ac48c5d62cd0907411fb42f4d1864c1ba0b3d555b17ce7be0`
for ordinary and
`1e359875fa92965d4b0aa9026c050702ae3ac165aa59c0a4cc74a6381589c3b7`
for persistent. Within-backend replay is exact. All 246,883,105 floats pass the
explicit benchmark comparison across backends with maximum error 4.66e-9;
cross-backend state is not bitwise equal. Selected graph and optimizer work
matches the normal nonzero-rate training case.

Diagnostic replay medians are 12.003 ms ordinary and 17.058 ms persistent.
Rate-times-duration estimates give **1.546 versus 2.825 billion warp
instructions**, **2.346 versus 6.342 billion integer thread instructions** and
**zero local load/store traffic** in both modes. Barrier waiting represents
3.121% versus 50.165% of active warp cycles; memory-barrier waiting is 0% versus
1.461%. These are counter shares, not elapsed-time attribution. They support
continued investigation of synchronization and execution overhead but do not
prove a specific stall's contribution to latency. Zero-rate SDK replay does
not replace the normal completed-step performance gate.

Every build, test, copy, capture and heavy analysis runs in a verified bounded
cgroup; GPU jobs are serialized and all artifacts remain on local `/x` storage.
The complete exploration and final qualification retain **950 normal paired
comparisons**, **ten resource diagnostics**, proofs, patches, binaries, source/
SPIR-V dumps, SDK receipts and state files. The manifest
`target/megakernel-results/narrow-scalar-tiles/summary.json` indexes **637
independently verified artifact hashes**; the independent audit log is
`target/narrow-scalar-artifact-audit.log`.

The subsequent scratch-clearing and pointwise-batching experiments are recorded
below. Neither is retained; the qualified runtime remains `bed8c91`.

## Scratch clearing and pointwise batching probes

The scratch probe omits the callable converter's explicit zeroing loops and
initial workgroup barrier for F32 scalar matmul bodies only. Each K stage already
overwrites all shared A/B cells it reads, including zero values for edge inputs,
then synchronizes before computing. Padding is never read; K=0 and empty splits
perform no shared reads. The original end-of-stage and worker-loop barriers
order subsequent reuse. Admission covers plain/AT/BT products, addends, scalar
horizontal slices and split-K bodies, preserving their original geometry and
arithmetic. Other families, regional execution and all payload fences retain
their existing behavior. Shared capacity and pooling are unchanged.

Source audits confirm removal from four forward and ten training handlers.
All eight whole-graph checks and all 168 strict whole-graph oracle tests pass.
Normal measurements retain **400 comparisons**: twenty-pair CI/model trials
and fresh controls, then fifty-pair model repeats with another fresh control.
Every comparison preserves implementation keys, logical work, one compute
dispatch and no fallback. Maximum training error remains 4.66e-9. The first
repeat-control launch used a mistyped ICD path and failed before GPU execution;
its log and empty result are preserved beside the successful corrected run.

| Case | Candidate ordinary / persistent ms | Control ordinary / persistent ms | Persistent candidate / control | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| SmolVLA forward, fifty pairs | 4.154 / 5.791 | 4.209 / 5.811 | 0.997 | 0.955–1.012 |
| SmolVLA training, fifty pairs | 15.173 / 18.953 | 14.848 / 19.224 | 0.986 | 0.944–1.001 |
| CI inference, twenty pairs | 0.049471 / 0.069051 | 0.057451 / 0.075850 | 0.910 | 0.870–0.931 |
| CI backward, twenty pairs | 0.066565 / 0.099536 | 0.066685 / 0.102351 | 0.973 | 0.962–0.991 |
| CI SGD, twenty pairs | 0.068116 / 0.102576 | 0.068500 / 0.101465 | 1.011 | 0.994–1.054 |

All samples and outliers remain. The large ordinary CI inference movement
illustrates why the absolute changes alone cannot establish a runtime benefit.
Candidate/control ratios of paired medians are 1.001 (0.970–1.060) forward and
0.977 (0.895–1.001) training. CI SGD's paired ratio regresses from 1.483 to 1.517,
a relative ratio of 1.023 (1.008–1.052). Intervals use 10,000 bootstrap resamples
and describe sampling within these runs, not between-run drift. The probe does
not establish a useful completed-step gain and is reverted.

Separate resource diagnostics report unchanged 116/114 registers and
20,880/29,232 bytes of shared storage. Code decreases from 108,544/256,512 to
104,576/246,656 bytes for forward/training. Candidate and fresh-control SDK
captures each collect three groups per backend, with 146 samples per group,
matching adapters, successful exits and no warnings. The actual state files
match the prior qualified hash within each backend; cross-backend state passes
the explicit benchmark comparison with maximum error 4.66e-9. Replay is exact
within each backend, and graph/optimizer arithmetic remains matched.

| Complete-step SDK diagnostic | Candidate | Fresh control |
|---|---:|---:|
| Ordinary replay ms | 12.002 | 12.009 |
| Persistent replay ms | 17.017 | 17.064 |
| Persistent warp instructions per replay | 2.808 billion | 2.829 billion |
| Persistent shared wavefronts per replay | 537.518 million | 541.199 million |
| Persistent barrier share of active warp cycles | 50.027% | 50.299% |
| Local load/store traffic, both backends | zero | zero |

Repeated instruction and clock rates vary by at most 0.327% and 0.267%
respectively across these captures' three groups. The small reductions in
estimated work and diagnostic replay time do not override the normal timing
gate. Counter counts are rates multiplied by replay duration, and barrier
shares are not elapsed-time attribution. The proof, patches, rejected source,
binaries, dumps, measurements and SDK states are indexed by
`target/megakernel-results/scalar-scratch/summary.json`, with **177 independently
verified hashes** and audit log `target/scalar-scratch-artifact-audit.log`.

The following pointwise experiment reuses the existing scratch-free sequential
batch loop, original logical coordinates and clamped completion RMW. Admission
requires generated F32 pointwise bodies without broadcasts and matching logical/
physical workgroup sizes. The first version requests four tiles per claim at
the existing 1,024-tile threshold. A second version requests two at 512, leaving
SGD's four/1,024 policy unchanged. Both pass nine whole-graph tests, including a
new exact-output fixture with changed inputs, 1/3/256 workers, a partial
1,025-tile grid and a three-tile grid using the same handler. Its initial failure
was a mistaken assertion about compilation order, before execution; the corrected
fixture requires the exact multiset of two large and two small grids.

Neither policy selects model work. A construction audit finds five generated
pointwise nodes totaling 161 tiles in forward and forty totaling 4,690 tiles
in training. The largest grid has **141 tiles**. After excluding broadcasts,
eligible work totals only seven forward tiles and 4,536 training tiles; none
meets either per-node threshold. There is no reason to relax broadcast admission
to reach these thresholds. The two variants' eighty normal comparisons are
therefore controls, not evidence of a batching speedup. Two resource diagnostics
match the retained pipeline reports, and both complete generated model shader
files are byte-for-byte identical to the retained runtime. Full oracle and SDK
qualification are not claimed for these prototypes.

Both pointwise policies, temporary diagnostics and the temporary fixture are
reverted. The evidence is indexed by
`target/megakernel-results/pointwise-batches/summary.json`, with **122 independently
verified hashes** and audit log `target/pointwise-batches-artifact-audit.log`.
The retained examples are rebuilt byte-for-byte identically to the preserved
qualified binaries, and all eight restored whole-graph checks pass. Every
build, test, copy, capture and heavy analysis uses a verified bounded
cgroup; GPU jobs are serialized and all artifacts stay on local `/x` storage.

The following experiment compares the existing contiguous and interleaved
scalar column mappings while preserving shapes, K staging and each output's
accumulation order. Baseline parity remains open.

## Interleaved scalar columns

An exploratory default-policy probe enables the existing interleaved scalar
column mapping in both backends. The same mapping selects B shared-memory
reads, fused addends and output columns. It preserves tile dimensions, K
staging, split count, precision and every output's accumulation order; explicit
scalar-shape choices remain authoritative. Exactly sixteen model shape keys per
phase change their interleave flag, while logical work and optimizer tasks stay
unchanged. Generated model sources confirm the alternate second-column loads.

Eight whole-graph smoke checks and **200 normal paired comparisons** pass.
Each case has twenty AB/BA pairs after ten warmups and at least one second of
warmup. All samples are retained. Full-state budgets, matched backend selection,
one persistent dispatch and no fallback remain intact; maximum training error
is 4.66e-9. These are exploratory runs, with no full oracle or matrix-regression
qualification claimed for the rejected probe.

| Case | Candidate ordinary / persistent ms | Fresh control ordinary / persistent ms | Candidate / control persistent median, bootstrap 95% interval |
|---|---:|---:|---:|
| Model forward | 3.844 / 5.677 | 4.185 / 5.802 | 0.978 [0.948, 1.027] |
| Model training + SGD | 15.189 / 19.731 | 15.128 / 18.930 | 1.042 [0.991, 1.055] |
| CI inference | 0.05876 / 0.07716 | 0.05219 / 0.07045 | 1.095 [1.061, 1.110] |
| CI backward | 0.06715 / 0.10169 | 0.06696 / 0.10278 | 0.989 [0.980, 1.016] |
| CI SGD | 0.06890 / 0.10325 | 0.06792 / 0.10185 | 1.014 [0.986, 1.039] |

The intervals describe within-run sampling uncertainty, not between-run drift.
The paired persistent/ordinary medians are 1.485 / 1.310 for the two model
phases, versus 1.403 / 1.271 in the fresh control. This does not establish a
useful completed-step benefit.

Two separate resource diagnostics retain 116 / 114 registers and 20,880 /
29,232 shared bytes for forward / training. Code grows from 108,544 / 256,512
to 109,440 / 258,560 bytes. Three supported SDK groups per backend each retain
146 post-warmup samples, no monitor warnings and the exact prior same-backend
state hashes. Cross-backend replay passes the existing benchmark budget.

| Training replay | Candidate | Fresh control |
|---|---:|---:|
| Ordinary / persistent ms | 12.287 / 17.676 | 12.008 / 17.083 |
| Ordinary / persistent estimated warp instructions | 1.590 / 2.877 billion | 1.542 / 2.831 billion |
| Ordinary / persistent estimated shared wavefronts | 407.615 / 476.807 million | 447.523 / 543.752 million |
| Persistent barrier share of active warp cycles | 51.07% | 50.09% |

Both modes retain zero measured local load/store traffic. The smaller shared
wavefront estimate does not produce lower instruction count or replay time.
These counters do not establish a bank-conflict or vectorization explanation.
Counts are rate-times-duration estimates; barrier shares are not elapsed-time
attribution. SDK replay uses zero-rate SGD and does not replace normal timing.

The probe is reverted exactly. Its proof, patch, binaries, samples and captures
remain at `target/scalar-interleave-probe/`. The independently checked manifest
is `target/megakernel-results/scalar-interleave/summary.json` with **168 artifact
hashes**; the restore audit is `target/scalar-interleave-artifact-audit.log`.
All jobs used verified bounded cgroups and local storage, with serialized GPU
work and no builds or analysis during normal timing or SDK capture.

## Cooperative subgroup ownership

The native cooperative matmul template stages inputs with all 64 invocations,
then each subgroup computes the same four accumulators. Only logical subgroup
zero stores their results, including the shared result used by scalar epilogues.
A one-template probe wraps cooperative operand loads and products in the same
subgroup-zero condition. It includes compensated products, preserves accumulator
initialization and every K iteration, and leaves both surrounding workgroup
barriers outside the condition. The separate 8×8 F32 and workgroup-scoped
templates are unchanged. Ordinary and persistent execution share this change;
packed persistent tiles retain independently virtualized subgroup indices.

Eight whole-graph smoke checks and **200 normal paired comparisons** pass with
the unchanged state budgets and maximum training error of 4.66e-9. Selected
implementation keys, logical work, optimizer tasks, one persistent dispatch and
absence of fallback match the retained runtime. Generated graph sources confirm
the four operand loads and products are inside the owner guard, between the
original barriers. Each case uses twenty AB/BA pairs, ten warmups and at least
one warmup second. All samples remain; no full oracle or matrix-regression
qualification is claimed for this rejected probe.

| Case | Candidate ordinary / persistent ms | Fresh control ordinary / persistent ms | Candidate / control persistent median, bootstrap 95% interval |
|---|---:|---:|---:|
| Model forward | 4.478 / 5.764 | 4.089 / 5.762 | 1.000 [0.976, 1.052] |
| Model training + SGD | 14.860 / 18.795 | 15.064 / 19.046 | 0.987 [0.917, 1.076] |
| CI inference | 0.05213 / 0.07190 | 0.05284 / 0.07241 | 0.993 [0.948, 1.074] |
| CI backward | 0.06595 / 0.10292 | 0.06670 / 0.10281 | 1.001 [0.993, 1.009] |
| CI SGD | 0.06799 / 0.10224 | 0.06794 / 0.10148 | 1.007 [0.991, 1.023] |

Ordinary forward is slower in this short run: candidate/control ratio 1.095
[1.043, 1.216]. Consequently its lower paired persistent/ordinary ratio, 1.280
versus 1.465, is not a persistent speedup. Training's paired ratio is essentially
unchanged, 1.270 versus 1.269. Bootstrap intervals describe within-run sampling
uncertainty, not between-run drift; these results do not establish a useful gain.

Two separate resource captures show forward registers falling from 116 to 100,
while training increases from 114 to 115. Shared memory stays at 20,880 / 29,232
bytes; code changes from 108,544 / 256,512 to 110,080 / 256,640 bytes.
Four supported SDK groups per backend collect 146 samples each, with matching
adapters, successful exits, no warnings and exact prior same-backend state
hashes. Cross-backend state passes the existing benchmark budget.

| Training replay | Candidate | Fresh control |
|---|---:|---:|
| Ordinary / persistent ms | 12.004 / 17.016 | 12.007 / 17.064 |
| Ordinary / persistent estimated warp instructions | 1.567 / 2.814 billion | 1.547 / 2.830 billion |
| Ordinary / persistent estimated shared wavefronts | 444.537 / 533.853 million | 449.838 / 539.899 million |
| Ordinary / persistent tensor-active percentage | 0.4314% / 0.3013% | 0.8393% / 0.5993% |
| Persistent barrier share of active warp cycles | 50.23% | 50.29% |

Both modes retain zero measured local load/store traffic. Multiplying the tensor
percentage by replay duration yields approximately 51.4% / 50.1% of the control's
ordinary / persistent tensor-active work. This supports actual duplication in
the original shader, but removing it changes persistent replay by only 0.28%
and does not improve the normal completed-step gate. Tensor-active percentage
is not elapsed-time attribution. Instruction and wavefront totals are
rate-times-duration estimates; SDK zero-rate SGD replay remains diagnostic.

The probe is reverted exactly. Its proof, source patch, measured models-feature
binaries, samples and captures are in `target/coop-owner-subgroup-probe/`.
The first build omitted the required models feature and failed before compiling;
its log is preserved. An additional all-feature build was not used for timing
or SDK capture. The independently checked manifest is
`target/megakernel-results/coop-owner-subgroup/summary.json`, with **185 artifact
hashes**; the restore audit is `target/coop-owner-artifact-audit.log`.
Both rebuilt benchmark binaries match the retained binaries byte-for-byte, and
all eight restored whole-graph checks pass. The final records are
`target/coop-owner-restored-{build,smoke}.log` and
`target/coop-owner-final-audit.log`.
All builds, tests, copies, captures and analysis use verified bounded cgroups,
serialized GPU jobs and local storage. Builds and analysis do not overlap
normal timing or SDK capture. The retained runtime and primary measurements
remain unchanged; baseline parity remains unmet.

The following investigation distributes distinct cooperative output rows across
subgroups while preserving output geometry and arithmetic. The owner-only result
does not by itself establish that a different distribution will be faster.

## Cooperative row distribution

Runtime `762faf20b03a0d36e0a56a9b5180aa486aafd0bd` distributes the native
cooperative matmul's two output row halves across logical subgroups zero and
one. Each computes two accumulators. At subgroup width 64, subgroup zero also
computes the second row's two accumulators; additional subgroups at narrower
widths never store. The original 64-invocation geometry, four operand tiles,
single K loop and unconditional staging/barriers remain. Packed graph tiles
retain separate virtual subgroup coordinates and four-tile packing.

Each output preserves its K order, conversion, fused-add initialization and
compensated main/high-low/low-high product order. Typed splits, output casts,
post-product addends, horizontal copies, prologue caching and shared epilogues
retain their behavior. The separate 8×8 F32 and workgroup-scoped templates are
unchanged. New fragments reside in the existing WGSL template. No scheduler,
descriptor, fence, Naga or validation-flag change is needed. Both ordinary and
persistent execution use the same updated template.

The initial variant put K loops inside subgroup-width-dependent rounds. It
produced correct results but failed the existing packing assertion: Naga's
scope-aware admission analysis could not prove the outer loop workgroup-uniform,
so the native handler packed one tile instead of four. Its 80 comparisons,
two resource captures and failed assertion remain in the record. The retained
variant keeps barriers outside all subgroup-dependent control. Ownership proofs
and shader validation cover alternate widths; hardware execution is qualified
only on RTX 5080.

Qualification passes **666 library tests** (eight existing ignores), **168 strict
whole-graph oracle tests**, **50 matrix regressions**, **eight whole-graph smoke
tests**, formatting and strict all-target/all-feature Clippy. One existing typed
source assertion referred to the old accumulator name; it now checks both row
paths and all four static multiply-adds. No numerical tolerance was changed.
Although this edit only affected a test, rebuilding changed the example binary
hashes. The failed equivalence audit is preserved, and the final binaries
received fresh primary measurements, resources and both-phase SDK captures.
Their generated model shaders match the exploratory fixed-barrier shaders
byte-for-byte; those exploratory timings remain separately identified.

The final primary dataset contains **250 comparisons**, fifty AB/BA pairs for
each case, plus **250 fresh old-binary control comparisons**. Every case passes
the existing state budgets and completion checks, with matching implementation
keys, one persistent compute dispatch and no fallback. Maximum training error
remains 4.66e-9. Paired ratios are medians of per-pair ratios, not ratios of the
separate latency medians.

| Case | Ordinary ms | Persistent ms | Paired persistent / ordinary, bootstrap 95% interval |
|---|---:|---:|---:|
| CI inference | 0.05225 | 0.07096 | 1.363 [1.352, 1.371] |
| CI backward | 0.06629 | 0.10208 | 1.545 [1.535, 1.555] |
| CI SGD | 0.06768 | 0.10187 | 1.508 [1.493, 1.531] |
| Model forward | 4.270 | 5.654 | 1.382 [1.328, 1.388] |
| Model training + SGD | 15.177 | 19.106 | 1.256 [1.235, 1.268] |

Against the fresh controls, persistent forward changes from 5.814 to 5.654 ms:
ratio **0.973 [0.913, 0.982]**, about 2.7% lower. Training changes from 19.216 to
19.106 ms: **0.994 [0.979, 0.997]**, about 0.6% lower. Normalizing by each run's
ordinary timings gives forward **0.996 [0.916, 1.016]**, an inconclusive change,
and training **0.969 [0.947, 0.991]**. These are modest gains; parity remains unmet.

CI inference varies between runs despite identical ordinary and persistent GPU
pipeline identities. The primary persistent median is 6.5% lower than control,
but its normalized ratio is 2.9% higher. An additional **600-comparison A/B/B/A
sequence** gives candidate/control persistent ratios 1.113 then 0.966, with
normalized ratios 0.965 then 0.988; the second comparison's intervals include
no change. CI backward and SGD intervals include no change. No stable CI
inference speedup or regression is established, nor a specific host-side cause.
All samples remain. Bootstrap intervals use 10,000 resamples and describe
within-run uncertainty, not between-run variation.

Two final resource captures show persistent forward/training register counts
falling from 116/114 to **100/103**. Shared memory remains **20,880/29,232 bytes**;
code changes from 108,544/256,512 to **106,752/256,512 bytes**.

Four SDK groups per backend and phase collect 146 post-warmup samples each.
All final captures have matching adapters, successful exits, no monitor warnings,
exact prior same-backend state hashes and zero measured local load/store traffic.
Cross-backend state passes the existing benchmark budget. Training SDK replay
uses zero-rate SGD and is diagnostic, separate from normal completed-step timing.

| Final SDK replay | Candidate ordinary / persistent | Fresh control ordinary / persistent |
|---|---:|---:|
| Forward ms | 3.852 / 5.248 | 3.677 / 5.324 |
| Training ms | 12.001 / 16.909 | 11.997 / 17.074 |
| Forward estimated warp instructions, billion | 0.382 / 0.679 | 0.382 / 0.696 |
| Training estimated warp instructions, billion | 1.549 / 2.803 | 1.553 / 2.824 |
| Forward estimated shared wavefronts, million | 126.057 / 141.433 | 129.848 / 144.561 |
| Training estimated shared wavefronts, million | 448.750 / 536.862 | 451.353 / 539.028 |
| Forward barrier share of active warp cycles | 4.10% / 57.17% | 3.97% / 57.42% |
| Training barrier share of active warp cycles | 3.12% / 50.35% | 3.13% / 50.21% |

Duration-weighted tensor activity is approximately half the control in both
phases and backends. Persistent instruction estimates fall about 2.5% forward
and 0.8% training, but training still executes about **1.81× ordinary warp
instructions**. Barrier waiting remains substantial. Ordinary forward SDK replay
is slower in this capture, consistent with treating normalized forward benefit
as unresolved. Rates multiplied by replay duration are estimates, and active
warp-cycle shares are not elapsed-time attribution. These counters do not prove
that a particular queue instruction or cache effect causes the remaining gap.

All **1,880 normal comparisons**, **six resource diagnostics** and **seven SDK
capture directories** (three exploratory, four final) are preserved under
`target/coop-row-subgroups-{probe,packed,qualified}/`. The manifest is
`target/megakernel-results/coop-row-subgroups/summary.json`, with **607 independently
verified artifact hashes**. It also checks source patches against base `70c75f4`,
generated packing, pipeline identities, exact SDK state files and restored final
binaries. Verification logs are `target/coop-rows-manifest.log` and
`target/coop-rows-artifact-audit.log`.

The final CI binary SHA-256 is
`9164358b0a3baf3e3c3a491ebbb4199736fb05fd81289c98edff18fa88af5fce`;
the model binary is
`f4c43ae7e5a36fbe571cb51496a0f7237982401cc1d01dfed76dbac70fe7448a`.
Builds, tests, copies and analyses use verified bounded cgroups on local `/x`
storage. GPU jobs are serialized; builds and heavy analysis do not overlap
normal timing or SDK capture. The following investigation measures explicit
worker counts on this geometry and deeper scalar K staging under the same
arithmetic, completion and completed-step gates.

## Worker counts and deeper scalar staging

On source base `e9a32a9`, the qualified row-distribution binaries are swept at
explicit worker counts, without changing runtime code or defaults. The earlier
CI sweep used different persistent pipelines; the earlier model sweep preceded
the current geometry and register reductions. Fresh 256-worker controls bracket
each sweep. Every run retains matching arithmetic, complete state/completion
checks, one persistent dispatch, no fallback and exact requested/reported
worker counts. All GPU pipeline identities match the qualified runtime for
each workload; only the runtime pool count changes.

CI uses fifty AB/BA pairs per case, **900 comparisons** total:

| Workers, execution order | Inference persistent ms | Backward persistent ms | SGD persistent ms |
|---|---:|---:|---:|
| 256, before | 0.07159 | 0.10218 | 0.10155 |
| 32 | 0.06816 | 0.09689 | 0.09803 |
| 128 | 0.07440 | 0.09794 | 0.09813 |
| 64 | 0.07335 | 0.09672 | 0.09720 |
| 16 | 0.07487 | 0.09836 | 0.09859 |
| 256, after | 0.07081 | 0.10235 | 0.10157 |

At 64 workers, backward is about 5.3–5.5% below both controls and SGD about
4.3% below them; normalized comparisons also improve. Inference comparisons
remain inconclusive because ordinary latency varies between runs. The 64-worker
paired persistent/ordinary ratios are still 1.349/1.460/1.435, so this does not
close the baseline gap. It is evidence for an explicit CI choice, not for
silently capping requested workers or a graph-independent automatic policy.

The model uses twenty pairs per phase, **240 comparisons** total. Counts 168
and 336 also sample two/four workers per SM on this 84-SM device; that is not
a portable workgroup-placement assumption.

| Workers, execution order | Forward persistent ms / paired ratio | Training persistent ms / paired ratio |
|---|---:|---:|
| 256, before | 5.714 / 1.366 | 19.076 / 1.273 |
| 128 | 5.446 / 1.394 | 20.467 / 1.373 |
| 336 | 5.415 / 1.354 | 18.933 / 1.264 |
| 168 | 5.379 / 1.386 | 18.900 / 1.289 |
| 512 | 5.725 / 1.370 | 19.080 / 1.281 |
| 256, after | 5.580 / 1.374 | 18.991 / 1.266 |

128 workers slows training substantially. Neither 168 nor 336 establishes a
normalized improvement against both controls; forward ordinary latency ranges
from 3.861 to 4.248 ms. Keep the existing model setting and preserve all samples.
The raw records, binaries, receipts and comparisons are in
`target/worker-count-current/`.

Two subsequent prototypes add an explicit 64-deep setting to the existing
scalar staging knob and pass it through the shared paired benchmark config to
both backends. The default remains 32 throughout. Explicit schedules, including
the existing 16-deep split-long products, stay pinned; quantized decoders retain
their fixed 32-deep layout. Each output retains ascending-K arithmetic and the
old 32-rounded tail, rather than adding another 32 zero FMAs. Neither prototype
changes the queue, descriptors, fences or submission boundary.

The first stages 64 values and uses a variable arithmetic-loop bound for the
final tail. The second executes two fixed 32-step inner loops and guards only
the second loop. Counted and unrolled 64-deep ragged products are added to the
existing smoke fixture and compared with an independent CPU result under
ordinary execution and 1/3/256 persistent workers. Both variants pass all eight
expanded whole-graph checks and **200 normal comparisons each**, twenty pairs
for all five cases with fresh 32-stage controls from the same binaries.

| Variant | Forward ordinary / persistent ms | Training ordinary / persistent ms |
|---|---:|---:|
| Variable bound, 32-stage control | 3.904 / 5.406 | 14.947 / 18.880 |
| Variable bound, 64-stage candidate | 4.267 / 6.680 | 14.887 / 21.193 |
| Fixed loops, 32-stage control | 4.219 / 5.404 | 14.818 / 18.991 |
| Fixed loops, 64-stage candidate | 3.918 / 5.750 | 14.464 / 19.088 |

The variable-bound candidate regresses persistent forward/training by about
24%/12%. Fixed loops reduce that loss, but forward still regresses about 6.4%
and training is effectively unchanged. Neither improves both model phases.
CI backward/SGD show small gains, insufficient to justify the model regressions.
The short samples include ordinary-training outliers; nothing is removed.
Bootstrap intervals describe within-run uncertainty, not drift or causal
attribution. No full oracle or matrix-regression qualification is claimed for
these rejected candidates.

Eight separate resource diagnostics show:

| Combined shader | Registers forward / training | Shared bytes forward / training | Code bytes forward / training |
|---|---:|---:|---:|
| 32-stage controls, both variants | 100 / 103 | 20,880 / 29,232 | 106,752 / 256,512 |
| 64-stage variable bound | 102 / 112 | 29,200 / 45,744 | 108,928 / 292,608 |
| 64-stage fixed loops | 100 / 114 | 29,200 / 45,744 | 133,376 / 342,656 |

Deeper staging trades barrier frequency for greater scratch and code size;
these resource observations do not assign the latency regression to one cause.
Generated graph sources verify the variable bound and the fixed-loop tail guard
with second-half shared indices. The first source-audit assertion expected the
pre-emission variable spelling; it was corrected to check the emitted loop and
index structure. Its failed log remains `target/workers-stage64-manifest.log`.
No new SDK counter capture is claimed for these probes.

Both prototypes, the temporary benchmark option and the expanded test are
reverted exactly. Rebuilt examples match the qualified binaries byte-for-byte,
and all eight restored whole-graph checks pass. The patches, proofs, measured
binaries and generated sources remain in `target/scalar-stage64-probe/` and
`target/scalar-stage64-fixed-loops/`.
The combined manifest is `target/megakernel-results/workers-stage64/summary.json`:
**1,540 normal comparisons, eight resource diagnostics and 463 independently
verified artifact hashes**. Successful audit logs are
`target/workers-stage64-manifest-fixed.log` and
`target/workers-stage64-artifact-audit.log`; restoration logs are
`target/workers-stage64-restored-{build,smoke}.log`.

All work uses verified bounded cgroups and local `/x` storage. GPU jobs are
serialized, and no builds, copies or heavy analysis overlap normal timing.
The qualified runtime remains `762faf2`, and baseline parity remains unmet.
The subsequent scheduler locality experiments are recorded below.

## Scheduler locality probes

Two experiments start from `a611752`, with the qualified runtime and binary
controls from `762faf2`. Neither changes arithmetic selection, worker counts,
graph geometry, memory fences or the complete-step submission boundary. Each
uses twenty AB/BA pairs per case, ten warmups and at least one second of warmup,
with a fresh control process before the candidate. All 400 normal comparisons
pass the existing state and completion gates, matching implementation keys,
one persistent compute dispatch and no fallback. Maximum training error remains
4.66e-9. Bootstrap intervals use 10,000 resamples with seed 20261026 and describe
within-run uncertainty, not drift or causal attribution.

The first probe starts each worker's full ready-bitmap scan at its last
successful ready word. Every word remains reachable, including wraparound.
The expanded fixture contains two independent chains and a join across 99
operations, with allocation reuse disabled to preserve their independence.
It passes all eight whole-graph smoke checks with poison, repeated steps,
external recording, serial mode and worker counts 1, 3, 64 and 4096.

The second probe lets the final producer reserve one newly ready successor
whose entire logical workload fits in one existing claim. Its ready bit stays
unpublished, so the already-running worker has exclusive ownership. Other
successors publish normally. The successor remains in the unfinished-node
count and follows the original claim, payload and completion fences. A private
reservation passes all eight smoke checks, including both multi-claim and
single-claim versions of the 99-operation fixture.

| Probe / case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate / control [95% interval] |
|---|---:|---:|---:|
| Cursor, CI inference | 0.05253 / 0.07089 | 0.05165 / 0.06989 | 0.986 [0.951, 1.057] |
| Cursor, CI backward | 0.06520 / 0.10265 | 0.06626 / 0.10173 | 0.991 [0.981, 1.034] |
| Cursor, CI SGD | 0.06827 / 0.10286 | 0.06825 / 0.10092 | 0.981 [0.973, 0.996] |
| Cursor, model forward | 4.221 / 5.642 | 4.309 / 5.718 | 1.014 [0.993, 1.064] |
| Cursor, model training | 14.779 / 19.006 | 14.768 / 19.553 | 1.029 [0.971, 1.045] |
| Handoff, CI inference | 0.05327 / 0.07132 | 0.05696 / 0.07378 | 1.035 [0.947, 1.113] |
| Handoff, CI backward | 0.06646 / 0.10286 | 0.06650 / 0.10156 | 0.987 [0.980, 0.996] |
| Handoff, CI SGD | 0.06855 / 0.10278 | 0.06834 / 0.10271 | 0.999 [0.973, 1.018] |
| Handoff, model forward | 4.245 / 5.764 | 3.947 / 5.742 | 0.996 [0.944, 1.070] |
| Handoff, model training | 14.767 / 19.073 | 14.954 / 18.894 | 0.991 [0.950, 1.008] |

The cursor's training ratio normalized to ordinary execution worsens by 3.2%
[1.015, 1.157]. The handoff's normalized candidate/control ratios are 1.012
[0.991, 1.056] forward and 0.992 [0.835, 1.109] training. Neither establishes a
useful model improvement, so both are reverted. The isolated CI reductions
do not justify retaining either change.

Eight separate resource diagnostics pass the same correctness gates. Forward /
training registers remain 100 / 103 and shared storage 20,880 / 29,232 bytes for
both candidates and controls. Generated executable sizes are:

| Variant | Forward bytes | Training bytes |
|---|---:|---:|
| Qualified control | 106,752 | 256,512 |
| Ready-word cursor | 106,880 | 256,640 |
| Private successor handoff | 107,776 | 257,536 |

No new SDK capture or counter-based causal claim accompanies these rejected
probes. The qualified SDK results above remain the current counter evidence.

The handoff investigation also preserves a correctness failure. An initial
workgroup scalar reservation produces nonfinite output in the packed-matrix
scratch test at 2080 columns, one worker and the first replay. Instrumenting
the reservation branch confirms that this graph never takes that branch;
neither of its two nodes fits one claim. Deferring metadata writes until the
next claim still fails, as does a vec4-padded shared reservation. Using a private
reservation passes the focused case and complete eight-test suite. This shows
sensitivity to the added shared state, but does not establish an alignment,
Naga or driver cause. Five failing test logs and their source variants remain
in `target/single-claim-handoff/`; no failed variant is retained in the runtime.
The next correctness investigation will minimize that reproducer and inspect
its emitted WGSL, SPIR-V and shared layout; the follow-up is recorded below.

Raw measurements, patches, proofs, binary receipts and generated shaders remain
under `target/ready-word-cursor/` and `target/single-claim-handoff/`. The combined
manifest is `target/megakernel-results/scheduler-locality/summary.json`.
Its 435 artifact hashes are independently verified; the manifest SHA-256 is
`b711bf95d0043d92cf49bd9a662a1a3aea9e33d95c13359c2986102c519aef4d`.
Both prototypes and their temporary test extensions are reverted. Rebuilt
examples match the qualified binaries byte-for-byte, and all eight restored
smoke tests pass. Restoration and audit logs use the
`target/scheduler-locality-*` prefix. Every heavy job uses a verified separate
cgroup with the existing 16/20/1 GiB limits; GPU jobs are serialized and artifacts
remain on local `/x`. The qualified runtime, Naga/Blade pins and primary
measurements remain unchanged. Baseline parity is still unmet.

## Shared-state failure reduction

At `f78a9f4`, the failing graph is reduced to 53×720 input, 2080 output columns,
one worker and one replay of the existing packed-matrix scratch test. The
smallest worker change reproduced here adds a live shared `u32`, stores the
worker ID in it and uses it for the initial ready-word index. There is no
successor handoff. The captured payload WGSL is identical to the passing
control; only this shared declaration, store and scan-start read differ.

| Diagnostic | Result |
|---|---|
| Original shared handoff | Nonfinite output at index zero |
| Unused extra shared scalar | Pass |
| Live shared worker ID, before or after scratch declarations | Nonfinite output at index zero |
| Live scalar with an initialization barrier | Incorrect finite output |
| Two or three live shared words | Incorrect finite output |
| Four or eight live shared words | Pass |
| Explicit `Aligned 16` operands on all five cooperative loads | Still fails |
| Standalone 16×16 cooperative matrix with 1/2/3/4/8 live words | All five cases pass |
| Packing disabled in graph admission | Stalls; explicitly cancelled |

The four/eight-word probes initialize and consume every word, unlike the
earlier vec4 variant that used only its first component. The pattern is
consistent with a layout sensitivity but does not establish its cause.
Vulkan requires these cooperative matrix pointers and strides to meet the
minimum of 16 bytes and the natural row/column alignment; static SPIR-V
validation does not prove the runtime address alignment.
See [VUID-RuntimeSpirv-OpCooperativeMatrixLoadKHR-08986](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#VUID-RuntimeSpirv-OpCooperativeMatrixLoadKHR-08986).
All 17 captured SPIR-V modules validate for Vulkan 1.3. Requesting driver internal
representations returns an empty list, so no native instruction or actual
shared-address inspection is claimed. The aligned-load diagnostic changes
only the five cooperative-load memory operands and does not fix the output.

The first attempt to disable packing changed a builder default that graph
admission overrides; its generated shader is unchanged and its failed run is
retained as an ineffective probe. The corrected admission change reduces each
half-precision scratch array from 512 to 256 elements, but stalls after the
shader dump. Repeated checks find the process still live, waiting without
new output or sustained CPU/GPU activity. Its exact cgroup is explicitly
stopped after more than two minutes, and the driver records exit -15. This is
a cancelled diagnostic, not a proven GPU deadlock or a passing check. The
standalone harness also retains its initial compile failure from an invalid
cleanup-method call; correcting that harness allows all five matrix cases
to execute successfully.

The record contains 15 focused runs: five passes, nine failures and one
cancellation, plus the five standalone matrix cases. These are correctness
diagnostics, with no normal benchmark or new SDK performance claim. All
temporary worker, test, packing, example and Blade instrumentation changes
are reverted. Rebuilt benchmark binaries exactly match the qualified hashes,
and all eight restored whole-graph smoke tests pass. Work remains inside
verified bounded cgroups, with serialized GPU jobs and local `/x` artifacts.

Sources, patches, outputs and process observations are in
`target/shared-state-repro/`. The manifest
`target/megakernel-results/shared-state-repro/summary.json` has 206 independently
verified artifact hashes and SHA-256
`18701f31cac9804d26dbeb34515543dc91090e4b2587643bcb13672cb077b487`.
The cause remains open; no alignment workaround or dependency change is retained.

The inspection identifies a separate performance hypothesis. Current Naga
emits per-access availability/visibility for ordinary workgroup memory accesses,
while workgroup control barriers provide ordering without the corresponding
propagation bits. Before changing this, audit whether existing barriers can
provide that propagation for every supported synchronization path, preserving
NonPrivatePointer and explicit decorations. The
[Vulkan memory model](https://docs.vulkan.org/spec/latest/appendices/memorymodel.html#availability-visibility-semantics)
defines the required barrier propagation; it does not by itself prove the
proposed compiler transformation. The subsequent prototype and its limits are
recorded below. It is not an explanation for the shared-state failure.

## Workgroup barrier propagation

At `e246ee6`, a temporary Naga change moves ordinary workgroup-memory
availability/visibility from individual accesses to the existing workgroup
control barriers. It retains NonPrivatePointer, barrier placement and scopes,
relaxed atomics, explicit storage decorations and fence-only propagation.
This includes barriers generated for workgroupUniformLoad and initialization.
The ordering argument relies on valid cross-invocation non-atomic communication
being synchronized with WorkgroupMemory semantics; it does not admit data races
or change the WGSL memory contract. Pointer helpers using workgroup globals
are covered; this validator rejects workgroup-pointer function parameters.

Any subgroup control or memory barrier, including nested or unused functions
and entries, disables the transformation for the entire module. Reusing a
Writer recomputes this decision. A separate pinned-compiler reproduction
confirms that subgroup-only barriers already fail Vulkan SPIR-V validation with
`VUID-StandaloneSpirv-MemorySemantics-10870`: the emitter selects SubgroupMemory
without a Vulkan-supported storage-class semantics bit. The test explicitly
preserves this rejection in fallback cases. No valid subgroup GPU execution
or repair of that separate issue is claimed.

Five focused Naga tests pass, checking operands, scopes, decorated accesses,
relaxed atomics, writer reuse and the fallback. All eight whole-graph smoke
tests and the publication/reuse stress test pass. Four initial test-development
failures are preserved: an unsupported subgroup enable directive, an invalid
workgroup-pointer parameter, the existing subgroup validation error, and the
test's initial SPIR-V 1.0 target. The final fixture uses supported syntax and
SPIR-V 1.3, and checks the existing subgroup rejection explicitly.

The normal comparison uses 20 AB/BA pairs per case for both fresh control and
candidate binaries: **200 comparisons** across three CI cases and both model
phases. All preserve selection, state, completion, tolerance and one-dispatch
gates. Ordinary and persistent paths use the same compiler in each binary.

| Model case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Candidate / control paired ratio, 95% interval |
|---|---:|---:|---:|
| Forward | 3.921 / 5.652 | 3.912 / 5.405 | 0.998 [0.971, 1.021] |
| Training + SGD | 14.860 / 18.863 | 14.842 / 19.059 | 1.005 [0.908, 1.142] |

The last column compares median paired persistent/ordinary ratios. Although
the raw forward persistent median falls, the normalized result establishes no
gain. Training has large outliers in both binaries, reaching 78–86 ms, which
are retained without filtering. Its raw persistent candidate/control interval
is especially wide, [0.991, 2.355]. The CI normalized ratios are 0.980, 0.996
and 1.016; all intervals include one. These data do not establish equivalence
or a small improvement, and the prototype fails the retention gate.

Four separate resource captures report unchanged forward/training resources:
100/103 registers, 20,880/29,232 shared bytes and 106,752/256,512 executable bytes.
All 43 dumped shader sources, including both whole-graph shaders, match between
control and candidate. Equal executable sizes do not prove identical native
instructions. No new SDK capture or full oracle qualification is claimed for
this rejected compiler prototype.

The temporary Cargo override and Naga implementation/tests are reverted after
preserving their patches and fixture. Both forks are clean, rebuilt examples
match the qualified binary hashes exactly, and all eight restored smoke tests
pass. Each heavy job uses a verified separate cgroup with 16/20/1 GiB limits;
GPU jobs serialize and artifacts remain on `/x`. Captures and raw samples are
under `target/workgroup-barrier-propagation/`. The manifest
`target/megakernel-results/workgroup-barrier-propagation/summary.json` has
**225 independently verified artifact hashes** and SHA-256
`1072455373873c0be6bf09093fe85a3617ad7703b7acb67776ce705d81210e7c`.
The qualified runtime, dependency pins and primary measurements remain unchanged.
Baseline parity is still unmet. The following probe supplies a scratch-ownership
argument and temporary subgroup-barrier lowering before measuring that direction.

## Subgroup-local cooperative row staging

At `5356830`, a temporary native cooperative matmul change keeps 64 logical
invocations, the original output tile, per-output accumulation order, K loop,
logical grid and four-tile packing. On 32-lane subgroups, each subgroup stages
its own A row tile and copies both B column tiles into disjoint shared slices.
Two staging passes cover the original 64 staging coordinates. The two K-loop
rendezvous use subgroup barriers. Other subgroup widths retain the original
staging and workgroup synchronization path. Prologue initialization, epilogue
scatter, graph publication/completion and scratch-reuse barriers remain.

The ownership audit finds a necessary adapter change: previously, inactive
packed groups masked writes and products but could still load a clamped active
scratch slice. The old workgroup barriers ordered those reads; subgroup barriers
would not. The probe also masks cooperative loads assigned to local matrices,
and rejects conversion if any cooperative load in such a handler is not
maskable. Generated model sources confirm all five loads in the changed handler
have activity guards. The two attention loads retain their original workgroup
synchronization. Partial packing and scratch reuse remain explicitly tested.

The temporary Naga patch maps subgroup barriers to AcquireRelease plus
WorkgroupMemory, retaining subgroup execution and memory scopes. Vulkan ignores
SubgroupMemory; it does not identify the shared storage class. See the
[Vulkan memory semantics](https://docs.vulkan.org/spec/latest/appendices/memorymodel.html#memory-semantics).
The cooperative operand checker also recognizes subgroup_size as workgroup-uniform:
[WGSL guarantees dispatch uniformity in compute](https://gpuweb.github.io/gpuweb/wgsl/#subgroups),
and Naga restricts cooperative loads/products to compute. Subgroup and invocation
IDs keep their previous classifications. No general relaxation of barrier
control validation or cross-backend repair is made.

Five focused fence tests and nine cooperative-matrix tests pass, including
SPIR-V validation, both operand scopes and negative divergent cases. All 48
shader-generation tests pass, retaining the four-tile packing assertion. All
eight expanded whole-graph smoke tests and publication/reuse stress pass. The
packed-matrix test covers plain and fused-prologue paths, short K tails,
single-tile claims, inactive groups, 1/3/64/4096 workers and three replays.

Development failures remain in the record: reserved names in the first WGSL
fixtures; missing subgroup-barrier capabilities in positive shader tests;
packing rejection from the conservative subgroup-size classification; a helper
fixture hitting existing workgroup control-flow rejection; and two fixture
variants retaining an unsupported 64×64 subgroup matrix type. The final operand
test uses valid shapes and retains the negative control tests. The first artifact
audit also mistakenly expected guards on the unchanged attention loads; the
corrected audit checks every load in the subgroup-synchronized handler.

Fresh control and candidate binaries each run 20 AB/BA pairs for all three CI
cases and both model phases: **200 normal comparisons**. All retain matching
selection, state/completion/tolerance checks, one persistent dispatch and no
fallback. Both execution modes share the candidate compiler and arithmetic.

| Model case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Candidate / control persistent median, 95% interval |
|---|---:|---:|---:|
| Forward | 4.242 / 5.703 | 4.613 / 6.028 | 1.057 [1.051, 1.116] |
| Training + SGD | 14.944 / 18.708 | 13.970 / 19.662 | 1.051 [0.997, 1.150] |

Forward's normalized paired ratio is 0.967 [0.909, 1.011], but ordinary execution
slows more, so this is not a persistent speedup. Training's normalized ratio is
1.068 [0.978, 1.920], with substantial timing variation retained. CI normalized
ratios are 1.018, 1.001 and 1.017; their intervals all include one. The probe
establishes no useful completed-step improvement and is rejected.

Four separate resource captures report forward/training shared storage growing
from 20,880/29,232 to 24,976/33,328 bytes. Registers change from 100/103 to 100/102;
executable sizes change from 106,752/256,512 to 107,776/254,848 bytes. Extra staging
and storage are potential costs, not an isolated causal attribution. No new SDK
capture or full oracle qualification is claimed for this rejected prototype.

All temporary runtime, shader, test, Cargo and Naga changes are reverted.
Rebuilt examples match the qualified hashes byte-for-byte, both forks are clean,
and all eight restored smoke tests pass. Heavy jobs use verified separate
16/20/1 GiB cgroups, GPU jobs serialize, and all artifacts stay on local `/x`.
Sources, patches, samples and failures are under `target/subgroup-row-staging/`.
The manifest `target/megakernel-results/subgroup-row-staging/summary.json` contains
**282 independently verified artifact hashes**, with SHA-256
`48d1e6d3edb41ca67d2d3539b91e3121b66148e3fd32b9388745910d498f022b`.
The qualified runtime and dependency pins remain unchanged. Next, audit which
real model dependencies permit safe release of row or element ranges; the prior
independent-node diagnostics motivate measuring that opportunity before changing
the scheduler. Baseline parity remains unmet.

## Real model dependency and row-range audit

At `c47b40a`, temporary diagnostics under the existing shader-dump option record
the selected dispatches, physical allocations, logical extents, binding offsets,
packing, parameters and dependency metadata. Scheduler code, shaders, arithmetic
and compiler pins are unchanged. Both 256-worker diagnostic comparisons pass
state, selection and completion checks with one compute dispatch and one submit.
An independent pairwise hazard reconstruction matches the captured transitive
closure, reduced edges and initial predecessor counts: 154 nodes / 161 edges
for forward and 527 nodes / 706 edges for forward/backward/SGD.

Forward has no sliced allocations, so replacing whole-allocation conflicts with
logical intervals provides no additional ordering freedom. Training packs
parameters and gradients into arenas. A hypothetical interval analysis removes
6,597 ordered node pairs and changes the reduced graph from 706 to 698 edges;
the longest unweighted path changes from 349 to 336 nodes. This uses the largest
logical extent at each physical offset, rounded to 16 bytes, and keeps appended
bindings conservative. It is an opportunity estimate, not a proof that every
selected kernel stays in those intervals or a performance prediction. The four
SGD chunks already have independent readiness: their immediate predecessors are
`[522]`, `[494,495]`, `[406,407]` and `[164,165]`. They are not chained together by
the optimizer's shared read-only segment table.

The narrower row-release candidate is the 16 repeated feed-forward chains:
`[50,720] × [720,4096] → SwiGLU [50,2048] → matrix/add [50,720]`.
Each chain has three selected dispatches and two compatible row bands, rows
0–31 and 32–49. The cooperative output includes padding through row 63.
Separate ordinary dispatch profiles locate the work:

| Graph | Eligible dispatches | Their median interval sum | All dispatch median sum | Share |
|---|---:|---:|---:|---:|
| Forward | 48 / 154 | 2.430144 ms | 4.003744 ms | 60.7% |
| Forward/backward, optimizer disabled | 48 / 523 | 2.108224 ms | 12.482208 ms | 16.9% |

Each profile contains five samples per dispatch, with exact state unchanged by
profiling. Attributions match the captured labels, grids and logical outputs.
The profiled/unprofiled wall ratios are 1.202 and 1.276: these are instrumented
ordinary intervals, not persistent per-node timings or projected speedups.
The initial profile attempt fails with `MissingGpuTimings`; enabling the existing
`MEGANEURA_GPU_TIMING` setting fixes the setup. Both logs are retained.

The static coordinate audit preserves the original grids and packing. The
cooperative matrix's two absolute ticket intervals are `[0,128)` and `[128,256)`,
each with 32 four-tile claims. SwiGLU uses `[0,256)` and `[256,400)`, with 256 and
144 claims. The final scalar matrix uses `[0,45)` and `[45,90)`, with 45 claims
each. All original coordinates occur exactly once, boundaries align with claims,
and same-operation partitions have no overlapping writes or read/write hazards.
Splitting these operations adds 48 scheduler nodes without adding payload tiles.
Only two bands are available; the earlier synthetic eight-chain result does not
establish a benefit for this model.

Two physical hazards prevent a simple row-for-row dependency rule:

- Every inference chain reuses the cooperative output allocation for the final
  matrix output, whose narrower rows overlap the first producer band. Both final
  bands must respect those WAW and SwiGLU-read WAR dependencies. The captured
  training chains use separate allocations at this boundary.
- Scalar matrix staging evaluates an element-zero load for masked rows or K
  lanes before selecting zero. The second final-matrix band therefore also reads
  the first SwiGLU band, even though that value is discarded. Its footprint must
  include that load; matching only the mathematical output rows is insufficient.

The next prototype should preserve these ranges, padded stores and all unrelated
physical hazards, rebuilding the full hazard relation before reducing edges.
Removing only old direct edges is unsafe: whole-node chains previously carried
transitive reuse ordering that finer completion can break. Completion and worker
counts must aggregate back to the original dispatches. Compare earlier release
against a control with identical partitioning, shader and whole-node barriers,
then apply the normal state/selection/latency gates. No partitioned runtime or
new SDK performance result is claimed by this audit.

All temporary diagnostics are reverted. Rebuilt examples match the qualified
hashes byte-for-byte, both forks are clean, and eight restored whole-graph smoke
tests pass. Builds, GPU jobs, copies and analyses use verified separate cgroups
with 16/20/1 GiB limits; GPU work is serialized and artifacts remain on local
`/x`. Captures, source snapshots, diagnostic binary, scripts, profiles and the
failed setup are under `target/dependency-range-audit/`. The manifest is
`target/megakernel-results/dependency-range-audit/summary.json`, with **86
independently verified artifact hashes** and SHA-256
`5b1e3bf5c20f79f260ef4044a073cdc01733218d863f5ff3682e36be5bb47391`.
The qualified runtime and primary performance results are unchanged. Baseline
parity remains unmet.

## Two-band release of feed-forward chains

At `e8ffb25`, a temporary host-side prototype splits each of the 16 audited
matrix/SwiGLU/matrix chains into two row bands. It changes no worker or payload
shader. Each part retains the original grid, parameter block and absolute ticket
coordinates; its initial claim counter starts at the band's first tile and its
remaining counter counts only that band. Completion checks use the absolute end,
while worker totals aggregate the parts. Cached parameter uploads and appended
optimizer records keep their original operation indexing.

The prototype rebuilds all physical hazards before reducing edges. Only accesses
inside an admitted chain use the proved row footprints; other relationships stay
whole-allocation. Cooperative padding, inference output reuse and the scalar
element-zero read remain in those footprints. A separate control has identical
parts and payloads but keeps whole-operation dependency edges between them.
The two source versions differ only in the host dependency-selection constant.

| Model graph | Expanded nodes | Whole-operation edges | Early-release edges |
|---|---:|---:|---:|
| Forward | 202 | 289 | 241 |
| Forward/backward/SGD | 575 | 834 | 786 |

Both modes execute the original 17,242 / 177,346 payload tiles. Independent
audits match every coordinate, parameter word, descriptor allocation, initial
counter and ready bit, full hazard closure and reduced edge set. Both contain
all 16 partitioned chains. Their 43 generated shader files are identical to each
other, and the model pipeline identities match the qualified runtime.

All eight expanded whole-graph smoke tests pass. The packed-matrix fixture adds
complete feed-forward chains at 33/50/53/64 rows with 4096 intermediate columns,
and 53 rows with 4224 columns. Each runs three exact-output replays at
1/3/64/4096 workers. Twenty captured partitioned graphs confirm that the intended
cases actually enter the new path. Existing tests also retain optimizer,
allocation-reuse, scratch, geometry and external-recording coverage.

The normal benchmark runs qualified / whole-operation / early / early /
whole-operation / qualified, with ten interleaved ordinary/persistent pairs per
case per visit: **300 comparisons** across all five cases. Selection, state
budgets and completion pass throughout, with one persistent compute dispatch and
no fallback. Maximum training error remains 4.66e-9.

| Graph | Qualified ordinary / persistent ms | Partitioned whole-operation ordinary / persistent ms | Early-release ordinary / persistent ms |
|---|---:|---:|---:|
| Forward | 3.901 / 5.381 | 3.993 / 5.400 | 4.266 / 5.691 |
| Forward/backward/SGD | 14.828 / 18.921 | 14.791 / 18.816 | 14.812 / 18.873 |

Forward early/whole persistent latency is **1.054 [0.992, 1.065]**; normalized
by each run's ordinary timing, **0.981 [0.918, 1.029]**. Against the qualified
runtime, raw latency is **1.058 [1.004, 1.080]**, while the normalized comparison
is inconclusive at **0.963 [0.907, 1.015]**. Training early/whole is
**1.003 [0.927, 1.045]** and early/qualified **0.998 [0.924, 1.051]**;
both normalized intervals also include no change. All CI comparisons are
inconclusive. These 10,000-resample intervals do not capture every source of
between-run variation: qualified forward medians change from 5.268 to 5.596 ms,
and the partitioned control changes from 5.733 to 5.369 ms. No useful gain is
established, so neither partitioning nor earlier release is retained.

Four separate native resource records confirm equal executable statistics in both modes:
forward/training use 100/103 registers, 20,880/29,232 shared bytes and
106,752/256,512 code bytes. With worker histograms enabled, early-release
scheduler storage is 438,364 / 1,272,128 bytes; the whole-operation control needs
192 more bytes for its extra edges. Initial diagnostics did not enable native
capture and contain no executable statistics; the separate capture-enabled runs
supply this check. The first resource audit incorrectly compared hash-map-derived
pipeline lists positionally. Comparing multisets, preserving duplicates, passes.
That failed audit and the initial Rust import-path build failure remain in the
record. No full oracle qualification or new SDK counter capture is claimed.

All temporary runtime and test changes are reverted. Rebuilt examples match the
qualified hashes byte-for-byte, both forks remain clean, and eight restored
whole-graph smoke tests pass. Every heavy job uses a verified separate 16/20/1 GiB
cgroup; GPU jobs serialize, and artifacts stay on local `/x`. Sources, binaries,
scripts, samples and failures are under `target/row-range-release/`. The manifest
`target/megakernel-results/row-range-release/summary.json` has **503 independently
verified artifact hashes** and SHA-256
`0e3943af3d131346c18fe4829ff89fbb681efa01de402af903e7e8c09f675c5b`.

Two-band release does not reproduce the earlier synthetic wide-graph benefit.
Next, test proven disjoint gradient-arena writes without increasing node counts
or changing whole-operation completion. That is a separate training hypothesis;
the qualified runtime and primary results remain unchanged, and baseline parity
is still unmet.

## Bounded gradient-write dependencies

The retained change narrows only proved write-only outputs when constructing
physical dependency edges. Scalar float32 `MatMulAT` outputs cover `M*N*4` bytes,
including each output of a horizontally packed dispatch. `SumRows` covers `N*4`
bytes for its serial layout and `N*workgroups.y*4` for partial-row outputs. These
templates guard their stores and never read the destination. Cooperative and
split-K matrix kernels, fused prologues/epilogues and reduced-storage outputs are
excluded. Checked arithmetic and allocation bounds fall back to the whole
allocation when the extent cannot be proved.

All reads, unrecognized writes and optimizer bindings remain whole-allocation.
Consequently only disjoint WAW relationships disappear; physical RAW/WAR and
overlapping WAW hazards retain their ordering. The planner reconstructs the full
hazard relation before reducing edges. It changes no node count, original tile,
parameter block, arithmetic, descriptor identity, shader, fence or completion
protocol. Parameter sharing invalidates cached metadata and recomputes ranges
from the rebound buffer pieces. Imported allocations remain rejected.

Independent audits reconstruct exact output bounds, full hazard closure and its
transitive reduction, initial claims, remaining counts, predecessor counts and
ready bits. The CI training graph removes two ordered pairs: bias-gradient
`SumRows` nodes 12 and 16 no longer precede independent weight-gradient
`MatMulAT` nodes 14 and 17. Removing transitive ordering can increase the number
of direct edges; edge count alone is not the scheduling opportunity.

| Graph | Nodes | Whole-allocation edges | Bounded-write edges | Ordered pairs removed |
|---|---:|---:|---:|---:|
| CI inference | 5 | 4 | 4 | 0 |
| CI backward | 18 | 19 | 20 | 2 |
| CI backward/SGD | 19 | 20 | 22 | 2 |
| Model forward | 154 | 161 | 161 | 0 |
| Model backward/SGD | 527 | 706 | 698 | 6,597 |

The model refines 99 matrix outputs and 37 row-sum outputs; CI refines three
matrix outputs and two row sums. No forward access changes. All 19 CI and 43
model generated shader files match their controls. Executable resource multisets
also match, retaining 100/103 registers, 20,880/29,232 shared bytes and
106,752/256,512 code bytes for model forward/training.

An initial 200-comparison screen suggested a CI benefit but no model gain. To
separate the dependency change from binary-layout effects, a temporary config
switch then selects whole-allocation or bounded-write hazards in one executable.
Whole / bounded / bounded / whole visits each contain 25 interleaved
ordinary/persistent pairs per case: **500 comparisons**, or 50 pairs per policy
and case. CI backward and SGD persistent latency improve **4.0% and 4.4%**.
Their bounded/whole ratios are 0.960 [0.948, 0.963] and 0.956 [0.943, 0.969];
normalizing by ordinary timing gives 0.947 [0.931, 0.963] and
0.941 [0.928, 0.955]. Model forward/training show no clear change. The switch,
diagnostic metadata dumps and temporary public configuration field are removed.

The same-binary inference control has identical graph metadata, yet its raw
latency ratio is 1.046 [1.011, 1.076]; normalization is inconclusive at
1.036 [0.936, 1.121]. This variation is preserved rather than attributed to a
dependency change that inference does not contain.

Fresh final binaries then run against saved `762faf2` binaries in the same
mirrored order, again with **500 comparisons** and 50 pairs per implementation
and case. Every comparison passes selected-implementation matching, full state
budgets and completion checks, with one persistent dispatch and no fallback.
Maximum model training error remains 4.66e-9.

| Case | Previous ordinary / persistent ms | Final ordinary / persistent ms | Final median paired persistent / ordinary |
|---|---:|---:|---:|
| CI inference | 0.05242 / 0.06786 | 0.05173 / 0.06963 | 1.371 |
| CI backward | 0.06630 / 0.10196 | 0.06635 / 0.09627 | 1.443 |
| CI backward/SGD | 0.06763 / 0.10226 | 0.06835 / 0.09675 | 1.420 |
| Model forward | 4.292 / 5.726 | 4.324 / 5.732 | 1.319 |
| Model backward/SGD | 14.736 / 18.894 | 14.749 / 18.887 | 1.358 |

Final CI backward and SGD persistent latency improve **5.6% and 5.4%**:
ratios 0.944 [0.928, 0.957] and 0.946 [0.940, 0.960]. The normalized ratios are
0.934 [0.923, 0.957] and 0.942 [0.928, 0.951]. Inference is inconclusive at
1.026 [0.976, 1.046]. Model forward/training remain unchanged within the
measurement intervals: 1.001 [0.987, 1.012] and 1.000 [0.985, 1.017].
Intervals use 10,000 bootstrap resamples and do not cover all between-run
variation. Paired ratios are medians of individual sample ratios, not ratios of
the aggregate medians; ordinary model-training samples have substantial spread.

Qualification passes 666 library tests (eight existing ignores), 168 strict
whole-graph oracle tests, 50 matrix regressions in their original mode and nine
whole-graph smoke tests, plus formatting and strict all-target/all-feature Clippy.
The added fixture compares exact parameters, gradients and optimizer state for
three gradient branches at 1/3/64/4096 workers, including SGD, Adam, clipping,
poisoned storage and repeated external records. A second part redirects two
matrix outputs and then shares their parameter allocation, checking overlapping
WAW/RAW ordering and cached replays against exact CPU values. The first version
of that fixture kept independent reads scheduled before its redirected writes;
the failed run is preserved, and the fixture now explicitly orders writes first.

An additional strict whole-graph run of all 50 matrix regressions passes 43 and
rejects seven at admission for unsupported scalar types or cooperative physical
workgroup sizes. All seven failures reproduce on the preceding qualified runtime.
No admission check or numerical tolerance is relaxed. These cases limit coverage;
the passing original-mode suite does not qualify them for whole-graph execution.

This investigation retains **1,200 normal comparisons** and **19 separate native
resource records**. No new Perf SDK counter capture is claimed. All heavy jobs
run in verified separate cgroups with 16/20/1 GiB limits, GPU jobs serialize, and
no build or heavy analysis overlaps normal measurement. Artifacts stay on local
`/x` under `target/gradient-write-ranges/`. The manifest
`target/megakernel-results/gradient-write-ranges/summary.json` contains **979
independently verified artifact hashes**, with SHA-256
`ad81207646b3ecd9b5fbf1c5e5acf68486a40106b4a657047f914376ffff7093`.
The measured final CI/model binaries have SHA-256
`c7529ae3eb8a374ec1d7a8086ff8aac7313a43c75c0f62750661c0596e19355c` and
`2ca00e91956a3eb426493746e7b9e8660321de3aab41156a71ebfb5d816266c7`.
Both dependency forks remain unchanged and clean. This is a CI training gain;
ordinary-baseline parity and the model performance gap remain open.

## Scalar matrix vector staging

Four-float input staging tests whether scalar matrix loads contribute useful
optimization headroom. The prototype reads aligned `vec4<f32>` values and
scatters their unchanged components into the existing padded shared arrays.
Accumulator arithmetic, reduction order, output stores, grids, scratch capacity,
dependencies and fences remain unchanged. Ordinary and persistent execution use
the same implementation selection. Eligibility excludes horizontal packs,
cooperative/split kernels, reduced storage, prologues/epilogues and locked
schedules; alignment checks admit only complete in-bounds vectors. Masked reads
still access element zero, covered by the unchanged whole-allocation read edges.

Three initial variants stage both A and B, only B, or only A. Each changes 50
forward and 200 training nodes. An independent indexing audit checks exact
shared-cell coverage and vector alignment at first/last row, column and K-stage
boundaries for 22 actual model shapes: **93,184 vector checks**. Each variant
passes a focused exact-reference fixture covering normal, A-transposed and
B-transposed products with partial tiles, two input seeds and ordinary execution
plus 1/3/64/4096 workers. A fourth fixture uses reductions of 516/517 elements
for the later long-K variant.

Each initial variant runs 200 normal comparisons in mirrored control / candidate
/ candidate / control visits, with ten interleaved ordinary/persistent pairs per
case per visit. These screens do not establish a useful model improvement:

| Staged input | Forward persistent candidate/control, 95% interval | Training persistent candidate/control, 95% interval |
|---|---:|---:|
| A+B | 0.972 [0.952, 1.054] | 0.991 [0.981, 1.086] |
| B only | 1.085 [1.071, 1.132] | 0.985 [0.915, 1.021] |
| A only | 0.970 [0.955, 1.067] | 0.988 [0.911, 1.040] |

The A-only forward signal warrants a same-binary control. A temporary benchmark
flag reconstructs the same selected plan with schedules locked to suppress only
the vector-staging wrapper. Both modes perform the same untimed reconstruction
and assert unchanged dispatches and buffer sizes. Audits reproduce the qualified
control's selected keys, executable resources, scheduler bytes and graph shader
hashes. Another **500 comparisons** use mirrored visits with 25 pairs per case,
giving 50 pairs per mode and case.

| Case | Scalar / vector persistent ms | Vector/scalar, 95% interval | Ratio normalized by paired ordinary timing, 95% interval |
|---|---:|---:|---:|
| CI inference | 0.06677 / 0.07241 | 1.084 [1.059, 1.111] | 0.965 [0.936, 0.990] |
| CI backward | 0.09615 / 0.09734 | 1.012 [0.991, 1.025] | 0.982 [0.960, 0.997] |
| CI backward/SGD | 0.09691 / 0.09742 | 1.005 [0.970, 1.030] | 0.989 [0.969, 1.004] |
| Model forward | 6.003 / 5.717 | 0.952 [0.942, 0.999] | 0.983 [0.967, 0.998] |
| Model backward/SGD | 18.799 / 18.739 | 0.997 [0.980, 1.013] | 0.972 [0.904, 1.076] |

The **4.8% forward gain** comes with an **8.4% CI inference regression**.
Ordinary CI inference also slows by 13.5%, so its favorable normalized ratio
does not justify retaining the change. Model training shows no clear benefit.
A-only staging keeps model shared memory unchanged and uses 100/101 registers
and 107,264/255,488 code bytes, versus the control's 100/103 registers and
106,752/256,512 code bytes. CI inference code grows from 17,536 to 26,240 bytes;
backward and SGD code also grow. These are native resource statistics, not SASS
instruction counts or an attribution of elapsed time.

A narrower A-only variant requires **K >= 512**. It changes 49 forward and 113
training nodes; the excluded forward node has K=32. CI selected keys, shader
files and executable resources now match the scalar mode exactly. Mixing scalar
and vector handlers increases model code to 116,224/290,560 bytes, with 100/100
registers and unchanged shared capacity. The same focused correctness checks
pass, followed by another **500 same-binary comparisons**:

| Case | Scalar / vector persistent ms | Vector/scalar, 95% interval | Ratio normalized by paired ordinary timing, 95% interval |
|---|---:|---:|---:|
| CI inference | 0.07596 / 0.07187 | 0.946 [0.930, 0.982] | 1.011 [0.977, 1.031] |
| CI backward | 0.09662 / 0.09802 | 1.014 [0.994, 1.032] | 1.009 [0.993, 1.042] |
| CI backward/SGD | 0.09681 / 0.09851 | 1.018 [0.987, 1.031] | 1.012 [0.990, 1.038] |
| Model forward | 5.972 / 5.783 | 0.968 [0.958, 1.012] | 0.973 [0.956, 0.995] |
| Model backward/SGD | 18.794 / 18.692 | 0.995 [0.989, 0.999] | 0.948 [0.898, 1.043] |

Identical CI inference shaders still show raw timing movement, while their
normalized interval includes no change. The model forward raw interval also
includes no change. Training improves only about 0.5%, and its normalized
interval is inconclusive. Ordinary model-training timing has substantial spread.
All intervals use 10,000 bootstrap resamples; they do not cover every source of
between-visit variation or establish performance on additional graphs.

All **1,600 normal comparisons** pass matching selection within each backend
pair, full state budgets and worker completion, with one persistent dispatch and
no fallback. Maximum model error remains 4.66e-9. There are **32 separate native
resource records** and no new Perf SDK captures. The rejected prototypes do not
receive full oracle/library/matrix qualification. The initial audit-script
failure on an omitted optional JSON field is preserved alongside its corrected
run; no GPU correctness failure is hidden.

All prototype code, the focused fixture and temporary benchmark switch are
removed. The rebuilt CI/model binaries exactly reproduce the qualified
`3cb7d57` hashes recorded above; formatting and all nine whole-graph smoke checks
pass. Both dependency forks remain unchanged. Every heavy job uses a verified
separate cgroup with 16/20/1 GiB memory limits; GPU jobs serialize and no build or
heavy analysis overlaps measurements. Local artifacts are under
`target/scalar-vector-staging{,-a-only,-b-only,-a-long}/`, with the manifest at
`target/megakernel-results/scalar-vector-staging/summary.json`. It contains
**1,405 independently verified artifact hashes**, with SHA-256
`4b5381e8ea6a78d60a445b8f5e73c8da0f0ad060743a0fd3a541d7ad78ca40a8`.

Vector staging remains evidence of possible forward headroom, not a retained
optimization. Further work must distinguish payload instruction cost from
combined-handler code/resource effects while keeping all CI and training gates.
The qualified runtime and baseline performance gap remain unchanged.

## Subgroup-uniform scalar output guards

Runtime `ac8ce60fbb2bd195358fc9b45eb555a1996d4ae1` skips a scalar matrix
subgroup's K reduction only when none of its threads owns a valid output.
`subgroupAny` votes on each thread's first output row and column, accounting for
interleaved columns. A thread's other outputs have greater coordinates, so a
false per-thread predicate implies no valid store. The vote can only enlarge
that active set: partial subgroups keep their original arithmetic. Every thread
still stages operands and reaches both workgroup barriers. K order, padding
arithmetic for active subgroups, output stores, tile geometry, dependencies,
descriptors, fences and whole-step boundaries remain unchanged. Both backends
use the same selected matrix template.

An independent geometry audit checks 1,009,152 thread/tile ownership cases over
square and rectangular tiles, both column layouts and boundary dimensions. It
finds wholly inactive warps account for 18.75% of forward and 16.3% of training
scalar warp K iterations in the audited model geometry. These are potential
reductions, not measured native instructions or a predicted elapsed-time gain.

The initial per-thread guard passes nine smoke tests, eight strict matrix
oracle tests and **500 paired comparisons**, but slows persistent model forward
by 8.0% and training by 19.2%. The candidate/control intervals are
1.080 [1.055, 1.088] and 1.192 [1.050, 1.279]. SDK captures confirm that persistent
FP32 thread activity falls 20.6/13.7%, yet resident replay grows from
5.246/16.912 ms to 5.471/18.988 ms. Persistent local-memory traffic stays zero.
This rejects padded arithmetic count alone as a sufficient explanation of the
performance gap. The per-thread source, binaries, timings and counters remain
preserved; this form is not retained.

Replacing its predicate with a subgroup vote avoids that regression. Another
**500 comparisons** find 3.9% lower persistent forward latency and 3.4/3.1%
lower CI backward/SGD latency; training-model improvement is inconclusive.
The final implementation then adds capability selection. Blade graphics
[`fde3a9f`](https://github.com/kvark/blade/commit/fde3a9fe609af7f31d50c53fc0e1469fc0e17cfa)
reports compute-stage subgroup voting from Vulkan's stage and `VOTE` property
bits. Subgroup width alone is not sufficient, as specified by
[Vulkan's subgroup properties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceSubgroupProperties.html).
Metal/GLES currently do not report this capability and retain the
original shader. Codegen defaults to that fallback. Blade macros remain pinned
at `c09c89f`; Naga remains unchanged at `bc7b70b`. No local dependency override,
new Naga syntax or memory-semantics change remains.

Qualification passes **667 library tests** (eight existing ignores), **168
strict whole-graph oracle tests**, **50 matrix regressions in their original
mode**, **nine whole-graph smoke tests**, formatting, strict all-target/all-feature
Clippy and Metal-target compilation. The new shader test covers normal,
transposed and fused-add forms, four tile geometries, both column layouts and
validation with/without subgroup capability. Blade passes 21 library tests
(one existing ignore), plus Metal and WebAssembly compilation. GPU execution
qualification remains RTX 5080 Vulkan; the original-mode matrix run does not
qualify previously rejected strict whole-graph matrix configurations.

All 19 CI and 43 model shader bodies match the successful subgroup prototype
after ignoring comments/whitespace. Its five complete graph shaders match
byte-for-byte; executable resource multisets, selected keys and scheduler bytes
also match. Model forward/training use 102/103 registers, 20,880/29,232 shared
bytes and 107,904/263,296 code bytes, versus the old 100/103 registers and
106,752/256,512 code bytes. CI retains 48 registers and adds 16 reported shared
bytes as the existing subgroup geometry checks become active.

Fresh final binaries run another **500 comparisons** against saved qualified
`3cb7d57` binaries: control / final / final / control, with 25 interleaved
ordinary/persistent pairs per case and visit. No performance override remains.

| Case | Final ordinary / persistent ms | Final median paired persistent / ordinary | Persistent final/control, bootstrap 95% interval |
|---|---:|---:|---:|
| CI inference | 0.05207 / 0.06912 | 1.329 | 0.981 [0.956, 0.990] |
| CI backward | 0.06551 / 0.09345 | 1.420 | 0.956 [0.946, 0.973] |
| CI backward/SGD | 0.06684 / 0.09417 | 1.406 | 0.970 [0.950, 0.978] |
| Model forward | 4.198 / 5.596 | 1.332 | 0.980 [0.974, 1.015] |
| Model backward/SGD | 14.754 / 18.749 | 1.282 | 0.999 [0.943, 1.010] |

CI backward/SGD improve **4.4% / 3.0%**. Normalizing by paired ordinary timing
also shows gains: 0.970 [0.959, 0.986] and 0.985 [0.970, 0.998]. The inference
normalized interval is inconclusive. Neither model phase establishes a
completed-step gain: normalized ratios are 0.979 [0.958, 1.003] forward and
1.011 [0.948, 1.047] training. Intervals use 10,000 bootstrap resamples and do
not cover every source of between-visit variation. Paired ratios are medians
of individual ratios, not ratios of aggregate medians.

Fresh final SDK captures collect four sequential metric groups per resident
replay, with matched selections and exact same-backend state against the saved
runtime. Training includes SGD at zero learning rate for stable replay. Group
clock and instruction-rate means vary by less than 1% within each capture.

| Phase | Ordinary control / final replay ms | Persistent control / final replay ms | Persistent warp instructions, billions, control / final | Persistent FP32 thread instructions, billions, control / final |
|---|---:|---:|---:|---:|
| Forward | 3.852 / 3.744 | 5.253 / 5.051 | 0.686 / 0.650 | 3.597 / 2.962 |
| Training + SGD | 12.005 / 11.990 | 16.892 / 16.266 | 2.807 / 2.739 | 18.975 / 16.937 |

Persistent resident replay improves **3.8% / 3.7%**. Estimated FP32 thread
activity falls 17.6/10.7%, and shared wavefronts fall 12.6/9.4%. Persistent local
loads/stores remain zero. Ordinary execution now has small local traffic:
about 79,780 load/store sectors forward and 323,180 training. These counters do
not identify its source, and no SASS attribution is claimed. Barrier waiting
increases from 57.07/50.09% to 59.19/50.87% of active warp cycles despite the
faster replay. Those percentages are not elapsed-time fractions. Resident replay
does not replace the completed-step model gate above.

All **1,500 normal comparisons** preserve matching selections, full state
budgets and completion, with one persistent dispatch and no fallback. Maximum
model error remains 4.66e-9. The record contains **30 native resource records**
and **16 SDK backend captures / 64 metric groups**, including the rejected
per-thread variant. The initial source-audit filename assumption failed because
shader filenames hash their contents; its log is preserved with the corrected
body comparison. No numerical tolerance or validation rule is relaxed.

Heavy jobs run in verified separate cgroups with 16/20/1 GiB limits; GPU jobs
serialize and no build or heavy analysis overlaps measurements. All artifacts
remain on local `/x` under `target/scalar-output-guard{,-subgroup,-qualified}/`.
The manifest `target/megakernel-results/scalar-output-guard/summary.json` has
**1,269 independently verified artifact hashes**, SHA-256
`7064d00abe13aee7aafa88b9d2516300a64046b690ba416b502766c0a0b0d483`.
Its frozen source patch matches the retained runtime commit. Measured final
CI/model binary SHA-256 values are
`8a0ecaf9ebc0c5ad6f4fedccd6957e2373d96773cd85e8db8be826d06c4175f6` and
`de9aea135071b6b0b8ef9f08bdd5db6aca253ccbc252073ee2b70e85b781bfcf`.
The retained change improves CI training and resident GPU replay; model
completed-step improvement and ordinary-baseline parity remain unproven.

## Interior staging for transposed scalar matrices

Runtime `540eccfd312b7e6ae372e434d36c3a10d145c684` adds a uniform direct-load
path for complete B stages in F32 transposed-B scalar products. These stages
are contiguous along K. The predicate proves that all tile columns and all K
positions are inside the matrix; the original per-element checked path handles
every edge stage. The loop already guarantees `t < end_k <= K`, and subtracting
the tile width from N avoids overflowing an upper-bound addition. Each direct
load uses the same original address and writes the same shared cell. A staging,
K order, zero padding, accumulator updates, stores, subgroup output guards,
barriers, descriptors, dependencies and completion remain unchanged. Both
backends use the same template. Other layouts and weight formats keep their
existing staging. No configuration hook or Naga/Blade change is added.

A coordinate audit checks **3,133,440 staged elements** over both B mappings,
legal tile/stage sizes, boundary dimensions and unsigned-32-bit limits. Every
admitted interior rectangle contains exactly the original coordinates. This
does not authorize allocations beyond the existing runtime limits or predict
native instruction counts.

The broad prototype enables the branch for every F32 scalar B layout. Its
500 paired comparisons pass, but persistent forward regresses **3.5%**
(candidate/control 1.035 [1.025, 1.046]); ordinary forward regresses 4.5%.
Training's raw ratio is 0.963 [0.897, 1.017], despite a favorable normalized
ratio. The forward regression rejects this form. Restricting the branch to
transposed-A products adds another 500 passing comparisons, but CI SGD
regresses **5.2%** and model training has no established gain. Its training
register count drops from 103 to 100, showing that this resource reduction
alone does not establish a speedup. Both prototypes remain rejected.

The transposed-B prototype instead improves persistent model training by
**5.7%** in 500 comparisons: 17.871 versus 18.944 ms, ratio
0.943 [0.937, 0.958]. Its normalized ratio is 0.950 [0.899, 0.963]. Both
transposed-only variants leave the complete forward shader byte-for-byte
unchanged; their differing forward timings are not evidence of a shader gain.

Final qualification passes **667 library tests** (eight existing ignores),
**168 strict whole-graph oracle tests**, **50 matrix regressions in their
original mode**, **nine whole-graph smoke tests**, formatting, strict
all-target/all-feature Clippy and Metal-target compilation. The existing
scalar edge fixture now also checks the transposed-B 64×64 kernel on a
35×65×1031 product with an addend, signed dyadic CPU results, changed inputs,
poisoned storage and ordinary/1/3/256-worker execution. Interior columns and K
stages use the direct path, while the final column tile and K tail exercise
the original path. Its first run failed a new geometry assertion expecting
32×32 SmallTile; the existing selection is 64×64 Default. Only that test
expectation was corrected, and the initial failure remains in the artifacts.
No tolerance, validation or admission rule is relaxed. These original-mode
matrix results do not qualify the previously rejected strict configurations.

All 19 CI and 43 model shader files match the successful transposed-B prototype
byte-for-byte. Pipeline resource multisets, selections and scheduler sizes also
match. Training retains **103 registers and 29,232 shared bytes**; code grows
from 263,296 to 267,776 bytes. Forward remains 102 registers, 20,880 shared
bytes and 107,904 code bytes. CI inference is unchanged; backward/SGD retain
48 registers and 9,600 shared bytes, with 61,312/65,408 code bytes.

Fresh final binaries run another **500 paired comparisons**, using
control/final/final/control visits and 25 interleaved ordinary/persistent pairs
per case and visit. The complete nonzero-rate training step remains the primary
boundary, with 256 workers, ten warmups and at least one second of warmup.

| Case | Final ordinary / persistent ms | Final median paired persistent / ordinary | Persistent final/control, bootstrap 95% interval |
|---|---:|---:|---:|
| CI inference | 0.05193 / 0.06961 | 1.342 | 1.015 [1.001, 1.066] |
| CI backward | 0.06494 / 0.09343 | 1.432 | 0.998 [0.986, 1.007] |
| CI backward/SGD | 0.06682 / 0.09455 | 1.413 | 1.005 [1.0003, 1.0095] |
| Model forward | 4.127 / 5.541 | 1.335 | 1.004 [0.988, 1.074] |
| Model backward/SGD | 14.613 / 17.892 | 1.225 | 0.944 [0.938, 0.966] |

Persistent model training falls from **18.953 to 17.892 ms**, a **5.6%** gain.
Normalizing by paired ordinary timing gives 0.971 [0.951, 0.991], a **2.9%**
improvement. Ordinary training's ratio is 0.984 [0.965, 1.002], so its apparent
gain remains inconclusive. CI SGD's raw median rises about **0.5% / 0.47 µs**;
its normalized interval, 0.999 [0.990, 1.008], is inconclusive. No CI improvement
is claimed. CI inference and model forward shader bytes are unchanged, so their
timing movements illustrate variability. All samples and outliers remain.
The 10,000-resample intervals describe these runs and do not cover every source
of between-visit variation. Paired-ratio medians need not equal ratios of time
medians. **Training remains about 22.5% slower than ordinary execution.**

Fresh NVIDIA SDK captures compare complete training plus zero-rate SGD replay,
with four sequential metric groups per backend and variant. All four backend
captures have matching adapters, successful exits, at least 100 post-warmup
samples per group and valid receipts. Actual state-file hashes match exactly
within each backend and retain the prior qualified state hashes. All complete
cross-backend state comparisons pass with maximum error 4.66e-9. Clock-rate
spread is at most 0.213%, and instruction-rate spread at most 1.066%, across
groups within a capture. No new forward counter claim is made.

| Complete-step SDK diagnostic | Control | Final |
|---|---:|---:|
| Ordinary resident replay ms | 11.9949 | 11.9427 |
| Persistent resident replay ms | 16.2214 | 15.8999 |
| Persistent warp instructions per replay, billions | 2.73978 | 2.72635 |
| Persistent integer-pipe thread instructions, billions | 6.57675 | 6.58486 |
| Persistent shared wavefronts, millions | 487.995 | 489.197 |
| Persistent barrier share of active warp cycles | 51.01% | 50.76% |
| Persistent local load/store sectors | zero | zero |

Persistent resident replay improves **2.0%**, less than the primary completed-step
gain. Aggregate integer and shared activity stays approximately flat; these
counters do not establish reduced integer instruction count as the cause.
Ordinary local load/store sectors rise from about 323,000 to 407,000 per replay.
No SASS or source-level spill attribution is available. Rate-times-duration
values estimate activity, barrier shares are not elapsed-time fractions, and
zero-rate resident replay does not replace the nonzero-rate primary gate.

The record preserves **2,000 normal comparisons, 40 native resource records,
four SDK backend captures / sixteen metric groups**, all variants, the fixture
failure and final qualification. Every comparison retains matching selected
arithmetic, full state/completion budgets, one persistent dispatch and no
fallback. All heavy jobs use verified separate cgroups with 16/20/1 GiB memory
limits; GPU jobs serialize and artifacts remain on `/x`.

Artifacts are under `target/scalar-interior-staging{,-at,-bt,-qualified}/`.
The manifest `target/megakernel-results/scalar-interior-staging/summary.json`
indexes **1,361 independently verified hashes**, SHA-256
`a2bb70df4e99c89c030e7359939c4e12668c7ffad212f79c128f7b1f71f87678`.
The frozen final patch matches the retained runtime commit. Final measured
CI/model binary SHA-256 values are
`b364fe6014d166ec297e37219d3b85cac1c99b267bd70b5f98348c23d2b5d8db` and
`3beabdb3af3e97ee825f25ae696fa25f4cb90d18914b1cc2e019be7739c4b1c7`.
The independent verification log is `target/interior-staging-verify.log`.
Baseline parity remains unmet; ordinary execution remains the default.

## Conditional scalar A staging

Two rejected prototypes on `6e8013f` test whether avoiding discarded A reads
helps the current scalar matrix path. The original checked load reads element
zero for an invalid row or K position, then selects zero for the shared cell.
Both prototypes instead read A only inside `if in_bounds`. Valid coordinates
use the identical original address; invalid coordinates still write positive
zero. Every staging cell is written before the unchanged barrier. All B formats
and normal/transposed/batched A mappings use this common template; arithmetic,
geometry, B staging, dependencies, descriptors and completion are unchanged.

The first form initializes a temporary float to zero, conditionally loads it,
then writes shared memory after the branch. The second puts the shared store
inside both branches, avoiding the temporary value across the merge. Neither
adds a barrier inside conditional control. Generated-source audits restore the
original masked expression and compare ordinary shader bodies after removing
only whitespace/comments; all other payload code matches.

Each prototype passes **40 code-generation checks, nine whole-graph smoke tests
and eight strict matrix oracle cases**. Each then runs control/candidate/
candidate/control visits with 25 interleaved ordinary/persistent pairs per case
and visit: **1,000 normal comparisons** total. All state, selection, completion,
one-dispatch and no-fallback gates pass, with maximum model error 4.66e-9.
The controls are immutable binaries of qualified runtime `540eccf`, including
its retained transposed-B interior staging. Worker count remains 256, with ten
warmup pairs and at least one second of warmup. No SDK sampling or build overlaps
normal measurements, and all samples and outliers remain.

| Form / model case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| Merged value, forward | 4.114 / 5.553 | 4.346 / 5.744 | 1.034 [1.022, 1.101] |
| Merged value, training + SGD | 14.636 / 17.758 | 14.774 / 19.002 | 1.070 [1.004, 1.160] |
| Branch-local stores, forward | 4.134 / 5.530 | 4.347 / 6.748 | 1.220 [1.183, 1.232] |
| Branch-local stores, training + SGD | 14.648 / 17.747 | 14.827 / 20.409 | 1.150 [1.145, 1.240] |

The first form regresses persistent forward/training by **3.4% / 7.0%**. Its
normalized training ratio is 1.064 [1.035, 1.086]. Moving the stores into the
branches regresses forward/training by **22.0% / 15.0%**; normalized ratios are
1.146 [1.130, 1.169] and 1.131 [1.108, 1.158]. Thus removing the temporary value
does not provide a remedy. CI also fails to establish a useful overall gain:
the first form slows backward 3.2%, while the second slows SGD 2.4% with a
normalized ratio of 1.024 [1.012, 1.039]. These 10,000-resample intervals describe
within-run uncertainty, not every source of between-visit variation.

Twenty separate native resource records preserve **102/103 registers** and
**20,880/29,232 shared bytes** for forward/training in both forms. The control's
107,904/267,776 code bytes grow to 109,568/272,768 with the merged value and
111,232/278,144 with branch-local stores. Resource totals do not identify native
predication, spills or the cause of the elapsed-time regressions. No new SDK
capture or full library/oracle/matrix-regression qualification is claimed for
either rejected candidate.

Removing the discarded element-zero read changes the shader's read footprint,
but the current whole-allocation read dependencies remain conservative. This
does not qualify a narrower hazard analysis or a row-partitioned scheduler.
The earlier two-band result remains rejected; any future revisit would need
its own physical hazard proof, matched controls and completed-step gain large
enough to cover the staging cost. Fewer source-level reads alone are not a
sufficient performance argument.

Both prototypes are reverted exactly. Release examples are rebuilt, all nine
restored whole-graph smoke tests pass, and runtime/dependency sources match
`540eccf`. Saved lockfiles also match. The first manifest attempt incorrectly
assumed ignored `Cargo.lock` was tracked; its failed log remains beside the
corrected audit. GPU work stays serialized in verified separate 16/20/1 GiB
cgroups, and every artifact remains on local `/x`.

Sources, binaries, samples, proofs and captures are under
`target/scalar-conditional-a{,-store}/`. The manifest
`target/megakernel-results/scalar-conditional-a/summary.json` contains **635
independently verified artifact hashes**, SHA-256
`03b93d9d7daad5afaca53ed00c0f823a7acf9801c51d42bce4bcdb546e1d8fa6`.
The independent audit log is `target/conditional-a-verify.log`.
The retained runtime and its performance claims remain unchanged; baseline
parity is still unmet.

## Subgroup broadcast of the claim ticket

A temporary Blade backend probe on `7a8804e` tests whether explicitly exposing
the claim ticket's subgroup uniformity helps driver control flow. It inserts
one `OpGroupNonUniformBroadcastFirst` of the `uvec4` result immediately after
the single `graph_claim` call. The function already returns
`workgroupUniformLoad(&graph_ticket)`, so every active invocation has the same
value; selecting the first active value in each subgroup preserves all four
components, including the sentinel values. Original workgroup barriers remain.
The RTX 5080 reports compute-stage ballot support for this instruction. See the
[SPIR-V instruction specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpGroupNonUniformBroadcastFirst).

The temporary postprocessor uses a separate DEBUG module to identify the call,
requiring that stripping its debug instructions reproduces the original
module. It validates the input, named and transformed modules with `spirv-val`
before pipeline creation. An independent inverse audit of **35 modules** removes
only the broadcast and any added capability/scope declaration, restores the
call result ID and bound, and reproduces every original byte. Memory operations,
scopes and semantics are unchanged. No Naga validator relaxation is involved.
This is a device-specific diagnostic, not a portable API change.

Nine whole-graph smoke tests pass. Control/candidate/candidate/control visits
then preserve **500 normal comparisons**, with 25 interleaved ordinary/
persistent pairs per case and visit, ten warmup pairs, at least one second of
warmup and 256 workers. Controls are the frozen qualified `540eccf` binaries.
Every selection, complete-state, completion, one-dispatch and no-fallback gate
passes; maximum model error is 4.66e-9. No build, heavy analysis or SDK capture
overlaps normal measurements.

| Case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| CI inference | 0.05170 / 0.06587 | 0.05728 / 0.07466 | 1.133 [1.074, 1.145] |
| CI backward | 0.06453 / 0.09327 | 0.06513 / 0.09433 | 1.011 [1.001, 1.035] |
| CI SGD | 0.06630 / 0.09445 | 0.06661 / 0.10066 | 1.066 [1.049, 1.073] |
| Model forward | 4.120 / 5.517 | 4.183 / 5.591 | 1.013 [1.005, 1.024] |
| Model training + SGD | 14.586 / 17.819 | 14.631 / 17.855 | 1.002 [0.986, 1.023] |

There is no useful model-training gain. The normalized forward/training ratios
are 1.003 [0.976, 1.030] and 0.982 [0.951, 1.020], both inconclusive. CI SGD
regresses **6.6% raw / 4.2% normalized**, with normalized interval
[1.021, 1.064]. CI inference's ordinary timing also rises 10.8%; its normalized
interval includes no change. All samples and outliers remain. These
10,000-resample intervals describe within-run uncertainty, not every source of
between-visit variation.

Ten separate resource records retain 48 CI registers, 102/103 model registers
and all original shared-memory sizes. CI backward/SGD code grows from
61,312/65,408 to 61,696/67,584 bytes; model forward grows from 107,904 to 108,032
bytes. Training remains 267,776 bytes. All **19 CI and 43 model WGSL files** are
byte-identical between control and candidate, as are ordinary pipeline resource
multisets. These counts do not establish native control-flow uniformity or
explain the timing changes. No new SDK capture or full-suite qualification is
claimed for this rejected probe.

The backend patch and ignored lockfile are restored exactly. Both forks are
clean, release examples are rebuilt from the original pins, and all nine
restored whole-graph smoke tests pass. The recovered initial smoke log contains
its passing result and separate cgroup receipt; no stale GPU job remained and
the recovered run was not duplicated. Every heavy job uses a verified separate
16/20/1 GiB cgroup, GPU work is serialized, and all artifacts remain on `/x`.

Sources, binaries, module proofs, samples and captures are under
`target/ticket-subgroup-hint/`. The manifest
`target/megakernel-results/ticket-subgroup-hint/summary.json` contains **435
independently verified artifact hashes**, SHA-256
`01a671f874b16eb8934d954ab9568fb02a3ffab667a4bdf37ddf3bf21cb34386`.
The independent verification log is `target/ticket-subgroup-hint/verify.log`.
Runtime `540eccf` remains qualified; baseline parity remains unmet.

## Scalar B shared-memory column permutation

Two rejected prototypes on `f466007` permute the physical shared-memory columns
of F32 scalar B tiles while retaining each thread's original output columns.
For `BN = 16 * TN`, the mapping is
`P(c) = c / TN + (c % TN) * 16`. It is bijective, and the original output column
`c = tx * TN + j` maps to `tx + j * 16`. Both staging and reduction use this
mapping. Unlike the earlier interleaved-output experiment, global storage
addresses and output ownership remain unchanged. So do masked element-zero
reads, padding, per-output K order, guards, barriers, shared capacity and hazards.

The broad form applies to non-interleaved F32 tiles with `TN > 1`; the second
restricts it to transposed B. Other formats and explicit interleaved output
layouts remain unchanged. NVIDIA documents successive 32-bit shared words in
successive banks, motivating a test of consecutive reduction loads across
`tx`; this source-level observation does not establish native vectorization or
a speedup. See the [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html).
An enumerated index proof checks unique staging writes and exact load mapping.
Generated-source audits invert only the changed shared B indices and reproduce
the original ordinary shader bodies after whitespace/comment normalization.

Each variant passes **40 code-generation checks, nine whole-graph smoke tests
and eight strict matrix oracle cases**. Each preserves 500 normal comparisons
in control/candidate/candidate/control order: 25 interleaved ordinary/persistent
pairs per case and visit, ten warmup pairs, at least one second of warmup and
256 workers. Controls are immutable qualified `540eccf` binaries. All full-state,
selection, completion, one-dispatch and no-fallback gates pass, with maximum
model error 4.66e-9. No build, heavy analysis or SDK sampling overlaps timings.

| Form / case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| Broad, forward | 4.094 / 5.398 | 4.177 / 5.684 | 1.053 [0.982, 1.098] |
| Broad, training + SGD | 14.600 / 17.741 | 15.889 / 18.320 | 1.033 [1.028, 1.074] |
| Transposed B, forward | 3.777 / 5.164 | 4.203 / 5.609 | 1.086 [1.022, 1.095] |
| Transposed B, training + SGD | 14.649 / 18.060 | 16.170 / 18.497 | 1.024 [1.009, 1.040] |

Broad persistent training regresses **3.3%**, while ordinary training regresses
**8.8%**. The transposed-B subset regresses **2.4% / 10.4%**, respectively.
Their normalized training ratios, 0.947 [0.930, 0.961] and 0.930 [0.856, 0.950],
improve by degrading the ordinary baseline more. That does not satisfy the
performance objective. Broad CI SGD also regresses 4.2% raw / 3.6% normalized;
the subset's CI SGD comparison is inconclusive.

The subset's complete forward and CI-inference shaders are **byte-identical**
to their controls, with matching compiled pipeline identities and resource
multisets in both backends. Their timing shifts, including an apparent
normalized forward improvement, therefore cannot be attributed to this code
change. The 10,000-resample intervals describe within-run uncertainty, not all
between-visit variation. All samples and outliers remain in the record.

Twenty resource records retain 48 CI registers, 102/103 model registers and
the original shared capacities. Broad model code grows from 107,904/267,776 to
109,952/273,280 bytes; the subset leaves forward unchanged and grows training
to 270,080 bytes. Ordinary-source audits cover all 19 CI and 43 model files per
variant. Broad/subset changes affect six/three CI files and eleven/four model
files, including the combined graph shaders.

Fresh SDK captures cover **the broad variant only**, with both model phases,
both backends and four single-pass metric groups per capture: eight backend
captures / 32 groups. Actual saved states match exactly within each backend;
cross-backend state passes the existing budget. Training replay includes SGD
at zero learning rate and does not replace the nonzero-rate completed-step gate.

| Phase / version | Ordinary / persistent replay ms | Persistent warp instructions, billions | Persistent integer-pipe instructions, billions | Persistent shared wavefronts, millions |
|---|---:|---:|---:|---:|
| Forward, control | 3.744 / 5.037 | 0.656 | 1.999 | 124.595 |
| Forward, broad | 3.812 / 5.231 | 0.668 | 2.125 | 124.758 |
| Training, control | 11.910 / 15.901 | 2.722 | 6.575 | 488.368 |
| Training, broad | 13.004 / 16.610 | 2.774 | 6.757 | 431.394 |

Persistent training's estimated shared activity falls **11.7%**, but integer
activity rises 2.8%, warp instructions rise 1.9% and replay slows **4.5%**.
Forward shared activity is approximately flat, integer activity rises 6.3%
and replay slows 3.9%. Persistent local load/store traffic remains zero.
Ordinary local load/store estimates rise from about 79,000 to 121,000 sectors
in forward and fall from about 405,000 to 200,000 in training. No native
instruction or spill attribution is established from these totals.

Clock spread across groups is at most 0.406%, and instruction-rate spread is
at most 1.705%. Persistent workgroup-barrier waiting rises from 58.94% to 59.61%
of active warp cycles in forward and 50.81% to 52.00% in training. Counter counts
are rate-times-duration estimates; these waiting shares are not elapsed-time
attribution. The evidence rejects reduced shared activity as a sufficient
argument for this layout and directs further work toward instruction cost.

Both prototypes are restored exactly to runtime `540eccf`; dependency locks
match the frozen qualified copy, both forks are clean, release examples are
rebuilt, and all nine restored smoke checks pass. No full library/oracle/matrix
qualification or subset SDK capture is claimed for the rejected prototypes.
Every heavy job uses a verified separate 16/20/1 GiB cgroup; GPU jobs remain
serialized and all artifacts are on local `/x`.

The records are under `target/scalar-shared-b-permutation{,-bt}/`. The manifest
`target/megakernel-results/scalar-shared-b-permutation/summary.json` preserves
**1,000 comparisons, 20 resource records, eight SDK captures / 32 metric groups,
and 814 independently verified artifact hashes**, SHA-256
`014007c7b3c923ca3c954fec292bfce5e1c61bc63886b8caba9efd2f9a1bde17`.
Its independent audit log is `target/scalar-shared-b-permutation/verify.log`.
Baseline parity remains unmet.

## Uniform interior A staging

Three rejected prototypes on `5095e6c` test a uniform direct-load path for
complete scalar A stages. The predicate checks `M >= BM`,
`tile_row <= M - BM` and `K - t >= KTILE`; the existing loop guarantees
`t < end_k <= K`. Every admitted A coordinate is therefore in bounds, and
the direct path uses its original storage index and shared cell. The original
per-element checked stage handles every edge, including its element-zero reads
and positive-zero padding. B staging, K order, stores, subgroup guards,
barriers, shared capacity, descriptors and dependencies remain unchanged.

The broad form enables this path for all F32 scalar layouts. Two subsets
restrict it to transposed-A or transposed-B products. Both backends use the
same template; other weight formats retain the original checked A stage.
An independent coordinate proof covers **3,133,440 staged elements**, including
both A mappings, legal tile/stage sizes, edge dimensions and unsigned-32-bit
limits. This proves the coordinate predicate without authorizing allocations
beyond the runtime's existing limits. Generated-source audits replace the new
branch with its unchanged edge body and reproduce the original ordinary shader
bodies after whitespace/comment normalization.

Each form passes **40 code-generation checks, nine whole-graph smoke tests and
eight strict matrix oracle cases**, followed by 500 normal comparisons. The
control/candidate/candidate/control protocol uses 25 interleaved ordinary/
persistent pairs per case and visit, ten warmup pairs, at least one second of
warmup and 256 workers. Controls are frozen qualified `540eccf` binaries.
All 1,500 comparisons preserve full-state budgets, matching selections,
completion, one persistent dispatch and no fallback. Maximum model error stays
4.66e-9. Builds, heavy analyses and profiling do not overlap normal timings.

| Form / case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| Broad, forward | 4.169 / 5.524 | 4.654 / 5.583 | 1.011 [0.997, 1.072] |
| Broad, training + SGD | 14.563 / 17.802 | 15.829 / 18.259 | 1.026 [1.004, 1.035] |
| Transposed A, training + SGD | 14.626 / 17.757 | 14.570 / 17.835 | 1.004 [0.987, 1.084] |
| Transposed B, training + SGD | 14.618 / 17.818 | 15.226 / 18.118 | 1.017 [0.996, 1.025] |

The broad form slows persistent training **2.6%** and CI SGD **5.4%**; ordinary
forward/training also regress **11.6% / 8.7%**. Transposed-A restriction gives
no established model-training gain and slows CI SGD **4.6%**, or 4.9% normalized.
The transposed-B subset gives no established persistent training gain and slows
ordinary training **4.2%**. Its small 0.8% raw CI SGD reduction accompanies a
2.0% ordinary slowdown and does not remedy the model result.

Normalized training ratios are 0.921 [0.873, 0.955], 1.012 [0.988, 1.108] and
0.973 [0.951, 0.987] for broad, transposed-A and transposed-B forms. The favorable
broad/transposed-B ratios reflect the larger ordinary slowdown and fail to
establish faster persistent steps. All samples and outliers remain. The
10,000-resample intervals describe within-run uncertainty, not all between-visit
variation.
Both restricted variants leave the complete forward and CI-inference shaders
byte-identical, with matching compiled pipeline identities and resource
multisets in both backends; their timing shifts are not attributed to this
change.

Thirty resource records retain 48 CI registers, 102/103 model registers and
all original shared capacities. Broad model code grows from 107,904/267,776 to
112,896/282,368 bytes. Restricted forward code stays unchanged; training grows
to 272,896 bytes for transposed A and 272,768 for transposed B. Transposed-A
CI SGD code shrinks from 65,408 to 65,024 bytes despite its slowdown. These
totals and source-level check removal do not establish native instruction
counts or explain the elapsed-time changes. No new SDK capture or full-suite
qualification is claimed for these rejected candidates.

All three probes are restored exactly to runtime `540eccf`, including the
retained B-stage optimization. Dependency locks match the qualified copy, both
forks are clean, release examples are rebuilt, and all nine restored smoke
tests pass. Heavy jobs use verified separate 16/20/1 GiB cgroups, GPU work is
serialized, and every artifact stays on local `/x`.

Records are under `target/scalar-interior-a{,-at,-bt}/`. The manifest
`target/megakernel-results/scalar-interior-a/summary.json` preserves **1,500
comparisons, 30 resource records and 991 independently verified artifact
hashes**, SHA-256
`21f0cafb3925e8503b5078ac6ac13e99a9490c22a5fc70997c3e151c1b252d28`.
The independent audit log is `target/scalar-interior-a/verify.log`.
Baseline parity remains unmet.

## Staging only valid A rows

Two rejected prototypes on `f758c61` initialize invalid A rows to positive zero
once, before the K loop, then stage only valid rows on each iteration. They
apply to F32 scalar products with normal A indexing, including transposed B;
transposed A and other weight formats retain the original stage. The first
form bounds the staging loop by the valid-row count. The second keeps the
original constant iteration count and guards each iteration by that count.
Both backends use the same template.

The valid prefix and explicitly cleared suffix partition the used shared A
cells. Every valid cell retains its original address and checked K-tail load;
invalid cells remain zero across stages. The first existing staging barrier
orders both initialization and valid writes. No barrier, reduction, store,
shared capacity, subgroup guard or dependency changes. K=0 uses none of the
initialized cells. Removing invalid-row element-zero reads does not qualify
narrower read hazards: dependencies still cover the original allocations.

Each form independently compares **3,674,112 shared-cell values across 4,784
stages and 184 layouts**, starting from poisoned scratch and using opaque
input values. The proof covers zero dimensions, edges and unsigned-32-bit
limits without authorizing larger allocations. Inverse source audits reproduce
the original ordinary shader bodies after restoring only A initialization and
staging; all 19 CI and 43 model files are accounted for.

Each variant passes **40 code-generation checks, nine whole-graph smoke tests
and eight strict matrix oracle cases**, then 500 normal comparisons against
frozen qualified `540eccf` binaries. The control/candidate/candidate/control
protocol uses 25 interleaved pairs per case and visit, ten warmup pairs, at
least one second of warmup and 256 workers. All 1,000 comparisons preserve
matching selections, full-state budgets, completion, one persistent dispatch
and no fallback. Maximum model error remains 4.66e-9.

| Loop form / case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| Variable bound, forward | 4.187 / 5.586 | 4.839 / 6.504 | 1.164 [1.129, 1.181] |
| Variable bound, training + SGD | 14.610 / 17.756 | 16.294 / 19.481 | 1.097 [1.013, 1.179] |
| Fixed bound, forward | 4.118 / 5.487 | 4.392 / 6.168 | 1.124 [1.114, 1.198] |
| Fixed bound, training + SGD | 14.603 / 17.785 | 14.935 / 19.650 | 1.105 [1.090, 1.110] |

The variable-bound form slows persistent forward/training **16.4% / 9.7%**;
ordinary execution also slows **15.6% / 11.5%**. Its favorable normalized
training ratio, 0.967 [0.948, 0.998], reflects the larger ordinary regression.
The fixed-bound form slows persistent forward/training **12.4% / 10.5%** and
CI SGD **6.2%**; normalized training and CI SGD regress **7.6% / 3.3%**. Neither
form advances parity. All samples remain; the 10,000-resample intervals
describe within-run uncertainty, not every source of between-visit variation.

Twenty resource records preserve 48 CI registers, 102/103 model registers and
the original shared capacities. Model code grows from 107,904/267,776 bytes
to 124,800/293,888 with variable bounds and 118,656/286,464 with fixed bounds.
The variable-bound CI code shrinks despite slower medians. Source operations
and these resource totals do not establish native instruction counts, loop
unrolling, spills or the cause of the timing changes. No fresh SDK capture or
full-suite qualification is claimed for the rejected prototypes.

Both probes are restored exactly to runtime `540eccf`. Dependency locks match
the qualified copy, both forks are clean, release examples are rebuilt, and
all nine restored smoke tests pass. Heavy jobs use verified separate cgroups
with 16 GiB memory high, 20 GiB maximum and 1 GiB swap; GPU work is serialized,
and builds/heavy analyses do not overlap measurements. Artifacts stay on `/x`.

Records are under `target/scalar-bounded-a{,-fixed}/`. The manifest
`target/megakernel-results/scalar-bounded-a/summary.json` preserves **1,000
comparisons, 20 resource records, 24 verified measurement scopes and 665
independently verified artifact hashes**, SHA-256
`6f35ed2ef68dec4f732ec441268732f939ac9d1cc19a12a6feb7722cc23d025e`.
The independent audit log is `target/scalar-bounded-a/verify.log`.
Baseline parity remains unmet.

## Direct coordinates for one-dimensional grids

Two rejected prototypes on `a6a7629` use the current node's uniform
`grid.y == 1 && grid.z == 1` predicate to construct logical coordinates
directly. Other grids retain the original quotient/remainder projection.
The broad form applies to every handler, including packed tiles. The subset
applies only inside the existing repeated optimizer batches. Neither changes
batch admission, selected arithmetic, shared state, barriers, fences or queue
operations. Runtime inspection preserves reuse of a handler with different
grids; no dimensions are assumed from its first recording.

For an entered claim, the tile index is below the logical count. Sequential
repeats retain their bounded final batch. Packed widths divide their selected
extent, and inactive groups clamp to the last active coordinate. For a
one-dimensional grid, x-packed coordinates therefore equal the tile plus its
clamped offset; packing on y or z has width one. The resulting y/z coordinates
are zero.
Zero-work nodes retain their outer skip, and overflow admission is unchanged.
Each independent proof checks **338,998 coordinates across 602 grids**, including
partial batches, inactive groups and unsigned-32-bit boundaries. This proves
the mapping without authorizing allocations beyond existing limits.

Both variants pass **49 code-generation checks, nine whole-graph smoke tests
and all 168 strict oracle tests**. Each then receives 500 normal comparisons
against frozen qualified `540eccf` binaries, using the established
control/candidate/candidate/control protocol, 25 interleaved pairs per case
and visit, ten warmup pairs, at least one second of warmup and 256 workers.
All 1,000 comparisons preserve matching selections, full-state budgets,
completion, one persistent dispatch and no fallback; maximum model error is
4.66e-9.

| Form / case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| Broad, forward | 4.118 / 5.493 | 4.118 / 5.517 | 1.004 [0.943, 1.065] |
| Broad, training + SGD | 14.622 / 17.815 | 14.639 / 17.873 | 1.003 [0.961, 1.087] |
| Repeated batches, training + SGD | 14.791 / 17.749 | 14.583 / 18.113 | 1.020 [1.006, 1.106] |

The broad form establishes no speedup in either model phase or CI. The subset
slows persistent training **2.0% raw / 2.6% normalized**. All samples and outliers
remain; bootstrap intervals describe within-run uncertainty, not all
between-visit variation. The subset's complete forward and CI shaders and
compiled pipeline/resource multisets are identical, so their timing shifts
are not attributed to this change.

Source audits account for 19 CI and 43 model modules per variant. Every ordinary
module is byte-identical, with matching selected pipeline keys and resource
multisets. Reversing only the coordinate transformation reproduces every whole
graph module after whitespace/comment normalization. Initial order-sensitive
resource assertions and their multiset corrections remain in the audit records.
Twenty resource records preserve shared capacities. Training registers fall
from 103 to **100 / 101** for broad/subset forms; code changes from 267,776 to
268,928 / 267,648 bytes. These reductions do not establish faster execution.

Fresh SDK captures cover **broad training only**: four backend captures and
sixteen sequential single-pass metric groups. Actual saved states match exactly
within each backend, and the cross-backend comparison satisfies the existing
budget. Zero-rate SGD keeps resident replay stable; it does not replace the
nonzero-rate completed-step gate.

| Version | Ordinary / persistent replay ms | Persistent warp instructions, billions | Persistent integer-pipe instructions, billions | Persistent shared wavefronts, millions |
|---|---:|---:|---:|---:|
| Control | 11.911 / 15.883 | 2.721 | 6.573 | 488.285 |
| Broad | 11.941 / 15.873 | 2.708 | 6.132 | 488.137 |

Estimated persistent integer-pipe work falls **6.7%**, but replay changes only
**-0.06%**. Warp instructions fall 0.48%, shared activity remains approximately
flat, and persistent local load/store traffic stays zero. Barrier waiting is
50.83% / 51.04% of active warp cycles. Across metric groups, clock spread is at
most 0.412% and instruction-rate spread 1.016%. Rate-times-duration counts are
estimates, and barrier shares do not attribute elapsed time. This transformation
reduces measured integer activity without advancing the completed-step gate;
it does not isolate the remaining cause of the performance gap.

Both probes are restored exactly to runtime `540eccf`, locks match the qualified
copy, both forks are clean, release examples are rebuilt, and all nine restored
smoke tests pass. No full retained-change qualification or subset/forward SDK
capture is claimed. Heavy jobs use verified separate 16/20/1 GiB cgroups,
GPU jobs are serialized, measurements exclude builds/heavy analyses, and all
artifacts stay on local `/x`.

Records are under `target/linear-grid{,-repeat}/`. The manifest
`target/megakernel-results/linear-grid/summary.json` preserves **1,000 normal
comparisons, 20 resource records, four SDK captures / sixteen metric groups,
26 verified measurement scopes and 762 independently verified artifact hashes**,
SHA-256 `92ec945f6265c987570a9fb524863c408aabafc73f5cb805c3cefcf17965c07f`.
The independent audit log is `target/linear-grid/verify.log`.
Baseline parity remains unmet.

## Scalar staging address hoisting

Two rejected prototypes on `8554bd0` precompute a lane's matrix base and stage
stride before the scalar K loop. The first applies to F32 products with normal
A storage and staging maps. The second restores A and applies the same
transformation only to F32 transposed B, preserving both its retained interior
branch and its original checked edge path. Both backends use the same template.

For the admitted K tiles, `KT` divides 256. Therefore
`floor((tid + e*256)/KT) = floor(tid/KT) + e*(256/KT)` and the remainder is
`tid % KT`. The original matrix index equals `base + e*stride + t`, including
unsigned wrapping. The base uses `tile_row` for A and `tile_col` for transposed
B. Original bounds checks, discarded element-zero reads, shared addresses,
K order, splits, barriers, subgroup guards and output arithmetic remain.
Precomputation reads only immutable parameters when K=0; no matrix read is
introduced. Dependencies and selected geometry remain unchanged.

Each proof checks **3,354,624 addresses across eleven layouts**, including
2,325,808 wrapping cases and original masked reads. These arithmetic boundary
checks do not authorize allocations beyond existing limits. Source audits
remove only the base/stride declarations and restore the original index
expressions, reproducing every ordinary shader body after whitespace/comment
normalization. All 19 CI and 43 model modules per variant are accounted for.

Each form passes **49 codegen checks, nine whole-graph smoke tests and eight
strict matmul oracle cases**, followed by 500 normal comparisons against frozen
qualified `540eccf` binaries. The established control/candidate/candidate/control
protocol uses 25 interleaved pairs per case and visit, ten warmup pairs, at least
one second of warmup and 256 workers. All 1,000 comparisons preserve matching
selections, full-state budgets, completion, one persistent dispatch and no
fallback; maximum model error remains 4.66e-9.

| Operand / case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| A, forward | 4.124 / 5.531 | 4.232 / 5.629 | 1.018 [1.010, 1.076] |
| A, training + SGD | 14.582 / 17.853 | 14.772 / 18.112 | 1.015 [0.993, 1.024] |
| Transposed B, training + SGD | 14.630 / 17.779 | 14.592 / 17.996 | 1.012 [0.997, 1.086] |
| Transposed B, CI SGD | 0.06661 / 0.09468 | 0.06637 / 0.10003 | 1.057 [1.023, 1.065] |

A hoisting slows persistent/ordinary forward **1.8% / 2.6%** and establishes no
training improvement. Transposed-B hoisting establishes no model-training gain
and slows CI SGD **5.7% raw / 5.2% normalized**. Its complete model-forward and
CI-inference shaders and compiled pipeline/resource multisets are identical
in both backends despite timing shifts, including 7.0% higher persistent CI
inference. Those shifts are not attributed to the shader change. All samples
remain; bootstrap intervals do not cover all between-visit variation.

Twenty resource records preserve 48 CI registers, 102/103 model registers and
all shared capacities. A hoisting grows model code from 107,904/267,776 to
108,288/268,416 bytes. Transposed-B model code retains the original sizes;
its CI SGD code grows from 65,408 to 66,560 bytes. These totals do not establish
native instruction identities, load-address latency or the cause of the timing
changes. No fresh SDK capture or full retained-change qualification is claimed.

Both probes are restored exactly to runtime `540eccf`, locks match the qualified
copy, both forks are clean, release examples are rebuilt, and all nine restored
smoke tests pass. Heavy jobs use separate verified 16/20/1 GiB cgroups, GPU work
is serialized, builds and heavy analyses do not overlap measurements, and all
artifacts stay on local `/x`.

Records are under `target/scalar-{a,bt}-address/`. The manifest
`target/megakernel-results/scalar-stage-address/summary.json` preserves **1,000
comparisons, 20 resource records, 24 verified measurement scopes and 667
independently verified artifact hashes**, SHA-256
`9aa3c53d539a513b4ecc9bb7b68635b6f6aa574ff50686defa64b80908fdd4a3`.
The independent audit log is `target/scalar-a-address/verify.log`.
Baseline parity remains unmet.

## Refreshed worker clock diagnostics

The older clock trace predates retained geometry, staging and dependency
changes. This refresh instruments qualified runtime `540eccf`, with the
published Blade and Naga revisions as its control. A temporary Blade patch
queries and enables the device-clock features and replaces four explicit
marker-call sites with `OpReadClockKHR` after normal Naga validation. An
independent audit validates all **30 original and transformed module pairs**
and reverses the added capabilities, extension, scope constant and clock
instructions to reproduce each original binary byte-for-byte. No Naga
language, IR, memory-semantics or validation change is introduced.

The first lane records worker entry/exit, each node/worker's first handler
start and last handler end, and summed handler intervals. Existing histogram
counts remain intact. The measured handler interval starts after parameter
broadcast and acquire fencing, and ends after the payload's publication fence
and workgroup rendezvous. It excludes ready-queue search, parameter staging,
successor publication and diagnostic bookkeeping. All recorded task counts
match the logical grids; intervals fit inside their worker lifetimes. The
current partial-batch completion clamp remains unchanged.

The probe passes **49 codegen checks and nine whole-graph smoke tests**. Frozen
qualified and instrumented binaries run in control/probe/probe/control order,
with ten paired samples per case/visit, ten minimum warmup pairs and at least
one second of warmup. All **200 diagnostic comparisons** preserve matching
implementations, full-state budgets, completion, one persistent dispatch and
no fallback. Maximum model error is 4.66e-9. These are instrumented diagnostics,
not new normal-performance qualification or a retained optimization.

| Case | Qualified / probe persistent GPU interval, ms | Probe change | Registers, qualified / probe |
|---|---:|---:|---:|
| CI inference | 0.03106 / 0.03408 | +9.7% | 48 / 48 |
| CI backward | 0.06787 / 0.07632 | +12.4% | 48 / 48 |
| CI SGD | 0.07226 / 0.08112 | +12.3% | 48 / 48 |
| Model forward | 5.327 / 5.062 | −5.0% | 102 / 112 |
| Model training + SGD | 17.378 / 17.434 | +0.3% | 103 / 111 |

These GPU intervals include the queue reset. Ordinary forward GPU time also
falls 4.1% between the visits; no speedup is attributed to instrumentation.
Persistent model wall medians change from 5.573/18.164 to 5.487/18.011 ms.
All twenty resource records preserve shared capacity, but model code grows
from 107,904/267,776 to 110,208/270,720 bytes. Reported scheduler storage grows
from 337,036/1,170,776 to 2,237,580/7,654,744 bytes. The probe therefore changes
more than clock-read instruction count and is particularly intrusive for CI.

Both model-forward traces have **168 long-lived workers** and **88 entering
in the final 5% of the dispatch envelope**. Summed measured handler intervals
occupy 33.62–33.71% of summed worker lifetimes, equivalent to about 56.5 busy
workers averaged over that envelope. Training has the same lifetime pattern,
with 45.63–50.18% of worker time inside measured handler regions, averaging
76.7–84.3 busy workers. Narrow ready work and time outside the handler coexist;
the complement is not all avoidable scheduling work, and these fractions are
**not elapsed-time savings**. In the much smaller CI graphs the corresponding
fractions are below 1%, with 256 long-lived workers.

Raw model envelopes are 5,108,128–5,122,752 ticks forward and
15,635,040–17,195,936 ticks for complete training. Their ratios to dispatch
timestamp nanoseconds are 0.99917–0.99982 on these captures. This observation
does not establish a universal tick rate. CI ratios are only 0.808–0.957,
where dispatch boundaries matter more. The
[clock extension](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_shader_clock.html)
does not guarantee a fixed rate or a compiler code-motion barrier.

Fresh ordinary dispatch profiles preserve bitwise exact complete state during
profiling and match the paired benchmark's selected implementation keys.
The initial invocation omitted GPU timing and failed with `MissingGpuTimings`;
its log is preserved separately from the successful timing-enabled invocation.
The ordinary backward profile excludes SGD, while the training clock trace
includes it. Representative current shapes are:

| Phase / shape | Nodes | Ordinary median interval, µs | Persistent median node span, µs | Ordinary registers / shared bytes |
|---|---:|---:|---:|---:|
| Forward `MatMul[50x4096x720]` | 16 | 77.84 | 110.04 | 55 / 2,176 |
| Forward `FusedMatMulAdd[50x720x2048]` | 16 | 63.49 | 69.37 | 40 / 6,400 |
| Forward `FlashAttentionCoop[50]` | 16 | 9.14 | 23.49 | 120 / 9,216 |
| Training `MatMulBT[50x720x4096]` | 16 | 149.49 | 183.99 | 48 / 9,472 |
| Training `MatMulBT[50x2048x720]` | 16 | 50.18 | 83.42 | 98 / 16,768 |
| Training attention gradient dKV, q=50/kv=0 | 8 | 100.35 | 143.61 | 35 / 512 |

Persistent spans use each capture's timestamp/envelope scale. Node spans
overlap, and ordinary pass-start intervals have different boundaries from
instrumented handler clocks; the columns do not establish exact kernel ratios
or additive elapsed savings. The wide forward matrix product, transposed
training products and attention gradients are concrete targets for examining
the cost of combining handlers with different resource requirements. The
four SGD nodes also vary substantially between the two training traces;
their overlapping spans sum to 3.81–5.30 million ticks, which is not a measured
optimizer-only elapsed interval.

The probe is preserved locally and removed from the working runtime. Runtime
sources, Blade and `Cargo.lock` are restored exactly, both forks are clean,
release examples are rebuilt and all nine restored smoke tests pass. All
heavy jobs use verified separate 16/20/1 GiB cgroups; GPU work is serialized,
and builds and heavy analysis do not overlap measurements. Artifacts remain
on local `/x`.

The record under `target/clock-current/` includes ten benchmark traces, twenty
resource records, current ordinary profiles and 162 independently verified
artifact hashes. Twenty job cgroups are recorded, including nine successful
measurement scopes and the failed profile setup. The manifest is
`target/megakernel-results/clock-current/summary.json`, SHA-256
`5ad0b2412d5460e317a575b145e7f59f26797a0ddafe2ddb9a19d2c5ca883ee8`.
No fresh SDK capture or full retained-change qualification is claimed.
The qualified runtime and performance claims remain unchanged; parity is
still unmet.

## Contiguous cooperative prologue staging

The refreshed trace identifies the forward `MatMul[50x4096x720]` as a native
cooperative kernel with a fused normalization prologue. Its A staging still
uses four scalar passes, whereas the plain cooperative path stages adjacent
K elements together. At `9ff1b50`, eligible prologue lanes stage four adjacent
K values, apply the original scalar factor chain to each value and write the
same shared cells before the existing barrier. The scalar storage binding is
retained: this introduces no vector-alignment requirement or assertion that
the driver emits a particular vector-load instruction.

The change applies to normal-layout, 16-wide native cooperative tiles with
a prologue, F32 weight storage and uncompensated accumulation. Other layouts,
weight formats, compensated and typed products retain their original paths.
Full four-element groups use direct checked loads; partial groups check each
element and preserve positive-zero padding. The original PerRow cache and
PerKCol indices, scalar multiplication order, matrix-input conversion, B
staging, cooperative accumulation order, stores, geometry, packing, barriers,
dependencies and queue protocol remain unchanged in both execution modes.

The ownership audit checks **3,276,800 shared cells in 12,800 stages**, including
zero dimensions, partial rows and K tails. It proves unique staging writes and
the same A/factor index tuple or zero value for every shared cell. The expanded
packed-matrix fixture adds K=717/718/719, alongside K=720, with changed inputs,
poisoned storage, 1/3/64/4096 workers and repeated scratch reuse. It requires
bitwise equality with ordinary execution and validates completion. The existing
prologue regression also passes.

Two independent control/candidate/candidate/control sequences preserve **1,000
normal comparisons**, with 25 AB/BA pairs per case and visit, ten minimum
warmup pairs, at least one second of warmup and 256 workers. Every comparison
retains matching implementations, complete state budgets, completion, one
persistent dispatch and no fallback; maximum model error remains 4.66e-9.

| Forward sequence | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate/control, bootstrap 95% interval | Paired-ratio candidate/control, 95% interval |
|---|---:|---:|---:|---:|
| First | 4.131 / 5.506 | 4.085 / 5.306 | 0.964 [0.957, 0.971] | 0.969 [0.928, 0.980] |
| Independent repeat | 4.108 / 5.531 | 4.088 / 5.273 | 0.953 [0.945, 0.969] | 0.953 [0.938, 0.992] |

The repeated completed-step measurements are:

| Case | Ordinary / persistent ms | Median paired persistent / ordinary |
|---|---:|---:|
| CI inference | 0.05723 / 0.07417 | 1.283× |
| CI backward | 0.06467 / 0.09346 | 1.445× |
| CI SGD | 0.06684 / 0.09474 | 1.422× |
| Model forward | 4.088 / 5.273 | 1.293× |
| Model training + SGD | 14.631 / 17.738 | 1.220× |

Training's first raw persistent median rises from 17.094 to 18.069 ms, while
the repeat changes from 17.834 to 17.738 ms. Their normalized ratios are
0.997 [0.975, 1.004] and 0.997 [0.900, 1.009]. CI inference also shifts between
visits. These changes are not attributed to the shader modification: all
nineteen CI shader files and the complete training shader are byte-identical,
and both backends' compiled pipeline/resource multisets match in these cases.
Only two of 43 model source files change: the prologue module and the combined
forward shader. No CI or training improvement is claimed. All samples remain;
bootstrap intervals do not cover every source of between-visit variation.

Ten separate resource records retain all shared capacities and scheduler sizes.
Persistent forward registers fall from 102 to 100, while code grows from
107,904 to 111,360 bytes. Training remains at 103 registers, 29,232 shared bytes
and 267,776 code bytes. The new forward binary remains roughly 29% slower than
ordinary execution in the repeat; complete training remains about 22% slower.

Four forward SDK backend captures, with four sequential metric groups each,
pass adapter, activity, exit and full-state checks. All four actual state files
are independently hashed and bitwise identical, SHA-256
`4e5d66e6ce5e95142b848651d316b4df64855a9d85669499cabc3a81089be629`.
Ordinary replay changes from 3.742 to 3.600 ms; persistent replay improves
**6.6%**, from 5.046 to 4.714 ms. Estimated persistent integer-pipe instructions
fall **18.3%**, warp instructions fall **5.2%**, and FP32 activity remains nearly
flat. Global-load sectors rise **46.2%**, shared wavefronts rise **2.2%**, and
persistent local load/store traffic stays zero. Thus this result supports less
integer work as a contributor; it does not establish reduced memory traffic
or identify particular native instructions. Ordinary local traffic increases
from about 79,500 to 113,000 sectors per replay despite its faster replay.

Barrier waiting remains about 59.0/59.2% of active warp cycles, not elapsed
time. Across metric groups, maximum clock spread is 0.376% and instruction-rate
spread is 1.205%. Counter rates multiplied by replay duration are estimates;
resident SDK replay does not replace the completed-step measurements.

Full qualification passes **667 library tests** (eight existing ignores),
**168 strict whole-graph oracle tests**, **50 matrix regressions in their
original mode**, and all **nine expanded whole-graph smoke tests**. The 49
codegen checks, prologue regression, formatting, strict all-target/all-feature
Clippy and Metal compilation with `models,gguf,profiler` also pass. Rebuilt
benchmark binaries match the measured candidates byte-for-byte. Both dependency
forks remain clean and their pins and `Cargo.lock` are unchanged. Metal GPU
execution and additional GPUs remain unqualified.

Records are under `target/coop-prologue-stage/` and
`target/coop-prologue-stage-repeat/`. The manifest
`target/megakernel-results/coop-prologue-stage/summary.json` preserves 1,000
normal comparisons, ten resource records, four SDK backend captures/sixteen
metric groups, 22 verified measurement scopes and **461 independently verified
artifact hashes**, SHA-256
`1e318dd91d7c3e92524913d6a333b5210f5acccb1913262a7a88b97f1aab60b8`.
Heavy jobs use separate verified 16/20/1 GiB cgroups, GPU work is serialized,
builds and heavy analysis do not overlap measurements, and all artifacts stay
on local `/x`. The change is retained; ordinary-baseline parity remains open.

## Column-shared cooperative A scratch probe

On October 26, 2026, a prototype based on `0caf997` gave the first column-packed
logical tile ownership of the native cooperative prologue's A scratch. It
redirected all four logical tiles' A reads to that slice and guarded every
write, including initialization. Admission was limited to normal F32-weight,
uncompensated f16-input, 16×16 cooperative prologues packed on grid axis 1.
The original barriers, B storage, arithmetic, completion and scratch allocation
were retained. No workgroup variables were added.

The prototype passes 49 codegen checks, nine whole-graph smoke tests and the
prologue regression. The existing packed fixture covers K tails, partial rows,
poisoning, changed-input replays and 1/3/64/4096 workers. **All 500 normal paired
comparisons pass** matching arithmetic, complete-state budgets, completion,
one persistent dispatch and no fallback. The maximum model error is 4.66e-9.
The interleaved control/candidate/candidate/control sequence uses 25 pairs per
case per visit, 256 workers and the established warmup and timing boundaries.

| Case | Control persistent ms | Candidate persistent ms | Candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| CI inference | 0.06976 | 0.06934 | 0.994 [0.979, 1.005] |
| CI backward | 0.09408 | 0.09364 | 0.995 [0.990, 1.005] |
| CI SGD | 0.09475 | 0.09472 | 1.000 [0.992, 1.005] |
| Model forward | 5.182 | 5.135 | 0.991 [0.928, 1.073] |
| Model training + SGD | 18.043 | 18.069 | 1.001 [0.987, 1.018] |

Forward's ordinary medians are 3.979/3.978 ms; the median paired
persistent/ordinary ratios are 1.346/1.342. Their candidate/control ratio is
0.997 [0.967, 1.003]. Training ordinary medians are 14.649/14.706 ms, and its
normalized ratio is 0.980 [0.960, 1.004]. Between-visit variation is substantial;
no completed-step improvement is established.

Only the combined forward shader changes: all 19 CI sources and 42 of 43 model
sources match exactly. Both backends' CI/training pipeline identities and
resources match, as do every ordinary pipeline's statistics. Forward remains
at 100 registers and 20,880 shared bytes; code falls from 111,360 to 110,464
bytes. Scheduler sizes are unchanged. An exact inverse of the generated-source
edits reconstructs the entire control shader: the changes are the ownership
condition, 14 guarded stores including resets, and the two A arrays' indices.
Scratch declarations and barriers are identical.

Four SDK backend captures with sixteen sequential metric groups preserve
bitwise-identical complete forward state. Persistent replay changes from
4.697 to 4.735 ms (**+0.8%**), while ordinary changes from 3.596 to 3.606 ms.
Estimated persistent integer-pipe activity falls 7.5%, FP32 activity 3.8%,
warp instructions 1.3% and shared wavefronts 1.8%. Global-load sectors change
only −0.06%; persistent local traffic remains zero. The emitted WGSL still
evaluates A and factor loads outside the ownership guard, consistent with the
absence of a global-load reduction. Native instruction attribution is not
available, so the counter changes do not prove which loads the driver retains.

Barrier waiting changes from 59.44% to 59.71% of active warp cycles, not elapsed
time. Maximum cross-group clock spread is 0.421% and instruction-rate spread
0.997%. All four actual 588,138,052-byte state files independently hash to
`4e5d66e6ce5e95142b848651d316b4df64855a9d85669499cabc3a81089be629`.
These rate-times-replay estimates do not replace the completed-step gate.

The prototype is **rejected; the qualified runtime is exactly restored**. Nine restored whole-graph
tests and formatting pass; rebuilt benchmark binaries match the qualified
controls byte-for-byte. Runtime `9ff1b50`, both fork checkouts and dependency
pins remain unchanged. Full candidate qualification and mixed packing widths
sharing a handler were not pursued after the performance gate failed.
Guarding complete staging regions or reducing scratch capacity remains
unmeasured; this result only rejects the tested store-guard prototype.

Artifacts remain on local `/x` under `target/column-shared-a/`. The manifest
`target/megakernel-results/column-shared-a/summary.json` records 500 normal and
ten diagnostic comparisons, 14 verified measurement scopes, 28 recorded job
scopes and **416 independently verified artifact hashes**. Its SHA-256 is
`140932375f74e924d0b76470c8fd732360108f0dfe5dd4ef1ef9ef1637624476`.
It retains the initial compile error, rejected patch, source audit, captures,
restoration checks and qualification limits. Every heavy job used its own
verified 16/20/1 GiB cgroup; GPU jobs were serialized, with no competing builds
or heavy analyses during measurement. Ordinary-baseline parity remains open.

## Complete A staging-region guard

A second probe, based on `780cb7c`, guards both complete A staging blocks,
including addresses, bounds and A/factor loads. It retains the prior ownership
rule and scratch allocation. Admission checks that each block only publishes
the named A scratch, contains no barriers or control flow leaving the region,
and does not write local values read outside the block. Only a single-function
shader with scalar boolean/integer staging locals is admitted. Other shader
selections retain their original paths.

The captured forward shader differs from the preceding store-only prototype
by exactly two block-to-condition changes. An independent inverse reconstructs
the entire qualified shader. Four-tile packing, declarations, barriers,
arithmetic and completion are unchanged. All 19 CI sources and 42 of 43 model
sources match the control. Ordinary and CI/training pipeline identities and
resources are identical. Persistent forward remains at 100 registers and
20,880 shared bytes; code changes from 111,360 to 109,952 bytes.

All 49 codegen checks, nine whole-graph GPU tests and the prologue regression
pass. Another **500 normal paired comparisons** preserve matching selected
arithmetic, full-state budgets, one persistent dispatch, completion and no
fallback. Maximum error remains 4.66e-9. Each of the five cases has two
25-pair visits per binary in control/candidate/candidate/control order, with
256 workers and the existing warmup and completed-step boundaries.

| Case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| CI inference | 0.05234 / 0.06808 | 0.05210 / 0.06955 | 1.022 [0.998, 1.061] |
| CI backward | 0.06550 / 0.09406 | 0.06463 / 0.09352 | 0.994 [0.987, 1.002] |
| CI SGD | 0.06674 / 0.09494 | 0.06699 / 0.09500 | 1.001 [0.986, 1.012] |
| Model forward | 4.084 / 5.285 | 4.062 / 5.256 | 0.995 [0.986, 1.003] |
| Model training + SGD | 14.615 / 17.760 | 14.605 / 17.975 | 1.012 [0.999, 1.104] |

Forward's median paired ratio changes from 1.297 to 1.295; its normalized
candidate/control ratio is 0.998 [0.982, 1.033]. Training's normalized ratio
is 0.999 [0.980, 1.102]. Unchanged CI backward has a normalized ratio of
1.013 [1.003, 1.028], illustrating variation despite identical shaders.
No completed-step improvement is established.

Four SDK captures with sixteen metric groups confirm the intended reduction
in activity. Estimated persistent global-load sectors fall **35.6%**, integer
activity **17.7%**, warp instructions **9.3%**, FP32 activity 3.6% and shared
wavefronts 4.8%. Local load/store traffic remains zero. Yet persistent replay
changes from **4.710 to 4.724 ms**, while ordinary changes from 3.599 to
3.606 ms. Barrier waiting rises from 59.51% to 62.03% of active warp cycles;
this fraction does not establish an elapsed synchronization cost. Maximum
cross-group clock spread is 0.431% and instruction-rate spread 1.807%
(0.497% for persistent captures). All four actual forward state files remain
bitwise identical to the preceding probe's recorded hash. Rate-times-replay
estimates do not replace the normal performance gate or identify native
instructions.

The prototype is rejected. The qualified source is restored exactly, nine
restored whole-graph tests and formatting pass, and rebuilt benchmark binaries
match the controls byte-for-byte. Runtime `9ff1b50`, dependency pins and both
forks remain unchanged. Full candidate retention qualification is not claimed.
Removing duplicate A staging activity alone does not explain the remaining
latency gap. Investigate work distribution and packing width next, separating
changes in physical claims from scratch/resource changes; smaller scratch
capacity remains unmeasured.

Local artifacts are under `target/column-shared-a-regions/`. Manifest
`target/megakernel-results/column-shared-a-regions/summary.json` records 500
normal/ten diagnostic comparisons, 14 verified measurement scopes, 27 recorded
job scopes and **414 independently verified artifact hashes**, SHA-256
`2e67888f79587648f3fe0dad8b146e74d5dc46bcf21ce9468dd85686317e3178`.
All heavy jobs run in separate verified 16/20/1 GiB cgroups; GPU jobs serialize,
builds and heavy analysis do not overlap measurement, and artifacts stay on
local `/x`. Baseline parity remains unmet.

## Two-tile cooperative packing and inactive staging

Two probes based on `76fda5c` test the wide cooperative prologue's work
distribution. The first limits eligible F32, uncompensated, 16-wide native
prologues to two tiles per claim. Axis ranking and the original four-slice
scratch capacity stay fixed. The forward grid `[2,128,1]` still contains 256
logical tiles, now in 128 claims instead of 64. The physical worker remains
256 invocations. This leaves half its logical groups inactive for that handler;
their masked stores and products still allow redundant staging loads.

The second probe guards all four complete A/B staging blocks with the existing
active-lane condition, using the preceding side-effect and local-state escape
checks. Every physical invocation still reaches the original barriers. The
first emitted shader differs from the control by exactly one tile-group clamp;
the second adds exactly four block-to-condition changes. Reversing those edits
reconstructs the entire qualified shader. Enumeration covers 264 grids and
309,636 active tile coordinates exactly once. Claim counts follow runtime batch
selection; aggregate worker reports do not establish per-node SM utilization.

Both prototypes pass 49 codegen checks, nine whole-graph tests and the prologue
regression. All **1,250 normal paired comparisons** preserve full-state budgets,
matching selected arithmetic, one persistent dispatch, completion and no
fallback. Maximum error is 4.66e-9. The first sequence has two 25-pair visits
per binary and case in control/pack2/pack2/control order. The second uses
control/pack2/guarded/guarded/pack2/control, also 25 pairs per visit across all
five cases. Warmup, 256 workers and completed-step boundaries stay unchanged.

In the first sequence, forward ordinary/persistent medians are
4.021/5.222 ms for the control and 4.029/5.220 ms for two-tile packing.
Persistent candidate/control is 1.000 [0.985, 1.066]; the normalized paired
ratio is 0.965 [0.960, 1.011]. Training and CI intervals also include no change.
The three-way sequence gives:

| Case | Control ordinary / persistent ms | Two-tile ordinary / persistent ms | Guarded ordinary / persistent ms | Guarded/control paired-ratio ratio, bootstrap 95% interval |
|---|---:|---:|---:|---:|
| CI inference | 0.05265 / 0.06879 | 0.05725 / 0.07451 | 0.05202 / 0.06911 | 1.028 [0.983, 1.072] |
| CI backward | 0.06463 / 0.09341 | 0.06504 / 0.09410 | 0.06463 / 0.09359 | 0.994 [0.985, 1.007] |
| CI SGD | 0.06661 / 0.09474 | 0.06684 / 0.09510 | 0.06655 / 0.09494 | 0.998 [0.990, 1.012] |
| Model forward | 4.105 / 5.311 | 4.104 / 5.257 | 3.957 / 4.986 | 1.009 [0.973, 1.042] |
| Model training + SGD | 14.848 / 18.054 | 14.637 / 18.118 | 14.817 / 18.024 | 1.002 [0.984, 1.022] |

Guarded forward's raw persistent median falls 6.1%, but ordinary also becomes
faster during those visits. Its paired ratio changes from 1.293 to 1.305, with
no established normalized improvement. Unchanged CI inference also varies
between visits. All samples remain; the pooled bootstrap does not account for
every source of between-visit variation. Neither prototype is retained on this
evidence, and full candidate retention qualification was not pursued.

All 19 CI sources and 42 of 43 model sources match the control. Ordinary and
CI/training pipeline identities and resources are unchanged. Forward remains
at 100 registers and 20,880 shared bytes; code is 111,360 bytes for control and
two-tile packing, and 110,336 bytes with guards. Scheduler sizes are unchanged.

Ten SDK backend captures across both studies contain forty metric groups. In
the first, two-tile packing improves persistent replay 1.5% but raises estimated
global-load sectors 59.9% and integer activity 18.3%. The three-way capture
reproduces this excess and shows that the complete guards remove it:

| SDK forward diagnostic | Control | Two-tile | Two-tile with guards |
|---|---:|---:|---:|
| Ordinary replay ms | 3.601 | 3.602 | 3.604 |
| Persistent replay ms | 4.708 | 4.649 | 4.663 |
| Persistent global-load sectors / control | 1.000 | 1.603 | 0.997 |
| Persistent integer activity / control | 1.000 | 1.186 | 0.976 |
| Persistent warp instructions / control | 1.000 | 1.143 | 0.989 |
| Persistent shared wavefronts / control | 1.000 | 1.103 | 1.062 |
| Barrier waiting, percent of active warp cycles | 59.27 | 52.81 | 60.56 |

Persistent local load/store traffic stays zero. All ten actual forward state
files are bitwise identical. The three-way capture's maximum clock spread is
0.288% and instruction-rate spread 1.134%. Guarded replay improves only 1.0%
against control and is 0.3% slower than unguarded two-tile packing. These
rate-times-replay estimates neither replace the completed-step gate nor turn
barrier-cycle shares into elapsed costs. Reducing inactive staging traffic does
not explain the remaining latency gap by itself.

The qualified source is restored exactly. Nine restored whole-graph tests and
formatting pass; rebuilt benchmarks match the qualified controls byte-for-byte.
Runtime `9ff1b50`, both clean fork checkouts and dependency pins stay unchanged.
Investigate cooperative geometry that supplies more claims while keeping the
physical groups active; scratch-capacity changes remain unmeasured.

Artifacts are under local `target/coop-prologue-pack2/` and
`target/coop-prologue-pack2-active/`. The combined manifest
`target/megakernel-results/coop-prologue-pack2/summary.json` records 1,250 normal
and 25 diagnostic comparisons, 35 measurement scopes, 59 recorded job scopes
and **1,003 independently verified artifact hashes**, SHA-256
`be84f6e06208525733a8398c318c6249f36b4d8c89104e77532b710a65ae953e`.
The interrupted session's final SDK capture completed successfully; its exit
receipt and actual files were checked before continuing. A fresh launcher
check verified separate 16/20/1 GiB cgroup limits and the shared GPU lock, with
no surviving measurement jobs. Heavy jobs use those verified scopes, GPU jobs
serialize, and builds or heavy analyses do not overlap measurement. All
artifacts remain on `/x`. Baseline parity remains unmet.

## Narrow native cooperative prologue tiles

Runtime `d01534a` uses a 32×16 output tile instead of 32×32 for eligible normal,
F32-weight, uncompensated f16-input prologues with 16×16 hardware tiles. The
logical workgroup remains 64 lanes; its two 32-lane subgroups each own one
16×16 output. The wider-subgroup path still computes the second row. Epilogues,
horizontal copies, add products and other arithmetic selections retain their
existing geometry. Both execution modes use the common selector.

The measured wide prologue changes from grid `[2,128,1]` to `[2,256,1]`.
Four logical tiles still share a 256-lane physical worker, so the available
claims rise from 64 to 128 while all four groups remain active. Sixteen model
prologue nodes select this geometry. Total graph tasks rise from 17,242 to
21,338; the 154 operations and 337,036 scheduler bytes remain unchanged.
Claim counts follow runtime batching; they are not measured per-node SM usage.

The generator removes the second B staging region, its cooperative load, two
accumulators, two multiply-add expressions and two stores. It preserves A/factor
staging, K=16 iteration order, conversion and every barrier. An independent
transformation matches the control's remaining executable tokens, ignoring
comments and whitespace; enumeration covers 168 geometries and 9,208,320
outputs exactly once. This changes geometry and physical scratch together,
so their individual contributions are not isolated.

Fusion occurs after initial geometry selection. An initial smoke failure caught
the stale grid: both backends left half the output unwritten. The selector now
refreshes fused grids through the common geometry function. Grids that would
exceed portable limits retain the original wide tile and fusion. Narrow
eligibility participates in internal cache keys and public pipeline labels;
the original public code-generation entry point retains its wide geometry.
The preserved failure and subsequent regression check document this boundary.

Four control/candidate/candidate/control sequences preserve **2,000 normal
comparisons**, with 25 pairs per case and visit, ten minimum warmup pairs,
at least one second of warmup and 256 workers. All state budgets, completion,
matching arithmetic within each binary, one persistent dispatch and no fallback
pass; maximum model error remains 4.66e-9. Across control/candidate binaries,
only the declared `:narrow-eligible` reporting suffix is normalized.

| Forward sequence | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate/control, bootstrap 95% interval | Paired-ratio candidate/control, 95% interval |
|---|---:|---:|---:|---:|
| Prototype | 3.985 / 5.102 | 3.794 / 4.923 | 0.965 [0.938, 1.015] | 0.979 [0.967, 1.033] |
| Independent repeat | 4.064 / 5.229 | 3.724 / 4.918 | 0.941 [0.925, 0.956] | 1.041 [0.989, 1.062] |
| Legality/API hardening | 4.007 / 5.217 | 3.832 / 4.975 | 0.954 [0.944, 1.022] | 1.007 [0.978, 1.032] |
| Final reporting fix | 4.014 / 5.229 | 3.795 / 4.894 | 0.936 [0.884, 0.957] | 1.012 [0.968, 1.050] |

The repeat and final sequences establish lower absolute forward latency under
their pooled bootstrap; the other two intervals include no change. Ordinary
also becomes faster. All four normalized intervals include no change, so the
result does not establish progress in closing the relative gap. All visits are
retained; pooled intervals do not capture every source of between-visit variation.

| Final candidate case | Ordinary / persistent ms | Median paired persistent / ordinary |
|---|---:|---:|
| CI inference | 0.05180 / 0.06684 | 1.293× |
| CI backward | 0.06493 / 0.09344 | 1.433× |
| CI SGD | 0.06672 / 0.09508 | 1.424× |
| Model forward | 3.795 / 4.894 | 1.322× |
| Model training + SGD | 14.624 / 17.788 | 1.217× |

All 19 CI sources and 41 of 43 model sources match the qualified control.
Only the ordinary prologue and combined forward shader change. CI and training
pipeline/resource multisets remain identical. Training's raw persistent change
across the four sequences is −0.1%, +1.3%, +0.3% and −1.5%; every interval includes
no change. Final CI SGD rises 0.5% in raw timing, but its normalized interval
includes no change; CI inference also varies. No CI or training gain is claimed.
Bridges from the prototype through hardening and the reporting fix prove exact
shader bytes, resources and execution metadata, allowing only the label suffix.

Persistent forward remains at 100 registers; shared memory falls from 20,880
to 18,832 bytes and code from 111,360 to 107,648 bytes. The one changed ordinary
executable falls from 48 to 40 registers, 2,432 to 1,664 shared bytes and 59,776
to 49,536 code bytes. Training stays at 103 registers, 29,232 shared bytes and
267,776 code bytes.

Eight SDK backend captures across prototype and final binaries contain 32
sequential metric groups. Persistent replay improves **6.0% in both captures**:
4.706→4.422 ms initially and 4.699→4.415 ms finally. Ordinary replay improves
7.4% in both, finally 3.602→3.335 ms. All eight actual forward state files are
independently hashed and bitwise identical, SHA-256
`4e5d66e6ce5e95142b848651d316b4df64855a9d85669499cabc3a81089be629`.

Final estimated persistent integer activity rises 22.9%, warp instructions
9.9%, global-load sectors 47.2%, shared wavefronts 8.3% and FP32 activity 5.1%.
Local load/store traffic remains zero. Barrier waiting falls from 59.31% to
54.20% of active warp cycles, not elapsed time. Active compute warps remain
near 32.5% of peak, so an average persistent occupancy gain is not demonstrated.
The data support faster replay despite increased traffic and instructions;
they do not isolate work distribution, scratch footprint or native instructions
as the cause. Final maximum cross-group clock spread is 0.392% and instruction
rate spread 1.388%. Counter-rate-times-replay estimates do not replace the
completed-step gate.

Full qualification passes **668 library tests** (eight existing ignores),
**168 strict oracle tests**, **50 matrix regressions**, and all **nine expanded
whole-graph smoke tests**. The packed fixture covers mixed two/four-tile claims
sharing a handler, K tails, partial rows, poisoned scratch, repeated updates and
1/3/64/4096 workers. The 49 focused codegen checks, prologue regression, format,
strict all-target/all-feature Clippy and Metal compilation also pass. The final
label fix reruns the library, smoke, prologue, Clippy, Metal and formatting gates;
exact shader/resource bridges connect it to the full oracle/matrix qualification.
An earlier library fixture failure and its corrected dimensions are preserved.
The committed source and current binaries match the frozen final candidate.
Both forks stay clean; dependency pins and lock remain unchanged.

Records stay on local `/x` under `target/coop-prologue-narrow{,-repeat,-qualified,-final}/`.
The manifest `target/megakernel-results/coop-prologue-narrow/summary.json` preserves
2,000 normal and 30 diagnostic comparisons, eight SDK captures / 32 metric groups,
48 measurement scopes, 94 recorded job scopes and **1,216 independently verified
artifact hashes**, SHA-256
`3f3de22e898c1f97627a590e07db2a718f232b6106e4e22b92a74ec9d0e11351`.
Recovered jobs had completed successfully; no surviving heavy jobs were found.
Every heavy job uses a separate verified 16/20/1 GiB cgroup, GPU work holds the
shared lock, and builds/copies/heavy analysis do not overlap measurements.
Retain this change for its reproduced absolute forward gain. Baseline parity,
Metal GPU execution and additional GPU qualification remain open.

## Cooperative prologue scratch-capacity control

A diagnostic based on `189ab53` appends 1,024 unused f16 elements to the last
existing f16 scratch bank in each combined shader. Admission accounting uses
the same larger capacity. No array, initialization store, handler, index,
geometry, packing rule, queue operation or fence is added. This restores
forward's shared allocation from 18,832 to 20,880 bytes while keeping the
retained narrow tile. Training's allocation grows from 29,232 to 31,280 bytes.

Independent reversal of that one capacity declaration reproduces each entire
control shader exactly. All 19 CI sources and 41 of 43 model sources are
unchanged. Ordinary pipelines and other cached pipeline/resource multisets
match. Persistent registers remain 100/103 and code sizes 107,648/267,776 bytes
for forward/training. The driver reports exactly 2,048 additional shared bytes
in each. Source identity changes in appended pipeline metadata are accounted
for explicitly; selected implementations, geometry, tasks, scheduler bytes
and execution metadata otherwise match.

The control/padded/padded/control sequence preserves **500 normal comparisons**,
25 pairs per case and visit, ten minimum warmup pairs, at least one warmup
second and 256 workers. All state budgets, completion, matching arithmetic,
one persistent dispatch and no fallback pass. Maximum model error is 4.66e-9.

| Case | Control ordinary / persistent ms | Padded ordinary / persistent ms | Persistent padded/control, bootstrap 95% interval | Paired-ratio padded/control, 95% interval |
|---|---:|---:|---:|---:|
| CI inference | 0.05143 / 0.06643 | 0.05208 / 0.06957 | 1.047 [1.004, 1.061] | 1.033 [0.930, 1.059] |
| CI backward | 0.06493 / 0.09390 | 0.06493 / 0.09391 | 1.000 [0.992, 1.007] | 0.998 [0.986, 1.009] |
| CI SGD | 0.06660 / 0.09484 | 0.06647 / 0.09500 | 1.002 [0.997, 1.032] | 1.007 [0.999, 1.035] |
| Model forward | 3.829 / 4.990 | 3.832 / 4.976 | 0.997 [0.992, 1.007] | 0.991 [0.957, 1.017] |
| Model training + SGD | 14.720 / 17.998 | 14.608 / 17.827 | 0.991 [0.980, 1.007] | 1.001 [0.909, 1.073] |

Neither model phase establishes a timing change. CI inference varies despite
identical shader sources and resources, and its normalized interval includes
no change. All samples remain; pooled intervals do not capture all between-visit
variation. The forward result does not reproduce the roughly 6% loss that would
be expected if the previous gain came entirely from the 2 KiB capacity reduction.

Four forward SDK backend captures contain sixteen metric groups and preserve
bitwise-identical actual state files. Persistent replay changes **4.411→4.428 ms
(+0.4%)**, while ordinary is 3.334→3.338 ms. Estimated persistent warp/integer
instructions, global-load sectors and shared wavefronts all change by about
+0.06%; local traffic stays zero. Active compute warps remain **32.50%/32.47%**
of peak, and barrier waiting is **54.14%/54.00%** of active warp cycles. Maximum
cross-group clock spread is 0.416% and instruction-rate spread 0.437%.
These estimates and cycle shares do not establish native instruction identity
or elapsed-time attribution.

This control weakens shared capacity as an explanation for the narrow tile's
gain and gives little reason to prioritize another small capacity reduction.
It does not isolate work distribution from the changed cooperative body, or
prove that all scratch reductions are ineffective. Driver bounds handling,
initialization and allocation can respond to array capacity even when source
indices and reported code size stay fixed.

The diagnostic passes 49 codegen checks, nine whole-graph smoke tests and the
prologue regression. It is removed exactly; nine restored smoke tests and
formatting pass, and rebuilt benchmarks match the qualified `d01534a` controls
byte-for-byte. Both forks and dependency pins/lock remain unchanged. An initial
audit's incorrect assumption about cached pipeline count is preserved alongside
the corrected audit, which also verifies every unchanged cached pipeline.
No full candidate-retention qualification is claimed for this diagnostic.

Records are local under `target/prologue-scratch-control/`. The manifest
`target/megakernel-results/prologue-scratch-control/summary.json` preserves
500 normal/ten diagnostic comparisons, four SDK captures/sixteen groups,
14 measurement scopes, 27 recorded job scopes and **426 independently verified
artifact hashes**, SHA-256
`3f77f1ecf45fccd636f980da1aad71b8a9687bcb511ed9ecbe412d2cf2006694`.
Every heavy job uses a separate verified 16/20/1 GiB cgroup, GPU jobs serialize,
and builds/analysis do not overlap measurements. The qualified runtime and
performance claims remain unchanged; baseline parity remains unmet.

The next concrete training target is the selected rowwise attention gradient.
Its current shader reserves four components per lane to cover heads up to
256 with 64 lanes, while this model's 64-wide heads need one component.
Investigate head-width specialization
through the existing common attention pipeline key and generator. Preserve
dimension masking, the exact reduction tree, query/head accumulation order,
causal/window bounds and complete training state; test partial chunk boundaries
and mixed head widths before any retention decision.

## Rowwise dKV component-count specialization

A prototype based on `58468b0` specializes rowwise dK/dV for the selected head
width through the existing attention pipeline key. Its template's `MAX_CHUNKS`
changes from four to `clamp(ceil(head_dim/64),1,4)`. The measured 64-wide heads
need one component per lane. dQ, geometry, masks, query/head accumulation order,
dual reduction tree, barriers and the queue protocol stay unchanged.

An independent audit reverses that one constant in the ordinary shader exactly.
In the combined training shader, only the corresponding constant and
`graph_handler_29_entry` change; all other source matches. Enumeration verifies
the same ordered valid components for every lane at all 256 supported head
widths, totaling 32,896 dimensions. Sixteen selected dKV labels change from
`:scalar` to `:head-dim-64:ept-None`; the audit explicitly checks this mapping.
Both backends select identical raw keys within each binary. All 19 CI sources
and 41 of 43 model sources remain identical, including complete model forward.

The ordinary dKV executable falls from **35 to 24 registers** and **6,784 to
3,712 code bytes**, retaining 512 shared bytes. Combined training code falls
from **267,776 to 263,168 bytes**, but stays at **103 registers and 29,232 shared
bytes**. Other compiled pipeline/resource multisets, logical tasks, geometry
and scheduler sizes match. Smaller code does not establish a latency gain.

The control/candidate/candidate/control sequence preserves **500 normal
comparisons**, 25 pairs per case/visit, ten minimum warmup pairs, one warmup
second and 256 workers. Every state budget, completion check, matching arithmetic,
one persistent dispatch and no-fallback check passes; maximum error is 4.66e-9.

| Case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| CI inference | 0.05743 / 0.07464 | 0.05362 / 0.06948 | 0.931 [0.888, 0.949] |
| CI backward | 0.06482 / 0.09374 | 0.06512 / 0.09433 | 1.006 [0.999, 1.014] |
| CI SGD | 0.06654 / 0.09483 | 0.06687 / 0.09533 | 1.005 [0.998, 1.040] |
| Model forward | 3.555 / 4.875 | 3.801 / 4.903 | 1.006 [0.980, 1.097] |
| Model training + SGD | 14.591 / 17.855 | 14.424 / 17.800 | 0.997 [0.975, 1.013] |

Training's normalized candidate/control paired ratio is **1.003 [0.905, 1.093]**;
no completed-step improvement is established. The unchanged CI inference
shader varies alongside ordinary execution; its normalized interval includes
no change. All samples remain, and pooled bootstrap intervals do not capture
all between-visit variation. No CI or forward gain is attributed to this edit.

Four SDK backend captures contain sixteen metric groups and include forward,
backward and zero-rate SGD for stable replay. Persistent replay changes
**15.907→15.804 ms (−0.65%)**; ordinary changes **11.921→11.734 ms (−1.57%)**.
Estimated persistent warp instructions fall 1.92% and FP32 activity 2.46%,
while integer activity rises 0.98%. Global-load sectors and shared wavefronts
remain nearly unchanged; persistent local traffic is zero. Barrier waiting is
50.77%/51.04% of active warp cycles, not elapsed time. Maximum cross-group clock
spread is 0.396% and instruction-rate spread 0.642%.

Each backend preserves bitwise-identical actual state before and after the
change. Ordinary and persistent state are compared with the unchanged benchmark
budget, not claimed bitwise equal to each other. SDK zero-rate replay and
rate-times-duration instruction estimates remain diagnostic. The small replay
change does not replace the normal completed-step acceptance gate.

Qualification passes **669 library tests** (eight ignores), all **nine whole-graph
smoke tests**, and **22 attention reference tests in each of ordinary and strict
whole-graph modes**. The reference sweep adds 63/65/129/193-wide cases with GQA,
cross/causal/window masks; a pipeline-key regression separates 64/65-wide heads.
An initial source-audit assumption omitted the global chunk constant; its
failure and corrected exact audit are preserved. No full candidate-retention
qualification is claimed.

The prototype is rejected and source/tests are restored exactly. Nine restored
smoke tests and formatting pass; rebuilt benchmarks match the qualified
`d01534a` controls byte-for-byte. Forks, dependency pins and lock stay unchanged.
The manifest `target/megakernel-results/rowwise-kv-chunks/summary.json` preserves
500 normal/ten diagnostic comparisons, four SDK captures/sixteen groups,
14 measurement scopes, 29 recorded job scopes and **430 independently verified
artifact hashes**, SHA-256
`cd48f2111b0bcea81cf47f616e4a2ba5379020b19a42087eee724b80ad3458f5`.
Raw artifacts stay under local `target/rowwise-kv-chunks/`. Every heavy job uses
a separate bounded cgroup, GPU work serializes, and builds/analysis do not
overlap measurements. Baseline parity remains unmet.

The retained combined shader runs dQ/dKV as one 64-lane logical tile in a
256-lane physical worker. Independent heads have equal query-loop bounds at
the same row, whereas different causal rows do not. The earlier broad
three-tile packing probe admitted heads by separating coordinate arguments,
but enlarged cooperative attention scratch and failed its timing gate. Test
the non-cooperative gradient opportunity separately, keeping cooperative
packing and its footprint unchanged. Audit scope/uniformity, three-tile tails,
scratch isolation and exact task coverage. An exact-order subgroup reduction
remains a separate possible experiment: the current dKV loop has eight
workgroup barriers per query/head iteration, and any replacement must preserve
the original tree and a supported fallback.

## Earlier artifact identities

All records remain on local storage at
`target/megakernel-results/whole-graph-20261025/`. The measured SmolVLA executable
has SHA-256 `fd52b1400a974c8c7a326afcc40b071310aecb61aa4cba9c5a19a64de6398637`.

```sh
cargo build --release --features models --example bench_smolvla_train --example bench_ci_latency
target/release/examples/bench_smolvla_train --whole-graph --workers 256 --warmup 10 --runs 50 --warmup-ms 1000
target/release/examples/bench_ci_latency --whole-graph --workers 256 --warmup 10 --runs 50 --warmup-ms 1000
python3 scripts/run_bounded_job.py sdk-capture --gpu -- \
  env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  python3 scripts/profile_smolvla_sdk.py --sdk /x/Tools/nsight-perf-sdk-2026.3 \
  --output target/new-whole-graph-capture --phase forward --workers 64 256 --whole-graph
```

| Record | SHA-256 |
|---|---|
| `smolvla-64.json` | `24a0aaa2065745d13eeeb6fba2b8f05e9ff32884c791b2dac8513b47146a9d0c` |
| `smolvla-256.json` | `69944be71d04a4c609a844c6538c2463d11b88eff3882b4d6f8799d5c53eaf3d` |
| `ci-256.json` | `e8c5e06ca134e7b6cd3a1ba157fe2fbd32c425fde3e7eef084c294257bab35c1` |
| `smolvla-256-diagnostics.json` | `0d650e390fc5843e818bef26d185e75f3cb8bdc0f703d282bb0e4c1616adeeaa` |
| `sdk-forward/ordinary-receipt.json` | `e209faec2241740d5c13306595bab3fb8defb3407b83d5563e61a05ed2ec28a1` |
| `sdk-forward/whole-64-receipt.json` | `d280ce9059afbe6f73229c35f6fb6605b5e7906f9af305593850effcf359eb35` |
| `sdk-forward/whole-256-receipt.json` | `a996d1fc0fc624c1502cdbc7e14074d6a0a028cffa3fadb3f9fb8c33870ca618` |


## Scalar attention head packing

Retain `8a63a0a` after full qualification and a larger completed-step repeat.
Scalar callable tiles expose direct workgroup-coordinate reads as separate
arguments, so the existing scope-aware admission check can distinguish the
uniform query coordinate from independent head coordinates. Whole-vector uses
remain conservative. Scalar packing considers widths 4/3/2/1; subgroup and
cooperative handlers retain 4/2/1. Each node chooses a divisor of its own grid,
while the shared handler allocates scratch for its largest admitted width.

The rowwise dQ/dKV templates remove only their redundant head-bound early exit:
the common selector dispatches exactly the query/KV head count on Y. Row bounds,
masks, reduction trees, query/head accumulation order and dimension guards stay
intact. The original head guard prevents packing admission even after splitting
coordinate arguments; removing it admits heads while causal row packing still
fails validation. No new shader, runtime enum, environment hook, Naga change or
Blade change is introduced.

In this model, the sixteen dQ nodes have 50×15 grids. Packing three heads changes
12,000 successful single-tile claims into 4,000 three-tile claims, preserving all
12,000 logical tiles. The five KV heads remain unpacked because no permitted
width divides five. Captured dQ/dKV cases confirm widths 3/1. The reported graph
logical tasks remain 79,824; buffers, scheduler storage, implementation keys,
geometry, completion accounting and one-dispatch execution remain unchanged.
All cooperative handler functions and cases are byte-identical to the control.

Training still uses **103 registers and 29,232 shared bytes**; compiled code
changes 267,776→267,904 bytes. Forward remains 100 registers, 18,832 shared bytes
and 107,648 code bytes. CI resource statistics are also unchanged. Ordinary dQ
registers fall 35→30, with shared/code sizes unchanged at 512/6,272 bytes; ordinary
dKV resources do not change. The ordinary source audit finds only the two head
guard removals. Separate arguments change additional scalar callable functions,
so this is not a claim that only dQ source bytes differ.

The first **500 normal comparisons** give training 17.805→17.001 ms,
candidate/control **0.9549 [0.9450, 1.0388]**. That interval includes no change.
A fresh **1,000-comparison repeat** uses fifty AB/BA pairs per case in each of
two control and two candidate visits:

| Case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate / control (95% interval) |
|---|---:|---:|---:|
| CI inference | 0.052301 / 0.069505 | 0.057171 / 0.074400 | 1.0704 (1.0533–1.0776) |
| CI backward | 0.064771 / 0.093856 | 0.065276 / 0.093705 | 0.9984 (0.9936–1.0040) |
| CI SGD | 0.066596 / 0.094966 | 0.066535 / 0.094966 | 1.0000 (0.9941–1.0074) |
| Model forward | 3.806824 / 4.958675 | 3.847089 / 5.008921 | 1.0101 (1.0045–1.0225) |
| Model training + SGD | 14.625019 / 17.881634 | 14.729420 / 17.044788 | 0.9532 (0.9384–0.9704) |

The repeat establishes a **4.68% absolute persistent training improvement**.
The normalized training ratio is **0.9559 [0.9015, 1.0268]**, so a relative-gap
improvement is not established. The candidate's paired persistent/ordinary
ratios remain **1.3050 forward and 1.1807 training**. Baseline parity is unmet.
Forward's approximately 1% movement tracks ordinary execution; its normalized
ratio is 1.0018 [0.9970, 1.0368]. Do not attribute this to head packing.

CI inference moves substantially between visits in both backends. Its repeat
persistent median rises 7.0% while ordinary rises 9.3%; an additional **1,200
CI-only comparisons**, with one hundred pairs per case in each visit, reverse
both movements using the same rebuilt binaries:

| Case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate / control (95% interval) |
|---|---:|---:|---:|
| CI inference | 0.057066 / 0.074295 | 0.051886 / 0.068910 | 0.9275 (0.9125–0.9330) |
| CI backward | 0.065135 / 0.093576 | 0.064565 / 0.093540 | 0.9996 (0.9961–1.0024) |
| CI SGD | 0.066440 / 0.094530 | 0.066505 / 0.094280 | 0.9974 (0.9952–1.0007) |

The confirmation's normalized inference ratio is 1.0067 [0.9931, 1.0172]. CI
backward/SGD absolute changes remain near zero across the studies. Preserve all
samples and both directions of movement; claim no consistent CI speedup or
regression. Ratios improved by slower ordinary execution do not count as gains.
Pooled bootstrap intervals do not capture all between-visit variation.

Four complete-step Perf SDK captures, with sixteen metric groups and zero-rate
SGD for stable replay, give persistent **15.8761→15.1283 ms (−4.71%)**; ordinary
is 11.9153→11.9863 ms (+0.60%). Estimated warp instructions fall **6.48%**, integer
thread instructions **3.76%**, FP32 thread instructions **3.40%**, global-load
sectors **5.79%** and shared wavefronts **1.84%**. Persistent local traffic stays
zero. Barrier waiting is 50.96→51.27% of active warp cycles, not elapsed time.
Maximum metric-group clock/instruction-rate spreads are 0.398%/0.985%.
These counters support less duplicated work and fewer claims at unchanged
register/shared capacity; they do not establish exact elapsed stall costs.

Each captured state contains **246,883,105 floats**. Control/candidate actual
files are bitwise identical within each backend: ordinary SHA-256
`f67022044ae8966ac48c5d62cd0907411fb42f4d1864c1ba0b3d555b17ce7be0`,
persistent `1e359875fa92965d4b0aa9026c050702ae3ac165aa59c0a4cc74a6381589c3b7`.
Across backends, the original budget passes with maximum error 4.66e-9; their
states are not bitwise identical. All **2,700 normal** and **twenty diagnostic**
comparisons preserve the same arithmetic selection, full-state/completion
checks, single compute dispatch and absence of fallback.

Qualification passes **669 library tests** (eight ignored), **168 strict
whole-graph oracle tests**, **50 matrix tests**, **ten whole-graph smoke tests**,
22 attention reference tests in each execution mode, formatting, strict Clippy
and Metal compilation. The new mixed fixture shares handlers across 1/2/3/4/5/15
query heads and 1/2/3/5 KV heads, widths 63/64/65/129/193, causal/cross/window masks,
poisoned scratch, three replays and worker counts 1/3/64/4096. Its outputs match
ordinary execution exactly. Admission checks both three- and four-tile maximum
handlers and smaller per-node widths; a separate mapping audit checks 146,016
unique coordinates and 91,520 claims across representative grids and axes.

Preserve the initial test-import failure and rejected head-guard admission.
The test-only qualification rebuild changes executable hashes despite identical
production source. The repeat measures those rebuilt binaries after confirming
byte-identical generated WGSL, matching resource-statistic multisets, geometry,
implementation selection, execution and buffers. The initial byte-hash assertion
and an order-sensitive statistics comparison are preserved alongside their
corrected audits. Native instruction bytes are unavailable; Metal compilation
does not qualify Metal GPU execution. Performance evidence is for the RTX 5080.

Artifacts are under `target/scalar-head-packing/`, including the primary study,
`qualification/`, `repeat/` and `ci-confirm/`. The manifest
`target/megakernel-results/scalar-head-packing/summary.json` indexes **806
independently verified artifact hashes**, **thirty measurement scopes** and
**62 recorded job scopes**; SHA-256
`17084fdf6631cb1f211c4093cd1b60516d383bf9a578b9ab5fcf690db6d469b5`.
All heavy jobs use separate verified cgroups with 16/20/1 GiB memory-high/hard/
swap limits. GPU jobs are serialized; builds, copies and heavy analyses do not
overlap timing or SDK capture. Artifacts remain on local `/x`.

Next, investigate partial scalar head batches for the five KV heads, preserving
uniform barrier loops, valid inactive-lane coordinates, scratch isolation and
exact logical completion despite any padded claim space. Keep cooperative
packing fixed and measure both model phases and all CI cases. Exact-order
subgroup reduction remains a separate experiment; the forward gap also remains
open. Do not silently cap requested workers or infer parity from diagnostic
counter reductions.

## Partial scalar head batches

Retain `2997a85` after full qualification and a larger completed-step repeat,
using `8a63a0a` as the runtime control. Scalar packing selects up to four heads
without requiring the width to divide the head count. Subgroup/cooperative
handlers retain their divisor-based 4/2/1 selection. No shader file, runtime
selector, environment hook, Naga interface or dependency revision is added.

Each node separates its padded ticket limit from its original logical task
count. The GPU metadata layout and queue allocation stay unchanged; a CPU
vector stores the claim limits for completion validation. A scalar handler
computes the valid head count for its ticket and uses that count for completion
and worker statistics. Inactive groups clamp both their logical coordinates
and scratch-bank index to a valid group, keeping barrier loops uniform without
reading an uninitialized tail bank. Empty nodes retain one completion-only
ticket and zero logical worker tasks. Checked padding and the existing worker
overshoot bound reject counters that could overflow.

The model's sixteen dQ nodes retain 12,000 logical tiles while successful claims
fall from 4,000 to 3,200. Its dKV nodes retain 2,640 logical tiles while claims
fall from 2,640 to 1,056. These are counts derived from the selected grids and
batch widths, not hardware counter estimates. Reported graph tasks remain
79,824. Implementation selection, GPU buffers, queue bytes, one compute
dispatch, one submission and complete Session/SGD boundaries are preserved.

The initial implementation kept a completion variable live across every handler,
including graphs with no partial scalar batches. Its **500 normal comparisons**
show training 17.044→16.587 ms, candidate/control **0.9732 [0.9553, 1.0102]**,
but CI SGD regresses 2.27% with an interval above no change. This version is
superseded. Code generation now emits that variable only when a graph has a
scalar packed handler. All CI and model-forward shader functions and resource
statistics then match the control. Their whole-source hashes can still differ
because the metadata comment changes; native instruction bytes are unavailable.

All ordinary shader files remain byte-identical. Cooperative handler functions,
cases and pooled scratch declarations also remain byte-identical. The refined
training shader is byte-identical to the initially profiled candidate, with the
same pipeline identity and resource statistics. Training registers fall
**103→98**, shared memory remains **29,232 bytes**, and code grows
267,904→269,824 bytes. Forward remains 100 registers, 18,832 shared bytes and
107,648 code bytes. No occupancy change is inferred from those counts.

A refined **500-comparison primary** improves persistent training
17.3165→16.6105 ms, ratio **0.9592 [0.9490, 0.9766]**. Its normalized ratio is
0.9555 [0.9279, 1.0362], so that sequence does not establish a relative-gap
improvement. A fresh **1,000-comparison repeat** uses the identical frozen
binaries, fifty paired samples per case in each of two control and two candidate
visits:

| Case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate / control (95% interval) |
|---|---:|---:|---:|
| CI inference | 0.056596 / 0.069860 | 0.053250 / 0.070426 | 1.0081 (0.9449–1.0189) |
| CI backward | 0.065166 / 0.093906 | 0.065310 / 0.094046 | 1.0015 (0.9957–1.0068) |
| CI SGD | 0.066490 / 0.094551 | 0.067090 / 0.095040 | 1.0052 (0.9988–1.0115) |
| Model forward | 3.851389 / 5.016856 | 3.843593 / 5.013291 | 0.9993 (0.9943–1.0055) |
| Model training + SGD | 14.709211 / 17.266281 | 14.735774 / 16.917443 | 0.9798 (0.9733–0.9913) |

The repeat establishes a **2.02% absolute persistent training improvement**.
Its normalized ratio is **0.9681 [0.9496, 0.9890]**, with ordinary latency nearly
unchanged, supporting a smaller relative gap in this sequence. Final paired
persistent/ordinary ratios remain **1.3021 forward and 1.1328 training**; parity
is unmet. Both candidate training visit medians are below both control visit
medians. CI and forward timing intervals include no change in both refined
sequences. Preserve the initial result and all visits; pooled bootstrap
intervals do not account for all between-visit variation.

Four complete-step SDK backend captures, sixteen metric groups and zero-rate
SGD give persistent **15.1314→14.7050 ms (−2.82%)**; ordinary changes only +0.14%.
Estimated persistent warp instructions fall **7.72%**, integer thread
instructions **7.82%**, FP32 thread instructions **4.23%**, global-load sectors
**7.72%**, and shared wavefronts **1.08%**. Persistent local traffic stays zero.
Barrier waiting rises 51.39→53.00% of active warp cycles despite faster replay;
this is not elapsed stall cost. Maximum metric-group clock and instruction-rate
spreads are 0.393% and 1.827%. Rate-times-duration instruction estimates and the
exact training-shader bridge support the explanation of less duplicated work
and fewer claims, without replacing normal completed-step timing.

Each actual SDK state has **246,883,105 floats**. Independent hashing confirms
bitwise-identical control/candidate files within each backend, with the same
ordinary/persistent hashes recorded for scalar head packing. Across backends,
maximum error stays at 4.66e-9 under the unchanged budget; their states are not
bitwise identical. All **2,000 normal and twenty diagnostic comparisons** pass
arithmetic-selection, full-state, completion, one-dispatch and no-fallback gates.

Qualification passes **669 library tests** (eight ignored), **168 strict oracle
checks**, **50 matrix regressions**, **ten whole-graph smoke tests**, 22 attention
references in each execution mode, formatting, strict Clippy and Metal-target
compilation. The mixed-head fixture adds six and seven KV heads to cover tails
of one, two and three heads. It preserves exact outputs across poisoned scratch,
three replays, partial dimensions, causal/cross/window masks and 1/3/64/4096
workers. The independent mapping audit covers 146,016 unique logical coordinates,
68,224 successful claims and 126,880 safely clamped inactive coordinates, plus
empty/overflow cases. Metal compilation is not Metal GPU qualification.

After session recovery, no jobs remained live. The initial audit's completed
artifacts survived without its shell exit; a separate verification run records
an explicit successful exit. A later audit incorrectly compared an unordered
pipeline-statistics list by position. Preserve that failure; comparison of the
full normalized statistic multisets passes, without dropping any statistics.
The source and capture audits, exact binary repeat and dependency checks pass.
Blade graphics, macros and Naga remain at fde3a9f/c09c89f/bc7b70b; both fork
checkouts stay clean and the lock is unchanged.

Raw evidence is local under `target/scalar-head-tail/`,
`target/scalar-head-tail-isolated/` and `target/scalar-head-tail-repeat/`.
The manifest `target/megakernel-results/scalar-head-tail/summary.json` contains
**843 independently verified artifact hashes**, **34 measurement scopes** and
**66 recorded job scopes**; SHA-256
`79028c78eea049775a38e4d7a88d79b2640a18153c4470bd95a8b8697eb6ec01`.
Every heavy job uses a separate verified 16/20/1 GiB cgroup. GPU jobs serialize,
and builds, copies and analyses do not overlap timing or SDK capture. All
artifacts stay on local `/x`.

Next, investigate exact-order subgroup reduction separately: the rowwise dKV
loop still executes eight workgroup barriers per query/head iteration. Preserve
the existing reduction tree, masks, accumulation order and a supported fallback;
first prove the subgroup mapping and cross-subgroup publication requirements.
Keep cooperative packing and explicit requested worker counts fixed. Continue
measuring the forward gap and all CI cases against this qualified runtime.
