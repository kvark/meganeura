# Open items

Carried forward from the October 2026 audit and the fixes that followed it.
Each entry states what was established, what is still open, and what would
close it. "Measured" means the claim was checked rather than inferred — several
items here exist because an earlier guess turned out to be wrong.

Status legend: **open** · **blocked** (needs a decision or measurement not yet
made) · **done** (kept for the record, with what it cost).

## Summary

| # | Item | Status | Where |
|---|---|---|---|
| 1 | Cooperative flash forward loses f16 precision on NVIDIA | **done — root-caused and fixed; suite green on both GPUs** | below |
| 2 | `load::onnx` slices with a file-controlled length | **done** | below |
| 3 | Per-step CPU work in the step loop | **host costs measured; end-to-end attribution open** | below |
| 4 | `step()` fences before recording, so CPU and GPU never overlap | **open — earlier overlap bound withdrawn** | below |
| 3c | CPU/GPU split benchmark (`bench_step_cpu`) | **corrected to include completion** | below |
| 5 | Structural: god functions, duplicated `(m,n,k)`, split tables | **done** | below |
| 6 | Importers accept malformed input without a fuzz corpus | **done — found two panics** | below |
| 7 | `EntrySpec` table for `ShaderEntry` | retracted | below |
| 8 | RMSNorm `(2..=32)` rows-per-workgroup bound | retracted | below |
| 11 | The ~9 ms periodic driver stall on this box | noted | below |

---

## 1. Cooperative flash forward loses f16 precision on NVIDIA

**Status: done, at the root cause. All 17 attention oracle cases now pass on the
RTX 5070, and the whole suite is green on both GPUs for the first time.**

Five `attention` oracle cases failed on NVIDIA and passed everywhere else.
`MEGANEURA_FLASH_FWD_COOP=0` made all 17 pass — the only switch that changed the
outcome, re-verified rather than inherited. `MEGANEURA_DISABLE_COOP=1` passed too,
with a **byte-identical** plan, so this was never about dispatch selection.

**It was not accumulation.** The diagnosis that matters:

| | |
|---|---|
| `coop_mat16x16<f32,C>` | the QK^T accumulator is f32 |
| `local_o: array<f32, chunk_hd>` | PV accumulates in f32 |
| `dst[...] = local_o[e] / safe_sum` | the output is f32 |
| `shared_q`, `shared_k_t`, `shared_v` | the f16 is workgroup *staging* |

`examples/bench_attention_coop` settles it without a shader. It rounds one
operand at a time **in the f64 CPU reference** and differences against the exact
evaluation, so the floors are properties of the operands, not the kernel:

| shape | measured | floor, all three rounded | Q | K | V |
|---|---|---|---|---|---|
| hd=64, q=31 | 2.360e-4 | 2.360e-4 | 6.18e-5 | 5.20e-5 | **2.360e-4** |
| hd=256, q=33 | 2.719e-4 | 2.714e-4 | 3.84e-5 | 4.60e-5 | **2.438e-4** |
| hd=128, q=64 | 2.549e-4 | 2.546e-4 | 6.66e-5 | 4.52e-5 | **2.403e-4** |
| q=1024, 8 heads | 2.827e-4 | 2.826e-4 | 9.60e-5 | 8.35e-5 | **2.403e-4** |

The measured error **equals the floor to three digits**. The kernel's own
arithmetic contributes nothing detectable — every f32 accumulator is doing its
job. And V alone accounts for essentially all of it, while Q and K together cost
3–4x less.

**Why V, and why the arithmetic cannot rescue it.** The cooperative matrix is
QK^T only. `shared_q` and `shared_k_t` are operands of
`coopLoadT<coop_mat16x16<f16,A/B>>`, so the hardware requires f16. **V is not an
operand of any matrix instruction** — PV is a scalar loop. The generated code
even said so: `local_o[e] + p * f32(shared_v[...])`, converting an f16 the shader
had itself just rounded one line earlier.

That round trip is also why nothing downstream hides it. Q's and K's errors are
summed over `head_dim` terms and then normalised by the softmax, so they average
each other away. V's error enters the output *linearly*, as a convex combination
with positive weights summing to one — and positive weights do not cancel a
rounding error the way an average over `head_dim` terms does.

**The fix, and three instances of it.** V's staging becomes `array<f32>`, and the
one `f32(...)` on read goes away. Enforcing the rule rather than patching the
symptom found two more:

| shader | array | read by |
|---|---|---|
| `flash_attention_coop` | `shared_v` | scalar PV loop |
| `flash_grad_q_coop` | `shared_k` | scalar `dS·K` loop |
| `flash_grad_kv_coop` | `shared_q`, `shared_do` | scalar `dS·Q` and `p·dO` loops |

Each is the second copy of a tensor whose *other* copy is the real cooperative
operand — `shared_k_t` beside `shared_k`, `shared_q_t` beside `shared_q`. The
cooperative matmul keeps its f16 operands; only the scalar consumers move to f32.
Every remaining `array<f16>` in workgroup memory feeds a `coopLoadT`/`coopStoreT`.

The forward was also inconsistent with its own non-cooperative counterpart, which
already staged V as `f32`.

**Result.** Residual error is now 6.3e-5 to 1.09e-4, matching the analytic Q+K
floor (6.25e-5 to 1.44e-4) — i.e. exactly the part the hardware forces, and
nothing else. `exact_f16` fell from 3.2% to 0.0%, so the outputs are no longer on
the f16 grid at all. All 17 attention cases pass; the whole suite is green on the
RTX 5070 and the Intel B570.

No tolerance was changed and no policy was touched. `CoopPolicy::NativeF32` still
exists for callers who want full precision, and `Auto` still uses the cooperative
matmul — it just no longer rounds a tensor that nothing required it to round.

**Cost.** Forward coop, minimum of 200 steps, RTX 5070: 0.03–0.39 ms across the
shape set, against 0.03–0.53 ms for the scalar kernel. For shapes that fit the
device limit, the extra shared memory is not free but is not decisive; `Auto`
remains 1.5–3x faster than the scalar path it replaces.

**The guard.** `every_f16_workgroup_array_is_a_cooperative_operand` asserts the
invariant over the generated WGSL for all three cooperative attention kernels at
head_dim 16/32/64/128/256 and the cooperative matmul family across tile size,
operand type and `compensated`. It is structural rather than numeric on purpose:
most of these variants are only reachable on hardware with cooperative-matrix
support, so a numeric test would not run where the mistake is easiest to make.
Reverting any one of the four arrays makes it fail.

Selection also checks the workgroup storage against the selected device's
limit. At head_dim 256, cooperative dQ needs 52,416 bytes, exceeding the RTX
5070's 49,152-byte limit; forward still fits at 33,792 bytes. Forward, dQ and
dK/dV fall back independently when they do not fit. The limit reaches both
ordinary compilation and measured search, participates in the build cache key,
and older cached plans are invalidated. A CPU check derives storage from the
generated shader types, and hardware parity cases cover the wide-head fallbacks.

## 2. `load::onnx` slices with a file-controlled length

**Status: done.** Seven sites across four functions now go through one
`delimited` helper; the three RoPE builders use `u32::try_from`. Four tests
cover it, and reverting the helper makes all four panic, so they are load-bearing
rather than decorative.

This, as it stood:

```rust
let (len, p) = read_proto_varint(buf, pos).ok()?;
let len = len as usize;
match field_no {
    1 => name = String::from_utf8_lossy(&buf[p..p + len]).into_owned(),
    2 => type_bytes = Some(&buf[p..p + len]),
    _ => {}
}
```

`len` is a `u64` from the file, truncated by `as usize`, and used directly as a
slice length with no bounds check. A malformed or hostile `.onnx` panics here
rather than returning the `None` the enclosing `Option`-returning signature
promises — the panic escapes through `load_onnx`'s error type entirely.

Two smaller truncations elsewhere: the dimension walk in `load/onnx` steps `pos`
by a fixed amount per field and only re-checks against `buf.len()` at the top of
the loop, so a truncated stream misparses silently — that one is fixed, see
below; and the three RoPE builders in `graph.rs` did `x_shape[1] as u32` for the
head dimension, where the adjacent `dim % 2 == 0` and `dim % head_dim == 0`
asserts do not catch a dimension that truncated to a valid-looking value — also
fixed.

**What was done.** `delimited(buf, len, start) -> Option<(&[u8], usize)>` — one
helper, in `load/onnx.rs` beside `read_proto_varint` — does
`usize::try_from`, `checked_add` and `get`, and every length-delimited site uses
it, so there is one bounds-checked path rather than seven hand-written slices.
The `Result`-returning callers get a typed `OnnxError::ParseError`; the
parsers that return a value directly `break`, which is what their existing
malformed-input handling already does. The fixed-width wire types (`pos += 8`,
`pos += 4`) were also unchecked and could step past the end without the loop
noticing; those are `checked_add` now.

Tests: a length past the end, a start past the end, `usize::MAX` (which must
not wrap the end back into range), and a sweep of lengths and offsets that must
return rather than panic. Reverting `delimited` to the unchecked form makes all
four panic at the slice.

---

## 3. Per-step CPU work in the training loop

**Status: host costs measured; the claimed share of completed-step time is
withdrawn.** The original benchmark stopped its timer after submission and
waited for completion outside the measured region. Its results describe the
CPU call, so they cannot establish that encoding dominates a completed step.

**3a. `optimizer_units()` rebuilds the segment table every step.**
`src/runtime/optimizer.rs` rebuilds plans and segments, looks up each parameter's
logical length and learning-rate multiplier, and compares the result with the
uploaded table. This is repeated host work even when the layout is unchanged.
The old sweep used 16 parameters in both its `8×64` and `8×1024` cases; changing
weight dimensions while keeping that count fixed does not isolate the cost of
building the segment table. Its share remains unmeasured.

**3b. `Pipelines::get` performs a HashMap lookup per dispatch.**
`&self.map[&self.selected[dispatch_index]]` still hashes a `Variant` during
recording. An earlier instrumented run attributed about 7% of host recording
time to this lookup. That is a host fraction, not a demonstrated end-to-end
saving. The previous estimate of a 6% whole-step improvement is withdrawn.

**3c. Separate encoding, submission and completion.**
`examples/bench_step_cpu.rs` now reports:

- `encode_ms`: `Session::record` on an idle device, without submission.
- `step_call_ms`: `Session::step`, including recording and submission.
- `wait_ms`: the following `Session::wait`, including remaining GPU execution
  and host synchronization.
- `step_ms`: the entire `step()` plus `wait()` interval.

Input upload stays outside the interval. Each column is the median of its own
samples, so the displayed medians need not add up. Completion wait is a wall
measurement, not a GPU timestamp. The benchmark uses one context for the sweep.

The old RTX 5070 measurements below remain useful only as host measurements
(40 runs, median, chain of `matmul + bias_mul + gelu` blocks). The former
`step_ms` column is relabeled `step_call_ms`; the former `idle_ms` was a second
encoding loop, with the device idle in both loops, and is omitted.

| blocks | width | dispatches | params | step_call_ms | encode_ms |
|---|---|---|---|---|---|
| 4 | 64 | 53 | 8 | 0.043 | 0.038 |
| 8 | 64 | 105 | 16 | 0.080 | 0.075 |
| 16 | 64 | 209 | 32 | 0.149 | 0.141 |
| 32 | 64 | 417 | 64 | 0.303 | 0.297 |

Encoding scaled with dispatch count in this sweep. These values do not show
that encoding is 90–98% of completed-step time, nor do they bound CPU/GPU
overlap savings.

A validation run of the repaired benchmark on 2026-10-03 (RTX 5070, release
build, 40 samples per case) measured the previously omitted wait:

| blocks | width | batch | encode_ms | step_call_ms | wait_ms | step_ms |
|---|---|---|---|---|---|---|
| 4 | 64 | 1 | 0.0366 | 0.0472 | 0.0655 | 0.1132 |
| 8 | 64 | 1 | 0.0730 | 0.0855 | 0.1065 | 0.1922 |
| 16 | 64 | 1 | 0.1429 | 0.1573 | 0.1852 | 0.3433 |
| 32 | 64 | 1 | 0.2876 | 0.3091 | 0.3427 | 0.6540 |
| 8 | 256 | 1 | 0.0726 | 0.0848 | 0.1349 | 0.2195 |
| 8 | 1024 | 1 | 0.0726 | 0.0850 | 0.6508 | 0.7363 |
| 16 | 1024 | 1 | 0.1485 | 0.1803 | 1.4251 | 1.6054 |
| 8 | 256 | 4 | 0.0695 | 0.0815 | 0.1866 | 0.2678 |

These are measurements from one run of the corrected timer, not an encoder
rotation comparison. They confirm that the old region omitted a substantial
part of completion time, especially as the matrix width grows.

**Host attribution retained from the earlier instrumented run.** Four timing
pairs inside `record_groups` measured the following phases. Instrumentation
added about 90 ns per dispatch, so these are approximate upper bounds. The
instrumentation was temporary; reproducing this breakdown requires restoring
those timing pairs.

| phase | ns/dispatch | share of recording |
|---|---|---|
| `pipelines::get` | 44 | 7% |
| `pass.with(pipeline)` | 58 | 9% |
| `bind_dispatch` | 187 | 28% |
| `pc.dispatch(workgroups)` | 369 | 56% |

Binding and dispatch recording dominated the measured host work. Optimizing
these phases, caching optimizer metadata, and reducing dispatch count remain
candidates; their end-to-end benefit needs a completed-step measurement on the
workload being optimized.

## 4. `step()` fences before recording

**Status: open. The earlier conclusion that encoder rotation was not worth
doing is withdrawn.**

`Session::step` waits for the previous submission before recording:

```rust
self.wait();
self.encoder.start();
self.encode_step(...);
self.sync_point = Some(self.gpu.submit(&mut self.encoder));
```

The original benchmark called `set_input` before starting the timer, consuming
that previous submission's fence, and stopped timing before the new submission
finished. Subtracting encoding time therefore left mostly submission overhead.
The claimed 1–5% overlap bound omitted GPU execution and cannot support a
scheduling decision.

The repaired benchmark includes completion and reports the wait separately.
Even with those measurements, `step_ms - encode_ms` is not the time an encoder
rotation would save: submission and synchronization also contribute, and the
CPU and GPU can overlap only part of their work. Closing this item requires a
comparison of completed-step throughput with and without encoder rotation on
the same graph, inputs, device and synchronization contract.

## 5. Structural items

**Status: open. None are urgent; all are real.**

- ~~**`bind_dispatch` is ~1210 lines.**~~ **Done.** One 58-arm match over
  `dispatch.shader` is now a routing match plus eight family functions:

  | | lines | | lines |
  |---|---|---|---|
  | `bind_dispatch` | 333 | `bind_norm` | 195 |
  | `bind_matmul` | 120 | `bind_activation` | 78 |
  | `bind_conv` | 196 | `bind_reduction` | 58 |
  | `bind_attention` | 202 | `bind_loss` | 48 |
  | | | `bind_pointwise` | 203 |

  The design point is that **the routing match is the only exhaustive one and it
  has no wildcard arm**, so adding a `ShaderEntry` variant fails to compile at
  the routing decision. Each family's inner match ends in `unreachable!` — if a
  routing arm and a family ever drift, the compiler still passes and the
  mismatch is caught the first time a test reaches it.

  Splitting it does **not** make it faster, and that was measured before doing
  it: timing `bind_dispatch` per shader entry across the whole `bench_step_cpu`
  sweep gives 138–174 ns for every entry with no hot arm, so the 187 ns that is
  28% of the per-step host path is the binding model rather than one arm doing
  something wasteful. Moving the arms moves the same work.

  Verified rather than assumed: the sequence of 65 `pc.bind` calls is
  byte-identical before and after, so nothing was reordered or dropped.

  The `(m, n, k)` rule had a **third** copy here as well as the two already
  fixed — see below. Routing every contraction arm through `Dispatch::mnk()`
  also replaced thirteen `params[0..3]` spellings.
- ~~**Module sizes.**~~ **Done.** `runtime.rs` 8592 → 5666, `codegen.rs` 8146 →
  5419, `compile.rs` 7928 → 4326. The split is by responsibility, not by line
  count:

  | moved | to | lines | why that seam |
  |---|---|---|---|
  | `impl Compiler` | `compile/emit.rs` | 3634 | everything here *emits* dispatches; everything in the parent *describes* the plan |
  | attention + conv generators | `codegen/attention.rs` | 2744 | the matmul family stays put; everything that is not a matmul is here |
  | `Session` binding | `runtime/binding.rs` | 1477 | given a `Dispatch`, what goes in binding 0 |
  | `Session` transfers | `runtime/transfer.rs` | 1444 | everything that touches a buffer from the CPU side |
  | `GpuOptions`, context creation | `runtime/context.rs` | 99 | nothing here touches a `Session` |
  | host-side Q4/Q8 packing | `runtime/quantize.rs` | 187 | callable from `f32` without a GGUF file, unlike the load-only formats |

  Two conventions, both forced by the compiler rather than chosen:

  - Methods and helpers that move become `pub(super)`, because an inherent impl
    in a child module is only reachable from the parent if the methods are
    visible there. `pub(super)` rather than `pub` keeps the surface unchanged —
    no caller outside the module can reach them either way.
  - Public items are re-exported (`pub use attention::*;`, `pub use
    context::*;`), so every existing path still resolves. The one path that had
    to change was relative: `include_str!("shaders/...")` inside the moved
    generators became `"../shaders/..."`, which the compiler caught immediately.

  No behaviour changed: 491 + 22 + 112 + 121 + 64 green on both GPUs, and the
  set of `pub` item names across `src/` is identical before and after — 711 each,
  none added and none removed. That is the check that makes "pure
  reorganisation" a claim rather than an assertion.
- ~~**`(m,n,k)` is reinterpreted differently at two binding sites.**~~ **Done.**
  The two sites disagreed: the horizontal-batch binding swapped `n` and `k` only
  for `ShaderEntry::MatMul`, and the cooperative-prologue binding only for
  `MatMul` and `FusedMatMulAdd`. Neither matched the shaders that actually store
  `(m, k, n)` in `params` — `MatMul`, `MatMulGemv`, `MatMulGemvAdd`,
  `FusedMatMulAdd`. `Dispatch::mnk()` now encodes the verified table beside
  `scalar_matmul()` / `conv_k_tile()` / `gemv_shape()`, and returns `None` for a
  non-contraction rather than handing back three unrelated numbers.

  Whether either site was *wrong in practice* was checked rather than assumed.
  Instrumenting `merge_horizontal` across the whole suite shows only `MatMul`
  (55), `MatMulBT` (29) and `MatMulAT` (44) are ever horizontally merged, so the
  first site's list happened to be complete for every case the suite produces —
  by luck, since `merge_horizontal`'s predicate in `compile.rs` restricts the kernel
  but not the shader. The prologue site is **never reached by any test at all**,
  so its old list could not be validated empirically; the centralisation makes it
  correct by construction instead.

  Three tests pin it, including a table of all eight contraction shaders against
  the order their constructors use.
- ~~**Quantized block geometry lives in three independent tables.**~~ **Done**,
  though it was five. `DType::block_geometry()` is now the single source for
  `(elements per block, bytes per block)`, read by `TensorType::size_bytes`, by
  `set_parameter_packed`'s copy loop and by all six `Graph::parameter_q*k`
  divisibility asserts; `WeightFormat::dtype()` maps a format to the `DType` it
  stores so the copy loop cannot name a different geometry than the buffer was
  sized with. Three tests, including one that restates each stride as the
  literal arithmetic it replaced (`blocks * 36 * 4` and so on) so a change to
  the table cannot quietly resize a buffer.

  Meganeura's `Q4_0` and `Q8_0` still keep their own arithmetic and resolve to
  no `DType` counterpart — that is deliberate, since `DType::Q4_0` is
  Meganeura's *asymmetric* Q4 and `DType::Q8_0` pads to 36 bytes where GGML uses
  34. The GGUF loader's table stays separate for the same reason: it describes
  GGML's wire format, and merging the two is what would erase that difference.

---

## 6. Importers accept malformed input without a fuzz corpus

**Status: done. `proptest` over arbitrary bytes found two reachable panics, both
fixed; the corpus is checked in.**

`proptest` as a dev-dependency, four properties in `tests/importer_fuzz.rs`, all
asserting the same contract: arbitrary input must come back as a `Result::Err`
and never a panic. 256 cases by default, `PROPTEST_CASES=20000` clean. The cases
are arbitrary bytes, not well-formed documents with a field perturbed, because
the failures here are structural — a length that runs past the end, a varint
that overflows — and those are what a hand-built corpus misses.

**Two panics, both shrunk to a minimal input:**

**1. NNEF: a body that closes before it opens.** `parse_graph_nnef` finds the
body with an independent `find('{')` and `rfind('}')`, so the two-byte input
`"}{"` leaves `body_start > body_end`. Every slice in the header parser spans
that range. One comparison at the source fixes it, and two unit tests pin it —
one for the rejected shapes, one asserting `{}` still parses so the check cannot
be over-tightened into rejecting valid input.

**2. ONNX: an overflowing length in the protobuf dependency.**
`oxionnx-proto-0.1.2/src/parser.rs:73` tests `pos + len > buf.len()`, which
overflows for a large declared length and passes the check before slicing
`buf[pos..pos + len]`. This is not ours to patch. `load_onnx_bytes` contains it
with `catch_unwind`, because the function's signature promises a `Result` and a
caller loading an untrusted file should not be able to abort their process with
four bytes of input. The fence is deliberately at that boundary rather than
around just the dependency call: `extract_shapes_from_proto` and
`translate_graph` follow, are ours, and are not supposed to panic — one
boundary covers all three.

The panic message is left reaching stderr on purpose. Suppressing it would need
a global hook swap around a library call, and it would also hide a panic from
our own code below, which is a bug worth being loud about.

**The corpus.** `tests/importer_fuzz.proptest-regressions` holds both minimal
inputs, so they replay on every run. proptest appends to that file per failing
test, so two simultaneous failures under parallel threads can clobber each
other; the suite runs `--test-threads=1`, and the test file says so.

Both fixes are verified by reverting them: each makes its recorded corpus entry
fail again.

## 7. `EntrySpec` table for `ShaderEntry` — retracted

**Status: retracted. The reasoning was wrong; do not build this.**

The claim was that adding a `ShaderEntry` variant costs fifteen coordinated
edits across four files, and that a descriptor table would collapse them to one
row. Measured: adding a canary variant produces **five `E0004`s** naming
`profile_family`, `shader_group` and `entry_point` in `compile.rs` and
`shader_data_layout` and `bind_dispatch` in `runtime.rs`. All five matches are
exhaustive over all 105 variants, so the compiler already catches a missing arm.

More importantly the proposed table is not expressible. `Pipelines::key` is a
**cascade over dispatch shape** — split-K, conv tile, horizontal batch,
epilogue, scalar matmul, GEMV modifiers, reduction, pointwise, attention,
weight format, coop, small tile — not a lookup keyed by entry. There is no
per-entry arm in it to fall out of step, so a table would have to encode the
cascade rather than remove it.

What genuinely remains unenforced: that the generated WGSL and the pipeline key
agree, and that the binding code matches the layout. Those are covered at
runtime by the assert in `Pipelines::select` and at build time by
`all_shaders_generate_valid_modules` — which is weaker, because most of those
variants are only reachable on hardware with cooperative-matrix support.

---

## 8. RMSNorm `(2..=32)` rows-per-workgroup bound — retracted

**Status: retracted. Measured, no effect.**

At `cols == 64` the bound yields `rows_per_workgroup == 1`, so a 256-thread
workgroup covers one row: 64 lanes compute, 192 exit early, and the reduction
tree still runs over all 256. The obvious fix — extending the bound to
`(2..=64)` so eight rows share a workgroup — measures no better.

RTX 5070, 65536 rows, 200 runs, same element count in both cases:

| width | rows/workgroup | time | bandwidth |
|---|---|---|---|
| 64 | 1 | 7.46 ms | 6.7 GB/s |
| 32 | 8 | 7.60 ms | 6.6 GB/s |

Ratio 0.98 and 1.00 across two runs. The kernel runs at ~6.6 GB/s, which is
memory-bound, so idle lanes cost nothing and widening the bound would only
churn the reduction order. Do not "fix" this without a new measurement showing
the kernel has become compute-bound.

The benchmark was not kept: `bench_ci_latency` reports the same shape at
0.05 ms, too small to resolve a change in reduction geometry. A permanent
benchmark for a conclusion of "no effect" would be scaffolding with nothing to
guard.

---

## 9. Cross-cutting: context exhaustion hid item 1

**Status: done. Kept because it is the reason item 1 was invisible.**

The NVIDIA driver issues roughly ten GPU contexts per process and then refuses
with `NoSupportedDeviceFound`. `SessionConfig::from_env` logged that, set
`gpu: None`, and `build` fell through to `default_gpu_context()` — which names
no device, so Blade picked whichever adapter initialised first. On a machine
with both an NVIDIA and an Intel card that means sessions 1–10 ran on the
requested device and everything after ran on Intel, silently, with no logger
installed to surface the warning.

Fixed by making a named device a hard failure (`config.rs`), making
`SessionConfig::from_env_with_gpu` public so callers can share one context, and
sharing a context across the test harnesses (`tests/support/gpu.rs`) and
`load::gguf::Generator`.

Worth remembering: any test result on a multi-adapter machine is only
trustworthy if the device was pinned and the context budget respected. The
suite now panics rather than falling through, so a violation is loud.

---

## 11. The measurement pitfall on this hardware

**Status: noted so the next benchmark does not rediscover it.**

An earlier version of `bench_attention_coop` reported ~8.6 ms for the scalar
kernel and ~1.6 ms for the cooperative one — the cooperative kernel appearing six
times *faster* than the scalar path it is supposed to beat. Two effects
combined.

**The NVIDIA driver takes a periodic ~9 ms stall.** On a 200-step run of an 8x8
matmul, which contains no attention kernel at all:

| policy | p10 | p50 | p90 | p99 | min |
|---|---|---|---|---|---|
| Auto | 0.057 ms | 1.32 ms | 8.95 ms | 9.70 ms | 0.055 ms |
| NativeF32 | 0.065 ms | 1.33 ms | 9.08 ms | 9.48 ms | 0.064 ms |
| Disabled | 0.060 ms | 0.077 ms | 9.16 ms | 9.85 ms | 0.059 ms |

Identical shape for all three policies, so it is not the graph. Which sample the
median lands on depends on where the stall boundaries fall relative to the sample
count, which is why successive runs of the same binary reported different
policies as the slow one.

**Timing a bare `step` measures submission, not the step.** Successive steps
pipeline, so a loop of them reports whichever stall the driver happened to take.
Each step has to be timed with its own `wait()` inside the measured region.

With both corrected, the minimum was stable to within 10% across runs and
policies in that attention experiment. `bench_step_cpu` reports medians;
its encoding samples measure host work, while its corrected completion samples
can include these driver stalls. Those wall times need that distinction when
comparing devices or estimating performance improvements.

---

## 12. Claims retracted during this work

Recorded because an audit that only keeps its hits teaches the wrong lesson.
Each of these looked like a defect and was checked instead of assumed.

- **`set_parameter` duplicates a 20-line encoder match.** It appears once.
- **The flash-attention family is "textbook copy-paste, six copies."** Only 27
  lines are verbatim common to all six generators; pairwise overlap is 33–72
  lines. The kernels genuinely differ — coop versus scalar uses different data
  movement, dQ versus dKV differ in what is reduced. The duplication that *was*
  real (the causal/window range at five sites, and the attention `Params`
  literals at eight) has been fixed.
- **`optimizer_len` returns a padded element count for gradients.** Every call
  site passes the *parameter* buffer, which has a `param_types` entry, so the
  `plan.buffers[param] / 4` fallback is never reached for a gradient. The
  one-ULP test failure this was blamed for is f32 accumulation drift; the test
  now uses a measured tolerance.
- **Adding a `ShaderEntry` variant needs fifteen edits.** Five compiler-caught
  `E0004`s; see item 7.
- **Removing the `Pipelines::get` hash is worth doing.** Measured at 7% of the
  host path, and 56% of that path is one `pc.dispatch` into the driver; see
  item 3.
- **Cooperative flash forward is "~3.2x per dispatch" faster than scalar.** The
  shape is real — 3.2x at `hd=128` with GQA — but it is not uniform: 1.06x on a
  shape below the flash threshold and 1.8x at `q=1024`, and the first version of
  the measurement had it *slower*, inverted, because of the stall in item 11.
- **`Pipelines::get` hashing a `Vec`-bearing enum is expensive.** A standalone
  microbenchmark puts a `HashMap` probe of that enum at 23 ns, of which 12 ns is
  hashing the key. The width of the enum is not the cost; the per-dispatch count
  is.
- **The cooperative attention error was an accumulator problem.** It was not.
  Every accumulator was already f32 and the output was f32; the error was f16
  *staging* of a tensor that feeds no matrix instruction, and it survived to the
  output because a convex combination does not cancel a rounding error the way a
  dot product over `head_dim` terms does. Measuring the floor in the reference
  rather than reasoning about the shader is what found it — and it found two more
  instances in the backward kernels that nobody had reported.
