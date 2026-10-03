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
| 1 | Cooperative flash forward loses f16 precision on NVIDIA | **measured — trade is real; the fix has a portability cost** | below |
| 2 | `load::onnx` slices with a file-controlled length | **done** | below |
| 3 | Per-step CPU work in the step loop | **measured — the cost is real; the two targets named were the wrong ones** | below |
| 4 | `step()` fences before recording, so CPU and GPU never overlap | **measured — not worth doing** | below |
| 3c | CPU/GPU split benchmark (`bench_step_cpu`) | done | below |
| 5 | Structural: god functions, duplicated `(m,n,k)`, split tables | open | below |
| 6 | Importers accept malformed input without a fuzz corpus | **done — found two panics** | below |
| 7 | `EntrySpec` table for `ShaderEntry` | retracted | below |
| 8 | RMSNorm `(2..=32)` rows-per-workgroup bound | retracted | below |
| 11 | The ~9 ms periodic driver stall on this box | noted | below |

---

## 1. Cooperative flash forward loses f16 precision on NVIDIA

**Status: measured. The speed is worth having; the error is not all of it
unavoidable. The remaining fix is a judgement call, recorded here with both
sides measured.**

Five `attention` oracle cases fail on the RTX 5070 and pass on Intel, lavapipe
and every other adapter. `MEGANEURA_FLASH_FWD_COOP=0` makes all 17 pass — the
only switch that changes the outcome, and re-verified rather than taken from the
earlier audit. `MEGANEURA_DISABLE_COOP=1` passes too, with a **byte-identical**
plan, so this is not dispatch selection: the cooperative *forward* kernel is
reached through `Auto` even when cooperative dispatch is disabled.

**Where the f16 actually is.** Reading `generate_flash_attention_coop_module`:

- `coop_mat16x16<f32,C>` — the QK^T accumulator is already f32.
- `local_o: array<f32, chunk_hd>`, and PV accumulates as
  `local_o[e] + p * f32(shared_v[...])` — f32 throughout.
- `dst[...] = local_o[e] / safe_sum` — the output is f32.
- The f16 is `shared_q`, `shared_k_t`, `shared_v`: Q, K and V are rounded to
  f16 when staged into workgroup memory.

So the loss is representational, in the *inputs*. Q and K must be f16 — they are
operands of `coopLoadT<coop_mat16x16<f16,A/B>>`. **V is not.** The cooperative
matmul is QK^T only; PV is a scalar loop. V's f16 storage is an artefact, and
the `f32(shared_v[...])` conversion in the accumulate line is the tell: it exists
only because the storage was f16.

**The measurement.** `examples/bench_attention_coop` sweeps the three policies
over the oracle's own flash shapes against an independent f64 CPU reference.
RTX 5070, minimum of 200 steps, coop:

| shape | Auto (f16 coop) | NativeF32 | Disabled | Auto vs scalar |
|---|---|---|---|---|
| hd=64, q=31 (scalar control) | 0.031 | 0.028 | 0.029 | 1.06x |
| hd=256, q=33 | 0.036 | 0.069 | 0.099 | 2.7x |
| hd=128, q=64, gqa | 0.032 | 0.092 | 0.103 | 3.2x |
| hd=128, q=63 (scalar control) | 0.086 | 0.050 | 0.063 | 0.73x |
| hd=64, q=130 | 0.046 | 0.081 | 0.096 | 2.1x |
| hd=32, q=260 | 0.037 | 0.110 | 0.090 | 2.4x |
| hd=16, q=257 | 0.064 | 0.096 | 0.097 | 1.5x |
| q=1024, 8 heads, hd=64 | 0.293 | 0.584 | 0.526 | 1.8x |

Two things. `NativeF32` tracks `Disabled` everywhere, confirming the auto-tune
report that this device advertises no f32 cooperative tile — so on NVIDIA
`NativeF32` *is* "use the scalar kernel". And the speed is real: 1.8x at the
largest shape, up to 3.2x where flash selects. The two "scalar control" rows are
shapes below the flash threshold, where no policy should matter and none does.

**What staging V as f32 buys.** Measured by changing `shared_v` to `array<f32>`
and removing the one conversion — six lines, no policy change, no tolerance
change:

| shape | worst_abs as f16 | worst_abs as f32 | ratio | time as f16 | time as f32 |
|---|---|---|---|---|---|
| hd=256, q=33 | 2.72e-4 | 6.31e-5 | 4.3x | 0.036 | 0.068 |
| hd=128, q=64 | 2.55e-4 | 8.11e-5 | 3.1x | 0.032 | 0.036 |
| hd=64, q=130 | 2.36e-4 | 8.03e-5 | 2.9x | 0.046 | 0.046 |
| hd=32, q=260 | 2.36e-4 | 1.00e-4 | 2.4x | 0.037 | 0.044 |
| q=1024, 8 heads | 2.83e-4 | 1.44e-4 | 2.0x | 0.254 | 0.344 |

So V accounts for roughly three quarters of the error: Q and K's f16 rounding
barely registers, because the coop matmul accumulates them in f32 over `hd` terms
and softmax then normalises. Recovering it costs **1.35x on the forward at the
largest shape** (0.254 → 0.344 ms), which still leaves the path 1.5x faster than
the scalar kernel.

**Why this was not simply done.** Workgroup storage. With `bkv=16`:

| head_dim | shared (V as f16) | shared (V as f32) |
|---|---|---|
| 64 | 7 KB | 9 KB |
| 128 | 13 KB | 17 KB |
| 256 | 25 KB | 33 KB |

25 KB is already over the 16 KB that WebGPU guarantees, so this path needs an
opt-in above the floor either way — but 33 KB excludes devices that offer exactly
32 KB, which 25 KB does not. That is a narrowing of the set of hardware the
cooperative forward works on, traded for 2–4x accuracy on a path that is
documented as reduced precision.

That is a policy decision, not a bug fix, so it is not taken unilaterally here.
Loosening the oracle's tolerance to accept 2.4e-4 would hide the defect rather
than address it, which is worse than either option.

**What would close it.** A decision on the table above. If V as f32: it is six
lines, it recovers three quarters of the error, and it needs a head_dim at which
the extra 8 KB is refused rather than assumed. If V stays f16: the oracle needs a
tolerance derived from the representation (one f16 ulp) for cooperative paths,
stated as the contract that path actually offers — with the default documented as
f16 rather than leaving it implicit.

## 2. `load::onnx` slices with a file-controlled length

**Status: done.** Seven sites across four functions now go through one
`delimited` helper; the three RoPE builders use `u32::try_from`. Four tests
cover it, and reverting the helper makes all four panic, so they are load-bearing
rather than decorative.

`src/load/onnx.rs:209-213`:

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

Two smaller truncations in the same reader: the dimension walk at
`onnx.rs:206-207` steps `pos` by a fixed amount per field and only re-checks
against `buf.len()` at the top of the loop, so a truncated stream misparses
silently; and `src/graph.rs:2058,2111,2146` do `x_shape[1] as u32` in the three
RoPE builders, where the adjacent `dim % 2 == 0` and `dim % head_dim == 0`
asserts do not catch a dimension that truncated to a valid-looking value.

**What was done.** `delimited(buf, len, start) -> Option<(&[u8], usize)>` does
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

**Status: measured. The cost is real and dominant, but neither target the audit
named accounts for much of it. See "attribution" below — the per-dispatch hash
is ~7% of the host path.

Three findings, all on the per-step path:

**3a. `optimizer_units()` rebuilds the whole segment table every step.**
`src/runtime/optimizer.rs:432`. Per step it rebuilds a `Vec<Plan>` with a
`Vec<(usize,u32)>` per arena chunk, rebuilds the entire `Vec<Segment>` — one
`optimizer_len` HashMap lookup per parameter, plus a `lr_multiplier_for_buf`
prefix scan — then `bytemuck`-memcmps the full table (`#params × 32` bytes) to
decide whether to re-upload. The table only changes when a parameter is
rebound (`optimizer.rs:271-273`), so most of that work is recomputing an
invariant. The `memcmp` is itself a working measurement of "did it change?",
which is how the potential saving can be bounded without new instrumentation —
but nobody has run it against a model large enough for the answer to matter.

Note the cost is paid even with no optimizer and no clipping configured, as
long as the session has trainable pairs.

**3b. `Pipelines::get` is a SipHash lookup per dispatch, per step.**
`src/runtime.rs:1508`. `&self.map[&self.selected[dispatch_index]]` hashes a
`Variant` — a wide enum some of whose variants embed a `Vec`, e.g.
`SpecializedConv(ShaderEntry, Vec<u32>, u32)`. The comment at `runtime.rs:1114`
says variants are "resolved after compilation or tuning, never while recording a
step", which is true of the *key* but not of the lookup. Storing
`selected: Vec<&ComputePipeline>` after `select()` removes it. This is the
per-dispatch half of the measured cost below, and it is worth roughly a third of
the per-dispatch figure by inspection, not by attribution — see "what is still
attributable".

**3c. The benchmark that settles both. `examples/bench_step_cpu.rs`.**

`Session::record` encodes a step into a caller's encoder and does not submit,
so it times the host path exactly — walking dispatches, resolving each
pipeline, building its binding struct, issuing the call. `Session::step` is the
whole step. The difference is what an encoder rotation could overlap.

RTX 5070, 40 runs, median, chain of `matmul + bias_mul + gelu` blocks:

| blocks | width | dispatches | params | step_ms | encode_ms | idle_ms | encode share |
|---|---|---|---|---|---|---|---|
| 4 | 64 | 53 | 8 | 0.043 | 0.038 | 0.037 | 89% |
| 8 | 64 | 105 | 16 | 0.080 | 0.075 | 0.072 | 94% |
| 16 | 64 | 209 | 32 | 0.149 | 0.141 | 0.140 | 95% |
| 32 | 64 | 417 | 64 | 0.303 | 0.297 | 0.279 | 98% |

The Intel B570 shows the same shape (`encode` 0.041 / 0.081 / 0.162 / 0.325 ms
for the same dispatch counts).

Two things follow, and neither was expected:

- **Encoding is ~90–98% of step time**, and it scales linearly with dispatch
  count at a flat **~680 ns per dispatch** (690 / 686 / 700 / 678 ns across a
  8× range). Not a fixed overhead and not queueing: a straight line through the
  origin in dispatch count.
- **`idle_ms` matches `encode_ms`**, which is the control that matters.
  Repeating the encode with the device already idle gives the same number, so
  there is no fence hiding inside `encode_step` and the 3a rebuild is not
  hiding behind a wait. This is what rules out the alternative explanation.

**Consequence for 3a.** `optimizer_units` rebuilds a table sized by *parameters*,
and the sweep shows step time tracking *dispatches* — the two cases that
separate them (`8×64` and `8×1024`, both 105 dispatches and 16 parameters,
0.071 and 0.075 ms) differ by 5%. So 3a is real but small at these sizes; it
would need a parameter-heavy, dispatch-light model to show up, which is not
what this sweep builds. Worth doing for a large model, not worth doing first.

**Attribution: the audit named the wrong targets.** Timing the four phases
inside `record_groups` — pipeline lookup, `pass.with`, `bind_dispatch`,
`pc.dispatch` — gives, per dispatch (instrumented, so upper bounds; ~90 ns of
the total is the instrumentation itself). The instrumentation was temporary and
is not in the tree; `bench_step_cpu` is the part that stays, and the split below
would need the same four `Instant::now()` pairs re-added to reproduce:

| phase | ns/dispatch | share |
|---|---|---|
| `pipelines::get` (the HashMap lookup) | 44 | 7% |
| `pass.with(pipeline)` | 58 | 9% |
| `bind_dispatch` | 187 | 28% |
| `pc.dispatch(workgroups)` — the driver call | 369 | 56% |

So 3b, the hash lookup the audit pointed at, is **7%** of the host path.
Removing it entirely would be a 7% saving on a cost that is itself ~90–98% of a
step — worth about 6% end to end on a small model, and it costs a
`Vec<&ComputePipeline>` plus a lifetime change to `Pipelines`. That is not
nothing, but it is not the win the audit implied, and it should not be the first
thing done.

84% is `bind_dispatch` plus the driver call, and 56% of the whole host path is a
single `pc.dispatch` — a call into Blade that records into the command encoder.
That is not ours to optimise; it is the cost of issuing a dispatch at all.
Which reframes the finding: **the host cost of a step is roughly what issuing
D dispatches costs**, and the only levers that matter are issuing fewer
dispatches or making each one cheaper to record. `bind_dispatch` at 28% is the
part that is ours, and it is a 60-arm match building a struct per dispatch —
which is item 5's god-function problem, now with a number attached.

**What is left to decide.** Whether reducing dispatch count (fewer, wider
dispatches) is available at the shapes that matter, and whether `bind_dispatch`
can avoid constructing its binding struct per call. Both are design questions
with a measurement in hand, which is more than the audit had.

## 4. `step()` fences before recording

**Status: measured. The premise is right and the conclusion is that it does not
matter — do not do this.**

`src/runtime.rs`, `Session::step`:

```rust
self.wait();
self.encoder.start();
self.encode_step(...);
self.sync_point = Some(self.gpu.submit(&mut self.encoder));
```

The `wait()` blocks on the *previous* step's fence before recording starts, so
the host record time is exposed rather than overlapped. The encoder is already
double-buffered (`buffer_count: 2` at `runtime.rs:3548`) and `Session::record`
documents that recording while earlier work is in flight is fine — `step()`
just does not do it.

That reasoning is sound but measures out as not worth acting on. From the
`bench_step_cpu` table, `step_ms - encode_ms` — everything an encoder rotation
could hide — is **0.005 ms at 53 dispatches and 0.006–0.014 ms at 417**, against
an encode cost of 0.038–0.297 ms. The overlappable fraction is 1–5% and does not
grow with dispatch count, because the thing to be hidden is the *submit*, which
is a fixed cost, while the thing actually costing time is the per-dispatch
encode loop, which a rotation does not reduce: the CPU still has to walk every
dispatch.

A two-slot encoder rotation would change submission structure and gain a few
microseconds per step on small models. On the `gpu-gap-2026-09.md` figures that
motivated it (~0.75 ms in `step` against ~3.01 ms waiting) the ratio inverts,
which suggests those measurements were taken with a different cost
distribution — most likely a larger model where dispatch count is higher and
the encode loop dominates, in which case the fix for item 4 would be item 3b,
not a rotation.

**What would reopen this.** A model with enough dispatches that `encode_ms`
exceeds the GPU wait, which is the regime where the two swap. That is a
different regime from the one the audit observed, so re-measure there before
reconsidering.

## 5. Structural items

**Status: open. None are urgent; all are real.**

- **`bind_dispatch` is ~1210 lines** (`src/runtime.rs:6845`) — a 60-arm match
  over `dispatch.shader`, each arm building a binding struct. Measured at
  **28% of the per-step host path** (187 ns of ~657 ns per dispatch), so this is
  the one structural item with a number attached and a plausible payoff.
- **Module sizes**: `runtime.rs` 8578, `codegen.rs` 7976, `compile.rs` 7784,
  `graph.rs` 3409. The file split recommended in July 2026 is still open; the
  one boundary that mattered (compile↔runtime) is now closed.
- **`(m,n,k)` is reinterpreted differently at two binding sites.**
  `src/runtime.rs:6859` and `:7017` both special-case
  `ShaderEntry::MatMul` to swap `n` and `k`. A `Dispatch::mnk()` accessor next
  to the existing `scalar_matmul()` / `conv_k_tile()` / `gemv_shape()` accessors
  would centralise it.
- **Quantized block geometry lives in three independent tables**: the `DType`
  doc comments, five `assert!`s in `Graph::parameter_q*k`, and an inline
  `(block, stride)` match at `runtime.rs:4932`. Two of the Q8_0 numbers differ
  deliberately (Meganeura pads to 36 bytes, GGML uses 34) and that is
  documented, but nothing enforces the relationship.

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

With both corrected, the minimum is stable to within 10% across runs and
policies, and the numbers become interpretable. `bench_step_cpu` uses a median,
which is defensible there because host encode time has no such bimodality — but
that is a property of that measurement, not a general licence.

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