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
| 1 | Cooperative flash forward loses f16 precision on NVIDIA | open — needs measurement | below |
| 2 | `load::onnx` slices with a file-controlled length | **done** | below |
| 3 | Per-step CPU work in the step loop | **measured — the cost is real; the two targets named were the wrong ones** | below |
| 4 | `step()` fences before recording, so CPU and GPU never overlap | **measured — not worth doing** | below |
| 3c | CPU/GPU split benchmark (`bench_step_cpu`) | done | below |
| 5 | Structural: god functions, duplicated `(m,n,k)`, split tables | open | below |
| 6 | Importers accept malformed input without a fuzz corpus | open | below |
| 7 | `EntrySpec` table for `ShaderEntry` | retracted | below |
| 8 | RMSNorm `(2..=32)` rows-per-workgroup bound | retracted | below |

---

## 1. Cooperative flash forward loses f16 precision on NVIDIA

**Status: open. This is the only item here that is a defect in a shipped path
rather than a harness artefact or a tidiness question.**

Five `attention` oracle cases fail on the RTX 5070 and pass on Intel, lavapipe
and every other adapter. The worst element of every failing comparison is
*exactly* `f16(reference)`:

| got | reference | `f16(reference)` |
|---|---|---|
| `-1.052734375` | `-1.0522552728652954` | `-1.052734375` |
| `-1.03515625` | `-1.0346871614456177` | `-1.03515625` |
| `0.5654296875` | `0.5656632781028748` | `0.5654296875` |
| `0.5009765625` | `0.5007421970367432` | `0.5009765625` |

Isolated by elimination, not inference:

- `MEGANEURA_FLASH_FWD_COOP=0` → 18 passed, 0 failed. The only switch that
  changes the outcome.
- `MEGANEURA_DISABLE_COOP=1` → 18 passed, 0 failed, and the dumped plans are
  **byte-identical** to the default. So it is not dispatch selection.
- On this device `runtime::auto_tune` reports `f16_tile: 16, f32_tile: 0` — no
  f32 cooperative tile is advertised at all — and `CoopPolicy::Auto` selects the
  f16 one.
- `compile::attention_dispatch` gates the choice on `!requires_full_precision`,
  which an inference graph does not set.

This was invisible for as long as the oracle silently fell through to the Intel
card after exhausting the NVIDIA context budget (see item 9 below); the oracle
was right to fail on it.

**What would close it.** A measurement of what each option costs on the shapes
that matter, because the fix is a policy choice rather than a bug fix:

- keep `Auto` (f16 coop forward) — fastest, loses ~2 decimal digits
- make `NativeF32` the default — the device has no f32 tile, so this selects
  the scalar kernel for these shapes
- gate coop forward on `requires_full_precision` for inference too

The third is a one-line change and would be wrong without the measurement: it
gives up the ~3.2x-per-dispatch speedup the code comments cite. Recorded rather
than papered over; loosening the oracle's tolerance to hide it would be worse
than the defect.

---

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
the total is the instrumentation itself):

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

**Status: open.**

`tests/oracle/fuzz.rs` is hand-rolled — no `proptest`, `quickcheck` or
`cargo-fuzz` anywhere in the tree — and covers random *graphs* through the
compiler, not random *files* through the importers. `load::onnx` has 6 unit
tests and `load::nnef` has 4, all with well-formed input. Item 2 is the
sharpest instance of what that leaves open.

The existing fuzzer's op mix is also narrow: 18 op kinds, all 2-D
elementwise/contraction, over dims drawn from `[1, 3, 4, 8, 17]` — every
dimension tiny or odd, so tile-edge behaviour is never randomly explored, and
no attention, convolution or long-axis reduction appears. 65 graphs by
default; `ORACLE_FUZZ_COUNT=400` is opt-in.

**What would close it.** `proptest` as a dev-dependency with a byte-oriented
strategy for the ONNX and NNEF readers — arbitrary bytes, not well-formed
protobuf — asserting the parse either succeeds or returns an error, never
panics. The exit criterion is a malformed-input corpus in the repository.

---

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

## 10. Claims retracted during this work

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