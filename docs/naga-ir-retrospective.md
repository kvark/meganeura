# Direct Naga IR and shader-source complexity

## Summary

Meganeura tried authoring shaders directly as `naga::Module` IR and later
returned to generated/templated WGSL parsed by Naga. The experiment is a good
paper lesson: an IR can be an excellent validation and portability boundary
without being a practical human authoring or debugging format.

WGSL was not required by the architecture. The temporary IR→WGSL→IR
roundtrip was a workaround that let Naga's WGSL writer/parser normalize
manually constructed IR. Even after direct module submission worked, the
authoring burden remained high enough that WGSL templates were preferable.

## Timeline and quantitative evidence

| Date / commit | Change | Size evidence |
|---|---|---:|
| 2026-03-17, `1f7bab3` | Added programmatic Naga modules for 11 shader types | +1,813 lines in `codegen.rs` |
| 2026-03-18, `36fde26` | Deleted static WGSL and moved toward direct Blade module consumption | Commit explicitly retained a WGSL normalization roundtrip because hand-built emit ranges caused SPIR-V caching failures |
| 2026-03-21, `701a28e` | Fixed missing emit ranges, passed raw modules to Blade, and added SPIR-V tests | `codegen.rs` had grown to 3,735 lines |
| 2026-03-27, parent of `133b619` | Direct-IR implementation immediately before the switch | 6,359 lines in `codegen.rs` |
| 2026-03-27, `133b619` | Switched to WGSL templating | 949 Rust lines + 1,402 WGSL lines across 24 files |

The switch reduced the relevant authored source from 6,359 to 2,351 lines,
about 63%, while making generated shaders directly inspectable.

## What made direct IR difficult

- Every scalar, vector, array, struct, global, local, binding, expression,
  statement, and entry point required explicit construction.
- Naga expression arenas use handles whose ordering and `Emit` ranges carry
  non-obvious validity/lifetime requirements.
- Local-variable initialization and pre-emitted expressions required special
  cases in a custom builder.
- A missing range produced backend failures such as
  `Expression [17] is not cached!`; the handle identifies no useful source
  location for a human.
- Bounds-checking transformations could expose an invalid range only in a
  later backend, far from the construction mistake.
- Diff review showed builder mechanics rather than the mathematical kernel.
- Adding attention, normalization, and backward kernels grew the abstraction
  faster than it removed boilerplate.

The specific cached-expression bug was fixed. The conclusion is therefore not
“direct Naga cannot work”; it is “maintaining all of Naga's internal
invariants manually was a poor trade for this project.”

## Why WGSL worked better

- Shader logic remains recognizable to GPU programmers.
- Naga's parser owns arena construction and expression emission invariants.
- Parse and validation errors carry source locations.
- Generated WGSL can be logged, minimized, compiled independently, and
  attached to bug reports.
- Templates still allow target-specific constants, prologues, epilogues, and
  cooperative-matrix variants.
- The runtime still consumes validated Naga modules, preserving the portable
  SPIR-V/Metal path.

The current hybrid is:

```text
typed graph/schedule
        ↓
WGSL template or generator
        ↓
Naga parse + validate
        ↓
Blade / SPIR-V / Metal
```

## Current complexity baseline

As of 2026-07-23:

- 75 WGSL files;
- 6,114 WGSL lines;
- 31 Rust source files under `src/`;
- 33,112 Rust lines under `src/`.

The direct-IR retreat solved authoring verbosity, not shader-family
proliferation. Seventy-five files are too many to present as 75 independent
ideas; most are variants of a smaller number of kernel archetypes. This pass
removed the standalone attention dK and dV shaders after making the fused
dK+dV compiler path an explicit invariant.

## Consolidation plan

Do not perform a large shader rewrite immediately before freezing paper
results. Consolidate in measured stages:

1. **Inventory and manifest.** Generate a machine-readable list of shader
   entry points, layouts, supported dtypes, tile geometry, and owning
   archetype. Make file count and WGSL LOC reproducible metrics.
2. **Matmul family.** Merge scalar, small, GEMV, cooperative, transposed,
   fused-add, and RMSNorm variants behind one typed matmul specification with
   explicit prologue/epilogue objects.
3. **Convolution family.** Express forward, input-gradient, weight-gradient,
   im2col, small, and cooperative variants through shared tiling/staging
   generators instead of copy-adjacent files.
4. **Reduction/normalization family.** Reuse the schedule-layer reduction
   archetype for softmax, sums, norms, and their composable prologues and
   epilogues where performance measurements permit.
5. **Attention family.** Share indexing, masks, and tile descriptions across
   forward, dQ, dK, dV, and fused dKV while retaining kernels that profiles
   show must differ.
6. **Delete only after parity.** For each migration, require shader
   validation, numerical parity, dispatch-count parity, and stable GPU
   performance on NVIDIA and AMD before deleting the old variant.

A useful near-term target is not “one shader.” It is one typed generator per
major archetype with a small number of evidence-backed specialized kernels.

## Paper framing

Use this as a compact case study, not a complaint about Naga:

> We found Naga valuable as a portable validated compiler IR, but
> programmatic authoring exposed low-level expression-emission invariants and
> expanded to 6.4 KLOC. Returning to generated WGSL reduced the corresponding
> implementation by roughly 63%, restored source-located diagnostics, and
> retained the same Naga-to-backend path.

The broader systems lesson is that the best interchange IR and the best
authoring IR need not be the same representation.
