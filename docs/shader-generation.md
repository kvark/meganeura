# Shader generation investigation

Investigated on October 4, 2026, against Meganeura `d30031a`. The counts and
examples below describe that revision, before the repository cleanup.

## Recommendation

Keep Naga as the runtime boundary and WGSL as the generated format. Replace
exact-text shader rewrites with explicit specialization points. Evaluate
syntax-aware composition where transformations actually need to understand
declarations or functions. A wholesale frontend migration is not justified
by the evidence collected here.

Synaga is a plausible authoring frontend. To replace the difficult generators,
it needs runtime specialization and composition as well as more compute
features. Shapes, device capabilities and fused operation graphs are known
after the application is built; generating Rust strings would retain much of
the present machinery.

## Current implementation

The tree contains 89 WGSL files with 6,205 lines and approximately 7,400 lines
of Rust around generation, excluding test modules but including comments.
These are whole-repository counts, not this branch's additions.

- [GEMV specialization](../src/codegen.rs) finds statements, scans to
  semicolons and rewrites declarations, loads and reductions. Some anchors
  are checked; other replacements can silently do nothing. Formatting and
  local names have become an implicit interface.
- Horizontal matmul splits source at `@compute`, removes attributes, renames
  functions and redirects buffer accesses. This is structural manipulation
  expressed as text replacement.
- [Attention generation](../src/codegen/attention.rs) occupies 2,727 lines.
  Shader algorithms are mixed with escaping, indentation and host decisions.
  Stable shader bodies can live in shader files with explicit substitutions.
- The [pointwise DAG emitter](../src/schedule.rs) is comparatively simple:
  operations already have a typed representation, and emitting their WGSL
  expressions is compact. A general Naga builder would add little here.

WGSL is parsed during pipeline construction. Blade receives the resulting
Naga module directly; shader parsing is not per-dispatch work. Removing the
parser would affect construction/tuning cost, not establish a GPU speedup.
Existing variant validation and numerical oracles support a controlled
refactor. This investigation demonstrated maintenance problems, not new
incorrect GPU results.

## Synaga

Tested [synaga `a265b35`](https://github.com/kvark/synaga/tree/a265b35cf3829e88408df734e2a936da0074fb71).
A Rust translation of the plain f32 GEMV passed Rust checking and Naga
validation against Meganeura's exact Naga pin, `323acfb`. Direct module output
and unbound resources named for Blade fit the existing runtime.

The probes rejected symbolic `threads(LANES)`, constant arithmetic, `f16`,
generic functions, subgroup addition and packed integer dot products.
Cooperative matrices are also listed as unsupported in the project's README.
The core compiled against our pin; enabling its optional WGSL exporter failed
on the newer `RayQueryFunction::Begin` variant.

The [runtime API](https://github.com/kvark/synaga/blob/a265b35cf3829e88408df734e2a936da0074fb71/src/lib.rs)
accepts Rust source. A useful next capability for Meganeura would specialize
compiled shaders with constants, types, unrolling and generated
prologue/epilogue functions, without reconstructing whole programs as strings.

## Alternatives

| Candidate | Route to Naga | Findings |
|---|---|---|
| [WESL / wgsl-parse](https://github.com/webgpu-tools/wesl-rs/tree/bc77d256b15c91f15cba87b2756379bd28b69841) | WGSL syntax tree → WGSL → Naga | Imports, conditional compilation and syntax-aware editing address current problems. The parser round-tripped our GEMV and a cooperative-matrix probe successfully. Current main requires Rust 1.97.1, above our declared 1.92 minimum. |
| [naga_oil](https://github.com/bevyengine/naga_oil/tree/32fc887e548ca9d67ba8a6beeed60abb9fda21b7) | Composed WGSL modules → Naga IR | Imports and function overrides provide composition. An imported-epilogue probe worked with released Naga 30. Current oil failed to compile against our git pin; unbound GEMV resources also required non-validating composition followed by our own validation. |
| [wgsl-rs](https://github.com/schell/wgsl-rs/tree/efccd432b197800804e18f441f7763fc94390eeb) | Rust subset → its IR → WGSL → Naga | Supports generic and runtime-instantiated templates, but its current scalar representation lacks f16. More compute-feature work would still be needed. |
| [shame](https://github.com/RayMarch/shame) | Rust shader-building DSL → WGSL → Naga | Runtime metaprogramming fits generation, but introduces another DSL and still emits WGSL. |
| [CubeCL](https://github.com/tracel-ai/cubecl) | Rust → CubeCL IR → WGSL, SPIR-V or Metal | Supports host-time decisions, generics and unrolling. Closest to runtime-specialized compute programming, but adopting its compiler machinery is a larger integration project. |

WESL's experimental [quotation macros](https://github.com/webgpu-tools/wesl-rs/blob/bc77d256b15c91f15cba87b2756379bd28b69841/crates/wesl-quote/README.md)
parse shader syntax during the Rust build and allow expressions and statements
to be injected at runtime. They offer a middle ground between strings and
manually constructing Naga arenas. Only the parser round-trip was tested here,
not the quotation or full WESL linking paths.

[Slang](https://shader-slang.org/slang/user-guide/targets) and
[Rust-GPU](https://github.com/rust-gpu/rust-gpu/releases) offer routes through
WGSL or SPIR-V. Both introduce larger toolchains and still need a solution for
Meganeura's graph-dependent composition. These, wgsl-rs, shame and CubeCL were
assessed from source/documentation rather than integration prototypes.

## Pipeline constants

A GEMV probe using an override for its lane count specialized to 32, 64, 128
and 256 lanes, with workgroup storage of 512, 1,024, 2,048 and 4,096 bytes.
This can reduce numeric source regeneration, while structural variants still
need generation. The probe used a loop for reduction; no performance
equivalence with the unrolled implementation was measured.

The current Blade path resolves overrides before assigning resource bindings.
Naga's override-resolution revalidation rejects those unbound buffers. The
probe reproduced that rejection and passed after assigning temporary bindings.
This integration issue must be fixed before using overrides in Meganeura.

## Evaluation criteria

Use the GEMV family to compare approaches: runtime widths, row grouping,
packed/f16 formats, tree/subgroup reductions and fused norm/add. Compare total
generator and adapter code, numerical results and GPU performance. A fixed
kernel alone does not exercise the complexity driving the current rewrites.

All experimental compilation and validation ran outside the repository under
`/tmp/meganeura-shader-probe*`. No GPU execution, performance comparison or
complete alternative-frontend migration was performed.
