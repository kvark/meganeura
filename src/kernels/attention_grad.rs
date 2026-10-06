//! Attention backward: dQ, and the fused dK + dV.
//!
//! Every path of a part reads the same buffers through the same binding
//! layout and parameters (`[q_seq, kv_seq, heads << 16 | kv_heads,
//! head_dim, window]`), so the paths differ only in geometry, module and
//! what they admit. The one exception is the last input: most paths read
//! `dot(dO, O)` per query row, reduced once before the dispatch, and the
//! f16 cooperative path reads `O` itself.

use super::{Family, Rejection};
use crate::codegen::{ShaderModule, attention_lanes};
use serde::{Deserialize, Serialize};

/// Which gradient a dispatch computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Part {
    /// dQ, one row per query position, over the query heads.
    Q,
    /// dK and dV together, one row per key position, over the KV heads.
    KV,
}

/// The operand type of a cooperative path's matrix products.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Operands {
    /// 8×8 f32 matrices: full precision, so preferred wherever they run.
    F32,
    /// 16×16 f16 matrices. Rounding dO to f16 loses small derivatives, so
    /// this is an opt-in.
    F16,
}

/// How a dispatch computes it, most preferred first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Path {
    /// Cooperative-matrix products over 16-row tiles.
    Cooperative(Operands),
    /// Flash tiling: several rows per workgroup sharing their K/V (or Q/dO)
    /// staging.
    Flash,
    /// One workgroup per row and head. The fallback.
    Rowwise,
}

/// One attention backward kernel: the shader entry
/// [`ShaderEntry::AttentionGrad`](crate::compile::ShaderEntry::AttentionGrad)
/// carries.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AttentionGrad {
    pub part: Part,
    pub path: Path,
}

impl std::fmt::Debug for AttentionGrad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let part = match self.part {
            Part::Q => "dQ",
            Part::KV => "dKV",
        };
        let path = match self.path {
            Path::Cooperative(Operands::F32) => "cooperative-f32",
            Path::Cooperative(Operands::F16) => "cooperative-f16",
            Path::Flash => "flash",
            Path::Rowwise => "rowwise",
        };
        write!(f, "{part}-{path}")
    }
}

/// One dispatch's attention backward.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Problem {
    pub part: Part,
    /// Rows the dispatch covers: query rows for dQ, key rows for dK/dV.
    pub rows: u32,
    /// Rows each of those iterates over: key rows for dQ, query rows for
    /// dK/dV.
    pub other_rows: u32,
    /// Heads the dispatch covers: query heads for dQ, KV heads for dK/dV.
    pub heads: u32,
    pub head_dim: u32,
    /// Elements per thread of the scalar paths.
    pub ept_cap: u32,
    /// Whether a schedule pinned `ept_cap` for this node, which pins a
    /// scalar path.
    pub pinned: bool,
}

/// What the device and compile options allow.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Target {
    /// 16×16 f16 cooperative matrices.
    pub cooperative_f16: bool,
    /// 8×8 f32 cooperative matrices.
    pub cooperative_f32: bool,
    pub shared_memory_bytes: u32,
    /// [`CompileOptions::flash_backward_coop`](crate::CompileOptions).
    pub reduced_precision: bool,
    /// [`CompileOptions::prefer_attention_grad`](crate::CompileOptions).
    pub prefer: Option<Path>,
}

/// Rows per cooperative tile.
const TILE: u32 = 16;

/// The f32 path's only head width, and the extent below which it was not
/// measured to pay off.
const F32_HEAD_DIM: u32 = 64;
const F32_MIN_ROWS: u32 = 128;

/// The widest head the rowwise kernels' 64 lanes cover, four dimensions
/// each.
const ROWWISE_MAX_HEAD_DIM: u32 = 256;

impl Family for Path {
    type Problem = Problem;
    type Target = Target;
    fn preferred(target: &Target) -> Option<Self> {
        target.prefer
    }

    const PREFERENCE: &'static [Self] = &[
        Path::Cooperative(Operands::F32),
        Path::Cooperative(Operands::F16),
        Path::Flash,
        Path::Rowwise,
    ];

    fn admits(self, problem: &Problem, target: &Target) -> Result<(), Rejection> {
        let head_dim = problem.head_dim;
        let kernel = AttentionGrad::new(problem.part, self);
        match self {
            Path::Cooperative(Operands::F32) => {
                if problem.pinned {
                    Err("a schedule pinned a scalar layout")
                } else if !target.cooperative_f32 {
                    Err("no 8x8 f32 cooperative matrices")
                } else if head_dim != F32_HEAD_DIM {
                    Err("head width is not 64")
                } else if problem.rows < F32_MIN_ROWS || problem.other_rows < F32_MIN_ROWS {
                    Err("fewer than 128 rows on a side")
                } else if !crate::compile::workgroups_within_portable_limits(
                    kernel.workgroups(problem),
                ) {
                    Err("grid exceeds the portable dispatch limit")
                } else if kernel.shared_bytes(head_dim) > u64::from(target.shared_memory_bytes) {
                    Err("tiles exceed workgroup memory")
                } else {
                    Ok(())
                }
            }
            Path::Cooperative(Operands::F16) => {
                if problem.pinned {
                    Err("a schedule pinned a scalar layout")
                } else if !target.reduced_precision {
                    Err("reduced-precision backward is not enabled")
                } else if !target.cooperative_f16 {
                    Err("no 16x16 f16 cooperative matrices")
                } else if head_dim < TILE || !head_dim.is_power_of_two() {
                    Err("head width is not a power of two of at least 16")
                } else if problem.rows < TILE {
                    Err("fewer rows than one tile")
                } else if kernel.shared_bytes(head_dim) > u64::from(target.shared_memory_bytes) {
                    Err("tiles exceed workgroup memory")
                } else {
                    Ok(())
                }
            }
            Path::Flash => {
                let rows = flash_rows(head_dim, problem.ept_cap);
                if rows < 2 {
                    Err("head too wide to tile several rows")
                } else if problem.rows < rows {
                    Err("fewer rows than one tile")
                } else {
                    Ok(())
                }
            }
            Path::Rowwise => {
                if head_dim > ROWWISE_MAX_HEAD_DIM {
                    Err("head wider than 256")
                } else {
                    Ok(())
                }
            }
        }
    }
}

/// Rows per workgroup of the flash paths.
fn flash_rows(head_dim: u32, ept_cap: u32) -> u32 {
    let (_, lanes) = attention_lanes(head_dim, ept_cap);
    (256 / lanes).max(1)
}

impl AttentionGrad {
    /// Every kernel of the family.
    pub const ALL: [Self; 8] = [
        Self::new(Part::Q, Path::Cooperative(Operands::F32)),
        Self::new(Part::Q, Path::Cooperative(Operands::F16)),
        Self::new(Part::Q, Path::Flash),
        Self::new(Part::Q, Path::Rowwise),
        Self::new(Part::KV, Path::Cooperative(Operands::F32)),
        Self::new(Part::KV, Path::Cooperative(Operands::F16)),
        Self::new(Part::KV, Path::Flash),
        Self::new(Part::KV, Path::Rowwise),
    ];

    pub const fn new(part: Part, path: Path) -> Self {
        Self { part, path }
    }

    /// The preferred kernel for `problem` on `target`.
    pub(crate) fn select(problem: &Problem, target: &Target) -> Self {
        Self::new(problem.part, super::select::<Path>(problem, target).chosen)
    }

    /// Workgroups covering `problem`.
    pub(crate) fn workgroups(self, problem: &Problem) -> [u32; 3] {
        let per_group = match self.path {
            Path::Cooperative(_) => TILE,
            Path::Flash => flash_rows(problem.head_dim, problem.ept_cap),
            Path::Rowwise => 1,
        };
        [problem.rows.div_ceil(per_group), problem.heads, 1]
    }

    /// Whether the last input is `dot(dO, O)` per query row rather than `O`.
    pub(crate) fn reads_row_dot(self) -> bool {
        self.path != Path::Cooperative(Operands::F16)
    }

    /// Whether the scalar layout's elements per thread shape the module,
    /// so the dispatch has to carry them.
    pub(crate) fn takes_ept_cap(self) -> bool {
        self.path == Path::Flash
    }

    /// Whether the module is generated for the head width.
    pub(crate) fn specializes_head_dim(self) -> bool {
        self.path != Path::Rowwise
    }

    /// Workgroup memory of the cooperative tiles.
    pub(crate) fn shared_bytes(self, head_dim: u32) -> u64 {
        let (per_dim, fixed) = match (self.path, self.part) {
            (Path::Cooperative(Operands::F32), _) => {
                return u64::from(crate::codegen::FLASH_GRAD_COOP_F32_SHARED_BYTES);
            }
            (Path::Cooperative(Operands::F16), Part::Q) => (192u64, 3264),
            (Path::Cooperative(Operands::F16), Part::KV) => (256, 4544),
            (Path::Flash | Path::Rowwise, _) => (0, 0),
        };
        per_dim * u64::from(head_dim) + fixed
    }

    /// The shader capabilities the module needs.
    pub fn capabilities(self) -> naga::valid::Capabilities {
        use naga::valid::Capabilities as C;
        match self.path {
            Path::Cooperative(Operands::F32) => C::COOPERATIVE_MATRIX | C::SUBGROUP,
            Path::Cooperative(Operands::F16) => {
                C::COOPERATIVE_MATRIX | C::SHADER_FLOAT16 | C::SUBGROUP
            }
            Path::Flash | Path::Rowwise => C::empty(),
        }
    }

    /// The module for `head_dim`, with `ept_cap` elements per thread on
    /// the flash path (the knobs' default when `None`).
    pub fn generate(
        self,
        head_dim: u32,
        ept_cap: Option<u32>,
        knobs: &crate::compile::TuningKnobs,
    ) -> ShaderModule {
        use crate::codegen as cg;
        match (self.path, self.part) {
            (Path::Cooperative(Operands::F32), Part::Q) => {
                cg::generate_flash_grad_q_coop_f32_module(head_dim)
            }
            (Path::Cooperative(Operands::F32), Part::KV) => {
                cg::generate_flash_grad_kv_coop_f32_module(head_dim)
            }
            (Path::Cooperative(Operands::F16), Part::Q) => {
                cg::generate_flash_grad_q_coop_f16_module(head_dim)
            }
            (Path::Cooperative(Operands::F16), Part::KV) => {
                cg::generate_flash_grad_kv_coop_f16_module(head_dim)
            }
            (Path::Flash, Part::Q) => cg::generate_flash_grad_q_module(
                head_dim,
                ept_cap.unwrap_or(knobs.flash_grad_q_ept_cap),
            ),
            (Path::Flash, Part::KV) => cg::generate_flash_grad_kv_module(
                head_dim,
                ept_cap.unwrap_or(knobs.flash_grad_kv_ept_cap),
            ),
            (Path::Rowwise, Part::Q) => {
                ShaderModule::new(include_str!("../shaders/mha_grad_q.wgsl"))
            }
            (Path::Rowwise, Part::KV) => {
                ShaderModule::new(include_str!("../shaders/mha_grad_kv.wgsl"))
            }
        }
    }

    /// The module's global bindings, in binding order.
    #[cfg(test)]
    pub(crate) fn globals(self) -> &'static [&'static str] {
        match self.part {
            Part::Q => &[
                "d_out", "src_a", "src_b", "bias", "lse", "fwd_dst", "dst", "params",
            ],
            Part::KV => &[
                "d_out", "src_a", "src_b", "bias", "lse", "fwd_dst", "dst", "dst2", "params",
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem(part: Part, rows: u32, head_dim: u32) -> Problem {
        Problem {
            part,
            rows,
            other_rows: rows,
            heads: 2,
            head_dim,
            ept_cap: 4,
            pinned: false,
        }
    }

    const ROOMY: Target = Target {
        cooperative_f16: true,
        cooperative_f32: false,
        shared_memory_bytes: 1 << 20,
        reduced_precision: true,
        prefer: None,
    };

    #[test]
    fn preference_lists_every_path_once_and_ends_in_the_fallback() {
        assert_eq!(Path::PREFERENCE.last(), Some(&Path::Rowwise));
        for part in [Part::Q, Part::KV] {
            let paths: Vec<_> = AttentionGrad::ALL
                .iter()
                .filter(|kernel| kernel.part == part)
                .map(|kernel| kernel.path)
                .collect();
            assert_eq!(paths, Path::PREFERENCE);
        }
    }

    /// The fallback admits every head the builders accept, whatever the
    /// target, so selection never fails on a supported problem.
    #[test]
    fn rowwise_admits_every_supported_problem() {
        let bare = Target {
            cooperative_f16: false,
            cooperative_f32: false,
            shared_memory_bytes: 0,
            reduced_precision: false,
            prefer: None,
        };
        for part in [Part::Q, Part::KV] {
            for rows in [1, 2, 15, 16, 129] {
                for head_dim in [1, 3, 16, 48, 64, 100, 256] {
                    let p = problem(part, rows, head_dim);
                    let f32 = Target {
                        cooperative_f32: true,
                        ..ROOMY
                    };
                    for target in [bare, ROOMY, f32] {
                        assert_eq!(Path::Rowwise.admits(&p, &target), Ok(()));
                        let chosen = AttentionGrad::select(&p, &target);
                        assert_eq!(chosen.part, part);
                        let [groups, heads, 1] = chosen.workgroups(&p) else {
                            unreachable!()
                        };
                        assert_eq!(heads, 2);
                        assert!(groups >= 1);
                    }
                }
            }
        }
    }

    #[test]
    fn cooperative_states_why_it_declines() {
        const F16: Path = Path::Cooperative(Operands::F16);
        let p = problem(Part::KV, 64, 64);
        assert_eq!(F16.admits(&p, &ROOMY), Ok(()));
        let cases: [(Problem, Target, Rejection); 6] = [
            (
                Problem { pinned: true, ..p },
                ROOMY,
                "a schedule pinned a scalar layout",
            ),
            (
                p,
                Target {
                    reduced_precision: false,
                    ..ROOMY
                },
                "reduced-precision backward is not enabled",
            ),
            (
                p,
                Target {
                    cooperative_f16: false,
                    ..ROOMY
                },
                "no 16x16 f16 cooperative matrices",
            ),
            (
                problem(Part::KV, 64, 48),
                ROOMY,
                "head width is not a power of two of at least 16",
            ),
            (problem(Part::KV, 15, 64), ROOMY, "fewer rows than one tile"),
            (
                p,
                Target {
                    shared_memory_bytes: 16_384,
                    ..ROOMY
                },
                "tiles exceed workgroup memory",
            ),
        ];
        for (p, target, reason) in cases {
            let selection = super::super::select::<Path>(&p, &target);
            assert_ne!(selection.chosen, F16);
            assert_eq!(selection.declined[1], (F16, reason));
        }
    }

    #[test]
    fn full_precision_cooperative_is_preferred_where_it_runs() {
        const F32: Path = Path::Cooperative(Operands::F32);
        let target = Target {
            cooperative_f32: true,
            ..ROOMY
        };
        for part in [Part::Q, Part::KV] {
            let p = problem(part, 128, 64);
            assert_eq!(AttentionGrad::select(&p, &target).path, F32);
            // Full precision needs no opt-in.
            let strict = Target {
                reduced_precision: false,
                cooperative_f16: false,
                ..target
            };
            assert_eq!(AttentionGrad::select(&p, &strict).path, F32);
            for (p, reason) in [
                (problem(part, 128, 128), "head width is not 64"),
                (problem(part, 127, 64), "fewer than 128 rows on a side"),
                (
                    Problem {
                        other_rows: 127,
                        ..p
                    },
                    "fewer than 128 rows on a side",
                ),
                (
                    problem(part, 16 * 65_536, 64),
                    "grid exceeds the portable dispatch limit",
                ),
            ] {
                assert_eq!(F32.admits(&p, &target), Err(reason));
            }
            let small = Target {
                shared_memory_bytes: crate::codegen::FLASH_GRAD_COOP_F32_SHARED_BYTES - 1,
                ..target
            };
            assert_eq!(F32.admits(&p, &small), Err("tiles exceed workgroup memory"));
        }
    }

    /// A preferred path is tried first and still has to admit the problem.
    #[test]
    fn preference_comes_first_but_still_admits() {
        let p = problem(Part::Q, 256, 64);
        let rowwise = Target {
            prefer: Some(Path::Rowwise),
            ..ROOMY
        };
        assert_eq!(AttentionGrad::select(&p, &rowwise).path, Path::Rowwise);
        let f32 = Target {
            prefer: Some(Path::Cooperative(Operands::F32)),
            ..ROOMY
        };
        assert_eq!(
            AttentionGrad::select(&p, &f32).path,
            Path::Cooperative(Operands::F16)
        );
    }

    #[test]
    fn names_are_stable() {
        let names: Vec<_> = AttentionGrad::ALL
            .iter()
            .map(|kernel| format!("{kernel:?}"))
            .collect();
        assert_eq!(
            names,
            [
                "dQ-cooperative-f32",
                "dQ-cooperative-f16",
                "dQ-flash",
                "dQ-rowwise",
                "dKV-cooperative-f32",
                "dKV-cooperative-f16",
                "dKV-flash",
                "dKV-rowwise"
            ]
        );
    }
}
