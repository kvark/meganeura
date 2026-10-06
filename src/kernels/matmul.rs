//! Dense matrix products: `C = A·B`, `Aᵀ·B` and `A·Bᵀ`, each optionally
//! `+ D`.
//!
//! The compiler emits each with the 64-wide scalar tile kernel. Session
//! construction then picks among the paths here, and tuning challenges the
//! choice by measurement; both ask [`cooperative_geometry`] whether a
//! cooperative kernel is legal, so the two cannot drift apart.
//!
//! Single-row products (GEMV), block-diagonal and batch-major layouts and
//! convolutions are separate computations with their own kernels.

use super::{Family, Rejection};
use crate::codegen::{CoopConfig, ShaderGroup};
use crate::compile::{Dispatch, Kernel};

/// Which product a dispatch computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Layout {
    /// `A·B`.
    Plain,
    /// `A·B + D`.
    PlainAdd,
    /// `Aᵀ·B`.
    TransposedA,
    /// `Aᵀ·B + D`.
    TransposedAAdd,
    /// `A·Bᵀ`.
    TransposedB,
    /// `A·Bᵀ + D`.
    TransposedBAdd,
}

impl Layout {
    pub(crate) fn of(group: ShaderGroup) -> Option<Self> {
        Some(match group {
            ShaderGroup::MatMul => Self::Plain,
            ShaderGroup::MatMulAdd => Self::PlainAdd,
            ShaderGroup::MatMulAT => Self::TransposedA,
            ShaderGroup::MatMulATAdd => Self::TransposedAAdd,
            ShaderGroup::MatMulBT => Self::TransposedB,
            ShaderGroup::MatMulBTAdd => Self::TransposedBAdd,
            _ => return None,
        })
    }

    fn has_addend(self) -> bool {
        matches!(
            self,
            Self::PlainAdd | Self::TransposedAAdd | Self::TransposedBAdd
        )
    }

    /// Transposed products with an addend have cooperative kernels that
    /// only measurement selects, and no 32-wide scalar kernel.
    fn transposed_with_addend(self) -> bool {
        matches!(self, Self::TransposedAAdd | Self::TransposedBAdd)
    }
}

/// One dispatch's product.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Problem {
    pub layout: Layout,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    /// Packed or f16 weights, which only the scalar kernels decode.
    pub reduced_storage: bool,
    /// Autodiff's mark on derivative work that must keep f32 operands.
    pub requires_full_precision: bool,
    /// A fused epilogue that binds buffers of its own, which only the
    /// scalar kernels stage.
    pub epilogue_inputs: bool,
    /// A measured schedule pinned a scalar or split-K kernel.
    pub pinned: bool,
    /// Workgroups of the compiled 64-wide tiling.
    pub compiled_workgroups: u32,
}

impl Problem {
    /// A product with nothing pinned or fused, as tuning measures it.
    pub(crate) fn plain(layout: Layout, [m, n, k]: [u32; 3], reduced_storage: bool) -> Self {
        Self {
            layout,
            m,
            n,
            k,
            reduced_storage,
            requires_full_precision: false,
            epilogue_inputs: false,
            pinned: false,
            compiled_workgroups: m.div_ceil(64) * n.div_ceil(64),
        }
    }

    /// The product `dispatch` computes, if it is a dense one.
    pub(crate) fn of(dispatch: &Dispatch) -> Option<Self> {
        let layout = Layout::of(dispatch.shader.shader_group())?;
        let p = &dispatch.params;
        let (m, n, k) = match layout {
            Layout::Plain | Layout::PlainAdd => (p[0], p[2], p[1]),
            Layout::TransposedA
            | Layout::TransposedAAdd
            | Layout::TransposedB
            | Layout::TransposedBAdd => (p[0], p[1], p[2]),
        };
        Some(Self {
            layout,
            m,
            n,
            k,
            reduced_storage: dispatch.weight_format.uses_reduced_storage(),
            requires_full_precision: dispatch.requires_full_precision,
            epilogue_inputs: dispatch
                .matmul_epilogue
                .as_ref()
                .is_some_and(|epilogue| !epilogue.inputs.is_empty()),
            pinned: dispatch.scalar_matmul().is_some()
                || matches!(dispatch.kernel, Kernel::SplitMatmul { .. }),
            compiled_workgroups: dispatch.workgroups.iter().product(),
        })
    }
}

/// What the device and session allow.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Target {
    pub cooperative: Option<CoopConfig>,
    /// [`CoopPolicy::AllowF16`](crate::CoopPolicy): f16 operands even for
    /// work that requires full precision.
    pub allow_raw_f16: bool,
}

/// How a dispatch computes its product, most preferred first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Path {
    /// Cooperative-matrix tiles in the device's configuration.
    Cooperative,
    /// 32-wide scalar tiles, for products too small to occupy the device
    /// with 64-wide ones.
    SmallTile,
    /// The kernel the compiler emitted: 64-wide scalar tiles, or the
    /// schedule a measurement pinned. The fallback.
    Compiled,
}

/// Output workgroups below which f16 cooperative staging costs more than
/// it saves. Representative transformer projections with 20--72 tiles
/// regress on discrete NVIDIA GPUs; wider backward and MLP products win.
const MIN_F16_COOPERATIVE_WORKGROUPS: u64 = 128;
/// The same for f32 cooperative tiles, whose staging is cheaper.
const MIN_F32_COOPERATIVE_WORKGROUPS: u64 = 16;
/// 64-wide tilings with fewer workgroups than this switch to 32-wide tiles,
/// four times as many, to occupy GPUs with many SMs.
const SMALL_TILE_BELOW_WORKGROUPS: u32 = 16;

impl Family for Path {
    type Problem = Problem;
    type Target = Target;
    const PREFERENCE: &'static [Self] = &[Path::Cooperative, Path::SmallTile, Path::Compiled];

    fn admits(self, problem: &Problem, target: &Target) -> Result<(), Rejection> {
        match self {
            Path::Cooperative => {
                let Some(ref config) = target.cooperative else {
                    return Err("no cooperative matrices");
                };
                if problem.pinned {
                    Err("a schedule pinned a scalar kernel")
                } else if problem.layout.transposed_with_addend() {
                    Err("only measurement promotes transposed products with an addend")
                } else if config.use_f16_input
                    && problem.requires_full_precision
                    && !target.allow_raw_f16
                {
                    // A hi/lo f16 split improves the mantissa but not f16's
                    // exponent range: values below its smallest subnormal
                    // vanish in both halves.
                    Err("f16 operands would lose required precision")
                } else if problem.epilogue_inputs {
                    Err("the epilogue binds buffers of its own")
                } else {
                    let geometry = cooperative_geometry(problem, config)?;
                    let workgroups: u64 = geometry.grid.into_iter().map(u64::from).product();
                    let minimum = if config.use_f16_input {
                        MIN_F16_COOPERATIVE_WORKGROUPS
                    } else {
                        MIN_F32_COOPERATIVE_WORKGROUPS
                    };
                    if workgroups < minimum {
                        Err("too few tiles to amortize staging")
                    } else {
                        Ok(())
                    }
                }
            }
            Path::SmallTile => {
                if problem.pinned {
                    Err("a schedule pinned a scalar kernel")
                } else if problem.reduced_storage {
                    Err("packed weights have no 32-wide kernel")
                } else if problem.layout.transposed_with_addend() {
                    Err("no 32-wide kernel for transposed products with an addend")
                } else if problem.compiled_workgroups >= SMALL_TILE_BELOW_WORKGROUPS {
                    Err("64-wide tiles already occupy the device")
                } else {
                    Ok(())
                }
            }
            Path::Compiled => Ok(()),
        }
    }
}

/// The preferred path for `problem` on `target`.
pub(crate) fn select(problem: &Problem, target: &Target) -> Path {
    super::select::<Path>(problem, target).chosen
}

/// Where a cooperative kernel writes and what it needs allocated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CooperativeGeometry {
    /// Workgroups: rows of output tiles on X, columns on Y.
    pub grid: [u32; 3],
    /// Bytes of output (and of a padded addend) covering whole tiles.
    pub output_bytes: usize,
    /// Whether the addend is read without bounds checks and so needs the
    /// same padding as the output.
    pub pad_addend: bool,
}

/// Whether a cooperative kernel in `config` computes `problem`, and its
/// geometry if so. Legality only: whether it pays off is
/// [`Family::admits`]'s question, or measurement's.
pub(crate) fn cooperative_geometry(
    problem: &Problem,
    config: &CoopConfig,
) -> Result<CooperativeGeometry, Rejection> {
    let tile = config.matmul_output_tile();
    let grid = [problem.m.div_ceil(tile), problem.n.div_ceil(tile), 1];
    // Direct stores write whole sub-tiles, so N must stay aligned for
    // stores never to straddle rows. The bottom edge is safe: the output
    // allocation rounds M up to whole tiles.
    if problem.reduced_storage {
        Err("packed weights have no cooperative kernel")
    } else if !problem.n.is_multiple_of(16) {
        Err("N is not a multiple of 16")
    } else if matches!(problem.layout, Layout::Plain | Layout::PlainAdd) && problem.k < 4 {
        Err("K is below one vec4 load")
    } else if !crate::compile::workgroups_within_portable_limits(grid) {
        Err("grid exceeds the portable dispatch limit")
    } else {
        let output_bytes = crate::compile::cooperative_output_bytes(problem.m, problem.n, 1, tile)
            .ok_or("padded output exceeds the address space")?;
        // The dense f32 8x8 kernel stages its addend through checked loads.
        let checked_addend = config.tile_size == 8 && !config.use_f16_input;
        Ok(CooperativeGeometry {
            grid,
            output_bytes,
            pad_addend: problem.layout.has_addend() && !checked_addend,
        })
    }
}

/// Scalar tiled workgroups: columns of output tiles on X, rows on Y.
pub(crate) fn tiled_workgroups(problem: &Problem, tile: u32) -> [u32; 3] {
    [problem.n.div_ceil(tile), problem.m.div_ceil(tile), 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn product(m: u32, n: u32, k: u32) -> Problem {
        Problem::plain(Layout::Plain, [m, n, k], false)
    }

    const F16: CoopConfig = CoopConfig {
        tile_size: 16,
        use_f16_input: true,
        compensated: false,
    };
    const F32: CoopConfig = CoopConfig {
        tile_size: 8,
        use_f16_input: false,
        compensated: false,
    };

    fn target(config: &CoopConfig) -> Target {
        Target {
            cooperative: Some(*config),
            allow_raw_f16: false,
        }
    }

    #[test]
    fn compiled_admits_everything() {
        let bare = Target {
            cooperative: None,
            allow_raw_f16: false,
        };
        for problem in [product(1, 1, 1), product(4096, 4096, 4096)] {
            assert_eq!(Path::Compiled.admits(&problem, &bare), Ok(()));
        }
        assert_eq!(select(&product(4096, 4096, 64), &bare), Path::Compiled);
        assert_eq!(select(&product(64, 64, 64), &bare), Path::SmallTile);
    }

    #[test]
    fn cooperative_states_why_it_declines() {
        let big = product(1024, 1024, 256);
        assert_eq!(select(&big, &target(&F16)), Path::Cooperative);
        for (problem, config, reason) in [
            (
                Problem {
                    requires_full_precision: true,
                    ..big
                },
                F16,
                "f16 operands would lose required precision",
            ),
            (
                Problem {
                    pinned: true,
                    ..big
                },
                F32,
                "a schedule pinned a scalar kernel",
            ),
            (
                Problem {
                    epilogue_inputs: true,
                    ..big
                },
                F32,
                "the epilogue binds buffers of its own",
            ),
            (product(1024, 1000, 256), F32, "N is not a multiple of 16"),
            (product(1024, 1024, 3), F32, "K is below one vec4 load"),
            (
                product(256, 256, 64),
                F16,
                "too few tiles to amortize staging",
            ),
        ] {
            assert_eq!(
                Path::Cooperative.admits(&problem, &target(&config)),
                Err(reason)
            );
        }
        // Full precision stays eligible on f32 tiles, and on f16 tiles when
        // the session allows it.
        let derivative = Problem {
            requires_full_precision: true,
            ..big
        };
        assert_eq!(select(&derivative, &target(&F32)), Path::Cooperative);
        let allowed = Target {
            allow_raw_f16: true,
            ..target(&F16)
        };
        assert_eq!(select(&derivative, &allowed), Path::Cooperative);
    }

    #[test]
    fn transposed_addends_are_left_to_measurement() {
        for layout in [Layout::TransposedAAdd, Layout::TransposedBAdd] {
            let problem = Problem::plain(layout, [1024, 1024, 256], false);
            assert_eq!(select(&problem, &target(&F32)), Path::Compiled);
            let small = Problem::plain(layout, [32, 32, 32], false);
            assert_eq!(select(&small, &target(&F32)), Path::Compiled);
            assert!(cooperative_geometry(&problem, &F32).is_ok());
        }
    }

    #[test]
    fn only_unchecked_addends_are_padded() {
        let add = Problem {
            layout: Layout::PlainAdd,
            ..product(100, 64, 64)
        };
        assert!(!cooperative_geometry(&add, &F32).unwrap().pad_addend);
        assert!(cooperative_geometry(&add, &F16).unwrap().pad_addend);
        let geometry = cooperative_geometry(&add, &F32).unwrap();
        assert_eq!(geometry.grid, [4, 2, 1]);
        assert_eq!(geometry.output_bytes, 128 * 64 * 4);
    }
}
