//! Convolutions lowered to implicit GEMM: the forward pass, and the input
//! and weight gradients.
//!
//! Two decisions are made here. The compiler picks a scalar register tile
//! ([`register_tile`]), which names the shader entry and which tuning may
//! later measure against the others. Session construction then decides
//! whether a cooperative kernel replaces the 64-wide one, through the
//! [`Path`] family. Depthwise and Winograd convolutions are separate
//! computations with their own kernels.

use super::{Family, Rejection};
use crate::codegen::{CoopConfig, ShaderGroup};
use crate::compile::{Dispatch, Kernel, ShaderEntry};

/// Which product a dispatch computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// `out[Co, oH·oW] = W[Co, Ci·kH·kW] · im2col(x)`.
    Forward,
    /// `dx[Ci, H·W] = Wᵀ · col2im(dy)`.
    GradInput,
    /// `dW[Co, Ci·kH·kW] = dy · im2col(x)ᵀ`.
    GradWeight,
}

/// Register tiles of the scalar kernels, widest first.
const TILES: [u32; 3] = [64, 32, 16];
/// Workgroups a scalar tiling needs to occupy the device.
const OCCUPIED: u32 = 64;
/// With f32 cooperative matrices, the 64-wide tile is kept from this many
/// workgroups on, so session construction can promote it.
const COOPERATIVE_CANDIDATE: u32 = 16;

/// The scalar register tile for a `rows × cols` product over `batch`
/// images: the widest that still occupies the device, or 16 when none
/// does. A device with f32 cooperative matrices keeps the 64-wide tile
/// from 16 workgroups on, since only that entry is promoted.
pub(crate) fn register_tile(rows: u32, cols: u32, batch: u32, f32_cooperative: bool) -> u32 {
    let batch = batch.max(1);
    let groups = |tile: u32| rows.div_ceil(tile) * cols.div_ceil(tile) * batch;
    if f32_cooperative && groups(64) >= COOPERATIVE_CANDIDATE {
        return 64;
    }
    TILES
        .into_iter()
        .find(|&tile| groups(tile) >= OCCUPIED)
        .unwrap_or(16)
}

/// The scalar shader entry for `kind` at `tile`.
pub(crate) fn entry(kind: Kind, tile: u32) -> ShaderEntry {
    match (kind, tile) {
        (Kind::Forward, 16) => ShaderEntry::Conv2dGemm16,
        (Kind::Forward, 32) => ShaderEntry::Conv2dGemmSmall,
        (Kind::Forward, _) => ShaderEntry::Conv2dGemm,
        (Kind::GradInput, 16) => ShaderEntry::Conv2dGradInputGemm16,
        (Kind::GradInput, 32) => ShaderEntry::Conv2dGradInputGemmSmall,
        (Kind::GradInput, _) => ShaderEntry::Conv2dGradInputGemm,
        (Kind::GradWeight, 16) => ShaderEntry::Conv2dGradWeightGemm16,
        (Kind::GradWeight, 32) => ShaderEntry::Conv2dGradWeightGemmSmall,
        (Kind::GradWeight, _) => ShaderEntry::Conv2dGradWeightGemm,
    }
}

/// Scalar convolution with its geometry baked into the pipeline, using
/// exact reciprocal multipliers as constants and the K stage the uniform
/// shader uses. The uniform software divisor stays available to tuning.
pub(crate) fn exact_kernel() -> Kernel {
    Kernel::SpecializedConv {
        k_tile: EXACT_K_TILE,
    }
}

/// The K stage of [`exact_kernel`]. Any other stage was measured and pinned.
const EXACT_K_TILE: u32 = 16;

/// One 64-wide dispatch that cooperative tiles may replace.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Problem {
    pub kind: Kind,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub batch: u32,
    /// Kernel height, width and stride, which the generated cooperative
    /// kernels are specialized for.
    pub window: [u32; 3],
    /// Tuning pinned a K stage other than the compiled one.
    pub pinned: bool,
    pub reduced_storage: bool,
    pub requires_full_precision: bool,
    /// A fused epilogue that binds buffers of its own.
    pub epilogue_inputs: bool,
}

impl Problem {
    /// The convolution `dispatch` computes, if its 64-wide entry has a
    /// cooperative form. The narrower tiles and the weight gradient do not.
    pub(crate) fn of(dispatch: &Dispatch) -> Option<Self> {
        let p = &dispatch.params;
        // params: [batch, in_channels, in_h, in_w, out_channels, kernel_h,
        // kernel_w, stride, padding_h, out_h, out_w, padding_w]
        let (kind, m, n, k) = match dispatch.shader.shader_group() {
            ShaderGroup::Conv2dGemm => (Kind::Forward, p[4], p[9] * p[10], p[1] * p[5] * p[6]),
            ShaderGroup::Conv2dGradInputGemm => {
                (Kind::GradInput, p[1], p[2] * p[3], p[4] * p[5] * p[6])
            }
            _ => return None,
        };
        Some(Self {
            kind,
            m,
            n,
            k,
            batch: p[0],
            window: [p[5], p[6], p[7]],
            pinned: dispatch.conv_k_tile().is_some_and(|k| k != EXACT_K_TILE),
            reduced_storage: dispatch.weight_format.uses_reduced_storage(),
            requires_full_precision: dispatch.requires_full_precision,
            epilogue_inputs: dispatch
                .matmul_epilogue
                .as_ref()
                .is_some_and(|epilogue| !epilogue.inputs.is_empty()),
        })
    }

    /// The generated cooperative entry for this kind and window.
    pub(crate) fn cooperative_entry(&self) -> ShaderEntry {
        let [kh, kw, stride] = self.window;
        match self.kind {
            Kind::Forward => ShaderEntry::Conv2dGemmCoopGen(kh, kw, stride),
            Kind::GradInput => ShaderEntry::Conv2dGradInputGemmCoopGen(kh, kw, stride),
            Kind::GradWeight => unreachable!("the weight gradient has no cooperative kernel"),
        }
    }
}

/// What the device and session allow.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Target {
    pub cooperative: Option<CoopConfig>,
    /// [`CoopPolicy::AllowF16`](crate::CoopPolicy).
    pub allow_raw_f16: bool,
}

/// How a 64-wide convolution runs, most preferred first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Path {
    /// A generated cooperative-matrix kernel for the window.
    Cooperative,
    /// The scalar kernel the compiler emitted. The fallback.
    Compiled,
}

/// Cooperative tiles below which staging costs more than it saves: f16
/// tiles lose on discrete NVIDIA GPUs at low occupancy, and the input
/// gradient's scalar staging runs 64 threads against the scalar kernel's
/// 256.
const MIN_F16_WORKGROUPS: u64 = 128;
const MIN_F32_WORKGROUPS: u64 = 16;
const MIN_GRAD_INPUT_WORKGROUPS: u64 = 16;

impl Family for Path {
    type Problem = Problem;
    type Target = Target;
    const PREFERENCE: &'static [Self] = &[Path::Cooperative, Path::Compiled];

    fn admits(self, problem: &Problem, target: &Target) -> Result<(), Rejection> {
        let Path::Cooperative = self else {
            return Ok(());
        };
        let Some(ref config) = target.cooperative else {
            return Err("no cooperative matrices");
        };
        let f32_8x8 = !config.use_f16_input && config.tile_size == 8;
        if problem.pinned {
            Err("tuning pinned a scalar K stage")
        } else if config.use_f16_input && problem.requires_full_precision && !target.allow_raw_f16 {
            Err("f16 operands would lose required precision")
        } else if problem.epilogue_inputs {
            Err("the epilogue binds buffers of its own")
        } else if problem.reduced_storage {
            Err("packed weights keep their compiled kernel")
        } else if f32_8x8 && problem.kind == Kind::Forward {
            // Policy: the forward kernel keeps the older two-by-two tile,
            // and scalar im2col measured faster on the 8x8 f32 device.
            Err("scalar im2col measured faster on 8x8 f32 tiles")
        } else {
            let geometry = cooperative_geometry(problem, config)?;
            let workgroups: u64 = geometry.grid.into_iter().map(u64::from).product();
            let minimum = match problem.kind {
                Kind::GradInput => MIN_GRAD_INPUT_WORKGROUPS,
                Kind::Forward | Kind::GradWeight if config.use_f16_input => MIN_F16_WORKGROUPS,
                Kind::Forward | Kind::GradWeight => MIN_F32_WORKGROUPS,
            };
            if workgroups < minimum {
                // Policy, not legality.
                Err("too few tiles to amortize staging")
            } else {
                Ok(())
            }
        }
    }
}

/// The preferred path for `problem` on `target`.
pub(crate) fn select(problem: &Problem, target: &Target) -> Path {
    super::select::<Path>(problem, target).chosen
}

/// Where a cooperative convolution writes and what it needs allocated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CooperativeGeometry {
    /// Rows of output tiles on X, columns on Y, images on Z.
    pub grid: [u32; 3],
    /// Output bytes covering whole tiles.
    pub output_bytes: usize,
}

/// Whether a cooperative kernel in `config` computes `problem`, and its
/// geometry if so.
pub(crate) fn cooperative_geometry(
    problem: &Problem,
    config: &CoopConfig,
) -> Result<CooperativeGeometry, Rejection> {
    let tile = config.output_tile();
    let grid = [
        problem.m.div_ceil(tile),
        problem.n.div_ceil(tile),
        problem.batch,
    ];
    // Direct stores write whole sub-tiles: N must stay aligned so stores
    // never straddle rows, and later images must not start inside padding
    // their consumers do not account for. The generated f32 input-gradient
    // kernel stages its right edge through checked stores instead.
    let stores_aligned =
        problem.n.is_multiple_of(16) && (problem.batch == 1 || problem.m.is_multiple_of(tile));
    let checked_stores = problem.kind == Kind::GradInput && !config.use_f16_input;
    if !stores_aligned && !checked_stores {
        Err("output tiles would straddle rows or images")
    } else if problem.k < 4 {
        Err("K is below one vec4 load")
    } else if !crate::compile::workgroups_within_portable_limits(grid) {
        Err("grid exceeds the portable dispatch limit")
    } else {
        let output_bytes =
            crate::compile::cooperative_output_bytes(problem.m, problem.n, problem.batch, tile)
                .ok_or("padded output exceeds the address space")?;
        Ok(CooperativeGeometry { grid, output_bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_a_wide_grid_on_the_64_tile() {
        // 256 x 3136 is 4 * 49 workgroups at tile 64.
        assert_eq!(register_tile(256, 3136, 1, false), 64);
    }

    #[test]
    fn narrows_a_7x7_weight_gradient_to_16() {
        // Co=64, Ci*k=147: tile 64 is 3 workgroups, tile 32 is 10, tile 16 is 40.
        assert_eq!(register_tile(64, 147, 1, false), 16);
    }

    #[test]
    fn keeps_64_when_native_f32_coop_can_use_the_grid() {
        assert_eq!(register_tile(256, 196, 1, true), 64);
        assert_eq!(register_tile(64, 147, 1, true), 16);
    }

    fn problem(kind: Kind, m: u32, n: u32, k: u32) -> Problem {
        Problem {
            kind,
            m,
            n,
            k,
            batch: 1,
            window: [3, 3, 1],
            pinned: false,
            reduced_storage: false,
            requires_full_precision: false,
            epilogue_inputs: false,
        }
    }

    const F16: CoopConfig = CoopConfig {
        tile_size: 16,
        use_f16_input: true,
        compensated: false,
    };
    const F32_8: CoopConfig = CoopConfig {
        tile_size: 8,
        use_f16_input: false,
        compensated: false,
    };

    fn target(config: CoopConfig) -> Target {
        Target {
            cooperative: Some(config),
            allow_raw_f16: false,
        }
    }

    #[test]
    fn compiled_admits_everything() {
        let bare = Target {
            cooperative: None,
            allow_raw_f16: false,
        };
        let p = problem(Kind::Forward, 256, 3136, 576);
        assert_eq!(select(&p, &bare), Path::Compiled);
        assert_eq!(Path::Compiled.admits(&p, &target(F16)), Ok(()));
    }

    #[test]
    fn cooperative_states_why_it_declines() {
        let wide = problem(Kind::Forward, 256, 3136, 576);
        assert_eq!(select(&wide, &target(F16)), Path::Cooperative);
        assert_eq!(
            Path::Cooperative.admits(&wide, &target(F32_8)),
            Err("scalar im2col measured faster on 8x8 f32 tiles")
        );
        let grad = problem(Kind::GradInput, 64, 3136, 576);
        assert_eq!(select(&grad, &target(F32_8)), Path::Cooperative);
        for (p, config, reason) in [
            (
                Problem {
                    pinned: true,
                    ..wide
                },
                F16,
                "tuning pinned a scalar K stage",
            ),
            (
                Problem {
                    requires_full_precision: true,
                    ..wide
                },
                F16,
                "f16 operands would lose required precision",
            ),
            (
                problem(Kind::Forward, 256, 3130, 576),
                F16,
                "output tiles would straddle rows or images",
            ),
            (
                problem(Kind::Forward, 256, 3136, 3),
                F16,
                "K is below one vec4 load",
            ),
            (
                problem(Kind::Forward, 64, 256, 576),
                F16,
                "too few tiles to amortize staging",
            ),
        ] {
            assert_eq!(Path::Cooperative.admits(&p, &target(config)), Err(reason));
        }
        // The f32 input gradient stages its right edge through checked
        // stores, so unaligned columns stay eligible.
        let ragged = problem(Kind::GradInput, 64, 3130, 576);
        assert_eq!(select(&ragged, &target(F32_8)), Path::Cooperative);
    }

    #[test]
    fn batched_outputs_need_whole_row_tiles() {
        let batched = Problem {
            batch: 4,
            ..problem(Kind::Forward, 100, 3136, 576)
        };
        assert_eq!(
            cooperative_geometry(&batched, &F16),
            Err("output tiles would straddle rows or images")
        );
        let aligned = Problem { m: 128, ..batched };
        let geometry = cooperative_geometry(&aligned, &F16).unwrap();
        assert_eq!(geometry.grid, [4, 98, 4]);
        assert_eq!(geometry.output_bytes, 128 * 3136 * 4 * 4);
    }
}
