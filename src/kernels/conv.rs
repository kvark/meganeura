//! Convolutions lowered to implicit GEMM: the forward pass, and the input
//! and weight gradients.
//!
//! Two decisions are made here. The compiler picks a scalar register tile
//! ([`register_tile`]), which names the shader entry and which tuning may
//! later measure against the others. Session construction then decides
//! whether a cooperative kernel replaces the 64-wide one, through the
//! [`Path`] family. Depthwise and Winograd convolutions are separate
//! computations with their own kernels.

use super::{Family, Rejection, occupancy};
use crate::codegen::{CoopConfig, ShaderGroup};
use crate::compile::{Dispatch, Kernel, ShaderEntry};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub(crate) struct Params {
    pub batch: u32,
    pub in_channels: u32,
    pub in_h: u32,
    pub in_w: u32,
    pub out_channels: u32,
    pub kernel_h: u32,
    pub kernel_w: u32,
    pub stride: u32,
    pub padding_h: u32,
    pub out_h: u32,
    pub out_w: u32,
    pub padding_w: u32,
    pub kernel_w_multiplier: u32,
    pub kernel_hw_multiplier: u32,
    pub column_width_multiplier: u32,
    pub output_spatial_multiplier: u32,
}

impl From<&Dispatch> for Params {
    fn from(dispatch: &Dispatch) -> Self {
        let p = &dispatch.params;
        let column_width = match dispatch.shader {
            // Scalar input gradients enumerate one stride phase's columns.
            ShaderEntry::Conv2dGradInputGemm
            | ShaderEntry::Conv2dGradInputGemmSmall
            | ShaderEntry::Conv2dGradInputGemm16 => p[3].div_ceil(p[7]),
            ShaderEntry::Conv2dGradInputGemmCoopGen(..) => p[3],
            _ => p[10],
        };
        Self {
            batch: p[0],
            in_channels: p[1],
            in_h: p[2],
            in_w: p[3],
            out_channels: p[4],
            kernel_h: p[5],
            kernel_w: p[6],
            stride: p[7],
            padding_h: p[8],
            out_h: p[9],
            out_w: p[10],
            padding_w: p[11],
            kernel_w_multiplier: crate::divisor::multiplier(p[6]),
            kernel_hw_multiplier: crate::divisor::multiplier(p[5] * p[6]),
            column_width_multiplier: crate::divisor::multiplier(column_width),
            output_spatial_multiplier: crate::divisor::multiplier(p[9] * p[10]),
        }
    }
}

impl Params {
    /// A 1×1 gather is contiguous when it preserves both spatial axes.
    /// Complete four-element rows keep vector loads inside each binding.
    pub(crate) fn vectorized_forward(&self) -> bool {
        self.kernel_h == 1
            && self.kernel_w == 1
            && self.stride == 1
            && self.padding_h == 0
            && self.padding_w == 0
            && self.in_h == self.out_h
            && self.in_w == self.out_w
            && self.in_channels != 0
            && self.in_channels.is_multiple_of(4)
            && self
                .out_h
                .checked_mul(self.out_w)
                .is_some_and(|n| n != 0 && n.is_multiple_of(4))
    }
}

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

/// The scalar register tile for a `rows × cols` product over `batch`
/// images: the widest that still occupies the device, or 16 when none
/// does. A device with f32 cooperative matrices keeps the 64-wide tile
/// from [`occupancy::MIN_F32_COOP_WORKGROUPS`] workgroups on, since only that
/// entry is promoted.
pub(crate) fn register_tile(rows: u32, cols: u32, batch: u32, f32_cooperative: bool) -> u32 {
    let batch = batch.max(1);
    if f32_cooperative
        && u64::from(rows.div_ceil(64)) * u64::from(cols.div_ceil(64)) * u64::from(batch)
            >= occupancy::MIN_F32_COOP_WORKGROUPS
    {
        return 64;
    }
    occupancy::widest_occupied_tile(
        &TILES,
        |tile| (tile, tile),
        rows,
        cols,
        batch,
        u64::from(occupancy::SCALAR_OCCUPIED),
        false,
    )
}

/// Reduction partitions for a weight gradient's `rows × cols` output summed
/// over `reduction` positions, when its 64-wide grid cannot occupy the
/// device. [`register_tile`] narrows such tiles instead, down to one output
/// per thread at width 16; partitions of a long reduction supply the
/// workgroups and let the tile stay wide. Outputs narrower than 64 keep
/// their tile, which would otherwise be mostly padding.
pub(crate) fn weight_partitions(rows: u32, cols: u32, reduction: u32) -> Option<u32> {
    if rows < 64 || cols < 64 {
        return None;
    }
    let groups = rows.div_ceil(64).saturating_mul(cols.div_ceil(64));
    occupancy::CONV_WEIGHT_SPLITS.multi_splits(groups, reduction)
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

/// The grid of a scalar input gradient at `tile`: one Z layer per image and
/// stride phase, each covering that phase's `ceil(H/s) × ceil(W/s)` pixels,
/// so a strided gradient enumerates only the taps that reach an output.
pub(crate) fn grad_input_workgroups(
    batch: u32,
    in_channels: u32,
    in_h: u32,
    in_w: u32,
    stride: u32,
    tile: u32,
) -> [u32; 3] {
    let columns = in_h.div_ceil(stride) * in_w.div_ceil(stride);
    [
        columns.div_ceil(tile),
        in_channels.div_ceil(tile),
        batch * stride * stride,
    ]
}

/// Scalar convolution with its geometry baked into the pipeline, using
/// exact reciprocal multipliers as constants and the K stage the uniform
/// shader uses. The uniform software divisor stays available to tuning.
pub(crate) fn exact_kernel() -> Kernel {
    Kernel::SpecializedConv {
        interleave_columns: false,
        k_tile: EXACT_K_TILE,
    }
}

/// The K stage of [`exact_kernel`]. Any other stage was measured and pinned.
const EXACT_K_TILE: u32 = 16;

/// Register blocking of the direct forward convolution
/// ([`crate::compile::Kernel::RegisterConv`]): each thread accumulates
/// `channels` output channels for `pixels` consecutive output pixels, and a
/// workgroup is `threads` threads along the pixels. It uses no workgroup
/// memory. Every thread of a workgroup reads the same weights, which the
/// cache serves, and adjacent threads read adjacent pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RegisterBlock {
    pub channels: u32,
    pub pixels: u32,
    pub threads: u32,
}

/// Output pixels (over the batch) below which a convolution takes the
/// smaller 8 × 2 block, so the grid still has threads to spread over, and
/// a 1×1 convolution takes the register product kernel instead.
pub(crate) const FEW_PIXELS: u32 = 1024;

/// The register block for a forward convolution with `out_channels`
/// channels over `pixels` output pixels (all images), at `stride`.
///
/// Measured on Adreno 740 (Quest 3S) over D-FINE-N's convolutions, one
/// dispatch at a time:
/// - 16 channels by 4 pixels where there are at least 16 channels; 2
///   pixels at strides above 1, whose gathers are not contiguous;
/// - 4 channels where there are fewer than 16;
/// - 8 channels by 2 pixels below [`FEW_PIXELS`] pixels: the deep 1×1
///   convolutions at 15×20 then have twice the threads, and ran up to
///   1.2× faster than with 16 by 4;
/// - always 64 threads.
///
/// Neighbouring blocks were up to 2.5× slower on some shapes, and not
/// predictably, so tuning should eventually choose per shape.
pub(crate) fn register_block(out_channels: u32, pixels: u32, stride: u32) -> RegisterBlock {
    let (channels, pixels) = if out_channels < 16 {
        (4, if stride == 1 { 4 } else { 2 })
    } else if pixels < FEW_PIXELS {
        (8, 2)
    } else {
        (16, if stride == 1 { 4 } else { 2 })
    };
    RegisterBlock {
        channels,
        pixels,
        threads: 64,
    }
}

/// The grid of a [`RegisterBlock`] over `out_channels` × `spatial` outputs
/// for `batch` images.
pub(crate) fn register_workgroups(
    block: RegisterBlock,
    out_channels: u32,
    spatial: u32,
    batch: u32,
) -> [u32; 3] {
    [
        spatial.div_ceil(block.pixels * block.threads),
        out_channels.div_ceil(block.channels),
        batch,
    ]
}

/// One convolution dispatch whose kernel session construction chooses.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Problem {
    pub kind: Kind,
    /// The 64-wide entry, the one with a cooperative form.
    pub wide: bool,
    /// Still on the kernel the compiler emitted (or a register one), so no
    /// measurement has fixed it.
    pub compiled: bool,
    /// A 1×1 convolution that keeps its extent, with vec4-aligned input
    /// channels and pixels: a product per image ([`Params::vectorized_forward`]).
    pub vectorized: bool,
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
    /// Any fused epilogue, which the register kernel does not apply.
    pub epilogue: bool,
}

impl Problem {
    /// The convolution `dispatch` computes: every forward entry, and the
    /// 64-wide input gradient. The weight gradient has no other path.
    pub(crate) fn of(dispatch: &Dispatch) -> Option<Self> {
        let p = &dispatch.params;
        // params: [batch, in_channels, in_h, in_w, out_channels, kernel_h,
        // kernel_w, stride, padding_h, out_h, out_w, padding_w]
        let group = dispatch.shader.shader_group();
        let (kind, m, n, k) = match group {
            ShaderGroup::Conv2dGemm | ShaderGroup::Conv2dGemmSmall | ShaderGroup::Conv2dGemm16 => {
                (Kind::Forward, p[4], p[9] * p[10], p[1] * p[5] * p[6])
            }
            ShaderGroup::Conv2dGradInputGemm => {
                (Kind::GradInput, p[1], p[2] * p[3], p[4] * p[5] * p[6])
            }
            _ => return None,
        };
        Some(Self {
            kind,
            wide: matches!(
                group,
                ShaderGroup::Conv2dGemm | ShaderGroup::Conv2dGradInputGemm
            ),
            compiled: dispatch.kernel == exact_kernel()
                || matches!(
                    dispatch.kernel,
                    Kernel::RegisterConv(_) | Kernel::RegisterMatmul(_)
                ),
            vectorized: kind == Kind::Forward && Params::from(dispatch).vectorized_forward(),
            m,
            n,
            k,
            batch: p[0],
            window: [p[5], p[6], p[7]],
            pinned: dispatch.conv_k_tile().is_some_and(|k| k != EXACT_K_TILE),
            reduced_storage: dispatch.weight_format.uses_reduced_storage(),
            requires_full_precision: dispatch.full_precision(),
            epilogue_inputs: dispatch
                .matmul_epilogue
                .as_ref()
                .is_some_and(|epilogue| !epilogue.inputs.is_empty()),
            epilogue: dispatch.matmul_epilogue.is_some(),
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

/// How a convolution runs, most preferred first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Path {
    /// A generated cooperative-matrix kernel for the window.
    Cooperative,
    /// A small 1×1 convolution as the register product kernel, weight as A.
    Product,
    /// The direct register-blocked forward kernel, [`register_block`].
    Register,
    /// The scalar kernel the compiler emitted. The fallback.
    Compiled,
}

impl Family for Path {
    type Problem = Problem;
    type Target = Target;
    const PREFERENCE: &'static [Self] = &[
        Path::Cooperative,
        Path::Product,
        Path::Register,
        Path::Compiled,
    ];

    fn tile(self, problem: &Problem, target: &Target) -> Option<[u32; 2]> {
        Some(match self {
            Path::Cooperative => [target.cooperative?.output_tile(); 2],
            Path::Product => [4, 4],
            Path::Register => {
                let block = problem.register_block();
                [block.channels, block.pixels * block.threads]
            }
            Path::Compiled => [64; 2],
        })
    }

    fn admits(self, problem: &Problem, target: &Target) -> Result<(), Rejection> {
        match self {
            Path::Compiled => return Ok(()),
            Path::Register => return register_admits(problem, target),
            Path::Product => return product_admits(problem, target).map(|_| ()),
            Path::Cooperative => {}
        }
        if !problem.wide {
            return Err("only the 64-wide entry has a cooperative form");
        }
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
                Kind::GradInput => occupancy::MIN_GRAD_INPUT_WORKGROUPS,
                Kind::Forward | Kind::GradWeight if config.use_f16_input => {
                    occupancy::MIN_F16_COOP_WORKGROUPS
                }
                Kind::Forward | Kind::GradWeight => occupancy::MIN_F32_COOP_WORKGROUPS,
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

/// Policy: the register kernel replaces the shared-memory tiles where no
/// cooperative matrices are available. It was measured there (Adreno 740,
/// and an RTX 3050 with cooperative matrices disabled); devices with them
/// keep their kernels unless tuning measures otherwise.
fn register_admits(problem: &Problem, target: &Target) -> Result<(), Rejection> {
    if problem.kind != Kind::Forward {
        Err("the register kernel computes the forward convolution only")
    } else if target.cooperative.is_some() {
        Err("measured only where cooperative matrices are absent")
    } else if !problem.compiled {
        Err("a measurement fixed the kernel")
    } else if problem.reduced_storage {
        Err("packed weights keep their compiled kernel")
    } else if problem.epilogue {
        Err("the register kernel has no epilogue")
    } else if !crate::compile::workgroups_within_portable_limits(problem.register_workgroups()) {
        Err("grid exceeds the portable dispatch limit")
    } else {
        Ok(())
    }
}

/// A 1×1 convolution that keeps its extent is a product per image. On
/// small images the register product kernel, which reads both operands as
/// vec4 and spreads threads over rows too, beats the direct kernel:
/// D-FINE-N's 1×1 convolutions at 15×20 ran 1.5–1.7× faster on Adreno 740.
/// From 30×40 up the direct kernel won, by up to 2.8× at 60×80.
pub(crate) fn product_admits(
    problem: &Problem,
    target: &Target,
) -> Result<crate::kernels::matmul::RegisterTile, Rejection> {
    register_admits(problem, target)?;
    if !problem.vectorized {
        return Err("not a vec4-aligned 1x1 convolution");
    }
    if problem.n.saturating_mul(problem.batch) >= FEW_PIXELS {
        return Err("the direct kernel is faster on this many pixels");
    }
    crate::kernels::matmul::register_product_tile([problem.m, problem.n, problem.k], problem.batch)
}

impl Problem {
    /// The register block for this forward convolution.
    pub(crate) fn register_block(&self) -> RegisterBlock {
        register_block(self.m, self.n.saturating_mul(self.batch), self.window[2])
    }

    /// The grid of [`Problem::register_block`].
    pub(crate) fn register_workgroups(&self) -> [u32; 3] {
        register_workgroups(self.register_block(), self.m, self.n, self.batch)
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
    // Direct stores write hardware sub-tiles, not the whole workgroup tile.
    // The second row/column sub-tile is guarded independently. Each image
    // must end on a hardware row boundary so stores cannot touch its neighbor.
    // The generated f32 input-gradient kernel bounds-checks edge stores.
    let stores_aligned = problem.n.is_multiple_of(16)
        && (problem.batch == 1 || problem.m.is_multiple_of(config.tile_size));
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
    fn register_blocks_cover_every_output() {
        // D-FINE-N's 16->8 2x2 convolution: 4 channels, 4 pixels.
        let narrow = register_block(8, 240 * 320, 1);
        assert_eq!((narrow.channels, narrow.pixels), (4, 4));
        assert_eq!(register_workgroups(narrow, 8, 240 * 320, 1), [300, 2, 1]);
        // Strided gathers take two pixels; partial blocks round up.
        let strided = register_block(17, 3 * 1200, 2);
        assert_eq!((strided.channels, strided.pixels), (16, 2));
        assert_eq!(register_workgroups(strided, 17, 1200, 3), [10, 2, 3]);
        // A deep 1x1 at 15x20 takes the smaller block for more threads.
        let deep = register_block(512, 300, 1);
        assert_eq!((deep.channels, deep.pixels), (8, 2));
        assert_eq!(register_workgroups(deep, 512, 300, 1), [3, 64, 1]);
    }

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

    #[test]
    fn long_weight_reductions_split_instead_of_narrowing() {
        // ResNet-50's stem at batch 4: 3 workgroups at 64, 50176 positions.
        assert_eq!(weight_partitions(64, 147, 50176), Some(86));
        // Partitions keep at least 256 positions.
        assert_eq!(weight_partitions(256, 64, 12544), Some(49));
        // Occupied grids, short reductions and narrow outputs stay unsplit.
        assert_eq!(weight_partitions(1024, 256, 784), None);
        assert_eq!(weight_partitions(64, 147, 511), None);
        assert_eq!(weight_partitions(8, 75, 524288), None);
    }

    fn problem(kind: Kind, m: u32, n: u32, k: u32) -> Problem {
        Problem {
            kind,
            wide: true,
            compiled: true,
            vectorized: false,
            m,
            n,
            k,
            batch: 1,
            window: [3, 3, 1],
            pinned: false,
            reduced_storage: false,
            requires_full_precision: false,
            epilogue_inputs: false,
            epilogue: false,
        }
    }

    const F16: CoopConfig = CoopConfig {
        tile_size: 16,
        use_f16_input: true,
    };
    const F32_8: CoopConfig = CoopConfig {
        tile_size: 8,
        use_f16_input: false,
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
        assert_eq!(Path::Compiled.admits(&p, &bare), Ok(()));
        assert_eq!(Path::Compiled.admits(&p, &target(F16)), Ok(()));
        let grad = problem(Kind::GradInput, 64, 3136, 576);
        assert_eq!(select(&grad, &bare), Path::Compiled);
    }

    #[test]
    fn small_aligned_1x1_convolutions_take_the_product() {
        let bare = Target {
            cooperative: None,
            allow_raw_f16: false,
        };
        // D-FINE-N's 896 -> 512 at 15×20.
        let small = Problem {
            vectorized: true,
            window: [1, 1, 1],
            ..problem(Kind::Forward, 512, 300, 896)
        };
        assert_eq!(select(&small, &bare), Path::Product);
        for (p, reason) in [
            (
                Problem {
                    vectorized: false,
                    ..small
                },
                "not a vec4-aligned 1x1 convolution",
            ),
            (
                Problem { n: 1200, ..small },
                "the direct kernel is faster on this many pixels",
            ),
            (
                Problem {
                    m: 4,
                    n: 16,
                    ..small
                },
                "too few workgroups; split K instead",
            ),
        ] {
            assert_eq!(Path::Product.admits(&p, &bare), Err(reason));
        }
        // Declined products fall to the direct kernel.
        assert_eq!(select(&Problem { n: 1200, ..small }, &bare), Path::Register);
    }

    #[test]
    fn register_takes_forward_convolutions_without_cooperative_matrices() {
        let bare = Target {
            cooperative: None,
            allow_raw_f16: false,
        };
        let p = problem(Kind::Forward, 256, 3136, 576);
        assert_eq!(select(&p, &bare), Path::Register);
        let narrow = Problem {
            wide: false,
            ..problem(Kind::Forward, 8, 76_800, 64)
        };
        assert_eq!(select(&narrow, &bare), Path::Register);
        for (p, target, reason) in [
            (
                problem(Kind::GradInput, 64, 3136, 576),
                bare,
                "the register kernel computes the forward convolution only",
            ),
            (
                p,
                target(F16),
                "measured only where cooperative matrices are absent",
            ),
            (
                Problem {
                    compiled: false,
                    ..p
                },
                bare,
                "a measurement fixed the kernel",
            ),
            (
                Problem {
                    reduced_storage: true,
                    ..p
                },
                bare,
                "packed weights keep their compiled kernel",
            ),
            (
                Problem {
                    epilogue: true,
                    ..p
                },
                bare,
                "the register kernel has no epilogue",
            ),
            (
                // 2^24 pixels need 65,536 workgroups of 4 × 64.
                problem(Kind::Forward, 16, 1 << 24, 27),
                bare,
                "grid exceeds the portable dispatch limit",
            ),
            (
                Problem { batch: 70_000, ..p },
                bare,
                "grid exceeds the portable dispatch limit",
            ),
        ] {
            assert_eq!(Path::Register.admits(&p, &target), Err(reason));
        }
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
        let narrow = Problem {
            wide: false,
            ..wide
        };
        assert_eq!(
            Path::Cooperative.admits(&narrow, &target(F16)),
            Err("only the 64-wide entry has a cooperative form")
        );
        // The f32 input gradient stages its right edge through checked
        // stores, so unaligned columns stay eligible.
        let ragged = problem(Kind::GradInput, 64, 3130, 576);
        assert_eq!(select(&ragged, &target(F32_8)), Path::Cooperative);
    }

    #[test]
    fn batched_outputs_need_whole_hardware_row_tiles() {
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
        for kind in [Kind::Forward, Kind::GradInput] {
            let half_workgroup = Problem {
                kind,
                m: 48,
                ..batched
            };
            let geometry = cooperative_geometry(&half_workgroup, &F16).unwrap();
            assert_eq!(geometry.grid, [2, 98, 4]);
            assert_eq!(geometry.output_bytes, 64 * 3136 * 4 * 4);
        }
    }
}
