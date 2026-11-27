//! Dense matrix products: `C = A·B`, `Aᵀ·B` and `A·Bᵀ`, each optionally
//! `+ D`.
//!
//! F32 products start with the 64-wide scalar tile kernel. Session
//! construction then picks among the paths here, and tuning challenges the
//! choice by measurement; both ask `cooperative_geometry` whether a
//! cooperative kernel is legal, so the two cannot drift apart.
//! Explicit reduced storage fixes a typed cooperative implementation of the
//! same product; other candidates cannot replace its storage contract.
//!
//! Single-row products (GEMV), block-diagonal and batch-major layouts and
//! convolutions are separate computations with their own kernels.

use super::{Family, Rejection, occupancy};
use crate::codegen::{
    CoopConfig, CooperativeMatmulShape, MatMulOptions, MatMulTile, MatmulKnobs, MatrixTypes,
    ScalarMatmulShape, ShaderGroup, ShaderModule, TypedMatmulLayout, WorkgroupMatmulShape,
};
use crate::compile::{
    BufferRef, Dispatch, EpilogueLoadKind, Kernel, MatMulEpilogue, MatMulPrologue,
    PrologueLoadKind, ShaderEntry, WeightFormat,
};
use crate::schedule::PointwiseDAG;
use serde::{Deserialize, Serialize};

/// K partitions for a checked typed product: fill a small 32×32 output grid
/// without making each partition mostly staging work. Storage and
/// multiplication types do not change the partition contract.
pub(crate) fn matrix_splits(m: u32, n: u32, k: u32) -> u32 {
    let groups = m.div_ceil(32).saturating_mul(n.div_ceil(32)).max(1);
    occupancy::TYPED_MATMUL_SPLITS.splits(groups, k)
}

/// K partitions for a scalar product whose grid of 32×32 tiles, the
/// narrowest tuning offers, has fewer workgroups than
/// [`occupancy::SCALAR_OCCUPIED`]: aim for 128 workgroups with at least eight
/// 16-deep K stages each. A 64×1024 by 1024×256 product has 16 workgroups and
/// gets 8 partitions; on RTX 5070 and Arc B570 that takes 0.4 of the unsplit
/// time, SumRows included. A 64×256 by 256×1024 product already has 64 and
/// stays whole.
pub(crate) fn scalar_partitions(m: u32, n: u32, k: u32) -> Option<u32> {
    let groups = m.div_ceil(32).saturating_mul(n.div_ceil(32)).max(1);
    occupancy::SCALAR_MATMUL_SPLITS.multi_splits(groups, k)
}

/// Which product a dispatch computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    pub(crate) fn from_entry(entry: &ShaderEntry) -> Option<Self> {
        Some(match *entry {
            ShaderEntry::MatMul => Self::Plain,
            ShaderEntry::FusedMatMulAdd => Self::PlainAdd,
            ShaderEntry::MatMulAT => Self::TransposedA,
            ShaderEntry::FusedMatMulATAdd => Self::TransposedAAdd,
            ShaderEntry::MatMulBT => Self::TransposedB,
            ShaderEntry::FusedMatMulBTAdd => Self::TransposedBAdd,
            _ => return None,
        })
    }

    pub(crate) fn entry_ref(self) -> &'static ShaderEntry {
        match self {
            Self::Plain => &ShaderEntry::MatMul,
            Self::PlainAdd => &ShaderEntry::FusedMatMulAdd,
            Self::TransposedA => &ShaderEntry::MatMulAT,
            Self::TransposedAAdd => &ShaderEntry::FusedMatMulATAdd,
            Self::TransposedB => &ShaderEntry::MatMulBT,
            Self::TransposedBAdd => &ShaderEntry::FusedMatMulBTAdd,
        }
    }

    pub(crate) fn group(self) -> ShaderGroup {
        match self {
            Self::Plain => ShaderGroup::MatMul,
            Self::PlainAdd => ShaderGroup::MatMulAdd,
            Self::TransposedA => ShaderGroup::MatMulAT,
            Self::TransposedAAdd => ShaderGroup::MatMulATAdd,
            Self::TransposedB => ShaderGroup::MatMulBT,
            Self::TransposedBAdd => ShaderGroup::MatMulBTAdd,
        }
    }

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

    pub(crate) fn has_addend(self) -> bool {
        matches!(
            self,
            Self::PlainAdd | Self::TransposedAAdd | Self::TransposedBAdd
        )
    }

    /// Transposed products with an addend have no 32-wide scalar kernel.
    fn transposed_with_addend(self) -> bool {
        matches!(self, Self::TransposedAAdd | Self::TransposedBAdd)
    }

    /// Encode `(m, n, k)` into the `Dispatch::params` order expected by this
    /// layout's shader entry (`[m, k, n, 0]` for plain, `[m, n, k, 0]` for
    /// transposed A/B).
    pub(crate) fn params_words(self, m: u32, n: u32, k: u32) -> Vec<u32> {
        match self {
            Self::Plain | Self::PlainAdd => vec![m, k, n, 0],
            Self::TransposedA | Self::TransposedAAdd | Self::TransposedB | Self::TransposedBAdd => {
                vec![m, n, k, 0]
            }
        }
    }
}

/// What a dense matrix product computes.
///
/// Single decoder for the six dense matmul entries (`MatMul`, `MatMulAT`,
/// `MatMulBT`, and their `Fused*Add` forms).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Product {
    pub layout: Layout,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    /// Storage format of B (`WeightFormat::F32` when explicit `types` govern storage).
    pub weights: WeightFormat,
    /// Explicit operand, multiplication and output types when fixed by a typed
    /// or workgroup cooperative kernel.
    pub types: Option<MatrixTypes>,
}

/// How a dense matrix product runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Variant {
    /// Compiled 64×64 scalar tiled kernel (uses session `MatmulKnobs`).
    DefaultScalar,
    /// Demoted 32×32 scalar tiled kernel (uses session `MatmulKnobs`).
    SmallScalar,
    /// Measured/specialized scalar tile shape (overrides `MatmulKnobs`).
    Scalar(ScalarMatmulShape),
    /// K-partitioned scalar tile shape (`k_partitions >= 2`).
    SplitScalar(ScalarMatmulShape),
    /// Device-configured cooperative matmul (`Kernel::Cooperative`).
    Cooperative(CoopConfig),
    /// Shared-tile native-f32 16×16 cooperative matmul.
    CooperativeTiled(CooperativeMatmulShape),
    /// Checked typed cooperative matmul.
    CooperativeTyped { specialize: bool },
    /// Workgroup/subgroup-tiled cooperative matmul.
    CooperativeWorkgroup(WorkgroupMatmulShape),
}

/// Launch geometry and uniform words for one dense product dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Launch {
    pub workgroups: [u32; 3],
    /// `[m, n, k, _pad]` bound as `MatMulParams`.
    pub params: [u32; 4],
}

impl Variant {
    /// Decode `(Variant, k_partitions)` from a dense product's [`Kernel`].
    pub(crate) fn from_kernel(
        kernel: &Kernel,
        coop_config: Option<&CoopConfig>,
    ) -> Option<(Self, u32)> {
        Some(match *kernel {
            Kernel::Default => (Self::DefaultScalar, 1),
            Kernel::SmallTile => (Self::SmallScalar, 1),
            Kernel::ScalarMatmul(shape) => (Self::Scalar(shape), 1),
            Kernel::SplitMatmul { shape, splits } => (Self::SplitScalar(shape), splits),
            Kernel::Cooperative => (Self::Cooperative(*coop_config?), 1),
            Kernel::CooperativeTiled { shape, splits } => (Self::CooperativeTiled(shape), splits),
            Kernel::CooperativeTyped {
                splits, specialize, ..
            } => (Self::CooperativeTyped { specialize }, splits),
            Kernel::CooperativeWorkgroup { shape, splits, .. } => {
                (Self::CooperativeWorkgroup(shape), splits)
            }
            _ => return None,
        })
    }
}

impl Product {
    /// Decode the dense product carried by `dispatch`, if its shader entry is
    /// one of the six dense matmul forms and its dimensions are present.
    pub(crate) fn of(dispatch: &Dispatch) -> Option<Self> {
        let layout = Layout::from_entry(&dispatch.shader)?;
        let p = &dispatch.params;
        let (a, b, c) = (p.first().copied()?, p.get(1).copied()?, p.get(2).copied()?);
        let (m, n, k) = match layout {
            Layout::Plain | Layout::PlainAdd => (a, c, b),
            Layout::TransposedA
            | Layout::TransposedAAdd
            | Layout::TransposedB
            | Layout::TransposedBAdd => (a, b, c),
        };
        let types = match dispatch.kernel {
            Kernel::CooperativeTyped { types, .. } => Some(types),
            Kernel::CooperativeWorkgroup { types, .. } => types,
            _ => None,
        };
        Some(Self {
            layout,
            m,
            n,
            k,
            weights: dispatch.weight_format,
            types,
        })
    }

    /// Compute the launch workgroup grid and uniform words for `variant` with
    /// `k_partitions` K slices (1 for unsplit products).
    pub(crate) fn launch(self, variant: Variant, k_partitions: u32) -> Launch {
        let workgroups = match variant {
            Variant::DefaultScalar => crate::compile::matmul_workgroups(self.m, self.n, 64),
            Variant::SmallScalar => crate::compile::matmul_workgroups(self.m, self.n, 32),
            Variant::Scalar(shape) => {
                crate::compile::matmul_workgroups_rect(self.m, self.n, shape.rows(), shape.cols())
            }
            Variant::SplitScalar(shape) => [
                self.n.div_ceil(shape.cols()),
                self.m.div_ceil(shape.rows()),
                k_partitions,
            ],
            Variant::Cooperative(config) => {
                let tile = config.matmul_output_tile();
                [self.m.div_ceil(tile), self.n.div_ceil(tile), 1]
            }
            Variant::CooperativeTiled(shape) => [
                self.m / CooperativeMatmulShape::ROWS,
                self.n / shape.columns,
                k_partitions,
            ],
            Variant::CooperativeTyped { .. } => {
                [self.m.div_ceil(32), self.n.div_ceil(32), k_partitions]
            }
            Variant::CooperativeWorkgroup(shape) => [
                self.m.div_ceil(shape.rows),
                self.n / shape.columns,
                k_partitions,
            ],
        };
        let pad = match variant {
            Variant::CooperativeTyped { .. } | Variant::CooperativeWorkgroup(_) => k_partitions,
            _ => workgroups[1],
        };
        Launch {
            workgroups,
            params: [self.m, self.n, self.k, pad],
        }
    }

    /// Uniform `[m, n, k, _pad]` words for a dispatch whose workgroups are
    /// already installed.
    pub(crate) fn uniform_params(self, kernel: &Kernel, workgroups: [u32; 3]) -> [u32; 4] {
        let pad = match *kernel {
            Kernel::CooperativeTyped { splits, .. }
            | Kernel::CooperativeWorkgroup { splits, .. } => splits,
            _ => workgroups[1],
        };
        [self.m, self.n, self.k, pad]
    }
}

/// Epilogue description without buffer handles: everything that affects WGSL.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct EpilogueCode {
    pub dag: PointwiseDAG,
    pub input_kinds: Vec<EpilogueLoadKind>,
}

impl EpilogueCode {
    fn from_compile(epi: &MatMulEpilogue) -> Self {
        Self {
            dag: epi.dag.clone(),
            input_kinds: epi.inputs.iter().map(|entry| entry.1.clone()).collect(),
        }
    }

    fn to_compile(&self) -> MatMulEpilogue {
        MatMulEpilogue {
            dag: self.dag.clone(),
            inputs: self
                .input_kinds
                .iter()
                .map(|kind| (BufferRef(0), kind.clone()))
                .collect(),
        }
    }
}

/// Implementation-specific fields that determine a dense product's WGSL.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CodeImpl {
    DefaultScalar,
    SmallScalar,
    Scalar(ScalarMatmulShape),
    SplitScalar {
        shape: ScalarMatmulShape,
        splits: u32,
    },
    Cooperative,
    CooperativeTiled {
        shape: CooperativeMatmulShape,
        splits: u32,
    },
    CooperativeTyped {
        types: MatrixTypes,
        staging: TypedMatmulLayout,
    },
    CooperativeWorkgroup {
        shape: WorkgroupMatmulShape,
        types: MatrixTypes,
        row_bounds: bool,
    },
}

/// Everything that determines a dense matrix product's compiled pipeline.
///
/// `Code` is both the pipeline-map key and the sole per-dispatch input to
/// [`module`]. Code generation receives no `&Dispatch`, so a field cannot
/// influence the generated shader without belonging to the key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Code {
    pub layout: Layout,
    pub weights: WeightFormat,
    pub implementation: CodeImpl,
    pub epilogue: Option<EpilogueCode>,
    pub prologue: Vec<PrologueLoadKind>,
}

/// Device and session context needed to lower a [`Code`] into WGSL.
pub(crate) struct CodeTarget<'a> {
    pub capabilities: &'a blade_graphics::Capabilities,
    pub knobs: MatmulKnobs,
    pub cooperative: Option<CoopConfig>,
}

/// The cooperative M×N×K shapes a device lists for `multiply` operands with an
/// F32 accumulator.
pub(crate) fn cooperative_shapes(
    capabilities: &blade_graphics::CooperativeMatrix,
    multiply: crate::DType,
) -> &[[u32; 3]] {
    match multiply {
        crate::DType::BF16 => &capabilities.bf16_f32_shapes,
        crate::DType::F16 => &capabilities.f16_f32_shapes,
        crate::DType::F32 => &capabilities.f32_shapes,
        _ => &[],
    }
}

pub(crate) fn checked_matrix_unsupported(multiply: crate::DType) -> String {
    format!(
        "{multiply:?} matrix products require native 16x16x16, 8x16x16 or 8x8x8 {multiply:?}/F32 cooperative matrices"
    )
}

/// Shared workgroup-cooperative tile resolution used by [`module`].
pub(crate) fn workgroup_shape_and_subgroup(
    shape: WorkgroupMatmulShape,
    types: MatrixTypes,
    caps: &blade_graphics::Capabilities,
) -> Option<([u32; 3], u32)> {
    if shape.subgroup_rows != 0 {
        crate::codegen::native_subgroup_shape(caps, types.multiply)
    } else {
        Some(([16, 16, 16], 32))
    }
}

pub(crate) fn tiled_coop_shared_bytes_for(
    prologue: &[PrologueLoadKind],
    shape: CooperativeMatmulShape,
) -> u64 {
    let row_factors = prologue
        .iter()
        .filter(|kind| **kind == PrologueLoadKind::PerRow)
        .count();
    u64::from(shape.shared_bytes()) + row_factors as u64 * 64 * 4
}

impl Code {
    /// Project a dense product `dispatch` onto its pipeline key, or `None` if
    /// `dispatch` is not one of the six dense matmul entries (or is a
    /// horizontal multi-weight pack).
    pub(crate) fn from_dispatch(
        dispatch: &Dispatch,
        caps: &blade_graphics::Capabilities,
    ) -> Option<Self> {
        let layout = Layout::from_entry(&dispatch.shader)?;
        if dispatch.horizontal_batch >= 2 {
            return None;
        }
        let dims = dispatch.product().map(|p| [p.m, p.n, p.k]);
        let epilogue = dispatch
            .matmul_epilogue
            .as_ref()
            .map(EpilogueCode::from_compile);
        let prologue = || -> Vec<PrologueLoadKind> {
            dispatch
                .matmul_prologue
                .as_ref()
                .map(|p| p.factors.iter().map(|entry| entry.1.clone()).collect())
                .unwrap_or_default()
        };
        let (implementation, weights, epilogue, prologue) = match dispatch.kernel {
            // Its own pipeline, shared with batch-major products.
            Kernel::RegisterMatmul(_) => return None,
            Kernel::CooperativeWorkgroup { shape, types, .. } => {
                let types = types.unwrap_or(MatrixTypes::F32_AS_F16);
                let row_bounds = dims.is_some_and(|[m, _, _]| shape.ragged_rows(m));
                (
                    CodeImpl::CooperativeWorkgroup {
                        shape,
                        types,
                        row_bounds,
                    },
                    WeightFormat::F32,
                    epilogue,
                    Vec::new(),
                )
            }
            Kernel::CooperativeTyped {
                types, specialize, ..
            } => {
                let raw_layout = match dims {
                    Some(dims) => TypedMatmulLayout {
                        dimensions: specialize.then_some(dims),
                        ..TypedMatmulLayout::for_product(types, layout.has_addend(), dims)
                    },
                    None => TypedMatmulLayout::default(),
                };
                // The stored kernel does not read direct_inputs. Normalize it
                // before any pipeline lookup, including tuning reuse.
                let stored = crate::codegen::checked_cooperative_shape(cooperative_shapes(
                    &caps.cooperative_matrix,
                    types.multiply,
                ))
                .is_some_and(|shape| raw_layout.uses_stored_kernel(shape));
                let staging = TypedMatmulLayout {
                    direct_inputs: raw_layout.direct_inputs && !stored,
                    ..raw_layout
                };
                (
                    CodeImpl::CooperativeTyped { types, staging },
                    WeightFormat::F32,
                    None,
                    Vec::new(),
                )
            }
            Kernel::CooperativeTiled { shape, splits } => (
                CodeImpl::CooperativeTiled { shape, splits },
                WeightFormat::F32,
                None,
                prologue(),
            ),
            Kernel::SplitMatmul { shape, splits } => (
                CodeImpl::SplitScalar { shape, splits },
                dispatch.weight_format,
                None,
                Vec::new(),
            ),
            Kernel::Cooperative if !dispatch.weight_format.is_quantized() => (
                CodeImpl::Cooperative,
                dispatch.weight_format,
                epilogue,
                prologue(),
            ),
            Kernel::ScalarMatmul(shape) => (
                CodeImpl::Scalar(shape),
                dispatch.weight_format,
                epilogue,
                Vec::new(),
            ),
            Kernel::SmallTile => (
                CodeImpl::SmallScalar,
                dispatch.weight_format,
                epilogue,
                Vec::new(),
            ),
            _ => (
                CodeImpl::DefaultScalar,
                dispatch.weight_format,
                epilogue,
                Vec::new(),
            ),
        };
        Some(Self {
            layout,
            weights,
            implementation,
            epilogue,
            prologue,
        })
    }

    /// Name used by the profiler and by pipeline-statistics dumps.
    pub(crate) fn label(&self) -> String {
        use std::hash::{Hash, Hasher};
        let e = self.layout.entry_ref();
        match self.implementation {
            CodeImpl::CooperativeWorkgroup {
                shape,
                types,
                row_bounds,
            } => {
                let suffix = if row_bounds { "-row-bounds" } else { "" };
                let epi = if self.epilogue.is_some() {
                    "-epilogue"
                } else {
                    ""
                };
                format!("{e:?}:workgroup-{shape:?}-{types:?}{suffix}{epi}")
            }
            CodeImpl::CooperativeTyped { types, staging } => {
                let multiply = MatrixTypes::name(types.multiply);
                let operands = if types.a == types.multiply && types.b == types.multiply {
                    multiply.to_owned()
                } else {
                    let (a, b) = (MatrixTypes::name(types.a), MatrixTypes::name(types.b));
                    format!("{a}-{b}-as-{multiply}")
                };
                let output = MatrixTypes::name(types.output);
                format!("{e:?}:cooperative-{operands}-{output}-output:{staging:?}")
            }
            CodeImpl::CooperativeTiled { shape, splits } => {
                format!(
                    "{e:?}:native-f32-tiled-{shape:?}-split-k-{splits}:{:?}",
                    self.prologue
                )
            }
            CodeImpl::SplitScalar { shape, splits } => {
                format!("{e:?}:split-k-{:?}-{splits}-{shape:?}", self.weights)
            }
            CodeImpl::Cooperative => {
                if self.epilogue.is_some() {
                    let mut hasher = std::hash::DefaultHasher::new();
                    self.hash(&mut hasher);
                    format!("{e:?}:cooperative-epilogue:{:016x}", hasher.finish())
                } else if !self.prologue.is_empty() {
                    if self.weights == WeightFormat::F32 {
                        format!("{e:?}:cooperative-prologue:{:?}", self.prologue)
                    } else {
                        format!(
                            "{e:?}:cooperative-prologue-{:?}-weights:{:?}",
                            self.weights, self.prologue
                        )
                    }
                } else if self.weights == WeightFormat::F32 {
                    format!("{e:?}:cooperative")
                } else {
                    format!("{e:?}:cooperative-{:?}-weights", self.weights)
                }
            }
            CodeImpl::Scalar(shape) => {
                if self.epilogue.is_some() {
                    let mut hasher = std::hash::DefaultHasher::new();
                    self.hash(&mut hasher);
                    format!("{e:?}:scalar-epilogue:{:016x}", hasher.finish())
                } else {
                    format!("{e:?}:scalar-{:?}-{shape:?}", self.weights)
                }
            }
            CodeImpl::SmallScalar => {
                if self.epilogue.is_some() {
                    let mut hasher = std::hash::DefaultHasher::new();
                    self.hash(&mut hasher);
                    format!("{e:?}:scalar-epilogue:{:016x}", hasher.finish())
                } else if self.weights.uses_reduced_storage() {
                    format!("{e:?}:weight-{:?}-small-tile", self.weights)
                } else {
                    format!("{e:?}:small-tile")
                }
            }
            CodeImpl::DefaultScalar => {
                if self.epilogue.is_some() {
                    let mut hasher = std::hash::DefaultHasher::new();
                    self.hash(&mut hasher);
                    format!("{e:?}:scalar-epilogue:{:016x}", hasher.finish())
                } else if self.weights.uses_reduced_storage() {
                    format!("{e:?}:weight-{:?}", self.weights)
                } else {
                    format!("{e:?}:scalar")
                }
            }
        }
    }
}

/// Check per-dispatch dimension legality that is not encoded in [`Code`]
/// (because valid shapes share one pipeline across compatible dimensions).
pub(crate) fn validate_dispatch(
    dispatch: &Dispatch,
    code: &Code,
    caps: &blade_graphics::Capabilities,
) -> Result<(), String> {
    match code.implementation {
        CodeImpl::CooperativeWorkgroup { shape, types, .. }
            if !dispatch.product().is_some_and(|p| {
                shape.supports_layout(caps, types, Some(code.layout.group()), p.m, p.n, p.k)
            }) =>
        {
            return Err("unsupported workgroup cooperative product".to_owned());
        }
        CodeImpl::CooperativeTiled { shape, .. }
            if !dispatch
                .product()
                .is_some_and(|p| shape.fits_dimensions(p.m, p.n, p.k)) =>
        {
            return Err("tiled cooperative matmul requires complete aligned tiles".to_owned());
        }
        _ => {}
    }
    Ok(())
}

/// Sole codegen entry point for dense matrix products.
///
/// Takes only [`Code`] and [`CodeTarget`] — never `&Dispatch` — so the
/// pipeline key is guaranteed to determine the generated WGSL, layout, and
/// entry point.
pub(crate) fn module(
    code: &Code,
    target: &CodeTarget<'_>,
) -> Result<(ShaderModule, blade_graphics::ShaderDataLayout, &'static str), String> {
    let group = code.layout.group();
    let entry = code.layout.entry_ref();
    let entry_point = entry.entry_point();
    let layout = if !code.prologue.is_empty() {
        crate::runtime::matmul_with_prologue_layout(code.prologue.len())
    } else {
        crate::runtime::shader_data_layout(entry)
    };
    let epi = code.epilogue.as_ref().map(EpilogueCode::to_compile);
    let pro = (!code.prologue.is_empty()).then(|| MatMulPrologue {
        factors: code
            .prologue
            .iter()
            .map(|kind| (BufferRef(0), kind.clone()))
            .collect(),
    });
    let cooperative = || {
        target
            .cooperative
            .expect("cooperative dispatch needs a qualified device configuration")
    };
    let shader_module = match code.implementation {
        CodeImpl::DefaultScalar => crate::codegen::generate_matmul_with_epilogue(
            group,
            epi.as_ref(),
            MatMulOptions {
                format: code.weights,
                tile: MatMulTile::Large,
                knobs: target.knobs,
            },
        ),
        CodeImpl::SmallScalar => {
            let tile = if epi.is_none() && code.layout.transposed_with_addend() {
                MatMulTile::Large
            } else {
                MatMulTile::Small
            };
            crate::codegen::generate_matmul_with_epilogue(
                group,
                epi.as_ref(),
                MatMulOptions {
                    format: code.weights,
                    tile,
                    knobs: target.knobs,
                },
            )
        }
        CodeImpl::Scalar(shape) => {
            let limit = target.capabilities.max_compute_shared_memory_size;
            if limit != 0 && shape.shared_bytes() > limit {
                return Err(format!(
                    "scalar tile requires {} shared bytes, device limit is {limit}",
                    shape.shared_bytes()
                ));
            }
            let knobs = MatmulKnobs {
                k_stage: shape.k_stage,
                interleave_columns: shape.interleave_columns,
                unroll_k: shape.unroll_k,
                vector_fragments: shape.vector_fragments,
                ..target.knobs
            };
            crate::codegen::generate_matmul_with_epilogue(
                group,
                epi.as_ref(),
                MatMulOptions {
                    format: code.weights,
                    tile: shape.geometry(),
                    knobs,
                },
            )
        }
        CodeImpl::SplitScalar { shape, splits } => {
            crate::codegen::generate_split_matmul(group, shape, splits, code.weights, target.knobs)
        }
        CodeImpl::Cooperative => {
            let config = cooperative();
            let subgroup = target.capabilities.subgroup_size;
            if let Some(ref epi) = epi {
                crate::codegen::generate_coop_matmul_with_dag_epilogue(
                    group,
                    &config,
                    epi,
                    code.weights,
                    subgroup,
                )
            } else if let Some(ref pro) = pro {
                let (add, variant) =
                    crate::codegen::coop_shape(group).expect("cooperative matrix group");
                crate::codegen::gen_matmul_coop_with_prologue(
                    add,
                    variant,
                    &config,
                    pro,
                    code.weights,
                    subgroup,
                )
            } else {
                crate::codegen::generate_module_coop_weighted(
                    group,
                    &config,
                    code.weights,
                    subgroup,
                )
            }
        }
        CodeImpl::CooperativeTiled { shape, splits } => {
            let config = cooperative();
            if config.tile_size != 16 || config.use_f16_input {
                return Err("tiled cooperative matmul requires native 16x16 f32".to_owned());
            }
            let shared_limit = target.capabilities.max_compute_shared_memory_size;
            if shared_limit != 0
                && tiled_coop_shared_bytes_for(&code.prologue, shape) > u64::from(shared_limit)
            {
                return Err("tiled cooperative matmul exceeds device shared memory".to_owned());
            }
            crate::codegen::generate_tiled_coop_matmul(
                group,
                shape,
                splits,
                pro.as_ref(),
                target.capabilities.subgroup_size,
            )
        }
        CodeImpl::CooperativeTyped { types, staging } => {
            let native = crate::codegen::checked_cooperative_shape(cooperative_shapes(
                &target.capabilities.cooperative_matrix,
                types.multiply,
            ))
            .ok_or_else(|| checked_matrix_unsupported(types.multiply))?;
            crate::codegen::generate_module_coop_typed_with_subgroup(
                group,
                types,
                native,
                staging,
                target.capabilities.subgroup_size,
            )
        }
        CodeImpl::CooperativeWorkgroup {
            shape,
            types,
            row_bounds,
        } => {
            if !epi
                .as_ref()
                .is_none_or(|e| shape.subgroup_rows != 0 && e.inputs.is_empty())
            {
                return Err("unsupported epilogue on workgroup cooperative product".to_owned());
            }
            let (native, subgroup) =
                workgroup_shape_and_subgroup(shape, types, target.capabilities)
                    .ok_or_else(|| checked_matrix_unsupported(types.multiply))?;
            crate::codegen::generate_workgroup_matmul_for_device(
                group,
                shape,
                types,
                native,
                subgroup,
                row_bounds,
                epi.as_ref(),
            )
        }
    };
    Ok((shader_module, layout, entry_point))
}

/// One dispatch's product.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Problem {
    pub layout: Layout,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    /// Packed or f16 weights, which need a storage-aware kernel.
    pub reduced_storage: bool,
    /// f16 weights, which f16-input cooperative tiles also read.
    pub f16_weights: bool,
    /// Autodiff's mark on derivative work that must keep f32 operands.
    pub requires_full_precision: bool,
    /// A fused epilogue that binds buffers of its own, which only the
    /// scalar kernels stage.
    pub epilogue_inputs: bool,
    /// A measured schedule or explicit storage policy fixed the kernel.
    pub pinned: bool,
    /// Any fused epilogue, which only the tiled kernels apply.
    pub epilogue: bool,
    /// The compiler fixed the schedule (`Dispatch::schedule_locked`).
    pub locked: bool,
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
            f16_weights: false,
            requires_full_precision: false,
            epilogue_inputs: false,
            pinned: false,
            epilogue: false,
            locked: false,
            compiled_workgroups: m.div_ceil(64) * n.div_ceil(64),
        }
    }

    /// The product `dispatch` computes, if it is a dense one.
    pub(crate) fn of(dispatch: &Dispatch) -> Option<Self> {
        let product = dispatch.product()?;
        Some(Self {
            layout: product.layout,
            m: product.m,
            n: product.n,
            k: product.k,
            reduced_storage: product.weights.uses_reduced_storage(),
            f16_weights: product.weights == WeightFormat::F16,
            requires_full_precision: dispatch.full_precision(),
            epilogue_inputs: dispatch
                .matmul_epilogue
                .as_ref()
                .is_some_and(|epilogue| !epilogue.inputs.is_empty()),
            pinned: dispatch.scalar_matmul().is_some()
                || matches!(
                    dispatch.kernel,
                    Kernel::SplitMatmul { .. }
                        | Kernel::CooperativeTiled { .. }
                        | Kernel::CooperativeTyped { .. }
                        | Kernel::CooperativeWorkgroup { .. }
                ),
            epilogue: dispatch.matmul_epilogue.is_some(),
            locked: dispatch.schedule_locked,
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
    /// [`TuningKnobs::prefer_matmul`](crate::TuningKnobs).
    pub prefer: Option<Path>,
}

/// How a dispatch computes its product, most preferred first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Path {
    /// Cooperative-matrix tiles in the device's configuration.
    Cooperative,
    /// The direct register kernel ([`RegisterTile`]), where cooperative
    /// matrices are absent.
    Register,
    /// 32-wide scalar tiles, for products too small to occupy the device
    /// with 64-wide ones.
    SmallTile,
    /// 128-row by 64-column scalar tiles, for products large enough to
    /// occupy the device with them.
    WideTile,
    /// The kernel the compiler emitted: 64-wide scalar tiles, or the
    /// schedule a measurement pinned. The fallback.
    Compiled,
}

/// 128 rows by 64 columns: an 8×4 register tile per thread, against 4×4
/// for the compiled 64×64 kernel, reads shared memory 1.5 times less often
/// per multiply-add. Products with at least [`occupancy::SCALAR_OCCUPIED`]
/// such tiles take 0.60–0.97 of the 64×64 time on Arc B570 and 0.85–0.92 on
/// RTX 5070 (4096³: 26.4 against 44.1 ms, and 12.4 against 14.6); with 16 or
/// fewer they run slower on both. Interleaved columns suit the B570 (2048³:
/// 4.7 TFLOP/s against 3.4) and cost the RTX 5070 a little (12.4 against
/// 13.5).
pub(crate) const WIDE_TILE: ScalarMatmulShape = ScalarMatmulShape {
    tile_size: 128,
    tile_n: 64,
    k_stage: 16,
    interleave_columns: true,
    unroll_k: false,
    vector_fragments: false,
};

impl Family for Path {
    type Problem = Problem;
    type Target = Target;
    const PREFERENCE: &'static [Self] = &[
        Path::Cooperative,
        Path::Register,
        Path::SmallTile,
        Path::WideTile,
        Path::Compiled,
    ];

    fn preferred(target: &Target) -> Option<Self> {
        target.prefer
    }

    fn tile(self, _: &Problem, target: &Target) -> Option<[u32; 2]> {
        Some(match self {
            Path::Cooperative => [target.cooperative?.matmul_output_tile(); 2],
            Path::Register => [REGISTER_TILE.rows, 4],
            Path::SmallTile => [32; 2],
            Path::WideTile => [WIDE_TILE.rows(), WIDE_TILE.cols()],
            Path::Compiled => [64; 2],
        })
    }

    fn admits(self, problem: &Problem, target: &Target) -> Result<(), Rejection> {
        match self {
            Path::Cooperative => {
                let Some(ref config) = target.cooperative else {
                    return Err("no cooperative matrices");
                };
                if problem.pinned {
                    Err("the kernel is fixed by a schedule or storage policy")
                } else if problem.layout.transposed_with_addend() && config.use_f16_input {
                    Err("f16 transposed addends are not qualified for automatic promotion")
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
                        occupancy::MIN_F16_COOP_WORKGROUPS
                    } else {
                        occupancy::MIN_F32_COOP_WORKGROUPS
                    };
                    if workgroups < minimum {
                        // Policy, not legality: tuning may still measure it.
                        Err("too few tiles to amortize staging")
                    } else {
                        Ok(())
                    }
                }
            }
            Path::Register => register_admits(problem, target),
            Path::SmallTile => {
                if problem.pinned {
                    Err("the kernel is fixed by a schedule or storage policy")
                } else if problem.reduced_storage {
                    Err("packed weights keep their compiled kernel")
                } else if problem.layout.transposed_with_addend() {
                    Err("no 32-wide kernel for transposed products with an addend")
                } else if problem.compiled_workgroups >= occupancy::SMALL_TILE_BELOW_WORKGROUPS {
                    Err("64-wide tiles already occupy the device")
                } else {
                    Ok(())
                }
            }
            Path::WideTile => {
                let [columns, rows, _] = wide_workgroups(problem);
                if problem.pinned {
                    Err("the kernel is fixed by a schedule or storage policy")
                } else if problem.reduced_storage {
                    Err("packed weights keep their compiled kernel")
                } else if u64::from(columns) * u64::from(rows)
                    < u64::from(occupancy::SCALAR_OCCUPIED)
                {
                    Err("too few 128x64 tiles to occupy the device")
                } else if !crate::compile::workgroups_within_portable_limits(wide_workgroups(
                    problem,
                )) {
                    Err("grid exceeds the portable dispatch limit")
                } else {
                    Ok(())
                }
            }
            Path::Compiled => Ok(()),
        }
    }
}

/// Register grids below this many workgroups leave the product to the
/// tiled kernels, which split K to occupy the device; the register kernel
/// walks the whole reduction in each thread.
const MIN_REGISTER_WORKGROUPS: u32 = 4;

/// Whether the register kernel computes `[m, n, k]` over `batch` products
/// on `target`, the rules [`Path::Register`] and batch-major products share.
pub(crate) fn register_admits_product(
    [m, n, k]: [u32; 3],
    batch: u32,
    target: &Target,
) -> Result<RegisterTile, Rejection> {
    // Policy: measured only where cooperative matrices are absent (Adreno
    // 740); devices with them keep their kernels unless tuning measures it.
    if target.cooperative.is_some() {
        return Err("measured only where cooperative matrices are absent");
    }
    if target.prefer.is_some_and(|path| path != Path::Register) {
        return Err("another path is preferred");
    }
    let tile = register_tile(m, n, k).ok_or("vec4 loads need N and K multiples of four")?;
    let grid = register_workgroups(tile, m, n, batch);
    if !crate::compile::workgroups_within_portable_limits(grid) {
        Err("grid exceeds the portable dispatch limit")
    } else if grid.iter().map(|&g| u64::from(g)).product::<u64>()
        < u64::from(MIN_REGISTER_WORKGROUPS)
    {
        Err("too few workgroups; split K instead")
    } else {
        Ok(tile)
    }
}

fn register_admits(problem: &Problem, target: &Target) -> Result<(), Rejection> {
    if problem.layout != Layout::Plain {
        Err("the register kernel computes plain products only")
    } else if problem.pinned || problem.locked {
        Err("the kernel is fixed by a schedule or storage policy")
    } else if problem.reduced_storage {
        Err("packed weights keep their compiled kernel")
    } else if problem.epilogue {
        Err("the register kernel has no epilogue")
    } else {
        register_admits_product([problem.m, problem.n, problem.k], 1, target).map(|_| ())
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
    let f16_tiles_read_f16 = problem.f16_weights && config.use_f16_input;
    if problem.reduced_storage && !f16_tiles_read_f16 {
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

/// Workgroups of [`WIDE_TILE`]: columns on X, rows on Y.
pub(crate) fn wide_workgroups(problem: &Problem) -> [u32; 3] {
    [
        problem.n.div_ceil(WIDE_TILE.cols()),
        problem.m.div_ceil(WIDE_TILE.rows()),
        1,
    ]
}

/// Register tile of the direct product ([`crate::compile::Kernel::RegisterMatmul`]):
/// each thread accumulates `rows` rows of C by four columns, read and
/// written as one vec4, from A rows read as vec4 along K and B rows read as
/// vec4 along N. Threads enumerate (row block, column) pairs with columns
/// fastest, so a narrow output still fills the workgroup and adjacent
/// threads read adjacent columns of B. No workgroup memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegisterTile {
    pub rows: u32,
    pub threads: u32,
}

/// The register tile for a plain or batch-major `m × n` product over `k`,
/// if it applies: vec4 loads need `n` and `k` to be multiples of four.
///
/// Four rows and 64 threads, measured on Adreno 740 (Quest 3S) over
/// D-FINE-N's products: 0.88 ms for its seven distinct shapes against about
/// 3.7 ms on the shared-memory tiles. Other tiles won single shapes by up
/// to 1.3× and lost others by more.
pub(crate) fn register_tile(m: u32, n: u32, k: u32) -> Option<RegisterTile> {
    let aligned = m != 0 && n.is_multiple_of(4) && n != 0 && k.is_multiple_of(4) && k != 0;
    aligned.then_some(REGISTER_TILE)
}

/// The tile [`register_tile`] picks.
const REGISTER_TILE: RegisterTile = RegisterTile {
    rows: 4,
    threads: 64,
};

/// The grid of a [`RegisterTile`] over `batch` products of `m × n`.
pub(crate) fn register_workgroups(tile: RegisterTile, m: u32, n: u32, batch: u32) -> [u32; 3] {
    let threads = (n / 4).saturating_mul(m.div_ceil(tile.rows));
    [threads.div_ceil(tile.threads), 1, batch]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_tiles_need_vec4_columns_and_depth() {
        // D-FINE-N's attention product: 8 heads of 300x16 over K = 300.
        let tile = register_tile(300, 16, 300).expect("aligned");
        assert_eq!(register_workgroups(tile, 300, 16, 8), [5, 1, 8]);
        // Rows round up to whole tiles.
        assert_eq!(register_workgroups(tile, 301, 128, 1), [38, 1, 1]);
        assert!(register_tile(1200, 1, 33).is_none());
        assert!(register_tile(300, 64, 6).is_none());
    }

    #[test]
    fn register_path_states_why_it_declines() {
        let bare = Target {
            cooperative: None,
            allow_raw_f16: false,
            prefer: None,
        };
        let plain = Problem::plain(Layout::Plain, [300, 128, 512], false);
        assert_eq!(select(&plain, &bare), Path::Register);
        let coop = Target {
            cooperative: Some(CoopConfig {
                tile_size: 16,
                use_f16_input: true,
            }),
            ..bare
        };
        let preferred = Target {
            prefer: Some(Path::SmallTile),
            ..bare
        };
        for (problem, target, reason) in [
            (
                plain,
                coop,
                "measured only where cooperative matrices are absent",
            ),
            (plain, preferred, "another path is preferred"),
            (
                Problem::plain(Layout::TransposedB, [300, 128, 512], false),
                bare,
                "the register kernel computes plain products only",
            ),
            (
                Problem::plain(Layout::Plain, [300, 130, 512], false),
                bare,
                "vec4 loads need N and K multiples of four",
            ),
            (
                Problem {
                    locked: true,
                    ..plain
                },
                bare,
                "the kernel is fixed by a schedule or storage policy",
            ),
            (
                Problem {
                    epilogue: true,
                    ..plain
                },
                bare,
                "the register kernel has no epilogue",
            ),
            (
                // Two workgroups walking K = 1024: split K instead.
                Problem::plain(Layout::Plain, [6, 256, 1024], false),
                bare,
                "too few workgroups; split K instead",
            ),
        ] {
            assert_eq!(Path::Register.admits(&problem, &target), Err(reason));
        }
        assert_eq!(
            register_admits_product([300, 16, 300], 70_000, &bare),
            Err("grid exceeds the portable dispatch limit")
        );
    }

    #[test]
    fn matrix_partitions_keep_short_and_occupied_products_unsplit() {
        for (m, n, k, expected) in [
            (8, 256, 1023, 1),
            (8, 256, 1024, 4),
            (8, 256, 2048, 8),
            (8, 256, 4096, 16),
            (8, 256, 65536, 16),
            (128, 256, 2560, 4),
            (512, 256, 2560, 1),
            (33, 65, 1031, 4),
            (u32::MAX, u32::MAX, 2048, 1),
        ] {
            assert_eq!(matrix_splits(m, n, k), expected, "{m}x{n}x{k}");
        }
    }

    fn product(m: u32, n: u32, k: u32) -> Problem {
        Problem::plain(Layout::Plain, [m, n, k], false)
    }

    const F16: CoopConfig = CoopConfig {
        tile_size: 16,
        use_f16_input: true,
    };
    const F32: CoopConfig = CoopConfig {
        tile_size: 8,
        use_f16_input: false,
    };

    fn target(config: &CoopConfig) -> Target {
        Target {
            cooperative: Some(*config),
            allow_raw_f16: false,
            prefer: None,
        }
    }

    #[test]
    fn compiled_admits_everything() {
        let bare = Target {
            cooperative: None,
            allow_raw_f16: false,
            prefer: None,
        };
        for problem in [product(1, 1, 1), product(4096, 4096, 4096)] {
            assert_eq!(Path::Compiled.admits(&problem, &bare), Ok(()));
        }
        // Without cooperative matrices, aligned products take registers.
        assert_eq!(select(&product(4096, 4096, 64), &bare), Path::Register);
        // Among the scalar tiles:
        let scalar = |problem: &Problem| {
            [Path::SmallTile, Path::WideTile, Path::Compiled]
                .into_iter()
                .find(|path| path.admits(problem, &bare).is_ok())
                .unwrap()
        };
        assert_eq!(scalar(&product(4096, 4096, 64)), Path::WideTile);
        // 32 tiles of 128x64 are too few; 64 are enough.
        assert_eq!(scalar(&product(512, 512, 64)), Path::Compiled);
        assert_eq!(scalar(&product(1024, 512, 64)), Path::WideTile);
        assert_eq!(scalar(&product(64, 64, 64)), Path::SmallTile);
        // Past 65535 rows of tiles, only the compiled kernel wraps rows.
        assert_eq!(select(&product(128 * 65536, 64, 64), &bare), Path::Compiled);
    }

    #[test]
    fn stored_f16_requires_plain_f16_cooperative_tiles() {
        let problem = Problem::of(&Dispatch {
            shader: crate::compile::ShaderEntry::MatMul,
            params: vec![257, 65, 512, 0],
            workgroups: [8, 5, 1],
            weight_format: crate::compile::WeightFormat::F16,
            ..Default::default()
        })
        .unwrap();
        assert!(problem.reduced_storage && problem.f16_weights);
        assert_eq!(select(&problem, &target(&F16)), Path::Cooperative);
        assert_eq!(select(&problem, &target(&F32)), Path::Compiled);
        let quantized = Problem {
            f16_weights: false,
            ..problem
        };
        assert_eq!(select(&quantized, &target(&F16)), Path::Compiled);
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
                "the kernel is fixed by a schedule or storage policy",
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
    fn transposed_addends_promote_on_f32_tiles() {
        for layout in [Layout::TransposedAAdd, Layout::TransposedBAdd] {
            let large = Problem {
                requires_full_precision: true,
                ..Problem::plain(layout, [1024, 1024, 256], false)
            };
            assert_eq!(select(&large, &target(&F32)), Path::Cooperative);
            // F16 stays out even when a session explicitly allows it.
            assert_eq!(select(&large, &target(&F16)), Path::WideTile);
            let allowed = Target {
                allow_raw_f16: true,
                ..target(&F16)
            };
            assert_eq!(select(&large, &allowed), Path::WideTile);
            // The shared occupancy floor still applies.
            let small = Problem::plain(layout, [32, 32, 32], false);
            assert_eq!(select(&small, &target(&F32)), Path::Compiled);
        }
    }

    impl crate::kernels::sweep::Swept for Path {
        type Kind = Layout;
        const KINDS: &'static [Layout] = &[
            Layout::Plain,
            Layout::PlainAdd,
            Layout::TransposedA,
            Layout::TransposedAAdd,
            Layout::TransposedB,
            Layout::TransposedBAdd,
        ];

        fn graph(layout: Layout, extents: [u32; 3]) -> crate::Graph {
            use crate::graph::Op;
            let [m, n, k] = extents.map(|extent| extent as usize);
            let mut g = crate::Graph::new();
            let addend = layout.has_addend().then(|| g.parameter("d", &[m, n]));
            let (product, fused) = match layout {
                Layout::Plain | Layout::PlainAdd => {
                    let (a, b) = (g.parameter("a", &[m, k]), g.parameter("b", &[k, n]));
                    (g.matmul(a, b), Op::FusedMatMulAdd)
                }
                Layout::TransposedA | Layout::TransposedAAdd => {
                    let (a, b) = (g.parameter("a", &[k, m]), g.parameter("b", &[k, n]));
                    (g.matmul_at(a, b), Op::FusedMatMulATAdd)
                }
                Layout::TransposedB | Layout::TransposedBAdd => {
                    let (a, b) = (g.parameter("a", &[m, k]), g.parameter("b", &[n, k]));
                    (g.matmul_bt(a, b), Op::FusedMatMulBTAdd)
                }
            };
            if let Some(addend) = addend {
                // The fused forms have no builder: the optimizer makes them.
                let node = &mut g.nodes_mut()[product as usize];
                node.op = fused;
                node.inputs.push(addend);
            }
            g.set_outputs(vec![product]);
            g
        }

        fn problem(layout: Layout, extents: [u32; 3]) -> Problem {
            Problem::plain(layout, extents, false)
        }

        fn prefer(self, options: &mut crate::reference::gpu::Options) {
            options.compile.knobs.prefer_matmul = Some(self);
        }

        fn target(
            self,
            session: &crate::Session,
            options: &crate::reference::gpu::Options,
        ) -> Target {
            Target {
                cooperative: session.coop_config(),
                allow_raw_f16: options.coop == crate::CoopPolicy::AllowF16,
                prefer: Some(self),
            }
        }

        fn ran(dispatch: &Dispatch) -> Option<Option<Self>> {
            Layout::of(dispatch.shader.shader_group())?;
            Some(match dispatch.kernel {
                Kernel::Cooperative => Some(Path::Cooperative),
                Kernel::SmallTile => Some(Path::SmallTile),
                Kernel::RegisterMatmul(_) => Some(Path::Register),
                Kernel::ScalarMatmul(shape) if shape == WIDE_TILE => Some(Path::WideTile),
                Kernel::Default => Some(Path::Compiled),
                _ => None,
            })
        }
    }

    /// Every path the device has, preferred in turn, in every layout.
    #[test]
    fn every_path_against_the_reference() {
        crate::kernels::sweep::run::<Path>();
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

    #[test]
    fn product_decode_launch_and_code_agree_across_layouts() {
        for layout in [
            Layout::Plain,
            Layout::PlainAdd,
            Layout::TransposedA,
            Layout::TransposedAAdd,
            Layout::TransposedB,
            Layout::TransposedBAdd,
        ] {
            let (m, n, k) = (96, 128, 256);
            let dispatch = Dispatch {
                shader: layout.entry_ref().clone(),
                params: layout.params_words(m, n, k),
                ..Default::default()
            };
            let decoded = dispatch.product().expect("dense product");
            assert_eq!(
                (decoded.layout, decoded.m, decoded.n, decoded.k),
                (layout, m, n, k)
            );
            assert_eq!(dispatch.mnk(), Some((m, n, k)));

            let default_launch = decoded.launch(Variant::DefaultScalar, 1);
            assert_eq!(default_launch.workgroups, [2, 2, 1]);
            assert_eq!(default_launch.params, [m, n, k, 2]);
            assert_eq!(
                decoded.uniform_params(&Kernel::Default, default_launch.workgroups),
                default_launch.params
            );

            let typed_launch = decoded.launch(Variant::CooperativeTyped { specialize: false }, 4);
            assert_eq!(typed_launch.workgroups, [3, 4, 4]);
            assert_eq!(typed_launch.params, [m, n, k, 4]);
            assert_eq!(
                decoded.uniform_params(
                    &Kernel::CooperativeTyped {
                        types: MatrixTypes::F32_AS_F16,
                        splits: 4,
                        specialize: false,
                    },
                    typed_launch.workgroups
                ),
                typed_launch.params
            );

            let caps = blade_graphics::Capabilities::default();
            let code = Code::from_dispatch(&dispatch, &caps).expect("dense product code");
            assert_eq!(code.layout, layout);
            assert_eq!(code.implementation, CodeImpl::DefaultScalar);
            let (shader, _, entry_point) = module(
                &code,
                &CodeTarget {
                    capabilities: &caps,
                    knobs: MatmulKnobs::default(),
                    cooperative: None,
                },
            )
            .expect("default scalar module");
            assert!(!shader.source.is_empty());
            assert_eq!(entry_point, layout.entry_ref().entry_point());
        }
    }
}
