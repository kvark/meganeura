//! Unified GPU occupancy thresholds and reduction-partition policy.
//!
//! Kernel families, lowering passes, codegen tile selection, and measured
//! search all ask the same two questions:
//! 1. Does an output tile grid occupy the device (or warrant a narrower,
//!    wider, or cooperative tile)?
//! 2. When an output grid is under-occupied over a long reduction `k`, how
//!    many reduction partitions should split `k` to fill the device without
//!    making each partition mostly staging overhead?
//!
//! Centralizing these thresholds and the single [`PartitionPolicy`] formula
//! here ensures that compile-time lowering, kernel-family admission, tile
//! sizing, and measured search share one occupancy model. The three
//! measured-search split policies scale with `MachineProfile::parallel_width`,
//! while the compile-time typed, scalar, convolution, and native-F32 policies
//! use fixed occupancy thresholds.

/// Output workgroups at which a scalar tiling fills the device: wide 128×64
/// tiles require at least this many, scalar matmul/conv grids below this
/// narrow their tile or split along K.
pub(crate) const SCALAR_OCCUPIED: u32 = 64;

/// Output workgroups below which F16 cooperative staging costs more than it
/// saves on discrete GPUs. Representative transformer projections with 20–72
/// tiles regress on discrete NVIDIA GPUs; wider backward and MLP products win.
/// Shared by dense matmul, forward convolution, subgroup convolution tile
/// sizing, and search promotion.
pub(crate) const MIN_F16_COOP_WORKGROUPS: u64 = 128;

/// Output workgroups below which F32 cooperative tiles (whose staging is
/// cheaper than F16) fail to occupy the device. Also keeps 64-wide scalar
/// convolution tiles eligible for F32 cooperative promotion from this many
/// workgroups on.
pub(crate) const MIN_F32_COOP_WORKGROUPS: u64 = 16;

/// Output workgroups below which cooperative convolution input-gradient
/// staging loses to the scalar kernel: the input gradient's scalar staging
/// runs 64 threads against the scalar kernel's 256.
pub(crate) const MIN_GRAD_INPUT_WORKGROUPS: u64 = 16;

/// 64-wide scalar matmul tilings with fewer workgroups than this switch to
/// 32-wide tiles (four times as many workgroups) to occupy GPUs with many SMs.
pub(crate) const SMALL_TILE_BELOW_WORKGROUPS: u32 = 16;

/// Target concurrent workgroups scaled by `parallel_width`
/// (`MachineProfile::parallel_width`, typically concurrent workgroups across
/// all compute units) when known, falling back to `default` otherwise.
pub(crate) fn target_workgroups(parallel_width: Option<u32>, waves: u32, default: u32) -> u32 {
    match parallel_width.filter(|&w| w != 0) {
        Some(width) => width.saturating_mul(waves).clamp(64, 512),
        None => default,
    }
}

/// Parameters governing how many reduction partitions (`1..=max_splits`)
/// split a reduction of length `k` across an `output_groups` tile grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PartitionPolicy {
    /// Output grids with at least this many workgroups stay unsplit (`1`).
    pub occupied_groups: u32,
    /// Total workgroups (`output_groups * splits`) aimed for when splitting.
    pub target_groups: u32,
    /// Reductions shorter than this stay unsplit (`1`).
    pub min_reduction: u32,
    /// Minimum reduction work each partition must own (`splits <= k / min_chunk`).
    pub min_chunk: u32,
    /// Hard upper bound on returned partitions.
    pub max_splits: u32,
    /// Restrict partition counts to powers of two (for native-F32 cooperative
    /// split-K schedules).
    pub power_of_two: bool,
}

impl PartitionPolicy {
    /// Largest partition count `k` allows under this policy (ignoring output
    /// occupancy), respecting `min_reduction`, `min_chunk`, `max_splits`, and
    /// `power_of_two`. Always returns at least `1`.
    pub(crate) fn max_splits(self, k: u32) -> u32 {
        if k < self.min_reduction {
            return 1;
        }
        let cap = (k / self.min_chunk.max(1)).min(self.max_splits);
        if cap < 2 {
            return 1;
        }
        if self.power_of_two {
            1u32 << cap.ilog2()
        } else {
            cap
        }
    }

    /// Reduction partitions for `output_groups` output workgroups over a
    /// reduction of length `k`. Always returns at least `1`.
    pub(crate) fn splits(self, output_groups: u32, k: u32) -> u32 {
        if output_groups == 0 || output_groups >= self.occupied_groups {
            return 1;
        }
        let cap = self.max_splits(k);
        if cap < 2 {
            return 1;
        }
        let want = self.target_groups.div_ceil(output_groups);
        if self.power_of_two {
            want.next_power_of_two().min(cap).max(1)
        } else {
            want.min(cap).max(1)
        }
    }

    /// Like [`Self::splits`], returning `Some(splits)` only when splitting (`>= 2`).
    pub(crate) fn multi_splits(self, output_groups: u32, k: u32) -> Option<u32> {
        let splits = self.splits(output_groups, k);
        (splits >= 2).then_some(splits)
    }
}

/// Checked typed dense products on a 32×32 output grid: fill small grids up
/// to 128 workgroups when `k >= 1024`, keeping at least 256 K per partition.
pub(crate) const TYPED_MATMUL_SPLITS: PartitionPolicy = PartitionPolicy {
    occupied_groups: 128,
    target_groups: 128,
    min_reduction: 1024,
    min_chunk: 256,
    max_splits: 16,
    power_of_two: false,
};

/// Scalar dense products on a 32×32 output grid: split grids below
/// [`SCALAR_OCCUPIED`] toward 128 workgroups with at least eight 16-deep K
/// stages (128 elements) per partition.
pub(crate) const SCALAR_MATMUL_SPLITS: PartitionPolicy = PartitionPolicy {
    occupied_groups: SCALAR_OCCUPIED,
    target_groups: 128,
    min_reduction: 0,
    min_chunk: 128,
    max_splits: 16,
    power_of_two: false,
};

/// Scalar convolution weight gradients on a 64×64 output grid: split grids
/// below [`SCALAR_OCCUPIED`] toward 256 workgroups with at least sixteen
/// 16-deep K stages (256 positions) per partition.
pub(crate) const CONV_WEIGHT_SPLITS: PartitionPolicy = PartitionPolicy {
    occupied_groups: SCALAR_OCCUPIED,
    target_groups: 256,
    min_reduction: 0,
    min_chunk: 256,
    max_splits: 65_535,
    power_of_two: false,
};

/// Native-F32 cooperative tiled products: power-of-two K partitions up to 16,
/// raising output grids toward 512 workgroups with at least 128 K each.
pub(crate) const NATIVE_F32_TILED_SPLITS: PartitionPolicy = PartitionPolicy {
    occupied_groups: 512,
    target_groups: 512,
    min_reduction: 0,
    min_chunk: 128,
    max_splits: 16,
    power_of_two: true,
};

/// Low-occupancy gathered subgroup convolutions during search: split when
/// below 4 waves (default 192) toward 8 waves (default 384), measured in
/// 32-deep K stages (`min_reduction = 32` stages, `min_chunk = 8` stages).
pub(crate) fn gathered_conv_splits(parallel_width: Option<u32>) -> PartitionPolicy {
    PartitionPolicy {
        occupied_groups: target_workgroups(parallel_width, 4, 192),
        target_groups: target_workgroups(parallel_width, 8, 384),
        min_reduction: 32,
        min_chunk: 8,
        max_splits: 65_535,
        power_of_two: false,
    }
}

/// Subgroup convolution weight gradients during search: aim for 4 waves
/// (default 192) with at least 128 reduction positions per chunk.
pub(crate) fn conv_weight_search_splits(parallel_width: Option<u32>) -> PartitionPolicy {
    let target = target_workgroups(parallel_width, 4, 192);
    PartitionPolicy {
        occupied_groups: target,
        target_groups: target,
        min_reduction: 0,
        min_chunk: 128,
        max_splits: 65_535,
        power_of_two: false,
    }
}

/// Long dense weight/input gradients during search: aim for 3 waves
/// (default 144) on 64×64 output tiles with at least 256 K per chunk.
pub(crate) fn long_gradient_splits(parallel_width: Option<u32>) -> PartitionPolicy {
    let target = target_workgroups(parallel_width, 3, 144);
    PartitionPolicy {
        occupied_groups: target,
        target_groups: target,
        min_reduction: 0,
        min_chunk: 256,
        max_splits: 65_535,
        power_of_two: false,
    }
}

/// How many row slices a cooperative `SumRows` launch needs.
///
/// Eight lanes already share each column. A tall matrix (`rows / 8 > 64`) with
/// fewer than 128 column groups (`ceil(cols / 32)`) splits rows toward 256
/// workgroups, keeping at least 8 lane trips (64 rows) per slice.
pub(crate) fn row_reduction_splits(rows: u32, cols: u32) -> u32 {
    let groups = cols.div_ceil(32).max(1);
    let trip = rows.div_ceil(8);
    if trip <= 64 || groups >= 128 {
        return 1;
    }
    // About 256 workgroups, each slice at least 8 rows a lane. Whisper's bias
    // gradients (1500 × 384) took 7.0 µs on 6 slices of 32 rows a lane vs
    // 4.8 µs on 22 (RTX 5070, profiled); 12544 × 64 dropped from 6.9 to 5.1 µs.
    // `trip.div_ceil(8)` lets a partial final chunk participate in the cap.
    let want = 256u32.div_ceil(groups);
    let cap = trip.div_ceil(8).clamp(2, 256);
    want.clamp(2, cap)
}

/// Pick the first (widest) tile in `tiles` whose output grid
/// `ceil(m / tr) * ceil(n / tc) * layers` reaches `min_groups` (and optionally
/// fits inside `(m, n)` without padding past the product's extent), falling
/// back to the last (narrowest) tile.
pub(crate) fn widest_occupied_tile<T: Copy>(
    tiles: &[T],
    dims: impl Fn(T) -> (u32, u32),
    m: u32,
    n: u32,
    layers: u32,
    min_groups: u64,
    require_fit: bool,
) -> T {
    let layers = u64::from(layers.max(1));
    tiles
        .iter()
        .copied()
        .find(|&tile| {
            let (tr, tc) = dims(tile);
            (!require_fit || (tr <= m && tc <= n))
                && u64::from(m.div_ceil(tr)) * u64::from(n.div_ceil(tc)) * layers >= min_groups
        })
        .unwrap_or_else(|| *tiles.last().expect("non-empty tile menu"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_workgroups_scales_and_clamps_parallel_width() {
        assert_eq!(target_workgroups(None, 4, 192), 192);
        assert_eq!(target_workgroups(Some(0), 4, 192), 192);
        assert_eq!(target_workgroups(Some(8), 4, 192), 64);
        assert_eq!(target_workgroups(Some(48), 4, 192), 192);
        assert_eq!(target_workgroups(Some(200), 4, 192), 512);
    }

    #[test]
    fn partition_policies_handle_boundaries_and_power_of_two() {
        // Zero groups and occupied grids stay unsplit.
        assert_eq!(TYPED_MATMUL_SPLITS.splits(0, 4096), 1);
        assert_eq!(TYPED_MATMUL_SPLITS.splits(128, 4096), 1);
        assert_eq!(TYPED_MATMUL_SPLITS.splits(8, 1023), 1);
        assert_eq!(TYPED_MATMUL_SPLITS.splits(8, 1024), 4);
        assert_eq!(TYPED_MATMUL_SPLITS.splits(8, 65_536), 16);

        // Scalar matmul splits below 64 groups and stops at 64.
        assert_eq!(SCALAR_MATMUL_SPLITS.multi_splits(16, 1024), Some(8));
        assert_eq!(SCALAR_MATMUL_SPLITS.multi_splits(64, 1024), None);
        assert_eq!(SCALAR_MATMUL_SPLITS.multi_splits(16, 200), None);

        // Native F32 tiled splits stay powers of two.
        assert_eq!(NATIVE_F32_TILED_SPLITS.max_splits(128), 1);
        assert_eq!(NATIVE_F32_TILED_SPLITS.max_splits(384), 2);
        assert_eq!(NATIVE_F32_TILED_SPLITS.max_splits(2048), 16);
        assert_eq!(NATIVE_F32_TILED_SPLITS.splits(16, 2048), 16);
        assert_eq!(NATIVE_F32_TILED_SPLITS.splits(64, 2048), 8);
        assert_eq!(NATIVE_F32_TILED_SPLITS.splits(96, 2048), 8);
        assert_eq!(NATIVE_F32_TILED_SPLITS.splits(512, 2048), 1);
        assert_eq!(NATIVE_F32_TILED_SPLITS.splits(4, 128), 1);

        // Gathered conv splits scale with parallel_width.
        let wide = gathered_conv_splits(Some(64));
        assert_eq!(wide.occupied_groups, 256);
        assert_eq!(wide.target_groups, 512);
        assert_eq!(wide.multi_splits(64, 64), Some(8));
        assert_eq!(wide.multi_splits(64, 31), None);
    }

    #[test]
    fn widest_occupied_tile_respects_threshold_and_fit() {
        let tiles = [(64u32, 64u32), (64, 32), (32, 32)];
        // 256x3136x1 -> 4*49 = 196 >= 128 at 64x64
        assert_eq!(
            widest_occupied_tile(&tiles, |t| t, 256, 3136, 1, MIN_F16_COOP_WORKGROUPS, true),
            (64, 64)
        );
        // 32x3136x1 -> 64x64 doesn't fit rows (require_fit=true), falls back to 32x32
        assert_eq!(
            widest_occupied_tile(&tiles, |t| t, 32, 3136, 1, MIN_F16_COOP_WORKGROUPS, true),
            (32, 32)
        );
    }
}
