// Cooperative matrix matmul: 2×2 tile grid ($OUTPUT_TILE×$OUTPUT_TILE output per WG)
// Dispatch: [ceil(m/$OUTPUT_TILE), ceil(n/$OUTPUT_TILE), 1], WG=64
// Parameterized by tile size ($TILE_SIZE) and element type ($ELEM_TYPE).
// - 16×16 f16 path: RDNA3/Volta+ (VK_KHR_cooperative_matrix)
// -  8×8 f32 path:  Apple Silicon (simdgroup_matrix)

$ENABLE_F16
enable wgpu_cooperative_matrix;

struct Params {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

var<storage> matrix_a: array<f32>;
var<storage> matrix_b: array<f32>;
var<storage, read_write> matrix_c: array<f32>;
$FUSED_ADD_DECL
var<uniform> params: Params;
var<workgroup> shared_a0: array<$ELEM_TYPE, $SHARED_SIZE>;
var<workgroup> shared_a1: array<$ELEM_TYPE, $SHARED_SIZE>;
var<workgroup> shared_b0: array<$ELEM_TYPE, $SHARED_SIZE>;
var<workgroup> shared_b1: array<$ELEM_TYPE, $SHARED_SIZE>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_row = wgid.x * $OUTPUT_TILE_U;
    let tile_col = wgid.y * $OUTPUT_TILE_U;
    let m = params.m;
    let n = params.n;
    let k = params.k;

    // C offsets for the 4 output tiles
    let c00 = tile_row * n + tile_col;
    let c01 = tile_row * n + (tile_col + $TILE_SIZE_U);
    let c10 = (tile_row + $TILE_SIZE_U) * n + tile_col;
    let c11 = (tile_row + $TILE_SIZE_U) * n + (tile_col + $TILE_SIZE_U);

    // Validity flags for secondary tiles
    let n1_valid = (tile_col + $TILE_SIZE_U) < n;
    let m1_valid = (tile_row + $TILE_SIZE_U) < m;

    // Initialize accumulators
    $ACC_INIT

    // Hoisted staging index components
    let src_col = lid.x & $TILE_MASK_U;
    let base_row = lid.x >> $TILE_SHIFT_U;
    let cc = tile_col + src_col;
    let in_n = cc < n;
    let cc1 = cc + $TILE_SIZE_U;
    let in_n1 = cc1 < n;

    var t = 0u;
    loop {
        if t >= k { break; }

        // Stage sa0: B[t:t+tile, tile_col:tile_col+tile] → shared_a0
        let zero_val = $ELEM_ZERO;
        for (var e = 0u; e < $STAGING_ITERS_U; e++) {
            let flat = lid.x + e * 64u;
            let tr = t + base_row + e * $ROW_STRIDE_U;
            let in_bounds = (tr < k) && in_n;
            if in_bounds {
                shared_a0[flat] = $CAST_OPEN matrix_b[$B_INDEX_0] $CAST_CLOSE;
            } else {
                shared_a0[flat] = zero_val;
            }
        }

        // Stage sa1: B[t:t+tile, tile_col+tile:tile_col+2*tile] → shared_a1
        for (var e = 0u; e < $STAGING_ITERS_U; e++) {
            let flat = lid.x + e * 64u;
            let tr = t + base_row + e * $ROW_STRIDE_U;
            let in_bounds = (tr < k) && in_n1;
            if in_bounds {
                shared_a1[flat] = $CAST_OPEN matrix_b[$B_INDEX_1] $CAST_CLOSE;
            } else {
                shared_a1[flat] = zero_val;
            }
        }

        // Stage sb0: A[tile_row:tile_row+tile, t:t+tile] → shared_b0
        let tc = t + src_col;
        let in_k = tc < k;
        for (var e = 0u; e < $STAGING_ITERS_U; e++) {
            let flat = lid.x + e * 64u;
            let gr = tile_row + base_row + e * $ROW_STRIDE_U;
            let in_bounds = (gr < m) && in_k;
            if in_bounds {
                shared_b0[flat] = $CAST_OPEN matrix_a[$A_INDEX_0] $CAST_CLOSE;
            } else {
                shared_b0[flat] = zero_val;
            }
        }

        // Stage sb1: A[tile_row+tile:tile_row+2*tile, t:t+tile] → shared_b1
        for (var e = 0u; e < $STAGING_ITERS_U; e++) {
            let flat = lid.x + e * 64u;
            let gr = tile_row + $TILE_SIZE_U + base_row + e * $ROW_STRIDE_U;
            let in_bounds = (gr < m) && in_k;
            if in_bounds {
                shared_b1[flat] = $CAST_OPEN matrix_a[$A_INDEX_1] $CAST_CLOSE;
            } else {
                shared_b1[flat] = zero_val;
            }
        }

        workgroupBarrier();

        // Cooperative matrix multiply-add: C += A × B
        // shared_b{0,1} hold A-matrix row tiles; shared_a{0,1} hold B-matrix column tiles.
        // Load A data into role-A (left operand), B data into role-B (right operand).
        let a0 = coopLoad<$COOP_AB>(&shared_b0[0], $TILE_SIZE_U);
        let a1 = coopLoad<$COOP_AB>(&shared_b1[0], $TILE_SIZE_U);
        let b0 = coopLoad<$COOP_BA>(&shared_a0[0], $TILE_SIZE_U);
        let b1 = coopLoad<$COOP_BA>(&shared_a1[0], $TILE_SIZE_U);
        acc00 = coopMultiplyAdd(a0, b0, acc00);
        acc01 = coopMultiplyAdd(a0, b1, acc01);
        acc10 = coopMultiplyAdd(a1, b0, acc10);
        acc11 = coopMultiplyAdd(a1, b1, acc11);

        workgroupBarrier();
        t += $TILE_SIZE_U;
    }

    // Store results
    coopStore(acc00, &matrix_c[c00], n);
    if n1_valid {
        coopStore(acc01, &matrix_c[c01], n);
    }
    if m1_valid {
        coopStore(acc10, &matrix_c[c10], n);
    }
    if n1_valid && m1_valid {
        coopStore(acc11, &matrix_c[c11], n);
    }
}
