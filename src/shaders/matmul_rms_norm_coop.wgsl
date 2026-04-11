// Fused RmsNorm + MatMul with cooperative matrix: C = RmsNorm(X, W_norm) × W_proj
//
// Same as matmul_coop.wgsl but applies RmsNorm during the A-matrix staging
// phase. The rsqrt is precomputed in a prologue, then each A element is
// normalized on-the-fly before being written to shared memory.
//
// Dispatch: [ceil(M/OUTPUT_TILE), ceil(N/OUTPUT_TILE), 1], WG=64
// Inputs: matrix_a = X [M, K], matrix_b = W_proj [K, N], bias = W_norm [K]
// Params: m, n, k, eps_bits

$ENABLE_F16
enable wgpu_cooperative_matrix;

struct Params {
    m: u32,
    n: u32,
    k: u32,
    eps_bits: u32,
}

var<storage> matrix_a: array<f32>;           // X [M, K]
var<storage> matrix_b: array<f32>;           // W_proj [K, N]
var<storage> bias: array<f32>;               // W_norm [K]
var<storage, read_write> matrix_c: array<f32>;
var<uniform> params: Params;
var<workgroup> shared_a0: array<$ELEM_TYPE, $SHARED_SIZE>;
var<workgroup> shared_a1: array<$ELEM_TYPE, $SHARED_SIZE>;
var<workgroup> shared_b0: array<$ELEM_TYPE, $SHARED_SIZE>;
var<workgroup> shared_b1: array<$ELEM_TYPE, $SHARED_SIZE>;
// rsqrt cache: one per M-tile row (OUTPUT_TILE rows max)
var<workgroup> rsqrt_cache: array<f32, $OUTPUT_TILE_U>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_row = wgid.x * $OUTPUT_TILE_U;
    let tile_col = wgid.y * $OUTPUT_TILE_U;
    let m = params.m;
    let n = params.n;
    let k = params.k;
    let eps = bitcast<f32>(params.eps_bits);

    // --- Prologue: precompute rsqrt for rows in this M-tile ---
    // Each of 64 threads handles one or more rows.
    for (var row_off = lid.x; row_off < $OUTPUT_TILE_U; row_off += 64u) {
        let row = tile_row + row_off;
        if row < m {
            var ss = 0.0;
            for (var j = 0u; j < k; j++) {
                let v = matrix_a[row * k + j];
                ss += v * v;
            }
            rsqrt_cache[row_off] = inverseSqrt(ss / f32(k) + eps);
        } else {
            rsqrt_cache[row_off] = 0.0;
        }
    }
    workgroupBarrier();

    // C offsets for the 4 output tiles
    let c00 = tile_row * n + tile_col;
    let c01 = tile_row * n + (tile_col + $TILE_SIZE_U);
    let c10 = (tile_row + $TILE_SIZE_U) * n + tile_col;
    let c11 = (tile_row + $TILE_SIZE_U) * n + (tile_col + $TILE_SIZE_U);

    let n1_valid = (tile_col + $TILE_SIZE_U) < n;
    let m1_valid = (tile_row + $TILE_SIZE_U) < m;

    // Initialize accumulators to zero
    var acc00 = coopConstructT<$COOP_OUT>(0.0);
    var acc01 = coopConstructT<$COOP_OUT>(0.0);
    var acc10 = coopConstructT<$COOP_OUT>(0.0);
    var acc11 = coopConstructT<$COOP_OUT>(0.0);

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

        // Stage B[t:t+tile, tile_col:tile_col+tile] → shared_a0 (unchanged)
        let zero_val = $ELEM_ZERO;
        for (var e = 0u; e < $STAGING_ITERS_U; e++) {
            let flat = lid.x + e * 64u;
            let tr = t + base_row + e * $ROW_STRIDE_U;
            let in_bounds = (tr < k) && in_n;
            if in_bounds {
                shared_a0[flat] = $CAST_OPEN matrix_b[tr * n + cc] $CAST_CLOSE;
            } else {
                shared_a0[flat] = zero_val;
            }
        }

        // Stage B[t:t+tile, tile_col+tile:tile_col+2*tile] → shared_a1
        for (var e = 0u; e < $STAGING_ITERS_U; e++) {
            let flat = lid.x + e * 64u;
            let tr = t + base_row + e * $ROW_STRIDE_U;
            let in_bounds = (tr < k) && in_n1;
            if in_bounds {
                shared_a1[flat] = $CAST_OPEN matrix_b[tr * n + cc1] $CAST_CLOSE;
            } else {
                shared_a1[flat] = zero_val;
            }
        }

        // Stage A[tile_row:tile_row+tile, t:t+tile] → shared_b0
        // WITH on-the-fly RmsNorm: A_norm[r,c] = A[r,c] * rsqrt[r] * W_norm[c]
        let tc = t + src_col;
        let in_k = tc < k;
        for (var e = 0u; e < $STAGING_ITERS_U; e++) {
            let flat = lid.x + e * 64u;
            let gr = tile_row + base_row + e * $ROW_STRIDE_U;
            let row_local = base_row + e * $ROW_STRIDE_U;
            let in_bounds = (gr < m) && in_k;
            if in_bounds {
                let raw = matrix_a[gr * k + tc];
                let norm = raw * rsqrt_cache[row_local] * bias[tc];
                shared_b0[flat] = $CAST_OPEN norm $CAST_CLOSE;
            } else {
                shared_b0[flat] = zero_val;
            }
        }

        // Stage A[tile_row+tile:tile_row+2*tile, t:t+tile] → shared_b1 (with RmsNorm)
        for (var e = 0u; e < $STAGING_ITERS_U; e++) {
            let flat = lid.x + e * 64u;
            let gr = tile_row + $TILE_SIZE_U + base_row + e * $ROW_STRIDE_U;
            let row_local = $TILE_SIZE_U + base_row + e * $ROW_STRIDE_U;
            let in_bounds = (gr < m) && in_k;
            if in_bounds {
                let raw = matrix_a[gr * k + tc];
                let norm = raw * rsqrt_cache[row_local] * bias[tc];
                shared_b1[flat] = $CAST_OPEN norm $CAST_CLOSE;
            } else {
                shared_b1[flat] = zero_val;
            }
        }

        workgroupBarrier();

        // Cooperative matrix multiply-add (same as matmul_coop.wgsl)
        let a0 = coopLoadT<$COOP_AB>(&shared_b0[0], $TILE_SIZE_U);
        let a1 = coopLoadT<$COOP_AB>(&shared_b1[0], $TILE_SIZE_U);
        let b0 = coopLoadT<$COOP_BA>(&shared_a0[0], $TILE_SIZE_U);
        let b1 = coopLoadT<$COOP_BA>(&shared_a1[0], $TILE_SIZE_U);
        acc00 = coopMultiplyAdd(a0, b0, acc00);
        acc01 = coopMultiplyAdd(a0, b1, acc01);
        acc10 = coopMultiplyAdd(a1, b0, acc10);
        acc11 = coopMultiplyAdd(a1, b1, acc11);

        workgroupBarrier();
        t += $TILE_SIZE_U;
    }

    // Store results
    coopStoreT(acc00, &matrix_c[c00], n);
    if n1_valid {
        coopStoreT(acc01, &matrix_c[c01], n);
    }
    if m1_valid {
        coopStoreT(acc10, &matrix_c[c10], n);
    }
    if n1_valid && m1_valid {
        coopStoreT(acc11, &matrix_c[c11], n);
    }
}
