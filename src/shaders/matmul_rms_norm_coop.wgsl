// Fused RmsNorm + MatMul with cooperative matrix: C = RmsNorm(X, W_norm) × W_proj
//
// Same as matmul_coop.wgsl but applies RmsNorm during the A-matrix staging
// phase. The rsqrt is precomputed in a prologue, then each A element is
// normalized on-the-fly before being written to shared memory.
//
// Dispatch: [ceil(M/OUTPUT_TILE), ceil(N/OUTPUT_TILE), 1], WG=64
// Inputs: src_a = X [M, K], src_b = W_proj [K, N], bias = W_norm [K]
// Params: m, n, k, eps_bits

$ENABLE_F16
enable wgpu_cooperative_matrix;

struct Params {
    m: u32,
    n: u32,
    k: u32,
    eps_bits: u32,
}

var<storage> src_a: array<f32>;              // X [M, K]
var<storage> src_b: array<f32>;             // W_proj [K, N]
var<storage> bias: array<f32>;               // W_norm [K]
var<storage, read_write> dst: array<f32>;
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

    // --- Prologue: precompute rsqrt using 64-thread cooperative reduction ---
    // All 64 threads cooperate on each row: each sums a stride, then
    // tree-reduce via shared_b0 (reused as temp). 6 barriers per row.
    for (var row_off = 0u; row_off < $OUTPUT_TILE_U; row_off++) {
        let row = tile_row + row_off;
        var ss = 0.0;
        if row < m {
            var j = lid.x;
            loop {
                if j >= k { break; }
                let v = src_a[row * k + j];
                ss += v * v;
                j += 64u;
            }
        }
        // Tree reduction across 64 threads using shared_b0 as temp
        shared_b0[lid.x] = $CAST_OPEN ss $CAST_CLOSE;
        workgroupBarrier();
        if lid.x < 32u { shared_b0[lid.x] = $CAST_OPEN f32(shared_b0[lid.x]) + f32(shared_b0[lid.x + 32u]) $CAST_CLOSE; }
        workgroupBarrier();
        if lid.x < 16u { shared_b0[lid.x] = $CAST_OPEN f32(shared_b0[lid.x]) + f32(shared_b0[lid.x + 16u]) $CAST_CLOSE; }
        workgroupBarrier();
        if lid.x < 8u { shared_b0[lid.x] = $CAST_OPEN f32(shared_b0[lid.x]) + f32(shared_b0[lid.x + 8u]) $CAST_CLOSE; }
        workgroupBarrier();
        if lid.x < 4u { shared_b0[lid.x] = $CAST_OPEN f32(shared_b0[lid.x]) + f32(shared_b0[lid.x + 4u]) $CAST_CLOSE; }
        workgroupBarrier();
        if lid.x < 2u { shared_b0[lid.x] = $CAST_OPEN f32(shared_b0[lid.x]) + f32(shared_b0[lid.x + 2u]) $CAST_CLOSE; }
        workgroupBarrier();
        if lid.x == 0u {
            let total = f32(shared_b0[0u]) + f32(shared_b0[1u]);
            if row < m {
                rsqrt_cache[row_off] = inverseSqrt(total / f32(k) + eps);
            } else {
                rsqrt_cache[row_off] = 0.0;
            }
        }
        workgroupBarrier();
    }

    // C offsets for the 4 output tiles
    let c00 = tile_row * n + tile_col;
    let c01 = tile_row * n + (tile_col + $TILE_SIZE_U);
    let c10 = (tile_row + $TILE_SIZE_U) * n + tile_col;
    let c11 = (tile_row + $TILE_SIZE_U) * n + (tile_col + $TILE_SIZE_U);

    let n1_valid = (tile_col + $TILE_SIZE_U) < n;
    let m1_valid = (tile_row + $TILE_SIZE_U) < m;

    // Initialize accumulators to zero
    var acc00 = $COOP_OUT();
    var acc01 = $COOP_OUT();
    var acc10 = $COOP_OUT();
    var acc11 = $COOP_OUT();

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
                shared_a0[flat] = $CAST_OPEN src_b[tr * n + cc] $CAST_CLOSE;
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
                shared_a1[flat] = $CAST_OPEN src_b[tr * n + cc1] $CAST_CLOSE;
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
                let raw = src_a[gr * k + tc];
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
                let raw = src_a[gr * k + tc];
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
    coopStoreT(acc00, &dst[c00], n);
    if n1_valid {
        coopStoreT(acc01, &dst[c01], n);
    }
    if m1_valid {
        coopStoreT(acc10, &dst[c10], n);
    }
    if n1_valid && m1_valid {
        coopStoreT(acc11, &dst[c11], n);
    }
}
