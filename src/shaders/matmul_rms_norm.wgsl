// Fused RmsNorm + MatMul: C = RmsNorm(X, W_norm) × W_proj
//
// Eliminates the intermediate normalized tensor by computing
// normalization on-the-fly during the matmul A-tile load phase.
//
// BM=64, BN=64, KTILE=16, TM=4, TN=4, workgroup [16,16,1]
// Dispatch: [ceil(N/64), ceil(M/64), 1]
//
// Inputs:
//   matrix_a = X [M, K]           (raw, unnormalized)
//   matrix_b = W_proj [K, N]      (projection weight)
//   bias     = W_norm [K]         (RMS norm scale weight)
// Output:
//   matrix_c = C [M, N]
// Params:
//   m, n, k, eps_bits (eps as bitcast f32)

struct Params {
    m: u32,
    n: u32,
    k: u32,
    eps_bits: u32,
}

var<storage> src_a: array<f32>;           // X [M, K]
var<storage> src_b: array<f32>;          // W_proj [K, N]
var<storage> bias: array<f32>;           // W_norm [K]
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;
var<workgroup> shared_a: array<f32, 1024>;
var<workgroup> shared_b: array<f32, 1024>;
var<workgroup> rsqrt_cache: array<f32, 64>;  // one per M-tile row

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let tile_row = wgid.y * 64u;
    let tile_col = wgid.x * 64u;
    let tid = ty * 16u + tx;
    let m = params.m;
    let n = params.n;
    let k = params.k;
    let eps = bitcast<f32>(params.eps_bits);

    // --- Prologue: precompute rsqrt for rows in this M-tile ---
    // Each of the 256 threads handles one or more rows.
    // For each row: sum x² over all K columns, compute rsqrt.
    for (var row_off = tid; row_off < 64u; row_off += 256u) {
        let row = tile_row + row_off;
        if row < m {
            var ss = 0.0;
            for (var j = 0u; j < k; j++) {
                let v = src_a[row * k + j];
                ss += v * v;
            }
            rsqrt_cache[row_off] = inverseSqrt(ss / f32(k) + eps);
        } else {
            rsqrt_cache[row_off] = 0.0;
        }
    }
    workgroupBarrier();

    // --- Main matmul with fused normalization ---
    var s0_0 = 0.0; var s0_1 = 0.0; var s0_2 = 0.0; var s0_3 = 0.0;
    var s1_0 = 0.0; var s1_1 = 0.0; var s1_2 = 0.0; var s1_3 = 0.0;
    var s2_0 = 0.0; var s2_1 = 0.0; var s2_2 = 0.0; var s2_3 = 0.0;
    var s3_0 = 0.0; var s3_1 = 0.0; var s3_2 = 0.0; var s3_3 = 0.0;

    var t = 0u;
    loop {
        if t >= k { break; }

        // Load A tile: apply RmsNorm on-the-fly
        // A_fused[i,j] = X[i,j] * rsqrt[i] * W_norm[j]
        for (var e = 0u; e < 4u; e++) {
            let flat = tid + e * 256u;
            let row_local = flat / 16u;
            let col_local = flat % 16u;
            let a_row = tile_row + row_local;
            let a_col = t + col_local;
            if a_row < m && a_col < k {
                let raw = src_a[a_row * k + a_col];
                let norm = raw * rsqrt_cache[row_local] * bias[a_col];
                shared_a[row_local * 16u + col_local] = norm;
            } else {
                shared_a[row_local * 16u + col_local] = 0.0;
            }
        }

        // Load B tile (standard, no transformation)
        for (var e = 0u; e < 4u; e++) {
            let flat = tid + e * 256u;
            let row_local = flat / 64u;
            let col_local = flat % 64u;
            let b_row = t + row_local;
            let b_col = tile_col + col_local;
            if b_row < k && b_col < n {
                shared_b[row_local * 64u + col_local] = src_b[b_row * n + b_col];
            } else {
                shared_b[row_local * 64u + col_local] = 0.0;
            }
        }

        workgroupBarrier();

        for (var kk = 0u; kk < 16u; kk++) {
            let a0 = shared_a[(ty * 4u + 0u) * 16u + kk];
            let a1 = shared_a[(ty * 4u + 1u) * 16u + kk];
            let a2 = shared_a[(ty * 4u + 2u) * 16u + kk];
            let a3 = shared_a[(ty * 4u + 3u) * 16u + kk];
            let b0 = shared_b[kk * 64u + tx * 4u + 0u];
            let b1 = shared_b[kk * 64u + tx * 4u + 1u];
            let b2 = shared_b[kk * 64u + tx * 4u + 2u];
            let b3 = shared_b[kk * 64u + tx * 4u + 3u];
            s0_0 += a0 * b0; s0_1 += a0 * b1; s0_2 += a0 * b2; s0_3 += a0 * b3;
            s1_0 += a1 * b0; s1_1 += a1 * b1; s1_2 += a1 * b2; s1_3 += a1 * b3;
            s2_0 += a2 * b0; s2_1 += a2 * b1; s2_2 += a2 * b2; s2_3 += a2 * b3;
            s3_0 += a3 * b0; s3_1 += a3 * b1; s3_2 += a3 * b2; s3_3 += a3 * b3;
        }

        workgroupBarrier();
        t += 16u;
    }

    // Store
    let s = array<array<f32, 4>, 4>(
        array<f32, 4>(s0_0, s0_1, s0_2, s0_3),
        array<f32, 4>(s1_0, s1_1, s1_2, s1_3),
        array<f32, 4>(s2_0, s2_1, s2_2, s2_3),
        array<f32, 4>(s3_0, s3_1, s3_2, s3_3),
    );
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            let row = tile_row + ty * 4u + i;
            let col = tile_col + tx * 4u + j;
            if row < m && col < n {
                dst[row * n + col] = s[i][j];
            }
        }
    }
}
