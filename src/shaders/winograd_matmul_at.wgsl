// Batched matmul with A transposed for Winograd: C[z, N, M] = A^T[z, N, K] × B[z, K, M]
// equivalently C[z, n, m] = sum_k A[z, k, n] × B[z, k, m]
// where A is stored as [z, K, N] (row-major) but accessed with transposed indices.
// z = 16 alpha planes, dispatched via workgroup_id.z
// BM=64, BN=64, KTILE=16, TM=4, TN=4, workgroup [16,16,1]
// Dispatch: [ceil(M/64), ceil(N/64), 16]

struct Params {
    out_n: u32,     // output rows = N (e.g. in_channels)
    out_m: u32,     // output cols = M (e.g. total_tiles)
    k: u32,         // inner dimension K (e.g. out_channels)
    _pad: u32,
}

var<storage> matrix_a: array<f32>;           // [16, K, N] (transposed access)
var<storage> matrix_b: array<f32>;           // [16, K, M]
var<storage, read_write> matrix_c: array<f32>;  // [16, N, M]
var<uniform> params: Params;
var<workgroup> shared_a: array<f32, 1024>;   // [64, 16]
var<workgroup> shared_b: array<f32, 1024>;   // [16, 64]

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let z = wgid.z;                    // alpha plane index (0..15)
    let tile_row = wgid.y * 64u;       // N tile start
    let tile_col = wgid.x * 64u;       // M tile start
    let tid = ty * 16u + tx;

    let n = params.out_n;
    let m = params.out_m;
    let k = params.k;
    let plane_a = z * k * n;           // A is [K, N] per plane
    let plane_b = z * k * m;           // B is [K, M] per plane
    let plane_c = z * n * m;           // C is [N, M] per plane

    var s0_0 = 0.0; var s0_1 = 0.0; var s0_2 = 0.0; var s0_3 = 0.0;
    var s1_0 = 0.0; var s1_1 = 0.0; var s1_2 = 0.0; var s1_3 = 0.0;
    var s2_0 = 0.0; var s2_1 = 0.0; var s2_2 = 0.0; var s2_3 = 0.0;
    var s3_0 = 0.0; var s3_1 = 0.0; var s3_2 = 0.0; var s3_3 = 0.0;

    var t = 0u;
    loop {
        if t >= k { break; }

        // Load A^T tile: need A^T[n_idx, k_idx] = A[k_idx, n_idx]
        // shared_a[row_local, col_local] where row_local indexes N, col_local indexes K
        // A^T[row, col] = A[col, row] = matrix_a[plane_a + col * n + row]
        for (var e = 0u; e < 4u; e++) {
            let flat = tid + e * 256u;
            let row_local = flat / 16u;   // N index
            let col_local = flat % 16u;   // K index
            let a_row = tile_row + row_local;  // N index
            let a_col = t + col_local;         // K index
            let ib = a_row < n && a_col < k;
            // Transposed access: A[K, N] → A^T[N, K]
            shared_a[row_local * 16u + col_local] = select(0.0, matrix_a[plane_a + a_col * n + a_row], ib);
        }

        // Load B tile: B[z, K, M] → shared_b[16, 64]
        for (var e = 0u; e < 4u; e++) {
            let flat = tid + e * 256u;
            let row_local = flat / 64u;
            let col_local = flat % 64u;
            let b_row = t + row_local;
            let b_col = tile_col + col_local;
            let ib = b_row < k && b_col < m;
            shared_b[row_local * 64u + col_local] = select(0.0, matrix_b[plane_b + b_row * m + b_col], ib);
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
            if row < n && col < m {
                matrix_c[plane_c + row * m + col] = s[i][j];
            }
        }
    }
}
