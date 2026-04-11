// 2×2 register-tiled matmul: C = A × B (+ D if fused_add)
// BM=32, BN=32, KTILE=32, TM=2, TN=2, workgroup [16,16,1]
//
// Smaller tiles than matmul.wgsl (64×64). Produces 4× more workgroups
// for the same output, improving GPU occupancy on small matrices.
// Used when ceil(M/64)*ceil(N/64) < threshold (low workgroup count).
//
// Shared memory uses padded stride 33 (33 mod 32 = 1) so that
// consecutive rows map to consecutive banks → zero bank conflicts.
//
// Template variables: same as matmul.wgsl

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
var<workgroup> shared_a: array<f32, 1056>;  // 32 * 33 (padded stride)
var<workgroup> shared_b: array<f32, 1056>;  // 32 * 33 (padded stride)

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let tile_row = wgid.y * 32u;
    let tile_col = wgid.x * 32u;
    let tid = ty * 16u + tx;

    // 4 accumulator registers (2×2 per thread)
    var s0_0 = 0.0; var s0_1 = 0.0;
    var s1_0 = 0.0; var s1_1 = 0.0;

    var t = 0u;
    loop {
        if t >= params.k { break; }

        // Load A tile into shared_a[32×32]: 4 elements per thread
        // shared_a layout: [m_local * 33 + k_local] (padded stride)
        for (var e = 0u; e < 4u; e++) {
            let flat = tid + e * 256u;
            let row_local = $A_ROW_S;
            let col_local = $A_COL_S;
            let a_row = tile_row + row_local;
            let a_col = t + col_local;
            let in_bounds = (a_row < params.m) && (a_col < params.k);
            shared_a[row_local * 33u + col_local] = select(0.0, matrix_a[$A_INDEX], in_bounds);
        }

        // Load B tile into shared_b[32×32]: 4 elements per thread
        // shared_b layout: [k_local * 33 + n_local] (padded stride)
        for (var e = 0u; e < 4u; e++) {
            let flat = tid + e * 256u;
            let row_local = $B_ROW_S;
            let col_local = $B_COL_S;
            let b_row = t + row_local;
            let b_col = tile_col + col_local;
            let in_bounds = (b_row < params.k) && (b_col < params.n);
            shared_b[row_local * 33u + col_local] = select(0.0, matrix_b[$B_INDEX], in_bounds);
        }

        workgroupBarrier();

        // Compute: unrolled over KTILE=32
        for (var kk = 0u; kk < 32u; kk++) {
            let a0 = shared_a[(ty * 2u + 0u) * 33u + kk];
            let a1 = shared_a[(ty * 2u + 1u) * 33u + kk];
            let b0 = shared_b[kk * 33u + tx * 2u + 0u];
            let b1 = shared_b[kk * 33u + tx * 2u + 1u];
            s0_0 += a0 * b0; s0_1 += a0 * b1;
            s1_0 += a1 * b0; s1_1 += a1 * b1;
        }

        workgroupBarrier();
        t += 32u;
    }

    // Store results with bounds check
    let s = array<array<f32, 2>, 2>(
        array<f32, 2>(s0_0, s0_1),
        array<f32, 2>(s1_0, s1_1),
    );
    for (var i = 0u; i < 2u; i++) {
        for (var j = 0u; j < 2u; j++) {
            let row = tile_row + ty * 2u + i;
            let col = tile_col + tx * 2u + j;
            if row < params.m && col < params.n {
                let idx = row * params.n + col;
                $STORE_BODY
            }
        }
    }
}
