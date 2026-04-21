// 4×4 register-tiled matmul: C = A × B (+ D if fused_add)
// BM=64, BN=64, KTILE=32, TM=4, TN=4, workgroup [16,16,1]
//
// Shared memory uses padded strides (A: 33, B: 65) to eliminate bank
// conflicts. With stride 33, consecutive A rows map to consecutive banks
// (33 mod 32 = 1), giving zero-conflict reads in the compute loop.
//
// Template variables: A_INDEX, B_INDEX (global index), A_ROW/COL,
// B_ROW/COL (tile mapping), FUSED_ADD_DECL/EXPR (addend), EPILOGUE_BODY

$ENABLE_F16
struct Params {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

var<storage> matrix_a: array<f32>;
var<storage> matrix_b: $B_STORAGE_TYPE;
var<storage, read_write> matrix_c: array<f32>;
$FUSED_ADD_DECL
var<uniform> params: Params;
var<workgroup> shared_a: array<f32, 2112>;  // 64 * 33 (padded stride)
var<workgroup> shared_b: array<f32, 2080>;  // 32 * 65 (padded stride)

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let tile_row = wgid.y * 64u;
    let tile_col = wgid.x * 64u;
    let tid = ty * 16u + tx;

    // 16 accumulator registers
    var s0_0 = 0.0; var s0_1 = 0.0; var s0_2 = 0.0; var s0_3 = 0.0;
    var s1_0 = 0.0; var s1_1 = 0.0; var s1_2 = 0.0; var s1_3 = 0.0;
    var s2_0 = 0.0; var s2_1 = 0.0; var s2_2 = 0.0; var s2_3 = 0.0;
    var s3_0 = 0.0; var s3_1 = 0.0; var s3_2 = 0.0; var s3_3 = 0.0;

    var t = 0u;
    loop {
        if t >= params.k { break; }

        // Load A tile into shared_a[64×32]: 8 elements per thread
        // shared_a layout: [m_local * 33 + k_local] (padded stride)
        for (var e = 0u; e < 8u; e++) {
            let flat = tid + e * 256u;
            let row_local = $A_ROW;
            let col_local = $A_COL;
            let a_row = tile_row + row_local;
            let a_col = t + col_local;
            let in_bounds = (a_row < params.m) && (a_col < params.k);
            shared_a[row_local * 33u + col_local] = select(0.0, matrix_a[$A_INDEX], in_bounds);
        }

        // Load B tile into shared_b[32×64]: 8 elements per thread
        // shared_b layout: [k_local * 65 + n_local] (padded stride)
        for (var e = 0u; e < 8u; e++) {
            let flat = tid + e * 256u;
            let row_local = $B_ROW;
            let col_local = $B_COL;
            let b_row = t + row_local;
            let b_col = tile_col + col_local;
            let in_bounds = (b_row < params.k) && (b_col < params.n);
            shared_b[row_local * 65u + col_local] = select(0.0, $B_CAST_OPEN matrix_b[$B_INDEX] $B_CAST_CLOSE, in_bounds);
        }

        workgroupBarrier();

        // Compute: unrolled over KTILE=32
        for (var kk = 0u; kk < 32u; kk++) {
            let a0 = shared_a[(ty * 4u + 0u) * 33u + kk];
            let a1 = shared_a[(ty * 4u + 1u) * 33u + kk];
            let a2 = shared_a[(ty * 4u + 2u) * 33u + kk];
            let a3 = shared_a[(ty * 4u + 3u) * 33u + kk];
            let b0 = shared_b[kk * 65u + tx * 4u + 0u];
            let b1 = shared_b[kk * 65u + tx * 4u + 1u];
            let b2 = shared_b[kk * 65u + tx * 4u + 2u];
            let b3 = shared_b[kk * 65u + tx * 4u + 3u];
            s0_0 += a0 * b0; s0_1 += a0 * b1; s0_2 += a0 * b2; s0_3 += a0 * b3;
            s1_0 += a1 * b0; s1_1 += a1 * b1; s1_2 += a1 * b2; s1_3 += a1 * b3;
            s2_0 += a2 * b0; s2_1 += a2 * b1; s2_2 += a2 * b2; s2_3 += a2 * b3;
            s3_0 += a3 * b0; s3_1 += a3 * b1; s3_2 += a3 * b2; s3_3 += a3 * b3;
        }

        workgroupBarrier();
        t += 32u;
    }

    // Store results with bounds check and optional fused epilogue
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
            if row < params.m && col < params.n {
                let idx = row * params.n + col;
                $STORE_BODY
            }
        }
    }
}
