// 4×4 register-tiled matmul: C = A × B (+ D if fused_add)
// BM=64, BN=64, KTILE=16, TM=4, TN=4, workgroup [16,16,1]
//
// Template variables:
//   $A_INDEX, $B_INDEX: global memory index expressions
//   $A_ROW, $A_COL: thread-to-tile mapping for A load (row=M, col=K)
//   $B_ROW, $B_COL: thread-to-tile mapping for B load (row=K, col=N)
//   $FUSED_ADD_DECL, $FUSED_ADD_EXPR: optional addend for FusedMatMulAdd

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
var<workgroup> shared_a: array<f32, 1024>;
var<workgroup> shared_b: array<f32, 1024>;

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

        // Load A tile into shared_a[64*16]: 4 elements per thread
        // shared_a layout: [m_local * 16 + k_local]
        for (var e = 0u; e < 4u; e++) {
            let flat = tid + e * 256u;
            let row_local = $A_ROW;
            let col_local = $A_COL;
            let a_row = tile_row + row_local;
            let a_col = t + col_local;
            let in_bounds = (a_row < params.m) && (a_col < params.k);
            shared_a[row_local * 16u + col_local] = select(0.0, matrix_a[$A_INDEX], in_bounds);
        }

        // Load B tile into shared_b[16*64]: 4 elements per thread
        // shared_b layout: [k_local * 64 + n_local]
        for (var e = 0u; e < 4u; e++) {
            let flat = tid + e * 256u;
            let row_local = $B_ROW;
            let col_local = $B_COL;
            let b_row = t + row_local;
            let b_col = tile_col + col_local;
            let in_bounds = (b_row < params.k) && (b_col < params.n);
            shared_b[row_local * 64u + col_local] = select(0.0, matrix_b[$B_INDEX], in_bounds);
        }

        workgroupBarrier();

        // Compute: unrolled over KTILE=16
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

    // Store results with bounds check
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
                matrix_c[idx] = s[i][j]$FUSED_ADD_EXPR;
            }
        }
    }
}
