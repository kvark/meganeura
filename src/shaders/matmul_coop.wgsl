// Cooperative matrix matmul: 2×2 tile grid (32×32 output per WG)
// Dispatch: [ceil(m/32), ceil(n/32), 1], WG=64

enable f16;
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
var<workgroup> shared_a0: array<f16, 256>;
var<workgroup> shared_a1: array<f16, 256>;
var<workgroup> shared_b0: array<f16, 256>;
var<workgroup> shared_b1: array<f16, 256>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_row = wgid.x * 32u;
    let tile_col = wgid.y * 32u;
    let m = params.m;
    let n = params.n;
    let k = params.k;

    // C offsets for the 4 output tiles
    let c00 = tile_row * n + tile_col;
    let c01 = tile_row * n + (tile_col + 16u);
    let c10 = (tile_row + 16u) * n + tile_col;
    let c11 = (tile_row + 16u) * n + (tile_col + 16u);

    // Validity flags for secondary tiles
    let n1_valid = (tile_col + 16u) < n;
    let m1_valid = (tile_row + 16u) < m;

    // Initialize accumulators
    $ACC_INIT

    // Hoisted staging index components
    let src_col = lid.x & 15u;
    let base_row = lid.x >> 4u;
    let cc = tile_col + src_col;
    let in_n = cc < n;
    let cc1 = cc + 16u;
    let in_n1 = cc1 < n;

    var t = 0u;
    loop {
        if t >= k { break; }

        // Stage sa0: B[t:t+16, tile_col:tile_col+16] → shared_a0
        let zero16 = f16(0.0);
        for (var e = 0u; e < 4u; e++) {
            let flat = lid.x + e * 64u;
            let tr = t + base_row + e * 4u;
            let in_bounds = (tr < k) && in_n;
            if in_bounds {
                shared_a0[flat] = f16(matrix_b[$B_INDEX_0]);
            } else {
                shared_a0[flat] = zero16;
            }
        }

        // Stage sa1: B[t:t+16, tile_col+16:tile_col+32] → shared_a1
        for (var e = 0u; e < 4u; e++) {
            let flat = lid.x + e * 64u;
            let tr = t + base_row + e * 4u;
            let in_bounds = (tr < k) && in_n1;
            if in_bounds {
                shared_a1[flat] = f16(matrix_b[$B_INDEX_1]);
            } else {
                shared_a1[flat] = zero16;
            }
        }

        // Stage sb0: A[tile_row:tile_row+16, t:t+16] → shared_b0
        let tc = t + src_col;
        let in_k = tc < k;
        for (var e = 0u; e < 4u; e++) {
            let flat = lid.x + e * 64u;
            let gr = tile_row + base_row + e * 4u;
            let in_bounds = (gr < m) && in_k;
            if in_bounds {
                shared_b0[flat] = f16(matrix_a[$A_INDEX_0]);
            } else {
                shared_b0[flat] = zero16;
            }
        }

        // Stage sb1: A[tile_row+16:tile_row+32, t:t+16] → shared_b1
        for (var e = 0u; e < 4u; e++) {
            let flat = lid.x + e * 64u;
            let gr = tile_row + 16u + base_row + e * 4u;
            let in_bounds = (gr < m) && in_k;
            if in_bounds {
                shared_b1[flat] = f16(matrix_a[$A_INDEX_1]);
            } else {
                shared_b1[flat] = zero16;
            }
        }

        workgroupBarrier();

        // Cooperative matrix multiply-add
        let a0 = coopLoad<coop_mat16x16<f16,A>>(&shared_a0[0], 16u);
        let a1 = coopLoad<coop_mat16x16<f16,A>>(&shared_a1[0], 16u);
        let b0 = coopLoad<coop_mat16x16<f16,B>>(&shared_b0[0], 16u);
        let b1 = coopLoad<coop_mat16x16<f16,B>>(&shared_b1[0], 16u);
        acc00 = coopMultiplyAdd(a0, b0, acc00);
        acc01 = coopMultiplyAdd(a1, b0, acc01);
        acc10 = coopMultiplyAdd(a0, b1, acc10);
        acc11 = coopMultiplyAdd(a1, b1, acc11);

        workgroupBarrier();
        t += 16u;
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
