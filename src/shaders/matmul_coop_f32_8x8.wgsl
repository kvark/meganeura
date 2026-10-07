// @section header
// Four 32-lane SIMD groups own separate 16x16 output quadrants.
// Cooperative 8x8 f32 operands and accumulators, with 32-deep K staging.
enable wgpu_cooperative_matrix;

struct Params {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

var<storage> matrix_a: array<f32>;
$MATRIX_BINDINGS
$FUSED_ADD_DECL
$PROLOGUE_DECL
$EPILOGUE_DECL
var<uniform> params: Params;
var<workgroup> shared_a: array<f32, 1024>;
var<workgroup> shared_b: array<f32, 1024>;
$PROLOGUE_CACHE_DECL
$RESULT_SHARED_DECL

// @section kernel
$ENTRY_SIGNATURE {
    let tile_row = wgid.x * 32u;
    let tile_col = wgid.y * 32u;
    let m = params.m;
    let n = params.n;
    let k = params.k;
    $PROLOGUE_CACHE_INIT

    // Usually one iteration (four 32-lane groups). Keep the geometry
    // correct on devices choosing another subgroup width: every invocation
    // stages inputs and reaches every workgroup barrier, including groups
    // without an output quadrant. Wider groups process multiple rounds.
    for (var group_base = 0u; group_base < 4u; group_base += 128u / sg_size) {
        let tile_sg = group_base + sg;
        // Extra groups on a narrower-subgroup device read a valid quadrant
        // but do not store it. Cooperative arithmetic stays in uniform flow.
        let sg_row = ((tile_sg % 4u) / 2u) * 16u;
        let sg_col = (tile_sg % 2u) * 16u;
        $ACC_INIT

        for (var t = 0u; t < k; t += 32u) {
            $A_STAGE
            $B_STAGE
            workgroupBarrier();

            for (var kk = 0u; kk < 32u; kk += 8u) {
                let a0 = coopLoadT<coop_mat8x8<f32,A>>(&shared_a[sg_row * 32u + kk], 32u);
                let a1 = coopLoadT<coop_mat8x8<f32,A>>(&shared_a[(sg_row + 8u) * 32u + kk], 32u);
                let b0 = coopLoadT<coop_mat8x8<f32,B>>(&shared_b[kk * 32u + sg_col], 32u);
                let b1 = coopLoadT<coop_mat8x8<f32,B>>(&shared_b[kk * 32u + sg_col + 8u], 32u);
                acc00 = coopMultiplyAdd(a0, b0, acc00);
                acc01 = coopMultiplyAdd(a0, b1, acc01);
                acc10 = coopMultiplyAdd(a1, b0, acc10);
                acc11 = coopMultiplyAdd(a1, b1, acc11);
            }
            workgroupBarrier();
        }

        $RESULT_STORE
    }
}
