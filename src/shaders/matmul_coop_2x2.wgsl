// Cooperative matrix matmul: 2×2 hardware tiles per WG ($OUTPUT_TILE = 2×$TILE_SIZE).
// Dispatch: [ceil(m/$OUTPUT_TILE), ceil(n/$OUTPUT_TILE), 1], WG=64.
// Used when that grid still has enough workgroups (wide FFN / lm_head).
// Skinny prefill projections (128×576) stay on the 1×1 shader.

$ENABLE_F16
enable wgpu_cooperative_matrix;

struct Params {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

var<storage> matrix_a: $A_STORAGE;
var<storage> matrix_b: $B_STORAGE;
var<storage, read_write> matrix_c: array<f32>;
$FUSED_ADD_DECL
$PROLOGUE_DECL
$EPILOGUE_DECL
var<uniform> params: Params;
var<workgroup> shared_a0: array<$ELEM_TYPE, $SHARED_SIZE>;
var<workgroup> shared_a1: array<$ELEM_TYPE, $SHARED_SIZE>;
var<workgroup> shared_b0: array<$ELEM_TYPE, $SHARED_SIZE>;
var<workgroup> shared_b1: array<$ELEM_TYPE, $SHARED_SIZE>;
$SHARED_LO_DECL
$RESULT_SHARED_DECL
$PROLOGUE_CACHE_DECL

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_row = wgid.x * $OUTPUT_TILE_U;
    let tile_col = wgid.y * $OUTPUT_TILE_U;
    let m = params.m;
    let n = params.n;
    let k = params.k;

    let c00 = tile_row * n + tile_col;
    let c01 = tile_row * n + (tile_col + $TILE_SIZE_U);
    let c10 = (tile_row + $TILE_SIZE_U) * n + tile_col;
    let c11 = (tile_row + $TILE_SIZE_U) * n + (tile_col + $TILE_SIZE_U);

    let n1_valid = (tile_col + $TILE_SIZE_U) < n;
    let m1_valid = (tile_row + $TILE_SIZE_U) < m;

    $ACC_INIT

    $STAGING_VARS

    $PROLOGUE_CACHE_INIT

    var t = 0u;
    loop {
        if t >= k { break; }

        $B_STAGE_0
        $B_STAGE_1
        $A_STAGE_0
        $A_STAGE_1

        workgroupBarrier();

        let a0 = coopLoadT<$COOP_AB>(&shared_b0[0], $TILE_SIZE_U);
        let a1 = coopLoadT<$COOP_AB>(&shared_b1[0], $TILE_SIZE_U);
        let b0 = coopLoadT<$COOP_BA>(&shared_a0[0], $TILE_SIZE_U);
        let b1 = coopLoadT<$COOP_BA>(&shared_a1[0], $TILE_SIZE_U);
        acc00 = coopMultiplyAdd(a0, b0, acc00);
        acc01 = coopMultiplyAdd(a0, b1, acc01);
        acc10 = coopMultiplyAdd(a1, b0, acc10);
        acc11 = coopMultiplyAdd(a1, b1, acc11);
        $COMPENSATED_MMA

        workgroupBarrier();
        t += $TILE_SIZE_U;
    }

    $RESULT_STORE
}
