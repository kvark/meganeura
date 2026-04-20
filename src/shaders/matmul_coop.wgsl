// Cooperative matrix matmul: 2×2 tile grid ($OUTPUT_TILE×$OUTPUT_TILE output per WG)
// Dispatch: [ceil(m/$OUTPUT_TILE), ceil(n/$OUTPUT_TILE), 1], WG=64
// Parameterized by tile size ($TILE_SIZE) and element type ($ELEM_TYPE).
// - 16×16 f16 path: RDNA3/Volta+ (VK_KHR_cooperative_matrix)
// -  8×8 f32 path:  Apple Silicon (simdgroup_matrix)

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
var<uniform> params: Params;
var<workgroup> shared_a0: array<$ELEM_TYPE, $SHARED_SIZE>;
var<workgroup> shared_a1: array<$ELEM_TYPE, $SHARED_SIZE>;
var<workgroup> shared_b0: array<$ELEM_TYPE, $SHARED_SIZE>;
var<workgroup> shared_b1: array<$ELEM_TYPE, $SHARED_SIZE>;
$PROLOGUE_CACHE_DECL

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_row = wgid.x * $OUTPUT_TILE_U;
    let tile_col = wgid.y * $OUTPUT_TILE_U;
    let m = params.m;
    let n = params.n;
    let k = params.k;

    // C offsets for the 4 output tiles
    let c00 = tile_row * n + tile_col;
    let c01 = tile_row * n + (tile_col + $TILE_SIZE_U);
    let c10 = (tile_row + $TILE_SIZE_U) * n + tile_col;
    let c11 = (tile_row + $TILE_SIZE_U) * n + (tile_col + $TILE_SIZE_U);

    // Validity flags for secondary tiles
    let n1_valid = (tile_col + $TILE_SIZE_U) < n;
    let m1_valid = (tile_row + $TILE_SIZE_U) < m;

    // Initialize accumulators
    $ACC_INIT

    // Hoisted staging index components
    $STAGING_VARS

    // Prologue cache init (e.g. per-row rsqrt). Loaded once per workgroup
    // from global into shared memory so the K-loop staging reads are cheap.
    $PROLOGUE_CACHE_INIT

    var t = 0u;
    loop {
        if t >= k { break; }

        // Stage sa0: B[t:t+tile, tile_col:tile_col+tile] → shared_a0
        $B_STAGE_0

        // Stage sa1: B[t:t+tile, tile_col+tile:tile_col+2*tile] → shared_a1
        $B_STAGE_1

        // Stage sb0: A[tile_row:tile_row+tile, t:t+tile] → shared_b0
        $A_STAGE_0

        // Stage sb1: A[tile_row+tile:tile_row+2*tile, t:t+tile] → shared_b1
        $A_STAGE_1

        workgroupBarrier();

        // Cooperative matrix multiply-add: C += A × B
        // shared_b{0,1} hold A-matrix row tiles; shared_a{0,1} hold B-matrix column tiles.
        // Load A data into role-A (left operand), B data into role-B (right operand).
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
