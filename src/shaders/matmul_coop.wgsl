// @section header
// Cooperative matrix matmul: 2×2 tile grid ($OUTPUT_M×$OUTPUT_N output per WG)
// Dispatch: [ceil(m/$OUTPUT_M), ceil(n/$OUTPUT_N), 1], WG=64
// One template for every tile. $TILE_M, $TILE_N and $TILE_K are the C rows,
// C columns and shared K of one hardware tile. Square tiles set them equal.
// - 16×16 f16 path: RDNA3/Volta+ (VK_KHR_cooperative_matrix)
// -  8×8 f32 path:  Apple Silicon (simdgroup_matrix)

$ENABLE_F16
enable wgpu_cooperative_matrix;

struct Params {
    m: u32,
    n: u32,
    k: u32,
    $PARTITION_FIELD: u32,
}

var<storage> matrix_a: $A_STORAGE;
$MATRIX_BINDINGS
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

// @section kernel
$ENTRY_SIGNATURE {
    let tile_row = wgid.x * $OUTPUT_M_U;
    let tile_col = wgid.y * $OUTPUT_N_U;
    let m = params.m;
    let n = params.n;
    let k = params.k;
    $K_RANGE

    let n1_valid = (tile_col + $TILE_N_U) < n;

    // Hoisted staging index components
    $STAGING_VARS

    // Prologue cache init (e.g. per-row rsqrt). Loaded once per workgroup
    // from global into shared memory so the K-loop staging reads are cheap.
    $PROLOGUE_CACHE_INIT

    // Two subgroups own separate rows. A single wide subgroup keeps both
    // rows, without putting staging or workgroup barriers in subgroup control.
    let sg_row = (sg % 2u) * $TILE_M_U;
    let row_valid = sg < 2u && tile_row + sg_row < m;
    let c0 = $OUTPUT_OFFSET + (tile_row + sg_row) * n + tile_col;
    let c1 = c0 + $TILE_N_U;
    let owns_second_row = sg == 0u && sg_size >= 64u;
    let second_row_valid = owns_second_row && tile_row + $TILE_M_U < m;
    let c2 = $OUTPUT_OFFSET + (tile_row + $TILE_M_U) * n + tile_col;
    let c3 = c2 + $TILE_N_U;
    $ACC_INIT

    var t = $K_START;
    loop {
        if t >= $K_END { break; }

        $B_STAGE_0
        $B_STAGE_1
        $A_STAGE_0
        $A_STAGE_1
        workgroupBarrier();

        var a = $COOP_AB();
        if sg_row == 0u {
            a = coopLoadT<$COOP_AB>(&shared_b0[0], $TILE_K_U);
        } else {
            a = coopLoadT<$COOP_AB>(&shared_b1[0], $TILE_K_U);
        }
        let b0 = coopLoadT<$COOP_BA>(&shared_a0[0], $TILE_N_U);
        let b1 = coopLoadT<$COOP_BA>(&shared_a1[0], $TILE_N_U);
        acc0 = coopMultiplyAdd(a, b0, acc0);
        acc1 = coopMultiplyAdd(a, b1, acc1);
        if owns_second_row {
            let a_second = coopLoadT<$COOP_AB>(&shared_b1[0], $TILE_K_U);
            acc2 = coopMultiplyAdd(a_second, b0, acc2);
            acc3 = coopMultiplyAdd(a_second, b1, acc3);
        }
        $COMPENSATED_MMA

        workgroupBarrier();
        t += $TILE_K_U;
    }
    $RESULT_STORE
    $RESULT_SCATTER
}

// @section prologue_stage_vector
{
    let gr = $ROW_OFFSET + v4_row;
    let tc4 = t + v4_col;
    let shared_base = v4_row * 16u + v4_col;
    if gr < m && tc4 <= k && k - tc4 >= 4u {
        let base = gr * k + tc4;
        let a_values = vec4<f32>(matrix_a[base], matrix_a[base + 1u],
                                  matrix_a[base + 2u], matrix_a[base + 3u]);
        $VECTOR_STORES
    } else {
        for (var e = 0u; e < 4u; e++) {
            let tc = tc4 + e;
            let flat = shared_base + e;
            if gr < m && tc < k {
                let a_val = matrix_a[gr * k + tc];
                $STORE
            } else {
                $ZERO
            }
        }
    }
}

// @section prologue_stage_component
{
    let tc = tc4 + $OFFSETu;
    let flat = shared_base + $OFFSETu;
    let a_val = a_values.$COMPONENT;
    $STORE
}

// @section acc_zero
var acc0 = $COOP_C();
var acc1 = $COOP_C();
var acc2 = $COOP_C();
var acc3 = $COOP_C();

// @section acc_add
var acc0 = $COOP_C();
var acc1 = $COOP_C();
var acc2 = $COOP_C();
var acc3 = $COOP_C();
if row_valid {
    acc0 = coopLoadT<$COOP_C>(&src[c0], n);
    if n1_valid {
        acc1 = coopLoadT<$COOP_C>(&src[c1], n);
    }
}

if second_row_valid {
    acc2 = coopLoadT<$COOP_C>(&src[c2], n);
    if n1_valid {
        acc3 = coopLoadT<$COOP_C>(&src[c3], n);
    }
}

// @section direct_store
if row_valid {
    coopStoreT(acc0, &$C_BUFFER[c0], n);
    if n1_valid {
        coopStoreT(acc1, &$C_BUFFER[c1], n);
    }
}

if second_row_valid {
    coopStoreT(acc2, &$C_BUFFER[c2], n);
    if n1_valid {
        coopStoreT(acc3, &$C_BUFFER[c3], n);
    }
}

// @section shared_result
var<workgroup> shared_c: array<f32, $RESULT_SHARED_SIZE>;

// @section shared_store
if sg < 2u {
    coopStoreT(acc0, &shared_c[sg_row * $OUTPUT_N_U], $OUTPUT_N_U);
    coopStoreT(acc1, &shared_c[sg_row * $OUTPUT_N_U + $TILE_N_U], $OUTPUT_N_U);
}

if owns_second_row {
    coopStoreT(acc2, &shared_c[$TILE_M_U * $OUTPUT_N_U], $OUTPUT_N_U);
    coopStoreT(acc3, &shared_c[$TILE_M_U * $OUTPUT_N_U + $TILE_N_U], $OUTPUT_N_U);
}

// @section result_scatter
workgroupBarrier();
for (var e = 0u; e < $STORE_ITERS_U; e++) {
    let local_idx = lid.x + e * 64u;
    if local_idx < $RESULT_SHARED_SIZE_U {
        let local_row = local_idx / $OUTPUT_N_U;
        let local_col = local_idx - local_row * $OUTPUT_N_U;
        let row = tile_row + local_row;
        let col = tile_col + local_col;
        if row < m && col < n {
            let idx = $RESULT_INDEX;
            var val = shared_c[local_idx];
            $EPILOGUE_BODY
            $C_BUFFER[idx] = $RESULT_VALUE;
        }
    }
}

// @section shared_low
var<workgroup> shared_a0_lo: array<f16, $SHARED_SIZE>;
var<workgroup> shared_a1_lo: array<f16, $SHARED_SIZE>;
var<workgroup> shared_b0_lo: array<f16, $SHARED_SIZE>;
var<workgroup> shared_b1_lo: array<f16, $SHARED_SIZE>;

// @section compensated
var a_lo = $COOP_AB();
if sg_row == 0u {
    a_lo = coopLoadT<$COOP_AB>(&shared_b0_lo[0], $TILE_K_U);
} else {
    a_lo = coopLoadT<$COOP_AB>(&shared_b1_lo[0], $TILE_K_U);
}
let b0_lo = coopLoadT<$COOP_BA>(&shared_a0_lo[0], $TILE_N_U);
let b1_lo = coopLoadT<$COOP_BA>(&shared_a1_lo[0], $TILE_N_U);
acc0 = coopMultiplyAdd(a, b0_lo, acc0);
acc0 = coopMultiplyAdd(a_lo, b0, acc0);
acc1 = coopMultiplyAdd(a, b1_lo, acc1);
acc1 = coopMultiplyAdd(a_lo, b1, acc1);
if owns_second_row {
    let a_second = coopLoadT<$COOP_AB>(&shared_b1[0], $TILE_K_U);
    let a_second_lo = coopLoadT<$COOP_AB>(&shared_b1_lo[0], $TILE_K_U);
    acc2 = coopMultiplyAdd(a_second, b0_lo, acc2);
    acc2 = coopMultiplyAdd(a_second_lo, b0, acc2);
    acc3 = coopMultiplyAdd(a_second, b1_lo, acc3);
    acc3 = coopMultiplyAdd(a_second_lo, b1, acc3);
}
