// K-split GEMV. Forward workgroups cover four columns; transposed workgroups
// cover one, two or four contiguous B rows. Width and reduction are shared
// specialization slots, including in the integer-dot templates.

// @section forward
$ENABLE_F16
struct Params {
    m: u32,
    n: u32,
    k: u32,
    $EPS_FIELD: u32,
}
var<storage> matrix_a: array<f32>;
$NORM_WEIGHT
var<storage> matrix_b: $B_STORAGE;
var<storage, read_write> matrix_c: array<vec4<f32>>;
$ADDEND_DECL
var<uniform> params: Params;
const LANES: u32 = $LANESu;
$NORM_SCRATCH
var<workgroup> reduce_buf: array<vec4<f32>, LANES>;
$WEIGHT_HELPERS
@compute @workgroup_size(LANES)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>$SUBGROUP_ARGS) {
    let col4 = wgid.x;
    let lane = lid.x;
    let n_v4 = params.n / 4u;
$EARLY_RETURN
    let k = params.k;
$NORM_PROLOGUE
    var acc = vec4<f32>(0.0);
    var kk = lane;
    loop {
        if kk >= k { break; }
        let a = matrix_a[kk]$A_SCALE;
$WEIGHT_LOAD
        acc = acc + vec4<f32>(a) * b;
        kk += LANES;
    }
$REDUCTION
    if lane == 0u {
        matrix_c[col4] = $TOTAL$ADDEND;
    }
}

// @section transposed
$ENABLE_F16
struct Params {
    m: u32,
    n: u32,
    k: u32,
    $EPS_FIELD: u32,
}
var<storage> matrix_a: array<vec4<f32>>;
$NORM_WEIGHT
var<storage> matrix_b: $B_STORAGE;
var<storage, read_write> matrix_c: array<f32>;
$ADDEND_DECL
var<uniform> params: Params;
const LANES: u32 = $LANESu;
$NORM_SCRATCH
var<workgroup> reduce_buf: array<$ACC_TYPE, LANES>;
@compute @workgroup_size(LANES)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>$SUBGROUP_ARGS) {
    let col = $COL_EXPR;
    let lane = lid.x;
$EARLY_RETURN
    let k_v4 = params.k / 4u;
    let row_off = col * k_v4;
$NORM_K
$NORM_PROLOGUE
    var acc = $ACC_ZERO;
    var kk_v4 = lane;
    loop {
        if kk_v4 >= k_v4 { break; }
$A_LOAD
$ACCUMULATE
        kk_v4 += LANES;
    }
$REDUCTION
    if lane == 0u {
$STORE
    }
}

// @section weight_load
        let b = $B_VALUE;

// @section packed_load
        let col = col4 * 4u;
        let b = vec4<f32>(
            $DEQUANT(kk, col),
            $DEQUANT(kk, col + 1u),
            $DEQUANT(kk, col + 2u),
            $DEQUANT(kk, col + 3u),
        );

// @section bt_a
        let a = matrix_a[kk_v4];

// @section bt_norm_a
        let at = kk_v4 * 4u;
        let a = matrix_a[kk_v4] * rs * vec4<f32>(norm_w[at], norm_w[at+1u], norm_w[at+2u], norm_w[at+3u]);

// @section bt_accumulate
        let b = $B_VALUE;
        acc = acc + dot(a, b);

// @section bt_row_accumulate
        if col + $ROWu < params.n { acc.$COMPONENT += dot(a, $B_VALUE); }

// @section bt_store
        matrix_c[col] = $TOTAL$ADDEND;

// @section bt_total
        let total = $TOTAL;

// @section bt_row_store
        if col + $ROWu < params.n { matrix_c[col + $ROWu] = total.$COMPONENT$ADDEND; }

// @section addend
var<storage> src: array<vec4<f32>>;

// @section bt_addend
var<storage> src: array<f32>;

// @section guard
    if col4 >= n_v4 { return; }

// @section bt_guard
    if col >= params.n { return; }

// @section norm_weight
var<storage> norm_w: array<f32>;

// @section norm_scratch
var<workgroup> scale_buf: array<f32, LANES>;
var<workgroup> inv_rms: f32;

// @section norm_k
    let k = params.k;

// @section norm_end
    let rs = inv_rms;
    if col4 >= n_v4 { return; }

// @section bt_norm_end
    let rs = inv_rms;
    if col >= params.n { return; }

// @section tree_start
    reduce_buf[lane] = acc;
    workgroupBarrier();

// @section tree_step
    if lane < $STRIDEu { reduce_buf[lane] += reduce_buf[lane + $STRIDEu]; }
    workgroupBarrier();

// @section subgroup_reduce
    // Actual subgroup IDs also cover partially populated subgroups.
    var group_total = subgroupAdd(acc);
    if wave_count > 1u {
        if sg_id == subgroupBroadcastFirst(sg_id) {
            reduce_buf[wave_id] = group_total;
        }
        workgroupBarrier();
        if lane == 0u {
            group_total = reduce_buf[0];
            for (var g = 1u; g < wave_count; g += 1u) {
                group_total += reduce_buf[g];
            }
        }
    }
