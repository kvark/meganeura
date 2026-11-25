// A workgroup owns a block of output rows. Each lane forms four scores
// against one row from the other operand, then accumulates four vec4 outputs.
$PARAMS
var<storage> d_out: array<vec4<f32>>;
var<storage> src_a: array<vec4<f32>>;
var<storage> src_b: array<vec4<f32>>;
var<storage> bias: array<vec4<f32>>;
var<storage> lse: array<f32>;
var<storage> fwd_dst: array<f32>;
var<storage, read_write> dst: array<vec4<f32>>;
$SECOND_OUTPUT
var<uniform> params: Params;

const KV = $KV;
const ROWS = 32u;
const COLS = $COLS;
const THREADS = ROWS / 4u * COLS;
var<workgroup> fixed_a: array<vec4<f32>, ROWS * (COLS + 1u)>;
var<workgroup> fixed_b: array<vec4<f32>, ROWS * (COLS + 1u)>;
var<workgroup> walk_a: array<vec4<f32>, COLS * (COLS + 1u)>;
var<workgroup> walk_b: array<vec4<f32>, COLS * (COLS + 1u)>;
var<workgroup> ds_tile: array<vec4<f32>, THREADS>;
$PROB_STORAGE
var<workgroup> stats: array<vec4<f32>, $STATS_ROWS>;

fn query_stats(row: u32) -> vec4<f32> {
    return vec4<f32>(lse[row * 2u], lse[row * 2u + 1u], fwd_dst[row], 0.0);
}

@compute @workgroup_size(THREADS)
fn main(@builtin(workgroup_id) block: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let heads = params.packed_heads >> 16u;
    let kv_heads = params.packed_heads & 65535u;
    let group = heads / kv_heads;
    let causal = params.kv_seq == 0u;
    let kv_len = select(params.kv_seq, params.q_seq, causal);
    let fixed_len = select(params.q_seq, kv_len, KV);
    let row0 = block.x * ROWS;
    let last = min(row0 + ROWS, fixed_len) - 1u;
    let window = params.window_size;
    var begin = 0u;
    var walk_len = select(kv_len, params.q_seq * group, KV);
    if causal {
        if KV {
            begin = row0 * group;
            if window != 0u { walk_len = (last + min(window, params.q_seq - last)) * group; }
        } else {
            if window != 0u { begin = row0 + 1u - min(window, row0 + 1u); }
            walk_len = last + 1u;
        }
    }
    let stride = select(heads, kv_heads, KV);
    let row4 = tid / COLS;
    let col = tid % COLS;
    let scale = inverseSqrt(f32(COLS * 4u));
    for (var i = tid; i < ROWS * COLS; i += THREADS) {
        let r = i / COLS;
        let row = (min(row0 + r, fixed_len - 1u) * stride + block.y);
        let s = r * (COLS + 1u) + col;
        if KV {
            fixed_a[s] = src_b[row * COLS + col];
            fixed_b[s] = bias[row * COLS + col];
        } else {
            fixed_a[s] = src_a[row * COLS + col];
            fixed_b[s] = d_out[row * COLS + col];
            if col == 0u { stats[r] = query_stats(row); }
        }
    }
    var grad: array<vec4<f32>, 4>;
    var grad_v: array<vec4<f32>, 4>;
    for (var base = begin; base < walk_len; base += COLS) {
        workgroupBarrier();
        for (var i = tid; i < COLS * COLS; i += THREADS) {
            let r = i / COLS;
            let pos = min(base + r, walk_len - 1u);
            let s = r * (COLS + 1u) + col;
            if KV {
                let row = (pos / group) * heads + block.y * group + pos % group;
                walk_a[s] = src_a[row * COLS + col];
                walk_b[s] = d_out[row * COLS + col];
                if col == 0u { stats[r] = query_stats(row); }
            } else {
                let row = pos * kv_heads + block.y / group;
                walk_a[s] = src_b[row * COLS + col];
                walk_b[s] = bias[row * COLS + col];
            }
        }
        workgroupBarrier();
        var score = vec4<f32>(0.0);
        var dprob = vec4<f32>(0.0);
        for (var k = 0u; k < COLS; k++) {
            let a = walk_a[col * (COLS + 1u) + k];
            let b = walk_b[col * (COLS + 1u) + k];
            for (var j = 0u; j < 4u; j++) {
                let at = (row4 * 4u + j) * (COLS + 1u) + k;
                score[j] += dot(fixed_a[at], a);
                dprob[j] += dot(fixed_b[at], b);
            }
        }
        var prob = vec4<f32>(0.0);
        var ds = vec4<f32>(0.0);
        for (var j = 0u; j < 4u; j++) {
            let r = row4 * 4u + j;
            let query = select(row0 + r, (base + col) / group, KV);
            let key = select(base + col, row0 + r, KV);
            let end = select(kv_len, query + 1u, causal);
            let start = select(0u, end - min(end, params.window_size), params.window_size != 0u);
            if row0 + r < fixed_len && base + col < walk_len && key >= start && key < end {
                let q = stats[select(r, col, KV)];
                prob[j] = exp(min(score[j] * scale - q.x, 0.0) - q.y);
                ds[j] = prob[j] * (dprob[j] - q.z);
            }
        }
        ds_tile[col * (ROWS / 4u) + row4] = ds;
        $STORE_PROB
        workgroupBarrier();
        for (var k = 0u; k < COLS; k++) {
            let at = k * (ROWS / 4u) + row4;
            let a = walk_a[k * (COLS + 1u) + col];
            let b = walk_b[k * (COLS + 1u) + col];
            for (var j = 0u; j < 4u; j++) {
                grad[j] = fma(vec4<f32>(ds_tile[at][j]), a, grad[j]);
                $ACCUM_V
            }
        }
    }
    for (var j = 0u; j < 4u; j++) {
        let r = row0 + row4 * 4u + j;
        if r < fixed_len {
            let out = (r * stride + block.y) * COLS + col;
            dst[out] = grad[j] * scale;
            $STORE_V
        }
    }
}
