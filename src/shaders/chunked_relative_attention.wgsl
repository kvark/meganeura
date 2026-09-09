// Causal chunked self-attention with Transformer-XL relative-key logits.
// Q/K/V are [seq_len, num_heads * head_dim]. `relative_k` contains projected
// relative keys [left_context, num_heads * head_dim], ordered from the
// furthest relative position to the current position. Gemma's blocked form
// extracts a chunk plus its left context and then masks that down to
// distances `< left_context - 1`; iterating that effective window directly is
// equivalent and avoids materializing padded block contexts.

struct Params {
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    left_context: u32,
    softcap_bits: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<storage> src_a: array<f32>;
var<storage> src_b: array<f32>;
var<storage> bias: array<f32>;
var<storage> relative_k: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;
var<workgroup> wg_dot: array<f32, 64>;

const MAX_VALUES_PER_THREAD: u32 = 8u;

fn tree_reduce(tid: u32) {
    workgroupBarrier();
    if tid < 32u { wg_dot[tid] += wg_dot[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u { wg_dot[tid] += wg_dot[tid + 16u]; }
    workgroupBarrier();
    if tid < 8u { wg_dot[tid] += wg_dot[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u { wg_dot[tid] += wg_dot[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u { wg_dot[tid] += wg_dot[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u { wg_dot[tid] += wg_dot[tid + 1u]; }
    workgroupBarrier();
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let query_row = wgid.x;
    let head = wgid.y;
    let tid = lid.x;
    if query_row >= params.seq_len || head >= params.num_heads { return; }

    let model_dim = params.num_heads * params.head_dim;
    let head_off = head * params.head_dim;
    let q_base = query_row * model_dim + head_off;
    let past = params.left_context - 1u;
    let effective_past = past - 1u;
    let first_key = query_row - min(query_row, effective_past);
    let softcap = bitcast<f32>(params.softcap_bits);

    var my_out: array<f32, 8>;
    for (var lane = 0u; lane < MAX_VALUES_PER_THREAD; lane++) {
        my_out[lane] = 0.0;
    }
    var max_score = -1e30;
    var sum_exp = 0.0;

    for (var key_row = first_key; key_row <= query_row; key_row++) {
        let key_base = key_row * model_dim + head_off;
        let distance = query_row - key_row;
        let relative_row = past - distance;
        let relative_base = relative_row * model_dim + head_off;
        var partial = 0.0;
        for (var lane = 0u; lane < MAX_VALUES_PER_THREAD; lane++) {
            let d = lane * 64u + tid;
            if d < params.head_dim {
                partial += src_a[q_base + d] * (src_b[key_base + d] + relative_k[relative_base + d]);
            }
        }
        wg_dot[tid] = partial;
        tree_reduce(tid);
        let score = softcap * tanh(wg_dot[0] / softcap);
        workgroupBarrier();

        let new_max = max(max_score, score);
        let correction = exp(max_score - new_max);
        let weight = exp(score - new_max);
        sum_exp = sum_exp * correction + weight;
        for (var lane = 0u; lane < MAX_VALUES_PER_THREAD; lane++) {
            let d = lane * 64u + tid;
            if d < params.head_dim {
                my_out[lane] = my_out[lane] * correction + weight * bias[key_base + d];
            }
        }
        max_score = new_max;
    }

    let safe_sum = select(sum_exp, 1.0, sum_exp == 0.0);
    for (var lane = 0u; lane < MAX_VALUES_PER_THREAD; lane++) {
        let d = lane * 64u + tid;
        if d < params.head_dim {
            dst[q_base + d] = my_out[lane] / safe_sum;
        }
    }
}
