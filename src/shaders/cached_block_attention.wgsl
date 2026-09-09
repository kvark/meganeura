// Cached causal attention for Q=[block_len, num_heads*head_dim]. K/V for the
// block have already been written into the cache beginning at kv_pos.

struct Params {
    window_size: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_len: u32,
    max_seq: u32,
    _pad0: u32,
    _pad1: u32,
}

var<storage> src_a: array<f32>;
var<storage> src_b: array<f32>;
var<storage> bias: array<f32>;
var<storage> kv_pos_buf: array<u32>;
var<storage> valid_len_buf: array<u32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;
var<workgroup> wg_dot: array<f32, 64>;
var<workgroup> wg_scores: array<f32, 512>;

const BKV: u32 = 8u;
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

fn tree_reduce_8(tid: u32) {
    workgroupBarrier();
    if tid < 32u { for (var i = 0u; i < BKV; i++) { wg_scores[i * 64u + tid] += wg_scores[i * 64u + tid + 32u]; } }
    workgroupBarrier();
    if tid < 16u { for (var i = 0u; i < BKV; i++) { wg_scores[i * 64u + tid] += wg_scores[i * 64u + tid + 16u]; } }
    workgroupBarrier();
    if tid < 8u { for (var i = 0u; i < BKV; i++) { wg_scores[i * 64u + tid] += wg_scores[i * 64u + tid + 8u]; } }
    workgroupBarrier();
    if tid < 4u { for (var i = 0u; i < BKV; i++) { wg_scores[i * 64u + tid] += wg_scores[i * 64u + tid + 4u]; } }
    workgroupBarrier();
    if tid < 2u { for (var i = 0u; i < BKV; i++) { wg_scores[i * 64u + tid] += wg_scores[i * 64u + tid + 2u]; } }
    workgroupBarrier();
    if tid < 1u { for (var i = 0u; i < BKV; i++) { wg_scores[i * 64u + tid] += wg_scores[i * 64u + tid + 1u]; } }
    workgroupBarrier();
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let query_row = wgid.x;
    let head = wgid.y;
    let tid = lid.x;
    let valid_len = min(valid_len_buf[0], params.block_len);
    if query_row >= valid_len || head >= params.num_heads { return; }

    let query_position = kv_pos_buf[0] + query_row;
    let kv_len = min(query_position + 1u, params.max_seq);
    let first_kv = select(
        0u,
        kv_len - params.window_size,
        params.window_size != 0u && kv_len > params.window_size,
    );
    let kv_head = head / (params.num_heads / params.num_kv_heads);
    let kv_head_off = kv_head * params.head_dim;
    let kv_dim = params.num_kv_heads * params.head_dim;
    let scale = inverseSqrt(f32(params.head_dim));
    let q_base = query_row * params.num_heads * params.head_dim + head * params.head_dim;

    var my_out: array<f32, 8>;
    for (var lane = 0u; lane < MAX_VALUES_PER_THREAD; lane++) {
        my_out[lane] = 0.0;
    }
    var max_score = -1e30;
    var sum_exp = 0.0;

    let tile_end = first_kv + ((kv_len - first_kv) / BKV) * BKV;
    var t = first_kv;
    for (; t < tile_end; t += BKV) {
        for (var i = 0u; i < BKV; i++) {
            let k_base = (t + i) * kv_dim + kv_head_off;
            var partial = 0.0;
            for (var lane = 0u; lane < MAX_VALUES_PER_THREAD; lane++) {
                let d = lane * 64u + tid;
                if d < params.head_dim {
                    partial += src_a[q_base + d] * src_b[k_base + d];
                }
            }
            wg_scores[i * 64u + tid] = partial;
        }
        tree_reduce_8(tid);
        for (var i = 0u; i < BKV; i++) {
            let score = wg_scores[i * 64u] * scale;
            let new_max = max(max_score, score);
            let correction = exp(max_score - new_max);
            let weight = exp(score - new_max);
            sum_exp = sum_exp * correction + weight;
            let v_base = (t + i) * kv_dim + kv_head_off;
            for (var lane = 0u; lane < MAX_VALUES_PER_THREAD; lane++) {
                let d = lane * 64u + tid;
                if d < params.head_dim {
                    my_out[lane] = my_out[lane] * correction + weight * bias[v_base + d];
                }
            }
            max_score = new_max;
        }
        workgroupBarrier();
    }

    for (; t < kv_len; t++) {
        let k_base = t * kv_dim + kv_head_off;
        var partial = 0.0;
        for (var lane = 0u; lane < MAX_VALUES_PER_THREAD; lane++) {
            let d = lane * 64u + tid;
            if d < params.head_dim {
                partial += src_a[q_base + d] * src_b[k_base + d];
            }
        }
        wg_dot[tid] = partial;
        tree_reduce(tid);
        let score = wg_dot[0] * scale;
        workgroupBarrier();
        let new_max = max(max_score, score);
        let correction = exp(max_score - new_max);
        let weight = exp(score - new_max);
        sum_exp = sum_exp * correction + weight;
        for (var lane = 0u; lane < MAX_VALUES_PER_THREAD; lane++) {
            let d = lane * 64u + tid;
            if d < params.head_dim {
                my_out[lane] = my_out[lane] * correction + weight * bias[k_base + d];
            }
        }
        max_score = new_max;
    }

    let safe_sum = select(sum_exp, 1.0, sum_exp == 0.0);
    let dst_base = query_row * params.num_heads * params.head_dim + head * params.head_dim;
    for (var lane = 0u; lane < MAX_VALUES_PER_THREAD; lane++) {
        let d = lane * 64u + tid;
        if d < params.head_dim {
            dst[dst_base + d] = my_out[lane] / safe_sum;
        }
    }
}
