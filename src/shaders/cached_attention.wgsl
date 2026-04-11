// Cached single-token attention: Q=[1, num_heads*head_dim], K/V from cache.
// Dispatch: [1, num_heads, 1]
// Each workgroup handles one head: loops over 0..kv_len cached positions.
// kv_len = kv_pos + 1 (read from kv_pos_buf storage buffer).
// Uses online softmax (same algorithm as attention.wgsl).

struct Params {
    _reserved: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
}

var<storage> src_a: array<f32>;      // Q: [1, num_heads * head_dim]
var<storage> src_b: array<f32>;      // K cache: [max_seq, num_kv_heads * head_dim]
var<storage> bias: array<f32>;       // V cache: [max_seq, num_kv_heads * head_dim]
var<storage> kv_pos_buf: array<u32>; // [1] — current kv position
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;
var<workgroup> wg_dot: array<f32, 64>;
var<workgroup> wg_scores: array<f32, 512>;  // BKV * 64

const BKV: u32 = 8u;

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
    let head = wgid.y;
    let tid = lid.x;

    let num_heads = params.num_heads;
    let num_kv_heads = params.num_kv_heads;
    let head_dim = params.head_dim;
    let kv_len = kv_pos_buf[0] + 1u; // attend to positions 0..kv_pos inclusive

    if head >= num_heads { return; }

    let kv_head = head / (num_heads / num_kv_heads);
    let kv_head_off = kv_head * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let scale = inverseSqrt(f32(head_dim));
    let q_base = head * head_dim;
    let q_val = src_a[q_base + tid];

    var my_out = 0.0;
    var max_score = -1e30;
    var sum_exp = 0.0;

    // Tiled KV loop: BKV positions per reduction
    let tile_end = (kv_len / BKV) * BKV;
    var t = 0u;
    for (; t < tile_end; t += BKV) {
        for (var i = 0u; i < BKV; i++) {
            let k_base = (t + i) * kv_dim + kv_head_off;
            wg_scores[i * 64u + tid] = q_val * src_b[k_base + tid];
        }
        tree_reduce_8(tid);

        for (var i = 0u; i < BKV; i++) {
            let score = wg_scores[i * 64u] * scale;
            let new_max = max(max_score, score);
            let correction = exp(max_score - new_max);
            let weight = exp(score - new_max);
            sum_exp = sum_exp * correction + weight;
            let v_base = (t + i) * kv_dim + kv_head_off;
            my_out = my_out * correction + weight * bias[v_base + tid];
            max_score = new_max;
        }
    }

    // Tail
    for (; t < kv_len; t++) {
        let k_base = t * kv_dim + kv_head_off;
        wg_dot[tid] = q_val * src_b[k_base + tid];
        tree_reduce(tid);
        let score = wg_dot[0] * scale;
        let new_max = max(max_score, score);
        let correction = exp(max_score - new_max);
        let weight = exp(score - new_max);
        sum_exp = sum_exp * correction + weight;
        my_out = my_out * correction + weight * bias[k_base + tid];
        max_score = new_max;
    }

    let safe_sum = select(sum_exp, 1.0, sum_exp == 0.0);
    dst[q_base + tid] = my_out / safe_sum;
}
