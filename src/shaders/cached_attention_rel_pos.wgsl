// Cached single-token attention WITH T5 relative-position bias.
// Variant of cached_attention.wgsl: adds a per-head learned rel-pos bias to the
// QK^T score before softmax, bucketed from (q_pos - key_pos). The query's
// absolute position is kv_pos (kv_pos_buf[0]); keys span 0..kv_pos. Used by the
// autoregressive temporal decode, whose self-attention carries the learned
// `temporal_decoder.relpos_bias` table.
//
// Dispatch: [1, num_heads, 1]; workgroup 64 lanes (head_dim must be 64).

struct Params {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    num_buckets: u32,
    max_distance: u32,
    bidirectional: u32,
    _pad0: u32,
    _pad1: u32,
}

var<storage> src_a: array<f32>;      // Q: [1, num_heads * head_dim]
var<storage> src_b: array<f32>;      // K cache: [max_seq, num_kv_heads * head_dim]
var<storage> bias: array<f32>;       // V cache: [max_seq, num_kv_heads * head_dim]
var<storage> src_d: array<f32>;      // rel-pos table: [num_heads, num_buckets]
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

// T5 relative position bucket — identical to the generated full-attention
// shader's helper (and shaders/t5_rel_pos.wgsl).
fn rel_pos_bucket(q_minus_k: i32, num_buckets: u32, max_distance: u32, bidirectional: u32) -> u32 {
    var n: i32 = q_minus_k;
    var ret: u32 = 0u;
    var nb: u32 = num_buckets;
    if bidirectional != 0u {
        nb = nb / 2u;
        if n < 0 { ret = nb; n = -n; }
    } else {
        if n < 0 { n = 0; }
    }
    let max_exact: u32 = nb / 2u;
    let n_u: u32 = u32(n);
    if n_u < max_exact { return ret + n_u; }
    let log_n = log(f32(n_u) / f32(max_exact));
    let log_max = log(f32(max_distance) / f32(max_exact));
    let val_large = u32(f32(max_exact) + log_n / log_max * f32(nb - max_exact));
    return ret + min(val_large, nb - 1u);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let head = wgid.y;
    let tid = lid.x;

    let num_heads = params.num_heads;
    let num_kv_heads = params.num_kv_heads;
    let head_dim = params.head_dim;
    let num_buckets = params.num_buckets;
    let max_distance = params.max_distance;
    let bidirectional = params.bidirectional;
    let q_pos = kv_pos_buf[0];
    let kv_len = q_pos + 1u; // attend to positions 0..q_pos inclusive

    if head >= num_heads { return; }

    let kv_head = head / (num_heads / num_kv_heads);
    let kv_head_off = kv_head * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let scale = inverseSqrt(f32(head_dim));
    let q_base = head * head_dim;
    let q_val = src_a[q_base + tid];
    let bias_row_off = head * num_buckets;

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
            let bucket = rel_pos_bucket(i32(q_pos) - i32(t + i), num_buckets, max_distance, bidirectional);
            let score = wg_scores[i * 64u] * scale + src_d[bias_row_off + bucket];
            let new_max = max(max_score, score);
            let correction = exp(max_score - new_max);
            let weight = exp(score - new_max);
            sum_exp = sum_exp * correction + weight;
            let v_base = (t + i) * kv_dim + kv_head_off;
            my_out = my_out * correction + weight * bias[v_base + tid];
            max_score = new_max;
        }
        // All lanes have read wg_scores; barrier before the next tile overwrites it.
        workgroupBarrier();
    }

    // Tail
    for (; t < kv_len; t++) {
        let k_base = t * kv_dim + kv_head_off;
        wg_dot[tid] = q_val * src_b[k_base + tid];
        tree_reduce(tid);
        let bucket = rel_pos_bucket(i32(q_pos) - i32(t), num_buckets, max_distance, bidirectional);
        let score = wg_dot[0] * scale + src_d[bias_row_off + bucket];
        let new_max = max(max_score, score);
        let correction = exp(max_score - new_max);
        let weight = exp(score - new_max);
        sum_exp = sum_exp * correction + weight;
        my_out = my_out * correction + weight * bias[k_base + tid];
        max_score = new_max;
        // All lanes have read wg_dot[0]; barrier before the next iter overwrites it.
        workgroupBarrier();
    }

    let safe_sum = select(sum_exp, 1.0, sum_exp == 0.0);
    dst[q_base + tid] = my_out / safe_sum;
}
