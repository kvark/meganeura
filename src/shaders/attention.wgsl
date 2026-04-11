// Tiled attention: online softmax with KV-tiling and GQA
// BKV=8 KV positions per tile, reducing workgroup barriers by 8x.
// Variants: causal (kv_len=pos+1), full (kv_len=q_seq), cross (kv_len=kv_seq)
//
// Dispatch: [q_seq, num_heads, 1], WG=64 (one thread per head_dim element)

struct Params {
    $PARAM_FIELDS
}

var<storage> src_a: array<f32>;  // Q
var<storage> src_b: array<f32>;  // K
var<storage> bias: array<f32>;   // V
var<storage, read_write> dst: array<f32>;
var<storage, read_write> lse: array<f32>;  // log-sum-exp for backward
var<uniform> params: Params;
// Shared memory for tiled score reduction: 8 scores × 64 partial sums
var<workgroup> wg_scores: array<f32, 512>;
$ROPE_DECL

const BKV: u32 = 8u;

fn tree_reduce_8(tid: u32) {
    // Reduce 8 independent dot products simultaneously.
    // wg_scores layout: [8][64] — each row is a partial dot product.
    workgroupBarrier();
    if tid < 32u {
        for (var i = 0u; i < BKV; i++) {
            wg_scores[i * 64u + tid] += wg_scores[i * 64u + tid + 32u];
        }
    }
    workgroupBarrier();
    if tid < 16u {
        for (var i = 0u; i < BKV; i++) {
            wg_scores[i * 64u + tid] += wg_scores[i * 64u + tid + 16u];
        }
    }
    workgroupBarrier();
    if tid < 8u {
        for (var i = 0u; i < BKV; i++) {
            wg_scores[i * 64u + tid] += wg_scores[i * 64u + tid + 8u];
        }
    }
    workgroupBarrier();
    if tid < 4u {
        for (var i = 0u; i < BKV; i++) {
            wg_scores[i * 64u + tid] += wg_scores[i * 64u + tid + 4u];
        }
    }
    workgroupBarrier();
    if tid < 2u {
        for (var i = 0u; i < BKV; i++) {
            wg_scores[i * 64u + tid] += wg_scores[i * 64u + tid + 2u];
        }
    }
    workgroupBarrier();
    if tid < 1u {
        for (var i = 0u; i < BKV; i++) {
            wg_scores[i * 64u + tid] += wg_scores[i * 64u + tid + 1u];
        }
    }
    workgroupBarrier();
}

// Single-score tree reduce for tail elements (when remaining < BKV).
// Reuses slot 0 of wg_scores.
var<workgroup> wg_dot: array<f32, 64>;

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
    let pos = wgid.x;
    let head = wgid.y;
    let tid = lid.x;

    $PARSE_PARAMS

    if pos >= q_seq || head >= num_heads { return; }

    let kv_head = head / (num_heads / num_kv_heads);
    let kv_head_off = kv_head * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let scale = inverseSqrt(f32(head_dim));
    let q_base = pos * (num_heads * head_dim) + head * head_dim;
    let q_raw = src_a[q_base + tid];
    $ROPE_Q_APPLY
    let q_val = $Q_VAL_EXPR;

    var my_out = 0.0;
    var max_score = -1e30;
    var sum_exp = 0.0;

    // --- Tiled KV loop: process BKV positions per reduction ---
    let kv_start = $KV_START;
    var t = kv_start;
    let tile_end = kv_start + ((kv_len - kv_start) / BKV) * BKV;
    for (; t < tile_end; t += BKV) {
        // Compute BKV partial dot products simultaneously
        for (var i = 0u; i < BKV; i++) {
            let k_base = (t + i) * kv_dim + kv_head_off;
            let k_val = $K_VAL_EXPR;
            wg_scores[i * 64u + tid] = q_val * k_val;
        }
        tree_reduce_8(tid);
        // wg_scores[i * 64] now holds score[i] (before scaling)

        // Online softmax + output accumulation for BKV positions
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

    // --- Tail: process remaining KV positions one at a time ---
    for (; t < kv_len; t++) {
        let k_base = t * kv_dim + kv_head_off;
        let k_val_tail = $K_VAL_TAIL_EXPR;
        wg_dot[tid] = q_val * k_val_tail;
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

    // Store (max_score, log_sum_exp) for backward pass.
    if tid == 0u {
        let idx = (pos * num_heads + head) * 2u;
        lse[idx] = max_score;
        lse[idx + 1u] = select(log(sum_exp), -1e30, sum_exp == 0.0);
    }
}
