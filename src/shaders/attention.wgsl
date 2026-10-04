// @section main
$PARAMS
var<storage> src_a: array<f32>;
var<storage> src_b: array<f32>;
var<storage> bias: array<f32>;
var<storage, read_write> dst: array<f32>;
var<storage, read_write> lse: array<f32>;
var<uniform> params: Params;
var<workgroup> wg_scores: array<f32, $TILE_ELEMENTS>;
var<workgroup> wg_dot: array<f32, $HEAD_DIM>;

fn tree_reduce_8(tid: u32) {
$SCORE_REDUCE
}

fn tree_reduce(tid: u32) {
$DOT_STEP
    workgroupBarrier();
}

@compute @workgroup_size($HEAD_DIM)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let pos = wgid.x;
    let head = wgid.y;
    let tid = lid.x;
    let q_seq = params.q_seq;
    let kv_seq = params.kv_seq;
    let num_heads = params.packed_heads >> 16u;
    let num_kv_heads = params.packed_heads & 0xFFFFu;
    let head_dim = params.head_dim;
    if pos >= q_seq || head >= num_heads { return; }

    let kv_len = select(kv_seq, pos + 1u, kv_seq == 0u);
    let window_size = params.window_size;
    let kv_start = select(0u, kv_len - min(kv_len, window_size), window_size > 0u);

    let kv_head = head / (num_heads / num_kv_heads);
    let kv_head_off = kv_head * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let scale = inverseSqrt(f32(head_dim));
    let q_base = pos * (num_heads * head_dim) + head * head_dim;
    let live = tid < head_dim;
    let d = min(tid, $LASTu);
    let q_val = select(0.0, src_a[q_base + d], live);

    var my_out = 0.0;
    var max_score = -1e30;
    var sum_exp = 0.0;

    let kv_range = kv_len - kv_start;
    let tile_end = kv_start + (kv_range / 8u) * 8u;
    var t = kv_start;
    for (; t < tile_end; t += 8u) {
        for (var i = 0u; i < 8u; i++) {
            let k_base = (t + i) * kv_dim + kv_head_off;
            wg_scores[i * $HEAD_DIMu + tid] = select(0.0, q_val * src_b[k_base + d], live);
        }
        tree_reduce_8(tid);

        for (var i = 0u; i < 8u; i++) {
            let score = wg_scores[i * $HEAD_DIMu] * scale;
            let new_max = max(max_score, score);
            let correction = exp(max_score - new_max);
            let weight = exp(score - new_max);
            sum_exp = sum_exp * correction + weight;
            let v_base = (t + i) * kv_dim + kv_head_off;
            my_out = my_out * correction + weight * bias[v_base + d];
            max_score = new_max;
        }
        // Every lane must consume the reduced value before scratch is reused.
        workgroupBarrier();
    }

    for (; t < kv_len; t++) {
        let k_base = t * kv_dim + kv_head_off;
        wg_dot[tid] = select(0.0, q_val * src_b[k_base + d], live);
        tree_reduce(tid);
        let score = wg_dot[0] * scale;

        let new_max = max(max_score, score);
        let correction = exp(max_score - new_max);
        let weight = exp(score - new_max);
        sum_exp = sum_exp * correction + weight;
        my_out = my_out * correction + weight * bias[k_base + d];
        max_score = new_max;
        // Every lane must consume the reduced value before scratch is reused.
        workgroupBarrier();
    }

    let safe_sum = select(sum_exp, 1.0, sum_exp == 0.0);
    if live {
        dst[q_base + tid] = my_out / safe_sum;
    }

    if tid == 0u {
        let idx = (pos * num_heads + head) * 2u;
        lse[idx] = max_score;
        lse[idx + 1u] = select(log(sum_exp), -1e30, sum_exp == 0.0);
    }
}

// @section score_reduce
$SCORE_STEP
    workgroupBarrier();

// @section score_step
    workgroupBarrier();
    if tid < $STRIDEu {
        for (var i = 0u; i < $KEY_COUNT; i++) {
            wg_scores[i * $HEAD_DIMu + tid] += wg_scores[i * $HEAD_DIMu + tid + $STRIDEu];
        }
    }

// @section dot_step
    workgroupBarrier();
    if tid < $STRIDEu { wg_dot[tid] += wg_dot[tid + $STRIDEu]; }
