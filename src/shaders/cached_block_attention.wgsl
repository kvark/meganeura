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
var<workgroup> wg_scores: array<f32, 1024>;
var<workgroup> wave_slots: atomic<u32>;

// Tokens per reduction round; the round's masked tail costs no extra
// barrier, so a ragged remainder costs one round instead of one per token.
const BKV: u32 = 16u;
const MAX_VALUES_PER_THREAD: u32 = 8u;

// One round of the BKV-wide score reduction: every thread's per-token
// partial in wg_scores collapses to one value per slot, visible to all
// threads. The generator supplies the body — the barrier tree by default,
// or the two-barrier subgroup form where the device supports it.
fn tree_reduce_bkv(tid: u32, sg_id: u32) {
$SCORE_REDUCE
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
) {
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

    // Every tile is a full BKV-wide reduction round; the final one masks the
    // slots past kv_len, so a ragged remainder costs one round instead of a
    // per-token loop of barrier rounds.
    let rounds = (kv_len - first_kv + BKV - 1u) / BKV;
    for (var round = 0u; round < rounds; round++) {
        for (var i = 0u; i < BKV; i++) {
            let t = first_kv + round * BKV + i;
            let k_base = t * kv_dim + kv_head_off;
            var partial = 0.0;
            for (var lane = 0u; lane < MAX_VALUES_PER_THREAD; lane++) {
                let d = lane * 64u + tid;
                if d < params.head_dim && t < kv_len {
                    partial += src_a[q_base + d] * src_b[k_base + d];
                }
            }
            wg_scores[i * 64u + tid] = partial;
        }
        tree_reduce_bkv(tid, sg_id);
        for (var i = 0u; i < BKV; i++) {
            let t = first_kv + round * BKV + i;
            let live = t < kv_len;
            let score = select(-1e30, wg_scores[i * 64u] * scale, live);
            let new_max = max(max_score, score);
            let correction = exp(max_score - new_max);
            let weight = select(0.0, exp(score - new_max), live);
            sum_exp = sum_exp * correction + weight;
            let v_base = t * kv_dim + kv_head_off;
            for (var lane = 0u; lane < MAX_VALUES_PER_THREAD; lane++) {
                let d = lane * 64u + tid;
                if d < params.head_dim && live {
                    my_out[lane] = my_out[lane] * correction + weight * bias[v_base + d];
                }
            }
            max_score = new_max;
        }
        workgroupBarrier();
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
