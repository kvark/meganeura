// Flash-decoding split-K partial: one workgroup per (row, split, head)
// computes the online-softmax partial for its slice of the KV range.
// A split that starts past kv_len writes the identity partial —
// m = -1e30, l = 0, acc = 0 — which the combine folds away with weight
// exp(-1e30 - m) = 0, so short contexts pay only for the empty slices'
// launches.
//
// Partial row, one per (row, head, split):
//   [0..head_dim]  acc = sum exp(score - m) * V   (unscaled)
//   [head_dim]     m, the slice's running max
//   [head_dim + 1] l, the slice's sum of exp

struct Params {
    window_size: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_len: u32,
    max_seq: u32,
    splits: u32,
    chunk: u32,
}

var<storage> src_a: array<f32>;
var<storage> src_b: array<f32>;
var<storage> bias: array<f32>;
var<storage> kv_pos_buf: array<u32>;
var<storage> valid_len_buf: array<u32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;
var<workgroup> wg_scores: array<f32, 2048>;

const BKV: u32 = 16u;
const MAX_VALUES_PER_THREAD: u32 = 8u;

fn tree_reduce_bkv(tid: u32) {
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
    let split = wgid.y;
    let head = wgid.z;
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
    // This split's token range, clamped to what exists.
    let split_begin = first_kv + split * params.chunk;
    let split_end = min(split_begin + params.chunk, kv_len);

    let kv_head = head / (params.num_heads / params.num_kv_heads);
    let kv_head_off = kv_head * params.head_dim;
    let kv_dim = params.num_kv_heads * params.head_dim;
    let scale = inverseSqrt(f32(params.head_dim));
    let q_base = query_row * params.num_heads * params.head_dim + head * params.head_dim;

    // Layout: partial row = [(row * heads + head) * splits + split] * (head_dim + 2)
    let part = ((query_row * params.num_heads + head) * params.splits + split)
        * (params.head_dim + 2u);

    if split_begin >= kv_len {
        for (var d = tid; d < params.head_dim; d += 64u) {
            dst[part + d] = 0.0;
        }
        // An empty slice: the identity partial. The combine folds it away
        // with weight exp(-1e30 - m) = 0.
        if tid == 0u {
            dst[part + params.head_dim] = -1e30;
            dst[part + params.head_dim + 1u] = 0.0;
        }
        return;
    }

    var my_out: array<f32, 8>;
    for (var lane = 0u; lane < MAX_VALUES_PER_THREAD; lane++) {
        my_out[lane] = 0.0;
    }
    var max_score = -1e30;
    var sum_exp = 0.0;

    var t = split_begin;
    while t < split_end {
        for (var i = 0u; i < BKV; i++) {
            let at = t + i;
            let k_base = at * kv_dim + kv_head_off;
            var partial = 0.0;
            for (var lane = 0u; lane < MAX_VALUES_PER_THREAD; lane++) {
                let d = lane * 64u + tid;
                if d < params.head_dim && at < split_end {
                    partial += src_a[q_base + d] * src_b[k_base + d];
                }
            }
            wg_scores[i * 64u + tid] = partial;
        }
        tree_reduce_bkv(tid);
        var tile_max = max_score;
        for (var i = 0u; i < BKV; i++) {
            if t + i < split_end {
                tile_max = max(tile_max, wg_scores[i * 64u] * scale);
            }
        }
        let correction = exp(max_score - tile_max);
        sum_exp *= correction;
        for (var lane = 0u; lane < MAX_VALUES_PER_THREAD; lane++) {
            my_out[lane] *= correction;
        }
        for (var i = 0u; i < BKV; i++) {
            let at = t + i;
            let live = at < split_end;
            let score = select(-1e30, wg_scores[i * 64u] * scale, live);
            let weight = select(0.0, exp(score - tile_max), live);
            sum_exp += weight;
            let v_base = at * kv_dim + kv_head_off;
            for (var lane = 0u; lane < MAX_VALUES_PER_THREAD; lane++) {
                let d = lane * 64u + tid;
                if d < params.head_dim && live {
                    my_out[lane] += weight * bias[v_base + d];
                }
            }
        }
        max_score = tile_max;
        workgroupBarrier();
        t += BKV;
    }

    // Unnormalized: the combine rescales by exp(m_i - m).
    for (var lane = 0u; lane < MAX_VALUES_PER_THREAD; lane++) {
        let d = lane * 64u + tid;
        if d < params.head_dim {
            dst[part + d] = my_out[lane];
        }
    }
    if tid == 0u {
        dst[part + params.head_dim] = max_score;
        dst[part + params.head_dim + 1u] = sum_exp;
    }
}
