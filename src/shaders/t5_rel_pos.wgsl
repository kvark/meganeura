// T5 relative position bias: materialize bias[H, Q, K] from learned table[H, num_buckets]
// using T5's log-bucketed position function. Output is added to QK^T before softmax.
//
// Bucket math (matches flaxformer.components.relative_position_biases):
//   n = q - k  (positive = looking back)
//   bidirectional: split buckets in half, first half for n<0 (lookahead), second for n>=0; use abs
//   unidirectional: only n>=0 (clamp negatives to 0)
//   within each half: first num_buckets/4 are linear, rest are log-spaced up to max_distance

struct Params {
    q_len: u32,
    kv_len: u32,
    num_heads: u32,
    num_buckets: u32,
    max_distance: u32,
    bidirectional: u32, // 0 or 1
    _pad0: u32,
    _pad1: u32,
}

var<storage> src: array<f32>;            // bias_table[num_heads * num_buckets]
var<storage, read_write> dst: array<f32>; // out[num_heads * q_len * kv_len]
var<uniform> params: Params;

fn rel_pos_bucket(rel_pos: i32) -> u32 {
    // rel_pos = q - k. Negative means k is in the future (lookahead).
    var n: i32 = rel_pos;
    var ret: u32 = 0u;
    var nb: u32 = params.num_buckets;
    if params.bidirectional != 0u {
        nb = nb / 2u;
        if n < 0 {
            ret = nb;
            n = -n;
        }
    } else {
        if n < 0 { n = 0; }
    }
    let max_exact: u32 = nb / 2u;
    let n_u: u32 = u32(n);
    if n_u < max_exact {
        return ret + n_u;
    }
    let log_n = log(f32(n_u) / f32(max_exact));
    let log_max = log(f32(params.max_distance) / f32(max_exact));
    let val_large_f = f32(max_exact) + log_n / log_max * f32(nb - max_exact);
    let val_large = u32(val_large_f);
    let val_clamped = min(val_large, nb - 1u);
    return ret + val_clamped;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let qk = params.q_len * params.kv_len;
    let total = params.num_heads * qk;
    if i >= total { return; }
    let head = i / qk;
    let in_qk = i - head * qk;
    let q = in_qk / params.kv_len;
    let k = in_qk - q * params.kv_len;
    let bucket = rel_pos_bucket(i32(q) - i32(k));
    dst[i] = src[head * params.num_buckets + bucket];
}
