// Softmax attention with an additive bias per (head, query row, key):
//   out[row, head] = Σ_j softmax_j(scale · q_row·k_j + bias[head, row, j]) · v_j
// over keys 0..keys (mode 0), 0..=row (mode 1, causal) or 0..=kv_pos[0]
// (mode 2, cached). Query head h reads KV head h / (heads / kv_heads).
// The bias is read at head · bias_head_stride + row · bias_row_stride + j,
// so a cached step can share one row of biases across its queries.
// Dispatch: [rows, heads, 1], 64 lanes; head_dim up to 64 · MAX_VALUES.

struct Params {
    rows: u32,
    keys: u32,
    packed_heads: u32, // heads << 16 | kv_heads
    head_dim: u32,
    scale_bits: u32,
    mode: u32,
    bias_head_stride: u32,
    bias_row_stride: u32,
}

var<storage> q: array<f32>;
var<storage> k: array<f32>;
var<storage> v: array<f32>;
var<storage> bias: array<f32>;
var<storage> kv_pos: array<u32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

const LANES: u32 = 64u;
const BKV: u32 = 8u;
const MAX_VALUES: u32 = 8u;

var<workgroup> wg_scores: array<f32, 512>; // BKV * LANES

fn reduce_tile(tid: u32) {
    var stride = LANES / 2u;
    loop {
        workgroupBarrier();
        if stride == 0u { break; }
        if tid < stride {
            for (var i = 0u; i < BKV; i++) {
                wg_scores[i * LANES + tid] += wg_scores[i * LANES + tid + stride];
            }
        }
        stride = stride / 2u;
    }
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wgid.x;
    let head = wgid.y;
    let tid = lid.x;
    let heads = params.packed_heads >> 16u;
    let kv_heads = params.packed_heads & 0xFFFFu;
    let dim = params.head_dim;
    // Uniform control flow: every lane of a workgroup takes the same exit.
    if row >= params.rows || head >= heads { return; }

    var count = params.keys;
    if params.mode == 1u {
        count = min(row + 1u, params.keys);
    } else if params.mode == 2u {
        count = min(kv_pos[0] + 1u, params.keys);
    }

    let scale = bitcast<f32>(params.scale_bits);
    let kv_head = head / (heads / kv_heads);
    let kv_width = kv_heads * dim;
    let q_base = row * heads * dim + head * dim;
    let bias_base = head * params.bias_head_stride + row * params.bias_row_stride;

    var q_val: array<f32, 8>;
    var acc: array<f32, 8>;
    for (var e = 0u; e < MAX_VALUES; e++) {
        let d = tid + e * LANES;
        q_val[e] = select(0.0, q[q_base + min(d, dim - 1u)], d < dim);
        acc[e] = 0.0;
    }
    var max_score = -3.4028235e38;
    var sum_exp = 0.0;

    for (var t = 0u; t < count; t += BKV) {
        for (var i = 0u; i < BKV; i++) {
            let j = t + i;
            var partial = 0.0;
            if j < count {
                let k_base = j * kv_width + kv_head * dim;
                for (var e = 0u; e < MAX_VALUES; e++) {
                    let d = tid + e * LANES;
                    if d < dim {
                        partial += q_val[e] * k[k_base + d];
                    }
                }
            }
            wg_scores[i * LANES + tid] = partial;
        }
        reduce_tile(tid);
        for (var i = 0u; i < BKV; i++) {
            let j = t + i;
            if j < count {
                let score = wg_scores[i * LANES] * scale + bias[bias_base + j];
                let new_max = max(max_score, score);
                let correction = exp(max_score - new_max);
                let weight = exp(score - new_max);
                sum_exp = sum_exp * correction + weight;
                let v_base = j * kv_width + kv_head * dim;
                for (var e = 0u; e < MAX_VALUES; e++) {
                    let d = tid + e * LANES;
                    if d < dim {
                        acc[e] = acc[e] * correction + weight * v[v_base + d];
                    }
                }
                max_score = new_max;
            }
        }
        // Every lane has read the tile before the next one overwrites it.
        workgroupBarrier();
    }

    let inv = 1.0 / select(sum_exp, 1.0, sum_exp == 0.0);
    for (var e = 0u; e < MAX_VALUES; e++) {
        let d = tid + e * LANES;
        if d < dim {
            dst[q_base + d] = acc[e] * inv;
        }
    }
}
