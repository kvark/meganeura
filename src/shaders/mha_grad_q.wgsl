// MHA gradient wrt Q
// Dispatch: [q_seq, num_heads, 1], WG=64

struct Params {
    q_seq: u32,
    kv_seq: u32,
    packed_heads: u32,
    head_dim: u32,
}

var<storage> d_out: array<f32>;   // dO
var<storage> src_a: array<f32>;   // Q
var<storage> src_b: array<f32>;   // K
var<storage> bias: array<f32>;    // V
var<storage> lse: array<f32>;     // LSE from forward
var<storage> fwd_dst: array<f32>; // O from forward
var<storage, read_write> dst: array<f32>;  // dQ
var<uniform> params: Params;
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

    let q_seq = params.q_seq;
    let kv_seq = params.kv_seq;
    let num_heads = params.packed_heads >> 16u;
    let num_kv_heads = params.packed_heads & 0xFFFFu;
    let head_dim = params.head_dim;

    if pos >= q_seq || head >= num_heads { return; }

    let kv_head = head / (num_heads / num_kv_heads);
    let kv_head_off = kv_head * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let scale = inverseSqrt(f32(head_dim));
    let q_base = pos * (num_heads * head_dim) + head * head_dim;
    let q_val = src_a[q_base + tid];
    let do_val = d_out[q_base + tid];
    let lse_idx = (pos * num_heads + head) * 2u;
    let max_s = lse[lse_idx];
    let log_sum = lse[lse_idx + 1u];

    // Pre-compute row_sum = sum_d(dO[d] * O[d])
    wg_dot[tid] = do_val * fwd_dst[q_base + tid];
    tree_reduce(tid);
    let row_sum = wg_dot[0];

    var my_dq = 0.0;

    // kv_seq == 0 signals causal: each position attends to t ∈ [0, pos].
    let kv_len = select(kv_seq, pos + 1u, kv_seq == 0u);
    for (var t = 0u; t < kv_len; t++) {
        let k_base = t * kv_dim + kv_head_off;

        // score = Q·K * scale
        wg_dot[tid] = q_val * src_b[k_base + tid];
        tree_reduce(tid);
        let score = wg_dot[0] * scale;

        // P_t = exp(score - lse)
        // P_t = exp(score - max_score) / sum_exp
        // Clamp to handle f32 rounding where recomputed score > max_score
        let p_t = exp(min(score - max_s, 0.0) - log_sum);

        // dP_t = sum_d(dO[d] * V[d])
        wg_dot[tid] = do_val * bias[k_base + tid];
        tree_reduce(tid);
        let dp_t = wg_dot[0];

        // dS_t = P_t * (dP_t - row_sum)
        let ds_t = p_t * (dp_t - row_sum);

        // Accumulate dQ
        my_dq += ds_t * scale * src_b[k_base + tid];
    }

    dst[q_base + tid] = my_dq;
}
