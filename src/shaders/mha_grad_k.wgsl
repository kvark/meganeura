// MHA gradient wrt K
// Dispatch: [kv_seq, num_kv_heads, 1], WG=64

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
var<storage, read_write> dst: array<f32>;  // dK
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
    let t = wgid.x;      // KV position
    let kv_head = wgid.y; // KV head
    let tid = lid.x;

    let q_seq = params.q_seq;
    let kv_seq = params.kv_seq;
    let num_heads = params.packed_heads >> 16u;
    let num_kv_heads = params.packed_heads & 0xFFFFu;
    let head_dim = params.head_dim;

    // kv_seq == 0 means causal (seq = q_seq). Guard uses q_seq in that case.
    let effective_kv_seq = select(kv_seq, q_seq, kv_seq == 0u);
    if t >= effective_kv_seq || kv_head >= num_kv_heads { return; }

    let heads_per_kv = num_heads / num_kv_heads;
    let kv_dim = num_kv_heads * head_dim;
    let q_dim = num_heads * head_dim;
    let kv_base = t * kv_dim + kv_head * head_dim;
    let scale = inverseSqrt(f32(head_dim));

    var my_dk = 0.0;

    // kv_seq == 0 signals causal: only Q positions >= t contribute.
    let start_pos = select(0u, t, kv_seq == 0u);
    for (var pos = start_pos; pos < q_seq; pos++) {
        for (var head_rel = 0u; head_rel < heads_per_kv; head_rel++) {
            let head = kv_head * heads_per_kv + head_rel;
            let q_base = pos * q_dim + head * head_dim;

            // score = Q·K * scale
            wg_dot[tid] = src_a[q_base + tid] * src_b[kv_base + tid];
            tree_reduce(tid);
            let score = wg_dot[0] * scale;

            // P_t = exp(score - lse)
            let p_t = exp(min(score - lse[pos * num_heads + head], 0.0));

            // row_sum = sum_d(dO[d] * O[d])
            wg_dot[tid] = d_out[q_base + tid] * fwd_dst[q_base + tid];
            tree_reduce(tid);
            let row_sum = wg_dot[0];

            // dP_t = sum_d(dO[d] * V[d])
            wg_dot[tid] = d_out[q_base + tid] * bias[kv_base + tid];
            tree_reduce(tid);
            let dp_t = wg_dot[0];

            // dS_t = P_t * (dP_t - row_sum)
            let ds_t = p_t * (dp_t - row_sum);

            // Accumulate dK
            my_dk += ds_t * scale * src_a[q_base + tid];
        }
    }

    dst[kv_base + tid] = my_dk;
}
