// Fused MHA gradient wrt K and V (Flash Attention 2 style).
//
// Computes dK and dV in a single pass over Q positions, sharing the
// recomputed Q·K score. Saves one dispatch + one score recomputation
// per attention layer compared to separate GradK + GradV.
//
// Dispatch: [kv_seq, num_kv_heads, 1], WG=64
// Outputs: dst (dK), dst2 (dV)

struct Params {
    q_seq: u32,
    kv_seq: u32,
    packed_heads: u32,
    head_dim: u32,
    window_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<storage> d_out: array<f32>;   // dO
var<storage> src_a: array<f32>;   // Q
var<storage> src_b: array<f32>;   // K
var<storage> bias: array<f32>;    // V
var<storage> lse: array<f32>;     // LSE from forward
var<storage> fwd_dst: array<f32>; // O from forward
var<storage, read_write> dst: array<f32>;  // dK
var<storage, read_write> dst2: array<f32>; // dV
var<uniform> params: Params;
var<workgroup> wg_a: array<f32, 64>;
var<workgroup> wg_b: array<f32, 64>;
var<workgroup> wg_c: array<f32, 64>;

// Fused triple tree_reduce: Q·K, dO·O, dO·V in one pass.
fn triple_tree_reduce(tid: u32) {
    workgroupBarrier();
    if tid < 32u { wg_a[tid] += wg_a[tid + 32u]; wg_b[tid] += wg_b[tid + 32u]; wg_c[tid] += wg_c[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u { wg_a[tid] += wg_a[tid + 16u]; wg_b[tid] += wg_b[tid + 16u]; wg_c[tid] += wg_c[tid + 16u]; }
    workgroupBarrier();
    if tid < 8u { wg_a[tid] += wg_a[tid + 8u]; wg_b[tid] += wg_b[tid + 8u]; wg_c[tid] += wg_c[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u { wg_a[tid] += wg_a[tid + 4u]; wg_b[tid] += wg_b[tid + 4u]; wg_c[tid] += wg_c[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u { wg_a[tid] += wg_a[tid + 2u]; wg_b[tid] += wg_b[tid + 2u]; wg_c[tid] += wg_c[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u { wg_a[tid] += wg_a[tid + 1u]; wg_b[tid] += wg_b[tid + 1u]; wg_c[tid] += wg_c[tid + 1u]; }
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

    let effective_kv_seq = select(kv_seq, q_seq, kv_seq == 0u);
    if t >= effective_kv_seq || kv_head >= num_kv_heads { return; }

    let heads_per_kv = num_heads / num_kv_heads;
    let kv_dim = num_kv_heads * head_dim;
    let q_dim = num_heads * head_dim;
    let kv_base = t * kv_dim + kv_head * head_dim;
    let scale = inverseSqrt(f32(head_dim));
    let k_val = src_b[kv_base + tid];

    var my_dk = 0.0;
    var my_dv = 0.0;

    let start_pos = select(0u, t, kv_seq == 0u);
    let window = params.window_size;
    let end_pos = select(q_seq, min(q_seq, t + window), window > 0u);
    for (var pos = start_pos; pos < end_pos; pos++) {
        for (var head_rel = 0u; head_rel < heads_per_kv; head_rel++) {
            let head = kv_head * heads_per_kv + head_rel;
            let q_base = pos * q_dim + head * head_dim;
            let do_val = d_out[q_base + tid];

            // Fused triple reduction: Q·K, dO·O, dO·V
            wg_a[tid] = src_a[q_base + tid] * k_val;       // Q·K
            wg_b[tid] = do_val * fwd_dst[q_base + tid];     // dO·O (row_sum)
            wg_c[tid] = do_val * bias[kv_base + tid];       // dO·V (dp_t)
            triple_tree_reduce(tid);
            let score = wg_a[0] * scale;
            let row_sum = wg_b[0];
            let dp_t = wg_c[0];

            // Softmax probability
            let lse_idx = (pos * num_heads + head) * 2u;
            let p_t = exp(min(score - lse[lse_idx], 0.0) - lse[lse_idx + 1u]);

            // dS = P * (dP - row_sum)
            let ds_t = p_t * (dp_t - row_sum);

            // Accumulate both dK and dV
            my_dk += ds_t * scale * src_a[q_base + tid];
            my_dv += p_t * do_val;
        }
    }

    dst[kv_base + tid] = my_dk;
    dst2[kv_base + tid] = my_dv;
}
