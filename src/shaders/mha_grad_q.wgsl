// MHA gradient wrt Q — recomputes Q·K scores (no score buffer)
// Dispatch: [q_seq, num_heads, 1], WG=64

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
var<storage> lse: array<f32>;     // LSE from forward (max_score, log_sum only)
var<storage> fwd_dst: array<f32>; // O from forward
var<storage, read_write> dst: array<f32>;  // dQ
var<uniform> params: Params;
var<workgroup> wg_a: array<f32, 64>;
var<workgroup> wg_b: array<f32, 64>;

// Fused dual tree_reduce: reduces wg_a and wg_b simultaneously,
// saving 7 barriers vs doing them sequentially.
fn dual_tree_reduce(tid: u32) {
    workgroupBarrier();
    if tid < 32u { wg_a[tid] += wg_a[tid + 32u]; wg_b[tid] += wg_b[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u { wg_a[tid] += wg_a[tid + 16u]; wg_b[tid] += wg_b[tid + 16u]; }
    workgroupBarrier();
    if tid < 8u { wg_a[tid] += wg_a[tid + 8u]; wg_b[tid] += wg_b[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u { wg_a[tid] += wg_a[tid + 4u]; wg_b[tid] += wg_b[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u { wg_a[tid] += wg_a[tid + 2u]; wg_b[tid] += wg_b[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u { wg_a[tid] += wg_a[tid + 1u]; wg_b[tid] += wg_b[tid + 1u]; }
    workgroupBarrier();
}

fn tree_reduce_a(tid: u32) {
    workgroupBarrier();
    if tid < 32u { wg_a[tid] += wg_a[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u { wg_a[tid] += wg_a[tid + 16u]; }
    workgroupBarrier();
    if tid < 8u { wg_a[tid] += wg_a[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u { wg_a[tid] += wg_a[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u { wg_a[tid] += wg_a[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u { wg_a[tid] += wg_a[tid + 1u]; }
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
    wg_a[tid] = do_val * fwd_dst[q_base + tid];
    tree_reduce_a(tid);
    let row_sum = wg_a[0];

    var my_dq = 0.0;

    let kv_len = select(kv_seq, pos + 1u, kv_seq == 0u);
    let window = params.window_size;
    let kv_start = select(0u, select(0u, pos + 1u - window, pos >= window), window > 0u);

    for (var t = kv_start; t < kv_len; t++) {
        let k_base = t * kv_dim + kv_head_off;

        // Fused: compute Q·K score AND dO·V in one reduction pass
        wg_a[tid] = q_val * src_b[k_base + tid];
        wg_b[tid] = do_val * bias[k_base + tid];
        dual_tree_reduce(tid);
        let score = wg_a[0] * scale;
        let dp_t = wg_b[0];

        // P_t = exp(score - max_score) / sum_exp
        let p_t = exp(min(score - max_s, 0.0) - log_sum);

        // dS_t = P_t * (dP_t - row_sum)
        let ds_t = p_t * (dp_t - row_sum);

        // Accumulate dQ
        my_dq += ds_t * scale * src_b[k_base + tid];
    }

    dst[q_base + tid] = my_dq;
}
