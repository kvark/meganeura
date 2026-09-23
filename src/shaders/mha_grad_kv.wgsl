// Fused MHA gradient wrt K and V (Flash Attention 2 style).
//
// Computes dK and dV in a single pass over Q positions, sharing the
// recomputed Q·K score. Saves one dispatch + one score recomputation
// per attention layer compared to separate GradK + GradV.
//
// Dispatch: [kv_seq, num_kv_heads, 1], WG=64. Lane `tid` owns dimensions
// `tid + 64 * c` for c < MAX_CHUNKS, so heads up to 256 wide are covered.
// Dimensions past head_dim are zero-padded and must not touch storage.
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

const LANES: u32 = 64u;
const MAX_CHUNKS: u32 = 4u;

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
    var k_val: array<f32, MAX_CHUNKS>;
    var v_val: array<f32, MAX_CHUNKS>;
    var my_dk: array<f32, MAX_CHUNKS>;
    var my_dv: array<f32, MAX_CHUNKS>;
    for (var c = 0u; c < MAX_CHUNKS; c++) {
        let dim = tid + c * LANES;
        k_val[c] = 0.0;
        v_val[c] = 0.0;
        my_dk[c] = 0.0;
        my_dv[c] = 0.0;
        if dim < head_dim {
            k_val[c] = src_b[kv_base + dim];
            v_val[c] = bias[kv_base + dim];
        }
    }

    let start_pos = select(0u, t, kv_seq == 0u);
    let window = params.window_size;
    let end_pos = select(q_seq, min(q_seq, t + window), window > 0u);
    for (var pos = start_pos; pos < end_pos; pos++) {
        for (var head_rel = 0u; head_rel < heads_per_kv; head_rel++) {
            let head = kv_head * heads_per_kv + head_rel;
            let q_base = pos * q_dim + head * head_dim;
            var q_val: array<f32, MAX_CHUNKS>;
            var do_val: array<f32, MAX_CHUNKS>;
            var qk = 0.0;
            var doo = 0.0;
            var dov = 0.0;
            for (var c = 0u; c < MAX_CHUNKS; c++) {
                let dim = tid + c * LANES;
                q_val[c] = 0.0;
                do_val[c] = 0.0;
                if dim < head_dim {
                    q_val[c] = src_a[q_base + dim];
                    do_val[c] = d_out[q_base + dim];
                    qk += q_val[c] * k_val[c];
                    doo += do_val[c] * fwd_dst[q_base + dim];
                    dov += do_val[c] * v_val[c];
                }
            }

            // Fused triple reduction: Q·K, dO·O, dO·V
            wg_a[tid] = qk;
            wg_b[tid] = doo;
            wg_c[tid] = dov;
            triple_tree_reduce(tid);
            let score = wg_a[0] * scale;
            let row_sum = wg_b[0];
            let dp_t = wg_c[0];
            // The next head/query iteration reuses all three arrays.
            // Ensure every lane has captured their reduced values first.
            workgroupBarrier();

            // Softmax probability
            let lse_idx = (pos * num_heads + head) * 2u;
            let p_t = exp(min(score - lse[lse_idx], 0.0) - lse[lse_idx + 1u]);

            // dS = P * (dP - row_sum)
            let ds_t = p_t * (dp_t - row_sum);

            // Accumulate both dK and dV
            for (var c = 0u; c < MAX_CHUNKS; c++) {
                my_dk[c] += ds_t * scale * q_val[c];
                my_dv[c] += p_t * do_val[c];
            }
        }
    }

    for (var c = 0u; c < MAX_CHUNKS; c++) {
        let dim = tid + c * LANES;
        if dim < head_dim {
            dst[kv_base + dim] = my_dk[c];
            dst2[kv_base + dim] = my_dv[c];
        }
    }
}
