// Parallel attention: online softmax single-pass with GQA
// Variants: causal (kv_len=pos+1), full (kv_len=q_seq), cross (kv_len=kv_seq)

struct Params {
    $PARAM_FIELDS
}

var<storage> src_a: array<f32>;  // Q
var<storage> src_b: array<f32>;  // K
var<storage> bias: array<f32>;   // V
var<storage, read_write> dst: array<f32>;
var<storage, read_write> lse: array<f32>;  // log-sum-exp for backward
var<uniform> params: Params;
var<workgroup> wg_dot: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let pos = wgid.x;
    let head = wgid.y;
    let tid = lid.x;

    $PARSE_PARAMS

    if pos >= q_seq || head >= num_heads { return; }

    let kv_head = head / (num_heads / num_kv_heads);
    let kv_head_off = kv_head * head_dim;
    let scale = inverseSqrt(f32(head_dim));
    let q_base = pos * (num_heads * head_dim) + head * head_dim;
    let q_val = src_a[q_base + tid];

    var my_out = 0.0;
    var max_score = -1e30;
    var sum_exp = 0.0;

    for (var t = 0u; t < kv_len; t++) {
        let k_base = t * (num_kv_heads * head_dim) + kv_head_off;

        // Parallel dot product Q·K
        wg_dot[tid] = q_val * src_b[k_base + tid];
        workgroupBarrier();

        // Tree reduction
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

        let score = wg_dot[0] * scale;

        // Online softmax update
        let new_max = max(max_score, score);
        let correction = exp(max_score - new_max);
        let weight = exp(score - new_max);
        sum_exp = sum_exp * correction + weight;
        my_out = my_out * correction + weight * bias[k_base + tid];
        max_score = new_max;
    }

    let safe_sum = select(sum_exp, 1.0, sum_exp == 0.0);
    dst[q_base + tid] = my_out / safe_sum;

    // Store log-sum-exp for backward pass (one scalar per pos×head).
    if tid == 0u {
        lse[pos * num_heads + head] = select(max_score + log(sum_exp), -1e30, sum_exp == 0.0);
    }
}
