// @section main
$PARAMS
var<storage> d_out: array<f32>;
var<storage> src_a: array<f32>;
var<storage> src_b: array<f32>;
var<storage> bias: array<f32>;
var<storage> lse: array<f32>;
var<storage> fwd_dst: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;
var<workgroup> shared_k: array<f32, $TILE_ELEMENTS>;
var<workgroup> shared_v: array<f32, $TILE_ELEMENTS>;

$REDUCTION
@compute @workgroup_size($THREADS)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let qi = lid.x / $TPQu;
    let lane = lid.x % $TPQu;
    let d_base = lane * $EPTu;
    let pos = wgid.x * $BQu + qi;
    let head = wgid.y;
    let q_seq = params.q_seq;
    let kv_seq = params.kv_seq;
    let num_heads = params.packed_heads >> 16u;
    let num_kv_heads = params.packed_heads & 0xFFFFu;
    let head_dim = params.head_dim;
    let valid = pos < q_seq && head < num_heads;

    let kv_head = head / (num_heads / max(num_kv_heads, 1u));
    let kv_head_off = kv_head * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let scale = inverseSqrt(f32(head_dim));
    var max_s = 0.0;
    var log_sum = 0.0;
    var q_base = 0u;
$Q_INIT
    if valid {
        q_base = pos * (num_heads * head_dim) + head * head_dim;
$Q_LOAD
        let lse_idx = (pos * num_heads + head) * 2u;
        max_s = lse[lse_idx];
        log_sum = lse[lse_idx + 1u];
    }

    var row_sum = 0.0;
    if valid {
        row_sum = fwd_dst[pos * num_heads + head];
    }

    let my_kv_len = select(kv_seq, select(pos + 1u, 0u, !valid), kv_seq == 0u);
    let window = params.window_size;
    let my_kv_start = select(0u, my_kv_len - min(my_kv_len, window), window > 0u);
    let last_pos = min(wgid.x * $BQu + $BQu - 1u, q_seq - 1u);
    let first_pos = wgid.x * $BQu;
    let max_kv_len = select(kv_seq, last_pos + 1u, kv_seq == 0u);
    let first_kv_len = select(kv_seq, first_pos + 1u, kv_seq == 0u);
    let min_kv_start = select(0u, first_kv_len - min(first_kv_len, window), window > 0u);

$DQ_INIT

$KV_LOOP
    if valid {
$STORE
    }
}

// @section q_init
    var q$INDEX = 0.0;
    var do$INDEX = 0.0;

// @section q_load
        q$INDEX = src_a[q_base + d_base + $INDEXu];
        do$INDEX = d_out[q_base + d_base + $INDEXu];

// @section dq_init
    var dq$INDEX = 0.0;

// @section tile_dot
            score_part += q$INDEX * shared_k[k_off + $INDEXu];
            dp_part += do$INDEX * shared_v[k_off + $INDEXu];

// @section tile_accumulate
                dq$INDEX += w * shared_k[k_off + $INDEXu];

// @section tail_dot
        sp2 += q$INDEX * shared_k[d_base + $INDEXu];
        dp2 += do$INDEX * shared_v[d_base + $INDEXu];

// @section tail_accumulate
            dq$INDEX += w * shared_k[d_base + $INDEXu];

// @section kv_load
        let k$INDEX = shared_k[d_base + $INDEXu];
        let v$INDEX = shared_v[d_base + $INDEXu];

// @section dot
        score_part += q$INDEX * k$INDEX;
        dp_part += do$INDEX * v$INDEX;

// @section accumulate
            dq$INDEX += w * k$INDEX;

// @section store
        dst[q_base + d_base + $INDEXu] = dq$INDEX;

// @section reduce_step
    workgroupBarrier();
    if local < $STRIDEu { wg_score[base + local] += wg_score[base + local + $STRIDEu]; wg_dp[base + local] += wg_dp[base + local + $STRIDEu]; }

// @section tile_load_first
        if lid.x < $TILE_ELEMENTSu {
            let ki = lid.x / $HEAD_DIMu;
            let kd = lid.x % $HEAD_DIMu;
            let kb = (t + ki) * kv_dim + kv_head_off;
            shared_k[lid.x] = src_b[kb + kd];
            shared_v[lid.x] = bias[kb + kd];
        }

// @section tile_load_next
        if lid.x + $OFFSETu < $TILE_ELEMENTSu {
            let ki = (lid.x + $OFFSETu) / $HEAD_DIMu;
            let kd = (lid.x + $OFFSETu) % $HEAD_DIMu;
            let kb = (t + ki) * kv_dim + kv_head_off;
            shared_k[lid.x + $OFFSETu] = src_b[kb + kd];
            shared_v[lid.x + $OFFSETu] = bias[kb + kd];
        }

// @section reduction
var<workgroup> wg_score: array<f32, $THREADS>;
var<workgroup> wg_dp: array<f32, $THREADS>;
fn reduce_score_dp(tid: u32) {
    let local = tid % $TPQu;
    let base = (tid / $TPQu) * $TPQu;
$REDUCE_STEP
    workgroupBarrier();
}

// @section tiled_kv_loop
    let kv_range = max_kv_len - min_kv_start;
    let tile_end = min_kv_start + (kv_range / $BKVu) * $BKVu;
    var t = min_kv_start;
    for (; t < tile_end; t += $BKVu) {
$TILE_LOAD
        workgroupBarrier();

        for (var i = 0u; i < $BKVu; i++) {
            let kv_pos = t + i;
            let k_off = i * $HEAD_DIMu + d_base;
            var score_part = 0.0;
            var dp_part = 0.0;
$TILE_DOT
            let score = score_part * scale;
            if valid && kv_pos >= my_kv_start && kv_pos < my_kv_len {
                let p_t = exp(min(score - max_s, 0.0) - log_sum);
                let ds_t = p_t * (dp_part - row_sum);
                let w = ds_t * scale;
$TILE_ACCUMULATE
            }
        }
        workgroupBarrier();
    }

    for (; t < max_kv_len; t++) {
        let k_base = t * kv_dim + kv_head_off;
        for (var d = lid.x; d < $HEAD_DIMu; d += $THREADSu) {
            shared_k[d] = src_b[k_base + d];
            shared_v[d] = bias[k_base + d];
        }
        workgroupBarrier();
        var sp2 = 0.0;
        var dp2 = 0.0;
$TAIL_DOT
        let score2 = sp2 * scale;
        if valid && t >= my_kv_start && t < my_kv_len {
            let p_t = exp(min(score2 - max_s, 0.0) - log_sum);
            let ds_t = p_t * (dp2 - row_sum);
            let w = ds_t * scale;
$TAIL_ACCUMULATE
        }
        workgroupBarrier();
    }

// @section grouped_kv_loop
    for (var t = min_kv_start; t < max_kv_len; t++) {
        let k_base = t * kv_dim + kv_head_off;
        for (var d = lid.x; d < $HEAD_DIMu; d += $THREADSu) {
            shared_k[d] = src_b[k_base + d];
            shared_v[d] = bias[k_base + d];
        }
        workgroupBarrier();

$KV_LOAD
        var score_part = 0.0;
        var dp_part = 0.0;
$DOT
        wg_score[qi * $TPQu + lane] = score_part;
        wg_dp[qi * $TPQu + lane] = dp_part;
        reduce_score_dp(lid.x);
        let score = wg_score[qi * $TPQu] * scale;
        let dp_t = wg_dp[qi * $TPQu];

        if valid && t >= my_kv_start && t < my_kv_len {
            let p_t = exp(min(score - max_s, 0.0) - log_sum);
            let ds_t = p_t * (dp_t - row_sum);
            let w = ds_t * scale;
$ACCUMULATE
        }
        workgroupBarrier();
    }
