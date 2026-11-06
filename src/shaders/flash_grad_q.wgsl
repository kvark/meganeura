// @section main
$ENABLE
$PARAMS
var<storage> d_out: array<$ELEM>;
var<storage> src_a: array<$ELEM>;
var<storage> src_b: array<$ELEM>;
var<storage> bias: array<$ELEM>;
var<storage> lse: array<f32>;
var<storage> fwd_dst: array<f32>;
var<storage, read_write> dst: array<$ELEM>;
var<uniform> params: Params;
var<workgroup> shared_k: array<$STILE, $SHARED_LEN>;
var<workgroup> shared_v: array<$STILE, $SHARED_LEN>;
$LANE_SUM

$REDUCTION
@compute @workgroup_size($THREADS)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>$SUBGROUP_ARG) {
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

// @section tile_dot_v4
            score_part += dot(q$INDEX, shared_k[(k_off) / 4u + $OFFSETu]);
            dp_part += dot(do$INDEX, shared_v[(k_off) / 4u + $OFFSETu]);

// @section tile_dot_v4_f16
            score_part += dot(vec4<f32>(q$INDEX), vec4<f32>(shared_k[(k_off) / 4u + $OFFSETu]));
            dp_part += dot(do$INDEX, vec4<f32>(shared_v[(k_off) / 4u + $OFFSETu]));

// @section tile_accumulate
                dq$INDEX += w * shared_k[k_off + $INDEXu];

// @section tile_accumulate_v4
                dq$INDEX += w * shared_k[(k_off) / 4u + $OFFSETu];

// @section tile_accumulate_v4_f16
                dq$INDEX += w * vec4<f32>(shared_k[(k_off) / 4u + $OFFSETu]);

// @section tail_dot
        sp2 += q$INDEX * shared_k[d_base + $INDEXu];
        dp2 += do$INDEX * shared_v[d_base + $INDEXu];

// @section tail_dot_v4
        sp2 += dot(q$INDEX, shared_k[d_base / 4u + $OFFSETu]);
        dp2 += dot(do$INDEX, shared_v[d_base / 4u + $OFFSETu]);

// @section tail_dot_v4_f16
        sp2 += dot(vec4<f32>(q$INDEX), vec4<f32>(shared_k[d_base / 4u + $OFFSETu]));
        dp2 += dot(do$INDEX, vec4<f32>(shared_v[d_base / 4u + $OFFSETu]));

// @section tail_accumulate
            dq$INDEX += w * shared_k[d_base + $INDEXu];

// @section tail_accumulate_v4
            dq$INDEX += w * shared_k[d_base / 4u + $OFFSETu];

// @section tail_accumulate_v4_f16
            dq$INDEX += w * vec4<f32>(shared_k[d_base / 4u + $OFFSETu]);

// @section store
        dst[q_base + d_base + $INDEXu] = dq$INDEX;

// @section store_v4
        dst[(q_base + d_base) / 4u + $OFFSETu] = dq$INDEX;

// @section q_init_v4
    var q$INDEX = vec4<f32>(0.0);
    var do$INDEX = vec4<f32>(0.0);

// @section q_init_v4_f16
    var q$INDEX = vec4<f16>(0.0);
    var do$INDEX = vec4<f32>(0.0);

// @section q_load_v4
        q$INDEX = src_a[(q_base + d_base) / 4u + $OFFSETu];
        do$INDEX = d_out[(q_base + d_base) / 4u + $OFFSETu];

// @section q_load_v4_f16
        q$INDEX = vec4<f16>(src_a[(q_base + d_base) / 4u + $OFFSETu]);
        do$INDEX = d_out[(q_base + d_base) / 4u + $OFFSETu];

// @section dq_init_v4
    var dq$INDEX = vec4<f32>(0.0);

// @section reduce_step
    workgroupBarrier();
    if local < $STRIDEu {
        for (var i = 0u; i < $BKVu; i++) {
            let offset = i * $THREADSu + base + local;
            wg_score[offset] += wg_score[offset + $STRIDEu];
            wg_dp[offset] += wg_dp[offset + $STRIDEu];
        }
    }

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
var<workgroup> wg_score: array<f32, $GROUP_ELEMENTS>;
var<workgroup> wg_dp: array<f32, $GROUP_ELEMENTS>;
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
$TILED_TAIL_LOAD
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
    for (var t = min_kv_start; t < max_kv_len; t += $BKVu) {
$GROUP_LOAD
        workgroupBarrier();

        if $USE_SUBGROUP && (subgroup_size >= $TPQu && subgroup_size % $TPQu == 0u) {
            for (var i = 0u; i < $BKVu; i++) {
                let kv_pos = t + i;
                let k_off = i * $HEAD_DIMu + d_base;
                var score_part = 0.0;
                var dp_part = 0.0;
$TILE_DOT
                let score = lane_sum(score_part) * scale;
                let dp_t = lane_sum(dp_part);
                if valid && kv_pos >= my_kv_start && kv_pos < my_kv_len {
                    let p_t = exp(min(score - max_s, 0.0) - log_sum);
                    let w = p_t * (dp_t - row_sum) * scale;
$TILE_ACCUMULATE
                }
            }
        } else {
            for (var i = 0u; i < $BKVu; i++) {
                let k_off = i * $HEAD_DIMu + d_base;
                var score_part = 0.0;
                var dp_part = 0.0;
$TILE_DOT
                wg_score[i * $THREADSu + lid.x] = score_part;
                wg_dp[i * $THREADSu + lid.x] = dp_part;
            }
            reduce_score_dp(lid.x);
            for (var i = 0u; i < $BKVu; i++) {
                let kv_pos = t + i;
                let k_off = i * $HEAD_DIMu + d_base;
                let score = wg_score[i * $THREADSu + qi * $TPQu] * scale;
                let dp_t = wg_dp[i * $THREADSu + qi * $TPQu];
                if valid && kv_pos >= my_kv_start && kv_pos < my_kv_len {
                    let p_t = exp(min(score - max_s, 0.0) - log_sum);
                    let w = p_t * (dp_t - row_sum) * scale;
$TILE_ACCUMULATE
                }
            }
        }
        workgroupBarrier();
    }

// @section grouped_kv_loop_subgroup
    for (var t = min_kv_start; t < max_kv_len; t += $BKVu) {
$GROUP_LOAD
        workgroupBarrier();
        // The reported subgroup width is only a default. A shader that
        // actually runs narrower falls back to a per-key shared sum.
        if $USE_SUBGROUP && (subgroup_size >= $TPQu && subgroup_size % $TPQu == 0u) {
            for (var i = 0u; i < $BKVu; i++) {
                let kv_pos = t + i;
                let k_off = i * $HEAD_DIMu + d_base;
                var score_part = 0.0;
                var dp_part = 0.0;
$TILE_DOT
                let score = lane_sum(score_part) * scale;
                let dp_t = lane_sum(dp_part);
                if valid && kv_pos >= my_kv_start && kv_pos < my_kv_len {
                    let p_t = exp(min(score - max_s, 0.0) - log_sum);
                    let w = p_t * (dp_t - row_sum) * scale;
$TILE_ACCUMULATE
                }
            }
        } else {
            for (var i = 0u; i < $BKVu; i++) {
                let kv_pos = t + i;
                let k_off = i * $HEAD_DIMu + d_base;
                var score_part = 0.0;
                var dp_part = 0.0;
$TILE_DOT
                wg_part_score[lid.x] = score_part;
                wg_part_dp[lid.x] = dp_part;
                workgroupBarrier();
                if (lane == 0u) {
                    var score_sum = 0.0;
                    var dp_sum = 0.0;
                    for (var s = 0u; s < $TPQu; s++) {
                        let part = qi * $TPQu + s;
                        score_sum += wg_part_score[part];
                        dp_sum += wg_part_dp[part];
                    }
                    wg_part_score[qi * $TPQu] = score_sum;
                    wg_part_dp[qi * $TPQu] = dp_sum;
                }
                workgroupBarrier();
                let score = wg_part_score[qi * $TPQu] * scale;
                let dp_t = wg_part_dp[qi * $TPQu];
                if valid && kv_pos >= my_kv_start && kv_pos < my_kv_len {
                    let p_t = exp(min(score - max_s, 0.0) - log_sum);
                    let w = p_t * (dp_t - row_sum) * scale;
$TILE_ACCUMULATE
                }
                workgroupBarrier();
            }
        }
        workgroupBarrier();
    }
