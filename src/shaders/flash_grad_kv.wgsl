// @section main
$PARAMS
var<storage> d_out: array<f32>;
var<storage> src_a: array<f32>;
var<storage> src_b: array<f32>;
var<storage> bias: array<f32>;
var<storage> lse: array<f32>;
var<storage> fwd_dst: array<f32>;
var<storage, read_write> dst: array<f32>;
var<storage, read_write> dst2: array<f32>;
var<uniform> params: Params;

$SHARED
$REDUCTION
@compute @workgroup_size($THREADS)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let ki = lid.x / $TPQu;
    let lane = lid.x % $TPQu;
    let d_base = lane * $EPTu;
    let t = wgid.x * $BKVu + ki;
    let kv_head = wgid.y;
    let q_seq = params.q_seq;
    let kv_seq = params.kv_seq;
    let num_heads = params.packed_heads >> 16u;
    let num_kv_heads = params.packed_heads & 0xFFFFu;
    let head_dim = params.head_dim;

    let effective_kv_seq = select(kv_seq, q_seq, kv_seq == 0u);
    let valid = t < effective_kv_seq && kv_head < num_kv_heads;
    let heads_per_kv = num_heads / max(num_kv_heads, 1u);
    let kv_dim = num_kv_heads * head_dim;
    let q_dim = num_heads * head_dim;
    let kv_base = t * kv_dim + kv_head * head_dim;
    let scale = inverseSqrt(f32(head_dim));

$KV_INIT
    if valid {
$KV_LOAD
    }

$GRAD_INIT

    let start_pos = select(0u, t, kv_seq == 0u);
    let window = params.window_size;
    let end_pos = select(q_seq, min(q_seq, t + window), window > 0u);

    let first_t = wgid.x * $BKVu;
    let last_t = min(wgid.x * $BKVu + $BKVu - 1u, effective_kv_seq - 1u);
    let wg_start = select(0u, first_t, kv_seq == 0u);
    let wg_end = select(q_seq, min(q_seq, last_t + window), window > 0u);

$QUERY_LOOP

    if valid {
$STORE
    }
}

// @section direct_query_loop
    for (var pos = wg_start; pos < wg_end; pos++) {
        for (var head_rel = 0u; head_rel < heads_per_kv; head_rel++) {
            let head = kv_head * heads_per_kv + head_rel;
            let q_base = pos * q_dim + head * head_dim;

$Q_LOAD
            var score_part = 0.0;
            var dp_part = 0.0;
$DOT
            let row_sum = fwd_dst[pos * num_heads + head];
$SCORE
            if valid && pos >= start_pos && pos < end_pos {
                let lse_idx = (pos * num_heads + head) * 2u;
                let p_t = exp(min(score - lse[lse_idx], 0.0) - lse[lse_idx + 1u]);
                let ds_t = p_t * (dp_t - row_sum);
                let w_dk = ds_t * scale;
$ACCUMULATE
            }
        }
    }

// @section kv_init
    var k$INDEX = 0.0;
    var v$INDEX = 0.0;

// @section kv_load
        k$INDEX = src_b[kv_base + d_base + $INDEXu];
        v$INDEX = bias[kv_base + d_base + $INDEXu];

// @section grad_init
    var dk$INDEX = 0.0;
    var dv$INDEX = 0.0;

// @section q_load_direct
            let q$INDEX = src_a[q_base + $INDEXu];
            let do$INDEX = d_out[q_base + $INDEXu];

// @section dot
            score_part += q$INDEX * k$INDEX;
            dp_part += do$INDEX * v$INDEX;

// @section accumulate
                dk$INDEX += w_dk * q$INDEX;
                dv$INDEX += p_t * do$INDEX;

// @section store
        dst[kv_base + d_base + $INDEXu] = dk$INDEX;
        dst2[kv_base + d_base + $INDEXu] = dv$INDEX;

// @section reduce_step
    workgroupBarrier();
    if local < $STRIDEu {
        for (var qi = 0u; qi < $QUERY_TILEu; qi++) {
            let offset = qi * $THREADSu + base + local;
            wg_score[offset] += wg_score[offset + $STRIDEu];
            wg_dp[offset] += wg_dp[offset + $STRIDEu];
        }
    }

// @section shared
var<workgroup> shared_q: array<f32, $QUERY_ELEMENTS>;
var<workgroup> shared_do: array<f32, $QUERY_ELEMENTS>;

// @section reduction
var<workgroup> wg_score: array<f32, $GROUP_ELEMENTS>;
var<workgroup> wg_dp: array<f32, $GROUP_ELEMENTS>;

fn reduce_pair(tid: u32) {
    let local = tid % $TPQu;
    let base = (tid / $TPQu) * $TPQu;
$REDUCE_STEP
    workgroupBarrier();
}

// @section direct_score
            let score = score_part * scale;
            let dp_t = dp_part;

// @section tiled_query_loop
    let query_end = wg_end * heads_per_kv;
    for (var first = wg_start * heads_per_kv; first < query_end; first += $QUERY_TILEu) {
        for (var flat = lid.x; flat < $QUERY_ELEMENTSu; flat += $THREADSu) {
            let qi = flat / $HEAD_DIMu;
            let d = flat % $HEAD_DIMu;
            let query = first + qi;
            let pos = query / heads_per_kv;
            let head = kv_head * heads_per_kv + query % heads_per_kv;
            let q_base = pos * q_dim + head * head_dim;
            var q_value = 0.0;
            var do_value = 0.0;
            if query < query_end {
                q_value = src_a[q_base + d];
                do_value = d_out[q_base + d];
            }
            shared_q[flat] = q_value;
            shared_do[flat] = do_value;
        }
        workgroupBarrier();
        for (var qi = 0u; qi < $QUERY_TILEu; qi++) {
            let q_offset = qi * $HEAD_DIMu + d_base;
$TILE_Q_LOAD
            var score_part = 0.0;
            var dp_part = 0.0;
$DOT
            wg_score[qi * $THREADSu + lid.x] = score_part;
            wg_dp[qi * $THREADSu + lid.x] = dp_part;
        }
        reduce_pair(lid.x);
        for (var qi = 0u; qi < $QUERY_TILEu; qi++) {
            let query = first + qi;
            let pos = query / heads_per_kv;
            let head = kv_head * heads_per_kv + query % heads_per_kv;
            let score = wg_score[qi * $THREADSu + ki * $TPQu] * scale;
            let dp_t = wg_dp[qi * $THREADSu + ki * $TPQu];
            if valid && query < query_end && pos >= start_pos && pos < end_pos {
                let row_sum = fwd_dst[pos * num_heads + head];
                let lse_idx = (pos * num_heads + head) * 2u;
                let p_t = exp(min(score - lse[lse_idx], 0.0) - lse[lse_idx + 1u]);
                let w_dk = p_t * (dp_t - row_sum) * scale;
                let q_offset = qi * $HEAD_DIMu + d_base;
$TILE_Q_LOAD
$ACCUMULATE
            }
        }
        workgroupBarrier();
    }

// @section tile_q_load
            let q$INDEX = shared_q[q_offset + $INDEXu];
            let do$INDEX = shared_do[q_offset + $INDEXu];
