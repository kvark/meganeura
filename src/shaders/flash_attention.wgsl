// @section main
$PARAMS
var<storage> src_a: array<f32>;
var<storage> src_b: array<f32>;
var<storage> bias: array<f32>;
$CACHED_BINDING
var<storage, read_write> dst: array<f32>;
$LSE_BINDING
var<uniform> params: Params;
var<workgroup> shared_k: array<f32, $TILE_ELEMENTS>;
var<workgroup> shared_v: array<f32, $TILE_ELEMENTS>;
var<workgroup> wg_scores: array<f32, $SCORES>;
var<workgroup> wg_dot: array<f32, $THREADS>;

fn tree_reduce_bkv_grouped(tid: u32) {
    let qi = tid / $TPQu;
    let local = tid % $TPQu;
    let base = qi * $TPQu;
$SCORE_STEP
    workgroupBarrier();
}

fn tree_reduce_grouped(tid: u32) {
    let qi = tid / $TPQu;
    let local = tid % $TPQu;
    let base = qi * $TPQu;
$DOT_STEP
    workgroupBarrier();
}

@compute @workgroup_size($THREADS)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let qi = lid.x / $TPQu;
    let lane = lid.x % $TPQu;
    let d_base = lane * $D_BASE_STRIDEu;
    let pos = wgid.x * $BQu + qi;
    let head = wgid.y;
    let q_seq = params.q_seq;
$DIMENSIONS
    let head_dim = params.head_dim;
    let valid = pos < q_seq && head < num_heads;

    let my_kv_len = select(kv_seq, select(pos + 1u, 0u, !valid), kv_seq == 0u);
    let window_size = $WINDOW;
    let my_kv_start = select(0u, my_kv_len - min(my_kv_len, window_size), window_size > 0u);

    let last_pos = min(wgid.x * $BQu + $BQu - 1u, q_seq - 1u);
    let first_pos = wgid.x * $BQu;
    let max_kv_len = select(kv_seq, last_pos + 1u, kv_seq == 0u);
    let first_kv_len = select(kv_seq, first_pos + 1u, kv_seq == 0u);
    let min_kv_start = select(0u, first_kv_len - min(first_kv_len, window_size), window_size > 0u);

    let kv_head = head / (num_heads / max(num_kv_heads, 1u));
    let kv_head_off = kv_head * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let scale = inverseSqrt(f32(head_dim));
$Q_INIT
    if valid {
        let q_base = pos * (num_heads * head_dim) + head * head_dim;
$Q_LOAD
    }

$OUT_INIT
    var max_score = -1e30;
    var sum_exp = 0.0;

    let kv_range = max_kv_len - min_kv_start;
    let tile_end = min_kv_start + (kv_range / $BKVu) * $BKVu;
    var t = min_kv_start;
    for (; t < tile_end; t += $BKVu) {
$TILE_LOAD
        workgroupBarrier();

        let grp_base = qi * $TPQu;
        for (var i = 0u; i < $BKVu; i++) {
            var pdot = 0.0;
$TILE_DOT
            wg_scores[i * $THREADSu + grp_base + lane] = pdot;
        }
        tree_reduce_bkv_grouped(lid.x);

        for (var i = 0u; i < $BKVu; i++) {
            let kv_pos = t + i;
            if valid && kv_pos >= my_kv_start && kv_pos < my_kv_len {
                let score = wg_scores[i * $THREADSu + grp_base] * scale;
                let new_max = max(max_score, score);
                let correction = exp(max_score - new_max);
                let weight = exp(score - new_max);
                sum_exp = sum_exp * correction + weight;
$TILE_ACCUMULATE
                max_score = new_max;
            }
        }
        workgroupBarrier();
    }

    for (; t < max_kv_len; t++) {
        for (var d = lid.x; d < $HEAD_DIMu; d += $THREADSu) {
            shared_k[d] = src_b[t * kv_dim + kv_head_off + d];
        }
        workgroupBarrier();

        let dot_base = qi * $TPQu;
        var pdot2 = 0.0;
$TAIL_DOT
        wg_dot[dot_base + lane] = pdot2;
        tree_reduce_grouped(lid.x);
        let score = wg_dot[qi * $TPQu] * scale;

        if valid && t >= my_kv_start && t < my_kv_len {
            let new_max = max(max_score, score);
            let correction = exp(max_score - new_max);
            let weight = exp(score - new_max);
            sum_exp = sum_exp * correction + weight;
            let v_base2 = t * kv_dim + kv_head_off;
$TAIL_ACCUMULATE
            max_score = new_max;
        }
        workgroupBarrier();
    }

    if valid {
        let q_base = pos * (num_heads * head_dim) + head * head_dim;
        let safe_sum = select(sum_exp, 1.0, sum_exp == 0.0);
$OUT_STORE
$LSE_STORE
    }
}

// @section cached_binding
var<storage> kv_pos_buf: array<u32>;

// @section lse_binding
var<storage, read_write> lse: array<f32>;

// @section cached_dimensions
    let kv_seq = kv_pos_buf[0] + 1u;
    let num_heads = params.num_heads;
    let num_kv_heads = params.num_kv_heads;

// @section dimensions
    let kv_seq = params.kv_seq;
    let num_heads = params.packed_heads >> 16u;
    let num_kv_heads = params.packed_heads & 0xFFFFu;

// @section lse_store
        if lane == 0u {
            let idx = (pos * num_heads + head) * 2u;
            lse[idx] = max_score;
            lse[idx + 1u] = select(log(sum_exp), -1e30, sum_exp == 0.0);
        }

// @section score_step
    workgroupBarrier();
    if local < $STRIDEu {
        for (var i = 0u; i < $BKVu; i++) {
            wg_scores[i * $THREADSu + base + local] += wg_scores[i * $THREADSu + base + local + $STRIDEu];
        }
    }

// @section dot_step
    workgroupBarrier();
    if local < $STRIDEu { wg_dot[base + local] += wg_dot[base + local + $STRIDEu]; }

// @section tile_load_first
        if lid.x < $TILE_ELEMENTSu {
            let ki = lid.x / $HEAD_DIMu;
            shared_k[lid.x] = src_b[(t + ki) * kv_dim + kv_head_off + (lid.x % head_dim)];
            shared_v[lid.x] = bias[(t + ki) * kv_dim + kv_head_off + (lid.x % head_dim)];
        }

// @section tile_load_next
        if lid.x + $OFFSETu < $TILE_ELEMENTSu {
            let ki2 = (lid.x + $OFFSETu) / $HEAD_DIMu;
            shared_k[lid.x + $OFFSETu] = src_b[(t + ki2) * kv_dim + kv_head_off + ((lid.x + $OFFSETu) % head_dim)];
            shared_v[lid.x + $OFFSETu] = bias[(t + ki2) * kv_dim + kv_head_off + ((lid.x + $OFFSETu) % head_dim)];
        }

// @section q_init
    var q$INDEX = 0.0;

// @section q_load
        q$INDEX = src_a[q_base + d_base + $OFFSETu];

// @section out_init
    var out$INDEX = 0.0;

// @section tile_dot
            pdot += q$INDEX * shared_k[i * $HEAD_DIMu + d_base + $OFFSETu];

// @section tile_accumulate
                out$INDEX = out$INDEX * correction + weight * shared_v[i * $HEAD_DIMu + d_base + $OFFSETu];

// @section tail_dot
        pdot2 += q$INDEX * shared_k[d_base + $OFFSETu];

// @section tail_accumulate
            out$INDEX = out$INDEX * correction + weight * bias[v_base2 + d_base + $OFFSETu];

// @section out_store
        dst[q_base + d_base + $OFFSETu] = out$INDEX / safe_sum;
