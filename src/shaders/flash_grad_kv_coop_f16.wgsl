// K Q^T and V dO^T use f16 cooperative operands. The scalar dS Q and P dO
// products use untransposed f32 copies. Keep dK/dV in per-thread registers:
// cooperative accumulators spanning the query loop caused a shared-memory race.
enable f16;
enable wgpu_cooperative_matrix;

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

var<workgroup> shared_k: array<f16, $TILE_ELEMENTS>;
var<workgroup> shared_v: array<f16, $TILE_ELEMENTS>;
var<workgroup> shared_q: array<f32, $TILE_ELEMENTS>;
var<workgroup> shared_q_t: array<f16, $TILE_ELEMENTS>;
var<workgroup> shared_do: array<f32, $TILE_ELEMENTS>;
var<workgroup> shared_do_t: array<f16, $TILE_ELEMENTS>;
var<workgroup> shared_score: array<f32, 256>;
var<workgroup> shared_dp: array<f32, 256>;
var<workgroup> shared_p: array<f32, 256>;
var<workgroup> shared_ds: array<f32, 256>;
var<workgroup> wg_row_sum: array<f32, 16>;
var<workgroup> wg_row_sum_partial: array<f32, 64>;
var<workgroup> wg_lse_max: array<f32, 16>;
var<workgroup> wg_lse_log: array<f32, 16>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_id) sg: u32
) {
    let kv_base = wgid.x * 16u;
    let kv_head = wgid.y;
    let q_seq = params.q_seq;
    let kv_seq = params.kv_seq;
    let num_heads = params.packed_heads >> 16u;
    let num_kv_heads = params.packed_heads & 0xFFFFu;
    let head_dim = params.head_dim;
    let window_size = params.window_size;
    let heads_per_kv = num_heads / max(num_kv_heads, 1u);
    let kv_head_off = kv_head * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let q_dim = num_heads * head_dim;
    let scale = inverseSqrt(f32(head_dim));
    let effective_kv_seq = select(kv_seq, q_seq, kv_seq == 0u);

    let kv_row = lid.x / 4u;
    let chunk = lid.x % 4u;
    let d_off = chunk * $CHUNK_HDu;
    let kv_pos_thread = kv_base + kv_row;
    let kv_valid_thread = kv_pos_thread < effective_kv_seq && kv_head < num_kv_heads;

    var local_dv: array<f32, $CHUNK_HD>;
    var local_dk: array<f32, $CHUNK_HD>;
    for (var e = 0u; e < $CHUNK_HDu; e = e + 1u) {
        local_dv[e] = 0.0;
        local_dk[e] = 0.0;
    }

    for (var i = lid.x; i < $TILE_ELEMENTSu; i = i + 64u) {
        let ki = i / $HEAD_DIMu;
        let d = i % $HEAD_DIMu;
        let kp = kv_base + ki;
        if kp < effective_kv_seq {
            let kv_off = kp * kv_dim + kv_head_off + d;
            shared_k[i] = f16(src_b[kv_off]);
            shared_v[i] = f16(bias[kv_off]);
        } else {
            shared_k[i] = f16(0.0);
            shared_v[i] = f16(0.0);
        }
    }
    workgroupBarrier();

    for (var qh = 0u; qh < heads_per_kv; qh = qh + 1u) {
        let q_head = kv_head * heads_per_kv + qh;

        let row_q_start = select(0u, kv_pos_thread, kv_seq == 0u);
        let tile_end = (q_seq / 16u) * 16u;
        var t = 0u;
        for (; t < tile_end; t = t + 16u) {
            for (var i = lid.x; i < $TILE_ELEMENTSu; i = i + 64u) {
                let qi = i / $HEAD_DIMu;
                let d = i % $HEAD_DIMu;
                let qp = t + qi;
                if qp < q_seq {
                    let q_off = qp * q_dim + q_head * head_dim + d;
                    let qv = src_a[q_off];
                    let dov = d_out[q_off];
                    shared_q[i] = qv;
                    shared_do[i] = dov;
                    shared_q_t[d * 16u + qi] = f16(qv);
                    shared_do_t[d * 16u + qi] = f16(dov);
                } else {
                    shared_q[i] = 0.0;
                    shared_do[i] = 0.0;
                    shared_q_t[d * 16u + qi] = f16(0.0);
                    shared_do_t[d * 16u + qi] = f16(0.0);
                }
            }
            workgroupBarrier();

            let row_qp = t + kv_row;
            var row_part = 0.0;
            if row_qp < q_seq && q_head < num_heads {
                let q_base = row_qp * q_dim + q_head * head_dim + d_off;
                for (var e = 0u; e < $CHUNK_HDu; e = e + 1u) {
                    row_part = row_part + shared_do[kv_row * $HEAD_DIMu + d_off + e] * fwd_dst[q_base + e];
                }
            }
            wg_row_sum_partial[lid.x] = row_part;
            workgroupBarrier();
            if lid.x < 16u {
                let base = lid.x * 4u;
                wg_row_sum[lid.x] = wg_row_sum_partial[base] + wg_row_sum_partial[base + 1u] + wg_row_sum_partial[base + 2u] + wg_row_sum_partial[base + 3u];
                let qp = t + lid.x;
                if qp < q_seq && q_head < num_heads {
                    let li = (qp * num_heads + q_head) * 2u;
                    wg_lse_max[lid.x] = lse[li];
                    wg_lse_log[lid.x] = lse[li + 1u];
                } else {
                    wg_lse_max[lid.x] = 0.0;
                    wg_lse_log[lid.x] = 0.0;
                }
            }
            workgroupBarrier();

            // Cooperative score tile.
        var score_acc = coop_mat16x16<f32,C>();
            for (var ht = 0u; ht < $HEAD_TILESu; ht = ht + 1u) {
                let a_k = coopLoadT<coop_mat16x16<f16,A>>(&shared_k[ht * 16u], $HEAD_DIMu);
                let b_qt = coopLoadT<coop_mat16x16<f16,B>>(&shared_q_t[ht * 16u * 16u], 16u);
                score_acc = coopMultiplyAdd(a_k, b_qt, score_acc);
            }
            if sg == 0u { coopStoreT(score_acc, &shared_score[0], 16u); }
            var dp_acc = coop_mat16x16<f32,C>();
            for (var ht = 0u; ht < $HEAD_TILESu; ht = ht + 1u) {
                let a_v = coopLoadT<coop_mat16x16<f16,A>>(&shared_v[ht * 16u], $HEAD_DIMu);
                let b_dot = coopLoadT<coop_mat16x16<f16,B>>(&shared_do_t[ht * 16u * 16u], 16u);
                dp_acc = coopMultiplyAdd(a_v, b_dot, dp_acc);
            }
            if sg == 0u { coopStoreT(dp_acc, &shared_dp[0], 16u); }
            workgroupBarrier();

            for (var k = 0u; k < 4u; k = k + 1u) {
                let idx = lid.x * 4u + k;
                if idx < 256u {
                    let kv = idx / 16u;
                    let q  = idx % 16u;
                    let kp = kv_base + kv;
                    let qp = t + q;
                    let masked = !(kp < effective_kv_seq && kv_head < num_kv_heads && qp < q_seq && q_head < num_heads);
                    let row_kv_len = select(kv_seq, qp + 1u, kv_seq == 0u);
                    let row_kv_start = select(0u, row_kv_len - min(row_kv_len, window_size), window_size > 0u);
                    let attn_masked = masked || (kp < row_kv_start) || (kp >= row_kv_len);
                    let s = shared_score[kv * 16u + q] * scale;
                    let p = exp(min(s - wg_lse_max[q], 0.0) - wg_lse_log[q]);
                    let dp_v = shared_dp[kv * 16u + q];
                    let ds = p * (dp_v - wg_row_sum[q]);
                    let p_safe = select(p, 0.0, attn_masked);
                    let ds_safe = select(ds, 0.0, attn_masked);
                    shared_p[kv * 16u + q] = p_safe;
                    shared_ds[kv * 16u + q] = ds_safe;
                }
            }
            workgroupBarrier();

            for (var q = 0u; q < 16u; q = q + 1u) {
                let p = shared_p[kv_row * 16u + q];
                let ds = shared_ds[kv_row * 16u + q];
                for (var e = 0u; e < $CHUNK_HDu; e = e + 1u) {
                    let dov = shared_do[q * $HEAD_DIMu + d_off + e];
                    let qv  = shared_q[q * $HEAD_DIMu + d_off + e];
                    local_dv[e] = local_dv[e] + p  * dov;
                    local_dk[e] = local_dk[e] + ds * qv;
                }
            }
            workgroupBarrier();
        }

        for (; t < q_seq; t = t + 1u) {
            let masked = !(kv_valid_thread && q_head < num_heads);
            if !masked {
                let row_kv_len = select(kv_seq, t + 1u, kv_seq == 0u);
                let row_kv_start = select(0u, row_kv_len - min(row_kv_len, window_size), window_size > 0u);
                let attn_masked = (kv_pos_thread < row_kv_start) || (kv_pos_thread >= row_kv_len);
                if !attn_masked {
                    var dot_qk = 0.0;
                    var dot_dov = 0.0;
                    var dot_doo = 0.0;
                    for (var d = 0u; d < $HEAD_DIMu; d = d + 1u) {
                        let kv = src_b[kv_pos_thread * kv_dim + kv_head_off + d];
                        let vv = bias[kv_pos_thread * kv_dim + kv_head_off + d];
                        let qv = src_a[t * q_dim + q_head * head_dim + d];
                        let dov = d_out[t * q_dim + q_head * head_dim + d];
                        let ov = fwd_dst[t * q_dim + q_head * head_dim + d];
                        dot_qk = dot_qk + qv * kv;
                        dot_dov = dot_dov + dov * vv;
                        dot_doo = dot_doo + dov * ov;
                    }
                    let li = (t * num_heads + q_head) * 2u;
                    let lmax = lse[li];
                    let llog = lse[li + 1u];
                    let p = exp(min(dot_qk * scale - lmax, 0.0) - llog);
                    let ds = p * (dot_dov - dot_doo);
                    for (var e = 0u; e < $CHUNK_HDu; e = e + 1u) {
                        let dov = d_out[t * q_dim + q_head * head_dim + d_off + e];
                        let qv = src_a[t * q_dim + q_head * head_dim + d_off + e];
                        local_dv[e] = local_dv[e] + p  * dov;
                        local_dk[e] = local_dk[e] + ds * qv;
                    }
                }
            }
        }
    }

    if kv_valid_thread {
        let kv_dst_off = kv_pos_thread * kv_dim + kv_head_off + d_off;
        for (var e = 0u; e < $CHUNK_HDu; e = e + 1u) {
            dst[kv_dst_off + e] = local_dk[e] * scale;
            dst2[kv_dst_off + e] = local_dv[e];
        }
    }
}
