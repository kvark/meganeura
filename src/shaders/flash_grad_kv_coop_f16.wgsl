// F16 matrix operands use binary scaling to retain small derivatives.
// Softmax, row dots, partial-product sums and outputs remain F32.
enable f16;
enable wgpu_cooperative_matrix;
$SCALE

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

var<workgroup> shared_k: array<f16, (32 * $HEAD_DIM)>;
var<workgroup> shared_v: array<f16, (32 * $HEAD_DIM)>;
var<workgroup> shared_result: array<f32, (32 * $HEAD_DIM)>;
var<workgroup> shared_anchor: array<f32, $HEAD_DIM>;
var<workgroup> wg_score_offset: array<f32, 16>;
var<workgroup> shared_q_t: array<f16, $TILE_ELEMENTS>;
var<workgroup> shared_do_t: array<f16, $TILE_ELEMENTS>;
var<workgroup> shared_score: array<f32, 512>;
var<workgroup> shared_dp: array<f32, 512>;
var<workgroup> shared_p: array<f16, 512>;
var<workgroup> shared_ds: array<f16, 512>;
var<workgroup> wg_row_sum: array<f32, 16>;
var<workgroup> wg_lse_max: array<f32, 16>;
var<workgroup> wg_lse_log: array<f32, 16>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_id) sg: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_lane: u32
) {
    let kv_base = wgid.x * 32u;
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
    let subgroups = 128u / subgroup_size;
    let effective_kv_seq = select(kv_seq, q_seq, kv_seq == 0u);
    let causal = kv_seq == 0u;
    let first_q = min(q_seq, select(0u, kv_base, causal));
    let last_q = select(q_seq, min(q_seq, kv_base + 31u + window_size), causal && window_size > 0u);

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

    for (var d = lid.x; d < $HEAD_DIMu; d += 128u) {
        shared_anchor[d] = src_b[kv_base * kv_dim + kv_head_off + d];
    }
    workgroupBarrier();
    var maximum_v = 0.0;
    for (var i = lid.x; i < (32u * $HEAD_DIMu); i = i + 128u) {
        let ki = i / $HEAD_DIMu;
        let d = i % $HEAD_DIMu;
        let kp = kv_base + ki;
        if kp < effective_kv_seq {
            let kv_off = kp * kv_dim + kv_head_off + d;
            shared_k[i] = f16(src_b[kv_off] - shared_anchor[d]);
            let value = bias[kv_off];
            shared_result[i] = value;
            maximum_v = max(maximum_v, abs(value));
        } else {
            shared_k[i] = f16(0.0);
            shared_result[i] = 0.0;
        }
    }
    let exponent_v = block_exponents(vec2<f32>(maximum_v, 0.0), sg, subgroup_size, subgroup_lane).x;
    for (var i = lid.x; i < 32u * $HEAD_DIMu; i += 128u) {
        shared_v[i] = f16(scale_power(shared_result[i], exponent_v));
    }
    workgroupBarrier();

    for (var qh = 0u; qh < heads_per_kv; qh = qh + 1u) {
        let q_head = kv_head * heads_per_kv + qh;

        let tile_end = first_q + ((last_q - first_q) / 16u) * 16u;
        var t = first_q;
        for (; t < tile_end; t = t + 16u) {
            var maximum_qdo = vec2<f32>(0.0);
            for (var i = lid.x; i < $TILE_ELEMENTSu; i = i + 128u) {
                let qi = i / $HEAD_DIMu;
                let d = i % $HEAD_DIMu;
                let qp = t + qi;
                if qp < q_seq {
                    let q_off = qp * q_dim + q_head * head_dim + d;
                    let qv = src_a[q_off];
                    let dov = d_out[q_off];
                    shared_result[i] = qv;
                    shared_result[$TILE_ELEMENTSu + i] = dov;
                    maximum_qdo = max(maximum_qdo, abs(vec2<f32>(qv, dov)));
                } else {
                    shared_result[i] = 0.0;
                    shared_result[$TILE_ELEMENTSu + i] = 0.0;
                }
            }
            let exponent_qdo = block_exponents(maximum_qdo, sg, subgroup_size, subgroup_lane);
            for (var i = lid.x; i < $TILE_ELEMENTSu; i += 128u) {
                let index = (i % $HEAD_DIMu) * 16u + i / $HEAD_DIMu;
                shared_q_t[index] = f16(scale_power(shared_result[i], exponent_qdo.x));
                shared_do_t[index] = f16(scale_power(shared_result[$TILE_ELEMENTSu + i], exponent_qdo.y));
            }

            if lid.x < 16u {
                let qp = t + lid.x;
                if qp < q_seq && q_head < num_heads {
                    wg_row_sum[lid.x] = fwd_dst[qp * num_heads + q_head];
                    let li = (qp * num_heads + q_head) * 2u;
                    wg_lse_max[lid.x] = lse[li];
                    wg_lse_log[lid.x] = lse[li + 1u];
                } else {
                    wg_row_sum[lid.x] = 0.0;
                    wg_lse_max[lid.x] = 0.0;
                    wg_lse_log[lid.x] = 0.0;
                    wg_score_offset[lid.x] = 0.0;
                }
            }
            let score_row = lid.x / 4u;
            let score_lane = lid.x % 4u;
            var anchor_dot = 0.0;
            if score_row < 16u && t + score_row < q_seq && q_head < num_heads {
                for (var d = score_lane; d < $HEAD_DIMu; d += 4u) {
                    let qv = src_a[(t + score_row) * q_dim + q_head * head_dim + d];
                    anchor_dot += qv * shared_anchor[d];
                }
            }
            shared_score[lid.x] = anchor_dot;
            workgroupBarrier();
            if lid.x < 16u {
                let i = lid.x * 4u;
                wg_score_offset[lid.x] = (shared_score[i] + shared_score[i + 1u]
                    + shared_score[i + 2u] + shared_score[i + 3u]) * scale;
            }
            workgroupBarrier();

            // Cover every row even when there are fewer subgroups than tiles.
            for (var base = 0u; base < (32u / $Mu); base += subgroups) {
                let tile_row = ((base + sg) % (32u / $Mu)) * $Mu;
                var score_acc = coop_mat16x$M<f32, C>();
                for (var ht = 0u; ht < $HEAD_TILESu; ht = ht + 1u) {
                    let a_k = coopLoadT<coop_mat16x$M<f16, A>>(&shared_k[tile_row * $HEAD_DIMu + ht * 16u], $HEAD_DIMu);
                    let b_qt = coopLoadT<coop_mat16x16<f16, B>>(&shared_q_t[ht * 16u * 16u], 16u);
                    score_acc = coopMultiplyAdd(a_k, b_qt, score_acc);
                }
                if base + sg < (32u / $Mu) {
                    coopStoreT(score_acc, &shared_score[tile_row * 16u], 16u);
                }
            }

            for (var base = 0u; base < (32u / $Mu); base += subgroups) {
                let tile_row = ((base + sg) % (32u / $Mu)) * $Mu;
                var dp_acc = coop_mat16x$M<f32, C>();
                for (var ht = 0u; ht < $HEAD_TILESu; ht = ht + 1u) {
                    let a_v = coopLoadT<coop_mat16x$M<f16, A>>(&shared_v[tile_row * $HEAD_DIMu + ht * 16u], $HEAD_DIMu);
                    let b_dot = coopLoadT<coop_mat16x16<f16, B>>(&shared_do_t[ht * 16u * 16u], 16u);
                    dp_acc = coopMultiplyAdd(a_v, b_dot, dp_acc);
                }
                if base + sg < (32u / $Mu) {
                    coopStoreT(dp_acc, &shared_dp[tile_row * 16u], 16u);
                }
            }

            workgroupBarrier();

            var maximum_dsp = vec2<f32>(0.0);
            for (var k = 0u; k < 4u; k = k + 1u) {
                let idx = lid.x * 4u + k;
                if idx < 512u {
                    let kv = idx / 16u;
                    let q = idx % 16u;
                    let kp = kv_base + kv;
                    let qp = t + q;
                    let masked = !(kp < effective_kv_seq && kv_head < num_kv_heads && qp < q_seq && q_head < num_heads);
                    let row_kv_len = select(kv_seq, qp + 1u, kv_seq == 0u);
                    let row_kv_start = select(0u, row_kv_len - min(row_kv_len, window_size), window_size > 0u);
                    let attn_masked = masked || (kp < row_kv_start) || (kp >= row_kv_len);
                    // shared_q_t also serves the score product. Restore its
                    // scale before adding the common key term computed in F32.
                    let s = scale_power(shared_score[kv * 16u + q], -exponent_qdo.x) * scale + wg_score_offset[q];
                    let p = exp(min(s - wg_lse_max[q], 0.0) - wg_lse_log[q]);
                    let dp_v = scale_power(shared_dp[kv * 16u + q], -(exponent_v + exponent_qdo.y));
                    let ds = p * (dp_v - wg_row_sum[q]);
                    let p_safe = select(p, 0.0, attn_masked);
                    let ds_safe = select(ds, 0.0, attn_masked);
                    shared_score[kv * 16u + q] = p_safe;
                    shared_dp[kv * 16u + q] = ds_safe;
                    maximum_dsp = max(maximum_dsp, abs(vec2<f32>(ds_safe, p_safe)));
                }
            }
            let exponent_dsp = block_exponents(maximum_dsp, sg, subgroup_size, subgroup_lane);
            for (var i = lid.x; i < 512u; i += 128u) {
                shared_ds[i] = f16(scale_power(shared_dp[i], exponent_dsp.x));
                shared_p[i] = f16(scale_power(shared_score[i], exponent_dsp.y));
            }
            workgroupBarrier();

            let tiles = (32u / $Mu) * $HEAD_TILESu;
            for (var base = 0u; base < tiles; base += subgroups) {
                let tile = (base + sg) % tiles;
                let native_row = (tile / $HEAD_TILESu) * $Mu;
                let native_col = (tile % $HEAD_TILESu) * 16u;
                let a = coopLoadT<coop_mat16x$M<f16, A>>(&shared_ds[native_row * 16u], 16u);
                let b = coopLoad<coop_mat16x16<f16, B>>(&shared_q_t[native_col * 16u], 16u);
                // Each partial product is unscaled before the F32 tile sum.
                var product = coopMultiplyAdd(a, b, coop_mat16x$M<f32, C>());
                if base + sg < tiles {
                    coopStoreT(product, &shared_result[native_row * $HEAD_DIMu + native_col], $HEAD_DIMu);
                }
            }
            workgroupBarrier();
            for (var e = 0u; e < $CHUNK_HDu; e++) {
                local_dk[e] += scale_power(shared_result[kv_row * $HEAD_DIMu + d_off + e], -(exponent_dsp.x + exponent_qdo.x));
            }
            workgroupBarrier();
            for (var base = 0u; base < tiles; base += subgroups) {
                let tile = (base + sg) % tiles;
                let native_row = (tile / $HEAD_TILESu) * $Mu;
                let native_col = (tile % $HEAD_TILESu) * 16u;
                let a = coopLoadT<coop_mat16x$M<f16, A>>(&shared_p[native_row * 16u], 16u);
                let b = coopLoad<coop_mat16x16<f16, B>>(&shared_do_t[native_col * 16u], 16u);
                // Each partial product is unscaled before the F32 tile sum.
                var product = coopMultiplyAdd(a, b, coop_mat16x$M<f32, C>());
                if base + sg < tiles {
                    coopStoreT(product, &shared_result[native_row * $HEAD_DIMu + native_col], $HEAD_DIMu);
                }
            }
            workgroupBarrier();
            for (var e = 0u; e < $CHUNK_HDu; e++) {
                local_dv[e] += scale_power(shared_result[kv_row * $HEAD_DIMu + d_off + e], -(exponent_dsp.y + exponent_qdo.y));
            }
            workgroupBarrier();

        }

        for (; t < last_q; t = t + 1u) {
            let masked = !(kv_valid_thread && q_head < num_heads);
            if !masked {
                let row_kv_len = select(kv_seq, t + 1u, kv_seq == 0u);
                let row_kv_start = select(0u, row_kv_len - min(row_kv_len, window_size), window_size > 0u);
                let attn_masked = (kv_pos_thread < row_kv_start) || (kv_pos_thread >= row_kv_len);
                if !attn_masked {
                    var dot_qk = 0.0;
                    var dot_dov = 0.0;
                    let dot_doo = fwd_dst[t * num_heads + q_head];
                    for (var d = 0u; d < $HEAD_DIMu; d = d + 1u) {
                        let kv = src_b[kv_pos_thread * kv_dim + kv_head_off + d];
                        let vv = bias[kv_pos_thread * kv_dim + kv_head_off + d];
                        let qv = src_a[t * q_dim + q_head * head_dim + d];
                        let dov = d_out[t * q_dim + q_head * head_dim + d];
                        dot_qk = dot_qk + qv * kv;
                        dot_dov = dot_dov + dov * vv;
                    }
                    let li = (t * num_heads + q_head) * 2u;
                    let lmax = lse[li];
                    let llog = lse[li + 1u];
                    let p = exp(min(dot_qk * scale - lmax, 0.0) - llog);
                    let ds = p * (dot_dov - dot_doo);
                    for (var e = 0u; e < $CHUNK_HDu; e = e + 1u) {
                        let dov = d_out[t * q_dim + q_head * head_dim + d_off + e];
                        let qv = src_a[t * q_dim + q_head * head_dim + d_off + e];
                        local_dv[e] = local_dv[e] + p * dov;
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
