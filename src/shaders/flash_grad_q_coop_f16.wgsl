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
var<uniform> params: Params;

var<workgroup> shared_q: array<f16, (32 * $HEAD_DIM)>;
var<workgroup> shared_do: array<f16, (32 * $HEAD_DIM)>;
var<workgroup> shared_result: array<f32, (32 * $HEAD_DIM)>;
var<workgroup> shared_anchor: array<f32, $HEAD_DIM>;
var<workgroup> wg_score_offset: array<f32, 32>;
var<workgroup> shared_k_t: array<f16, $TILE_ELEMENTS>;
var<workgroup> shared_v_t: array<f16, $TILE_ELEMENTS>;
var<workgroup> shared_score: array<f32, 512>;
var<workgroup> shared_dp: array<f32, 512>;
var<workgroup> shared_ds: array<f16, 512>;
var<workgroup> wg_row_sum: array<f32, 32>;
var<workgroup> wg_lse_max: array<f32, 32>;
var<workgroup> wg_lse_log: array<f32, 32>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_id) sg: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_lane: u32
) {
    let pos_base = wgid.x * 32u;
    let head = wgid.y;
    let q_seq = params.q_seq;
    let kv_seq = params.kv_seq;
    let num_heads = params.packed_heads >> 16u;
    let num_kv_heads = params.packed_heads & 0xFFFFu;
    let head_dim = params.head_dim;
    let window_size = params.window_size;
    let kv_head = head / (num_heads / max(num_kv_heads, 1u));
    let kv_head_off = kv_head * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let scale = inverseSqrt(f32(head_dim));
    let subgroups = 128u / subgroup_size;

    let last_pos = min(pos_base + 31u, q_seq - 1u);
    let max_kv_len = select(kv_seq, last_pos + 1u, kv_seq == 0u);
    let first_kv_len = select(kv_seq, pos_base + 1u, kv_seq == 0u);
    let min_kv_start = select(0u, first_kv_len - min(first_kv_len, window_size), window_size > 0u);

    // Cache the fixed key anchor once; the following scaling barriers publish it.
    for (var d = lid.x; d < $HEAD_DIMu; d += 128u) {
        shared_anchor[d] = src_b[min_kv_start * kv_dim + kv_head_off + d];
    }
    let exponent_do = exponent_for(row_maxima(pos_base, 32u, head).y);
    for (var i = lid.x; i < (32u * $HEAD_DIMu); i = i + 128u) {
        let r = i / $HEAD_DIMu;
        let col = i % $HEAD_DIMu;
        let qp = pos_base + r;
        if qp < q_seq {
            let qi = qp * (num_heads * head_dim) + head * head_dim + col;
            shared_q[i] = f16(src_a[qi]);
            shared_do[i] = f16(scale_power(d_out[qi], exponent_do));
        } else {
            shared_q[i] = f16(0.0);
            shared_do[i] = f16(0.0);
        }
    }
    workgroupBarrier(); // Publish the anchor before its dot products.

    if lid.x < 32u {
        let r = lid.x;
        let qp = pos_base + r;
        if qp < q_seq && head < num_heads {
            wg_row_sum[r] = fwd_dst[qp * num_heads + head];
            let li = (qp * num_heads + head) * 2u;
            wg_lse_max[r] = lse[li];
            wg_lse_log[r] = lse[li + 1u];
        } else {
            wg_row_sum[r] = 0.0;
            wg_lse_max[r] = 0.0;
            wg_lse_log[r] = 0.0;
            wg_score_offset[r] = 0.0;
        }
    }

    let row = lid.x / 4u;
    let chunk = lid.x % 4u;
    let d_off = chunk * $CHUNK_HDu;
    let qpos_thread = pos_base + row;
    let q_valid_thread = qpos_thread < q_seq && head < num_heads;

    // Share the anchor dot over four lanes per query, using score scratch
    // before the first score product. This also publishes all row metadata.
    var anchor_dot = 0.0;
    if q_valid_thread {
        for (var d = chunk; d < $HEAD_DIMu; d += 4u) {
            let qv = src_a[qpos_thread * (num_heads * head_dim) + head * head_dim + d];
            anchor_dot += qv * shared_anchor[d];
        }
    }
    shared_score[lid.x] = anchor_dot;
    workgroupBarrier();
    if chunk == 0u {
        let i = row * 4u;
        wg_score_offset[row] = (shared_score[i] + shared_score[i + 1u]
            + shared_score[i + 2u] + shared_score[i + 3u]) * scale;
    }
    workgroupBarrier();

    var local_dq: array<f32, $CHUNK_HD>;
    for (var e = 0u; e < $CHUNK_HDu; e = e + 1u) {
        local_dq[e] = 0.0;
    }

    let tile_end = min_kv_start + ((max_kv_len - min_kv_start) / 16u) * 16u;
    var t = min_kv_start;
    for (; t < tile_end; t = t + 16u) {
        var maximum_kv = vec2<f32>(0.0);
        for (var i = lid.x; i < $TILE_ELEMENTSu; i = i + 128u) {
            let ki = i / $HEAD_DIMu;
            let d = i % $HEAD_DIMu;
            let kv_pos = t + ki;
            let k_v = src_b[kv_pos * kv_dim + kv_head_off + d];
            let v_v = bias[kv_pos * kv_dim + kv_head_off + d];
            // A common key component must not amplify dS rounding error.
            // The exact dS row sums to zero; keep the anchor fixed across tiles.
            let centered = k_v - shared_anchor[d];
            shared_result[i] = centered;
            shared_result[$TILE_ELEMENTSu + i] = v_v;
            maximum_kv = max(maximum_kv, abs(vec2<f32>(centered, v_v)));
        }
        let exponent_kv = block_exponents(maximum_kv, sg, subgroup_size, subgroup_lane);
        for (var i = lid.x; i < $TILE_ELEMENTSu; i += 128u) {
            let index = (i % $HEAD_DIMu) * 16u + i / $HEAD_DIMu;
            shared_k_t[index] = f16(scale_power(shared_result[i], exponent_kv.x));
            shared_v_t[index] = f16(scale_power(shared_result[$TILE_ELEMENTSu + i], exponent_kv.y));
        }
        workgroupBarrier();

        // Cover every row even when there are fewer subgroups than tiles.
        for (var base = 0u; base < (32u / $Mu); base += subgroups) {
            let tile_row = ((base + sg) % (32u / $Mu)) * $Mu;
            var score_acc = coop_mat16x$M<f32, C>();
            for (var ht = 0u; ht < $HEAD_TILESu; ht = ht + 1u) {
                let a = coopLoadT<coop_mat16x$M<f16, A>>(&shared_q[tile_row * $HEAD_DIMu + ht * 16u], $HEAD_DIMu);
                let b = coopLoadT<coop_mat16x16<f16, B>>(&shared_k_t[ht * 16u * 16u], 16u);
                score_acc = coopMultiplyAdd(a, b, score_acc);
            }
            if base + sg < (32u / $Mu) {
                coopStoreT(score_acc, &shared_score[tile_row * 16u], 16u);
            }
        }

        for (var base = 0u; base < (32u / $Mu); base += subgroups) {
            let tile_row = ((base + sg) % (32u / $Mu)) * $Mu;
            var dp_acc = coop_mat16x$M<f32, C>();
            for (var ht = 0u; ht < $HEAD_TILESu; ht = ht + 1u) {
                let a = coopLoadT<coop_mat16x$M<f16, A>>(&shared_do[tile_row * $HEAD_DIMu + ht * 16u], $HEAD_DIMu);
                let b = coopLoadT<coop_mat16x16<f16, B>>(&shared_v_t[ht * 16u * 16u], 16u);
                dp_acc = coopMultiplyAdd(a, b, dp_acc);
            }
            if base + sg < (32u / $Mu) {
                coopStoreT(dp_acc, &shared_dp[tile_row * 16u], 16u);
            }
        }

        workgroupBarrier();

        var maximum_ds = 0.0;
        for (var k = 0u; k < 4u; k = k + 1u) {
            let idx = lid.x * 4u + k;
            if idx < 512u {
                let r = idx / 16u;
                let j = idx % 16u;
                let qpos = pos_base + r;
                let kv_pos = t + j;
                let row_kv_len = select(kv_seq, qpos + 1u, kv_seq == 0u);
                let row_kv_start = select(0u, row_kv_len - min(row_kv_len, window_size), window_size > 0u);
                let q_valid = qpos < q_seq && head < num_heads;
                let masked = !(q_valid && kv_pos >= row_kv_start && kv_pos < row_kv_len);
                // Restore Q·K = Q·(K-anchor) + Q·anchor. The common term stays
                // F32 so rounding a large anchor cannot erase key differences.
                let s = scale_power(shared_score[r * 16u + j], -exponent_kv.x) * scale + wg_score_offset[r];
                let p = exp(min(s - wg_lse_max[r], 0.0) - wg_lse_log[r]);
                let dp_v = scale_power(shared_dp[r * 16u + j], -(exponent_do + exponent_kv.y));
                let ds = select(p * (dp_v - wg_row_sum[r]), 0.0, masked);
                shared_dp[r * 16u + j] = ds;
                maximum_ds = max(maximum_ds, abs(ds));
            }
        }
        let exponent_ds = block_exponents(vec2<f32>(maximum_ds, 0.0), sg, subgroup_size, subgroup_lane).x;
        for (var i = lid.x; i < 512u; i += 128u) {
            shared_ds[i] = f16(scale_power(shared_dp[i], exponent_ds));
        }
        workgroupBarrier();

        let tiles = (32u / $Mu) * $HEAD_TILESu;
        for (var base = 0u; base < tiles; base += subgroups) {
            let tile = (base + sg) % tiles;
            let native_row = (tile / $HEAD_TILESu) * $Mu;
            let native_col = (tile % $HEAD_TILESu) * 16u;
            let a = coopLoadT<coop_mat16x$M<f16, A>>(&shared_ds[native_row * 16u], 16u);
            let b = coopLoad<coop_mat16x16<f16, B>>(&shared_k_t[native_col * 16u], 16u);
            // Each partial product is unscaled before the F32 tile sum.
            var result = coopMultiplyAdd(a, b, coop_mat16x$M<f32, C>());
            if base + sg < tiles {
                coopStoreT(result, &shared_result[native_row * $HEAD_DIMu + native_col], $HEAD_DIMu);
            }
        }
        workgroupBarrier();
        for (var e = 0u; e < $CHUNK_HDu; e++) {
            local_dq[e] += scale_power(shared_result[row * $HEAD_DIMu + d_off + e], -(exponent_ds + exponent_kv.x));
        }
        workgroupBarrier();
    }

    for (; t < max_kv_len; t = t + 1u) {
        let row_kv_len = select(kv_seq, qpos_thread + 1u, kv_seq == 0u);
        let row_kv_start = select(0u, row_kv_len - min(row_kv_len, window_size), window_size > 0u);
        let masked = !(q_valid_thread && t >= row_kv_start && t < row_kv_len);
        if !masked {
            var dot_qk = 0.0;
            var dot_dov = 0.0;
            for (var d = 0u; d < $HEAD_DIMu; d = d + 1u) {
                let qv = src_a[qpos_thread * (num_heads * head_dim) + head * head_dim + d];
                let dov = d_out[qpos_thread * (num_heads * head_dim) + head * head_dim + d];
                let kv = src_b[t * kv_dim + kv_head_off + d];
                let vv = bias[t * kv_dim + kv_head_off + d];
                dot_qk = dot_qk + qv * kv;
                dot_dov = dot_dov + dov * vv;
            }
            let p = exp(min(dot_qk * scale - wg_lse_max[row], 0.0) - wg_lse_log[row]);
            let ds = p * (dot_dov - wg_row_sum[row]);
            for (var e = 0u; e < $CHUNK_HDu; e = e + 1u) {
                let kv = src_b[t * kv_dim + kv_head_off + d_off + e] - shared_anchor[d_off + e];
                local_dq[e] = local_dq[e] + ds * kv;
            }
        }
    }

    if q_valid_thread {
        let dst_row_stride = num_heads * head_dim;
        let q_base = qpos_thread * dst_row_stride + head * head_dim + d_off;
        for (var e = 0u; e < $CHUNK_HDu; e = e + 1u) {
            dst[q_base + e] = local_dq[e] * scale;
        }
    }
}
