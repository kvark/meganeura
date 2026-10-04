// QK^T and dO V^T use f16 cooperative operands. The scalar dS K accumulation
// uses the untransposed f32 K copy, with dQ held in registers across KV tiles.
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
var<uniform> params: Params;

var<workgroup> shared_q: array<f16, $TILE_ELEMENTS>;
var<workgroup> shared_do: array<f16, $TILE_ELEMENTS>;
var<workgroup> shared_k: array<f32, $TILE_ELEMENTS>;
var<workgroup> shared_k_t: array<f16, $TILE_ELEMENTS>;
var<workgroup> shared_v_t: array<f16, $TILE_ELEMENTS>;
var<workgroup> shared_score: array<f32, 256>;
var<workgroup> shared_dp: array<f32, 256>;
var<workgroup> shared_ds: array<f32, 256>;
var<workgroup> wg_row_sum: array<f32, 16>;
var<workgroup> wg_lse_max: array<f32, 16>;
var<workgroup> wg_lse_log: array<f32, 16>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_id) sg: u32
) {
    let pos_base = wgid.x * 16u;
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

    let last_pos = min(pos_base + 15u, q_seq - 1u);
    let max_kv_len = select(kv_seq, last_pos + 1u, kv_seq == 0u);
    let first_kv_len = select(kv_seq, pos_base + 1u, kv_seq == 0u);
    let min_kv_start = select(0u, first_kv_len - min(first_kv_len, window_size), window_size > 0u);

    for (var i = lid.x; i < $TILE_ELEMENTSu; i = i + 64u) {
        let r = i / $HEAD_DIMu;
        let col = i % $HEAD_DIMu;
        let qp = pos_base + r;
        if qp < q_seq {
            let qi = qp * (num_heads * head_dim) + head * head_dim + col;
            shared_q[i] = f16(src_a[qi]);
            shared_do[i] = f16(d_out[qi]);
        } else {
            shared_q[i] = f16(0.0);
            shared_do[i] = f16(0.0);
        }
    }
    workgroupBarrier();

    if lid.x < 16u {
        let r = lid.x;
        let qp = pos_base + r;
        if qp < q_seq && head < num_heads {
            var s = 0.0;
            let q_base = qp * (num_heads * head_dim) + head * head_dim;
            for (var d = 0u; d < $HEAD_DIMu; d = d + 1u) {
                s = s + d_out[q_base + d] * fwd_dst[q_base + d];
            }
            wg_row_sum[r] = s;
            let li = (qp * num_heads + head) * 2u;
            wg_lse_max[r] = lse[li];
            wg_lse_log[r] = lse[li + 1u];
        } else {
            wg_row_sum[r] = 0.0;
            wg_lse_max[r] = 0.0;
            wg_lse_log[r] = 0.0;
        }
    }

    let row = lid.x / 4u;
    let chunk = lid.x % 4u;
    let d_off = chunk * $CHUNK_HDu;
    let qpos_thread = pos_base + row;
    let q_valid_thread = qpos_thread < q_seq && head < num_heads;

    var local_dq: array<f32, $CHUNK_HD>;
    for (var e = 0u; e < $CHUNK_HDu; e = e + 1u) {
        local_dq[e] = 0.0;
    }

    let tile_end = min_kv_start + ((max_kv_len - min_kv_start) / 16u) * 16u;
    var t = min_kv_start;
    for (; t < tile_end; t = t + 16u) {
        for (var i = lid.x; i < $TILE_ELEMENTSu; i = i + 64u) {
            let ki = i / $HEAD_DIMu;
            let d = i % $HEAD_DIMu;
            let kv_pos = t + ki;
            let k_v = src_b[kv_pos * kv_dim + kv_head_off + d];
            let v_v = bias[kv_pos * kv_dim + kv_head_off + d];
            shared_k[i] = k_v;
            shared_k_t[d * 16u + ki] = f16(k_v);
            shared_v_t[d * 16u + ki] = f16(v_v);
        }
        workgroupBarrier();

        // Cooperative score tile.
        var score_acc = coop_mat16x16<f32,C>();
        for (var ht = 0u; ht < $HEAD_TILESu; ht = ht + 1u) {
            let a = coopLoadT<coop_mat16x16<f16,A>>(&shared_q[ht * 16u], $HEAD_DIMu);
            let b = coopLoadT<coop_mat16x16<f16,B>>(&shared_k_t[ht * 16u * 16u], 16u);
            score_acc = coopMultiplyAdd(a, b, score_acc);
        }
        if sg == 0u { coopStoreT(score_acc, &shared_score[0], 16u); }
        var dp_acc = coop_mat16x16<f32,C>();
        for (var ht = 0u; ht < $HEAD_TILESu; ht = ht + 1u) {
            let a = coopLoadT<coop_mat16x16<f16,A>>(&shared_do[ht * 16u], $HEAD_DIMu);
            let b = coopLoadT<coop_mat16x16<f16,B>>(&shared_v_t[ht * 16u * 16u], 16u);
            dp_acc = coopMultiplyAdd(a, b, dp_acc);
        }
        if sg == 0u { coopStoreT(dp_acc, &shared_dp[0], 16u); }
        workgroupBarrier();

        for (var k = 0u; k < 4u; k = k + 1u) {
            let idx = lid.x * 4u + k;
            if idx < 256u {
                let r = idx / 16u;
                let j = idx % 16u;
                let qpos = pos_base + r;
                let kv_pos = t + j;
                let row_kv_len = select(kv_seq, qpos + 1u, kv_seq == 0u);
                let row_kv_start = select(0u, row_kv_len - min(row_kv_len, window_size), window_size > 0u);
                let q_valid = qpos < q_seq && head < num_heads;
                let masked = !(q_valid && kv_pos >= row_kv_start && kv_pos < row_kv_len);
                let s = shared_score[r * 16u + j] * scale;
                let p = exp(min(s - wg_lse_max[r], 0.0) - wg_lse_log[r]);
                let dp_v = shared_dp[r * 16u + j];
                let ds = select(p * (dp_v - wg_row_sum[r]), 0.0, masked);
                shared_ds[r * 16u + j] = ds;
            }
        }
        workgroupBarrier();

        for (var j = 0u; j < 16u; j = j + 1u) {
            let p = shared_ds[row * 16u + j];
            for (var e = 0u; e < $CHUNK_HDu; e = e + 1u) {
                let kv = shared_k[j * $HEAD_DIMu + d_off + e];
                local_dq[e] = local_dq[e] + p * kv;
            }
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
                let qv = f32(shared_q[row * $HEAD_DIMu + d]);
                let dov = f32(shared_do[row * $HEAD_DIMu + d]);
                let kv = src_b[t * kv_dim + kv_head_off + d];
                let vv = bias[t * kv_dim + kv_head_off + d];
                dot_qk = dot_qk + qv * kv;
                dot_dov = dot_dov + dov * vv;
            }
            let p = exp(min(dot_qk * scale - wg_lse_max[row], 0.0) - wg_lse_log[row]);
            let ds = p * (dot_dov - wg_row_sum[row]);
            for (var e = 0u; e < $CHUNK_HDu; e = e + 1u) {
                let kv = src_b[t * kv_dim + kv_head_off + d_off + e];
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
