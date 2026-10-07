// QK^T uses $INPUT_TYPE cooperative operands; V and the scalar PV accumulation stay f32.
// Each of 64 threads owns one (query row, head-dimension chunk). Keeping O,
// its maximum and sum in registers avoids a shared-memory roundtrip per tile.
$ENABLE_F16
enable wgpu_cooperative_matrix;

$PARAMS
var<storage> src_a: array<f32>;
var<storage> src_b: array<f32>;
var<storage> bias: array<f32>;
var<storage, read_write> dst: array<f32>;
var<storage, read_write> lse: array<f32>;
var<uniform> params: Params;

var<workgroup> shared_q: array<$INPUT_TYPE, $TILE_ELEMENTS>;
var<workgroup> shared_k_t: array<$INPUT_TYPE, $TILE_ELEMENTS>;
var<workgroup> shared_v: array<f32, $TILE_ELEMENTS>;
var<workgroup> shared_score: array<f32, 256>;

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

    let row = lid.x / 4u;
    let chunk = lid.x % 4u;
    let d_off = chunk * $CHUNK_HDu;
    let qpos = pos_base + row;
    let q_valid = qpos < q_seq && head < num_heads;

    let row_kv_len = select(kv_seq, qpos + 1u, kv_seq == 0u);
    let row_kv_start = select(0u, row_kv_len - min(row_kv_len, window_size), window_size > 0u);
    let last_pos = min(pos_base + 15u, q_seq - 1u);
    let max_kv_len = select(kv_seq, last_pos + 1u, kv_seq == 0u);
    let first_kv_len = select(kv_seq, pos_base + 1u, kv_seq == 0u);
    let min_kv_start = select(0u, first_kv_len - min(first_kv_len, window_size), window_size > 0u);

    var local_o: array<f32, $CHUNK_HD>;
    for (var e = 0u; e < $CHUNK_HDu; e = e + 1u) {
        local_o[e] = 0.0;
    }
    var local_max: f32 = -1e30;
    var local_sum: f32 = 0.0;

    for (var i = lid.x; i < $TILE_ELEMENTSu; i = i + 64u) {
        let r = i / $HEAD_DIMu;
        let col = i % $HEAD_DIMu;
        let qp = pos_base + r;
        if qp < q_seq { shared_q[i] = $INPUT_TYPE(src_a[qp * (num_heads * head_dim) + head * head_dim + col]); } else { shared_q[i] = $INPUT_TYPE(0.0); }
    }
    workgroupBarrier();

    let tile_end = min_kv_start + ((max_kv_len - min_kv_start) / 16u) * 16u;
    var t = min_kv_start;
    for (; t < tile_end; t = t + 16u) {
        for (var i = lid.x; i < $TILE_ELEMENTSu; i = i + 64u) {
            let ki = i / $HEAD_DIMu;
            let d = i % $HEAD_DIMu;
            let kv_pos = t + ki;
            shared_k_t[d * 16u + ki] = $INPUT_TYPE(src_b[kv_pos * kv_dim + kv_head_off + d]);
            shared_v[ki * head_dim + d] = bias[kv_pos * kv_dim + kv_head_off + d];
        }
        workgroupBarrier();

        // Cooperative score tile.
        var score_acc = coop_mat16x16<f32,C>();
        for (var ht = 0u; ht < $HEAD_TILESu; ht = ht + 1u) {
            let a = coopLoadT<coop_mat16x16<$INPUT_TYPE,A>>(&shared_q[ht * 16u], $HEAD_DIMu);
            let b = coopLoadT<coop_mat16x16<$INPUT_TYPE,B>>(&shared_k_t[ht * 16u * 16u], 16u);
            score_acc = coopMultiplyAdd(a, b, score_acc);
        }
        if sg == 0u { coopStoreT(score_acc, &shared_score[0], 16u); }
        workgroupBarrier();

        var rowmax = -1e30;
        for (var j = 0u; j < 16u; j = j + 1u) {
            let kv_pos = t + j;
            if q_valid && kv_pos >= row_kv_start && kv_pos < row_kv_len {
                rowmax = max(rowmax, shared_score[row * 16u + j] * scale);
            }
        }
        let new_max = max(local_max, rowmax);
        let correction = select(exp(local_max - new_max), 0.0, !q_valid);
        for (var e = 0u; e < $CHUNK_HDu; e = e + 1u) {
            local_o[e] = local_o[e] * correction;
        }
        var rowsum = 0.0;
        for (var j = 0u; j < 16u; j = j + 1u) {
            let kv_pos = t + j;
            let masked = !(q_valid && kv_pos >= row_kv_start && kv_pos < row_kv_len);
            let score = shared_score[row * 16u + j] * scale;
            let p = select(exp(score - new_max), 0.0, masked);
            rowsum = rowsum + p;
            for (var e = 0u; e < $CHUNK_HDu; e = e + 1u) {
                local_o[e] = local_o[e] + p * shared_v[j * $HEAD_DIMu + d_off + e];
            }
        }
        local_sum = local_sum * correction + rowsum;
        local_max = select(local_max, new_max, q_valid);
        workgroupBarrier();
    }

    for (; t < max_kv_len; t = t + 1u) {
        let masked = !(q_valid && t >= row_kv_start && t < row_kv_len);
        var dot = 0.0;
        for (var d = 0u; d < $HEAD_DIMu; d = d + 1u) {
            let qv = f32(shared_q[row * $HEAD_DIMu + d]);
            let kv = src_b[t * kv_dim + kv_head_off + d];
            dot = dot + qv * kv;
        }
        let score = dot * scale;
        let new_max = select(local_max, max(local_max, score), !masked);
        let correction = exp(local_max - new_max);
        let p = select(exp(score - new_max), 0.0, masked);
        for (var e = 0u; e < $CHUNK_HDu; e = e + 1u) {
            let v = bias[t * kv_dim + kv_head_off + d_off + e];
            local_o[e] = local_o[e] * correction + p * v;
        }
        local_sum = local_sum * correction + p;
        local_max = select(local_max, new_max, q_valid);
    }

    if q_valid {
        let safe_sum = select(local_sum, 1.0, local_sum == 0.0);
        let q_base = qpos * (num_heads * head_dim) + head * head_dim;
        for (var e = 0u; e < $CHUNK_HDu; e = e + 1u) {
            dst[q_base + d_off + e] = local_o[e] / safe_sum;
        }
        if chunk == 0u {
            let idx = (qpos * num_heads + head) * 2u;
            lse[idx] = local_max;
            lse[idx + 1u] = select(log(local_sum), -1e30, local_sum == 0.0);
        }
    }
}
