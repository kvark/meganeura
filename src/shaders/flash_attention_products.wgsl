// @section main
enable f16;
enable wgpu_cooperative_matrix;
$PARAMS
var<storage> src_a: array<f32>;
var<storage> src_b: array<f32>;
var<storage> bias: array<f32>;
var<storage, read_write> dst: array<f32>;
var<storage, read_write> lse: array<f32>;
var<uniform> params: Params;
var<workgroup> shared_q: array<f16, $ELEMENTS>;
var<workgroup> shared_k: array<f16, $KV_ELEMENTS>;
var<workgroup> shared_v: array<f16, $KV_ELEMENTS>;
var<workgroup> shared_score: array<f32, $SCORE_ELEMENTS>;
var<workgroup> shared_p: array<f16, $SCORE_ELEMENTS>;
var<workgroup> shared_pv: array<f32, $ELEMENTS>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(subgroup_invocation_id) slane: u32,
        @builtin(subgroup_id) sg: u32,
        @builtin(subgroup_size) width: u32) {
    let tid = sg * width + slane;
    let pos_base = group.x * 16u;
    let head = group.y;
    let heads = params.packed_heads >> 16u;
    let kv_heads = params.packed_heads & 0xFFFFu;
    let kv_head = head / (heads / max(kv_heads, 1u));
    let kv_dim = kv_heads * params.head_dim;
    let scale = inverseSqrt(f32(params.head_dim));
    let row = tid / 4u;
    let d_base = (tid % 4u) * $CHUNKu;
    let qpos = pos_base + row;
    let q_valid = qpos < params.q_seq && head < heads;
    let first_len = min(params.kv_seq, select(params.kv_seq, pos_base + 1u, pos_base < params.causal_rows));
    let last_pos = min(pos_base + 15u, params.q_seq - 1u);
    let last_len = min(params.kv_seq, select(params.kv_seq, last_pos + 1u, last_pos < params.causal_rows));
    let first_key = first_len - select(0u, min(first_len, params.window_size), params.window_size > 0u);
    // No window starts at the first key; with a window, skip keys unused by every row.
    let begin = select(0u, first_key, params.window_size > 0u);
    let subgroups = 64u / width;
    var local_o: array<f32, $CHUNK>;
    for (var e = 0u; e < $CHUNKu; e++) { local_o[e] = 0.0; }
    for (var i = tid; i < $ELEMENTSu; i += 64u) {
        let qr = i / $PADDEDu;
        let d = i % $PADDEDu;
        var q = 0.0;
        if pos_base + qr < params.q_seq && d < params.head_dim {
            q = src_a[((pos_base + qr) * heads + head) * params.head_dim + d];
        }
        shared_q[i] = f16(q);
    }
    var local_max = -1e30;
    var local_sum = 0.0;
    workgroupBarrier();
    for (var t = begin; t < last_len; t += $KEYSu) {
        for (var i = tid; i < $KV_ELEMENTSu; i += 64u) {
            let kr = i / $PADDEDu;
            let d = i % $PADDEDu;
            var k = 0.0;
            var v = 0.0;
            if t + kr < last_len && d < params.head_dim {
                let index = (t + kr) * kv_dim + kv_head * params.head_dim + d;
                k = src_b[index];
                v = bias[index];
            }
            shared_k[i] = f16(k);
            shared_v[i] = f16(v);
        }
        workgroupBarrier();
$QK_BODY        workgroupBarrier();
        // Four physical subgroup lanes own a row and reduce its F32 statistics.
        var maximum = local_max;
        for (var j = tid % 4u; j < $KEYSu; j += 4u) {
            let len = min(params.kv_seq, select(params.kv_seq, qpos + 1u, qpos < params.causal_rows));
            let start = select(0u, len - min(len, params.window_size), params.window_size > 0u);
            if q_valid && t + j >= start && t + j < len {
                maximum = max(maximum, shared_score[row + j * 16u] * scale);
            }
        }
        maximum = max(maximum, subgroupShuffleXor(maximum, 1u));
        maximum = max(maximum, subgroupShuffleXor(maximum, 2u));
        let rescale = exp(local_max - maximum);
        var sum = 0.0;
        for (var j = tid % 4u; j < $KEYSu; j += 4u) {
            let len = min(params.kv_seq, select(params.kv_seq, qpos + 1u, qpos < params.causal_rows));
            let start = select(0u, len - min(len, params.window_size), params.window_size > 0u);
            var p = 0.0;
            if q_valid && t + j >= start && t + j < len {
                p = exp(shared_score[row + j * 16u] * scale - maximum);
            }
            sum += p;
            shared_p[row + j * 16u] = f16(p);
        }
        sum += subgroupShuffleXor(sum, 1u);
        sum += subgroupShuffleXor(sum, 2u);
        local_sum = local_sum * rescale + sum;
        local_max = maximum;
        workgroupBarrier();
$PV_BODY        workgroupBarrier();
        for (var e = 0u; e < $CHUNKu; e++) {
            local_o[e] = local_o[e] * rescale + shared_pv[row + (d_base + e) * 16u];
        }
        // The next key-stage barrier precedes every reuse of P/PV and row state.
    }
    if q_valid {
        let sum = select(local_sum, 1.0, local_sum == 0.0);
        for (var e = 0u; e < $CHUNKu; e++) {
            let d = d_base + e;
            if d < params.head_dim { dst[(qpos * heads + head) * params.head_dim + d] = local_o[e] / sum; }
        }
        if tid % 4u == 0u {
            let i = (qpos * heads + head) * 2u;
            lse[i] = local_max;
            lse[i + 1u] = select(log(local_sum), -1e30, local_sum == 0.0);
        }
    }
}
// @section qk16
        for (var qr = sg * $ROWSu; qr < 16u; qr += subgroups * $ROWSu) {
            var score = coop_mat16x$ROWS<f32,C>();
            for (var d = 0u; d < $PADDEDu; d += 16u) {
                let q = coopLoadT<coop_mat16x$ROWS<f16,A>>(&shared_q[qr * $PADDEDu + d], $PADDEDu);
                let k = coopLoad<coop_mat16x16<f16,B>>(&shared_k[d], $PADDEDu);
                score = coopMultiplyAdd(q, k, score);
            }
            coopStore(score, &shared_score[qr], 16u);
        }
// @section pv16
        for (var block = sg; block < $PV_TILES_u; block += subgroups) {
            let qr = (block / $HEAD_TILES_u) * $ROWSu;
            let d = (block % $HEAD_TILES_u) * 16u;
            let p = coopLoad<coop_mat16x$ROWS<f16,A>>(&shared_p[qr], 16u);
            let v = coopLoadT<coop_mat16x16<f16,B>>(&shared_v[d], $PADDEDu);
            let value = coopMultiplyAdd(p, v, coop_mat16x$ROWS<f32,C>());
            coopStore(value, &shared_pv[qr + d * 16u], 16u);
        }
// @section qk_wide
        for (var block = sg; block < $QK_TILES_u; block += subgroups) {
            let qr = (block / $KEY_TILES_u) * $ROWSu;
            let key = (block % $KEY_TILES_u) * 16u;
            var score = coop_mat16x$ROWS<f32,C>();
            for (var d = 0u; d < $PADDEDu; d += 16u) {
                let q = coopLoadT<coop_mat16x$ROWS<f16,A>>(&shared_q[qr * $PADDEDu + d], $PADDEDu);
                let k = coopLoad<coop_mat16x16<f16,B>>(&shared_k[key * $PADDEDu + d], $PADDEDu);
                score = coopMultiplyAdd(q, k, score);
            }
            coopStore(score, &shared_score[qr + key * 16u], 16u);
        }
// @section pv_wide
        for (var block = sg; block < $PV_TILES_u; block += subgroups) {
            let qr = (block / $HEAD_TILES_u) * $ROWSu;
            let d = (block % $HEAD_TILES_u) * 16u;
            var value = coop_mat16x$ROWS<f32,C>();
            for (var key = 0u; key < $KEYSu; key += 16u) {
                let p = coopLoad<coop_mat16x$ROWS<f16,A>>(&shared_p[qr + key * 16u], 16u);
                let v = coopLoadT<coop_mat16x16<f16,B>>(&shared_v[key * $PADDEDu + d], $PADDEDu);
                value = coopMultiplyAdd(p, v, value);
            }
            coopStore(value, &shared_pv[qr + d * 16u], 16u);
        }
