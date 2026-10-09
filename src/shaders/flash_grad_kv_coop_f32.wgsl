// F32 cooperative backward attention for 64-wide heads. Every score, probability,
// derivative and matrix operand remains f32, including tiny upstream gradients.
enable wgpu_cooperative_matrix;

$PARAMS
var<storage> d_out: array<f32>;
var<storage> src_a: array<f32>;
var<storage> src_b: array<f32>;
var<storage> bias: array<f32>;
var<storage> lse: array<f32>;
// Precomputed dot(dO, O), one scalar per query/head (not the forward output).
var<storage> fwd_dst: array<f32>;
var<storage, read_write> dst: array<f32>;
var<storage, read_write> dst2: array<f32>;
var<uniform> params: Params;

var<workgroup> shared_k: array<f32, 1024>;
var<workgroup> shared_v: array<f32, 1024>;
var<workgroup> shared_q: array<f32, 1024>;
var<workgroup> shared_do: array<f32, 1024>;
// K Q^T / V dO^T, overwritten in place with P / dS.
var<workgroup> shared_p: array<f32, 256>;
var<workgroup> shared_ds: array<f32, 256>;
var<workgroup> row_dot: array<f32, 16>;
var<workgroup> row_max: array<f32, 16>;
var<workgroup> row_log_sum: array<f32, 16>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_id) sg: u32,
    @builtin(subgroup_size) sg_size: u32
) {
    let kv_base = wgid.x * 16u;
    let kv_head = wgid.y;
    let q_seq = params.q_seq;
    let kv_seq = select(params.kv_seq, q_seq, params.kv_seq == 0u);
    let num_heads = params.packed_heads >> 16u;
    let num_kv_heads = params.packed_heads & 0xFFFFu;
    let heads_per_kv = num_heads / num_kv_heads;
    let q_stride = num_heads * 64u;
    let kv_stride = num_kv_heads * 64u;
    let scale = 0.125;
    let causal = params.kv_seq == 0u;
    let first_q = select(0u, kv_base, causal);
    let last_q = select(q_seq, min(q_seq, kv_base + 15u + params.window_size), causal && params.window_size > 0u);

    // SIMD groups own tile-height by 32 output regions. The
    // round loop also handles wider subgroups; extra narrow groups store
    // nothing but still participate in all cooperative operations/barriers.
    for (var group_base = 0u; group_base < $OUTPUT_GROUPSu; group_base += 128u / sg_size) {
        let tile_sg = group_base + sg;
        let out_row = ((tile_sg % $OUTPUT_GROUPSu) / 2u) * $TILEu;
        let out_col = (tile_sg % 2u) * 32u;
        $ACC_INIT

        for (var i = lid.x; i < 1024u; i += 128u) {
            let kp = kv_base + i / 64u;
            let d = i % 64u;
            var k = 0.0;
            var v = 0.0;
            if kp < kv_seq {
                let index = kp * kv_stride + kv_head * 64u + d;
                k = src_b[index];
                v = bias[index];
            }
            shared_k[i] = k;
            shared_v[i] = v;
        }
        workgroupBarrier();

        for (var relative_head = 0u; relative_head < heads_per_kv; relative_head++) {
            let head = kv_head * heads_per_kv + relative_head;
            for (var t = first_q; t < last_q; t += 16u) {
                for (var i = lid.x; i < 1024u; i += 128u) {
                    let qp = t + i / 64u;
                    let d = i % 64u;
                    var q = 0.0;
                    var dov = 0.0;
                    if qp < q_seq {
                        let index = qp * q_stride + head * 64u + d;
                        q = src_a[index];
                        dov = d_out[index];
                    }
                    shared_q[i] = q;
                    shared_do[i] = dov;
                }
                if lid.x < 16u {
                    let qp = t + lid.x;
                    var dot = 0.0;
                    var maximum = 0.0;
                    var logarithm = 0.0;
                    if qp < q_seq {
                        let index = qp * num_heads + head;
                        dot = fwd_dst[index];
                        maximum = lse[index * 2u];
                        logarithm = lse[index * 2u + 1u];
                    }
                    row_dot[lid.x] = dot;
                    row_max[lid.x] = maximum;
                    row_log_sum[lid.x] = logarithm;
                }
                workgroupBarrier();

                for (var score_base = 0u; score_base < $SCORE_GROUPSu; score_base += 128u / sg_size) {
                    let score_sg = score_base + sg;
                    let score_row = ((score_sg % $SCORE_GROUPSu) / $SCORE_COLSu) * $TILEu;
                    let score_col = (score_sg % $SCORE_COLSu) * $TILEu;
                    var score = coop_mat$TILEx$TILE<f32,C>();
                    var dp = coop_mat$TILEx$TILE<f32,C>();
                    for (var h = 0u; h < 64u; h += $TILEu) {
                        let k_index = score_row * 64u + h;
                        let q_index = score_col * 64u + h;
                        let k = coopLoadT<coop_mat$TILEx$TILE<f32,A>>(&shared_k[k_index], 64u);
                        let v = coopLoadT<coop_mat$TILEx$TILE<f32,A>>(&shared_v[k_index], 64u);
                        // Column-major loads transpose the existing Q/dO
                        // tiles without extra workgroup-memory copies.
                        let qt = coopLoad<coop_mat$TILEx$TILE<f32,B>>(&shared_q[q_index], 64u);
                        let dot = coopLoad<coop_mat$TILEx$TILE<f32,B>>(&shared_do[q_index], 64u);
                        score = coopMultiplyAdd(k, qt, score);
                        dp = coopMultiplyAdd(v, dot, dp);
                    }
                    let score_index = score_row * 16u + score_col;
                    if score_sg < $SCORE_GROUPSu {
                        coopStoreT(score, &shared_p[score_index], 16u);
                        coopStoreT(dp, &shared_ds[score_index], 16u);
                    }
                }
                workgroupBarrier();

                for (var i = lid.x; i < 256u; i += 128u) {
                    let kp = kv_base + i / 16u;
                    let q = i % 16u;
                    let qp = t + q;
                    let row_end = select(kv_seq, qp + 1u, causal);
                    let row_start = select(0u, row_end - min(row_end, params.window_size), params.window_size > 0u);
                    var p = 0.0;
                    var ds = 0.0;
                    if qp < q_seq && kp < kv_seq && kp >= row_start && kp < row_end {
                        p = exp(min(shared_p[i] * scale - row_max[q], 0.0) - row_log_sum[q]);
                        ds = p * (shared_ds[i] - row_dot[q]);
                    }
                    shared_p[i] = p;
                    shared_ds[i] = ds;
                }
                workgroupBarrier();

                for (var q = 0u; q < 16u; q += $TILEu) {
                    let a_index = out_row * 16u + q;
                    let p = coopLoadT<coop_mat$TILEx$TILE<f32,A>>(&shared_p[a_index], 16u);
                    let ds = coopLoadT<coop_mat$TILEx$TILE<f32,A>>(&shared_ds[a_index], 16u);
                    $ACCUMULATE
                }
                // No invocation overwrites Q/dO or score storage while
                // another SIMD group still consumes it.
                workgroupBarrier();
            }
        }

        // K/V are no longer needed. Reuse their tiles for checked stores of
        // the final accumulators, keeping workgroup memory below 19 KiB.
        $ACC_STORE
        workgroupBarrier();
        for (var i = lid.x; i < 1024u; i += 128u) {
            let kp = kv_base + i / 64u;
            let d = i % 64u;
            let owner = (i / ($TILEu * 64u)) * 2u + d / 32u;
            if kp < kv_seq && owner >= group_base && owner < group_base + 128u / sg_size {
                let index = kp * kv_stride + kv_head * 64u + d;
                dst[index] = shared_k[i] * scale;
                dst2[index] = shared_v[i];
            }
        }
        workgroupBarrier();
    }
}
