// @section main
// Four subgroups share one K/V tile. Each subgroup owns 16 query rows,
// and two lanes per row each accumulate half its output in F32.
enable f16;
enable wgpu_cooperative_matrix;
var<storage> src_a: array<vec4<f32>>;
var<storage> src_b: array<vec4<f32>>;
var<storage> bias: array<vec4<f32>>;
var<storage, read_write> dst: array<vec4<f32>>;
var<storage, read_write> lse: array<f32>;
const HD4 = $COLS;
const QK_STRIDE = HD4 + 2u;
var<workgroup> qk: array<vec4<f16>, 64u * QK_STRIDE>;
$VALUE_STORAGE
const SCORE_STRIDE = 9u;
var<workgroup> scores: array<vec4<f32>, 64u * SCORE_STRIDE>;
$PROB_STORAGE

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32,
        @builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) lane: u32) {
    let heads = $HEADS;
    let kv_heads = $KV_HEADS;
    let q0 = wg.x * 64u;
    let head = wg.y;
    let kv_head = head / (heads / kv_heads);
    let scale = inverseSqrt(f32(HD4 * 4u));
    for (var i = tid; i < 64u * HD4; i += 128u) {
        let r = i / HD4;
        let col = i % HD4;
        let source = (min(q0 + r, $Q_SEQ - 1u) * heads + head) * HD4 + col;
        qk[r * QK_STRIDE + col] = vec4<f16>(src_a[source]);
    }
    workgroupBarrier();
    $LOAD_Q
    $ACCUM_INIT
    workgroupBarrier();
    let row = sg * 16u + lane % 16u;
    let half = lane / 16u;
    let query = q0 + row;
    let end = min($KV_SEQ, select($KV_SEQ, query + 1u, query < $CAUSAL_ROWS));
    let start = select(0u, end - min(end, $WINDOW), $WINDOW > 0u);
    let last = min(q0 + 63u, $Q_SEQ - 1u);
    let tile_end = min($KV_SEQ, select($KV_SEQ, last + 1u, last < $CAUSAL_ROWS));
    let first_end = min($KV_SEQ, select($KV_SEQ, q0 + 1u, q0 < $CAUSAL_ROWS));
    let tile_start = select(0u, first_end - min(first_end, $WINDOW), $WINDOW > 0u);
    $SCALAR_OUTPUT
    var maximum = -1e30;
    var true_max = -1e30;
    var sum = 0.0;
    for (var base = tile_start; base < tile_end; base += 32u) {
        for (var i = tid; i < 32u * HD4; i += 128u) {
            let r = i / HD4;
            let col = i % HD4;
            let source = (min(base + r, tile_end - 1u) * kv_heads + kv_head) * HD4 + col;
            qk[r * QK_STRIDE + col] = vec4<f16>(src_b[source]);
            $STORE_VALUE
        }
        workgroupBarrier();
        for (var key = 0u; key < 32u; key += 16u) {
            var score = coop_mat16x16<f32,C>();
            $QK
            coopStoreT(score, &scores[sg * 16u * SCORE_STRIDE + key / 4u], SCORE_STRIDE);
        }
        workgroupBarrier();
        var tile_max = maximum;
        for (var k = half * 16u; k < (half + 1u) * 16u; k++) {
            if base + k >= start && base + k < end {
                tile_max = max(tile_max, scores[row * SCORE_STRIDE + k / 4u][k % 4u] * scale);
            }
        }
        tile_max = max(tile_max, subgroupShuffleXor(tile_max, 16u));
        true_max = max(true_max, tile_max);
        // With F16 probabilities we can delay rescaling until exp(score - maximum)
        // approaches exp(8). Keep the true maximum for the backward pass.
        let new_max = select(maximum, tile_max, tile_max > maximum + $SLACK);
        let correction = exp(maximum - new_max);
        let rescale = sum > 0.0 && new_max > maximum;
        sum *= correction;
        maximum = new_max;
        $PV
        workgroupBarrier();
    }
    $FINISH_PV
    if query < $Q_SEQ {
        let out = query * heads + head;
        $SCALAR_STORE
        if half == 0u {
            lse[out * 2u] = true_max;
            lse[out * 2u + 1u] = select(log(sum) + (maximum - true_max), -1e30, sum == 0.0);
        }
    }
}

// @section scalar_store
        for (var d = 0u; d < HD4 / 2u; d++) {
            dst[out * HD4 + half * (HD4 / 2u) + d] = output[d] / select(sum, 1.0, sum == 0.0);
        }

// @section scalar_pv
        for (var d = 0u; d < HD4 / 2u; d++) { output[d] *= correction; }
        for (var k = 0u; k < 32u; k++) {
            let p = select(0.0, exp(scores[row * SCORE_STRIDE + k / 4u][k % 4u] * scale - maximum),
                           base + k >= start && base + k < end);
            sum += p;
            for (var d = 0u; d < HD4 / 2u; d++) {
                output[d] = fma(vec4<f32>(p), values[k * (HD4 + 1u) + half * (HD4 / 2u) + d], output[d]);
            }
        }

// @section matrix_pv
        var local_sum = 0.0;
        for (var j = 0u; j < 4u; j++) {
            let key = base + half * 16u + 4u * j + vec4<u32>(0u, 1u, 2u, 3u);
            let valid = (key >= vec4<u32>(start)) & (key < vec4<u32>(end));
            let p = vec4<f16>(select(vec4<f32>(0.0), exp(scores[row * SCORE_STRIDE + half * 4u + j] * scale - maximum), valid));
            probs[row * 8u + half * 4u + j] = p;
            local_sum += dot(vec4<f32>(p), vec4<f32>(1.0));
        }
        sum += local_sum + subgroupShuffleXor(local_sum, 16u);
        workgroupBarrier();
        let fix = subgroupAny(rescale);
        if lane == 0u { flags[sg] = select(0u, 1u, fix); }
        let active_flags = workgroupUniformLoad(&flags);
        if (active_flags[0] | active_flags[1] | active_flags[2] | active_flags[3]) != 0u {
            $RESCALE_PV
        }
        for (var k = 0u; k < 32u; k += 16u) {
            let p = coopLoadT<coop_mat16x16<f16,A>>(&probs[sg * 16u * 8u + k / 4u], 8u);
            $ACCUM_PV
        }

// @section load_q
    let operand$INDEX = coopLoadT<coop_mat16x16<f16,A>>(&qk[sg * 16u * QK_STRIDE + $OFFSETu], QK_STRIDE);

// @section qk
            score = coopMultiplyAdd(operand$INDEX, coopLoad<coop_mat16x16<f16,B>>(&qk[key * QK_STRIDE + $OFFSETu], QK_STRIDE), score);

// @section accum_init
    var pv$INDEX = coop_mat16x16<f32,C>();

// @section rescale_pv
            coopStoreT(pv$INDEX, &scores[sg * 16u * SCORE_STRIDE], SCORE_STRIDE);
            workgroupBarrier();
            for (var d = 0u; d < 2u; d++) {
                scores[row * SCORE_STRIDE + half * 2u + d] *= correction;
            }
            workgroupBarrier();
            pv$INDEX = coopLoadT<coop_mat16x16<f32,C>>(&scores[sg * 16u * SCORE_STRIDE], SCORE_STRIDE);
            workgroupBarrier();

// @section accum_pv
            pv$INDEX = coopMultiplyAdd(p, coopLoadT<coop_mat16x16<f16,B>>(&qk[(32u + k) * QK_STRIDE + $OFFSETu], QK_STRIDE), pv$INDEX);

// @section finish_pv
    coopStoreT(pv$INDEX, &scores[sg * 16u * SCORE_STRIDE], SCORE_STRIDE);
    workgroupBarrier();
    if query < $Q_SEQ {
        for (var d = 0u; d < 2u; d++) {
            let index = (query * heads + head) * HD4 + $OFFSETu + half * 2u + d;
            dst[index] = scores[row * SCORE_STRIDE + half * 2u + d] / select(sum, 1.0, sum == 0.0);
        }
    }
    workgroupBarrier();
