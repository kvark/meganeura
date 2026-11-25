// @section main
enable f16;
enable wgpu_cooperative_matrix;
var<storage> d_out: array<vec4<f32>>;
var<storage> src_a: array<vec4<f32>>;
var<storage> src_b: array<vec4<f32>>;
var<storage> bias: array<vec4<f32>>;
var<storage> lse: array<f32>;
var<storage> fwd_dst: array<f32>;
var<storage, read_write> dst: array<vec4<f32>>;$SECOND_OUTPUT_DECL$
var<workgroup> hs: array<vec4<f16>, $OPERAND_VECS$>;
var<workgroup> fs: array<vec4<f32>, $SCORE_VECS$>;
var<workgroup> stats: array<vec4<f32>, $STAT_VECS$>;
var<workgroup> anchor: array<vec4<f32>, $HEAD_VECS$>;
$SCALING$
fn scaled(value: vec4<f32>, exponent: i32) -> vec4<f32> {
    if abs(exponent) <= 120 {
        return value * bitcast<f32>(u32(exponent + 127) << 23u);
    }
    // ldexp keeps extreme shifts separate even on drivers that reassociate multiplies.
    let first = clamp(exponent, -120, 120);
    let second = clamp(exponent - first, -120, 120);
    return ldexp(ldexp(ldexp(value, vec4<i32>(first)), vec4<i32>(second)), vec4<i32>(exponent - first - second));
}

// Keep each scale factor normal even when the product of the two scales
// would overflow or underflow F32.
fn restore_product(value: coop_mat16x16<f32,C>, exponent: i32) -> coop_mat16x16<f32,C> {
    var product = value;
    var remaining = exponent;
    while abs(remaining) > 120 {
        let shift = clamp(remaining, -120, 120);
        product = product * bitcast<f32>(u32(shift + 127) << 23u);
        remaining -= shift;
    }
    return product * bitcast<f32>(u32(remaining + 127) << 23u);
}

fn max_abs(v: vec4<f32>) -> f32 {
    let a = abs(v);
    return max(max(a.x, a.y), max(a.z, a.w));
}
@compute @workgroup_size($THREADS$)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(subgroup_id) sg: u32,
        @builtin(subgroup_invocation_id) lane: u32) {
    // The problem's sizes as literals: row indices divide by the group,
    // and bounds, loop limits and causality fold.
    let q_seq = $QUERY_COUNT$u;
    let causal = $CAUSAL$;
    let kv_seq = $KEY_COUNT$u;
    let heads = $HEADS$u;
    let kv_heads = $KV_HEADS$u;
    let window = $WINDOW$u;
    let group = $QUERY_GROUP$u;
    let scale = inverseSqrt(f32($HEAD_DIM$));
    let tid = sg * 32u + lane;
    let my_row = lane % 16u;
    let half = lane / 16u;
    let fbase = sg * $SCORES_PER_SUBGROUP$u;
    let tbase = $STAGE$u + sg * $PROB_PER_SUBGROUP$u;

// @section score_rows
        workgroupBarrier();
        {
            let srow = (fbase + my_row * $SCORE_STRIDE$u) / 4u + half * $KEY_VECS_PER_LANE$u;
            let urow = (fbase + $DP_OFFSET$u + my_row * $SCORE_STRIDE$u) / 4u + half * $KEY_VECS_PER_LANE$u;
            var big = 0.0;

// @section stage_keys
    let kv_head = wg.y;
    let k0 = wg.x * $ROWS$u;
    let last_k = min(k0 + $ROWS$u, kv_seq) - 1u;
    let begin = select(0u, k0, causal) * group;
    let end = select(q_seq, min(q_seq, last_k + window), window > 0u) * group;
    // Center K in F32 before scaling it for the matrix products.
    var maxima = vec2<f32>(0.0);
    for (var i = tid; i < $HEAD_VECS$u; i += $THREADS$u) { anchor[i] = src_b[(k0 * kv_heads + kv_head) * $HEAD_VECS$u + i]; }
    workgroupBarrier();
    for (var i = tid; i < $FIXED_VECS$u; i += $THREADS$u) {
        let key = k0 + i / $HEAD_VECS$u;
        if key < kv_seq {
            let at = (key * kv_heads + kv_head) * $HEAD_VECS$u + i % $HEAD_VECS$u;
            maxima = max(maxima, vec2<f32>(max_abs((src_b[at] - anchor[i % $HEAD_VECS$u]) * scale), max_abs(bias[at])));
        }
    }
    let fixed_exp = block_exponents(maxima, sg, 32u, lane);
    let kup = fixed_exp.x;
    let vup = fixed_exp.y;
    // K, then V, pass through one staging region on their way to operand
    // matrices: the step's rows need no more, and the smaller footprint fits
    // more workgroups per SM.
    for (var i = tid; i < $FIXED_VECS$u; i += $THREADS$u) {
        let key = k0 + i / $HEAD_VECS$u;
        var kx = vec4<f32>(0.0);
        if key < kv_seq {
            kx = (src_b[(key * kv_heads + kv_head) * $HEAD_VECS$u + i % $HEAD_VECS$u] - anchor[i % $HEAD_VECS$u]) * scale;
        }
        hs[(i / $HEAD_VECS$u) * $OPERAND_VEC_STRIDE$u + i % $HEAD_VECS$u] = vec4<f16>(scaled(kx, kup));
    }
    workgroupBarrier();

// @section stage_values
    workgroupBarrier();
    for (var i = tid; i < $FIXED_VECS$u; i += $THREADS$u) {
        let key = k0 + i / $HEAD_VECS$u;
        var vx = vec4<f32>(0.0);
        if key < kv_seq {
            vx = bias[(key * kv_heads + kv_head) * $HEAD_VECS$u + i % $HEAD_VECS$u];
        }
        hs[(i / $HEAD_VECS$u) * $OPERAND_VEC_STRIDE$u + i % $HEAD_VECS$u] = vec4<f16>(scaled(vx, vup));
    }
    workgroupBarrier();

// @section load_query_gradient
        let i$J$ = tid + $O$u;
        var d$J$ = vec4<f32>(0.0);
        var qx$J$ = vec4<f32>(0.0);
        var dot$J$ = 0.0;
        if i$J$ < $WALKED_VECS$u {
            let r = i$J$ / $HEAD_VECS$u;
            if q + r < end {
                let at = (((q + r) / group) * heads + kv_head * group + (q + r) % group) * $HEAD_VECS$u + i$J$ % $HEAD_VECS$u;
                qx$J$ = src_a[at];
                d$J$ = d_out[at];
            }
            dot$J$ = dot(qx$J$, anchor[i$J$ % $HEAD_VECS$u]);
        }
        let q$J$ = lane & 28u;
        let fv$J$ = vec4<f32>(
            subgroupShuffle(dot$J$, q$J$),
            subgroupShuffle(dot$J$, q$J$ + 1u),
            subgroupShuffle(dot$J$, q$J$ + 2u),
            subgroupShuffle(dot$J$, q$J$ + 3u),
        );
        if i$J$ < $WALKED_VECS$u && (lane & 3u) == 0u {
            fs[i$J$ / 4u] = fv$J$;
        }
        local_max = max(local_max, vec2<f32>(max_abs(qx$J$), max_abs(d$J$)));

// @section load_query_stats
        for (var i4 = tid; i4 < $KEY_VECS$u; i4 += $THREADS$u) {
            var mv = vec4<f32>(0.0);
            var lv = vec4<f32>(-1e30);
            var dv = vec4<f32>(0.0);
            for (var e = 0u; e < 4u; e++) {
                let i = i4 * 4u + e;
                if q + i < end {
                    let row = ((q + i) / group) * heads + kv_head * group + (q + i) % group;
                    mv[e] = lse[row * 2u];
                    lv[e] = lse[row * 2u + 1u];
                    dv[e] = fwd_dst[row];
                }
            }
            stats[i4] = mv;
            stats[$KEY_VECS$u + i4] = lv;
            stats[$GRAD_STATS_OFFSET$u + i4] = dv;
        }
        workgroupBarrier();
        let walk_exp = block_exponents(local_max, sg, 32u, lane);
        let qup = walk_exp.x;
        let dup = walk_exp.y;
        let score_down = -(fixed_exp.x + walk_exp.x);
        let dp_down = -(fixed_exp.y + walk_exp.y);

// @section center_query_stats
        for (var i4 = tid; i4 < $KEY_VECS$u; i4 += $THREADS$u) {
            var offs = vec4<f32>(0.0);
            for (var e = 0u; e < 4u; e++) {
                let i = i4 * 4u + e;
                var off = 0.0;
                for (var v = 0u; v < $DOT_VECS$u; v++) { off += dot(fs[i * $DOT_VECS$u + v], vec4<f32>(1.0)); }
                offs[e] = off * scale;
            }
            stats[i4] -= offs;
        }
        workgroupBarrier();

// @section kv_derivative
            var p$J$ = vec4<f32>(0.0);
            var ds$J$ = vec4<f32>(0.0);
            {
                let qi = half * $KEY_VECS_PER_LANE$u * 4u + $J$u * 4u;
                let sv = scaled(fs[srow + $J$u], score_down);
                let dv = scaled(fs[urow + $J$u], dp_down);
                let qq = vec4<u32>(q + qi) + vec4<u32>(0u, 1u, 2u, 3u);
                let qpos = qq / vec4<u32>(group);
                let ok = (qq < vec4<u32>(end)) & vec4<bool>(key$R$ < kv_seq)
                    & (vec4<bool>(!causal) | (qpos >= vec4<u32>(key$R$)))
                    & (vec4<bool>(window == 0u) | (qpos < vec4<u32>(key$R$ + window)));
                let pe = select(vec4<f32>(0.0), exp(min(sv - stats[qi / 4u], vec4<f32>(0.0)) - stats[$KEY_VECS$u + qi / 4u]), ok);
                p$J$ = pe;
                ds$J$ = pe * (dv - stats[$GRAD_STATS_OFFSET$u + qi / 4u]);
            }
            big = max(big, max_abs(ds$J$));
            prob_max = max(prob_max, max_abs(p$J$));

// @section stage_queries
    let head = wg.y;
    let kv_head = head / group;
    let q0 = wg.x * $ROWS$u;
    let last_q = min(q0 + $ROWS$u, q_seq) - 1u;
    let first_kv_len = select(kv_seq, min(kv_seq, q0 + 1u), causal);
    let k_begin = select(0u, first_kv_len - min(first_kv_len, window), window > 0u);
    let end = select(kv_seq, last_q + 1u, causal);
    var maxima = vec2<f32>(0.0);
    for (var i = tid; i < $HEAD_VECS$u; i += $THREADS$u) { anchor[i] = src_b[(min(k_begin, kv_seq - 1u) * kv_heads + kv_head) * $HEAD_VECS$u + i]; }
    workgroupBarrier();
    for (var i = tid; i < $FIXED_VECS$u; i += $THREADS$u) {
        let r = i / $HEAD_VECS$u;
        if q0 + r < q_seq {
            let at = ((q0 + r) * heads + head) * $HEAD_VECS$u + i % $HEAD_VECS$u;
            maxima = max(maxima, vec2<f32>(max_abs(src_a[at] * scale), max_abs(d_out[at])));
        }
    }
    let fixed_exp = block_exponents(maxima, sg, 32u, lane);
    let qup = fixed_exp.x;
    let dup = fixed_exp.y;
    // Q, then dO, pass through one staging region, as K and V do in dK/dV.
    for (var i = tid; i < $FIXED_VECS$u; i += $THREADS$u) {
        let r = i / $HEAD_VECS$u;
        var qx = vec4<f32>(0.0);
        if q0 + r < q_seq {
            qx = src_a[((q0 + r) * heads + head) * $HEAD_VECS$u + i % $HEAD_VECS$u] * scale;
        }
        hs[r * $OPERAND_VEC_STRIDE$u + i % $HEAD_VECS$u] = vec4<f16>(scaled(qx, qup));
    }
    workgroupBarrier();

// @section stage_output_gradient
    workgroupBarrier();
    for (var i = tid; i < $FIXED_VECS$u; i += $THREADS$u) {
        let r = i / $HEAD_VECS$u;
        var dx = vec4<f32>(0.0);
        if q0 + r < q_seq {
            dx = d_out[((q0 + r) * heads + head) * $HEAD_VECS$u + i % $HEAD_VECS$u];
        }
        hs[r * $OPERAND_VEC_STRIDE$u + i % $HEAD_VECS$u] = vec4<f16>(scaled(dx, dup));
    }
    workgroupBarrier();

// @section query_stats
    let pos$R$ = q0 + sg * $SUBGROUP_ROWS$u + $ROW_OFFSET$u + my_row;
    var m$R$ = 0.0;
    var ll$R$ = -1e30;
    var dd$R$ = 0.0;
    var off$R$ = 0.0;
    if pos$R$ < q_seq {
        let row = pos$R$ * heads + head;
        m$R$ = lse[row * 2u];
        ll$R$ = lse[row * 2u + 1u];
        dd$R$ = fwd_dst[row];
        for (var d4 = 0u; d4 < $HEAD_VECS$u; d4++) { off$R$ += dot(src_a[row * $HEAD_VECS$u + d4], anchor[d4]); }
        off$R$ *= scale;
    }
    var dsdown$R$: i32;

// @section load_keys_values
        let i$J$ = tid + $O$u;
        var kx$J$ = vec4<f32>(0.0);
        var vx$J$ = vec4<f32>(0.0);
        if i$J$ < $WALKED_VECS$u && t + i$J$ / $HEAD_VECS$u < end {
            let at = ((t + i$J$ / $HEAD_VECS$u) * kv_heads + kv_head) * $HEAD_VECS$u + i$J$ % $HEAD_VECS$u;
            kx$J$ = src_b[at] - anchor[i$J$ % $HEAD_VECS$u];
            vx$J$ = bias[at];
        }

        local_max = max(local_max, vec2<f32>(max_abs(kx$J$), max_abs(vx$J$)));

// @section query_derivative
            var ds$J$ = vec4<f32>(0.0);
            {
                let kp = t + half * $KEY_VECS_PER_LANE$u * 4u + $J$u * 4u;
                let qlim = select(kv_seq, pos$R$ + 1u, causal);
                let qlo = select(0u, qlim - min(qlim, window), window > 0u);
                let kvec = vec4<u32>(kp, kp + 1u, kp + 2u, kp + 3u);
                let ok = (kvec < vec4<u32>(end))
                    & vec4<bool>(pos$R$ < q_seq)
                    & (kvec >= vec4<u32>(qlo))
                    & (kvec < vec4<u32>(qlim));
                let p = select(vec4<f32>(0.0), exp(min(scaled(fs[srow + $J$u], score_down) + off$R$ - m$R$, vec4<f32>(0.0)) - ll$R$), ok);
                ds$J$ = p * (scaled(fs[urow + $J$u], dp_down) - dd$R$);
            }
            big = max(big, max_abs(ds$J$));
