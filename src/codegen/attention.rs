//! Attention and cooperative convolution kernels.

use super::*;

/// Workgroup storage for the fixed 16-query, 16-key cooperative tiles.
/// The fixed terms include score tiles and backward row statistics.
pub(crate) fn attention_coop_shared_bytes(group: ShaderGroup, head_dim: u32) -> u64 {
    let (per_dim, fixed) = match group {
        ShaderGroup::FlashAttentionCoop => (128u64, 1024),
        ShaderGroup::FlashGradQCoop => (192, 3264),
        ShaderGroup::FlashGradKVCoop => (256, 4544),
        _ => unreachable!("shared-memory accounting requires cooperative attention"),
    };
    per_dim * u64::from(head_dim) + fixed
}

pub fn attention_lanes(head_dim: u32, ept_cap: u32) -> (u32, u32) {
    assert!(head_dim >= 1, "attention needs a nonempty head");
    let padded = head_dim.next_power_of_two();
    let widest = padded / padded.min(ept_cap.max(1));
    let tpq = widest.min(1 << head_dim.trailing_zeros());
    (head_dim / tpq, tpq)
}

/// Elements per thread of the cached multi-query kernel.
pub(super) const CACHED_ATTENTION_EPT: u32 = 8;

/// Queries per workgroup of the cached multi-query kernel for `head_dim`.
pub fn cached_attention_queries(head_dim: u32) -> u32 {
    let (_, tpq) = attention_lanes(head_dim, CACHED_ATTENTION_EPT);
    FlashAttentionShape::default().threads / tpq
}

/// The cached multi-query attention kernel for `head_dim`.
pub(crate) fn generate_cached_query_attention_module(head_dim: u32) -> ShaderModule {
    generate_flash_attention(
        head_dim,
        CACHED_ATTENTION_EPT,
        FlashAttentionShape::default(),
        true,
    )
}

pub fn generate_flash_attention_module(
    head_dim: u32,
    ept_cap: u32,
    shape: FlashAttentionShape,
) -> ShaderModule {
    generate_flash_attention(head_dim, ept_cap, shape, false)
}

pub(super) fn generate_flash_attention(
    head_dim: u32,
    ept_cap: u32,
    shape: FlashAttentionShape,
    cached: bool,
) -> ShaderModule {
    use std::fmt::Write;
    assert!(matches!(shape.threads, 128 | 256) && shape.keys.is_power_of_two() && shape.keys <= 16);

    let hd = head_dim;
    // which honors MEGANEURA_FLASH_EPT_CAP overrides.
    let (ept, tpq) = attention_lanes(hd, ept_cap);
    let d_stride = if shape.interleave { tpq } else { 1 };
    let bq: u32 = (shape.threads / tpq).max(1);
    // Fall back to BQ=1 kernel when multi-query isn't beneficial
    if bq <= 1 {
        assert!(
            !cached,
            "cached attention needs several queries per workgroup"
        );
        return generate_attention_module(head_dim);
    }
    let wg_size = bq * tpq;
    let bkv: u32 = shape.keys;
    let mut src = String::new();

    // Params struct (matches AttentionParams: 8 u32 = 32 bytes)
    if cached {
        src.push_str(CACHED_ATTENTION_PARAMS_WGSL);
    } else {
        src.push_str(ATTENTION_PARAMS_WGSL);
    }
    src.push_str("var<storage> src_a: array<f32>;\n"); // Q
    src.push_str("var<storage> src_b: array<f32>;\n"); // K
    src.push_str("var<storage> bias: array<f32>;\n"); // V
    if cached {
        src.push_str("var<storage> kv_pos_buf: array<u32>;\n");
    }
    src.push_str("var<storage, read_write> dst: array<f32>;\n"); // O
    if !cached {
        src.push_str("var<storage, read_write> lse: array<f32>;\n");
    }
    src.push_str("var<uniform> params: Params;\n\n");

    // Shared memory:
    //   shared_k: K tile [BKV, hd] loaded once, reused by BQ groups
    //   wg_scores: [BKV][BQ][TPQ] keeps neighboring threads contiguous
    //   wg_dot: [BQ][TPQ] tail reduction
    let _ = writeln!(src, "var<workgroup> shared_k: array<f32, {}>;\n", bkv * hd);
    let _ = writeln!(src, "var<workgroup> shared_v: array<f32, {}>;\n", bkv * hd);
    let _ = writeln!(
        src,
        "var<workgroup> wg_scores: array<f32, {}>;\n",
        bq * bkv * tpq
    );
    let _ = writeln!(src, "var<workgroup> wg_dot: array<f32, {}>;\n", bq * tpq);

    // Grouped tree_reduce for BKV scores: each group of TPQ threads reduces independently.
    src.push_str("fn tree_reduce_bkv_grouped(tid: u32) {\n");
    let _ = writeln!(src, "    let qi = tid / {tpq}u;");
    let _ = writeln!(src, "    let local = tid % {tpq}u;");
    let _ = writeln!(src, "    let base = qi * {tpq}u;");
    let mut stride = tpq / 2;
    while stride > 0 {
        src.push_str("    workgroupBarrier();\n");
        let _ = writeln!(src, "    if local < {stride}u {{");
        let _ = writeln!(src, "        for (var i = 0u; i < {bkv}u; i++) {{");
        let _ = writeln!(
            src,
            "            wg_scores[i * {wg_size}u + base + local] += wg_scores[i * {wg_size}u + base + local + {stride}u];"
        );
        src.push_str("        }\n    }\n");
        stride /= 2;
    }
    src.push_str("    workgroupBarrier();\n}\n\n");

    // Grouped tree_reduce for tail (single dot product)
    src.push_str("fn tree_reduce_grouped(tid: u32) {\n");
    let _ = writeln!(src, "    let qi = tid / {tpq}u;");
    let _ = writeln!(src, "    let local = tid % {tpq}u;");
    let _ = writeln!(src, "    let base = qi * {tpq}u;");
    stride = tpq / 2;
    while stride > 0 {
        src.push_str("    workgroupBarrier();\n");
        let _ = writeln!(
            src,
            "    if local < {stride}u {{ wg_dot[base + local] += wg_dot[base + local + {stride}u]; }}"
        );
        stride /= 2;
    }
    src.push_str("    workgroupBarrier();\n}\n\n");

    // Main kernel
    let _ = writeln!(src, "@compute @workgroup_size({wg_size})");
    src.push_str(
        "fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n",
    );
    let _ = writeln!(src, "    let qi = lid.x / {tpq}u;"); // query within tile
    let _ = writeln!(src, "    let lane = lid.x % {tpq}u;"); // lane within query group
    let _ = writeln!(
        src,
        "    let d_base = lane * {}u;",
        if shape.interleave { 1 } else { ept }
    );
    let _ = writeln!(src, "    let pos = wgid.x * {bq}u + qi;"); // global query position
    src.push_str("    let head = wgid.y;\n");
    src.push_str("    let q_seq = params.q_seq;\n");
    if cached {
        src.push_str("    let kv_seq = kv_pos_buf[0] + 1u;\n");
        src.push_str("    let num_heads = params.num_heads;\n");
        src.push_str("    let num_kv_heads = params.num_kv_heads;\n");
    } else {
        src.push_str("    let kv_seq = params.kv_seq;\n");
        src.push_str("    let num_heads = params.packed_heads >> 16u;\n");
        src.push_str("    let num_kv_heads = params.packed_heads & 0xFFFFu;\n");
    }
    src.push_str("    let head_dim = params.head_dim;\n");
    src.push_str("    let valid = pos < q_seq && head < num_heads;\n\n");

    // Per-position KV range (causal + sliding window)
    src.push_str(
        "    let my_kv_len = select(kv_seq, select(pos + 1u, 0u, !valid), kv_seq == 0u);\n",
    );
    src.push_str(if cached {
        "    let window_size = 0u;\n"
    } else {
        "    let window_size = params.window_size;\n"
    });
    src.push_str("    let my_kv_start = select(0u, my_kv_len - min(my_kv_len, window_size), window_size > 0u);\n\n");

    // Workgroup-wide loop bounds
    let _ = writeln!(
        src,
        "    let last_pos = min(wgid.x * {bq}u + {bq}u - 1u, q_seq - 1u);"
    );
    let _ = writeln!(src, "    let first_pos = wgid.x * {bq}u;");
    src.push_str("    let max_kv_len = select(kv_seq, last_pos + 1u, kv_seq == 0u);\n");
    src.push_str("    let first_kv_len = select(kv_seq, first_pos + 1u, kv_seq == 0u);\n");
    src.push_str("    let min_kv_start = select(0u, first_kv_len - min(first_kv_len, window_size), window_size > 0u);\n\n");

    // GQA head mapping
    src.push_str("    let kv_head = head / (num_heads / max(num_kv_heads, 1u));\n");
    src.push_str("    let kv_head_off = kv_head * head_dim;\n");
    src.push_str("    let kv_dim = num_kv_heads * head_dim;\n");
    src.push_str("    let scale = inverseSqrt(f32(head_dim));\n");

    // Load Q values for this thread's EPT elements (registers)
    for e in 0..ept {
        let _ = writeln!(src, "    var q{e} = 0.0;");
    }
    src.push_str("    if valid {\n");
    src.push_str("        let q_base = pos * (num_heads * head_dim) + head * head_dim;\n");
    for e in 0..ept {
        let _ = writeln!(
            src,
            "        q{e} = src_a[q_base + d_base + {}u];",
            e * d_stride
        );
    }
    src.push_str("    }\n\n");

    // Online softmax accumulators: EPT output elements per thread
    for e in 0..ept {
        let _ = writeln!(src, "    var out{e} = 0.0;");
    }
    src.push_str("    var max_score = -1e30;\n");
    src.push_str("    var sum_exp = 0.0;\n\n");

    // --- Tiled KV loop with shared K staging ---
    let _ = writeln!(src, "    let kv_range = max_kv_len - min_kv_start;");
    let _ = writeln!(
        src,
        "    let tile_end = min_kv_start + (kv_range / {bkv}u) * {bkv}u;"
    );
    src.push_str("    var t = min_kv_start;\n");
    let _ = writeln!(src, "    for (; t < tile_end; t += {bkv}u) {{");

    // Cooperatively load K tile into shared memory
    let k_tile_size = bkv * hd;
    let loads_per_thread = k_tile_size.div_ceil(wg_size);
    for l in 0..loads_per_thread {
        let offset = l * wg_size;
        if offset == 0 {
            let _ = writeln!(src, "        if lid.x < {k_tile_size}u {{");
            let _ = writeln!(src, "            let ki = lid.x / {hd}u;");
            src.push_str(
                "            shared_k[lid.x] = src_b[(t + ki) * kv_dim + kv_head_off + (lid.x % head_dim)];\n",
            );
            src.push_str(
                "            shared_v[lid.x] = bias[(t + ki) * kv_dim + kv_head_off + (lid.x % head_dim)];\n",
            );
            src.push_str("        }\n");
        } else {
            let _ = writeln!(src, "        if lid.x + {offset}u < {k_tile_size}u {{");
            let _ = writeln!(src, "            let ki2 = (lid.x + {offset}u) / {hd}u;");
            let _ = writeln!(
                src,
                "            shared_k[lid.x + {offset}u] = src_b[(t + ki2) * kv_dim + kv_head_off + ((lid.x + {offset}u) % head_dim)];"
            );
            let _ = writeln!(
                src,
                "            shared_v[lid.x + {offset}u] = bias[(t + ki2) * kv_dim + kv_head_off + ((lid.x + {offset}u) % head_dim)];"
            );
            src.push_str("        }\n");
        }
    }
    src.push_str("        workgroupBarrier();\n\n");

    // Each thread computes partial dot product (EPT elements) for BKV positions
    let _ = writeln!(src, "        let grp_base = qi * {tpq}u;");
    let _ = writeln!(src, "        for (var i = 0u; i < {bkv}u; i++) {{");
    // Compute partial dot product across EPT elements
    src.push_str("            var pdot = 0.0;\n");
    for e in 0..ept {
        let _ = writeln!(
            src,
            "            pdot += q{e} * shared_k[i * {hd}u + d_base + {}u];",
            e * d_stride
        );
    }
    let _ = writeln!(
        src,
        "            wg_scores[i * {wg_size}u + grp_base + lane] = pdot;"
    );
    src.push_str("        }\n");
    src.push_str("        tree_reduce_bkv_grouped(lid.x);\n\n");

    // Online softmax + V accumulation for BKV positions
    let _ = writeln!(src, "        for (var i = 0u; i < {bkv}u; i++) {{");
    src.push_str("            let kv_pos = t + i;\n");
    src.push_str("            if valid && kv_pos >= my_kv_start && kv_pos < my_kv_len {\n");
    let _ = writeln!(
        src,
        "                let score = wg_scores[i * {wg_size}u + grp_base] * scale;"
    );
    src.push_str("                let new_max = max(max_score, score);\n");
    src.push_str("                let correction = exp(max_score - new_max);\n");
    src.push_str("                let weight = exp(score - new_max);\n");
    src.push_str("                sum_exp = sum_exp * correction + weight;\n");
    // Accumulate EPT V elements in registers
    for e in 0..ept {
        let _ = writeln!(
            src,
            "                out{e} = out{e} * correction + weight * shared_v[i * {hd}u + d_base + {}u];",
            e * d_stride
        );
    }
    src.push_str("                max_score = new_max;\n");
    src.push_str("            }\n");
    src.push_str("        }\n");
    src.push_str("        workgroupBarrier();\n");
    src.push_str("    }\n\n");

    // --- Tail: remaining KV positions one at a time ---
    src.push_str("    for (; t < max_kv_len; t++) {\n");
    // Load single K position into shared_k
    let _ = writeln!(
        src,
        "        for (var d = lid.x; d < {hd}u; d += {wg_size}u) {{"
    );
    src.push_str("            shared_k[d] = src_b[t * kv_dim + kv_head_off + d];\n");
    src.push_str("        }\n");
    src.push_str("        workgroupBarrier();\n\n");

    // Each thread computes partial dot product
    let _ = writeln!(src, "        let dot_base = qi * {tpq}u;");
    src.push_str("        var pdot2 = 0.0;\n");
    for e in 0..ept {
        let _ = writeln!(
            src,
            "        pdot2 += q{e} * shared_k[d_base + {}u];",
            e * d_stride
        );
    }
    src.push_str("        wg_dot[dot_base + lane] = pdot2;\n");
    src.push_str("        tree_reduce_grouped(lid.x);\n");
    let _ = writeln!(src, "        let score = wg_dot[qi * {tpq}u] * scale;\n");

    src.push_str("        if valid && t >= my_kv_start && t < my_kv_len {\n");
    src.push_str("            let new_max = max(max_score, score);\n");
    src.push_str("            let correction = exp(max_score - new_max);\n");
    src.push_str("            let weight = exp(score - new_max);\n");
    src.push_str("            sum_exp = sum_exp * correction + weight;\n");
    src.push_str("            let v_base2 = t * kv_dim + kv_head_off;\n");
    for e in 0..ept {
        let _ = writeln!(
            src,
            "            out{e} = out{e} * correction + weight * bias[v_base2 + d_base + {}u];",
            e * d_stride
        );
    }
    src.push_str("            max_score = new_max;\n");
    src.push_str("        }\n");
    src.push_str("        workgroupBarrier();\n");
    src.push_str("    }\n\n");

    // Final output + LSE
    src.push_str("    if valid {\n");
    src.push_str("        let q_base = pos * (num_heads * head_dim) + head * head_dim;\n");
    src.push_str("        let safe_sum = select(sum_exp, 1.0, sum_exp == 0.0);\n");
    for e in 0..ept {
        let _ = writeln!(
            src,
            "        dst[q_base + d_base + {}u] = out{e} / safe_sum;",
            e * d_stride
        );
    }

    if !cached {
        // LSE output: only first thread in each group.
        src.push_str("        if lane == 0u {\n");
        src.push_str("            let idx = (pos * num_heads + head) * 2u;\n");
        src.push_str("            lse[idx] = max_score;\n");
        src.push_str("            lse[idx + 1u] = select(log(sum_exp), -1e30, sum_exp == 0.0);\n");
        src.push_str("        }\n");
    }
    src.push_str("    }\n");
    src.push_str("}\n");

    let module = parse_source(&src).unwrap_or_else(|e| {
        panic!(
            "generated flash attention WGSL failed to parse:\n{}\n---\n{}",
            e, src
        )
    });
    ShaderModule {
        module,
        source: src,
        hint: if cached {
            "cached_query_attention"
        } else {
            "flash_attention"
        },
    }
}

/// Cooperative forward attention: QKᵀ uses f16 matrix operands, while softmax
/// and PV accumulate in f32 registers. Dispatch `[ceil(q_seq/16), num_heads, 1]`.
pub fn generate_flash_attention_coop_module(head_dim: u32) -> ShaderModule {
    generate_coop_attention(
        include_str!("../shaders/flash_attention_coop.wgsl"),
        head_dim,
        "flash_attention_coop",
    )
}

/// Cooperative dQ: dispatch `[ceil(q_seq/16), num_heads, 1]`.
pub fn generate_flash_grad_q_coop_module(head_dim: u32) -> ShaderModule {
    generate_coop_attention(
        include_str!("../shaders/flash_grad_q_coop.wgsl"),
        head_dim,
        "flash_grad_q_coop",
    )
}

/// Cooperative dK/dV: dispatch `[ceil(dispatch_kv/16), num_kv_heads, 1]`.
pub fn generate_flash_grad_kv_coop_module(head_dim: u32) -> ShaderModule {
    generate_coop_attention(
        include_str!("../shaders/flash_grad_kv_coop.wgsl"),
        head_dim,
        "flash_grad_kv_coop",
    )
}

fn generate_coop_attention(template: &str, head_dim: u32, hint: &'static str) -> ShaderModule {
    assert!(
        head_dim >= 16 && head_dim.is_multiple_of(16),
        "{hint} requires head_dim multiple of 16, got {head_dim}"
    );
    let source = preprocess(
        template,
        &[
            ("$PARAMS", ATTENTION_PARAMS_WGSL.trim_end()),
            ("$HEAD_DIM", &head_dim.to_string()),
            ("$HEAD_TILES", &(head_dim / 16).to_string()),
            ("$CHUNK_HD", &(head_dim / 4).to_string()),
            ("$TILE_ELEMENTS", &(head_dim * 16).to_string()),
        ],
    );
    let module = parse_source(&source)
        .unwrap_or_else(|e| panic!("generated {hint} WGSL failed to parse:\n{e}\n---\n{source}"));
    ShaderModule {
        module,
        source,
        hint,
    }
}

/// Generate a Flash Attention 2 backward dQ kernel using vectorized register
/// pattern (each thread computes full Q·K / dO·V dot products in registers).
///
/// Mirrors the forward kernel's EPT/TPQ/BQ pattern. When TPQ=1 (EPT==hd)
/// there are NO workgroup barriers inside the KV loop — each thread owns
/// one full query row and sequentially accumulates dQ by loading K/V
/// scalars directly from global memory.
pub fn generate_flash_grad_q_module(head_dim: u32, ept_cap: u32) -> ShaderModule {
    use std::fmt::Write;
    let hd = head_dim;
    // Backward uses its own cap because these kernels carry more live state.
    let (ept, tpq) = attention_lanes(hd, ept_cap);
    let bq: u32 = (256 / tpq).max(1);
    if bq <= 1 {
        // Fall back to hand-written shader
        return ShaderModule::new(include_str!("../shaders/mha_grad_q.wgsl"));
    }
    let wg_size = bq * tpq;
    let mut src = String::new();

    // Params + bindings (match MultiHeadAttnGradData)
    src.push_str(ATTENTION_PARAMS_WGSL);
    src.push_str("var<storage> d_out: array<f32>;\n");
    src.push_str("var<storage> src_a: array<f32>;\n"); // Q
    src.push_str("var<storage> src_b: array<f32>;\n"); // K
    src.push_str("var<storage> bias: array<f32>;\n"); // V
    src.push_str("var<storage> lse: array<f32>;\n");
    src.push_str("var<storage> fwd_dst: array<f32>;\n"); // D = rowsum(dO * O)
    src.push_str("var<storage, read_write> dst: array<f32>;\n"); // dQ
    src.push_str("var<uniform> params: Params;\n\n");

    // Shared K/V staging with BKV tiling: amortize barrier cost by loading
    // BKV KV positions worth of K and V at once, then looping in-register.
    let bkv: u32 = if tpq == 1 { 8 } else { 1 };
    let _ = writeln!(src, "var<workgroup> shared_k: array<f32, {}>;", bkv * hd);
    let _ = writeln!(src, "var<workgroup> shared_v: array<f32, {}>;\n", bkv * hd);

    // Shared memory only needed when TPQ > 1 (for cross-lane reductions).
    if tpq > 1 {
        // wg_score[qi*tpq + lane] partial Q·K
        // wg_dp   [qi*tpq + lane] partial dO·V
        let _ = writeln!(src, "var<workgroup> wg_score: array<f32, {}>;", bq * tpq);
        let _ = writeln!(src, "var<workgroup> wg_dp: array<f32, {}>;", bq * tpq);

        // Grouped tree_reduce for wg_score and wg_dp simultaneously
        src.push_str("fn reduce_score_dp(tid: u32) {\n");
        let _ = writeln!(src, "    let local = tid % {tpq}u;");
        let _ = writeln!(src, "    let base = (tid / {tpq}u) * {tpq}u;");
        let mut stride = tpq / 2;
        while stride > 0 {
            src.push_str("    workgroupBarrier();\n");
            let _ = writeln!(
                src,
                "    if local < {stride}u {{ wg_score[base + local] += wg_score[base + local + {stride}u]; wg_dp[base + local] += wg_dp[base + local + {stride}u]; }}"
            );
            stride /= 2;
        }
        src.push_str("    workgroupBarrier();\n}\n\n");
    }

    // Main kernel
    let _ = writeln!(src, "@compute @workgroup_size({wg_size})");
    src.push_str("fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n");
    let _ = writeln!(src, "    let qi = lid.x / {tpq}u;");
    let _ = writeln!(src, "    let lane = lid.x % {tpq}u;");
    let _ = writeln!(src, "    let d_base = lane * {ept}u;");
    let _ = writeln!(src, "    let pos = wgid.x * {bq}u + qi;");
    src.push_str("    let head = wgid.y;\n");
    src.push_str("    let q_seq = params.q_seq;\n");
    src.push_str("    let kv_seq = params.kv_seq;\n");
    src.push_str("    let num_heads = params.packed_heads >> 16u;\n");
    src.push_str("    let num_kv_heads = params.packed_heads & 0xFFFFu;\n");
    src.push_str("    let head_dim = params.head_dim;\n");
    src.push_str("    let valid = pos < q_seq && head < num_heads;\n\n");

    src.push_str("    let kv_head = head / (num_heads / max(num_kv_heads, 1u));\n");
    src.push_str("    let kv_head_off = kv_head * head_dim;\n");
    src.push_str("    let kv_dim = num_kv_heads * head_dim;\n");
    src.push_str("    let scale = inverseSqrt(f32(head_dim));\n");
    src.push_str("    var max_s = 0.0;\n    var log_sum = 0.0;\n    var q_base = 0u;\n");
    // Load Q and dO into thread-local registers
    for e in 0..ept {
        let _ = writeln!(src, "    var q{e} = 0.0;");
        let _ = writeln!(src, "    var do{e} = 0.0;");
    }
    src.push_str("    if valid {\n");
    src.push_str("        q_base = pos * (num_heads * head_dim) + head * head_dim;\n");
    for e in 0..ept {
        let _ = writeln!(src, "        q{e} = src_a[q_base + d_base + {e}u];");
        let _ = writeln!(src, "        do{e} = d_out[q_base + d_base + {e}u];");
    }
    src.push_str("        let lse_idx = (pos * num_heads + head) * 2u;\n");
    src.push_str("        max_s = lse[lse_idx];\n");
    src.push_str("        log_sum = lse[lse_idx + 1u];\n");
    src.push_str("    }\n\n");

    // D = rowsum(dO * O) is shared with the dK/dV kernel.
    src.push_str("    var row_sum = 0.0;\n");
    src.push_str("    if valid {\n");
    src.push_str("        row_sum = fwd_dst[pos * num_heads + head];\n");
    src.push_str("    }\n\n");

    // Per-position KV range
    src.push_str(
        "    let my_kv_len = select(kv_seq, select(pos + 1u, 0u, !valid), kv_seq == 0u);\n",
    );
    src.push_str("    let window = params.window_size;\n");
    src.push_str(
        "    let my_kv_start = select(0u, my_kv_len - min(my_kv_len, window), window > 0u);\n",
    );
    // Workgroup-wide bounds (all threads must agree for potential barriers)
    let _ = writeln!(
        src,
        "    let last_pos = min(wgid.x * {bq}u + {bq}u - 1u, q_seq - 1u);"
    );
    let _ = writeln!(src, "    let first_pos = wgid.x * {bq}u;");
    src.push_str("    let max_kv_len = select(kv_seq, last_pos + 1u, kv_seq == 0u);\n");
    src.push_str("    let first_kv_len = select(kv_seq, first_pos + 1u, kv_seq == 0u);\n");
    src.push_str("    let min_kv_start = select(0u, first_kv_len - min(first_kv_len, window), window > 0u);\n\n");

    // Per-thread dQ accumulators (EPT elements)
    for e in 0..ept {
        let _ = writeln!(src, "    var dq{e} = 0.0;");
    }
    src.push('\n');

    if tpq == 1 {
        // Tiled KV loop with BKV positions per barrier (only when tpq=1).
        let _ = writeln!(src, "    let kv_range = max_kv_len - min_kv_start;");
        let _ = writeln!(
            src,
            "    let tile_end = min_kv_start + (kv_range / {bkv}u) * {bkv}u;"
        );
        src.push_str("    var t = min_kv_start;\n");
        let _ = writeln!(src, "    for (; t < tile_end; t += {bkv}u) {{");
        // Cooperative tile load (wg_size threads load BKV*hd elements)
        let tile_size = bkv * hd;
        let loads_per_thread = tile_size.div_ceil(wg_size);
        for l in 0..loads_per_thread {
            let off = l * wg_size;
            if off == 0 {
                let _ = writeln!(src, "        if lid.x < {tile_size}u {{");
                let _ = writeln!(src, "            let ki = lid.x / {hd}u;");
                let _ = writeln!(src, "            let kd = lid.x % {hd}u;");
                src.push_str("            let kb = (t + ki) * kv_dim + kv_head_off;\n");
                src.push_str("            shared_k[lid.x] = src_b[kb + kd];\n");
                src.push_str("            shared_v[lid.x] = bias[kb + kd];\n");
                src.push_str("        }\n");
            } else {
                let _ = writeln!(src, "        if lid.x + {off}u < {tile_size}u {{");
                let _ = writeln!(src, "            let ki = (lid.x + {off}u) / {hd}u;");
                let _ = writeln!(src, "            let kd = (lid.x + {off}u) % {hd}u;");
                src.push_str("            let kb = (t + ki) * kv_dim + kv_head_off;\n");
                let _ = writeln!(
                    src,
                    "            shared_k[lid.x + {off}u] = src_b[kb + kd];"
                );
                let _ = writeln!(src, "            shared_v[lid.x + {off}u] = bias[kb + kd];");
                src.push_str("        }\n");
            }
        }
        src.push_str("        workgroupBarrier();\n\n");

        // Inner loop: BKV positions, in registers, no barriers
        let _ = writeln!(src, "        for (var i = 0u; i < {bkv}u; i++) {{");
        src.push_str("            let kv_pos = t + i;\n");
        let _ = writeln!(src, "            let k_off = i * {hd}u + d_base;");
        src.push_str("            var score_part = 0.0;\n");
        src.push_str("            var dp_part = 0.0;\n");
        for e in 0..ept {
            let _ = writeln!(
                src,
                "            score_part += q{e} * shared_k[k_off + {e}u];"
            );
            let _ = writeln!(
                src,
                "            dp_part += do{e} * shared_v[k_off + {e}u];"
            );
        }
        src.push_str("            let score = score_part * scale;\n");
        src.push_str("            if valid && kv_pos >= my_kv_start && kv_pos < my_kv_len {\n");
        src.push_str("                let p_t = exp(min(score - max_s, 0.0) - log_sum);\n");
        src.push_str("                let ds_t = p_t * (dp_part - row_sum);\n");
        src.push_str("                let w = ds_t * scale;\n");
        for e in 0..ept {
            let _ = writeln!(src, "                dq{e} += w * shared_k[k_off + {e}u];");
        }
        src.push_str("            }\n");
        src.push_str("        }\n");
        src.push_str("        workgroupBarrier();\n");
        src.push_str("    }\n\n");

        // Tail: remaining KV positions one at a time
        src.push_str("    for (; t < max_kv_len; t++) {\n");
        src.push_str("        let k_base = t * kv_dim + kv_head_off;\n");
        let _ = writeln!(
            src,
            "        for (var d = lid.x; d < {hd}u; d += {wg_size}u) {{"
        );
        src.push_str("            shared_k[d] = src_b[k_base + d];\n");
        src.push_str("            shared_v[d] = bias[k_base + d];\n");
        src.push_str("        }\n");
        src.push_str("        workgroupBarrier();\n");
        src.push_str("        var sp2 = 0.0;\n");
        src.push_str("        var dp2 = 0.0;\n");
        for e in 0..ept {
            let _ = writeln!(src, "        sp2 += q{e} * shared_k[d_base + {e}u];");
            let _ = writeln!(src, "        dp2 += do{e} * shared_v[d_base + {e}u];");
        }
        src.push_str("        let score2 = sp2 * scale;\n");
        src.push_str("        if valid && t >= my_kv_start && t < my_kv_len {\n");
        src.push_str("            let p_t = exp(min(score2 - max_s, 0.0) - log_sum);\n");
        src.push_str("            let ds_t = p_t * (dp2 - row_sum);\n");
        src.push_str("            let w = ds_t * scale;\n");
        for e in 0..ept {
            let _ = writeln!(src, "            dq{e} += w * shared_k[d_base + {e}u];");
        }
        src.push_str("        }\n");
        src.push_str("        workgroupBarrier();\n");
        src.push_str("    }\n\n");
    } else {
        // TPQ>1 path: single KV position per iteration with cross-lane reduction.
        src.push_str("    for (var t = min_kv_start; t < max_kv_len; t++) {\n");
        src.push_str("        let k_base = t * kv_dim + kv_head_off;\n");
        let _ = writeln!(
            src,
            "        for (var d = lid.x; d < {hd}u; d += {wg_size}u) {{"
        );
        src.push_str("            shared_k[d] = src_b[k_base + d];\n");
        src.push_str("            shared_v[d] = bias[k_base + d];\n");
        src.push_str("        }\n");
        src.push_str("        workgroupBarrier();\n\n");

        for e in 0..ept {
            let _ = writeln!(src, "        let k{e} = shared_k[d_base + {e}u];");
            let _ = writeln!(src, "        let v{e} = shared_v[d_base + {e}u];");
        }
        src.push_str("        var score_part = 0.0;\n");
        src.push_str("        var dp_part = 0.0;\n");
        for e in 0..ept {
            let _ = writeln!(src, "        score_part += q{e} * k{e};");
            let _ = writeln!(src, "        dp_part += do{e} * v{e};");
        }
        let _ = writeln!(src, "        wg_score[qi * {tpq}u + lane] = score_part;");
        let _ = writeln!(src, "        wg_dp[qi * {tpq}u + lane] = dp_part;");
        src.push_str("        reduce_score_dp(lid.x);\n");
        let _ = writeln!(src, "        let score = wg_score[qi * {tpq}u] * scale;");
        let _ = writeln!(src, "        let dp_t = wg_dp[qi * {tpq}u];\n");
        src.push_str("        if valid && t >= my_kv_start && t < my_kv_len {\n");
        src.push_str("            let p_t = exp(min(score - max_s, 0.0) - log_sum);\n");
        src.push_str("            let ds_t = p_t * (dp_t - row_sum);\n");
        src.push_str("            let w = ds_t * scale;\n");
        for e in 0..ept {
            let _ = writeln!(src, "            dq{e} += w * k{e};");
        }
        src.push_str("        }\n");
        src.push_str("        workgroupBarrier();\n");
        src.push_str("    }\n\n");
    }

    src.push_str("    if valid {\n");
    for e in 0..ept {
        let _ = writeln!(src, "        dst[q_base + d_base + {e}u] = dq{e};");
    }
    src.push_str("    }\n");
    src.push_str("}\n");

    let module = parse_source(&src).unwrap_or_else(|e| {
        panic!(
            "generated flash grad_q WGSL failed to parse:\n{}\n---\n{}",
            e, src
        )
    });
    ShaderModule {
        module,
        source: src,
        hint: "flash_grad_q",
    }
}

/// Generate a Flash Attention 2 backward dK/dV kernel using vectorized
/// register pattern (each thread computes full Q·K, dO·O, dO·V dot
/// products in registers).
///
/// Mirrors the forward kernel's EPT/TPQ/BKV pattern. When TPQ=1 (EPT==hd)
/// there are NO workgroup barriers inside the Q loop — each thread owns
/// one full (kv_pos, head_range) output row and sequentially accumulates
/// dK/dV by loading Q/dO/O scalars directly from global memory.
pub fn generate_flash_grad_kv_module(head_dim: u32, ept_cap: u32) -> ShaderModule {
    use std::fmt::Write;
    let hd = head_dim;
    // Backward uses its own cap because these kernels carry more live state.
    // The fused dK+dV kernel reports 210 regs at EPT=32 on Blackwell,
    // so the auto-tune typically chooses a smaller value here.
    let (ept, tpq) = attention_lanes(hd, ept_cap); // tpq threads per KV position
    let bkv: u32 = (256 / tpq).max(1);
    if bkv <= 1 {
        return ShaderModule::new(include_str!("../shaders/mha_grad_kv.wgsl"));
    }
    let wg_size = bkv * tpq;
    let mut src = String::new();

    // Params + bindings (match MultiHeadAttnGradKVData)
    src.push_str(ATTENTION_PARAMS_WGSL);
    src.push_str("var<storage> d_out: array<f32>;\n");
    src.push_str("var<storage> src_a: array<f32>;\n"); // Q
    src.push_str("var<storage> src_b: array<f32>;\n"); // K
    src.push_str("var<storage> bias: array<f32>;\n"); // V
    src.push_str("var<storage> lse: array<f32>;\n");
    // D[pos, head] = dot(dO, O), reduced once before this dispatch.
    src.push_str("var<storage> fwd_dst: array<f32>;\n");
    src.push_str("var<storage, read_write> dst: array<f32>;\n"); // dK
    src.push_str("var<storage, read_write> dst2: array<f32>;\n"); // dV
    src.push_str("var<uniform> params: Params;\n\n");

    // Shared Q/dO staging: only needed when TPQ > 1 (multiple threads
    // per KV position need coordinated access). When TPQ == 1, each thread
    // loads Q/dO directly from global memory — all threads read the same
    // addresses, hitting L2 cache, and no barriers are needed.
    if tpq > 1 {
        let _ = writeln!(src, "var<workgroup> shared_q: array<f32, {hd}>;");
        let _ = writeln!(src, "var<workgroup> shared_do: array<f32, {hd}>;\n");
    }

    // Shared memory only needed when TPQ > 1 (cross-lane reductions)
    if tpq > 1 {
        let _ = writeln!(src, "var<workgroup> wg_score: array<f32, {}>;", bkv * tpq);
        let _ = writeln!(src, "var<workgroup> wg_dp: array<f32, {}>;\n", bkv * tpq);
        src.push_str("fn reduce_pair(tid: u32) {\n");
        let _ = writeln!(src, "    let local = tid % {tpq}u;");
        let _ = writeln!(src, "    let base = (tid / {tpq}u) * {tpq}u;");
        let mut stride = tpq / 2;
        while stride > 0 {
            src.push_str("    workgroupBarrier();\n");
            let _ = writeln!(
                src,
                "    if local < {stride}u {{ wg_score[base + local] += wg_score[base + local + {stride}u]; wg_dp[base + local] += wg_dp[base + local + {stride}u]; }}"
            );
            stride /= 2;
        }
        src.push_str("    workgroupBarrier();\n}\n\n");
    }

    // Main kernel
    let _ = writeln!(src, "@compute @workgroup_size({wg_size})");
    src.push_str("fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n");
    let _ = writeln!(src, "    let ki = lid.x / {tpq}u;"); // KV position within tile
    let _ = writeln!(src, "    let lane = lid.x % {tpq}u;");
    let _ = writeln!(src, "    let d_base = lane * {ept}u;");
    let _ = writeln!(src, "    let t = wgid.x * {bkv}u + ki;"); // global KV position
    src.push_str("    let kv_head = wgid.y;\n");
    src.push_str("    let q_seq = params.q_seq;\n");
    src.push_str("    let kv_seq = params.kv_seq;\n");
    src.push_str("    let num_heads = params.packed_heads >> 16u;\n");
    src.push_str("    let num_kv_heads = params.packed_heads & 0xFFFFu;\n");
    src.push_str("    let head_dim = params.head_dim;\n\n");

    src.push_str("    let effective_kv_seq = select(kv_seq, q_seq, kv_seq == 0u);\n");
    src.push_str("    let valid = t < effective_kv_seq && kv_head < num_kv_heads;\n");
    src.push_str("    let heads_per_kv = num_heads / max(num_kv_heads, 1u);\n");
    src.push_str("    let kv_dim = num_kv_heads * head_dim;\n");
    src.push_str("    let q_dim = num_heads * head_dim;\n");
    src.push_str("    let kv_base = t * kv_dim + kv_head * head_dim;\n");
    src.push_str("    let scale = inverseSqrt(f32(head_dim));\n\n");

    // Load K/V slices for this thread into registers
    for e in 0..ept {
        let _ = writeln!(src, "    var k{e} = 0.0;");
        let _ = writeln!(src, "    var v{e} = 0.0;");
    }
    src.push_str("    if valid {\n");
    for e in 0..ept {
        let _ = writeln!(src, "        k{e} = src_b[kv_base + d_base + {e}u];");
        let _ = writeln!(src, "        v{e} = bias[kv_base + d_base + {e}u];");
    }
    src.push_str("    }\n\n");

    // Per-thread dK/dV accumulators (EPT elements each)
    for e in 0..ept {
        let _ = writeln!(src, "    var dk{e} = 0.0;");
        let _ = writeln!(src, "    var dv{e} = 0.0;");
    }
    src.push('\n');

    // Q loop range bounds
    src.push_str("    let start_pos = select(0u, t, kv_seq == 0u);\n");
    src.push_str("    let window = params.window_size;\n");
    src.push_str("    let end_pos = select(q_seq, min(q_seq, t + window), window > 0u);\n\n");

    // Workgroup-wide loop bounds (used for barrier consistency when TPQ>1)
    let _ = writeln!(src, "    let first_t = wgid.x * {bkv}u;");
    let _ = writeln!(
        src,
        "    let last_t = min(wgid.x * {bkv}u + {bkv}u - 1u, effective_kv_seq - 1u);"
    );
    src.push_str("    let wg_start = select(0u, first_t, kv_seq == 0u);\n");
    src.push_str("    let wg_end = select(q_seq, min(q_seq, last_t + window), window > 0u);\n\n");

    // Inner Q loop: iterate all Q positions that attend to this KV position.
    src.push_str("    for (var pos = wg_start; pos < wg_end; pos++) {\n");
    src.push_str("        for (var head_rel = 0u; head_rel < heads_per_kv; head_rel++) {\n");
    src.push_str("            let head = kv_head * heads_per_kv + head_rel;\n");
    src.push_str("            let q_base = pos * q_dim + head * head_dim;\n\n");

    if tpq == 1 {
        // TPQ=1: each thread handles the full head_dim. Load Q/dO directly
        // from global memory — all threads read the same addresses, hitting L2.
        // This eliminates ALL barriers in the inner loop.
        for e in 0..ept {
            let _ = writeln!(src, "            let q{e} = src_a[q_base + {e}u];");
            let _ = writeln!(src, "            let do{e} = d_out[q_base + {e}u];");
        }
    } else {
        // TPQ>1: cooperative staging into shared memory (needs barriers).
        let _ = writeln!(
            src,
            "            for (var d = lid.x; d < {hd}u; d += {wg_size}u) {{"
        );
        src.push_str("                shared_q[d] = src_a[q_base + d];\n");
        src.push_str("                shared_do[d] = d_out[q_base + d];\n");
        src.push_str("            }\n");
        src.push_str("            workgroupBarrier();\n\n");

        for e in 0..ept {
            let _ = writeln!(src, "            let q{e} = shared_q[d_base + {e}u];");
            let _ = writeln!(src, "            let do{e} = shared_do[d_base + {e}u];");
        }
    }
    src.push_str("            var score_part = 0.0;\n");
    src.push_str("            var dp_part = 0.0;\n");
    for e in 0..ept {
        let _ = writeln!(src, "            score_part += q{e} * k{e};");
        let _ = writeln!(src, "            dp_part += do{e} * v{e};");
    }
    src.push_str("            let row_sum = fwd_dst[pos * num_heads + head];\n");
    if tpq > 1 {
        let _ = writeln!(
            src,
            "            wg_score[ki * {tpq}u + lane] = score_part;"
        );
        let _ = writeln!(src, "            wg_dp[ki * {tpq}u + lane] = dp_part;");
        src.push_str("            reduce_pair(lid.x);\n");
        let _ = writeln!(
            src,
            "            let score = wg_score[ki * {tpq}u] * scale;"
        );
        let _ = writeln!(src, "            let dp_t = wg_dp[ki * {tpq}u];\n");
    } else {
        src.push_str("            let score = score_part * scale;\n");
        src.push_str("            let dp_t = dp_part;\n");
    }
    src.push_str("            if valid && pos >= start_pos && pos < end_pos {\n");
    src.push_str("                let lse_idx = (pos * num_heads + head) * 2u;\n");
    src.push_str(
        "                let p_t = exp(min(score - lse[lse_idx], 0.0) - lse[lse_idx + 1u]);\n",
    );
    src.push_str("                let ds_t = p_t * (dp_t - row_sum);\n");
    src.push_str("                let w_dk = ds_t * scale;\n");
    for e in 0..ept {
        let _ = writeln!(src, "                dk{e} += w_dk * q{e};");
        let _ = writeln!(src, "                dv{e} += p_t * do{e};");
    }
    src.push_str("            }\n");
    if tpq > 1 {
        src.push_str("            workgroupBarrier();\n");
    }
    src.push_str("        }\n");
    src.push_str("    }\n\n");

    src.push_str("    if valid {\n");
    for e in 0..ept {
        let _ = writeln!(src, "        dst[kv_base + d_base + {e}u] = dk{e};");
        let _ = writeln!(src, "        dst2[kv_base + d_base + {e}u] = dv{e};");
    }
    src.push_str("    }\n");
    src.push_str("}\n");

    let module = parse_source(&src).unwrap_or_else(|e| {
        panic!(
            "generated flash grad_kv WGSL failed to parse:\n{}\n---\n{}",
            e, src
        )
    });
    ShaderModule {
        module,
        source: src,
        hint: "flash_grad_kv",
    }
}

/// Conv2d cooperative-matrix GEMM direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conv2dCoopDirection {
    /// Forward: C[Co, oH*oW] = A[Co, K] * B[K, oH*oW], K = Ci*kH*kW
    Forward,
    /// Backward w.r.t. input: C[Ci, H*W] = A[Ci, K] * B[K, H*W], K = Co*kH*kW
    GradInput,
}

/// Generate a specialized conv2d cooperative-matrix GEMM kernel with
/// compile-time kernel size and stride constants.
///
/// When `kernel_h`, `kernel_w`, and `stride` are baked into the WGSL as
/// constants, the SPIR-V compiler can constant-fold the im2col index
/// decomposition (divisions become constant-divisor ops) and eliminate
/// dead stride branches entirely.
///
/// The generated shader uses the SAME bindings and Params struct as
/// the template-based variants (`conv2d_gemm_coop.wgsl` and
/// `conv2d_grad_input_gemm_coop.wgsl`) so the runtime bind/dispatch
/// code needs no changes.
pub fn generate_conv2d_coop_module(
    kernel_h: u32,
    kernel_w: u32,
    stride: u32,
    direction: Conv2dCoopDirection,
    config: &CoopConfig,
) -> ShaderModule {
    use std::fmt::Write;

    let tile = config.tile_size;
    let output_tile = config.output_tile();
    let shared_size = tile * tile;
    let wg_size: u32 = 64;
    let staging_iters = shared_size / wg_size;
    let row_stride = wg_size / tile;
    let tile_mask = tile - 1;
    let tile_shift = tile.trailing_zeros();
    let use_vec4 = tile >= 16;

    let (elem_type, enable_f16, elem_zero, cast_open, cast_close) = if config.use_f16_input {
        ("f16", "enable f16;\n", "f16(0.0)", "f16(", ")")
    } else {
        ("f32", "", "0.0", "", "")
    };
    let ab_type = if config.use_f16_input { "f16" } else { "f32" };
    let coop_ab = format!("coop_mat{tile}x{tile}<{ab_type},A>");
    let coop_ba = format!("coop_mat{tile}x{tile}<{ab_type},B>");
    let coop_c = format!("coop_mat{tile}x{tile}<f32,C>");

    let kernel_hw = kernel_h * kernel_w;
    let backward = direction == Conv2dCoopDirection::GradInput;

    let mut src = String::with_capacity(8192);

    // Header
    let dir_str = if backward {
        "backward (grad_input)"
    } else {
        "forward"
    };
    let _ = writeln!(
        src,
        "// Conv2d {dir_str} via implicit GEMM — cooperative matrix variant."
    );
    let _ = writeln!(
        src,
        "// Specialized for {kernel_h}x{kernel_w} stride-{stride} convolutions."
    );
    src.push('\n');
    src.push_str(enable_f16);
    src.push_str("enable wgpu_cooperative_matrix;\n\n");
    src.push_str(crate::divisor::SHADER);

    // Params struct — identical layout to Conv2dParams for binding compatibility.
    // kernel_h/kernel_w/stride fields are still present but ignored in favor of constants.
    src.push_str(
        "struct Params {\n\
         \x20   batch: u32,\n\
         \x20   in_channels: u32,\n\
         \x20   in_h: u32,\n\
         \x20   in_w: u32,\n\
         \x20   out_channels: u32,\n\
         \x20   kernel_h: u32,\n\
         \x20   kernel_w: u32,\n\
         \x20   stride: u32,\n\
         \x20   padding_h: u32,\n\
         \x20   out_h: u32,\n\
         \x20   out_w: u32,\n\
         \x20   padding_w: u32,\n\
         \x20   kernel_w_multiplier: u32,\n\
         \x20   kernel_hw_multiplier: u32,\n\
         \x20   column_width_multiplier: u32,\n\
         \x20   output_spatial_multiplier: u32,\n\
         }\n\n",
    );

    // Storage bindings — must match Conv2dData / Conv2dGradInputData layout
    if backward {
        src.push_str("var<storage> grad_out: array<f32>;\n");
        src.push_str("var<storage> weight: array<f32>;\n");
    } else {
        src.push_str("var<storage> src: array<f32>;\n");
        src.push_str("var<storage> weight: array<vec4<f32>>;\n");
    }
    src.push_str("var<storage, read_write> dst: array<f32>;\n");
    src.push_str("var<uniform> params: Params;\n");
    let _ = writeln!(
        src,
        "var<workgroup> shared_a0: array<{elem_type}, {shared_size}>;"
    );
    let _ = writeln!(
        src,
        "var<workgroup> shared_a1: array<{elem_type}, {shared_size}>;"
    );
    let _ = writeln!(
        src,
        "var<workgroup> shared_b0: array<{elem_type}, {shared_size}>;"
    );
    let _ = writeln!(
        src,
        "var<workgroup> shared_b1: array<{elem_type}, {shared_size}>;"
    );
    src.push('\n');

    // Compile-time constants for kernel geometry
    let _ = writeln!(src, "const KERNEL_H: u32 = {kernel_h}u;");
    let _ = writeln!(src, "const KERNEL_W: u32 = {kernel_w}u;");
    let _ = writeln!(src, "const KERNEL_HW: u32 = {kernel_hw}u;");
    let _ = writeln!(src, "const STRIDE: u32 = {stride}u;");
    src.push('\n');

    // Main function
    let _ = writeln!(src, "@compute @workgroup_size(64)");
    src.push_str(
        "fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(subgroup_id) sg: u32) {\n",
    );

    if backward {
        // Backward: M = Ci, N = H*W (input spatial), K = Co*kH*kW
        let _ = writeln!(src, "    let tile_row = wgid.x * {output_tile}u;");
        let _ = writeln!(src, "    let tile_col = wgid.y * {output_tile}u;");
        src.push_str("    let n = wgid.z;\n\n");
        src.push_str("    let m_total = params.in_channels;\n");
        src.push_str("    let n_total = params.in_h * params.in_w;\n");
        let _ = writeln!(src, "    let k_total = params.out_channels * KERNEL_HW;");
        src.push_str("    let go_spatial = params.out_h * params.out_w;\n\n");

        // Invert forward cross-correlation without flipping the weight indices.
        src.push_str("    let pad_h = i32(params.padding_h);\n");
        src.push_str("    let pad_w = i32(params.padding_w);\n");
    } else {
        // Forward: M = Co, N = oH*oW, K = Ci*kH*kW
        let _ = writeln!(src, "    let tile_row = wgid.x * {output_tile}u;");
        let _ = writeln!(src, "    let tile_col = wgid.y * {output_tile}u;");
        src.push_str("    let n = wgid.z;\n\n");
        src.push_str("    let m_total = params.out_channels;\n");
        src.push_str("    let n_total = params.out_h * params.out_w;\n");
        let _ = writeln!(src, "    let k_total = params.in_channels * KERNEL_HW;");
        src.push_str("    let input_stride = params.in_channels * params.in_h * params.in_w;\n\n");
    }

    // C offsets for the 4 output tiles
    src.push_str("    let c00 = n * m_total * n_total + tile_row * n_total + tile_col;\n");
    let _ = writeln!(
        src,
        "    let c01 = n * m_total * n_total + tile_row * n_total + (tile_col + {tile}u);"
    );
    let _ = writeln!(
        src,
        "    let c10 = n * m_total * n_total + (tile_row + {tile}u) * n_total + tile_col;"
    );
    let _ = writeln!(
        src,
        "    let c11 = n * m_total * n_total + (tile_row + {tile}u) * n_total + (tile_col + {tile}u);"
    );
    src.push('\n');
    let _ = writeln!(src, "    let n1_valid = (tile_col + {tile}u) < n_total;");
    let _ = writeln!(src, "    let m1_valid = (tile_row + {tile}u) < m_total;");
    src.push('\n');

    // Accumulator init
    let _ = writeln!(src, "    var acc00 = {coop_c}();");
    let _ = writeln!(src, "    var acc01 = {coop_c}();");
    let _ = writeln!(src, "    var acc10 = {coop_c}();");
    let _ = writeln!(src, "    var acc11 = {coop_c}();");
    src.push('\n');

    // Hoisted staging index components
    if backward {
        let _ = writeln!(src, "    let src_col = lid.x & {tile_mask}u;");
        let _ = writeln!(src, "    let base_row = lid.x >> {tile_shift}u;");
    } else {
        if use_vec4 {
            src.push_str("    let v4_row = lid.x >> 2u;\n");
            src.push_str("    let v4_col = (lid.x & 3u) << 2u;\n");
        }
        let _ = writeln!(src, "    let src_col = lid.x & {tile_mask}u;");
        let _ = writeln!(src, "    let base_row = lid.x >> {tile_shift}u;");
    }
    src.push('\n');

    // Main loop
    src.push_str("    var t = 0u;\n");
    src.push_str("    loop {\n");
    src.push_str("        if t >= k_total { break; }\n\n");
    let _ = writeln!(src, "        let zero_val = {elem_zero};");
    src.push('\n');

    if backward {
        // === BACKWARD STAGING ===
        // Stage sa0: B-tile im2col(grad_out)^T
        emit_grad_input_im2col_stage(
            &mut src,
            "shared_a0",
            "tile_col",
            "cc0",
            "in_n0",
            "ih0",
            "iw0",
            tile,
            tile_mask,
            staging_iters,
            row_stride,
            kernel_hw,
            stride,
            cast_open,
            cast_close,
            elem_zero,
        );

        // Stage sa1: second column block
        let _ = writeln!(src, "        let cc1 = tile_col + {tile}u + src_col;");
        emit_grad_input_im2col_stage(
            &mut src,
            "shared_a1",
            &format!("tile_col + {tile}u"),
            "cc1",
            "in_n1",
            "ih1",
            "iw1",
            tile,
            tile_mask,
            staging_iters,
            row_stride,
            kernel_hw,
            stride,
            cast_open,
            cast_close,
            elem_zero,
        );

        // Stage sb0: A-tile weight_T
        emit_grad_input_weight_stage(
            &mut src,
            "shared_b0",
            "tile_row",
            false,
            tile,
            staging_iters,
            row_stride,
            kernel_hw,
            cast_open,
            cast_close,
            elem_zero,
        );

        // Stage sb1: A-tile weight_T second row block
        emit_grad_input_weight_stage(
            &mut src,
            "shared_b1",
            "tile_row",
            true,
            tile,
            staging_iters,
            row_stride,
            kernel_hw,
            cast_open,
            cast_close,
            elem_zero,
        );
    } else {
        // === FORWARD STAGING ===
        // Stage sb0: A-tile weight[Co, K] via vec4
        emit_forward_weight_stage(
            &mut src,
            "shared_b0",
            "tile_row",
            false,
            tile,
            use_vec4,
            staging_iters,
            row_stride,
            cast_open,
            cast_close,
            elem_zero,
        );

        // Stage sb1: A-tile weight second row block
        emit_forward_weight_stage(
            &mut src,
            "shared_b1",
            "tile_row",
            true,
            tile,
            use_vec4,
            staging_iters,
            row_stride,
            cast_open,
            cast_close,
            elem_zero,
        );

        // Stage sa0: B-tile im2col(input)
        emit_forward_im2col_stage(
            &mut src,
            "shared_a0",
            "tile_col",
            "cc0",
            "in_n0",
            tile,
            tile_mask,
            staging_iters,
            row_stride,
            kernel_hw,
            stride,
            cast_open,
            cast_close,
            elem_zero,
        );

        // Stage sa1: B-tile second column block
        let _ = writeln!(src, "        let cc1 = tile_col + {tile}u + src_col;");
        emit_forward_im2col_stage(
            &mut src,
            "shared_a1",
            &format!("tile_col + {tile}u"),
            "cc1",
            "in_n1",
            tile,
            tile_mask,
            staging_iters,
            row_stride,
            kernel_hw,
            stride,
            cast_open,
            cast_close,
            elem_zero,
        );
    }

    // Barrier + cooperative matmul
    src.push_str("\n        workgroupBarrier();\n\n");
    let _ = writeln!(
        src,
        "        let a0 = coopLoadT<{coop_ab}>(&shared_b0[0], {tile}u);"
    );
    let _ = writeln!(
        src,
        "        let a1 = coopLoadT<{coop_ab}>(&shared_b1[0], {tile}u);"
    );
    let _ = writeln!(
        src,
        "        let b0 = coopLoadT<{coop_ba}>(&shared_a0[0], {tile}u);"
    );
    let _ = writeln!(
        src,
        "        let b1 = coopLoadT<{coop_ba}>(&shared_a1[0], {tile}u);"
    );
    src.push_str("        acc00 = coopMultiplyAdd(a0, b0, acc00);\n");
    src.push_str("        acc01 = coopMultiplyAdd(a0, b1, acc01);\n");
    src.push_str("        acc10 = coopMultiplyAdd(a1, b0, acc10);\n");
    src.push_str("        acc11 = coopMultiplyAdd(a1, b1, acc11);\n\n");
    src.push_str("        workgroupBarrier();\n");
    let _ = writeln!(src, "        t += {tile}u;");
    src.push_str("    }\n\n");

    // Store results
    let store_comment = if backward {
        "grad_input [N, Ci, H, W]"
    } else {
        "output [N, Co, oH, oW]"
    };
    let _ = writeln!(
        src,
        "    // Store results to {store_comment} in NCHW layout"
    );
    if config.use_f16_input || !backward {
        // Forward and f16 cooperative kernels retain the direct-store
        // alignment requirement enforced by runtime selection.
        src.push_str("    if sg == 0u {\n");
        src.push_str("    coopStoreT(acc00, &dst[c00], n_total);\n");
        src.push_str("    if n1_valid {\n");
        src.push_str("        coopStoreT(acc01, &dst[c01], n_total);\n");
        src.push_str("    }\n");
        src.push_str("    if m1_valid {\n");
        src.push_str("        coopStoreT(acc10, &dst[c10], n_total);\n");
        src.push_str("    }\n");
        src.push_str("    if n1_valid && m1_valid {\n");
        src.push_str("        coopStoreT(acc11, &dst[c11], n_total);\n");
        src.push_str("    }\n");
        src.push_str("    }\n");
    } else {
        // A direct cooperative store writes whole tiles. At a partial right
        // edge it crosses the logical NCHW row boundary, and at a partial
        // bottom edge its extra rows are the next image's first channels
        // (or past the buffer). Only fully covered workgroups keep the fast
        // path; edge workgroups stage through the now-dead f32 input tiles
        // and perform bounds-checked scalar stores.
        let _ = writeln!(
            src,
            "    if (tile_col + {output_tile}u) <= n_total && (tile_row + {output_tile}u) <= m_total {{"
        );
        src.push_str("        if sg == 0u {\n");
        src.push_str("        coopStoreT(acc00, &dst[c00], n_total);\n");
        src.push_str("        coopStoreT(acc01, &dst[c01], n_total);\n");
        src.push_str("        coopStoreT(acc10, &dst[c10], n_total);\n");
        src.push_str("        coopStoreT(acc11, &dst[c11], n_total);\n");
        src.push_str("        }\n");
        src.push_str("    } else {\n");
        src.push_str("        if sg == 0u {\n");
        let _ = writeln!(src, "        coopStoreT(acc00, &shared_b0[0], {tile}u);");
        let _ = writeln!(src, "        coopStoreT(acc01, &shared_b1[0], {tile}u);");
        let _ = writeln!(src, "        coopStoreT(acc10, &shared_a0[0], {tile}u);");
        let _ = writeln!(src, "        coopStoreT(acc11, &shared_a1[0], {tile}u);");
        src.push_str("        }\n");
        src.push_str("        workgroupBarrier();\n");
        let _ = writeln!(
            src,
            "        for (var flat = lid.x; flat < {shared_size}u; flat += 64u) {{"
        );
        let _ = writeln!(src, "            let local_row = flat >> {tile_shift}u;");
        let _ = writeln!(src, "            let local_col = flat & {tile_mask}u;");
        src.push_str("            let row0 = tile_row + local_row;\n");
        src.push_str("            let col0 = tile_col + local_col;\n");
        src.push_str("            if row0 < m_total && col0 < n_total {\n");
        src.push_str(
            "                dst[n * m_total * n_total + row0 * n_total + col0] = shared_b0[flat];\n",
        );
        src.push_str("            }\n");
        let _ = writeln!(src, "            let col1 = col0 + {tile}u;");
        src.push_str("            if row0 < m_total && col1 < n_total {\n");
        src.push_str(
            "                dst[n * m_total * n_total + row0 * n_total + col1] = shared_b1[flat];\n",
        );
        src.push_str("            }\n");
        let _ = writeln!(src, "            let row1 = row0 + {tile}u;");
        src.push_str("            if row1 < m_total && col0 < n_total {\n");
        src.push_str(
            "                dst[n * m_total * n_total + row1 * n_total + col0] = shared_a0[flat];\n",
        );
        src.push_str("            }\n");
        src.push_str("            if row1 < m_total && col1 < n_total {\n");
        src.push_str(
            "                dst[n * m_total * n_total + row1 * n_total + col1] = shared_a1[flat];\n",
        );
        src.push_str("            }\n");
        src.push_str("        }\n");
        src.push_str("    }\n");
    }
    src.push_str("}\n");

    ShaderModule::new(&src)
}

/// Emit the im2col staging loop for grad_input (backward) direction.
///
/// Loads from `grad_out` with compile-time kernel decomposition.
pub(super) fn emit_grad_input_im2col_stage(
    src: &mut String,
    shared_name: &str,
    tile_col_expr: &str,
    cc_var: &str,
    in_n_var: &str,
    ih_var: &str,
    iw_var: &str,
    _tile: u32,
    _tile_mask: u32,
    staging_iters: u32,
    row_stride: u32,
    _kernel_hw: u32,
    stride: u32,
    cast_open: &str,
    cast_close: &str,
    _elem_zero: &str,
) {
    use std::fmt::Write;

    // First stage (sa0) uses tile_col + src_col directly; subsequent stages
    // have cc_var pre-computed above the call.
    let is_first = cc_var == "cc0";
    if is_first {
        let _ = writeln!(src, "        let {cc_var} = {tile_col_expr} + src_col;");
    }
    let _ = writeln!(src, "        let {in_n_var} = {cc_var} < n_total;");

    // Pre-decompose spatial position (invariant across e iterations)
    let _ = writeln!(
        src,
        "        let {ih_var} = divide_exact({cc_var}, params.in_w, params.column_width_multiplier);"
    );
    let _ = writeln!(
        src,
        "        let {iw_var} = {cc_var} - {ih_var} * params.in_w;"
    );

    let _ = writeln!(
        src,
        "        for (var e = 0u; e < {staging_iters}u; e++) {{"
    );
    src.push_str("            let flat = lid.x + e * 64u;\n");
    let _ = writeln!(
        src,
        "            let tr = t + base_row + e * {row_stride}u;"
    );
    src.push_str("            var val = zero_val;\n");
    let _ = writeln!(src, "            if tr < k_total && {in_n_var} {{");

    // Decompose tr into (co, kh, kw) using compile-time constants
    let _ = writeln!(src, "                let co = tr / KERNEL_HW;");
    src.push_str("                let k_rem = tr - co * KERNEL_HW;\n");
    let _ = writeln!(src, "                let kh = k_rem / KERNEL_W;");
    src.push_str("                let kw = k_rem - kh * KERNEL_W;\n");

    if stride == 1 {
        // Stride-1 only path: oh = ih + pad_h - kh
        let _ = writeln!(
            src,
            "                let oh = i32({ih_var}) + pad_h - i32(kh);"
        );
        let _ = writeln!(
            src,
            "                let ow = i32({iw_var}) + pad_w - i32(kw);"
        );
        src.push_str(
            "                if oh >= 0 && u32(oh) < params.out_h && ow >= 0 && u32(ow) < params.out_w {\n",
        );
        let _ = writeln!(
            src,
            "                    val = {cast_open}grad_out[n * params.out_channels * go_spatial + co * go_spatial + u32(oh) * params.out_w + u32(ow)]{cast_close};"
        );
        src.push_str("                }\n");
    } else {
        // General stride path
        let _ = writeln!(
            src,
            "                let h_off = i32({ih_var}) + i32(params.padding_h) - i32(kh);"
        );
        let _ = writeln!(
            src,
            "                let w_off = i32({iw_var}) + i32(params.padding_w) - i32(kw);"
        );
        let _ = writeln!(src, "                let i_stride = i32(STRIDE);");
        src.push_str(
            "                if h_off >= 0 && w_off >= 0 && (h_off % i_stride) == 0 && (w_off % i_stride) == 0 {\n",
        );
        let _ = writeln!(src, "                    let oh = u32(h_off) / STRIDE;");
        let _ = writeln!(src, "                    let ow = u32(w_off) / STRIDE;");
        src.push_str("                    if oh < params.out_h && ow < params.out_w {\n");
        let _ = writeln!(
            src,
            "                        val = {cast_open}grad_out[n * params.out_channels * go_spatial + co * go_spatial + oh * params.out_w + ow]{cast_close};"
        );
        src.push_str("                    }\n");
        src.push_str("                }\n");
    }

    src.push_str("            }\n");
    let _ = writeln!(src, "            {shared_name}[flat] = val;");
    src.push_str("        }\n\n");
}

/// Emit the weight staging for grad_input (backward) direction.
///
/// Weight is stored as [Co, Ci, kH, kW]; we load weight_T[Ci, Co*kH*kW].
pub(super) fn emit_grad_input_weight_stage(
    src: &mut String,
    shared_name: &str,
    tile_row_expr: &str,
    is_second_block: bool,
    tile: u32,
    staging_iters: u32,
    row_stride: u32,
    _kernel_hw: u32,
    cast_open: &str,
    cast_close: &str,
    _elem_zero: &str,
) {
    use std::fmt::Write;

    // Only emit tc decomposition once (for the first weight stage)
    if !is_second_block {
        src.push_str("        let tc = t + src_col;\n");
        src.push_str("        let in_k = tc < k_total;\n");
        src.push_str("        let tc_co = tc / KERNEL_HW;\n");
        src.push_str("        let tc_k_rem = tc - tc_co * KERNEL_HW;\n");
        src.push_str("        let tc_kh = tc_k_rem / KERNEL_W;\n");
        src.push_str("        let tc_kw = tc_k_rem - tc_kh * KERNEL_W;\n");
        let _ = writeln!(
            src,
            "        let tc_weight_offset = tc_kh * KERNEL_W + tc_kw;"
        );
    }

    let row_offset = if is_second_block {
        format!("{tile_row_expr} + {tile}u + ")
    } else {
        format!("{tile_row_expr} + ")
    };

    let _ = writeln!(
        src,
        "        for (var e = 0u; e < {staging_iters}u; e++) {{"
    );
    src.push_str("            let flat = lid.x + e * 64u;\n");
    let _ = writeln!(
        src,
        "            let gr = {row_offset}base_row + e * {row_stride}u;"
    );
    src.push_str("            var val = zero_val;\n");
    src.push_str("            if gr < m_total && in_k {\n");
    let _ = writeln!(
        src,
        "                val = {cast_open}weight[(tc_co * m_total + gr) * KERNEL_HW + tc_weight_offset]{cast_close};"
    );
    src.push_str("            }\n");
    let _ = writeln!(src, "            {shared_name}[flat] = val;");
    src.push_str("        }\n\n");
}

/// Emit weight staging for the forward direction.
///
/// Weight is stored as dense [Co, K] row-major. Tiles of at least 16 use
/// vec4 loads; an 8x8 tile has only 64 shared elements, so each of the 64
/// threads stages one scalar instead. Mapping an 8x8 tile with the vec4
/// layout would address it as 16 rows by 16 columns and write out of bounds.
pub(super) fn emit_forward_weight_stage(
    src: &mut String,
    shared_name: &str,
    tile_row_expr: &str,
    is_second_block: bool,
    tile: u32,
    use_vec4: bool,
    staging_iters: u32,
    row_stride: u32,
    cast_open: &str,
    cast_close: &str,
    _elem_zero: &str,
) {
    use std::fmt::Write;

    let row_offset = if is_second_block {
        format!("{tile_row_expr} + {tile}u + ")
    } else {
        format!("{tile_row_expr} + ")
    };

    if !use_vec4 {
        src.push_str("        {\n");
        src.push_str("            let tc = t + src_col;\n");
        src.push_str("            let in_k = tc < k_total;\n");
        let _ = writeln!(
            src,
            "            for (var e = 0u; e < {staging_iters}u; e++) {{"
        );
        src.push_str("                let flat = lid.x + e * 64u;\n");
        let _ = writeln!(
            src,
            "                let gr = {row_offset}base_row + e * {row_stride}u;"
        );
        src.push_str("                var val = zero_val;\n");
        src.push_str("                if gr < m_total && in_k {\n");
        src.push_str("                    let idx = gr * k_total + tc;\n");
        let _ = writeln!(
            src,
            "                    val = {cast_open}weight[idx >> 2u][idx & 3u]{cast_close};"
        );
        src.push_str("                }\n");
        let _ = writeln!(src, "                {shared_name}[flat] = val;");
        src.push_str("            }\n");
        src.push_str("        }\n\n");
        return;
    }

    src.push_str("        {\n");
    let row_offset = if is_second_block {
        format!("({tile_row_expr} + {tile}u) + v4_row")
    } else {
        format!("{tile_row_expr} + v4_row")
    };
    let _ = writeln!(src, "            let gr = {row_offset};");
    src.push_str("            let tc4 = t + v4_col;\n");
    let _ = writeln!(src, "            let flat = v4_row * {tile}u + v4_col;");
    // Fast path: vec4 load, valid only when each weight row [Co, K] starts
    // on a vec4 boundary, i.e. k_total % 4 == 0. The branch is on a
    // uniform value so there is no warp divergence.
    src.push_str(
        "            if (k_total & 3u) == 0u && gr < m_total && (tc4 + 4u) <= k_total {\n",
    );
    src.push_str("                let v = weight[(gr * k_total + tc4) >> 2u];\n");
    let _ = writeln!(
        src,
        "                {shared_name}[flat] = {cast_open}v.x{cast_close};"
    );
    let _ = writeln!(
        src,
        "                {shared_name}[flat + 1u] = {cast_open}v.y{cast_close};"
    );
    let _ = writeln!(
        src,
        "                {shared_name}[flat + 2u] = {cast_open}v.z{cast_close};"
    );
    let _ = writeln!(
        src,
        "                {shared_name}[flat + 3u] = {cast_open}v.w{cast_close};"
    );
    src.push_str("            } else {\n");
    // Fallback for K not a multiple of 4: rows are not vec4-aligned, so
    // `weight[(gr*k_total+tc4)>>2]` would read across the row boundary.
    // Index each element individually (alignment-independent). Conv2d
    // with Ci*kH*kW not divisible by 4 (e.g. Ci=14 → K=126) hits this.
    src.push_str("                for (var i = 0u; i < 4u; i = i + 1u) {\n");
    src.push_str("                    let kc = tc4 + i;\n");
    src.push_str("                    if gr < m_total && kc < k_total {\n");
    src.push_str("                        let idx = gr * k_total + kc;\n");
    let _ = writeln!(
        src,
        "                        {shared_name}[flat + i] = {cast_open}weight[idx >> 2u][idx & 3u]{cast_close};"
    );
    src.push_str("                    } else {\n");
    let _ = writeln!(
        src,
        "                        {shared_name}[flat + i] = zero_val;"
    );
    src.push_str("                    }\n");
    src.push_str("                }\n");
    src.push_str("            }\n");
    src.push_str("        }\n\n");
}

/// Emit the im2col staging loop for forward direction.
///
/// Loads from `src` (input) with compile-time kernel decomposition.
pub(super) fn emit_forward_im2col_stage(
    src: &mut String,
    shared_name: &str,
    tile_col_expr: &str,
    cc_var: &str,
    in_n_var: &str,
    _tile: u32,
    _tile_mask: u32,
    staging_iters: u32,
    row_stride: u32,
    _kernel_hw: u32,
    _stride: u32,
    cast_open: &str,
    cast_close: &str,
    _elem_zero: &str,
) {
    use std::fmt::Write;

    let is_first = cc_var == "cc0";
    if is_first {
        let _ = writeln!(src, "        let {cc_var} = {tile_col_expr} + src_col;");
    }
    let _ = writeln!(src, "        let {in_n_var} = {cc_var} < n_total;");

    let _ = writeln!(
        src,
        "        for (var e = 0u; e < {staging_iters}u; e++) {{"
    );
    src.push_str("            let flat = lid.x + e * 64u;\n");
    let _ = writeln!(
        src,
        "            let tr = t + base_row + e * {row_stride}u;"
    );
    src.push_str("            var val = zero_val;\n");
    let _ = writeln!(src, "            if tr < k_total && {in_n_var} {{");

    // Decompose k_idx into (ci, kh, kw) using compile-time constants
    let _ = writeln!(src, "                let ci = tr / KERNEL_HW;");
    src.push_str("                let k_rem = tr - ci * KERNEL_HW;\n");
    let _ = writeln!(src, "                let kh = k_rem / KERNEL_W;");
    src.push_str("                let kw = k_rem - kh * KERNEL_W;\n");

    // Decompose hw_idx -> (oh, ow) -> (ih, iw)
    let _ = writeln!(
        src,
        "                let oh = divide_exact({cc_var}, params.out_w, params.column_width_multiplier);"
    );
    let _ = writeln!(
        src,
        "                let ow = {cc_var} - oh * params.out_w;"
    );

    // Use compile-time stride constant
    let _ = writeln!(
        src,
        "                let ih = i32(oh * STRIDE + kh) - i32(params.padding_h);"
    );
    let _ = writeln!(
        src,
        "                let iw = i32(ow * STRIDE + kw) - i32(params.padding_w);"
    );
    src.push_str(
        "                if ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w {\n",
    );
    let _ = writeln!(
        src,
        "                    val = {cast_open}src[n * input_stride + ci * params.in_h * params.in_w + u32(ih) * params.in_w + u32(iw)]{cast_close};"
    );
    src.push_str("                }\n");

    src.push_str("            }\n");
    let _ = writeln!(src, "            {shared_name}[flat] = val;");
    src.push_str("        }\n\n");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
