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

/// Cooperative-matrix flash attention forward (Phase 1).
///
/// Replaces only the QK^T product with a `coop_mat` MMA — softmax and
/// PV stay scalar — to keep the change tractable and validate the
/// coop path before tackling PV. Critical design point: the O
/// accumulator is held in *registers* across the entire KV loop (not
/// re-staged through shared memory each iteration), and the per-row
/// rescale runs INSIDE the same thread that owns the accumulator.
/// Each row's softmax math is duplicated 4× (once per d-chunk) but
/// that's a small constant (16 ops/row) vs the cost of the shared-mem
/// roundtrip.
///
/// Workgroup layout (64 threads = 16 rows × 4 d-chunks):
///   * `BQ = 16`, `BKV = 16` — match the coop_mat tile size.
///   * Each thread owns one (row, d_chunk) pair: holds
///     `O_acc[chunk_hd]` and `local_max` / `local_sum` in registers.
///   * Q is staged once per workgroup into shared as f16 [BQ × hd].
///   * K is staged per KV tile, **transposed** into shared as f16
///     [hd × BKV] so the MMA reads it as the B operand for Q @ K^T.
///   * V is staged per KV tile as f16 [BKV × hd] for the scalar PV.
///   * The score tile (after MMA) goes to shared_score[BQ × BKV];
///     each thread reads its row's BKV scores and runs softmax
///     locally.
///
/// Caller dispatch must use workgroups = `[ceil(q_seq/16), num_heads, 1]`.
/// `head_dim` must be a multiple of 16.
pub fn generate_flash_attention_coop_module(head_dim: u32) -> ShaderModule {
    use std::fmt::Write;
    assert!(
        head_dim >= 16 && head_dim.is_multiple_of(16),
        "coop flash attention requires head_dim multiple of 16, got {head_dim}"
    );
    let hd = head_dim;
    let hd_tiles = hd / 16;
    let bq: u32 = 16;
    let bkv: u32 = 16;
    let wg_size: u32 = 64;
    assert!(
        wg_size == bq * 4,
        "coop flash assumes wg_size=64 / BQ=16 / 4 hd-chunks per row"
    );
    let chunks_per_row: u32 = 4;
    let chunk_hd: u32 = hd / chunks_per_row;

    let mut src = String::new();
    src.push_str("enable f16;\n");
    src.push_str("enable wgpu_cooperative_matrix;\n\n");
    src.push_str(ATTENTION_PARAMS_WGSL);
    src.push_str("var<storage> src_a: array<f32>;\n"); // Q
    src.push_str("var<storage> src_b: array<f32>;\n"); // K
    src.push_str("var<storage> bias: array<f32>;\n"); // V
    src.push_str("var<storage, read_write> dst: array<f32>;\n"); // O
    src.push_str("var<storage, read_write> lse: array<f32>;\n");
    src.push_str("var<uniform> params: Params;\n\n");

    let _ = writeln!(src, "var<workgroup> shared_q: array<f16, {}>;", bq * hd);
    let _ = writeln!(src, "var<workgroup> shared_k_t: array<f16, {}>;", hd * bkv);
    // Only Q and K feed cooperative instructions; scalar PV needs f32 V.
    let _ = writeln!(src, "var<workgroup> shared_v: array<f32, {}>;", bkv * hd);
    let _ = writeln!(
        src,
        "var<workgroup> shared_score: array<f32, {}>;",
        bq * bkv
    );
    src.push('\n');

    let _ = writeln!(src, "@compute @workgroup_size({wg_size})");
    src.push_str("fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(subgroup_id) sg: u32) {\n");
    let _ = writeln!(src, "    let pos_base = wgid.x * {bq}u;");
    src.push_str("    let head = wgid.y;\n");
    src.push_str("    let q_seq = params.q_seq;\n");
    src.push_str("    let kv_seq = params.kv_seq;\n");
    src.push_str("    let num_heads = params.packed_heads >> 16u;\n");
    src.push_str("    let num_kv_heads = params.packed_heads & 0xFFFFu;\n");
    src.push_str("    let head_dim = params.head_dim;\n");
    src.push_str("    let window_size = params.window_size;\n");
    src.push_str("    let kv_head = head / (num_heads / max(num_kv_heads, 1u));\n");
    src.push_str("    let kv_head_off = kv_head * head_dim;\n");
    src.push_str("    let kv_dim = num_kv_heads * head_dim;\n");
    src.push_str("    let scale = inverseSqrt(f32(head_dim));\n\n");

    // Per-thread (row, chunk) — index hoisted to top.
    let _ = writeln!(src, "    let row = lid.x / {chunks_per_row}u;");
    let _ = writeln!(src, "    let chunk = lid.x % {chunks_per_row}u;");
    let _ = writeln!(src, "    let d_off = chunk * {chunk_hd}u;");
    src.push_str("    let qpos = pos_base + row;\n");
    src.push_str("    let q_valid = qpos < q_seq && head < num_heads;\n\n");

    // Per-row valid KV range (causal + sliding window).
    src.push_str(&kv_range("qpos", 4));

    // Workgroup-wide bounds (drives all threads through the same
    // outer KV loop).
    src.push_str("    let last_pos = min(pos_base + 15u, q_seq - 1u);\n");
    src.push_str("    let max_kv_len = select(kv_seq, last_pos + 1u, kv_seq == 0u);\n");
    src.push_str("    let first_kv_len = select(kv_seq, pos_base + 1u, kv_seq == 0u);\n");
    src.push_str(
        "    let min_kv_start = select(0u, first_kv_len - min(first_kv_len, window_size), window_size > 0u);\n\n",
    );

    // Per-thread O accumulator and softmax state — REGISTERS.
    let _ = writeln!(src, "    var local_o: array<f32, {chunk_hd}>;");
    let _ = writeln!(src, "    for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{");
    src.push_str("        local_o[e] = 0.0;\n");
    src.push_str("    }\n");
    src.push_str("    var local_max: f32 = -1e30;\n");
    src.push_str("    var local_sum: f32 = 0.0;\n\n");

    // ---- Stage Q once per workgroup ----
    let q_total = bq * hd;
    let _ = writeln!(
        src,
        "    for (var i = lid.x; i < {q_total}u; i = i + {wg_size}u) {{"
    );
    let _ = writeln!(src, "        let r = i / {hd}u;");
    let _ = writeln!(src, "        let col = i % {hd}u;");
    src.push_str("        let qp = pos_base + r;\n");
    src.push_str(
        "        if qp < q_seq { shared_q[i] = f16(src_a[qp * (num_heads * head_dim) + head * head_dim + col]); } else { shared_q[i] = f16(0.0); }\n",
    );
    src.push_str("    }\n");
    src.push_str("    workgroupBarrier();\n\n");

    // ---- KV tile loop ----
    let _ = writeln!(
        src,
        "    let tile_end = min_kv_start + ((max_kv_len - min_kv_start) / {bkv}u) * {bkv}u;"
    );
    src.push_str("    var t = min_kv_start;\n");
    let _ = writeln!(src, "    for (; t < tile_end; t = t + {bkv}u) {{");

    // Stage K transposed [hd × bkv] and V natural [bkv × hd] into shared.
    let kv_total = bkv * hd;
    let _ = writeln!(
        src,
        "        for (var i = lid.x; i < {kv_total}u; i = i + {wg_size}u) {{"
    );
    let _ = writeln!(src, "            let ki = i / {hd}u;");
    let _ = writeln!(src, "            let d  = i % {hd}u;");
    src.push_str("            let kv_pos = t + ki;\n");
    src.push_str(
        "            shared_k_t[d * 16u + ki] = f16(src_b[kv_pos * kv_dim + kv_head_off + d]);\n",
    );
    src.push_str(
        "            shared_v[ki * head_dim + d] = bias[kv_pos * kv_dim + kv_head_off + d];\n",
    );
    src.push_str("        }\n");
    src.push_str("        workgroupBarrier();\n\n");

    // Cooperative QK^T → shared_score.
    src.push_str("        var score_acc = coop_mat16x16<f32,C>();\n");
    let _ = writeln!(
        src,
        "        for (var ht = 0u; ht < {hd_tiles}u; ht = ht + 1u) {{"
    );
    let _ = writeln!(
        src,
        "            let a = coopLoadT<coop_mat16x16<f16,A>>(&shared_q[ht * 16u], {hd}u);"
    );
    let _ = writeln!(
        src,
        "            let b = coopLoadT<coop_mat16x16<f16,B>>(&shared_k_t[ht * 16u * {bkv}u], {bkv}u);"
    );
    src.push_str("            score_acc = coopMultiplyAdd(a, b, score_acc);\n");
    src.push_str("        }\n");
    let _ = writeln!(
        src,
        "        if sg == 0u {{ coopStoreT(score_acc, &shared_score[0], {bkv}u); }}"
    );
    src.push_str("        workgroupBarrier();\n\n");

    // Per-thread row softmax + PV. Each thread owns one (row, chunk).
    // The 4 chunks of a row redundantly compute the same row max/sum
    // (16 mul/add/exp ops per row — small constant).
    src.push_str("        var rowmax = -1e30;\n");
    let _ = writeln!(src, "        for (var j = 0u; j < {bkv}u; j = j + 1u) {{");
    src.push_str("            let kv_pos = t + j;\n");
    src.push_str("            if q_valid && kv_pos >= row_kv_start && kv_pos < row_kv_len {\n");
    let _ = writeln!(
        src,
        "                rowmax = max(rowmax, shared_score[row * {bkv}u + j] * scale);"
    );
    src.push_str("            }\n");
    src.push_str("        }\n");
    src.push_str("        let new_max = max(local_max, rowmax);\n");
    src.push_str("        let correction = select(exp(local_max - new_max), 0.0, !q_valid);\n");

    // Apply correction to the O accumulator (registers).
    let _ = writeln!(
        src,
        "        for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    src.push_str("            local_o[e] = local_o[e] * correction;\n");
    src.push_str("        }\n");

    // PV: for each j compute p_j = exp(score-new_max), accumulate into local_o.
    src.push_str("        var rowsum = 0.0;\n");
    let _ = writeln!(src, "        for (var j = 0u; j < {bkv}u; j = j + 1u) {{");
    src.push_str("            let kv_pos = t + j;\n");
    src.push_str(
        "            let masked = !(q_valid && kv_pos >= row_kv_start && kv_pos < row_kv_len);\n",
    );
    let _ = writeln!(
        src,
        "            let score = shared_score[row * {bkv}u + j] * scale;"
    );
    src.push_str("            let p = select(exp(score - new_max), 0.0, masked);\n");
    src.push_str("            rowsum = rowsum + p;\n");
    let _ = writeln!(
        src,
        "            for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    let _ = writeln!(
        src,
        "                local_o[e] = local_o[e] + p * shared_v[j * {hd}u + d_off + e];"
    );
    src.push_str("            }\n");
    src.push_str("        }\n");
    src.push_str("        local_sum = local_sum * correction + rowsum;\n");
    src.push_str("        local_max = select(local_max, new_max, q_valid);\n");
    src.push_str("        workgroupBarrier();\n");
    src.push_str("    }\n\n");

    // ---- Tail KV (positions not in a full BKV tile) ----
    src.push_str("    for (; t < max_kv_len; t = t + 1u) {\n");
    src.push_str("        let masked = !(q_valid && t >= row_kv_start && t < row_kv_len);\n");
    // Compute QK for this single position from shared_q (everyone reads the same Q, ok).
    src.push_str("        var dot = 0.0;\n");
    let _ = writeln!(src, "        for (var d = 0u; d < {hd}u; d = d + 1u) {{");
    let _ = writeln!(src, "            let qv = f32(shared_q[row * {hd}u + d]);");
    src.push_str("            let kv = src_b[t * kv_dim + kv_head_off + d];\n");
    src.push_str("            dot = dot + qv * kv;\n");
    src.push_str("        }\n");
    src.push_str("        let score = dot * scale;\n");
    src.push_str("        let new_max = select(local_max, max(local_max, score), !masked);\n");
    src.push_str("        let correction = exp(local_max - new_max);\n");
    src.push_str("        let p = select(exp(score - new_max), 0.0, masked);\n");
    let _ = writeln!(
        src,
        "        for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    src.push_str("            let v = bias[t * kv_dim + kv_head_off + d_off + e];\n");
    src.push_str("            local_o[e] = local_o[e] * correction + p * v;\n");
    src.push_str("        }\n");
    src.push_str("        local_sum = local_sum * correction + p;\n");
    src.push_str("        local_max = select(local_max, new_max, q_valid);\n");
    src.push_str("    }\n\n");

    // ---- Final write: divide by sum_exp, write output + LSE ----
    src.push_str("    if q_valid {\n");
    src.push_str("        let safe_sum = select(local_sum, 1.0, local_sum == 0.0);\n");
    src.push_str("        let q_base = qpos * (num_heads * head_dim) + head * head_dim;\n");
    let _ = writeln!(
        src,
        "        for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    src.push_str("            dst[q_base + d_off + e] = local_o[e] / safe_sum;\n");
    src.push_str("        }\n");
    // LSE: only the first chunk-thread per row writes (avoids 4-way duplicate write).
    src.push_str("        if chunk == 0u {\n");
    src.push_str("            let idx = (qpos * num_heads + head) * 2u;\n");
    src.push_str("            lse[idx] = local_max;\n");
    src.push_str("            lse[idx + 1u] = select(log(local_sum), -1e30, local_sum == 0.0);\n");
    src.push_str("        }\n");
    src.push_str("    }\n");
    src.push_str("}\n");

    let module = parse_source(&src)
        .unwrap_or_else(|e| panic!("generated coop flash WGSL failed to parse:\n{e}\n---\n{src}"));
    ShaderModule {
        module,
        source: src,
        hint: "flash_attention_coop",
    }
}

/// Cooperative-matrix flash backward dQ kernel.
///
/// Three matmuls per KV tile, all coop_mat:
///   1. `score = Q @ K^T` (`BQxBKV` from `Q[BQ,hd]`, `K^T[hd,BKV]`)
///   2. `dp = dO @ V^T` (`BQxBKV` from `dO[BQ,hd]`, `V^T[hd,BKV]`)
///   3. `dQ += ds @ K * scale` (`BQxhd_chunk` from `ds[BQ,BKV]`, `K[BKV,hd_chunk]`),
///      one MMA per hd_chunk → 4 separate `coop_mat<f32,C>` accumulators
///      that live across the entire KV loop.
///
/// Per-row `row_sum` is precomputed once at the top (`sum_d dO[i,d]·O[i,d]`).
/// `ds = p · (dp − row_sum)` where `p = exp(score·scale − lse[row])` is
/// computed elementwise in scalar code (1 thread per (row,col) of the
/// 16×16 ds tile = 64 threads · 4 elements/thread).
///
/// Workgroup layout (64 threads = 16 rows × 4 d-chunks):
///   * BQ = BKV = 16 — match coop_mat tile size.
///   * Each workgroup processes one head and 16 query positions, all KV.
///
/// Caller dispatch must use `[ceil(q_seq/16), num_heads, 1]`.
/// `head_dim` must be a multiple of 16.
pub fn generate_flash_grad_q_coop_module(head_dim: u32) -> ShaderModule {
    use std::fmt::Write;
    assert!(
        head_dim >= 16 && head_dim.is_multiple_of(16),
        "coop flash grad_q requires head_dim multiple of 16, got {head_dim}"
    );
    let hd = head_dim;
    let hd_tiles = hd / 16;
    let bq: u32 = 16;
    let bkv: u32 = 16;
    let wg_size: u32 = 64;
    assert!(
        wg_size == 64,
        "coop grad_q assumes wg_size=64 (one warp set)"
    );

    let mut src = String::new();
    src.push_str("enable f16;\n");
    src.push_str("enable wgpu_cooperative_matrix;\n\n");
    src.push_str(ATTENTION_PARAMS_WGSL);
    src.push_str("var<storage> d_out: array<f32>;\n");
    src.push_str("var<storage> src_a: array<f32>;\n"); // Q
    src.push_str("var<storage> src_b: array<f32>;\n"); // K
    src.push_str("var<storage> bias: array<f32>;\n"); // V
    src.push_str("var<storage> lse: array<f32>;\n");
    src.push_str("var<storage> fwd_dst: array<f32>;\n"); // O
    src.push_str("var<storage, read_write> dst: array<f32>;\n"); // dQ
    src.push_str("var<uniform> params: Params;\n\n");

    let _ = writeln!(src, "var<workgroup> shared_q: array<f16, {}>;", bq * hd);
    let _ = writeln!(src, "var<workgroup> shared_do: array<f16, {}>;", bq * hd);
    // Two copies of K: `shared_k_t` is the cooperative operand for the score
    // matmul and has to be f16, `shared_k` is the untransposed copy the scalar
    // `dS·K` loop reads and does not.
    //
    // Staging the second copy as f16 and converting it straight back on read
    // (`f32(shared_k[...])`) is the same avoidable loss the cooperative forward
    // had with V: the accumulator is f32, so the f16 only rounded the value.
    let _ = writeln!(src, "var<workgroup> shared_k: array<f32, {}>;", bkv * hd);
    let _ = writeln!(src, "var<workgroup> shared_k_t: array<f16, {}>;", hd * bkv);
    let _ = writeln!(src, "var<workgroup> shared_v_t: array<f16, {}>;", hd * bkv);
    let _ = writeln!(
        src,
        "var<workgroup> shared_score: array<f32, {}>;",
        bq * bkv
    );
    let _ = writeln!(src, "var<workgroup> shared_dp: array<f32, {}>;", bq * bkv);
    // shared_ds is f32 instead of f16 — we don't feed it into a coop
    // MMA anymore (dQ uses scalar PV-style accumulation instead) and
    // f32 keeps the per-row scaling tighter.
    let _ = writeln!(src, "var<workgroup> shared_ds: array<f32, {}>;", bq * bkv);
    let _ = writeln!(src, "var<workgroup> wg_row_sum: array<f32, {bq}>;");
    let _ = writeln!(src, "var<workgroup> wg_lse_max: array<f32, {bq}>;");
    let _ = writeln!(src, "var<workgroup> wg_lse_log: array<f32, {bq}>;");
    src.push('\n');

    let _ = writeln!(src, "@compute @workgroup_size({wg_size})");
    src.push_str("fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(subgroup_id) sg: u32) {\n");
    let _ = writeln!(src, "    let pos_base = wgid.x * {bq}u;");
    src.push_str("    let head = wgid.y;\n");
    src.push_str("    let q_seq = params.q_seq;\n");
    src.push_str("    let kv_seq = params.kv_seq;\n");
    src.push_str("    let num_heads = params.packed_heads >> 16u;\n");
    src.push_str("    let num_kv_heads = params.packed_heads & 0xFFFFu;\n");
    src.push_str("    let head_dim = params.head_dim;\n");
    src.push_str("    let window_size = params.window_size;\n");
    src.push_str("    let kv_head = head / (num_heads / max(num_kv_heads, 1u));\n");
    src.push_str("    let kv_head_off = kv_head * head_dim;\n");
    src.push_str("    let kv_dim = num_kv_heads * head_dim;\n");
    src.push_str("    let scale = inverseSqrt(f32(head_dim));\n\n");

    // Workgroup-wide KV iteration bounds.
    src.push_str("    let last_pos = min(pos_base + 15u, q_seq - 1u);\n");
    src.push_str("    let max_kv_len = select(kv_seq, last_pos + 1u, kv_seq == 0u);\n");
    src.push_str("    let first_kv_len = select(kv_seq, pos_base + 1u, kv_seq == 0u);\n");
    src.push_str(
        "    let min_kv_start = select(0u, first_kv_len - min(first_kv_len, window_size), window_size > 0u);\n\n",
    );

    // ---- Stage Q + dO into shared (once per workgroup) ----
    let q_total = bq * hd;
    let _ = writeln!(
        src,
        "    for (var i = lid.x; i < {q_total}u; i = i + {wg_size}u) {{"
    );
    let _ = writeln!(src, "        let r = i / {hd}u;");
    let _ = writeln!(src, "        let col = i % {hd}u;");
    src.push_str("        let qp = pos_base + r;\n");
    src.push_str("        if qp < q_seq {\n");
    src.push_str("            let qi = qp * (num_heads * head_dim) + head * head_dim + col;\n");
    src.push_str("            shared_q[i] = f16(src_a[qi]);\n");
    src.push_str("            shared_do[i] = f16(d_out[qi]);\n");
    src.push_str("        } else {\n");
    src.push_str("            shared_q[i] = f16(0.0);\n");
    src.push_str("            shared_do[i] = f16(0.0);\n");
    src.push_str("        }\n");
    src.push_str("    }\n");
    src.push_str("    workgroupBarrier();\n\n");

    // ---- Precompute per-row row_sum = sum_d(dO[r, d] * O[r, d]) and
    //      cache lse{max,log}. 16 threads (one per row).
    let _ = writeln!(src, "    if lid.x < {bq}u {{");
    src.push_str("        let r = lid.x;\n");
    src.push_str("        let qp = pos_base + r;\n");
    src.push_str("        if qp < q_seq && head < num_heads {\n");
    src.push_str("            var s = 0.0;\n");
    src.push_str("            let q_base = qp * (num_heads * head_dim) + head * head_dim;\n");
    let _ = writeln!(
        src,
        "            for (var d = 0u; d < {hd}u; d = d + 1u) {{"
    );
    src.push_str("                s = s + d_out[q_base + d] * fwd_dst[q_base + d];\n");
    src.push_str("            }\n");
    src.push_str("            wg_row_sum[r] = s;\n");
    src.push_str("            let li = (qp * num_heads + head) * 2u;\n");
    src.push_str("            wg_lse_max[r] = lse[li];\n");
    src.push_str("            wg_lse_log[r] = lse[li + 1u];\n");
    src.push_str("        } else {\n");
    src.push_str("            wg_row_sum[r] = 0.0;\n");
    src.push_str("            wg_lse_max[r] = 0.0;\n");
    src.push_str("            wg_lse_log[r] = 0.0;\n");
    src.push_str("        }\n");
    src.push_str("    }\n\n");

    // ---- Per-thread (row, chunk) hoisted indices. Each thread owns
    //      one (row, hd_chunk) of dQ in registers, accumulated across
    //      the entire KV loop.
    let chunks_per_row: u32 = 4;
    let chunk_hd: u32 = hd / chunks_per_row;
    let _ = writeln!(src, "    let row = lid.x / {chunks_per_row}u;");
    let _ = writeln!(src, "    let chunk = lid.x % {chunks_per_row}u;");
    let _ = writeln!(src, "    let d_off = chunk * {chunk_hd}u;");
    src.push_str("    let qpos_thread = pos_base + row;\n");
    src.push_str("    let q_valid_thread = qpos_thread < q_seq && head < num_heads;\n\n");

    // Per-thread dQ accumulator (registers).
    let _ = writeln!(src, "    var local_dq: array<f32, {chunk_hd}>;");
    let _ = writeln!(src, "    for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{");
    src.push_str("        local_dq[e] = 0.0;\n");
    src.push_str("    }\n\n");

    // ---- KV tile loop ----
    let _ = writeln!(
        src,
        "    let tile_end = min_kv_start + ((max_kv_len - min_kv_start) / {bkv}u) * {bkv}u;"
    );
    src.push_str("    var t = min_kv_start;\n");
    let _ = writeln!(src, "    for (; t < tile_end; t = t + {bkv}u) {{");

    // Stage K both ways and V transposed.
    let kv_total = bkv * hd;
    let _ = writeln!(
        src,
        "        for (var i = lid.x; i < {kv_total}u; i = i + {wg_size}u) {{"
    );
    let _ = writeln!(src, "            let ki = i / {hd}u;");
    let _ = writeln!(src, "            let d  = i % {hd}u;");
    src.push_str("            let kv_pos = t + ki;\n");
    src.push_str("            let k_v = src_b[kv_pos * kv_dim + kv_head_off + d];\n");
    src.push_str("            let v_v = bias[kv_pos * kv_dim + kv_head_off + d];\n");
    src.push_str("            shared_k[i] = k_v;\n");
    let _ = writeln!(src, "            shared_k_t[d * {bkv}u + ki] = f16(k_v);");
    let _ = writeln!(src, "            shared_v_t[d * {bkv}u + ki] = f16(v_v);");
    src.push_str("        }\n");
    src.push_str("        workgroupBarrier();\n\n");

    // score = Q @ K^T (B operand from shared_k_t).
    src.push_str("        var score_acc = coop_mat16x16<f32,C>();\n");
    let _ = writeln!(
        src,
        "        for (var ht = 0u; ht < {hd_tiles}u; ht = ht + 1u) {{"
    );
    let _ = writeln!(
        src,
        "            let a = coopLoadT<coop_mat16x16<f16,A>>(&shared_q[ht * 16u], {hd}u);"
    );
    let _ = writeln!(
        src,
        "            let b = coopLoadT<coop_mat16x16<f16,B>>(&shared_k_t[ht * 16u * {bkv}u], {bkv}u);"
    );
    src.push_str("            score_acc = coopMultiplyAdd(a, b, score_acc);\n");
    src.push_str("        }\n");
    let _ = writeln!(
        src,
        "        if sg == 0u {{ coopStoreT(score_acc, &shared_score[0], {bkv}u); }}"
    );

    // dp = dO @ V^T.
    src.push_str("        var dp_acc = coop_mat16x16<f32,C>();\n");
    let _ = writeln!(
        src,
        "        for (var ht = 0u; ht < {hd_tiles}u; ht = ht + 1u) {{"
    );
    let _ = writeln!(
        src,
        "            let a = coopLoadT<coop_mat16x16<f16,A>>(&shared_do[ht * 16u], {hd}u);"
    );
    let _ = writeln!(
        src,
        "            let b = coopLoadT<coop_mat16x16<f16,B>>(&shared_v_t[ht * 16u * {bkv}u], {bkv}u);"
    );
    src.push_str("            dp_acc = coopMultiplyAdd(a, b, dp_acc);\n");
    src.push_str("        }\n");
    let _ = writeln!(
        src,
        "        if sg == 0u {{ coopStoreT(dp_acc, &shared_dp[0], {bkv}u); }}"
    );
    src.push_str("        workgroupBarrier();\n\n");

    // ds = p * (dp - row_sum). 64 threads × 4 elements each = 256 entries
    // (the 16x16 ds tile). Each thread handles 4 consecutive entries.
    let ds_total = bq * bkv;
    let _ = writeln!(src, "        for (var k = 0u; k < 4u; k = k + 1u) {{");
    let _ = writeln!(src, "            let idx = lid.x * 4u + k;");
    let _ = writeln!(src, "            if idx < {ds_total}u {{");
    let _ = writeln!(src, "                let r = idx / {bkv}u;");
    let _ = writeln!(src, "                let j = idx % {bkv}u;");
    src.push_str("                let qpos = pos_base + r;\n");
    src.push_str("                let kv_pos = t + j;\n");
    src.push_str(&kv_range("qpos", 16));
    src.push_str("                let q_valid = qpos < q_seq && head < num_heads;\n");
    src.push_str(
        "                let masked = !(q_valid && kv_pos >= row_kv_start && kv_pos < row_kv_len);\n",
    );
    let _ = writeln!(
        src,
        "                let s = shared_score[r * {bkv}u + j] * scale;"
    );
    src.push_str("                let p = exp(min(s - wg_lse_max[r], 0.0) - wg_lse_log[r]);\n");
    let _ = writeln!(src, "                let dp_v = shared_dp[r * {bkv}u + j];");
    src.push_str("                let ds = select(p * (dp_v - wg_row_sum[r]), 0.0, masked);\n");
    let _ = writeln!(src, "                shared_ds[r * {bkv}u + j] = ds;");
    src.push_str("            }\n");
    src.push_str("        }\n");
    src.push_str("        workgroupBarrier();\n\n");

    // dQ += ds @ K * scale, scalar per-thread accumulation.
    // Each thread (row, chunk) reads its row of shared_ds (16 entries)
    // and shared_k (one chunk per j), accumulates into local_dq.
    let _ = writeln!(src, "        for (var j = 0u; j < {bkv}u; j = j + 1u) {{");
    let _ = writeln!(src, "            let p = shared_ds[row * {bkv}u + j];");
    let _ = writeln!(
        src,
        "            for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    let _ = writeln!(
        src,
        "                let kv = shared_k[j * {hd}u + d_off + e];"
    );
    src.push_str("                local_dq[e] = local_dq[e] + p * kv;\n");
    src.push_str("            }\n");
    src.push_str("        }\n");
    src.push_str("        workgroupBarrier();\n");
    src.push_str("    }\n\n");

    // ---- Tail KV (positions outside a full BKV tile) ----
    // Each thread handles its own (row, chunk) — replicated softmax
    // (4-way), unique chunk-of-K accumulation.
    src.push_str("    for (; t < max_kv_len; t = t + 1u) {\n");
    src.push_str(&kv_range("qpos_thread", 8));
    src.push_str(
        "        let masked = !(q_valid_thread && t >= row_kv_start && t < row_kv_len);\n",
    );
    src.push_str("        if !masked {\n");
    src.push_str("            var dot_qk = 0.0;\n");
    src.push_str("            var dot_dov = 0.0;\n");
    let _ = writeln!(
        src,
        "            for (var d = 0u; d < {hd}u; d = d + 1u) {{"
    );
    let _ = writeln!(
        src,
        "                let qv = f32(shared_q[row * {hd}u + d]);"
    );
    let _ = writeln!(
        src,
        "                let dov = f32(shared_do[row * {hd}u + d]);"
    );
    src.push_str("                let kv = src_b[t * kv_dim + kv_head_off + d];\n");
    src.push_str("                let vv = bias[t * kv_dim + kv_head_off + d];\n");
    src.push_str("                dot_qk = dot_qk + qv * kv;\n");
    src.push_str("                dot_dov = dot_dov + dov * vv;\n");
    src.push_str("            }\n");
    src.push_str(
        "            let p = exp(min(dot_qk * scale - wg_lse_max[row], 0.0) - wg_lse_log[row]);\n",
    );
    src.push_str("            let ds = p * (dot_dov - wg_row_sum[row]);\n");
    let _ = writeln!(
        src,
        "            for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    src.push_str("                let kv = src_b[t * kv_dim + kv_head_off + d_off + e];\n");
    src.push_str("                local_dq[e] = local_dq[e] + ds * kv;\n");
    src.push_str("            }\n");
    src.push_str("        }\n");
    src.push_str("    }\n\n");

    // ---- Final write: scale and store local_dq to global dst ----
    src.push_str("    if q_valid_thread {\n");
    src.push_str("        let dst_row_stride = num_heads * head_dim;\n");
    src.push_str("        let q_base = qpos_thread * dst_row_stride + head * head_dim + d_off;\n");
    let _ = writeln!(
        src,
        "        for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    src.push_str("            dst[q_base + e] = local_dq[e] * scale;\n");
    src.push_str("        }\n");
    src.push_str("    }\n");
    src.push_str("}\n");

    let module = parse_source(&src).unwrap_or_else(|e| {
        panic!("generated coop flash grad_q WGSL failed to parse:\n{e}\n---\n{src}")
    });
    ShaderModule {
        module,
        source: src,
        hint: "flash_grad_q_coop",
    }
}

/// Cooperative-matrix flash backward dK + dV kernel.
///
/// Workgroup processes one head and 16 KV positions, iterating through
/// all queries in tiles of 16. Per Q-tile:
///   1. `row_sum[q] = sum_d(dO[q,d]·O[q,d])` precomputed (16 threads).
///   2. `score = K @ Q^T` via `coop_mat` → `shared_score[BKV, BQ]`
///   3. `dp = V @ dO^T` via `coop_mat` → `shared_dp[BKV, BQ]`
///   4. `p[kv,q] = exp(score·scale − lse[q])`,
///      `ds[kv,q] = p · (dp − row_sum[q])` scalar elementwise
///   5. `dV[kv,d] += sum_q(p[kv,q] · dO[q,d])` scalar per-thread
///   6. `dK[kv,d] += sum_q(ds[kv,q] · Q[q,d]) · scale` scalar per-thread
///
/// dV and dK accumulate in per-thread registers (chunk_hd=16 each)
/// across the entire query loop — same design as the forward and
/// GradQ coop kernels (the alternative `coop_mat` accumulator
/// spanning the loop hits a naga / shared-memory race).
///
/// Workgroup layout (64 threads = 16 kv-rows × 4 d-chunks).
/// Caller dispatch must use `[ceil(dispatch_kv/16), num_kv_heads, 1]`.
/// `head_dim` must be a multiple of 16.
pub fn generate_flash_grad_kv_coop_module(head_dim: u32) -> ShaderModule {
    use std::fmt::Write;
    assert!(
        head_dim >= 16 && head_dim.is_multiple_of(16),
        "coop flash grad_kv requires head_dim multiple of 16, got {head_dim}"
    );
    let hd = head_dim;
    let hd_tiles = hd / 16;
    let bq: u32 = 16;
    let bkv: u32 = 16;
    let wg_size: u32 = 64;
    let chunks_per_row: u32 = 4;
    let chunk_hd: u32 = hd / chunks_per_row;

    let mut src = String::new();
    src.push_str("enable f16;\n");
    src.push_str("enable wgpu_cooperative_matrix;\n\n");
    src.push_str(ATTENTION_PARAMS_WGSL);
    src.push_str("var<storage> d_out: array<f32>;\n");
    src.push_str("var<storage> src_a: array<f32>;\n"); // Q
    src.push_str("var<storage> src_b: array<f32>;\n"); // K
    src.push_str("var<storage> bias: array<f32>;\n"); // V
    src.push_str("var<storage> lse: array<f32>;\n");
    src.push_str("var<storage> fwd_dst: array<f32>;\n"); // O
    src.push_str("var<storage, read_write> dst: array<f32>;\n"); // dK
    src.push_str("var<storage, read_write> dst2: array<f32>;\n"); // dV
    src.push_str("var<uniform> params: Params;\n\n");

    let _ = writeln!(src, "var<workgroup> shared_k: array<f16, {}>;", bkv * hd);
    let _ = writeln!(src, "var<workgroup> shared_v: array<f16, {}>;", bkv * hd);
    // Four copies of the Q-side data, and only two of them are cooperative
    // operands. `shared_k`, `shared_k_t`, `shared_v` and `shared_do_t` feed
    // `coopLoadT` for the score and `dp` matmuls and have to be f16;
    // `shared_q` and `shared_do` are the untransposed copies that the scalar
    // `dS·Q` (for dK) and `p·dO` (for dV) loops read, and do not.
    //
    // Those two were staged as f16 and converted straight back on read, which
    // rounds a value the f32 accumulator then had to work around.
    let _ = writeln!(src, "var<workgroup> shared_q: array<f32, {}>;", bq * hd);
    let _ = writeln!(src, "var<workgroup> shared_q_t: array<f16, {}>;", hd * bq);
    let _ = writeln!(src, "var<workgroup> shared_do: array<f32, {}>;", bq * hd);
    let _ = writeln!(src, "var<workgroup> shared_do_t: array<f16, {}>;", hd * bq);
    let _ = writeln!(
        src,
        "var<workgroup> shared_score: array<f32, {}>;",
        bkv * bq
    );
    let _ = writeln!(src, "var<workgroup> shared_dp: array<f32, {}>;", bkv * bq);
    let _ = writeln!(src, "var<workgroup> shared_p: array<f32, {}>;", bkv * bq);
    let _ = writeln!(src, "var<workgroup> shared_ds: array<f32, {}>;", bkv * bq);
    let _ = writeln!(src, "var<workgroup> wg_row_sum: array<f32, {bq}>;");
    let _ = writeln!(
        src,
        "var<workgroup> wg_row_sum_partial: array<f32, {wg_size}>;"
    );
    let _ = writeln!(src, "var<workgroup> wg_lse_max: array<f32, {bq}>;");
    let _ = writeln!(src, "var<workgroup> wg_lse_log: array<f32, {bq}>;");
    src.push('\n');

    let _ = writeln!(src, "@compute @workgroup_size({wg_size})");
    src.push_str("fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(subgroup_id) sg: u32) {\n");
    let _ = writeln!(src, "    let kv_base = wgid.x * {bkv}u;");
    src.push_str("    let kv_head = wgid.y;\n");
    src.push_str("    let q_seq = params.q_seq;\n");
    src.push_str("    let kv_seq = params.kv_seq;\n");
    src.push_str("    let num_heads = params.packed_heads >> 16u;\n");
    src.push_str("    let num_kv_heads = params.packed_heads & 0xFFFFu;\n");
    src.push_str("    let head_dim = params.head_dim;\n");
    src.push_str("    let window_size = params.window_size;\n");
    src.push_str("    let heads_per_kv = num_heads / max(num_kv_heads, 1u);\n");
    src.push_str("    let kv_head_off = kv_head * head_dim;\n");
    src.push_str("    let kv_dim = num_kv_heads * head_dim;\n");
    src.push_str("    let q_dim = num_heads * head_dim;\n");
    src.push_str("    let scale = inverseSqrt(f32(head_dim));\n");
    src.push_str("    let effective_kv_seq = select(kv_seq, q_seq, kv_seq == 0u);\n\n");

    // Per-thread (kv_row, chunk) hoisted indices.
    let _ = writeln!(src, "    let kv_row = lid.x / {chunks_per_row}u;");
    let _ = writeln!(src, "    let chunk = lid.x % {chunks_per_row}u;");
    let _ = writeln!(src, "    let d_off = chunk * {chunk_hd}u;");
    src.push_str("    let kv_pos_thread = kv_base + kv_row;\n");
    src.push_str(
        "    let kv_valid_thread = kv_pos_thread < effective_kv_seq && kv_head < num_kv_heads;\n\n",
    );

    // Per-thread dV and dK accumulators (registers).
    let _ = writeln!(src, "    var local_dv: array<f32, {chunk_hd}>;");
    let _ = writeln!(src, "    var local_dk: array<f32, {chunk_hd}>;");
    let _ = writeln!(src, "    for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{");
    src.push_str("        local_dv[e] = 0.0;\n");
    src.push_str("        local_dk[e] = 0.0;\n");
    src.push_str("    }\n\n");

    // ---- Stage K and V once per workgroup ----
    let kv_total = bkv * hd;
    let _ = writeln!(
        src,
        "    for (var i = lid.x; i < {kv_total}u; i = i + {wg_size}u) {{"
    );
    let _ = writeln!(src, "        let ki = i / {hd}u;");
    let _ = writeln!(src, "        let d  = i % {hd}u;");
    src.push_str("        let kp = kv_base + ki;\n");
    src.push_str("        if kp < effective_kv_seq {\n");
    src.push_str("            let kv_off = kp * kv_dim + kv_head_off + d;\n");
    src.push_str("            shared_k[i] = f16(src_b[kv_off]);\n");
    src.push_str("            shared_v[i] = f16(bias[kv_off]);\n");
    src.push_str("        } else {\n");
    src.push_str("            shared_k[i] = f16(0.0);\n");
    src.push_str("            shared_v[i] = f16(0.0);\n");
    src.push_str("        }\n");
    src.push_str("    }\n");
    src.push_str("    workgroupBarrier();\n\n");

    // GQA: this kernel processes one KV head, but multiple Q heads
    // map to it. Iterate q-heads (heads_per_kv) and within each, all
    // queries in BQ tiles. Outer loop is q-head, inner loop is q-pos.
    src.push_str("    for (var qh = 0u; qh < heads_per_kv; qh = qh + 1u) {\n");
    src.push_str("        let q_head = kv_head * heads_per_kv + qh;\n\n");

    // Per-row valid Q range (causal: only q >= kv_pos can attend; i.e.
    // for KV position kv_pos_thread, valid q_pos in [kv_pos_thread, q_seq))
    src.push_str("        let row_q_start = select(0u, kv_pos_thread, kv_seq == 0u);\n");
    // For sliding window: q must be in (kv_pos_thread - window, kv_pos_thread]
    // — implemented elementwise inside the softmax phase below.

    // ---- Q-tile loop ----
    let _ = writeln!(src, "        let tile_end = (q_seq / {bq}u) * {bq}u;");
    src.push_str("        var t = 0u;\n");
    let _ = writeln!(src, "        for (; t < tile_end; t = t + {bq}u) {{");

    // Stage Q, dO, O for this tile (and their transposes for coop).
    let q_total = bq * hd;
    let _ = writeln!(
        src,
        "            for (var i = lid.x; i < {q_total}u; i = i + {wg_size}u) {{"
    );
    let _ = writeln!(src, "                let qi = i / {hd}u;");
    let _ = writeln!(src, "                let d  = i % {hd}u;");
    src.push_str("                let qp = t + qi;\n");
    src.push_str("                if qp < q_seq {\n");
    src.push_str("                    let q_off = qp * q_dim + q_head * head_dim + d;\n");
    src.push_str("                    let qv = src_a[q_off];\n");
    src.push_str("                    let dov = d_out[q_off];\n");
    src.push_str("                    shared_q[i] = qv;\n");
    src.push_str("                    shared_do[i] = dov;\n");
    let _ = writeln!(
        src,
        "                    shared_q_t[d * {bq}u + qi] = f16(qv);"
    );
    let _ = writeln!(
        src,
        "                    shared_do_t[d * {bq}u + qi] = f16(dov);"
    );
    src.push_str("                } else {\n");
    src.push_str("                    shared_q[i] = 0.0;\n");
    src.push_str("                    shared_do[i] = 0.0;\n");
    let _ = writeln!(
        src,
        "                    shared_q_t[d * {bq}u + qi] = f16(0.0);"
    );
    let _ = writeln!(
        src,
        "                    shared_do_t[d * {bq}u + qi] = f16(0.0);"
    );
    src.push_str("                }\n");
    src.push_str("            }\n");
    src.push_str("            workgroupBarrier();\n\n");

    // Precompute row_sum[qi] = dot(dO[qi], O[qi]). Map four threads to
    // contiguous chunks of each Q row, matching the workgroup's existing
    // (row, chunk) layout. The old one-thread-per-row loop left 48/64
    // threads idle and issued strided loads across 16 distant Q rows.
    src.push_str("            let row_qp = t + kv_row;\n");
    src.push_str("            var row_part = 0.0;\n");
    src.push_str("            if row_qp < q_seq && q_head < num_heads {\n");
    src.push_str("                let q_base = row_qp * q_dim + q_head * head_dim + d_off;\n");
    let _ = writeln!(
        src,
        "                for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    let _ = writeln!(
        src,
        "                    row_part = row_part + shared_do[kv_row * {hd}u + d_off + e] * fwd_dst[q_base + e];"
    );
    src.push_str("                }\n");
    src.push_str("            }\n");
    src.push_str("            wg_row_sum_partial[lid.x] = row_part;\n");
    src.push_str("            workgroupBarrier();\n");
    let _ = writeln!(src, "            if lid.x < {bq}u {{");
    let _ = writeln!(src, "                let base = lid.x * {chunks_per_row}u;");
    src.push_str("                wg_row_sum[lid.x] = wg_row_sum_partial[base] + wg_row_sum_partial[base + 1u] + wg_row_sum_partial[base + 2u] + wg_row_sum_partial[base + 3u];\n");
    src.push_str("                let qp = t + lid.x;\n");
    src.push_str("                if qp < q_seq && q_head < num_heads {\n");
    src.push_str("                    let li = (qp * num_heads + q_head) * 2u;\n");
    src.push_str("                    wg_lse_max[lid.x] = lse[li];\n");
    src.push_str("                    wg_lse_log[lid.x] = lse[li + 1u];\n");
    src.push_str("                } else {\n");
    src.push_str("                    wg_lse_max[lid.x] = 0.0;\n");
    src.push_str("                    wg_lse_log[lid.x] = 0.0;\n");
    src.push_str("                }\n");
    src.push_str("            }\n");
    src.push_str("            workgroupBarrier();\n\n");

    // score = K @ Q^T (BKV x BQ).
    src.push_str("            var score_acc = coop_mat16x16<f32,C>();\n");
    let _ = writeln!(
        src,
        "            for (var ht = 0u; ht < {hd_tiles}u; ht = ht + 1u) {{"
    );
    let _ = writeln!(
        src,
        "                let a_k = coopLoadT<coop_mat16x16<f16,A>>(&shared_k[ht * 16u], {hd}u);"
    );
    let _ = writeln!(
        src,
        "                let b_qt = coopLoadT<coop_mat16x16<f16,B>>(&shared_q_t[ht * 16u * {bq}u], {bq}u);"
    );
    src.push_str("                score_acc = coopMultiplyAdd(a_k, b_qt, score_acc);\n");
    src.push_str("            }\n");
    let _ = writeln!(
        src,
        "            if sg == 0u {{ coopStoreT(score_acc, &shared_score[0], {bq}u); }}"
    );

    // dp = V @ dO^T (BKV x BQ).
    src.push_str("            var dp_acc = coop_mat16x16<f32,C>();\n");
    let _ = writeln!(
        src,
        "            for (var ht = 0u; ht < {hd_tiles}u; ht = ht + 1u) {{"
    );
    let _ = writeln!(
        src,
        "                let a_v = coopLoadT<coop_mat16x16<f16,A>>(&shared_v[ht * 16u], {hd}u);"
    );
    let _ = writeln!(
        src,
        "                let b_dot = coopLoadT<coop_mat16x16<f16,B>>(&shared_do_t[ht * 16u * {bq}u], {bq}u);"
    );
    src.push_str("                dp_acc = coopMultiplyAdd(a_v, b_dot, dp_acc);\n");
    src.push_str("            }\n");
    let _ = writeln!(
        src,
        "            if sg == 0u {{ coopStoreT(dp_acc, &shared_dp[0], {bq}u); }}"
    );
    src.push_str("            workgroupBarrier();\n\n");

    // p[kv, q] = exp(score * scale - lse[q]); ds[kv, q] = p * (dp - row_sum[q]).
    // 64 threads × 4 entries each = 256 = full BKV*BQ tile.
    let pq_total = bkv * bq;
    src.push_str("            for (var k = 0u; k < 4u; k = k + 1u) {\n");
    let _ = writeln!(src, "                let idx = lid.x * 4u + k;");
    let _ = writeln!(src, "                if idx < {pq_total}u {{");
    let _ = writeln!(src, "                    let kv = idx / {bq}u;");
    let _ = writeln!(src, "                    let q  = idx % {bq}u;");
    src.push_str("                    let kp = kv_base + kv;\n");
    src.push_str("                    let qp = t + q;\n");
    src.push_str(
        "                    let masked = !(kp < effective_kv_seq && kv_head < num_kv_heads && qp < q_seq && q_head < num_heads);\n",
    );
    // Causal: qp >= kp. Sliding window: qp - window < kp <= qp.
    src.push_str(&kv_range("qp", 20));
    src.push_str(
        "                    let attn_masked = masked || (kp < row_kv_start) || (kp >= row_kv_len);\n",
    );
    let _ = writeln!(
        src,
        "                    let s = shared_score[kv * {bq}u + q] * scale;"
    );
    src.push_str("                    let p = exp(min(s - wg_lse_max[q], 0.0) - wg_lse_log[q]);\n");
    let _ = writeln!(
        src,
        "                    let dp_v = shared_dp[kv * {bq}u + q];"
    );
    src.push_str("                    let ds = p * (dp_v - wg_row_sum[q]);\n");
    src.push_str("                    let p_safe = select(p, 0.0, attn_masked);\n");
    src.push_str("                    let ds_safe = select(ds, 0.0, attn_masked);\n");
    let _ = writeln!(
        src,
        "                    shared_p[kv * {bq}u + q] = p_safe;"
    );
    let _ = writeln!(
        src,
        "                    shared_ds[kv * {bq}u + q] = ds_safe;"
    );
    src.push_str("                }\n");
    src.push_str("            }\n");
    src.push_str("            workgroupBarrier();\n\n");

    // dV[kv,d] += sum_q(p[kv,q] · dO[q,d])  scalar
    // dK[kv,d] += sum_q(ds[kv,q] · Q[q,d])  scalar (scale applied at final write)
    let _ = writeln!(
        src,
        "            for (var q = 0u; q < {bq}u; q = q + 1u) {{"
    );
    let _ = writeln!(src, "                let p = shared_p[kv_row * {bq}u + q];");
    let _ = writeln!(
        src,
        "                let ds = shared_ds[kv_row * {bq}u + q];"
    );
    let _ = writeln!(
        src,
        "                for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    let _ = writeln!(
        src,
        "                    let dov = shared_do[q * {hd}u + d_off + e];"
    );
    let _ = writeln!(
        src,
        "                    let qv  = shared_q[q * {hd}u + d_off + e];"
    );
    src.push_str("                    local_dv[e] = local_dv[e] + p  * dov;\n");
    src.push_str("                    local_dk[e] = local_dk[e] + ds * qv;\n");
    src.push_str("                }\n");
    src.push_str("            }\n");
    src.push_str("            workgroupBarrier();\n");
    src.push_str("        }\n\n");

    // Tail Q (positions outside a full BQ tile) — scalar fallback.
    src.push_str("        for (; t < q_seq; t = t + 1u) {\n");
    src.push_str("            let masked = !(kv_valid_thread && q_head < num_heads);\n");
    src.push_str("            if !masked {\n");
    src.push_str(&kv_range("t", 16));
    src.push_str(
        "                let attn_masked = (kv_pos_thread < row_kv_start) || (kv_pos_thread >= row_kv_len);\n",
    );
    src.push_str("                if !attn_masked {\n");
    src.push_str("                    var dot_qk = 0.0;\n");
    src.push_str("                    var dot_dov = 0.0;\n");
    src.push_str("                    var dot_doo = 0.0;\n");
    let _ = writeln!(
        src,
        "                    for (var d = 0u; d < {hd}u; d = d + 1u) {{"
    );
    src.push_str(
        "                        let kv = src_b[kv_pos_thread * kv_dim + kv_head_off + d];\n",
    );
    src.push_str(
        "                        let vv = bias[kv_pos_thread * kv_dim + kv_head_off + d];\n",
    );
    src.push_str("                        let qv = src_a[t * q_dim + q_head * head_dim + d];\n");
    src.push_str("                        let dov = d_out[t * q_dim + q_head * head_dim + d];\n");
    src.push_str("                        let ov = fwd_dst[t * q_dim + q_head * head_dim + d];\n");
    src.push_str("                        dot_qk = dot_qk + qv * kv;\n");
    src.push_str("                        dot_dov = dot_dov + dov * vv;\n");
    src.push_str("                        dot_doo = dot_doo + dov * ov;\n");
    src.push_str("                    }\n");
    src.push_str("                    let li = (t * num_heads + q_head) * 2u;\n");
    src.push_str("                    let lmax = lse[li];\n");
    src.push_str("                    let llog = lse[li + 1u];\n");
    src.push_str("                    let p = exp(min(dot_qk * scale - lmax, 0.0) - llog);\n");
    src.push_str("                    let ds = p * (dot_dov - dot_doo);\n");
    let _ = writeln!(
        src,
        "                    for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    src.push_str(
        "                        let dov = d_out[t * q_dim + q_head * head_dim + d_off + e];\n",
    );
    src.push_str(
        "                        let qv = src_a[t * q_dim + q_head * head_dim + d_off + e];\n",
    );
    src.push_str("                        local_dv[e] = local_dv[e] + p  * dov;\n");
    src.push_str("                        local_dk[e] = local_dk[e] + ds * qv;\n");
    src.push_str("                    }\n");
    src.push_str("                }\n");
    src.push_str("            }\n");
    src.push_str("        }\n");
    src.push_str("    }\n\n"); // end qh loop

    // ---- Final write: scale (only dK), store to global dK / dV ----
    src.push_str("    if kv_valid_thread {\n");
    src.push_str("        let kv_dst_off = kv_pos_thread * kv_dim + kv_head_off + d_off;\n");
    let _ = writeln!(
        src,
        "        for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    src.push_str("            dst[kv_dst_off + e] = local_dk[e] * scale;\n");
    src.push_str("            dst2[kv_dst_off + e] = local_dv[e];\n");
    src.push_str("        }\n");
    src.push_str("    }\n");
    src.push_str("}\n");

    let module = parse_source(&src).unwrap_or_else(|e| {
        panic!("generated coop flash grad_kv WGSL failed to parse:\n{e}\n---\n{src}")
    });
    ShaderModule {
        module,
        source: src,
        hint: "flash_grad_kv_coop",
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
