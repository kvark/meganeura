//! Attention and cooperative convolution kernels.

use super::{
    ATTENTION_PARAMS_WGSL, CACHED_ATTENTION_PARAMS_WGSL, CoopConfig, FlashAttentionShape,
    ShaderGroup, ShaderModule, parse_source, preprocess, template_section,
};

/// Workgroup storage for the fixed 16-query, 16-key cooperative tiles.
/// The fixed terms include score tiles and backward row statistics.
pub(crate) fn attention_coop_shared_bytes(group: ShaderGroup, head_dim: u32) -> u64 {
    let (per_dim, fixed) = match group {
        ShaderGroup::FlashAttentionCoop => (128u64, 1024),
        ShaderGroup::FlashGradQCoopF16 => (192, 3264),
        ShaderGroup::FlashGradKVCoopF16 => (256, 4544),
        _ => unreachable!("shared-memory accounting requires cooperative attention"),
    };
    per_dim * u64::from(head_dim) + fixed
}

/// Split a head into power-of-two thread groups, with `ept * tpq = head_dim`.
/// For non-power-of-two widths, `ept` can exceed the requested cap.
pub fn attention_lanes(head_dim: u32, ept_cap: u32) -> (u32, u32) {
    assert!(head_dim >= 1, "attention needs a nonempty head");
    let padded = head_dim.next_power_of_two();
    let widest = padded / padded.min(ept_cap.max(1));
    let tpq = widest.min(1 << head_dim.trailing_zeros());
    (head_dim / tpq, tpq)
}

/// Elements per thread of the cached multi-query kernel.
const CACHED_ATTENTION_EPT: u32 = 8;

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

fn attention_module(source: String, hint: &'static str) -> ShaderModule {
    let module = parse_source(&source)
        .unwrap_or_else(|e| panic!("generated {hint} WGSL failed to parse:\n{e}\n---\n{source}"));
    ShaderModule {
        module,
        source,
        hint,
    }
}

fn unroll_elements(fragment: &str, count: u32, stride: u32) -> String {
    (0..count)
        .map(|index| {
            preprocess(
                fragment,
                &[
                    ("$INDEX", &index.to_string()),
                    ("$OFFSET", &(index * stride).to_string()),
                ],
            )
        })
        .collect()
}

fn unroll_reduction(fragment: &str, lanes: u32) -> String {
    let mut source = String::new();
    let mut stride = lanes / 2;
    while stride > 0 {
        source.push_str(&preprocess(fragment, &[("$STRIDE", &stride.to_string())]));
        stride /= 2;
    }
    source
}

fn attention_tile_load(template: &str, elements: u32, threads: u32) -> String {
    (0..elements.div_ceil(threads))
        .map(|index| {
            preprocess(
                template_section(
                    template,
                    if index == 0 {
                        "tile_load_first"
                    } else {
                        "tile_load_next"
                    },
                ),
                &[("$OFFSET", &(index * threads).to_string())],
            )
        })
        .collect()
}

fn attention_score_reduce(lanes: u32, keys: &str) -> String {
    let template = include_str!("../shaders/attention.wgsl");
    preprocess(
        template_section(template, "score_reduce"),
        &[
            (
                "$SCORE_STEP",
                &unroll_reduction(template_section(template, "score_step"), lanes),
            ),
            ("$HEAD_DIM", &lanes.to_string()),
            ("$KEY_COUNT", keys),
        ],
    )
}

/// One query per workgroup, with eight keys per tile and a padded lane per dimension.
pub fn generate_attention_module(head_dim: u32) -> ShaderModule {
    assert!(head_dim >= 1, "attention needs a nonempty head");
    let lanes = head_dim.next_power_of_two().max(2);
    let template = include_str!("../shaders/attention.wgsl");
    attention_module(
        preprocess(
            template_section(template, "main"),
            &[
                ("$PARAMS", ATTENTION_PARAMS_WGSL),
                ("$SCORE_REDUCE", &attention_score_reduce(lanes, "8u")),
                (
                    "$DOT_STEP",
                    &unroll_reduction(template_section(template, "dot_step"), lanes),
                ),
                ("$TILE_ELEMENTS", &(8 * lanes).to_string()),
                ("$HEAD_DIM", &lanes.to_string()),
                ("$LAST", &(head_dim - 1).to_string()),
            ],
        ),
        "attention",
    )
}

pub fn generate_module_block_attention() -> ShaderModule {
    generate_cached_attention_module(ShaderGroup::CachedBlockAttention, None)
}

pub(crate) fn generate_cached_attention_module(
    group: ShaderGroup,
    head_dim: Option<u32>,
) -> ShaderModule {
    let template = match group {
        ShaderGroup::CachedBlockAttention => include_str!("../shaders/cached_block_attention.wgsl"),
        ShaderGroup::CachedBlockAttentionSplit => {
            include_str!("../shaders/cached_block_attention_split.wgsl")
        }
        ShaderGroup::CachedBlockAttentionCombine => {
            include_str!("../shaders/cached_block_attention_combine.wgsl")
        }
        _ => unreachable!("not cached attention: {group:?}"),
    };
    let (dimension, values) = match head_dim {
        Some(hd) => {
            assert!((1..=512).contains(&hd));
            (format!("{hd}u"), hd.div_ceil(64))
        }
        None => ("params.head_dim".to_string(), 8),
    };
    ShaderModule::new(&preprocess(
        template,
        &[
            ("$SCORE_REDUCE", &attention_score_reduce(64, "BKV")),
            ("$HEAD_DIM", &dimension),
            ("$VALUES_PER_THREAD", &format!("{values}u")),
        ],
    ))
}

fn generate_flash_attention(
    head_dim: u32,
    ept_cap: u32,
    shape: FlashAttentionShape,
    cached: bool,
) -> ShaderModule {
    assert!(matches!(shape.threads, 128 | 256) && shape.keys.is_power_of_two() && shape.keys <= 16);
    let (ept, tpq) = attention_lanes(head_dim, ept_cap);
    let bq = (shape.threads / tpq).max(1);
    if bq <= 1 {
        assert!(
            !cached,
            "cached attention needs several queries per workgroup"
        );
        return generate_attention_module(head_dim);
    }
    let threads = bq * tpq;
    let template = include_str!("../shaders/flash_attention.wgsl");
    let fragment = |name| template_section(template, name);
    let elements =
        |name| unroll_elements(fragment(name), ept, if shape.interleave { tpq } else { 1 });
    let source = preprocess(
        fragment("main"),
        &[
            (
                "$PARAMS",
                if cached {
                    CACHED_ATTENTION_PARAMS_WGSL
                } else {
                    ATTENTION_PARAMS_WGSL
                },
            ),
            (
                "$CACHED_BINDING",
                if cached {
                    fragment("cached_binding")
                } else {
                    ""
                },
            ),
            (
                "$LSE_BINDING",
                if cached { "" } else { fragment("lse_binding") },
            ),
            (
                "$DIMENSIONS",
                fragment(if cached {
                    "cached_dimensions"
                } else {
                    "dimensions"
                }),
            ),
            (
                "$LSE_STORE",
                if cached { "" } else { fragment("lse_store") },
            ),
            ("$WINDOW", if cached { "0u" } else { "params.window_size" }),
            (
                "$SCORE_STEP",
                &unroll_reduction(fragment("score_step"), tpq),
            ),
            ("$DOT_STEP", &unroll_reduction(fragment("dot_step"), tpq)),
            ("$Q_INIT", &elements("q_init")),
            ("$Q_LOAD", &elements("q_load")),
            ("$OUT_INIT", &elements("out_init")),
            ("$TILE_DOT", &elements("tile_dot")),
            ("$TILE_ACCUMULATE", &elements("tile_accumulate")),
            ("$TAIL_DOT", &elements("tail_dot")),
            ("$TAIL_ACCUMULATE", &elements("tail_accumulate")),
            ("$OUT_STORE", &elements("out_store")),
            (
                "$TILE_LOAD",
                &attention_tile_load(template, shape.keys * head_dim, threads),
            ),
            ("$TILE_ELEMENTS", &(shape.keys * head_dim).to_string()),
            ("$SCORES", &(threads * shape.keys).to_string()),
            (
                "$D_BASE_STRIDE",
                &(if shape.interleave { 1 } else { ept }).to_string(),
            ),
            ("$HEAD_DIM", &head_dim.to_string()),
            ("$THREADS", &threads.to_string()),
            ("$TPQ", &tpq.to_string()),
            ("$BQ", &bq.to_string()),
            ("$BKV", &shape.keys.to_string()),
        ],
    );
    attention_module(
        source,
        if cached {
            "cached_query_attention"
        } else {
            "flash_attention"
        },
    )
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

/// F16-input cooperative dQ: dispatch `[ceil(q_seq/16), num_heads, 1]`.
pub fn generate_flash_grad_q_coop_f16_module(head_dim: u32) -> ShaderModule {
    generate_coop_attention(
        include_str!("../shaders/flash_grad_q_coop_f16.wgsl"),
        head_dim,
        "flash_grad_q_coop_f16",
    )
}

/// F16-input cooperative dK/dV: dispatch `[ceil(dispatch_kv/16), num_kv_heads, 1]`.
pub fn generate_flash_grad_kv_coop_f16_module(head_dim: u32) -> ShaderModule {
    generate_coop_attention(
        include_str!("../shaders/flash_grad_kv_coop_f16.wgsl"),
        head_dim,
        "flash_grad_kv_coop_f16",
    )
}

/// F32 8x8 cooperative dK/dV, with 16-key tiles and full-precision matrix products.
/// The initial specialization covers the 64-wide transformer heads.
pub fn generate_flash_grad_kv_coop_f32_module(head_dim: u32) -> ShaderModule {
    assert_eq!(head_dim, 64);
    let mut init = String::new();
    let mut accumulate = String::new();
    let mut store = String::new();
    for column in 0..4 {
        init.push_str(&format!(
            "var dk{column} = coop_mat8x8<f32,C>();\n\
             var dv{column} = coop_mat8x8<f32,C>();\n"
        ));
        accumulate.push_str(&format!(
            "let b{column} = q * 64u + out_col + {}u;\n\
             let q{column} = coopLoadT<coop_mat8x8<f32,B>>(&shared_q[b{column}], 64u);\n\
             let do{column} = coopLoadT<coop_mat8x8<f32,B>>(&shared_do[b{column}], 64u);\n\
             dk{column} = coopMultiplyAdd(ds, q{column}, dk{column});\n\
             dv{column} = coopMultiplyAdd(p, do{column}, dv{column});\n",
            column * 8
        ));
        store.push_str(&format!(
            "let c{column} = out_row * 64u + out_col + {}u;\n\
             if tile_sg < 4u {{\n\
                 coopStoreT(dk{column}, &shared_k[c{column}], 64u);\n\
                 coopStoreT(dv{column}, &shared_v[c{column}], 64u);\n\
             }}\n",
            column * 8
        ));
    }
    attention_module(
        preprocess(
            include_str!("../shaders/flash_grad_kv_coop_f32.wgsl"),
            &[
                ("$PARAMS", ATTENTION_PARAMS_WGSL.trim_end()),
                ("$ACC_INIT", &init),
                ("$ACCUMULATE", &accumulate),
                ("$ACC_STORE", &store),
            ],
        ),
        "flash_grad_kv_coop_f32",
    )
}

pub(crate) const FLASH_GRAD_COOP_F32_SHARED_BYTES: u32 = (4 * 1024 + 2 * 256 + 3 * 16) * 4;

/// F32 8x8 cooperative dQ, dispatched as `[ceil(q_seq/16), num_heads, 1]`.
pub fn generate_flash_grad_q_coop_f32_module(head_dim: u32) -> ShaderModule {
    assert_eq!(head_dim, 64);
    let mut init = String::new();
    let mut accumulate = String::new();
    let mut store = String::new();
    for column in 0..4 {
        init.push_str(&format!("var dq{column} = coop_mat8x8<f32,C>();\n"));
        accumulate.push_str(&format!(
            "let b{column} = k * 64u + out_col + {}u;\n\
             let k{column} = coopLoadT<coop_mat8x8<f32,B>>(&shared_k[b{column}], 64u);\n\
             dq{column} = coopMultiplyAdd(ds, k{column}, dq{column});\n",
            column * 8
        ));
        store.push_str(&format!(
            "let c{column} = out_row * 64u + out_col + {}u;\n\
             if tile_sg < 4u {{ coopStoreT(dq{column}, &shared_q[c{column}], 64u); }}\n",
            column * 8
        ));
    }
    attention_module(
        preprocess(
            include_str!("../shaders/flash_grad_q_coop_f32.wgsl"),
            &[
                ("$PARAMS", ATTENTION_PARAMS_WGSL.trim_end()),
                ("$ACC_INIT", &init),
                ("$ACCUMULATE", &accumulate),
                ("$ACC_STORE", &store),
            ],
        ),
        "flash_grad_q_coop_f32",
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
    attention_module(source, hint)
}

fn backward_attention_tile(head_dim: u32, threads: u32) -> u32 {
    // Budget 16 KiB for two input tiles and two partial-dot arrays.
    (2048 / (head_dim + threads)).clamp(1, 4)
}

/// dQ stages K/V tiles; multi-lane queries also reduce partial dot products.
pub fn generate_flash_grad_q_module(head_dim: u32, ept_cap: u32) -> ShaderModule {
    let (ept, tpq) = attention_lanes(head_dim, ept_cap);
    let bq = (256 / tpq).max(1);
    if bq <= 1 {
        return ShaderModule::new(include_str!("../shaders/mha_grad_q.wgsl"));
    }
    let threads = bq * tpq;
    let bkv = if tpq == 1 {
        8
    } else {
        backward_attention_tile(head_dim, threads)
    };
    let template = include_str!("../shaders/flash_grad_q.wgsl");
    let fragment = |name| template_section(template, name);
    let elements = |name| unroll_elements(fragment(name), ept, 1);
    let source = preprocess(
        fragment("main"),
        &[
            ("$PARAMS", ATTENTION_PARAMS_WGSL),
            (
                "$REDUCTION",
                if tpq > 1 { fragment("reduction") } else { "" },
            ),
            (
                "$REDUCE_STEP",
                &unroll_reduction(fragment("reduce_step"), tpq),
            ),
            (
                "$KV_LOOP",
                fragment(if tpq == 1 {
                    "tiled_kv_loop"
                } else {
                    "grouped_kv_loop"
                }),
            ),
            ("$Q_INIT", &elements("q_init")),
            ("$Q_LOAD", &elements("q_load")),
            ("$DQ_INIT", &elements("dq_init")),
            (
                "$TILE_LOAD",
                &attention_tile_load(template, bkv * head_dim, threads),
            ),
            ("$TILE_DOT", &elements("tile_dot")),
            ("$TILE_ACCUMULATE", &elements("tile_accumulate")),
            ("$TAIL_DOT", &elements("tail_dot")),
            ("$TAIL_ACCUMULATE", &elements("tail_accumulate")),
            ("$STORE", &elements("store")),
            ("$GROUP_ELEMENTS", &(bkv * threads).to_string()),
            ("$TILE_ELEMENTS", &(bkv * head_dim).to_string()),
            ("$HEAD_DIM", &head_dim.to_string()),
            ("$THREADS", &threads.to_string()),
            ("$EPT", &ept.to_string()),
            ("$TPQ", &tpq.to_string()),
            ("$BQ", &bq.to_string()),
            ("$BKV", &bkv.to_string()),
        ],
    );
    attention_module(source, "flash_grad_q")
}

/// dK/dV batches Q/dO rows for grouped reductions; single-lane heads load directly.
pub fn generate_flash_grad_kv_module(head_dim: u32, ept_cap: u32) -> ShaderModule {
    let (ept, tpq) = attention_lanes(head_dim, ept_cap);
    let bkv = (256 / tpq).max(1);
    if bkv <= 1 {
        return ShaderModule::new(include_str!("../shaders/mha_grad_kv.wgsl"));
    }
    let template = include_str!("../shaders/flash_grad_kv.wgsl");
    let fragment = |name| template_section(template, name);
    let elements = |name| unroll_elements(fragment(name), ept, 1);
    let query_tile = backward_attention_tile(head_dim, bkv * tpq);
    let source = preprocess(
        fragment("main"),
        &[
            ("$PARAMS", ATTENTION_PARAMS_WGSL),
            (
                "$QUERY_LOOP",
                fragment(if tpq > 1 {
                    "tiled_query_loop"
                } else {
                    "direct_query_loop"
                }),
            ),
            ("$TILE_Q_LOAD", &elements("tile_q_load")),
            ("$SHARED", if tpq > 1 { fragment("shared") } else { "" }),
            (
                "$REDUCTION",
                if tpq > 1 { fragment("reduction") } else { "" },
            ),
            (
                "$REDUCE_STEP",
                &unroll_reduction(fragment("reduce_step"), tpq),
            ),
            ("$KV_INIT", &elements("kv_init")),
            ("$KV_LOAD", &elements("kv_load")),
            ("$GRAD_INIT", &elements("grad_init")),
            ("$Q_LOAD", &elements("q_load_direct")),
            ("$DOT", &elements("dot")),
            ("$SCORE", fragment("direct_score")),
            ("$ACCUMULATE", &elements("accumulate")),
            ("$STORE", &elements("store")),
            ("$QUERY_TILE", &query_tile.to_string()),
            ("$QUERY_ELEMENTS", &(query_tile * head_dim).to_string()),
            ("$GROUP_ELEMENTS", &(query_tile * bkv * tpq).to_string()),
            ("$HEAD_DIM", &head_dim.to_string()),
            ("$THREADS", &(bkv * tpq).to_string()),
            ("$EPT", &ept.to_string()),
            ("$TPQ", &tpq.to_string()),
            ("$BKV", &bkv.to_string()),
        ],
    );
    attention_module(source, "flash_grad_kv")
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
