//! Attention WGSL kernel generators.

use super::{
    ATTENTION_PARAMS_WGSL, CACHED_ATTENTION_PARAMS_WGSL, FlashAttentionShape, ShaderGroup,
    ShaderModule, preprocess, template_section,
};

/// Workgroup storage for the fixed 16-query, 16-key cooperative tiles.
/// The fixed terms include score tiles and backward row statistics.
pub(crate) fn attention_coop_shared_bytes(group: ShaderGroup, head_dim: u32) -> u64 {
    let (per_dim, fixed) = match group {
        ShaderGroup::FlashAttentionCoop => (128u64, 1024),
        ShaderGroup::FlashAttentionCoopF32 => (192, 1024),
        ShaderGroup::FlashAttentionProducts => {
            return 160 * u64::from(head_dim).next_multiple_of(16) + 1536;
        }
        ShaderGroup::AttentionGrad(kernel) => return kernel.shared_bytes(head_dim),
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
        false,
        true,
    )
}

pub fn generate_flash_attention_module(
    head_dim: u32,
    ept_cap: u32,
    shape: FlashAttentionShape,
) -> ShaderModule {
    generate_flash_attention(head_dim, ept_cap, shape, false, false, false)
}

/// `f16_keys` stores Q/K as F16. Dot products widen operands before F32
/// multiplication and accumulation; softmax and output accumulation stay F32.
/// V remains F32 so a peaked softmax preserves its value.
pub fn generate_flash_attention_module_for_device(
    head_dim: u32,
    ept_cap: u32,
    shape: FlashAttentionShape,
    f16_keys: bool,
    subgroup_tiles: bool,
) -> ShaderModule {
    generate_flash_attention(head_dim, ept_cap, shape, false, f16_keys, subgroup_tiles)
}

fn attention_module(source: String, hint: &'static str) -> ShaderModule {
    ShaderModule::with_hint(source, hint)
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

/// How a scalar flash kernel stages its operands: vec4 loads when every lane
/// owns whole vectors, and F16 tiles (Q/K forward, Q/K/V backward) when
/// requested. Dot products widen F16 operands before F32 arithmetic.
#[derive(Clone, Copy)]
struct Staging {
    vector: bool,
    f16: bool,
}

impl Staging {
    fn new(vector: bool, f16: bool) -> Self {
        Self {
            vector,
            f16: f16 && vector,
        }
    }

    /// Bound F32 values and F32 registers.
    fn elem(self) -> &'static str {
        if self.vector { "vec4<f32>" } else { "f32" }
    }

    /// Staged tiles and the registers loaded from them.
    fn tile(self) -> &'static str {
        if self.f16 { "vec4<f16>" } else { self.elem() }
    }

    fn enable(self) -> &'static str {
        if self.f16 { "enable f16;\n" } else { "" }
    }

    /// Words of a tile of `elements` values.
    fn words(self, elements: u32) -> String {
        if self.vector { elements / 4 } else { elements }.to_string()
    }

    /// The `name` section of `template` for this staging.
    fn section<'a>(self, template: &'a str, name: &str) -> &'a str {
        if self.vector {
            template_section(template, &format!("{name}_v4"))
        } else {
            template_section(template, name)
        }
    }
}

/// The barrier-free sum across one query's `tpq` lanes, and the argument it
/// reads the live subgroup width from.
fn lane_sum(tpq: u32) -> String {
    let template = include_str!("../shaders/attention_lanes.wgsl");
    preprocess(
        template_section(template, "lane_sum"),
        &[(
            "$LANE_STEPS",
            &unroll_reduction(template_section(template, "lane_step"), tpq),
        )],
    )
}

const SUBGROUP_ARG: &str = ", @builtin(subgroup_size) subgroup_size: u32";

fn generate_flash_attention(
    head_dim: u32,
    ept_cap: u32,
    shape: FlashAttentionShape,
    cached: bool,
    f16_keys: bool,
    subgroup_tiles: bool,
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
    let vector = (f16_keys || subgroup_tiles) && ept.is_multiple_of(4);
    let staging = Staging::new(vector, f16_keys);
    let template = include_str!("../shaders/flash_attention.wgsl");
    let fragment = |name| template_section(template, name);
    // Contiguous lanes own `ept` elements. Interleaved lanes still own that
    // many, but in groups of four so a vec4 load stays coalesced across the
    // query's lanes. Scalar interleave keeps the one-element stride.
    let (base_stride, elem_stride) = if shape.interleave && vector {
        (4, tpq * 4)
    } else if shape.interleave {
        (1, tpq)
    } else if vector {
        // One vec4 per step. `ept` is the lane's first element, not the step.
        (ept, 4)
    } else {
        (ept, 1)
    };
    let elements = |name: &str| {
        let (count, stride) = if vector {
            (ept / 4, elem_stride / 4)
        } else {
            (ept, elem_stride)
        };
        unroll_elements(staging.section(template, name), count, stride)
    };
    let tile_elements = shape.keys * head_dim;
    let tile_load = if vector {
        fragment("tile_load_v4").to_owned()
    } else {
        attention_tile_load(template, tile_elements, threads)
    };
    // One vec4 per thread keeps the next tile in registers. Wider tiles keep
    // the direct shared load; a private array there spills.
    let prefetch = vector && tile_elements / 4 <= threads;
    let source = preprocess(
        fragment("main"),
        &[
            (
                "$KV_TILES",
                fragment(if prefetch {
                    "kv_tiles_prefetched"
                } else {
                    "kv_tiles"
                }),
            ),
            (
                "$TILE_BODY",
                fragment(if subgroup_tiles {
                    "tile_subgroup"
                } else {
                    "tile_tree"
                }),
            ),
            ("$TILE_TREE", fragment("tile_tree")),
            (
                "$TAIL_BODY",
                fragment(if subgroup_tiles {
                    "tail_subgroup"
                } else {
                    "tail_tree"
                }),
            ),
            ("$TILE_LOAD", &tile_load),
            ("$TAIL_LOAD", staging.section(template, "tail_load")),
            ("$ENABLE", staging.enable()),
            (
                "$PARAMS",
                if cached {
                    CACHED_ATTENTION_PARAMS_WGSL
                } else {
                    ATTENTION_PARAMS_WGSL
                },
            ),
            (
                "$LANE_SUM",
                &if subgroup_tiles {
                    lane_sum(tpq)
                } else {
                    String::new()
                },
            ),
            (
                "$SUBGROUP_ARG",
                if subgroup_tiles { SUBGROUP_ARG } else { "" },
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
                "$CAUSAL_ROWS",
                if cached { "0u" } else { "params.causal_rows" },
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
            ("$ELEM", staging.elem()),
            ("$STILE", staging.tile()),
            ("$SHARED_LEN", &staging.words(tile_elements)),
            ("$TILE_ELEMENTS", &tile_elements.to_string()),
            ("$SCORES", &(threads * shape.keys).to_string()),
            ("$D_BASE_STRIDE", &base_stride.to_string()),
            ("$HEAD_DIM", &head_dim.to_string()),
            ("$THREADS", &threads.to_string()),
            ("$TPQ", &tpq.to_string()),
            ("$BQ", &bq.to_string()),
            ("$BKV", &shape.keys.to_string()),
        ],
    );
    let module = attention_module(
        source,
        if cached {
            "cached_query_attention"
        } else {
            "flash_attention"
        },
    );
    if subgroup_tiles {
        module.with_subgroup(0, [threads, 1, 1])
    } else {
        module
    }
}

/// Inference F16 QK/PV products with F32 online softmax and accumulation.
/// The fixed query block is 16 rows; `rows` selects an advertised 8x16 or
/// 16x16 native product. Padding keeps head/key/query tails inside scratch.
pub fn generate_flash_attention_product_module(head_dim: u32, rows: u32) -> ShaderModule {
    assert!(head_dim > 0 && head_dim.is_multiple_of(4) && matches!(rows, 8 | 16));
    let padded = head_dim.next_multiple_of(16);
    let template = include_str!("../shaders/flash_attention_products.wgsl");
    attention_module(
        preprocess(
            template,
            &[
                ("$PARAMS", ATTENTION_PARAMS_WGSL),
                ("$PV_TILES_", &((16 / rows) * (padded / 16)).to_string()),
                ("$HEAD_TILES_", &(padded / 16).to_string()),
                ("$ELEMENTS", &(16 * padded).to_string()),
                ("$PADDED", &padded.to_string()),
                ("$CHUNK", &head_dim.div_ceil(4).to_string()),
                ("$ROWS", &rows.to_string()),
            ],
        ),
        "flash_attention_products",
    )
    .with_subgroup(0, [64, 1, 1])
}

/// Cooperative forward attention: QKᵀ uses f16 matrix operands, while softmax
/// and PV accumulate in f32 registers. Dispatch `[ceil(q_seq/16), num_heads, 1]`.
pub fn generate_flash_attention_coop_module(head_dim: u32) -> ShaderModule {
    generate_flash_attention_coop(head_dim, true)
}

pub fn generate_flash_attention_coop_f32_module(head_dim: u32) -> ShaderModule {
    generate_flash_attention_coop(head_dim, false)
}

fn generate_flash_attention_coop(head_dim: u32, use_f16: bool) -> ShaderModule {
    let source = preprocess(
        include_str!("../shaders/flash_attention_coop.wgsl"),
        &[
            ("$ENABLE_F16", if use_f16 { "enable f16;" } else { "" }),
            ("$INPUT_TYPE", if use_f16 { "f16" } else { "f32" }),
        ],
    );
    generate_coop_attention(&source, head_dim, "flash_attention_coop", 64)
}

/// Scaled F16 cooperative dQ, with 32 query rows per workgroup.
pub fn generate_flash_grad_q_coop_f16_module(head_dim: u32) -> ShaderModule {
    generate_flash_grad_f16_module(crate::kernels::attention_grad::Part::Q, head_dim, 16)
}

/// Scaled F16 cooperative dK/dV, with 32 key rows per workgroup.
pub fn generate_flash_grad_kv_coop_f16_module(head_dim: u32) -> ShaderModule {
    generate_flash_grad_f16_module(crate::kernels::attention_grad::Part::KV, head_dim, 16)
}

/// F32 cooperative dK/dV, with 16-key tiles and full-precision matrix products.
/// The initial specialization covers the 64-wide transformer heads.
pub fn generate_flash_grad_kv_coop_f32_module(head_dim: u32) -> ShaderModule {
    generate_flash_grad_kv_coop_f32_module_with_tile(head_dim, 8)
}

pub fn generate_flash_grad_kv_coop_f32_module_with_tile(head_dim: u32, tile: u32) -> ShaderModule {
    assert_eq!(head_dim, 64);
    assert!(matches!(tile, 8 | 16));
    let mut init = String::new();
    let mut accumulate = String::new();
    let mut store = String::new();
    for column in 0..32 / tile {
        init.push_str(&format!(
            "var dk{column} = coop_mat{tile}x{tile}<f32,C>();\n\
             var dv{column} = coop_mat{tile}x{tile}<f32,C>();\n"
        ));
        accumulate.push_str(&format!(
            "let b{column} = q * 64u + out_col + {}u;\n\
             let q{column} = coopLoadT<coop_mat{tile}x{tile}<f32,B>>(&shared_q[b{column}], 64u);\n\
             let do{column} = coopLoadT<coop_mat{tile}x{tile}<f32,B>>(&shared_do[b{column}], 64u);\n\
             dk{column} = coopMultiplyAdd(ds, q{column}, dk{column});\n\
             dv{column} = coopMultiplyAdd(p, do{column}, dv{column});\n",
            column * tile
        ));
        store.push_str(&format!(
            "let c{column} = out_row * 64u + out_col + {}u;\n\
             if tile_sg < $OUTPUT_GROUPSu {{\n\
                 coopStoreT(dk{column}, &shared_k[c{column}], 64u);\n\
                 coopStoreT(dv{column}, &shared_v[c{column}], 64u);\n\
             }}\n",
            column * tile
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
                ("$OUTPUT_GROUPS", &(32 / tile).to_string()),
                ("$SCORE_GROUPS", &((16 / tile).pow(2)).to_string()),
                ("$SCORE_COLS", &(16 / tile).to_string()),
                ("$TILE", &tile.to_string()),
            ],
        ),
        "flash_grad_kv_coop_f32",
    )
    .with_subgroup(0, [128, 1, 1])
}

pub(crate) const FLASH_GRAD_COOP_F32_SHARED_BYTES: u32 = (4 * 1024 + 2 * 256 + 3 * 16) * 4;

/// F32 cooperative dQ, dispatched as `[ceil(q_seq/16), num_heads, 1]`.
pub fn generate_flash_grad_q_coop_f32_module(head_dim: u32) -> ShaderModule {
    generate_flash_grad_q_coop_f32_module_with_tile(head_dim, 8)
}

pub fn generate_flash_grad_q_coop_f32_module_with_tile(head_dim: u32, tile: u32) -> ShaderModule {
    assert_eq!(head_dim, 64);
    assert!(matches!(tile, 8 | 16));
    let mut init = String::new();
    let mut accumulate = String::new();
    let mut store = String::new();
    for column in 0..32 / tile {
        init.push_str(&format!(
            "var dq{column} = coop_mat{tile}x{tile}<f32,C>();\n"
        ));
        accumulate.push_str(&format!(
            "let b{column} = k * 64u + out_col + {}u;\n\
             let k{column} = coopLoadT<coop_mat{tile}x{tile}<f32,B>>(&shared_k[b{column}], 64u);\n\
             dq{column} = coopMultiplyAdd(ds, k{column}, dq{column});\n",
            column * tile
        ));
        store.push_str(&format!(
            "let c{column} = out_row * 64u + out_col + {}u;\n\
             if tile_sg < $OUTPUT_GROUPSu {{ coopStoreT(dq{column}, &shared_q[c{column}], 64u); }}\n",
            column * tile
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
                ("$OUTPUT_GROUPS", &(32 / tile).to_string()),
                ("$SCORE_GROUPS", &((16 / tile).pow(2)).to_string()),
                ("$SCORE_COLS", &(16 / tile).to_string()),
                ("$TILE", &tile.to_string()),
            ],
        ),
        "flash_grad_q_coop_f32",
    )
    .with_subgroup(0, [128, 1, 1])
}

/// Binary-scaled F16 gradient products, with F32 accumulation.
/// Dispatch 32 query rows (dQ) or key rows (dK/dV) per workgroup.
pub fn generate_flash_grad_f16_module(
    part: crate::kernels::attention_grad::Part,
    head_dim: u32,
    native_rows: u32,
) -> ShaderModule {
    use crate::kernels::attention_grad::Part;
    assert!(matches!(native_rows, 8 | 16));
    let template = match part {
        Part::Q => include_str!("../shaders/flash_grad_q_coop_f16.wgsl"),
        Part::KV => include_str!("../shaders/flash_grad_kv_coop_f16.wgsl"),
    };
    let source = preprocess(
        template,
        &[
            ("$M", &native_rows.to_string()),
            ("$SCALE", include_str!("../shaders/attention_scale.wgsl")),
            ("$SCALE_GROUPS", "32"),
        ],
    );
    generate_coop_attention(&source, head_dim, "flash_grad_scaled_f16", 128)
}

fn generate_coop_attention(
    template: &str,
    head_dim: u32,
    hint: &'static str,
    threads: u32,
) -> ShaderModule {
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
    attention_module(source, hint).with_subgroup(0, [threads, 1, 1])
}

fn backward_attention_tile(head_dim: u32, threads: u32) -> u32 {
    // Budget 16 KiB for two input tiles and two partial-dot arrays.
    (2048 / (head_dim + threads)).clamp(1, 4)
}

/// Inner rows once the partial-dot arrays are gone. Two f32 tiles cost
/// `8 * tile * head_dim` bytes. 16 rows is the fastest cap that still
/// fits the 16 KiB budget on the head widths this kernel serves.
fn backward_attention_tile_subgroup(head_dim: u32) -> u32 {
    (2048 / head_dim).clamp(1, 16)
}

/// The device's reported subgroup width owns every lane of a query. The
/// generated kernel still checks the live width and falls back to a shared
/// sum, because the report is only a default.
fn subgroup_covers_lanes(subgroup_size: u32, tpq: u32) -> bool {
    tpq > 1 && subgroup_size >= tpq && subgroup_size.is_multiple_of(tpq)
}

const BACKWARD_SHARED_BUDGET: u32 = 16 * 1024;

/// Largest row tile whose staging bytes, plus `reserved` fallback storage,
/// stay inside the scalar backward shared-memory budget.
fn fit_shared_rows(want: u32, bytes_per_row: u32, reserved: u32) -> u32 {
    if bytes_per_row == 0 {
        return want.max(1);
    }
    let room = BACKWARD_SHARED_BUDGET.saturating_sub(reserved);
    want.clamp(1, (room / bytes_per_row).max(1))
}

/// dQ stages K/V tiles; multi-lane queries also reduce partial dot products.
pub fn generate_flash_grad_q_module(head_dim: u32, ept_cap: u32) -> ShaderModule {
    let part = crate::kernels::attention_grad::Part::Q;
    generate_flash_grad_module(part, head_dim, ept_cap, 0, 256, false)
}

/// dK/dV batches Q/dO rows for grouped reductions; single-lane heads load directly.
pub fn generate_flash_grad_kv_module(head_dim: u32, ept_cap: u32) -> ShaderModule {
    let part = crate::kernels::attention_grad::Part::KV;
    generate_flash_grad_module(part, head_dim, ept_cap, 0, 256, false)
}

/// A scalar flash attention derivative. A workgroup owns a block of rows,
/// queries for dQ and keys for dK/dV, and stages the other side's rows in
/// tiles: K/V for dQ, Q/dO for dK/dV. Several lanes per row sum their partial
/// dots with subgroup shuffles when the device's default `subgroup_size`
/// covers them, checking the live width, and with a shared tree otherwise.
/// Zero `subgroup_size` keeps the tree. `threads` is the flash workgroup
/// width, 128 or 256.
pub fn generate_flash_grad_module(
    part: crate::kernels::attention_grad::Part,
    head_dim: u32,
    ept_cap: u32,
    subgroup_size: u32,
    threads: u32,
    f16_tiles: bool,
) -> ShaderModule {
    use crate::kernels::attention_grad::Part;
    let (ept, tpq) = attention_lanes(head_dim, ept_cap);
    let rows = (threads / tpq).max(1);
    if rows <= 1 {
        return ShaderModule::new(match part {
            Part::Q => include_str!("../shaders/mha_grad_q.wgsl"),
            Part::KV => include_str!("../shaders/mha_grad_kv.wgsl"),
        });
    }
    let threads = rows * tpq;
    let vector = ept.is_multiple_of(4);
    let staging = Staging::new(vector, f16_tiles);
    let subgroup = subgroup_covers_lanes(subgroup_size, tpq);
    // Without the partial-dot arrays, subgroup kernels stage more rows: dQ
    // two tiles, dK/dV Q in the tile type and dO in F32, beside a compact
    // fallback of two F32 values per thread.
    let staged = if subgroup {
        let element = if staging.f16 { 2 } else { 4 };
        let bytes_per_row = match part {
            Part::Q => 2 * element,
            Part::KV => element + 4,
        } * head_dim;
        fit_shared_rows(
            backward_attention_tile_subgroup(head_dim),
            bytes_per_row,
            8 * threads,
        )
    } else if part == Part::Q && tpq == 1 {
        8
    } else {
        backward_attention_tile(head_dim, threads)
    };
    let staged_elements = staged * head_dim;
    let template = match part {
        Part::Q => include_str!("../shaders/flash_grad_q.wgsl"),
        Part::KV => include_str!("../shaders/flash_grad_kv.wgsl"),
    };
    let fragment = |name| template_section(template, name);
    let math = |name: &str| {
        unroll_elements(
            staging.section(template, name),
            if vector { ept / 4 } else { ept },
            1,
        )
    };
    let lanes = include_str!("../shaders/attention_lanes.wgsl");
    let mut vars: Vec<(&str, String)> = vec![
        ("$ENABLE", staging.enable().to_owned()),
        ("$PARAMS", ATTENTION_PARAMS_WGSL.to_owned()),
        (
            "$LANE_SUM",
            if subgroup {
                lane_sum(tpq)
            } else {
                String::new()
            },
        ),
        (
            "$SUBGROUP_ARG",
            if subgroup { SUBGROUP_ARG } else { "" }.to_owned(),
        ),
        (
            "$REDUCTION",
            if subgroup {
                template_section(lanes, "partial_sums")
            } else if tpq > 1 {
                fragment("reduction")
            } else {
                ""
            }
            .to_owned(),
        ),
        (
            "$REDUCE_STEP",
            unroll_reduction(fragment("reduce_step"), tpq),
        ),
    ];
    // Loops first: they contain the fragments substituted after them.
    match part {
        Part::Q => vars.extend([
            (
                "$KV_LOOP",
                fragment(if tpq == 1 {
                    "tiled_kv_loop"
                } else if subgroup {
                    "grouped_kv_loop_subgroup"
                } else {
                    "grouped_kv_loop"
                })
                .to_owned(),
            ),
            (
                "$TILE_LOAD",
                if vector {
                    fragment("tile_load_v4").to_owned()
                } else {
                    attention_tile_load(template, staged_elements, threads)
                },
            ),
            (
                "$GROUP_LOAD",
                staging.section(template, "group_load").to_owned(),
            ),
            (
                "$TILED_TAIL_LOAD",
                staging.section(template, "tail_load").to_owned(),
            ),
            ("$Q_INIT", math("q_init")),
            ("$Q_LOAD", math("q_load")),
            ("$DQ_INIT", math("dq_init")),
            ("$TILE_DOT", math("tile_dot")),
            ("$TILE_ACCUMULATE", math("tile_accumulate")),
            ("$TAIL_DOT", math("tail_dot")),
            ("$TAIL_ACCUMULATE", math("tail_accumulate")),
            ("$STORE", math("store")),
            ("$SHARED_LEN", staging.words(staged_elements)),
            ("$TILE_ELEMENTS", staged_elements.to_string()),
            ("$BQ", rows.to_string()),
            ("$BKV", staged.to_string()),
        ]),
        Part::KV => vars.extend([
            (
                "$QUERY_LOOP",
                fragment(if subgroup {
                    "tiled_query_loop_subgroup"
                } else if tpq > 1 {
                    "tiled_query_loop"
                } else {
                    "direct_query_loop"
                })
                .to_owned(),
            ),
            (
                "$QUERY_SHARED",
                if tpq > 1 { fragment("shared") } else { "" }.to_owned(),
            ),
            (
                "$QUERY_LOAD",
                staging.section(template, "query_load").to_owned(),
            ),
            ("$TILE_Q_LOAD", math("tile_q_load")),
            ("$KV_INIT", math("kv_init")),
            ("$KV_LOAD", math("kv_load")),
            ("$GRAD_INIT", math("grad_init")),
            ("$Q_LOAD", math("q_load_direct")),
            ("$DOT", math("dot")),
            ("$SCORE", fragment("direct_score").to_owned()),
            ("$ACCUMULATE", math("accumulate")),
            ("$STORE", math("store")),
            ("$QUERY_LEN", staging.words(staged_elements)),
            ("$QUERY_TILE", staged.to_string()),
            ("$QUERY_ELEMENTS", staged_elements.to_string()),
            ("$BKV", rows.to_string()),
        ]),
    }
    vars.extend([
        ("$ELEM", staging.elem().to_owned()),
        ("$STILE", staging.tile().to_owned()),
        ("$GROUP_ELEMENTS", (staged * threads).to_string()),
        ("$HEAD_DIM", head_dim.to_string()),
        ("$THREADS", threads.to_string()),
        ("$EPT", ept.to_string()),
        ("$TPQ", tpq.to_string()),
    ]);
    let vars: Vec<_> = vars
        .iter()
        .map(|&(key, ref value)| (key, value.as_str()))
        .collect();
    let hint = match part {
        Part::Q => "flash_grad_q",
        Part::KV => "flash_grad_kv",
    };
    let module = attention_module(preprocess(fragment("main"), &vars), hint);
    if subgroup {
        module.with_subgroup(0, [threads, 1, 1])
    } else {
        module
    }
}

pub(crate) fn block_attention(
    entry: &crate::compile::ShaderEntry,
    p: crate::compile::AttentionParams,
) -> ShaderModule {
    use crate::{compile::ShaderEntry, kernels::attention_grad::Part};
    let hd = p.head_dim;
    let cols = format!("{}u", hd / 4);
    match *entry {
        ShaderEntry::FlashAttentionCoop | ShaderEntry::FlashAttentionProducts => {
            let products = *entry == ShaderEntry::FlashAttentionProducts;
            let source = include_str!("../shaders/flash_attention_block.wgsl");
            // Cooperative matrix values must be named individually in WGSL.
            let fragments = |name| unroll_elements(template_section(source, name), hd / 16, 4);
            let scalar = |fragment| if products { "" } else { fragment };
            attention_module(
                preprocess(
                    template_section(source, "main"),
                    &[
                        ("$COLS", &cols),
                        (
                            "$PV",
                            template_section(
                                source,
                                if products { "matrix_pv" } else { "scalar_pv" },
                            ),
                        ),
                        (
                            "$VALUE_STORAGE",
                            if products {
                                ""
                            } else {
                                "var<workgroup> values: array<vec4<f32>, 32u * (HD4 + 1u)>;"
                            },
                        ),
                        (
                            "$STORE_VALUE",
                            if products {
                                "qk[(32u + r) * QK_STRIDE + col] = vec4<f16>(bias[source]);"
                            } else {
                                "values[r * (HD4 + 1u) + col] = bias[source];"
                            },
                        ),
                        (
                            "$SCALAR_OUTPUT",
                            scalar("var output: array<vec4<f32>, HD4 / 2u>;"),
                        ),
                        (
                            "$SCALAR_STORE",
                            scalar(template_section(source, "scalar_store")),
                        ),
                        (
                            "$PROB_STORAGE",
                            if products {
                                "var<workgroup> probs: array<vec4<f16>, 512>;\nvar<workgroup> flags: array<u32, 4>;"
                            } else {
                                ""
                            },
                        ),
                        (
                            "$FINISH_PV",
                            &if products {
                                fragments("finish_pv")
                            } else {
                                String::new()
                            },
                        ),
                        (
                            "$ACCUM_INIT",
                            &if products {
                                fragments("accum_init")
                            } else {
                                String::new()
                            },
                        ),
                        ("$RESCALE_PV", &fragments("rescale_pv")),
                        ("$ACCUM_PV", &fragments("accum_pv")),
                        ("$SLACK", if products { "8.0" } else { "0.0" }),
                        ("$LOAD_Q", &fragments("load_q")),
                        ("$QK", &fragments("qk")),
                        ("$HEADS", &format!("{}u", p.packed_heads >> 16)),
                        ("$KV_HEADS", &format!("{}u", p.packed_heads & 65535)),
                        ("$Q_SEQ", &format!("{}u", p.q_seq)),
                        ("$KV_SEQ", &format!("{}u", p.kv_seq)),
                        ("$CAUSAL_ROWS", &format!("{}u", p.causal_rows)),
                        ("$WINDOW", &format!("{}u", p.window_size)),
                    ],
                ),
                "flash_attention_block",
            ).with_subgroup(32, [128, 1, 1])
        }
        ShaderEntry::AttentionGrad(k)
            if matches!(k.path, crate::kernels::attention_grad::Path::Cooperative(_)) =>
        {
            super::attention_block_grad::coop_grad(k.part == Part::KV, p)
                .with_subgroup(32, [128, 1, 1])
        }
        ShaderEntry::AttentionGrad(k) => {
            let kv = k.part == Part::KV;
            let kv_only = |fragment| if kv { fragment } else { "" };
            attention_module(
                preprocess(
                    include_str!("../shaders/flash_grad_block.wgsl"),
                    &[
                        ("$PARAMS", ATTENTION_PARAMS_WGSL),
                        ("$COLS", &cols),
                        ("$KV", if kv { "true" } else { "false" }),
                        ("$STATS_ROWS", if kv { "COLS" } else { "ROWS" }),
                        (
                            "$SECOND_OUTPUT",
                            kv_only("var<storage, read_write> dst2: array<vec4<f32>>;"),
                        ),
                        (
                            "$PROB_STORAGE",
                            kv_only("var<workgroup> p_tile: array<vec4<f32>, THREADS>;"),
                        ),
                        (
                            "$STORE_PROB",
                            kv_only("p_tile[col * (ROWS / 4u) + row4] = prob;"),
                        ),
                        (
                            "$ACCUM_V",
                            kv_only("grad_v[j] = fma(vec4<f32>(p_tile[at][j]), b, grad_v[j]);"),
                        ),
                        ("$STORE_V", kv_only("dst2[out] = grad_v[j];")),
                    ],
                ),
                "flash_grad_block",
            )
        }
        _ => unreachable!("block attention requires an admitted entry"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod product_tests {
    use super::*;

    #[test]
    fn register_block_attention_shaders_validate() {
        use crate::{
            compile::ShaderEntry,
            kernels::attention_grad::{AttentionGrad, Operands, Part, Path},
        };
        for entry in [
            ShaderEntry::FlashAttentionCoop,
            ShaderEntry::FlashAttentionProducts,
            ShaderEntry::AttentionGrad(AttentionGrad::new(Part::Q, Path::Flash)),
            ShaderEntry::AttentionGrad(AttentionGrad::new(Part::KV, Path::Flash)),
            ShaderEntry::AttentionGrad(AttentionGrad::new(
                Part::Q,
                Path::Cooperative(Operands::F16),
            )),
            ShaderEntry::AttentionGrad(AttentionGrad::new(
                Part::KV,
                Path::Cooperative(Operands::F16),
            )),
        ] {
            for hd in [16, 32, 64, 128] {
                let mut shader = block_attention(
                    &entry,
                    crate::compile::AttentionParams {
                        q_seq: 133,
                        kv_seq: 137,
                        packed_heads: (2 << 16) | 1,
                        head_dim: hd,
                        window_size: 0,
                        causal_rows: 0,
                        _pad1: 0,
                        _pad2: 0,
                    },
                );
                let scalar =
                    matches!(entry, ShaderEntry::AttentionGrad(k) if k.path == Path::Flash);
                assert_eq!(
                    shader.subgroup.map(|s| (s.size, s.workgroup_size)),
                    (!scalar).then_some((32, [128, 1, 1]))
                );
                crate::codegen::validate_shader_for_test(
                    &mut shader.module,
                    naga::valid::Capabilities::all(),
                    (1, 3),
                )
                .unwrap();
            }
        }
    }

    #[test]
    fn products_validate_with_rectangular_tiles_and_padded_heads() {
        let caps = naga::valid::Capabilities::SHADER_FLOAT16
            | naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::SUBGROUP;
        for rows in [8, 16] {
            for head_dim in [4, 20, 64] {
                let mut module = generate_flash_attention_product_module(head_dim, rows);
                let mut binding = 0;
                for (_, global) in module.module.global_variables.iter_mut() {
                    if matches!(
                        global.space,
                        naga::AddressSpace::Uniform | naga::AddressSpace::Storage { .. }
                    ) {
                        global.binding = Some(naga::ResourceBinding { group: 0, binding });
                        binding += 1;
                    }
                }
                let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps)
                    .validate(&module.module)
                    .unwrap_or_else(|e| panic!("rows={rows} head={head_dim}: {e:?}"));
                naga::back::spv::write_vec(
                    &module.module,
                    &info,
                    &naga::back::spv::Options::default(),
                    None,
                )
                .unwrap();
            }
        }
    }
}
