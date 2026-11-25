//! Scaled F16 derivatives, with two 16-row tiles in registers per subgroup.
use super::{ShaderModule, preprocess, template_section};
use std::fmt::Write as _;

const SUBGROUPS: u32 = 4;
const ROW_TILES: u32 = 2;
const KEY_TILE: u32 = 32;
pub(crate) const BLOCK_ROWS: u32 = SUBGROUPS * ROW_TILES * 16;

/// `&array[offset / 4], stride / 4` for a cooperative access to a vec4 array.
fn at(array: &str, offset: &str, stride: u32) -> String {
    debug_assert!(stride.is_multiple_of(4));
    format!("&{array}[({offset}) / 4u], {}u", stride / 4)
}

pub(crate) struct CoopGradLayout {
    operand_stride: u32,
    prob_stride: u32,
    score_stride: u32,
    output_stride: u32,
    stage: u32,
    prob_per_subgroup: u32,
    halves: u32,
    scores_per_subgroup: u32,
    floats: u32,
    anchor_len: u32,
}

impl CoopGradLayout {
    pub(crate) fn new(kv: bool, head_dim: u32) -> Self {
        let operand_stride = head_dim + 8;
        let prob_stride = KEY_TILE + 8;
        let score_stride = KEY_TILE + 4;
        let output_stride = 20;
        // The fixed side's two blocks stage one after the other.
        let fixed = BLOCK_ROWS * operand_stride;
        let walked = 2 * KEY_TILE * operand_stride;
        let stage = fixed.max(walked);
        let prob_per_subgroup = (if kv { 2 } else { 1 }) * 16 * prob_stride;
        let halves = stage + SUBGROUPS * prob_per_subgroup;
        let scores_per_subgroup = (2 * 16 * score_stride).max(16 * output_stride);
        // dK/dV stages per-step partial anchor dots in `fs` before S/dP tiles.
        let anchor_dots = if kv { KEY_TILE * (head_dim / 4) } else { 0 };
        let floats = (SUBGROUPS * scores_per_subgroup).max(anchor_dots);
        Self {
            operand_stride,
            prob_stride,
            score_stride,
            output_stride,
            stage,
            prob_per_subgroup,
            halves,
            scores_per_subgroup,
            floats,
            anchor_len: head_dim,
        }
    }

    pub(crate) fn shared_bytes(&self) -> u32 {
        2 * self.halves
            + 4 * self.floats
            + 4 * (3 * KEY_TILE + 4)
            + 4 * self.anchor_len
            + SUBGROUPS * 8
    }
}

const SCALING_WGSL: &str = include_str!("../shaders/attention_scale.wgsl");

/// Cooperative-matrix f16 backward with f32 accumulation and statistics.
///
/// - dQ: a workgroup owns 128 queries of one head and
///   walks 32 keys at a time. Q (scaled) and dO (binary-scaled per
///   workgroup) stay in operand matrices. K is centered on the first key:
///   each dS row sums to zero, so dQ = Σ dS (K − anchor), and the shared
///   component no longer amplifies dS rounding.
/// - dK/dV: a workgroup owns 128 keys of one KV head
///   and walks 32 queries at a time. K (scaled) and V
///   stay in operand matrices; Q and dO (binary-scaled per step) stage.
///
/// dS is binary-scaled per subgroup and step; each partial product is
/// unscaled before it joins the f32 accumulators.
pub(crate) fn coop_grad(kv: bool, p: crate::compile::AttentionParams) -> ShaderModule {
    let (query_count, encoded_keys, window, head_dim) =
        (p.q_seq, p.kv_seq, p.window_size, p.head_dim);
    let (heads, kv_heads) = (p.packed_heads >> 16, p.packed_heads & 65535);
    let query_group = heads / kv_heads.max(1);
    let causal = encoded_keys == 0;
    let key_count = if causal { query_count } else { encoded_keys };
    let l = CoopGradLayout::new(kv, head_dim);
    let head_vecs = head_dim / 4;
    let operand_vec_stride = l.operand_stride / 4;
    let head_tiles = head_dim / 16;
    let key_tiles = KEY_TILE / 16;
    let row_tiles = ROW_TILES;
    let threads = SUBGROUPS * 32;
    let rows = BLOCK_ROWS;
    let key_vecs_per_lane = KEY_TILE / 8;
    let mut s = String::new();
    let w = &mut s;
    let template = preprocess(
        include_str!("../shaders/flash_grad_block_coop.wgsl"),
        &[
            ("$SCALING$", SCALING_WGSL),
            ("$SCALE_GROUPS", &SUBGROUPS.to_string()),
            ("$KEY_VECS$", &(KEY_TILE / 4).to_string()),
            ("$GRAD_STATS_OFFSET$", &(KEY_TILE / 2).to_string()),
            ("$CAUSAL$", &causal.to_string()),
            ("$HEADS$", &heads.to_string()),
            ("$KV_HEADS$", &kv_heads.to_string()),
            ("$KEY_COUNT$", &key_count.to_string()),
            ("$QUERY_COUNT$", &query_count.to_string()),
            (
                "$SECOND_OUTPUT_DECL$",
                if kv {
                    "\nvar<storage, read_write> dst2: array<vec4<f32>>;"
                } else {
                    ""
                },
            ),
            ("$WINDOW$", &window.to_string()),
            ("$SCORES_PER_SUBGROUP$", &l.scores_per_subgroup.to_string()),
            ("$FIXED_VECS$", &(rows * head_vecs).to_string()),
            ("$SCORE_VECS$", &(l.floats / 4).to_string()),
            ("$OPERAND_VECS$", &(l.halves / 4).to_string()),
            ("$HEAD_DIM$", &head_dim.to_string()),
            ("$HEAD_VECS$", &head_vecs.to_string()),
            ("$OPERAND_VEC_STRIDE$", &operand_vec_stride.to_string()),
            ("$KEY_VECS_PER_LANE$", &key_vecs_per_lane.to_string()),
            ("$DOT_VECS$", &(head_vecs / 4).to_string()),
            ("$QUERY_GROUP$", &query_group.to_string()),
            ("$ROWS$", &rows.to_string()),
            ("$STAGE$", &l.stage.to_string()),
            ("$STAT_VECS$", &(3 * KEY_TILE / 4 + 1).to_string()),
            ("$SUBGROUP_ROWS$", &(row_tiles * 16).to_string()),
            ("$THREADS$", &threads.to_string()),
            ("$PROB_PER_SUBGROUP$", &l.prob_per_subgroup.to_string()),
            ("$SCORE_STRIDE$", &l.score_stride.to_string()),
            ("$DP_OFFSET$", &(16 * l.score_stride).to_string()),
            ("$WALKED_VECS$", &(KEY_TILE * head_vecs).to_string()),
        ],
    );
    let section = |name| template_section(&template, name);
    w.push_str(section("main"));
    // Load 16-row operand tiles of a staged block into named matrices.
    let load_rows = |w: &mut String, name: &str, region: u32| {
        for r in 0..row_tiles {
            for h in 0..head_tiles {
                writeln!(
                    w,
                    "    let {name}{r}_{h} = coopLoadT<coop_mat16x16<f16,A>>({});",
                    at(
                        "hs",
                        &format!(
                            "{region}u + (sg * {}u + {}u) * {}u + {}u",
                            row_tiles * 16,
                            r * 16,
                            l.operand_stride,
                            h * 16
                        ),
                        l.operand_stride
                    )
                )
                .unwrap();
            }
        }
    };
    let zero = "coop_mat16x16<f32,C>()";
    // Both directions form the scores and dP with the same two matrix products.
    let scores = |w: &mut String, r: u32, left: &str, right: &str| {
        for k in 0..key_tiles {
            writeln!(
                w,
                "        var s{r}_{k} = {zero};\n        var u{r}_{k} = {zero};"
            )
            .unwrap();
            for h in 0..head_tiles {
                let a = at(
                    "hs",
                    &format!("{}u + {}u", k * 16 * l.operand_stride, h * 16),
                    l.operand_stride,
                );
                let b = at(
                    "hs",
                    &format!(
                        "{}u + {}u",
                        KEY_TILE * l.operand_stride + k * 16 * l.operand_stride,
                        h * 16
                    ),
                    l.operand_stride,
                );
                writeln!(
                        w,
                        "        s{r}_{k} = coopMultiplyAdd({left}{r}_{h}, coopLoad<coop_mat16x16<f16,B>>({a}), s{r}_{k});
        u{r}_{k} = coopMultiplyAdd({right}{r}_{h}, coopLoad<coop_mat16x16<f16,B>>({b}), u{r}_{k});"
                    )
                    .unwrap();
            }
            writeln!(
                w,
                "        coopStoreT(s{r}_{k}, {});\n        coopStoreT(u{r}_{k}, {});",
                at("fs", &format!("fbase + {}u", k * 16), l.score_stride),
                at(
                    "fs",
                    &format!("fbase + {}u + {}u", 16 * l.score_stride, k * 16),
                    l.score_stride
                )
            )
            .unwrap();
        }
        w.push_str(section("score_rows"));
    };
    if kv {
        // Each step's partial anchor dots fill `fs` before its S tiles do.
        assert!(
            l.floats >= KEY_TILE * head_vecs && head_vecs.is_multiple_of(4),
            "attention scratch at head width {head_dim}"
        );
        w.push_str(section("stage_keys"));
        load_rows(w, "ka", 0);
        w.push_str(section("stage_values"));
        load_rows(w, "va", 0);
        for r in 0..row_tiles {
            for h in 0..head_tiles {
                writeln!(
                    w,
                    "    var dk{r}_{h} = {zero};\n    var dv{r}_{h} = {zero};"
                )
                .unwrap();
            }
            writeln!(
                w,
                "    var dsdown{r}: i32;
    var pdown{r}: i32;
    let key{r} = k0 + sg * {}u + {}u + my_row;",
                row_tiles * 16,
                r * 16
            )
            .unwrap();
        }
        // Keep Q and dO in registers while reducing their tile maxima.
        let n = KEY_TILE * head_vecs;
        let reads = n.div_ceil(threads);
        writeln!(w, "    for (var q = begin; q < end; q += {cols}u) {{\n        workgroupBarrier();\n        var local_max = vec2<f32>(0.0);", cols = KEY_TILE)
            .unwrap();
        for j in 0..reads {
            w.push_str(&preprocess(
                section("load_query_gradient"),
                &[("$J$", &j.to_string()), ("$O$", &(j * threads).to_string())],
            ));
        }
        w.push_str(section("load_query_stats"));
        for j in 0..reads {
            writeln!(
                w,
                "        if i{j} < {n}u {{
            let col = (i{j} / {head_vecs}u) * {operand_stride}u / 4u + i{j} % {head_vecs}u;
            hs[col] = vec4<f16>(scaled(qx{j}, qup));
            hs[{drow}u / 4u + col] = vec4<f16>(scaled(d{j}, dup));
        }}",
                drow = KEY_TILE * l.operand_stride,
                operand_stride = l.operand_stride
            )
            .unwrap();
        }
        w.push_str(section("center_query_stats"));
        for r in 0..row_tiles {
            scores(w, r, "ka", "va");
            writeln!(w, "            var prob_max = 0.0;").unwrap();
            for j in 0..key_vecs_per_lane {
                w.push_str(&preprocess(
                    section("kv_derivative"),
                    &[("$J$", &j.to_string()), ("$R$", &r.to_string())],
                ));
            }
            writeln!(
                w,
                "            let e_ds = exponent_for(subgroupMax(big));
            let e_p = exponent_for(subgroupMax(prob_max));
            let prow = (tbase + my_row * {prob_stride}u) / 4u + half * {key_vecs_per_lane}u;
            let drow = (tbase + {dsub}u + my_row * {prob_stride}u) / 4u + half * {key_vecs_per_lane}u;",
                prob_stride = l.prob_stride,
                dsub = 16 * l.prob_stride
            )
            .unwrap();
            for j in 0..key_vecs_per_lane {
                writeln!(w, "            hs[prow + {j}u] = vec4<f16>(scaled(p{j}, e_p));\n            hs[drow + {j}u] = vec4<f16>(scaled(ds{j}, e_ds));")
                    .unwrap();
            }
            writeln!(
                w,
                "            pdown{r} = -(e_p + walk_exp.y);
            dsdown{r} = -(e_ds + walk_exp.x);\n        }}\n        workgroupBarrier();"
            )
            .unwrap();
            for k in 0..key_tiles {
                writeln!(
                    w,
                    "        let pa{r}_{k} = coopLoadT<coop_mat16x16<f16,A>>({});
        let sa{r}_{k} = coopLoadT<coop_mat16x16<f16,A>>({});",
                    at("hs", &format!("tbase + {}u", k * 16), l.prob_stride),
                    at(
                        "hs",
                        &format!("tbase + {}u + {}u", 16 * l.prob_stride, k * 16),
                        l.prob_stride
                    )
                )
                .unwrap();
            }
            for h in 0..head_tiles {
                writeln!(
                    w,
                    "        var pv{r}_{h} = {zero};\n        var sq{r}_{h} = {zero};"
                )
                .unwrap();
                for k in 0..key_tiles {
                    let drow = at(
                        "hs",
                        &format!(
                            "{}u + {}u",
                            KEY_TILE * l.operand_stride + k * 16 * l.operand_stride,
                            h * 16
                        ),
                        l.operand_stride,
                    );
                    let qrow = at(
                        "hs",
                        &format!("{}u + {}u", k * 16 * l.operand_stride, h * 16),
                        l.operand_stride,
                    );
                    writeln!(
                        w,
                        "        pv{r}_{h} = coopMultiplyAdd(pa{r}_{k}, coopLoadT<coop_mat16x16<f16,B>>({drow}), pv{r}_{h});
        sq{r}_{h} = coopMultiplyAdd(sa{r}_{k}, coopLoadT<coop_mat16x16<f16,B>>({qrow}), sq{r}_{h});"
                    )
                    .unwrap();
                }
                writeln!(w, "        dv{r}_{h} = dv{r}_{h} + restore_product(pv{r}_{h}, pdown{r});\n        dk{r}_{h} = dk{r}_{h} + restore_product(sq{r}_{h}, dsdown{r});").unwrap();
            }
        }
    } else {
        w.push_str(section("stage_queries"));
        load_rows(w, "qa", 0);
        w.push_str(section("stage_output_gradient"));
        load_rows(w, "da", 0);
        for r in 0..row_tiles {
            w.push_str(&preprocess(
                section("query_stats"),
                &[
                    ("$R$", &r.to_string()),
                    ("$ROW_OFFSET$", &(r * 16).to_string()),
                ],
            ));
            for h in 0..head_tiles {
                writeln!(w, "    var dq{r}_{h} = {zero};").unwrap();
            }
        }
        // Each thread issues all of its K and V loads for the step before its
        // stores, so their latencies overlap.
        let n = KEY_TILE * head_vecs;
        let reads = n.div_ceil(threads);
        writeln!(
            w,
            "    for (var t = k_begin; t < end; t += {cols}u) {{\n        workgroupBarrier();\n        var local_max = vec2<f32>(0.0);",
            cols = KEY_TILE
        )
        .unwrap();
        for j in 0..reads {
            w.push_str(&preprocess(
                section("load_keys_values"),
                &[("$J$", &j.to_string()), ("$O$", &(j * threads).to_string())],
            ));
        }
        writeln!(
            w,
            "        let walk_exp = block_exponents(local_max, sg, 32u, lane);
        let kup = walk_exp.x;
        let vup = walk_exp.y;
        let score_down = -(fixed_exp.x + walk_exp.x);
        let dp_down = -(fixed_exp.y + walk_exp.y);"
        )
        .unwrap();
        for j in 0..reads {
            writeln!(
                w,
                "        if i{j} < {n}u {{
            let r = i{j} / {head_vecs}u;
            hs[r * {operand_vec_stride}u + i{j} % {head_vecs}u] = vec4<f16>(scaled(kx{j}, kup));
            hs[({vrow}u + r * {operand_stride}u) / 4u + i{j} % {head_vecs}u] = vec4<f16>(scaled(vx{j}, vup));
        }}",
                vrow = KEY_TILE * l.operand_stride,
                operand_stride = l.operand_stride
            )
            .unwrap();
        }
        writeln!(w, "        workgroupBarrier();").unwrap();
        for r in 0..row_tiles {
            scores(w, r, "qa", "da");
            for j in 0..key_vecs_per_lane {
                w.push_str(&preprocess(
                    section("query_derivative"),
                    &[("$J$", &j.to_string()), ("$R$", &r.to_string())],
                ));
            }
            writeln!(
                w,
                "            let e_ds = exponent_for(subgroupMax(big));
            let drow = (tbase + my_row * {prob_stride}u) / 4u + half * {key_vecs_per_lane}u;",
                prob_stride = l.prob_stride
            )
            .unwrap();
            for j in 0..key_vecs_per_lane {
                writeln!(
                    w,
                    "            hs[drow + {j}u] = vec4<f16>(scaled(ds{j}, e_ds));"
                )
                .unwrap();
            }
            writeln!(
                w,
                "            dsdown{r} = -(e_ds + walk_exp.x);\n        }}\n        workgroupBarrier();"
            )
            .unwrap();
            for k in 0..key_tiles {
                writeln!(
                    w,
                    "        let sa{r}_{k} = coopLoadT<coop_mat16x16<f16,A>>({});",
                    at("hs", &format!("tbase + {}u", k * 16), l.prob_stride)
                )
                .unwrap();
            }
            for h in 0..head_tiles {
                writeln!(w, "        var part{r}_{h} = {zero};").unwrap();
                for k in 0..key_tiles {
                    let krow = at(
                        "hs",
                        &format!("{}u + {}u", k * 16 * l.operand_stride, h * 16),
                        l.operand_stride,
                    );
                    writeln!(
                        w,
                        "        part{r}_{h} = coopMultiplyAdd(sa{r}_{k}, coopLoadT<coop_mat16x16<f16,B>>({krow}), part{r}_{h});"
                    )
                    .unwrap();
                }
                writeln!(
                    w,
                    "        dq{r}_{h} = dq{r}_{h} + restore_product(part{r}_{h}, dsdown{r});"
                )
                .unwrap();
            }
        }
    }
    let (row, length, heads, head, outputs): (_, _, _, _, &[(&str, &str, &str)]) = if kv {
        (
            "key",
            "kv_seq",
            "kv_heads",
            "kv_head",
            &[("dk", "dst", "scale"), ("dv", "dst2", "1.0")],
        )
    } else {
        ("pos", "q_seq", "heads", "head", &[("dq", "dst", "scale")])
    };
    writeln!(w, "    }}\n    workgroupBarrier();").unwrap();
    for r in 0..row_tiles {
        for &(acc, out, factor) in outputs {
            for h in 0..head_tiles {
                writeln!(
                        w,
                        "    coopStoreT({acc}{r}_{h}, {});
    workgroupBarrier();
    if {row}{r} < {length} {{
        for (var j = 0u; j < 2u; j++) {{
            {out}[({row}{r} * {heads} + {head}) * {head_vecs}u + {c4}u + half * 2u + j] = fs[(fbase + my_row * {output_stride}u) / 4u + half * 2u + j] * {factor};
        }}
    }}
    workgroupBarrier();",
                        at("fs", "fbase", l.output_stride),
                        c4 = h * 4,
                        output_stride = l.output_stride
                    )
                    .unwrap();
            }
        }
    }
    writeln!(w, "}}").unwrap();
    ShaderModule::with_hint(
        s,
        if kv {
            "attention_tiled_coop_grad_kv"
        } else {
            "attention_tiled_coop_grad_q"
        },
    )
}
