//! Admission shared by measured extraction and forward attention lowering.

use crate::codegen::{CoopCaps, ShaderGroup, attention_coop_shared_bytes};

/// Native16 QK with scalar-f32 PV. This is a capability/shape check, not a
/// profitability threshold: short rows remain candidates for measurement.
pub(crate) fn admits_cooperative_f32(
    rows: u32,
    head_dim: u32,
    heads: u32,
    caps: CoopCaps,
    shared_memory_bytes: u32,
) -> bool {
    caps.f32_tile == 16
        && head_dim >= 16
        && head_dim.is_power_of_two()
        && rows >= 16
        && crate::compile::workgroups_within_portable_limits([rows.div_ceil(16), heads, 1])
        && attention_coop_shared_bytes(ShaderGroup::FlashAttentionCoopF32, head_dim)
            <= u64::from(shared_memory_bytes)
}

/// Rectangular F16 QK/PV, with padded heads and F32 online statistics.
/// Admission is separate from numerical qualification and whole-program timing.
pub(crate) fn admits_products(
    rows: u32,
    head_dim: u32,
    heads: u32,
    caps: CoopCaps,
    shared_memory_bytes: u32,
    subgroup_size: u32,
) -> bool {
    matches!(caps.f16_attention_rows, 8 | 16)
        && head_dim > 0
        && head_dim.is_multiple_of(4)
        && (4..=64).contains(&subgroup_size)
        && subgroup_size.is_power_of_two()
        && rows > 0
        && crate::compile::workgroups_within_portable_limits([rows.div_ceil(16), heads, 1])
        && attention_coop_shared_bytes(ShaderGroup::FlashAttentionProducts, head_dim)
            <= u64::from(shared_memory_bytes)
}

/// Each cooperative block maps 128 invocations to four complete 32-lane subgroups.
pub(crate) fn block_cooperative(caps: &blade_graphics::Capabilities, coop: CoopCaps) -> bool {
    caps.subgroup_size == 32
        && coop.supports_16x16_f16()
        && caps
            .full_subgroups
            .is_some_and(|sizes| sizes.supports(32, [128, 1, 1]))
}

/// Grid for the register-blocked alternative to an existing attention dispatch.
/// The forward shares one key/value tile across four native 16-row products.
/// Scalar gradients assign four output rows per lane. Scaled F16 gradients
/// keep two 16-row matrices per subgroup.
pub(crate) fn block_grid(
    d: &crate::compile::Dispatch,
    shared_bytes: u32,
    cooperative: bool,
) -> Option<[u32; 3]> {
    block_grid_for_shader(d, &d.shader, shared_bytes, cooperative)
}

pub(crate) fn block_grid_for_shader(
    d: &crate::compile::Dispatch,
    shader: &crate::compile::ShaderEntry,
    shared_bytes: u32,
    cooperative: bool,
) -> Option<[u32; 3]> {
    use crate::compile::{AttentionParams, DispatchParams as _, ShaderEntry};
    use crate::kernels::attention_grad::{Operands, Part, Path};
    if d.schedule_locked
        || !matches!(
            shader,
            ShaderEntry::FlashAttentionCoop
                | ShaderEntry::FlashAttentionProducts
                | ShaderEntry::AttentionGrad(_)
        )
    {
        return None;
    }
    let p = AttentionParams::from_words(&d.params)?;
    let heads = p.packed_heads >> 16;
    let kv_heads = p.packed_heads & 65535;
    let hd = p.head_dim;
    if !matches!(hd, 16 | 32 | 64 | 128)
        || kv_heads == 0
        || heads == 0
        || !heads.is_multiple_of(kv_heads)
        || p.q_seq == 0
    {
        return None;
    }
    let cols = hd / 4;
    let (bytes, rows, block, heads) = match *shader {
        // A short prefill already has a 16-row native kernel. Keep its
        // arithmetic alternatives ahead of a mostly empty 64-query block.
        ShaderEntry::FlashAttentionCoop | ShaderEntry::FlashAttentionProducts
            if cooperative && p.q_seq >= 64 && p.kv_seq > 0 && !d.requires_full_precision =>
        {
            // F16 P·V reuses the Q allocation for K/V after Q enters registers.
            let pv_bytes = if *shader == ShaderEntry::FlashAttentionProducts {
                512 * 8 + 4 * 4
            } else {
                32 * (cols + 1) * 16
            };
            let bytes = 64 * (cols + 2) * 8 + 64 * 9 * 16 + pv_bytes;
            (bytes, p.q_seq, 64, heads)
        }
        ShaderEntry::AttentionGrad(k)
            if k.path == Path::Cooperative(Operands::F16)
                && !d.requires_full_precision
                && (p.kv_seq == 0 || p.window_size == 0)
                && cooperative =>
        {
            use crate::codegen::attention_block_grad::{BLOCK_ROWS, CoopGradLayout};
            let kv = k.part == Part::KV;
            let rows = if kv && p.kv_seq != 0 {
                p.kv_seq
            } else {
                p.q_seq
            };
            (
                CoopGradLayout::new(kv, hd).shared_bytes(),
                rows,
                BLOCK_ROWS,
                if kv { kv_heads } else { heads },
            )
        }
        ShaderEntry::AttentionGrad(k)
            if k.path == Path::Flash
                && !d.attention_f16_tiles
                && (p.kv_seq == 0 || p.window_size == 0) =>
        {
            let kv = k.part == Part::KV;
            let rows = if kv && p.kv_seq != 0 {
                p.kv_seq
            } else {
                p.q_seq
            };
            let stats = if kv { cols } else { 32 };
            let bytes =
                16 * (2 * (32 + cols) * (cols + 1) + (1 + u32::from(kv)) * 8 * cols + stats);
            (bytes, rows, 32, if kv { kv_heads } else { heads })
        }
        _ => return None,
    };
    let grid = [rows.div_ceil(block), heads, 1];
    (bytes <= shared_bytes && crate::compile::workgroups_within_portable_limits(grid))
        .then_some(grid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{Kernel, ShaderEntry};
    use crate::kernels::attention_grad::{Operands, Path};
    use crate::{Graph, Session, compile, reference};

    #[test]
    fn block_attention_matches_reference_on_grouped_tails() {
        let gpu = reference::gpu::shared_context();
        let caps = gpu.capabilities();
        // The head-width-64 scalar tile uses just under 32 KiB.
        if caps.max_compute_shared_memory_size < 32 * 1024 {
            return;
        }
        let cooperative = block_cooperative(&caps, crate::runtime::auto_tune(&gpu, 0).coop_caps)
            && caps.max_compute_shared_memory_size >= 48 * 1024;
        // Reciprocal Q/K scales preserve the logits while crossing the F16 range.
        // The tiny V and loss also require undoing a combined scale beyond 2^-126.
        for (window, q_exp, v_exp, loss_scale) in [
            (0, 0, 0, 0.5),
            (19, 0, 0, 1e-8),
            (19, 20, -51, 2.0f32.powi(-51)),
            (19, 0, -58, 2.0f32.powi(-58)),
            (19, -20, 60, 2.0f32.powi(-60)),
        ] {
            let range_test = v_exp != 0;
            if range_test && !cooperative {
                continue;
            }
            let (queries, keys) = (133, if window == 0 { 137 } else { 133 });
            let mut g = Graph::new();
            let q = g.parameter("q", &[queries, 2 * 64]);
            let k = g.parameter("k", &[keys, 64]);
            let v = g.parameter("v", &[keys, 64]);
            let output = if window == 0 {
                g.multi_head_attn(q, k, v, 2, 1, 64, true)
            } else {
                g.sliding_window_attention(q, k, v, 2, 1, 64, window)
            };
            let loss = reference::gradients::weighted_loss(&mut g, output, 173, loss_scale);
            g.set_outputs(vec![loss, output]);
            let g = crate::autodiff::differentiate(&g);
            let mut feeds = reference::Feeds::new();
            feeds.fill_random(&g, 3671, 1.5);
            for (name, exp) in [("q", q_exp), ("k", -q_exp), ("v", v_exp)] {
                let mut values = feeds.f32(name).unwrap();
                for x in &mut values {
                    *x *= 2.0f32.powi(exp);
                }
                feeds.set(name, &values);
            }
            let expected = reference::evaluate(&g, &feeds).unwrap();
            let dense = compile::compile(&g);
            for (products, cooperative_grad) in [(false, false), (true, false), (false, true)] {
                if (range_test && !cooperative_grad)
                    || ((products || cooperative_grad) && !cooperative)
                {
                    continue;
                }
                let mut plan = dense.clone();
                let mut forwards = 0;
                let mut gradients = 0;
                for d in &mut plan.dispatches {
                    if let ShaderEntry::AttentionGrad(ref mut kernel) = d.shader {
                        kernel.path = if cooperative_grad {
                            Path::Cooperative(Operands::F16)
                        } else {
                            Path::Flash
                        };
                    } else if matches!(
                        d.shader,
                        ShaderEntry::FlashAttention | ShaderEntry::MultiHeadAttn
                    ) && cooperative
                        && !range_test
                    {
                        d.shader = if products {
                            ShaderEntry::FlashAttentionProducts
                        } else {
                            ShaderEntry::FlashAttentionCoop
                        };
                    }
                    if let Some(grid) =
                        block_grid(d, caps.max_compute_shared_memory_size, cooperative)
                    {
                        d.kernel = Kernel::BlockAttention;
                        d.workgroups = grid;
                        if matches!(d.shader, ShaderEntry::AttentionGrad(_)) {
                            gradients += 1;
                        } else {
                            forwards += 1;
                        }
                    }
                }
                assert_eq!(gradients, 2);
                if cooperative && !range_test {
                    assert_eq!(forwards, 1);
                }
                let mut session = Session::with_context(plan, gpu.clone());
                for name in ["q", "k", "v"] {
                    session.set_parameter(name, &feeds.f32(name).unwrap());
                }
                session.step();
                session.wait();
                for (index, id) in g.outputs().iter().enumerate() {
                    let want = &expected[*id as usize].data;
                    let mut got = vec![0.0; want.len()];
                    if index < 2 {
                        session.read_output_by_index(index, &mut got);
                    } else {
                        session.read_param_grad(["q", "k", "v"][index - 2], &mut got);
                    }
                    let peak = want.iter().copied().map(f64::abs).fold(0.0, f64::max);
                    let tolerance = if cooperative_grad && index >= 2 {
                        2e-2
                    } else if forwards == 0 {
                        1e-4
                    } else {
                        3e-3
                    };
                    assert!(
                        got.iter()
                            .zip(want)
                            .all(|(&v, &r)| (f64::from(v) - r).abs() <= peak * tolerance),
                        "output {index}, window {window}, q_exp {q_exp}, products {products}, coop_grad {cooperative_grad}, peak {peak:e}"
                    );
                }
            }
        }
    }
}
