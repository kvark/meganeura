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
