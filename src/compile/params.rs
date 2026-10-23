//! Uniform layouts shared by plan encoding, runtime binding and search.
//!
//! Dispatch words are byte-for-byte identical to the shader uniform. Decoding
//! requires the exact size: an old or truncated plan must not acquire defaults.
//! When migrating a family, also register its layout in the codegen binding
//! test and bump `cache::CACHE_FORMAT_VERSION` if its encoded words change.

pub(crate) trait DispatchParams: bytemuck::Pod {
    fn from_words(words: &[u32]) -> Option<Self> {
        bytemuck::try_pod_read_unaligned(bytemuck::cast_slice(words)).ok()
    }

    fn to_words(self) -> Vec<u32> {
        bytemuck::cast_slice(std::slice::from_ref(&self)).to_vec()
    }

    #[cfg(test)]
    const FIELDS: &'static [(&'static str, usize)];
}

// Generate the offset table from the declaration so adding a field cannot
// silently leave it out of the Rust/WGSL layout test.
macro_rules! params {
    ($name:ident { $($field:ident: $ty:ty),* $(,)? }) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Zeroable, bytemuck::Pod)]
        #[repr(C)]
        pub(crate) struct $name {
            $(pub $field: $ty,)*
        }

        impl DispatchParams for $name {
            #[cfg(test)]
            const FIELDS: &'static [(&'static str, usize)] = &[
                $((stringify!($field), std::mem::offset_of!(Self, $field)),)*
            ];
        }
    };
}

params!(RoPEParams {
    seq: u32,
    dim: u32,
    theta_bits: u32,
    pos_offset: u32,
    axes_dim: [u32; 3],
    head_dim: u32,
});

impl RoPEParams {
    pub fn new(seq: u32, dim: u32, theta: f32, head_dim: u32) -> Self {
        Self {
            seq,
            dim,
            theta_bits: theta.to_bits(),
            pos_offset: 0,
            axes_dim: [0; 3],
            head_dim,
        }
    }
}

params!(AttentionParams {
    q_seq: u32,
    kv_seq: u32,
    packed_heads: u32,
    head_dim: u32,
    window_size: u32,
    causal_rows: u32,
    _pad1: u32,
    _pad2: u32,
});

params!(CachedBlockAttentionParams {
    window_size: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_len: u32,
    max_seq: u32,
    splits: u32,
    _pad: u32,
});

params!(WinogradInputParams {
    batch: u32,
    in_channels: u32,
    in_h: u32,
    in_w: u32,
    padding: u32,
    tiles_h: u32,
    tiles_w: u32,
    total_tiles: u32,
});

params!(WinogradOutputParams {
    batch: u32,
    out_channels: u32,
    out_h: u32,
    out_w: u32,
    tiles_h: u32,
    tiles_w: u32,
    total_tiles: u32,
    _pad: u32,
});

params!(WinogradWeightParams {
    out_channels: u32,
    in_channels: u32,
    adjoint: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
});

#[cfg(test)]
mod tests {
    use super::*;

    fn check_encoding<P: DispatchParams + std::fmt::Debug + PartialEq>(params: P) {
        let words = params.to_words();
        assert_eq!(P::from_words(&words), Some(params));
        for len in 0..words.len() {
            assert_eq!(P::from_words(&words[..len]), None);
        }
        let mut oversized = words;
        oversized.push(0);
        assert_eq!(P::from_words(&oversized), None);
    }

    #[test]
    fn uniform_words_are_exact() {
        let rope = RoPEParams {
            pos_offset: 7,
            axes_dim: [2, 4, 6],
            ..RoPEParams::new(3, 48, 10_000.0, 16)
        };
        assert_eq!(
            rope.to_words(),
            [3, 48, 10_000f32.to_bits(), 7, 2, 4, 6, 16]
        );
        check_encoding(rope);
        let attention = AttentionParams {
            q_seq: 3,
            kv_seq: 5,
            packed_heads: (4 << 16) | 2,
            head_dim: 16,
            window_size: 7,
            causal_rows: 2,
            _pad1: 0,
            _pad2: 0,
        };
        assert_eq!(attention.to_words(), [3, 5, (4 << 16) | 2, 16, 7, 2, 0, 0]);
        check_encoding(attention);
        check_encoding(CachedBlockAttentionParams {
            window_size: 1,
            num_heads: 2,
            num_kv_heads: 3,
            head_dim: 4,
            block_len: 5,
            max_seq: 6,
            splits: 7,
            _pad: 0,
        });
    }
}
