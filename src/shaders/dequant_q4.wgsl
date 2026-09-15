fn q4_unpack_nibble(data: u32, d: f32, m: f32, in_word: u32) -> f32 {
    let nibble = (data >> (in_word * 4u)) & 0xFu;
    return f32(nibble) * d + m;
}

fn dequant_q4(k_idx: u32, n_idx: u32) -> f32 {
    // Q4_1 asymmetric: value = nibble * d + m
    // Layout: [num_blocks u32s: (d_f16|m_f16)][num_blocks*4 u32s: nibble data]
    let blocks_per_col = params.k / 32u;
    let num_blocks = blocks_per_col * params.n;
    let block = n_idx * blocks_per_col + k_idx / 32u;
    let in_block = k_idx % 32u;

    let dm = decode_f16_pair(matrix_b[block]);
    let d = dm.x;
    let m = dm.y;
    let data_u32 = matrix_b[num_blocks + block * 4u + in_block / 8u];
    return q4_unpack_nibble(data_u32, d, m, in_block % 8u);
}

fn dequant_q4_pack8(k_base: u32, n_idx: u32) -> array<f32, 8> {
    let blocks_per_col = params.k / 32u;
    let num_blocks = blocks_per_col * params.n;
    let block = n_idx * blocks_per_col + k_base / 32u;
    let dm = decode_f16_pair(matrix_b[block]);
    let d = dm.x;
    let m = dm.y;
    let data_u32 = matrix_b[num_blocks + block * 4u + (k_base % 32u) / 8u];
    var out: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) {
        out[i] = q4_unpack_nibble(data_u32, d, m, i);
    }
    return out;
}
