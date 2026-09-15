fn dequant_q8(k_idx: u32, n_idx: u32) -> f32 {
    // Q8_0: value = int8 * scale
    let blocks_per_col = params.k / 32u;
    let block = n_idx * blocks_per_col + k_idx / 32u;
    let in_block = k_idx % 32u;

    // Each block is 9 u32s: [scale_u32, data0..data7]
    let block_base = block * 9u;
    let scale = decode_f16(matrix_b[block_base] & 0xFFFFu);

    // Extract int8 from data u32s (4 bytes per u32)
    let byte_idx = in_block;
    let u32_idx = byte_idx / 4u;
    let byte_in_u32 = byte_idx % 4u;
    let data_u32 = matrix_b[block_base + 1u + u32_idx];
    let raw_byte = (data_u32 >> (byte_in_u32 * 8u)) & 0xFFu;

    // Sign-extend: if bit 7 is set, the value is negative
    let signed = i32(raw_byte) - select(0, 256, raw_byte >= 128u);
    return f32(signed) * scale;
}
