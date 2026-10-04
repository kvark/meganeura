//! Host-side packing for Meganeura's own Q4 and Q8 formats.
//!
//! These are the formats a caller can produce from `f32` without a GGUF file,
//! unlike the K-quants and GGML Q4_0, which are load-only. The block geometry
//! they follow is `DType::block_geometry`.

use crate::compile::WeightFormat;

pub(super) fn encode_parameter(
    name: &str,
    data: &[f32],
    format: WeightFormat,
    rows: usize,
    cols: usize,
) -> Vec<u8> {
    match format {
        WeightFormat::Q4 => quantize_q4_0(data, rows, cols),
        WeightFormat::Q8 => quantize_q8_0(data, rows, cols),
        WeightFormat::F16 => data
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect(),
        WeightFormat::Q40
        | WeightFormat::Q4K
        | WeightFormat::Q6K
        | WeightFormat::Q5K
        | WeightFormat::Q3K => {
            panic!(
                "parameter `{name}` is {format:?}; load it with set_parameter_packed (no host encoder)"
            )
        }
        WeightFormat::F32 => unreachable!("f32 parameters do not need encoding"),
    }
}

pub fn quantize_q4_0(data: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(data.len(), rows * cols);
    assert!(
        rows.is_multiple_of(32),
        "Q4 requires rows (K dim) to be a multiple of 32"
    );

    let blocks_per_col = rows / 32;
    let num_blocks = blocks_per_col * cols;
    // Layout: [num_blocks u32s for (d|m)][num_blocks * 4 u32s for nibbles]
    let total_bytes = num_blocks * 5 * 4; // 5 u32s per block

    let mut buf = vec![0u8; total_bytes];

    for col in 0..cols {
        for blk in 0..blocks_per_col {
            let block_idx = col * blocks_per_col + blk;

            // Find min and max for this block
            let mut bmin = f32::INFINITY;
            let mut bmax = f32::NEG_INFINITY;
            for i in 0..32 {
                let row = blk * 32 + i;
                let val = data[row * cols + col];
                if val < bmin {
                    bmin = val;
                }
                if val > bmax {
                    bmax = val;
                }
            }

            // Asymmetric: d = (max - min) / 15, m = min
            let d = if bmax > bmin {
                (bmax - bmin) / 15.0
            } else {
                1.0
            };
            let m = bmin;
            let d_f16 = half::f16::from_f32(d);
            let m_f16 = half::f16::from_f32(m);

            // Pack d and m into one u32 (d in low 16, m in high 16)
            let dm_u32 = d_f16.to_bits() as u32 | ((m_f16.to_bits() as u32) << 16);
            let dm_offset = block_idx * 4;
            buf[dm_offset..dm_offset + 4].copy_from_slice(&dm_u32.to_le_bytes());

            // Write nibbles into the data region
            let data_byte_base = num_blocks * 4 + block_idx * 16;
            let d_inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            for i in 0..32 {
                let row = blk * 32 + i;
                let val = data[row * cols + col];
                let nibble = ((val - m) * d_inv).round().clamp(0.0, 15.0) as u8;

                let byte_in_block = i / 2;
                let byte_offset = data_byte_base + byte_in_block;
                if i % 2 == 0 {
                    buf[byte_offset] |= nibble;
                } else {
                    buf[byte_offset] |= nibble << 4;
                }
            }
        }
    }

    buf
}

/// Dequantize Q4_1 buffer back to f32 (CPU reference, mirrors shader logic).
pub fn dequantize_q4_0(buf: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    let blocks_per_col = rows / 32;
    let num_blocks = blocks_per_col * cols;

    let buf_u32: Vec<u32> = buf
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut out = vec![0.0f32; rows * cols];

    for col in 0..cols {
        for blk in 0..blocks_per_col {
            let block = col * blocks_per_col + blk;

            // Decode d and m from packed u32
            let dm = buf_u32[block];
            let d = half::f16::from_bits((dm & 0xFFFF) as u16).to_f32();
            let m = half::f16::from_bits((dm >> 16) as u16).to_f32();

            for i in 0..32 {
                let byte_in_block = i / 2;
                let u32_in_block = byte_in_block / 4;
                let data_u32 = buf_u32[num_blocks + block * 4 + u32_in_block];
                let shift = (byte_in_block % 4) * 8 + if i % 2 != 0 { 4 } else { 0 };
                let nibble = (data_u32 >> shift) & 0xF;
                let val = nibble as f32 * d + m;
                let row = blk * 32 + i;
                out[row * cols + col] = val;
            }
        }
    }
    out
}

/// Quantize f32 data to Q8_0 format (symmetric 8-bit, 32-element blocks).
/// Per block: scale = absmax / 127, int8 = round(val / scale), clamped to [-128, 127].
/// Layout: `[scale_u32 per block][8 data_u32s per block]` = 9 u32s per block.
pub fn quantize_q8_0(data: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(data.len(), rows * cols);
    assert!(
        rows.is_multiple_of(32),
        "Q8 requires rows (K dim) to be a multiple of 32"
    );

    let blocks_per_col = rows / 32;
    let num_blocks = blocks_per_col * cols;
    let total_bytes = num_blocks * 9 * 4; // 9 u32s per block

    let mut buf = vec![0u8; total_bytes];

    for col in 0..cols {
        for blk in 0..blocks_per_col {
            let block_idx = col * blocks_per_col + blk;
            let block_base = block_idx * 36; // 9 u32s = 36 bytes

            // Find absmax
            let mut absmax = 0.0f32;
            for i in 0..32 {
                let row = blk * 32 + i;
                let val = data[row * cols + col].abs();
                if val > absmax {
                    absmax = val;
                }
            }

            let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
            let scale_f16 = half::f16::from_f32(scale);

            // Write scale as f16 in low 16 bits of first u32
            buf[block_base..block_base + 2].copy_from_slice(&scale_f16.to_le_bytes());

            // Write int8 values
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for i in 0..32 {
                let row = blk * 32 + i;
                let val = data[row * cols + col];
                let q = (val * inv_scale).round().clamp(-128.0, 127.0) as i8;
                buf[block_base + 4 + i] = q as u8;
            }
        }
    }
    buf
}

/// Dequantize Q8_0 buffer back to f32 (CPU reference).
pub fn dequantize_q8_0(buf: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    let blocks_per_col = rows / 32;
    let num_blocks = blocks_per_col * cols;
    let _ = num_blocks;

    let mut out = vec![0.0f32; rows * cols];

    for col in 0..cols {
        for blk in 0..blocks_per_col {
            let block_idx = col * blocks_per_col + blk;
            let block_base = block_idx * 36;

            let scale_bits = u16::from_le_bytes([buf[block_base], buf[block_base + 1]]);
            let scale = half::f16::from_bits(scale_bits).to_f32();

            for i in 0..32 {
                let q = buf[block_base + 4 + i] as i8;
                let val = q as f32 * scale;
                let row = blk * 32 + i;
                out[row * cols + col] = val;
            }
        }
    }
    out
}
