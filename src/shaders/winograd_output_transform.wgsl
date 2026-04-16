// Winograd F(2,3) output transform: M[16, Co, P] → output[N, Co, oH, oW]
// Each thread handles one (tile_idx, co) pair, transforming 16 coefficients to 2×2 output.
// Y = A^T × m × A where A^T = [[1,1,1,0],[0,1,-1,-1]]

struct Params {
    batch: u32,
    out_channels: u32,
    out_h: u32,
    out_w: u32,
    tiles_h: u32,
    tiles_w: u32,
    total_tiles: u32,
    _pad: u32,
}

var<storage> src: array<f32>;              // M [16, Co, P]
var<storage, read_write> dst: array<f32>;  // output [N, Co, oH, oW]
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.total_tiles * params.out_channels;
    if idx >= total { return; }

    let tile_idx = idx / params.out_channels;
    let co = idx % params.out_channels;

    // Decompose tile_idx → (n, tile_r, tile_c)
    let tiles_per_batch = params.tiles_h * params.tiles_w;
    let n = tile_idx / tiles_per_batch;
    let tile_rem = tile_idx - n * tiles_per_batch;
    let tile_r = tile_rem / params.tiles_w;
    let tile_c = tile_rem - tile_r * params.tiles_w;

    // Load 16 transform-domain coefficients: M[alpha, co, tile_idx]
    let co_p = params.out_channels * params.total_tiles;
    let base = co * params.total_tiles + tile_idx;
    let m00 = src[0u * co_p + base];  let m01 = src[1u * co_p + base];
    let m02 = src[2u * co_p + base];  let m03 = src[3u * co_p + base];
    let m10 = src[4u * co_p + base];  let m11 = src[5u * co_p + base];
    let m12 = src[6u * co_p + base];  let m13 = src[7u * co_p + base];
    let m20 = src[8u * co_p + base];  let m21 = src[9u * co_p + base];
    let m22 = src[10u * co_p + base]; let m23 = src[11u * co_p + base];
    let m30 = src[12u * co_p + base]; let m31 = src[13u * co_p + base];
    let m32 = src[14u * co_p + base]; let m33 = src[15u * co_p + base];

    // A^T × m × A for F(2,3)
    // A^T = [[1,1,1,0],[0,1,-1,-1]]
    // First: s = A^T × m (row transform)
    let s00 = m00 + m10 + m20;  let s01 = m01 + m11 + m21;
    let s02 = m02 + m12 + m22;  let s03 = m03 + m13 + m23;
    let s10 = m10 - m20 - m30;  let s11 = m11 - m21 - m31;
    let s12 = m12 - m22 - m32;  let s13 = m13 - m23 - m33;
    // Then: Y = s × A (column transform)
    let y00 = s00 + s01 + s02;
    let y01 = s01 - s02 - s03;
    let y10 = s10 + s11 + s12;
    let y11 = s11 - s12 - s13;

    // Write 2×2 output tile with bounds checking
    let oh_base = tile_r * 2u;
    let ow_base = tile_c * 2u;
    let out_base = n * params.out_channels * params.out_h * params.out_w
                 + co * params.out_h * params.out_w;

    if oh_base < params.out_h && ow_base < params.out_w {
        dst[out_base + oh_base * params.out_w + ow_base] = y00;
    }
    if oh_base < params.out_h && (ow_base + 1u) < params.out_w {
        dst[out_base + oh_base * params.out_w + ow_base + 1u] = y01;
    }
    if (oh_base + 1u) < params.out_h && ow_base < params.out_w {
        dst[out_base + (oh_base + 1u) * params.out_w + ow_base] = y10;
    }
    if (oh_base + 1u) < params.out_h && (ow_base + 1u) < params.out_w {
        dst[out_base + (oh_base + 1u) * params.out_w + ow_base + 1u] = y11;
    }
}
