// Winograd F(2,3) grad_output transform: grad_output[N, Co, oH, oW] → dM[16, Co, P]
// where P = batch * tiles_h * tiles_w, tiles_h = ceil(oH/2), tiles_w = ceil(oW/2).
// Each thread handles one (tile_idx, co) pair.
// dM = A × dY × A^T where A = [[1,0],[1,1],[1,-1],[0,-1]]
// (This is the adjoint/transpose of the forward output transform Y = A^T × M × A)

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

var<storage> src: array<f32>;              // grad_output [N, Co, oH, oW]
var<storage, read_write> dst: array<f32>;  // dM [16, Co, P]
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

    // Load 2×2 grad_output tile with bounds checking
    let oh_base = tile_r * 2u;
    let ow_base = tile_c * 2u;
    let src_base = n * params.out_channels * params.out_h * params.out_w
                 + co * params.out_h * params.out_w;

    var y00 = 0.0; var y01 = 0.0; var y10 = 0.0; var y11 = 0.0;
    if oh_base < params.out_h && ow_base < params.out_w {
        y00 = src[src_base + oh_base * params.out_w + ow_base];
    }
    if oh_base < params.out_h && (ow_base + 1u) < params.out_w {
        y01 = src[src_base + oh_base * params.out_w + ow_base + 1u];
    }
    if (oh_base + 1u) < params.out_h && ow_base < params.out_w {
        y10 = src[src_base + (oh_base + 1u) * params.out_w + ow_base];
    }
    if (oh_base + 1u) < params.out_h && (ow_base + 1u) < params.out_w {
        y11 = src[src_base + (oh_base + 1u) * params.out_w + ow_base + 1u];
    }

    // dM = A × dY × A^T
    // A = [[1,0],[1,1],[1,-1],[0,-1]]
    // A^T = [[1,1,1,0],[0,1,-1,-1]]
    //
    // First: t = A × dY (4×2)
    // t[0] = (y00, y01)
    // t[1] = (y00+y10, y01+y11)
    // t[2] = (y00-y10, y01-y11)
    // t[3] = (-y10, -y11)
    //
    // Then: dM = t × A^T (4×4)
    // dM[i,0] = t[i,0]
    // dM[i,1] = t[i,0] + t[i,1]
    // dM[i,2] = t[i,0] - t[i,1]
    // dM[i,3] = -t[i,1]

    let dm00 = y00;                     let dm01 = y00 + y01;
    let dm02 = y00 - y01;               let dm03 = -y01;
    let dm04 = y00 + y10;               let dm05 = y00 + y01 + y10 + y11;
    let dm06 = y00 - y01 + y10 - y11;   let dm07 = -y01 - y11;
    let dm08 = y00 - y10;               let dm09 = y00 + y01 - y10 - y11;
    let dm10 = y00 - y01 - y10 + y11;   let dm11 = -y01 + y11;
    let dm12 = -y10;                    let dm13 = -y10 - y11;
    let dm14 = -y10 + y11;              let dm15 = y11;

    // Write dM[alpha, co, tile_idx] for alpha=0..15
    let co_p = params.out_channels * params.total_tiles;
    let base = co * params.total_tiles + tile_idx;
    dst[0u * co_p + base] = dm00;  dst[1u * co_p + base] = dm01;
    dst[2u * co_p + base] = dm02;  dst[3u * co_p + base] = dm03;
    dst[4u * co_p + base] = dm04;  dst[5u * co_p + base] = dm05;
    dst[6u * co_p + base] = dm06;  dst[7u * co_p + base] = dm07;
    dst[8u * co_p + base] = dm08;  dst[9u * co_p + base] = dm09;
    dst[10u * co_p + base] = dm10; dst[11u * co_p + base] = dm11;
    dst[12u * co_p + base] = dm12; dst[13u * co_p + base] = dm13;
    dst[14u * co_p + base] = dm14; dst[15u * co_p + base] = dm15;
}
