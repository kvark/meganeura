// Winograd F(2,3) input transform: input[N, Ci, H, W] → V[16, Ci, P]
// where P = batch * tiles_h * tiles_w, tiles_h = ceil(out_h/2), tiles_w = ceil(out_w/2).
// Each thread handles one (tile_idx, ci) pair, computing 16 transform coefficients.
// V = B^T × d × B where d is a 4×4 input patch.

struct Params {
    batch: u32,
    in_channels: u32,
    in_h: u32,
    in_w: u32,
    padding: u32,
    tiles_h: u32,
    tiles_w: u32,
    total_tiles: u32,  // batch * tiles_h * tiles_w
}

var<storage> src: array<f32>;              // input [N, Ci, H, W]
var<storage, read_write> dst: array<f32>;  // V [16, Ci, P]
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.total_tiles * params.in_channels;
    if idx >= total { return; }

    let tile_idx = idx / params.in_channels;
    let ci = idx % params.in_channels;

    // Decompose tile_idx → (n, tile_r, tile_c)
    let tiles_per_batch = params.tiles_h * params.tiles_w;
    let n = tile_idx / tiles_per_batch;
    let tile_rem = tile_idx - n * tiles_per_batch;
    let tile_r = tile_rem / params.tiles_w;
    let tile_c = tile_rem - tile_r * params.tiles_w;

    // Input patch origin (may be negative due to padding)
    let ih_base = i32(tile_r * 2u) - i32(params.padding);
    let iw_base = i32(tile_c * 2u) - i32(params.padding);

    // Load 4×4 input patch with zero-padding
    let input_base = n * params.in_channels * params.in_h * params.in_w
                   + ci * params.in_h * params.in_w;
    var d: array<f32, 16>;
    for (var r = 0u; r < 4u; r++) {
        for (var c = 0u; c < 4u; c++) {
            let ih = ih_base + i32(r);
            let iw = iw_base + i32(c);
            var val = 0.0;
            if ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w {
                val = src[input_base + u32(ih) * params.in_w + u32(iw)];
            }
            d[r * 4u + c] = val;
        }
    }

    // B^T × d × B for F(2,3)
    // B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
    // First: t = B^T × d (row transform)
    let t00 = d[0] - d[8];    let t01 = d[1] - d[9];    let t02 = d[2] - d[10];   let t03 = d[3] - d[11];
    let t10 = d[4] + d[8];    let t11 = d[5] + d[9];    let t12 = d[6] + d[10];   let t13 = d[7] + d[11];
    let t20 = -d[4] + d[8];   let t21 = -d[5] + d[9];   let t22 = -d[6] + d[10];  let t23 = -d[7] + d[11];
    let t30 = d[4] - d[12];   let t31 = d[5] - d[13];   let t32 = d[6] - d[14];   let t33 = d[7] - d[15];
    // Then: V = t × B (column transform)
    let v00 = t00 - t02;  let v01 = t01 + t02;  let v02 = -t01 + t02;  let v03 = t01 - t03;
    let v04 = t10 - t12;  let v05 = t11 + t12;  let v06 = -t11 + t12;  let v07 = t11 - t13;
    let v08 = t20 - t22;  let v09 = t21 + t22;  let v10 = -t21 + t22;  let v11 = t21 - t23;
    let v12 = t30 - t32;  let v13 = t31 + t32;  let v14 = -t31 + t32;  let v15 = t31 - t33;

    // Write V[alpha, ci, tile_idx] for alpha=0..15
    // Layout: V[alpha * Ci * P + ci * P + tile_idx]
    let ci_p = params.in_channels * params.total_tiles;
    let base = ci * params.total_tiles + tile_idx;
    dst[0u * ci_p + base] = v00;  dst[1u * ci_p + base] = v01;
    dst[2u * ci_p + base] = v02;  dst[3u * ci_p + base] = v03;
    dst[4u * ci_p + base] = v04;  dst[5u * ci_p + base] = v05;
    dst[6u * ci_p + base] = v06;  dst[7u * ci_p + base] = v07;
    dst[8u * ci_p + base] = v08;  dst[9u * ci_p + base] = v09;
    dst[10u * ci_p + base] = v10; dst[11u * ci_p + base] = v11;
    dst[12u * ci_p + base] = v12; dst[13u * ci_p + base] = v13;
    dst[14u * ci_p + base] = v14; dst[15u * ci_p + base] = v15;
}
