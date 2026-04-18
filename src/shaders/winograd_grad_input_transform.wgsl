// Winograd F(2,3) grad_input inverse transform: dV[16, Ci, P] → grad_input[N, Ci, H, W]
// Pixel-centric approach: each thread handles one (n, ci, ih, iw) output pixel.
// For each pixel, determine which tiles cover it (up to 2×2 tiles), inverse-transform
// the contributions and sum them.
// dd = B × dV × B^T where B = [[1,0,0,0],[0,1,-1,1],[-1,1,1,0],[0,0,0,-1]]
// (This is the adjoint of the forward input transform V = B^T × d × B)

struct Params {
    batch: u32,
    in_channels: u32,
    in_h: u32,
    in_w: u32,
    padding: u32,
    tiles_h: u32,
    tiles_w: u32,
    total_tiles: u32,
}

var<storage> src: array<f32>;                      // dV [16, Ci, P]
var<storage, read_write> dst: array<f32>;           // grad_input [N, Ci, H, W]
var<uniform> params: Params;

// Compute one row of the inverse transform B × dV × B^T for a given local position (lr, lc)
// within a tile. Returns the scalar value at that position.
// B = [[1,0,0,0],[0,1,-1,1],[-1,1,1,0],[0,0,0,-1]]
// B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
//
// The full inverse transform of a 4×4 matrix V is:
// dd = B × V × B^T
// dd[lr, lc] = sum_a sum_b B[lr,a] * V[a,b] * B^T[b,lc]
//            = sum_a sum_b B[lr,a] * V[a,b] * B[lc,b]
//
// We precompute the B row coefficients for lr and lc,
// then just dot them with V.

fn compute_b_row(idx: u32) -> vec4<f32> {
    // B rows: [1,0,0,0], [0,1,-1,1], [-1,1,1,0], [0,0,0,-1]
    switch idx {
        case 0u: { return vec4<f32>(1.0, 0.0, 0.0, 0.0); }
        case 1u: { return vec4<f32>(0.0, 1.0, -1.0, 1.0); }
        case 2u: { return vec4<f32>(-1.0, 1.0, 1.0, 0.0); }
        default: { return vec4<f32>(0.0, 0.0, 0.0, -1.0); }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.batch * params.in_channels * params.in_h * params.in_w;
    if idx >= total { return; }

    // Decompose idx → (n, ci, ih, iw)
    let spatial = params.in_h * params.in_w;
    let ch_spatial = params.in_channels * spatial;
    let n = idx / ch_spatial;
    let rem1 = idx - n * ch_spatial;
    let ci = rem1 / spatial;
    let rem2 = rem1 - ci * spatial;
    let ih = rem2 / params.in_w;
    let iw = rem2 - ih * params.in_w;

    let tiles_per_batch = params.tiles_h * params.tiles_w;
    let ci_p = params.in_channels * params.total_tiles;

    var result = 0.0;

    // Determine which tiles cover this pixel
    // A tile at (tile_r, tile_c) covers input rows [tile_r*2 - padding, tile_r*2 - padding + 3]
    // and input cols [tile_c*2 - padding, tile_c*2 - padding + 3].
    // So tile_r covers ih when tile_r*2 - padding <= ih <= tile_r*2 - padding + 3
    // i.e. tile_r >= (ih + padding - 3) / 2 and tile_r <= (ih + padding) / 2

    let ih_padded = ih + params.padding;
    let iw_padded = iw + params.padding;

    // tile_r range: max(0, ceil((ih_padded - 3) / 2)) to min(tiles_h - 1, ih_padded / 2)
    // Since ih_padded and tile_r are unsigned, handle carefully
    var tr_min = 0u;
    if ih_padded >= 3u {
        tr_min = (ih_padded - 3u + 1u) / 2u; // ceil division
    }
    let tr_max = min(params.tiles_h - 1u, ih_padded / 2u);

    var tc_min = 0u;
    if iw_padded >= 3u {
        tc_min = (iw_padded - 3u + 1u) / 2u;
    }
    let tc_max = min(params.tiles_w - 1u, iw_padded / 2u);

    for (var tr = tr_min; tr <= tr_max; tr++) {
        let lr = ih_padded - tr * 2u; // local row within tile (0..3)
        let b_row = compute_b_row(lr);

        for (var tc = tc_min; tc <= tc_max; tc++) {
            let lc = iw_padded - tc * 2u; // local col within tile (0..3)
            let b_col = compute_b_row(lc);

            let tile_idx = n * tiles_per_batch + tr * params.tiles_w + tc;
            let v_base = ci * params.total_tiles + tile_idx;

            // Load 4×4 transform coefficients for this tile
            // V[a, b] = src[a*4+b * ci_p + v_base] but layout is [16, Ci, P]
            // with alpha = a*4+b (row-major 4×4)
            // dd[lr, lc] = sum_a sum_b B[lr,a] * V[a,b] * B[lc,b]
            var val = 0.0;
            for (var a = 0u; a < 4u; a++) {
                if b_row[a] == 0.0 { continue; }
                for (var b = 0u; b < 4u; b++) {
                    if b_col[b] == 0.0 { continue; }
                    let alpha = a * 4u + b;
                    let v = src[alpha * ci_p + v_base];
                    val += b_row[a] * v * b_col[b];
                }
            }
            result += val;
        }
    }

    dst[idx] = result;
}
