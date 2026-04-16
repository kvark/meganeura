// Winograd F(2,3) weight transform: weight[Co, Ci, 3, 3] → U[16, Co, Ci]
// U = G × w × G^T where G = [[1,0,0],[1/2,1/2,1/2],[1/2,-1/2,1/2],[0,0,1]]
// Each thread handles one (co, ci) pair.
// Dispatch: [ceil(Co*Ci / 256), 1, 1]

struct Params {
    out_channels: u32,
    in_channels: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
}

var<storage> src: array<f32>;              // weight [Co, Ci, 3, 3] = [Co*Ci*9]
var<storage, read_write> dst: array<f32>;  // U [16, Co, Ci] = [16*Co*Ci]
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.out_channels * params.in_channels;
    if idx >= total { return; }

    let co = idx / params.in_channels;
    let ci = idx % params.in_channels;

    // Load 3×3 filter
    let base = idx * 9u;
    let w00 = src[base];     let w01 = src[base + 1u]; let w02 = src[base + 2u];
    let w10 = src[base + 3u]; let w11 = src[base + 4u]; let w12 = src[base + 5u];
    let w20 = src[base + 6u]; let w21 = src[base + 7u]; let w22 = src[base + 8u];

    // tmp = G × w (4×3)
    // G = [[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]]
    let t00 = w00;                                let t01 = w01;                                let t02 = w02;
    let t10 = 0.5 * (w00 + w10 + w20);           let t11 = 0.5 * (w01 + w11 + w21);           let t12 = 0.5 * (w02 + w12 + w22);
    let t20 = 0.5 * (w00 - w10 + w20);           let t21 = 0.5 * (w01 - w11 + w21);           let t22 = 0.5 * (w02 - w12 + w22);
    let t30 = w20;                                let t31 = w21;                                let t32 = w22;

    // u = tmp × G^T (4×4) where G^T[k][c] = G[c][k]
    let u00 = t00;                                let u01 = 0.5 * (t00 + t01 + t02);           let u02 = 0.5 * (t00 - t01 + t02);           let u03 = t02;
    let u10 = t10;                                let u11 = 0.5 * (t10 + t11 + t12);           let u12 = 0.5 * (t10 - t11 + t12);           let u13 = t12;
    let u20 = t20;                                let u21 = 0.5 * (t20 + t21 + t22);           let u22 = 0.5 * (t20 - t21 + t22);           let u23 = t22;
    let u30 = t30;                                let u31 = 0.5 * (t30 + t31 + t32);           let u32_ = 0.5 * (t30 - t31 + t32);          let u33 = t32;

    // Write U[alpha, co, ci] for alpha=0..15
    let co_ci = params.out_channels * params.in_channels;
    let out_base = co * params.in_channels + ci;
    dst[0u * co_ci + out_base] = u00;  dst[1u * co_ci + out_base] = u01;
    dst[2u * co_ci + out_base] = u02;  dst[3u * co_ci + out_base] = u03;
    dst[4u * co_ci + out_base] = u10;  dst[5u * co_ci + out_base] = u11;
    dst[6u * co_ci + out_base] = u12;  dst[7u * co_ci + out_base] = u13;
    dst[8u * co_ci + out_base] = u20;  dst[9u * co_ci + out_base] = u21;
    dst[10u * co_ci + out_base] = u22; dst[11u * co_ci + out_base] = u23;
    dst[12u * co_ci + out_base] = u30; dst[13u * co_ci + out_base] = u31;
    dst[14u * co_ci + out_base] = u32_; dst[15u * co_ci + out_base] = u33;
}
