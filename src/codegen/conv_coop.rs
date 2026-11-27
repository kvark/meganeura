//! Cooperative-matrix 2D convolution WGSL generator, lowered through the
//! unified subgroup-tiled product kernel (`workgroup::subgroup_kernel`).

use super::workgroup::{
    Gather, Images, SubgroupLayout, WorkgroupMatmulShape, checked_subgroup_kernel_width,
    subgroup_kernel,
};
use super::{CoopConfig, MatMulCoopVariant, MatrixTypes, ShaderModule};

/// Conv2d cooperative-matrix GEMM direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conv2dCoopDirection {
    /// Forward: C[Co, oH*oW] = A[Co, K] * B[K, oH*oW], K = Ci*kH*kW
    Forward,
    /// Backward w.r.t. input: C[Ci, H*W] = A[Ci, K] * B[K, H*W], K = Co*kH*kW
    GradInput,
}

/// Generate a specialized conv2d cooperative-matrix GEMM kernel with
/// compile-time kernel size and stride constants, specialized to `subgroup`-wide
/// subgroups (8, 16, 32, or 64; `0` defaults to 32).
///
/// The generated shader uses the same bindings and `Params` layout as the
/// scalar conv2d entries (`Conv2dData` / `Conv2dGradInputData`) and covers
/// `[ceil(M / output_tile), ceil(N / output_tile), batch]` workgroups via a
/// 2×2 subgroup grid inside `subgroup_kernel`. Interior workgroups store
/// directly with `coopStoreT`; ragged edge tiles stage through shared memory.
pub fn generate_conv2d_coop_module(
    kernel_h: u32,
    kernel_w: u32,
    stride: u32,
    direction: Conv2dCoopDirection,
    config: &CoopConfig,
    subgroup: u32,
) -> ShaderModule {
    let tile = config.tile_size;
    let output_tile = config.output_tile();
    let subgroup = checked_subgroup_kernel_width(subgroup);
    let types = if config.use_f16_input {
        MatrixTypes::F32_AS_F16
    } else {
        MatrixTypes::F32
    };
    let k_stage = if config.use_f16_input { 32 } else { 16 };
    let shape = WorkgroupMatmulShape {
        rows: output_tile,
        columns: output_tile,
        k_stage,
        threads: 4 * subgroup,
        stage_operands: false,
        subgroup_rows: 2,
        prefetch: false,
    };
    let kernel_hw = kernel_h * kernel_w;
    let backward = direction == Conv2dCoopDirection::GradInput;
    let params_and_consts = format!(
        "{}\n{}\n\
         const KERNEL_H: u32 = {kernel_h}u;\n\
         const KERNEL_W: u32 = {kernel_w}u;\n\
         const KERNEL_HW: u32 = {kernel_hw}u;\n\
         const STRIDE: u32 = {stride}u;\n",
        crate::divisor::SHADER,
        include_str!("../shaders/conv2d_params.wgsl"),
    );

    let (header, dimensions, a, b, gather) = if backward {
        let gather_wt4 = if kernel_h == 1 && kernel_w == 1 {
            "fn gather_wt4(_img: u32, ci: u32, tap0: u32, end: u32) -> vec4<f32> {\n\
                 var out = vec4<f32>(0.0);\n\
                 if ci >= params.in_channels { return out; }\n\
                 for (var e = 0u; e < 4u; e++) {\n\
                     if tap0 + e < end { out[e] = weight[(tap0 + e) * params.in_channels + ci]; }\n\
                 }\n\
                 return out;\n\
             }\n"
            .to_owned()
        } else {
            "fn gather_wt4(_img: u32, ci: u32, tap0: u32, end: u32) -> vec4<f32> {\n\
                 var out = vec4<f32>(0.0);\n\
                 if ci >= params.in_channels { return out; }\n\
                 var co = tap0 / KERNEL_HW;\n\
                 let t0 = tap0 - co * KERNEL_HW;\n\
                 var ky = t0 / KERNEL_W;\n\
                 var kx = t0 - ky * KERNEL_W;\n\
                 for (var e = 0u; e < 4u; e++) {\n\
                     if tap0 + e < end {\n\
                         out[e] = weight[(co * params.in_channels + ci) * KERNEL_HW + ky * KERNEL_W + kx];\n\
                     }\n\
                     kx += 1u;\n\
                     if kx == KERNEL_W {\n\
                         kx = 0u;\n\
                         ky += 1u;\n\
                         if ky == KERNEL_H {\n\
                             ky = 0u;\n\
                             co += 1u;\n\
                         }\n\
                     }\n\
                 }\n\
                 return out;\n\
             }\n"
            .to_owned()
        };
        let dy_tap_check = if stride == 1 {
            "if tap0 + e < end && oh_num >= 0 && u32(oh_num) < params.out_h && ow_num >= 0 && u32(ow_num) < params.out_w {\n\
                 out[e] = grad_out[img_base + co * go_hw + u32(oh_num) * params.out_w + u32(ow_num)];\n\
             }"
        } else {
            "if tap0 + e < end && oh_num >= 0 && ow_num >= 0 && (u32(oh_num) % STRIDE) == 0u && (u32(ow_num) % STRIDE) == 0u {\n\
                 let oh = u32(oh_num) / STRIDE;\n\
                 let ow = u32(ow_num) / STRIDE;\n\
                 if oh < params.out_h && ow < params.out_w {\n\
                     out[e] = grad_out[img_base + co * go_hw + oh * params.out_w + ow];\n\
                 }\n\
             }"
        };
        let gather_dy4 = format!(
            "fn gather_dy4(image: u32, pos: u32, tap0: u32, end: u32) -> vec4<f32> {{\n\
                 var out = vec4<f32>(0.0);\n\
                 let n_total = params.in_h * params.in_w;\n\
                 if pos >= n_total {{ return out; }}\n\
                 let ih = divide_exact(pos, params.in_w, params.column_width_multiplier);\n\
                 let iw = pos - ih * params.in_w;\n\
                 let pad_h = i32(params.padding_h);\n\
                 let pad_w = i32(params.padding_w);\n\
                 let go_hw = params.out_h * params.out_w;\n\
                 let img_base = image * params.out_channels * go_hw;\n\
                 var co = tap0 / KERNEL_HW;\n\
                 let t0 = tap0 - co * KERNEL_HW;\n\
                 var ky = t0 / KERNEL_W;\n\
                 var kx = t0 - ky * KERNEL_W;\n\
                 for (var e = 0u; e < 4u; e++) {{\n\
                     let oh_num = i32(ih) + pad_h - i32(ky);\n\
                     let ow_num = i32(iw) + pad_w - i32(kx);\n\
                     {dy_tap_check}\n\
                     kx += 1u;\n\
                     if kx == KERNEL_W {{\n\
                         kx = 0u;\n\
                         ky += 1u;\n\
                         if ky == KERNEL_H {{\n\
                             ky = 0u;\n\
                             co += 1u;\n\
                         }}\n\
                     }}\n\
                 }}\n\
                 return out;\n\
             }}\n"
        );
        let header = format!(
            "{params_and_consts}\
             var<storage> grad_out: array<f32>;\n\
             var<storage> weight: array<f32>;\n\
             var<storage, read_write> dst: array<f32>;\n\
             var<uniform> params: Params;\n\
             {gather_wt4}{gather_dy4}"
        );
        let dimensions = "let m = params.in_channels;\n\
                          let n = params.in_h * params.in_w;\n\
                          let k = params.out_channels * KERNEL_HW;"
            .to_owned();
        (
            header,
            dimensions,
            "weight",
            "grad_out",
            [
                Some(Gather {
                    name: "gather_wt",
                    vector: true,
                }),
                Some(Gather {
                    name: "gather_dy",
                    vector: true,
                }),
            ],
        )
    } else {
        let gather_w4 = "fn gather_w4(_img: u32, co: u32, tap0: u32, end: u32) -> vec4<f32> {\n\
                 var out = vec4<f32>(0.0);\n\
                 let k_total = params.in_channels * KERNEL_HW;\n\
                 if co >= params.out_channels { return out; }\n\
                 let base = co * k_total + tap0;\n\
                 if tap0 + 4u <= end {\n\
                     out = vec4<f32>(weight[base], weight[base + 1u], weight[base + 2u], weight[base + 3u]);\n\
                 } else {\n\
                     for (var e = 0u; e < 4u; e++) {\n\
                         if tap0 + e < end { out[e] = weight[base + e]; }\n\
                     }\n\
                 }\n\
                 return out;\n\
             }\n";
        let gather_x4 = if kernel_h == 1 && kernel_w == 1 && stride == 1 {
            "fn gather_x4(image: u32, pos: u32, tap0: u32, end: u32) -> vec4<f32> {\n\
                 var out = vec4<f32>(0.0);\n\
                 let n_total = params.out_h * params.out_w;\n\
                 if pos >= n_total { return out; }\n\
                 let in_hw = params.in_h * params.in_w;\n\
                 let img_base = image * params.in_channels * in_hw;\n\
                 if params.padding_h == 0u && params.padding_w == 0u && params.in_h == params.out_h && params.in_w == params.out_w {\n\
                     let col_base = img_base + pos;\n\
                     for (var e = 0u; e < 4u; e++) {\n\
                         if tap0 + e < end { out[e] = src[col_base + (tap0 + e) * in_hw]; }\n\
                     }\n\
                     return out;\n\
                 }\n\
                 let oh = divide_exact(pos, params.out_w, params.column_width_multiplier);\n\
                 let ow = pos - oh * params.out_w;\n\
                 let ih = i32(oh) - i32(params.padding_h);\n\
                 let iw = i32(ow) - i32(params.padding_w);\n\
                 if ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w {\n\
                     let col_base = img_base + u32(ih) * params.in_w + u32(iw);\n\
                     for (var e = 0u; e < 4u; e++) {\n\
                         if tap0 + e < end { out[e] = src[col_base + (tap0 + e) * in_hw]; }\n\
                     }\n\
                 }\n\
                 return out;\n\
             }\n"
        } else {
            "fn gather_x4(image: u32, pos: u32, tap0: u32, end: u32) -> vec4<f32> {\n\
                 var out = vec4<f32>(0.0);\n\
                 let n_total = params.out_h * params.out_w;\n\
                 if pos >= n_total { return out; }\n\
                 let oh = divide_exact(pos, params.out_w, params.column_width_multiplier);\n\
                 let ow = pos - oh * params.out_w;\n\
                 let oy = i32(oh * STRIDE) - i32(params.padding_h);\n\
                 let ox = i32(ow * STRIDE) - i32(params.padding_w);\n\
                 let in_hw = params.in_h * params.in_w;\n\
                 let img_base = image * params.in_channels * in_hw;\n\
                 var ci = tap0 / KERNEL_HW;\n\
                 let t0 = tap0 - ci * KERNEL_HW;\n\
                 var ky = t0 / KERNEL_W;\n\
                 var kx = t0 - ky * KERNEL_W;\n\
                 for (var e = 0u; e < 4u; e++) {\n\
                     let ih = oy + i32(ky);\n\
                     let iw = ox + i32(kx);\n\
                     if tap0 + e < end && ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w {\n\
                         out[e] = src[img_base + ci * in_hw + u32(ih) * params.in_w + u32(iw)];\n\
                     }\n\
                     kx += 1u;\n\
                     if kx == KERNEL_W {\n\
                         kx = 0u;\n\
                         ky += 1u;\n\
                         if ky == KERNEL_H {\n\
                             ky = 0u;\n\
                             ci += 1u;\n\
                         }\n\
                     }\n\
                 }\n\
                 return out;\n\
             }\n"
        };
        let header = format!(
            "{params_and_consts}\
             var<storage> src: array<f32>;\n\
             var<storage> weight: array<f32>;\n\
             var<storage, read_write> dst: array<f32>;\n\
             var<uniform> params: Params;\n\
             {gather_w4}{gather_x4}"
        );
        let dimensions = "let m = params.out_channels;\n\
                          let n = params.out_h * params.out_w;\n\
                          let k = params.in_channels * KERNEL_HW;"
            .to_owned();
        (
            header,
            dimensions,
            "weight",
            "src",
            [
                Some(Gather {
                    name: "gather_w",
                    vector: true,
                }),
                Some(Gather {
                    name: "gather_x",
                    vector: true,
                }),
            ],
        )
    };

    subgroup_kernel(
        shape,
        types,
        [tile, tile, tile],
        subgroup,
        &SubgroupLayout {
            variant: MatMulCoopVariant::BT,
            header,
            dimensions,
            a,
            b,
            c: "dst",
            addend: None,
            prologue: None,
            epilogue: None,
            images: Some(Images {
                a: "0u".into(),
                b: "0u".into(),
                c: "image * m * n".into(),
                batch: None,
            }),
            fixed_splits: None,
            unaligned_loads: false,
            copies: 1,
            row_bounds: true,
            column_bounds: true,
            gather,
            scatter: None,
        },
    )
}
