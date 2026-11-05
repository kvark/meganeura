use super::{MatrixTypes, ShaderGroup, ShaderModule, preprocess};
use crate::DType;

fn extension(storage: DType) -> &'static str {
    match storage {
        DType::F32 => "",
        DType::F16 => "enable f16;",
        DType::BF16 => "enable wgpu_bfloat16;",
        _ => unreachable!(),
    }
}

pub(crate) fn transform(group: ShaderGroup, storage: DType) -> ShaderModule {
    let source = match group {
        ShaderGroup::WinogradInputTransform => {
            include_str!("../shaders/winograd_input_transform.wgsl")
        }
        ShaderGroup::WinogradWeightTransform => {
            include_str!("../shaders/winograd_weight_transform.wgsl")
        }
        _ => unreachable!(),
    };
    let mut source = preprocess(
        source,
        &[
            ("$EXTENSIONS", extension(storage)),
            ("$OUTPUT", MatrixTypes::name(storage)),
            (
                "$STORE",
                if storage == DType::BF16 {
                    "round_bf16"
                } else {
                    MatrixTypes::name(storage)
                },
            ),
        ],
    );
    if storage == DType::BF16 {
        source.push_str(include_str!("../shaders/bf16_round.wgsl"));
    }
    ShaderModule::new(&source)
}

pub(crate) fn matmul(
    storage: DType,
    native: [u32; 3],
    tile: [u32; 2],
    subgroup: u32,
) -> ShaderModule {
    let [m, n, k] = native;
    let [rows, columns] = tile;
    assert!(matches!(storage, DType::F16 | DType::BF16));
    assert!(super::checked_cooperative_shape(&[native]).is_some());
    assert!(rows.is_multiple_of(m) && columns.is_multiple_of(n));
    // The reported subgroup width is a sizing hint. Cover wider runtime
    // subgroups too; the shader distributes tiles using the actual width.
    let threads = (rows / m * (columns / n) * subgroup).max(128);
    assert!(threads > 0 && threads <= 256);
    let ty = MatrixTypes::name(storage);
    ShaderModule::new(&preprocess(
        include_str!("../shaders/winograd_matmul_half.wgsl"),
        &[
            ("$EXTENSIONS", extension(storage)),
            ("$TYPE", ty),
            ("$THREADS", &threads.to_string()),
            ("$ROWS", &rows.to_string()),
            ("$COLUMNS", &columns.to_string()),
            ("$M", &m.to_string()),
            ("$N", &n.to_string()),
            ("$K", &k.to_string()),
            ("$A_TILE", &format!("coop_mat{k}x{m}<{ty}, A>")),
            ("$B_TILE", &format!("coop_mat{n}x{k}<{ty}, B>")),
            ("$C_TILE", &format!("coop_mat{n}x{m}<f32, C>")),
        ],
    ))
}

/// F(4,3) adjoints: V=BᵀXB, D=A dY Aᵀ, dW=Gᵀ(DVᵀ)G.
/// All intermediates and products remain F32, including strict training.
pub(crate) fn weight_gradient(group: ShaderGroup) -> ShaderModule {
    use super::{MatMulIndexing, MatMulOptions, MatMulTile, MatmulKnobs};
    if let ShaderGroup::WinogradGradProduct(splits) = group {
        assert!(splits > 0);
        let (a_row, a_col, b_row, b_col) = super::matmul_stage_maps(ShaderGroup::MatMulBT);
        return super::matmul_vars_tiled(
            MatMulIndexing {
                a_idx: "((wgid.z % 36u) * params.m + a_row) * params.k + a_col",
                b_idx: "((wgid.z % 36u) * params.n + b_col) * params.k + b_row",
                a_row,
                a_col,
                b_row,
                b_col,
                tile_row: "wgid.y",
                c_idx: "((wgid.z % 36u) * params.m + row) * params.n + col",
                split_id: "wgid.z / 36u",
                split_stride: "36u * params.m * params.n",
            },
            "",
            "",
            "",
            "",
            MatMulOptions {
                tile: MatMulTile::Large,
                knobs: MatmulKnobs {
                    k_stage: 16,
                    ..Default::default()
                },
                ..Default::default()
            },
            splits,
            1,
        );
    }
    // Transform matrices from Lavin and Gray, CVPR 2016, F(4x4,3x3).
    let bt: &[&[f64]] = &[
        &[4., 0., -5., 0., 1., 0.],
        &[0., -4., -4., 1., 1., 0.],
        &[0., 4., -4., -1., 1., 0.],
        &[0., -2., -1., 2., 1., 0.],
        &[0., 2., -1., -2., 1., 0.],
        &[0., 4., 0., -5., 0., 1.],
    ];
    let a: &[&[f64]] = &[
        &[1., 0., 0., 0.],
        &[1., 1., 1., 1.],
        &[1., -1., 1., -1.],
        &[1., 2., 4., 8.],
        &[1., -2., 4., -8.],
        &[0., 0., 0., 1.],
    ];
    let gt: &[&[f64]] = &[
        &[0.25, -1. / 6., -1. / 6., 1. / 24., 1. / 24., 0.],
        &[0., -1. / 6., 1. / 6., 1. / 12., -1. / 12., 0.],
        &[0., -1. / 6., -1. / 6., 1. / 6., 1. / 6., 1.],
    ];
    let output = group == ShaderGroup::WinogradGradOutputTransform;
    let matrix = match group {
        ShaderGroup::WinogradGradInputTransform => bt,
        ShaderGroup::WinogradGradDeltaTransform => a,
        ShaderGroup::WinogradGradOutputTransform => gt,
        _ => unreachable!(),
    };
    let size = matrix[0].len();
    let mut body = String::new();
    if output {
        body.push_str(
            "let channels = params.channels * params.h;\nif idx >= channels { return; }\n",
        );
        for i in 0..36 {
            body.push_str(&format!("var d{i} = 0.0;\n"));
        }
        body.push_str("for (var s = 0u; s < params.splits; s++) {\n");
        for i in 0..36 {
            body.push_str(&format!(
                "d{i} += src[(s * 36u + {i}u) * channels + idx];\n"
            ));
        }
        body.push_str("}\n");
    } else {
        body.push_str(
            "if idx >= params.channels * params.total_tiles { return; }\n\
            let channel = idx / params.total_tiles;\n\
            let tile = idx % params.total_tiles;\n\
            let per_image = params.tiles_h * params.tiles_w;\n\
            let image = tile / per_image;\n\
            let local = tile % per_image;\n\
            let h0 = i32(local / params.tiles_w * 4u) - i32(params.padding);\n\
            let w0 = i32(local % params.tiles_w * 4u) - i32(params.padding);\n\
            let base = (image * params.channels + channel) * params.h * params.w;\n",
        );
        for i in 0..size {
            for j in 0..size {
                let index = i * size + j;
                body.push_str(&format!("var d{index} = 0.0;\n\
                if h0 + {i} >= 0 && h0 + {i} < i32(params.h) && w0 + {j} >= 0 && w0 + {j} < i32(params.w) {{\n\
                d{index} = src[base + u32(h0 + {i}) * params.w + u32(w0 + {j})];\n}}\n"));
            }
        }
    }
    fn expression(row: &[f64], input: impl Fn(usize) -> String) -> String {
        let terms: Vec<_> = row
            .iter()
            .enumerate()
            .filter(|&(_, c)| *c != 0.)
            .map(|(k, c)| format!("({c:.17} * {})", input(k)))
            .collect();
        if terms.is_empty() {
            "0.0".into()
        } else {
            terms.join(" + ")
        }
    }
    for (i, row) in matrix.iter().enumerate() {
        for j in 0..size {
            let value = expression(row, |k| format!("d{}", k * size + j));
            body.push_str(&format!("let t{i}_{j} = {value};\n"));
        }
    }
    for i in 0..matrix.len() {
        for (j, row) in matrix.iter().enumerate() {
            let value = expression(row, |k| format!("t{i}_{k}"));
            let index = i * matrix.len() + j;
            let address = if output {
                format!("idx * 9u + {index}u")
            } else {
                format!("{index}u * params.channels * params.total_tiles + idx")
            };
            body.push_str(&format!("dst[{address}] = {value};\n"));
        }
    }
    ShaderModule::new(&format!(
        "\
        struct Params {{ channels:u32, h:u32, w:u32, padding:u32, tiles_h:u32, tiles_w:u32, total_tiles:u32, splits:u32 }}\n\
        var<storage> src:array<f32>;\n\
        var<storage,read_write> dst:array<f32>;\n\
        var<uniform> params:Params;\n\
        @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {{ let idx=gid.x;\n{body}\n}}"
    ))
}
