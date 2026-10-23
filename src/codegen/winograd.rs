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
