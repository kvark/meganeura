use super::{MatrixTypes, ShaderGroup, ShaderModule, preprocess};
use crate::DType;
#[path = "cooperative/convolution_reuse_words.rs"]
mod staging;

/// Fixed bounded output reuse; arithmetic partitions are unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ConvolutionReuseShape {
    pub rows: u32,
    pub columns: u32,
}

impl ConvolutionReuseShape {
    pub fn for_dimensions(m: u32, n: u32) -> Option<Self> {
        fn gcd(mut a: u32, mut b: u32) -> u32 {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        }
        if m == 0 || n == 0 {
            return None;
        }
        let mp = m.checked_add(31)? / 32 * 32;
        let np = n.checked_add(31)? / 32 * 32;
        let rows = gcd(mp, 64);
        let columns = gcd(np, 4096 / rows);
        (rows * columns > 1024).then_some(Self { rows, columns })
    }

    pub fn shared_bytes(self, multiply: DType) -> u32 {
        let (bytes, padding) = if multiply == DType::F32 {
            (4, 4)
        } else {
            (2, 8)
        };
        (self.rows * (32 + padding) + 32 * (self.columns + padding)) * bytes + 4096
    }

    pub fn valid(self) -> bool {
        matches!(self.rows, 32 | 64)
            && matches!(self.columns, 32 | 64 | 128)
            && self.rows * self.columns <= 4096
    }
}

pub(super) fn generate(
    group: ShaderGroup,
    types: MatrixTypes,
    shape: ConvolutionReuseShape,
    declaration: &str,
    dimensions: &str,
    range: &str,
) -> ShaderModule {
    assert!(shape.valid());
    let dw = matches!(
        group,
        ShaderGroup::Conv2dGradientTyped {
            weight_gradient: true,
            ..
        }
    );
    let tile_rows = shape.rows;
    let tile_columns = shape.columns;
    let staged = staging::generate(group, types, tile_rows, tile_columns);
    let mut extensions = String::from("enable wgpu_cooperative_matrix;\n");
    let fields = [types.a, types.b, types.multiply, types.output];
    if fields.contains(&DType::F16) {
        extensions.push_str("enable f16;\n");
    }
    if fields.contains(&DType::BF16) {
        extensions.push_str("enable wgpu_bfloat16;\n");
    }
    let rounding = if fields.contains(&DType::BF16) {
        include_str!("../shaders/bf16_round.wgsl")
    } else {
        ""
    };
    let conversion = if types.output == DType::BF16 {
        "round_bf16"
    } else {
        MatrixTypes::name(types.output)
    };
    let mut init = String::new();
    let mut compensate = String::new();
    let mut store = String::new();
    for i in 0..tile_rows / 32 {
        for j in 0..tile_columns / 32 {
            let acc = format!("acc{i}_{j}");
            init += &format!("var {acc}=coop_mat16x16<f32,C>();\n");
            if dw {
                init += &format!(
                    "var total_{acc}=coop_mat16x16<f32,C>();\nvar error_{acc}=coop_mat16x16<f32,C>();\n"
                );
                compensate += &format!(
                    "let corrected_{acc}={acc}-error_{acc};\nlet sum_{acc}=total_{acc}+corrected_{acc};\nerror_{acc}=(sum_{acc}-total_{acc})-corrected_{acc};\ntotal_{acc}=sum_{acc};\n{acc}=coop_mat16x16<f32,C>();\n"
                );
            }
            let value = if dw { format!("total_{acc}") } else { acc };
            store += &format!(
                "coopStoreT({value},&result[subgroup*256u],16u);\nworkgroupBarrier();\n\
                for(var offset=lane;offset<1024u;offset+=128u) {{\n\
                    let slot=offset/256u;\n\
                    let r=row+slot/2u*{}u+{}u+(offset%256u)/16u;\n\
                    let c=col+slot%2u*{}u+{}u+offset%16u;\n\
                    if r<m && c<n {{ dst[(group.z*m+r)*n+c]={conversion}(result[offset]); }}\n\
                }}\nworkgroupBarrier();\n",
                tile_rows / 2,
                i * 16,
                tile_columns / 2,
                j * 16
            );
        }
    }
    if dw {
        compensate =
            format!("if start+32u>=end || ((start+32u)-first)%256u==0u {{\n{compensate}}}\n");
    }
    let source = format!(
        "{extensions}\n{declaration}\n{}\n{}\nvar<workgroup> result:array<f32,1024>;\n{rounding}\n\
        @compute @workgroup_size(128)\n\
        fn main(@builtin(workgroup_id) group:vec3<u32>,\n\
            @builtin(local_invocation_index) lane:u32,@builtin(subgroup_id) subgroup:u32) {{\n\
            {dimensions}\n{range}\nlet row=group.x*{tile_rows}u; let col=group.y*{tile_columns}u;\n\
            {init}\n{}\n\
            for(var start=first;start<end;start+=32u) {{\n\
                {}\nworkgroupBarrier();\n{}\n{compensate}\n{}\nworkgroupBarrier();\n\
            }}\n{store}\n}}\n",
        include_str!("../shaders/conv_tile_address.wgsl"),
        staged.declaration,
        staged.initialize,
        staged.load,
        staged.multiply,
        staged.advance
    );
    let cast = |storage| {
        if types.multiply == DType::BF16 && storage == DType::F32 {
            "round_bf16"
        } else {
            MatrixTypes::name(types.multiply)
        }
    };
    ShaderModule::new(&preprocess(
        &source,
        &[
            ("$A_TYPE", MatrixTypes::name(types.a)),
            ("$B_TYPE", MatrixTypes::name(types.b)),
            ("$A_STORAGE", MatrixTypes::name(types.a)),
            ("$B_STORAGE", MatrixTypes::name(types.b)),
            ("$OUT_TYPE", MatrixTypes::name(types.output)),
            ("$MULTIPLY_TYPE", MatrixTypes::name(types.multiply)),
            ("$A_CAST", cast(types.a)),
            ("$B_CAST", cast(types.b)),
        ],
    ))
}
