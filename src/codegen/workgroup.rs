use super::{MatMulCoopVariant, ShaderGroup, ShaderModule, coop_shape};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct WorkgroupMatmulShape {
    pub rows: u32,
    pub columns: u32,
    pub k_stage: u32,
    pub threads: u32,
}

impl WorkgroupMatmulShape {
    pub fn supports(self, caps: &blade_graphics::Capabilities, bf16: bool) -> bool {
        use blade_graphics::CooperativeMatrixOperand;
        let matrix = &caps.cooperative_matrix.workgroup;
        let operand = if bf16 {
            CooperativeMatrixOperand::Bf16
        } else {
            CooperativeMatrixOperand::F16
        };
        [self.rows, self.columns, self.k_stage]
            .iter()
            .all(|s| matches!(s, 16 | 32 | 64 | 128))
            && matrix.supports(
                operand,
                [self.rows, self.columns, self.k_stage],
                self.threads,
            )
            && self
                .shared_bytes(bf16)
                .checked_add(matrix.reserved_shared_memory)
                .is_some_and(|bytes| bytes <= caps.max_compute_shared_memory_size)
    }

    pub fn fits(self, m: u32, n: u32, k: u32) -> bool {
        self.rows != 0
            && self.columns != 0
            && self.k_stage != 0
            && m != 0
            && n != 0
            && k != 0
            && m.is_multiple_of(self.rows)
            && n.is_multiple_of(self.columns)
            && k.is_multiple_of(self.k_stage)
    }

    pub fn shared_bytes(self, bf16: bool) -> u32 {
        if bf16 {
            0
        } else {
            2 * self.k_stage * (self.rows + self.columns)
        }
    }
}

/// Aligned, unfused workgroup GEMM. F32 inputs are staged as F16; BF16 loads directly.
pub fn generate_workgroup_matmul(
    group: ShaderGroup,
    shape: WorkgroupMatmulShape,
    bf16: bool,
) -> ShaderModule {
    let (addend, variant) = coop_shape(group).expect("workgroup matmul group");
    assert!(!addend);
    let WorkgroupMatmulShape {
        rows: m,
        columns: n,
        k_stage: k,
        threads,
    } = shape;
    let mut source = String::from("enable wgpu_cooperative_matrix;\n");
    source += if bf16 {
        "enable wgpu_bfloat16;\n"
    } else {
        "enable f16;\n"
    };
    let storage = if bf16 { "bf16" } else { "vec4<f32>" };
    source += &format!(
        "struct Params {{ m: u32, n: u32, k: u32, pad: u32 }}\nvar<uniform> params: Params;\nvar<storage> matrix_a: array<{storage}>;\nvar<storage> matrix_b: array<{storage}>;\nvar<storage, read_write> matrix_c: array<f32>;\n"
    );
    if !bf16 {
        source += &format!(
            "var<workgroup> sa: array<f16, {}>;\nvar<workgroup> sb: array<f16, {}>;\n",
            m * k,
            n * k
        );
    }
    source += &format!(
        "@compute @workgroup_size({threads})\nfn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {{\nlet m = params.m;\nlet n = params.n;\nlet k = params.k;\nlet row = group.x * {m}u;\nlet col = group.y * {n}u;\nvar acc = coop_mat{n}x{m}<f32,C,workgroup>();\nfor (var t = 0u; t < k; t += {k}u) {{\n"
    );
    if bf16 {
        let (load_a, index_a, stride_a) = if variant == MatMulCoopVariant::AT {
            ("coopLoad", "t * m + row", "m")
        } else {
            ("coopLoadT", "row * k + t", "k")
        };
        let (load_b, index_b, stride_b) = if variant == MatMulCoopVariant::BT {
            ("coopLoad", "col * k + t", "k")
        } else {
            ("coopLoadT", "t * n + col", "n")
        };
        source += &format!(
            "let a = {load_a}<coop_mat{k}x{m}<bf16,A,workgroup>>(&matrix_a[{index_a}], {stride_a});\nlet b = {load_b}<coop_mat{n}x{k}<bf16,B,workgroup>>(&matrix_b[{index_b}], {stride_b});\n"
        );
    } else {
        for a in [true, false] {
            let transpose = if a {
                variant == MatMulCoopVariant::AT
            } else {
                variant == MatMulCoopVariant::BT
            };
            let (name, shared, rows, cols) = if a {
                ("matrix_a", "sa", m, k)
            } else {
                ("matrix_b", "sb", k, n)
            };
            let width = if transpose { rows } else { cols };
            let index = match (a, transpose) {
                (true, false) => "(row + r) * k + t + c",
                (true, true) => "(t + r) * m + row + c",
                (false, false) => "(t + r) * n + col + c",
                (false, true) => "(col + r) * k + t + c",
            };
            source += &format!(
                "for (var i = lid; i < {}u; i += {threads}u) {{\nlet r = i / {}u;\nlet c = (i % {}u) * 4u;\nlet v = {name}[({index}) / 4u];\n",
                rows * cols / 4,
                width / 4,
                width / 4
            );
            for (lane, field) in ["x", "y", "z", "w"].iter().enumerate() {
                let index = if transpose {
                    format!("(c + {lane}u) * {cols}u + r")
                } else {
                    format!("r * {cols}u + c + {lane}u")
                };
                source += &format!("{shared}[{index}] = f16(v.{field});\n");
            }
            source += "}\n";
        }
        source += &format!(
            "workgroupBarrier();\nlet a = coopLoadT<coop_mat{k}x{m}<f16,A,workgroup>>(&sa[0], {k}u);\nlet b = coopLoadT<coop_mat{n}x{k}<f16,B,workgroup>>(&sb[0], {n}u);\n"
        );
    }
    source += "acc = coopMultiplyAdd(a,b,acc);\n";
    if !bf16 {
        source += "workgroupBarrier();\n";
    }
    source += "}\ncoopStoreT(acc, &matrix_c[row * n + col], n);\n}\n";
    ShaderModule::with_hint(source, "workgroup-matmul")
}
