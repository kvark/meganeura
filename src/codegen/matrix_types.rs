use crate::DType;

/// Operand storage, multiplication and output types of a checked cooperative
/// product. Accumulation is always F32, and K partitions store F32 partials
/// until their reduction, so `output` applies only to an unpartitioned product.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MatrixTypes {
    pub a: DType,
    pub b: DType,
    pub multiply: DType,
    pub output: DType,
}

impl MatrixTypes {
    /// Automatic accelerated F32 storage with F16 multiplication.
    pub const F32_AS_F16: Self = Self {
        a: DType::F32,
        b: DType::F32,
        multiply: DType::F16,
        output: DType::F32,
    };
    /// Explicit BF16 operands with an F32 result.
    pub const BF16: Self = Self::reduced(DType::BF16);

    /// Both operands stored and multiplied as `storage`, with an F32 result.
    pub const fn reduced(storage: DType) -> Self {
        Self {
            a: storage,
            b: storage,
            multiply: storage,
            output: DType::F32,
        }
    }

    /// Whether the checked kernel can express these types.
    pub fn is_valid(self) -> bool {
        let floating = |ty| matches!(ty, DType::F32 | DType::F16 | DType::BF16);
        [self.a, self.b, self.multiply, self.output]
            .into_iter()
            .all(floating)
            // F16 and BF16 are distinct formats: an operand is stored as the
            // multiplication type, or one of the two is F32.
            && [self.a, self.b]
                .into_iter()
                .all(|ty| ty == self.multiply || ty == DType::F32 || self.multiply == DType::F32)
    }

    pub fn validate(self) {
        assert!(
            self.is_valid(),
            "unsupported cooperative matrix types {self:?}"
        );
    }

    /// Whether any field stores or multiplies `ty`.
    pub fn uses(self, ty: DType) -> bool {
        [self.a, self.b, self.multiply, self.output].contains(&ty)
    }

    /// The WGSL scalar type.
    pub(crate) fn name(ty: DType) -> &'static str {
        match ty {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            _ => panic!("non-floating matrix type {ty:?}"),
        }
    }

    /// Bytes per stored element.
    pub fn bytes(ty: DType) -> usize {
        match ty {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            _ => panic!("non-floating matrix type {ty:?}"),
        }
    }
}
