//! Shader codegen via Naga IR.
//!
//! Builds `naga::Module` objects programmatically for each [`ShaderGroup`].
//! Modules are passed directly to blade via `naga_module` for SPIR-V
//! compilation. Emit ranges must be correct for all expressions —
//! the SPIR-V backend panics on uncached expressions.

use naga::*;

const S: Span = Span::UNDEFINED;

// ---------------------------------------------------------------------------
// Builder: wraps a naga::Module and provides convenience methods
// ---------------------------------------------------------------------------

struct Builder {
    m: Module,
    // Cached types
    ty_f32: Handle<Type>,
    ty_u32: Handle<Type>,
    ty_vec3u: Handle<Type>,
    ty_arr_f32: Handle<Type>,
}

impl Builder {
    fn new() -> Self {
        let mut m = Module::default();

        let ty_f32 = m.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::F32),
            },
            S,
        );
        let ty_u32 = m.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::U32),
            },
            S,
        );
        let ty_vec3u = m.types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector {
                    size: VectorSize::Tri,
                    scalar: Scalar::U32,
                },
            },
            S,
        );
        let ty_arr_f32 = m.types.insert(
            Type {
                name: None,
                inner: TypeInner::Array {
                    base: ty_f32,
                    size: ArraySize::Dynamic,
                    stride: 4,
                },
            },
            S,
        );

        Self {
            m,
            ty_f32,
            ty_u32,
            ty_vec3u,
            ty_arr_f32,
        }
    }

    /// Create a Params struct type with named u32 fields (4 fields, 16 bytes).
    fn params_u32x4(&mut self, name: &str, fields: &[&str]) -> Handle<Type> {
        let members: Vec<StructMember> = fields
            .iter()
            .enumerate()
            .map(|(i, n)| StructMember {
                name: Some(n.to_string()),
                ty: self.ty_u32,
                binding: None,
                offset: (i as u32) * 4,
            })
            .collect();
        self.m.types.insert(
            Type {
                name: Some(name.to_string()),
                inner: TypeInner::Struct { members, span: 16 },
            },
            S,
        )
    }

    /// Create a Params struct where the second field is f32 (for SGD lr).
    fn params_u32_f32_u32_u32(&mut self, name: &str, fields: &[&str]) -> Handle<Type> {
        assert_eq!(fields.len(), 4);
        let tys = [self.ty_u32, self.ty_f32, self.ty_u32, self.ty_u32];
        let members: Vec<StructMember> = fields
            .iter()
            .enumerate()
            .map(|(i, n)| StructMember {
                name: Some(n.to_string()),
                ty: tys[i],
                binding: None,
                offset: (i as u32) * 4,
            })
            .collect();
        self.m.types.insert(
            Type {
                name: Some(name.to_string()),
                inner: TypeInner::Struct { members, span: 16 },
            },
            S,
        )
    }

    /// Add a storage buffer global variable (read-only).
    fn storage_ro(&mut self, name: &str) -> Handle<GlobalVariable> {
        self.m.global_variables.append(
            GlobalVariable {
                name: Some(name.to_string()),
                space: AddressSpace::Storage {
                    access: StorageAccess::LOAD,
                },
                binding: None,
                ty: self.ty_arr_f32,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            S,
        )
    }

    /// Add a storage buffer global variable (read-write).
    fn storage_rw(&mut self, name: &str) -> Handle<GlobalVariable> {
        self.m.global_variables.append(
            GlobalVariable {
                name: Some(name.to_string()),
                space: AddressSpace::Storage {
                    access: StorageAccess::LOAD | StorageAccess::STORE,
                },
                binding: None,
                ty: self.ty_arr_f32,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            S,
        )
    }

    /// Add a uniform buffer global variable.
    fn uniform(&mut self, name: &str, ty: Handle<Type>) -> Handle<GlobalVariable> {
        self.m.global_variables.append(
            GlobalVariable {
                name: Some(name.to_string()),
                space: AddressSpace::Uniform,
                binding: None,
                ty,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            S,
        )
    }

    /// Add a workgroup variable (array<f32, N>).
    fn workgroup_array(&mut self, name: &str, count: u32) -> Handle<GlobalVariable> {
        let ty = self.m.types.insert(
            Type {
                name: None,
                inner: TypeInner::Array {
                    base: self.ty_f32,
                    size: ArraySize::Constant(std::num::NonZeroU32::new(count).unwrap()),
                    stride: 4,
                },
            },
            S,
        );
        self.m.global_variables.append(
            GlobalVariable {
                name: Some(name.to_string()),
                space: AddressSpace::WorkGroup,
                binding: None,
                ty,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            S,
        )
    }

    /// Add a compute entry point.
    fn entry_point(&mut self, name: &str, workgroup_size: [u32; 3], func: Function) {
        self.m.entry_points.push(EntryPoint {
            name: name.to_string(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size,
            workgroup_size_overrides: None,
            function: func,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });
    }

    /// Add a storage buffer global variable for u32 array (read-only).
    fn storage_ro_u32(&mut self) -> Handle<GlobalVariable> {
        let ty_arr_u32 = self.m.types.insert(
            Type {
                name: None,
                inner: TypeInner::Array {
                    base: self.ty_u32,
                    size: ArraySize::Dynamic,
                    stride: 4,
                },
            },
            S,
        );
        self.m.global_variables.append(
            GlobalVariable {
                name: Some("indices".to_string()),
                space: AddressSpace::Storage {
                    access: StorageAccess::LOAD,
                },
                binding: None,
                ty: ty_arr_u32,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            S,
        )
    }

    fn finish(self) -> Module {
        self.m
    }
}

// ---------------------------------------------------------------------------
// FnBuilder: builds a single function body
// ---------------------------------------------------------------------------

struct FnBuilder {
    f: Function,
    #[allow(dead_code)]
    ty_f32: Handle<Type>,
    #[allow(dead_code)]
    ty_u32: Handle<Type>,
    ty_vec3u: Handle<Type>,
}

impl FnBuilder {
    fn new(b: &Builder) -> Self {
        Self {
            f: Function::default(),
            ty_f32: b.ty_f32,
            ty_u32: b.ty_u32,
            ty_vec3u: b.ty_vec3u,
        }
    }

    /// Add the gid argument: @builtin(global_invocation_id) gid: vec3<u32>
    fn arg_gid(&mut self) -> Handle<Expression> {
        self.f.arguments.push(FunctionArgument {
            name: Some("gid".to_string()),
            ty: self.ty_vec3u,
            binding: Some(Binding::BuiltIn(BuiltIn::GlobalInvocationId)),
        });
        self.expr(Expression::FunctionArgument(
            self.f.arguments.len() as u32 - 1,
        ))
    }

    /// Add the lid argument: @builtin(local_invocation_id) lid: vec3<u32>
    fn arg_lid(&mut self) -> Handle<Expression> {
        self.f.arguments.push(FunctionArgument {
            name: Some("lid".to_string()),
            ty: self.ty_vec3u,
            binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationId)),
        });
        self.expr(Expression::FunctionArgument(
            self.f.arguments.len() as u32 - 1,
        ))
    }

    /// Add the wgid argument: @builtin(workgroup_id) wgid: vec3<u32>
    #[allow(dead_code)]
    fn arg_wgid(&mut self) -> Handle<Expression> {
        self.f.arguments.push(FunctionArgument {
            name: Some("wgid".to_string()),
            ty: self.ty_vec3u,
            binding: Some(Binding::BuiltIn(BuiltIn::WorkGroupId)),
        });
        self.expr(Expression::FunctionArgument(
            self.f.arguments.len() as u32 - 1,
        ))
    }

    fn expr(&mut self, e: Expression) -> Handle<Expression> {
        self.f.expressions.append(e, S)
    }

    fn named(&mut self, name: &str, e: Expression) -> Handle<Expression> {
        let h = self.f.expressions.append(e, S);
        self.f.named_expressions.insert(h, name.to_string());
        h
    }

    /// Give a name to an existing expression handle.
    fn label(&mut self, name: &str, h: Handle<Expression>) -> Handle<Expression> {
        self.f.named_expressions.insert(h, name.to_string());
        h
    }

    /// Push Emit statements to the function body, skipping pre-emitted
    /// expressions (GlobalVariable, Literal, etc.).
    fn emit(&mut self, first: Handle<Expression>, last: Handle<Expression>) {
        push_emit(&self.f.expressions, &mut self.f.body, first, last);
    }

    /// GlobalVariable pointer expression.
    fn global(&mut self, gv: Handle<GlobalVariable>) -> Handle<Expression> {
        self.expr(Expression::GlobalVariable(gv))
    }

    /// Extract .x from a vec3 value (produces a value, not a pointer).
    fn vec_x(&mut self, vec: Handle<Expression>) -> Handle<Expression> {
        self.expr(Expression::AccessIndex {
            base: vec,
            index: 0,
        })
    }

    /// Extract .y from a vec3 value (produces a value, not a pointer).
    fn vec_y(&mut self, vec: Handle<Expression>) -> Handle<Expression> {
        self.expr(Expression::AccessIndex {
            base: vec,
            index: 1,
        })
    }

    /// Extract .z from a vec3 value (produces a value, not a pointer).
    #[allow(dead_code)]
    fn vec_z(&mut self, vec: Handle<Expression>) -> Handle<Expression> {
        self.expr(Expression::AccessIndex {
            base: vec,
            index: 2,
        })
    }

    /// AccessIndex on a struct or array
    fn field(&mut self, base: Handle<Expression>, index: u32) -> Handle<Expression> {
        self.expr(Expression::AccessIndex { base, index })
    }

    /// Dynamic array access: base[index]
    fn index(&mut self, base: Handle<Expression>, idx: Handle<Expression>) -> Handle<Expression> {
        self.expr(Expression::Access { base, index: idx })
    }

    fn load(&mut self, ptr: Handle<Expression>) -> Handle<Expression> {
        self.expr(Expression::Load { pointer: ptr })
    }

    fn literal_f32(&mut self, v: f32) -> Handle<Expression> {
        self.expr(Expression::Literal(Literal::F32(v)))
    }

    fn literal_u32(&mut self, v: u32) -> Handle<Expression> {
        self.expr(Expression::Literal(Literal::U32(v)))
    }

    fn binary(
        &mut self,
        op: BinaryOperator,
        l: Handle<Expression>,
        r: Handle<Expression>,
    ) -> Handle<Expression> {
        self.expr(Expression::Binary {
            op,
            left: l,
            right: r,
        })
    }

    fn unary(&mut self, op: UnaryOperator, e: Handle<Expression>) -> Handle<Expression> {
        self.expr(Expression::Unary { op, expr: e })
    }

    fn math1(&mut self, fun: MathFunction, arg: Handle<Expression>) -> Handle<Expression> {
        self.expr(Expression::Math {
            fun,
            arg,
            arg1: None,
            arg2: None,
            arg3: None,
        })
    }

    fn math2(
        &mut self,
        fun: MathFunction,
        arg: Handle<Expression>,
        arg1: Handle<Expression>,
    ) -> Handle<Expression> {
        self.expr(Expression::Math {
            fun,
            arg,
            arg1: Some(arg1),
            arg2: None,
            arg3: None,
        })
    }

    fn select(
        &mut self,
        cond: Handle<Expression>,
        accept: Handle<Expression>,
        reject: Handle<Expression>,
    ) -> Handle<Expression> {
        self.expr(Expression::Select {
            condition: cond,
            accept,
            reject,
        })
    }

    fn cast_f32(&mut self, e: Handle<Expression>) -> Handle<Expression> {
        self.expr(Expression::As {
            expr: e,
            kind: ScalarKind::Float,
            convert: Some(4),
        })
    }

    fn local_var(
        &mut self,
        name: &str,
        ty: Handle<Type>,
        init: Option<Handle<Expression>>,
    ) -> Handle<LocalVariable> {
        self.f.local_variables.append(
            LocalVariable {
                name: Some(name.to_string()),
                ty,
                init,
            },
            S,
        )
    }

    fn local_ptr(&mut self, lv: Handle<LocalVariable>) -> Handle<Expression> {
        self.expr(Expression::LocalVariable(lv))
    }

    fn store(&self, ptr: Handle<Expression>, value: Handle<Expression>) -> Statement {
        Statement::Store {
            pointer: ptr,
            value,
        }
    }

    fn if_return(&self, cond: Handle<Expression>) -> Statement {
        Statement::If {
            condition: cond,
            accept: Block::from_vec(vec![Statement::Return { value: None }]),
            reject: Block::new(),
        }
    }

    fn if_break(cond: Handle<Expression>) -> Statement {
        Statement::If {
            condition: cond,
            accept: Block::from_vec(vec![Statement::Break]),
            reject: Block::new(),
        }
    }

    fn finish(self) -> Function {
        self.f
    }
}

/// Push Emit statements to a block for a range of expressions,
/// automatically splitting around pre-emitted expressions.
fn push_emit(
    exprs: &Arena<Expression>,
    block: &mut Block,
    first: Handle<Expression>,
    last: Handle<Expression>,
) {
    let first_idx = first.index();
    let last_idx = last.index();
    let mut run_start: Option<Handle<Expression>> = None;
    let mut prev_h: Option<Handle<Expression>> = None;

    for (h, expr) in exprs.iter().skip(first_idx).take(last_idx - first_idx + 1) {
        if expr.needs_pre_emit() {
            if let (Some(start), Some(end)) = (run_start, prev_h) {
                block.push(Statement::Emit(Range::new_from_bounds(start, end)), S);
            }
            run_start = None;
            prev_h = None;
        } else {
            if run_start.is_none() {
                run_start = Some(h);
            }
            prev_h = Some(h);
        }
    }
    if let (Some(start), Some(end)) = (run_start, prev_h) {
        block.push(Statement::Emit(Range::new_from_bounds(start, end)), S);
    }
}

// ---------------------------------------------------------------------------
// Shader groups — each group is a naga::Module with one or more entry points
// ---------------------------------------------------------------------------

/// A shader group corresponds to a single `naga::Module` that may
/// contain multiple entry points (e.g. `Unary` has relu, sigmoid, neg).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderGroup {
    Unary,
    Binary,
    BiasAdd,
    Sgd,
    Transpose,
    MatMul,
    MatMulAdd,
    MatMulCoop,
    MatMulCoopAdd,
    Reduce,
    Softmax,
    CrossEntropy,
    RmsNorm,
    Embedding,
    RoPE,
    CausalAttention,
    LayerNorm,
    FullAttention,
    CrossAttention,
    MultiHeadAttn,
    MultiHeadAttnGradQ,
    MultiHeadAttnGradK,
    MultiHeadAttnGradV,
}

/// Generate a `naga::Module` for a shader group.
pub fn generate_module(group: ShaderGroup) -> Module {
    match group {
        ShaderGroup::Unary => gen_unary(),
        ShaderGroup::Binary => gen_binary(),
        ShaderGroup::BiasAdd => gen_bias_add(),
        ShaderGroup::Sgd => gen_sgd(),
        ShaderGroup::Transpose => gen_transpose(),
        ShaderGroup::MatMul => gen_matmul(),
        ShaderGroup::MatMulAdd => gen_matmul_add(),
        ShaderGroup::MatMulCoop => gen_matmul_coop(),
        ShaderGroup::MatMulCoopAdd => gen_matmul_coop_add(),
        ShaderGroup::Reduce => gen_reduce(),
        ShaderGroup::Softmax => gen_softmax(),
        ShaderGroup::CrossEntropy => gen_cross_entropy(),
        ShaderGroup::RmsNorm => gen_rms_norm(),
        ShaderGroup::Embedding => gen_embedding(),
        ShaderGroup::RoPE => gen_rope(),
        ShaderGroup::CausalAttention => gen_causal_attention(),
        ShaderGroup::LayerNorm => gen_layer_norm(),
        ShaderGroup::FullAttention => gen_full_attention(),
        ShaderGroup::CrossAttention => gen_cross_attention(),
        ShaderGroup::MultiHeadAttn => gen_mha_forward(),
        ShaderGroup::MultiHeadAttnGradQ => gen_mha_grad_q(),
        ShaderGroup::MultiHeadAttnGradK => gen_mha_grad_k(),
        ShaderGroup::MultiHeadAttnGradV => gen_mha_grad_v(),
    }
}

/// Generate WGSL source for a shader group.
pub fn generate_wgsl(group: ShaderGroup) -> String {
    let module = generate_module(group);
    let capabilities = match group {
        ShaderGroup::MatMulCoop | ShaderGroup::MatMulCoopAdd => {
            naga::valid::Capabilities::COOPERATIVE_MATRIX
                | naga::valid::Capabilities::SHADER_FLOAT16
        }
        _ => naga::valid::Capabilities::empty(),
    };
    module_to_wgsl(&module, capabilities)
}

/// Convert a naga Module to WGSL source text.
pub fn module_to_wgsl(module: &Module, capabilities: naga::valid::Capabilities) -> String {
    let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
    let info = naga::valid::Validator::new(flags, capabilities)
        .validate(module)
        .expect("generated module failed validation");
    naga::back::wgsl::write_string(module, &info, naga::back::wgsl::WriterFlags::empty())
        .expect("WGSL write failed")
}

// ---------------------------------------------------------------------------
// unary.wgsl: relu, sigmoid, neg
// ---------------------------------------------------------------------------

fn gen_unary() -> Module {
    let mut b = Builder::new();
    let ty_params = b.params_u32x4("Params", &["len", "_pad0", "_pad1", "_pad2"]);
    let gv_src = b.storage_ro("src");
    let gv_dst = b.storage_rw("dst");
    let gv_params = b.uniform("params", ty_params);

    // relu
    {
        let mut f = FnBuilder::new(&b);
        let gid = f.arg_gid();

        // let i = gid.x;
        let i = f.vec_x(gid);
        f.label("i", i);

        let params_ptr = f.global(gv_params);
        let len_ptr = f.field(params_ptr, 0);
        let len = f.load(len_ptr);
        let cond = f.binary(BinaryOperator::GreaterEqual, i, len);
        let src_ptr = f.global(gv_src);
        let elem_ptr = f.index(src_ptr, i);
        let val = f.load(elem_ptr);
        let zero = f.literal_f32(0.0);
        let result = f.math2(MathFunction::Max, val, zero);
        let dst_ptr = f.global(gv_dst);
        let dst_elem = f.index(dst_ptr, i);

        f.emit(i, i);
        f.emit(len_ptr, cond);
        f.f.body.push(f.if_return(cond), S);
        f.emit(src_ptr, result);
        f.emit(dst_elem, dst_elem);
        f.f.body.push(f.store(dst_elem, result), S);

        b.entry_point("relu", [256, 1, 1], f.finish());
    }

    // sigmoid
    {
        let mut f = FnBuilder::new(&b);
        let gid = f.arg_gid();
        let i = f.vec_x(gid);
        f.label("i", i);
        let params_ptr = f.global(gv_params);
        let len_ptr = f.field(params_ptr, 0);
        let len = f.load(len_ptr);
        let cond = f.binary(BinaryOperator::GreaterEqual, i, len);
        let src_ptr = f.global(gv_src);
        let elem_ptr = f.index(src_ptr, i);
        let val = f.load(elem_ptr);
        let neg_val = f.unary(UnaryOperator::Negate, val);
        let exp_neg = f.math1(MathFunction::Exp, neg_val);
        let one = f.literal_f32(1.0);
        let denom = f.binary(BinaryOperator::Add, one, exp_neg);
        let one2 = f.literal_f32(1.0);
        let result = f.binary(BinaryOperator::Divide, one2, denom);
        let dst_ptr = f.global(gv_dst);
        let dst_elem = f.index(dst_ptr, i);

        f.emit(i, i);
        f.emit(len_ptr, cond);
        f.f.body.push(f.if_return(cond), S);
        f.emit(src_ptr, result);
        f.emit(dst_elem, dst_elem);
        f.f.body.push(f.store(dst_elem, result), S);

        b.entry_point("sigmoid", [256, 1, 1], f.finish());
    }

    // neg
    {
        let mut f = FnBuilder::new(&b);
        let gid = f.arg_gid();
        let i = f.vec_x(gid);
        f.label("i", i);
        let params_ptr = f.global(gv_params);
        let len_ptr = f.field(params_ptr, 0);
        let len = f.load(len_ptr);
        let cond = f.binary(BinaryOperator::GreaterEqual, i, len);
        let src_ptr = f.global(gv_src);
        let elem_ptr = f.index(src_ptr, i);
        let val = f.load(elem_ptr);
        let result = f.unary(UnaryOperator::Negate, val);
        let dst_ptr = f.global(gv_dst);
        let dst_elem = f.index(dst_ptr, i);

        f.emit(i, i);
        f.emit(len_ptr, cond);
        f.f.body.push(f.if_return(cond), S);
        f.emit(src_ptr, result);
        f.emit(dst_elem, dst_elem);
        f.f.body.push(f.store(dst_elem, result), S);

        b.entry_point("neg", [256, 1, 1], f.finish());
    }

    // silu: x * sigmoid(x) = x / (1 + exp(-x))
    {
        let mut f = FnBuilder::new(&b);
        let gid = f.arg_gid();
        let i = f.vec_x(gid);
        f.label("i", i);
        let params_ptr = f.global(gv_params);
        let len_ptr = f.field(params_ptr, 0);
        let len = f.load(len_ptr);
        let cond = f.binary(BinaryOperator::GreaterEqual, i, len);
        let src_ptr = f.global(gv_src);
        let elem_ptr = f.index(src_ptr, i);
        let val = f.load(elem_ptr);
        let neg_val = f.unary(UnaryOperator::Negate, val);
        let exp_neg = f.math1(MathFunction::Exp, neg_val);
        let one = f.literal_f32(1.0);
        let denom = f.binary(BinaryOperator::Add, one, exp_neg);
        let one2 = f.literal_f32(1.0);
        let sigmoid = f.binary(BinaryOperator::Divide, one2, denom);
        let result = f.binary(BinaryOperator::Multiply, val, sigmoid);
        let dst_ptr = f.global(gv_dst);
        let dst_elem = f.index(dst_ptr, i);

        f.emit(i, i);
        f.emit(len_ptr, cond);
        f.f.body.push(f.if_return(cond), S);
        f.emit(src_ptr, result);
        f.emit(dst_elem, dst_elem);
        f.f.body.push(f.store(dst_elem, result), S);

        b.entry_point("silu", [256, 1, 1], f.finish());
    }

    // gelu: x * 0.5 * (1 + erf(x / sqrt(2)))
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    {
        let mut f = FnBuilder::new(&b);
        let gid = f.arg_gid();
        let i = f.vec_x(gid);
        f.label("i", i);
        let params_ptr = f.global(gv_params);
        let len_ptr = f.field(params_ptr, 0);
        let len = f.load(len_ptr);
        let cond = f.binary(BinaryOperator::GreaterEqual, i, len);
        let src_ptr = f.global(gv_src);
        let elem_ptr = f.index(src_ptr, i);
        let val = f.load(elem_ptr);

        // sqrt(2/pi) ≈ 0.7978845608
        let sqrt_2_pi = f.literal_f32(0.797_884_6);
        // 0.044715
        let coeff = f.literal_f32(0.044715);
        let half = f.literal_f32(0.5);
        let one = f.literal_f32(1.0);

        // x^3
        let x2 = f.binary(BinaryOperator::Multiply, val, val);
        let x3 = f.binary(BinaryOperator::Multiply, x2, val);
        // 0.044715 * x^3
        let cx3 = f.binary(BinaryOperator::Multiply, coeff, x3);
        // x + 0.044715 * x^3
        let inner = f.binary(BinaryOperator::Add, val, cx3);
        // sqrt(2/pi) * (x + 0.044715 * x^3)
        let scaled = f.binary(BinaryOperator::Multiply, sqrt_2_pi, inner);
        // tanh(...)
        let tanh_val = f.math1(MathFunction::Tanh, scaled);
        // 1 + tanh(...)
        let one_plus_tanh = f.binary(BinaryOperator::Add, one, tanh_val);
        // 0.5 * x
        let half_x = f.binary(BinaryOperator::Multiply, half, val);
        // 0.5 * x * (1 + tanh(...))
        let result = f.binary(BinaryOperator::Multiply, half_x, one_plus_tanh);

        let dst_ptr = f.global(gv_dst);
        let dst_elem = f.index(dst_ptr, i);

        f.emit(i, i);
        f.emit(len_ptr, cond);
        f.f.body.push(f.if_return(cond), S);
        f.emit(src_ptr, result);
        f.emit(dst_elem, dst_elem);
        f.f.body.push(f.store(dst_elem, result), S);

        b.entry_point("gelu", [256, 1, 1], f.finish());
    }

    b.finish()
}

// ---------------------------------------------------------------------------
// binary.wgsl: add, mul, greater
// ---------------------------------------------------------------------------

fn gen_binary() -> Module {
    let mut b = Builder::new();
    let ty_params = b.params_u32x4("Params", &["len", "_pad0", "_pad1", "_pad2"]);
    let gv_src_a = b.storage_ro("src_a");
    let gv_src_b = b.storage_ro("src_b");
    let gv_dst = b.storage_rw("dst");
    let gv_params = b.uniform("params", ty_params);

    fn binary_ep(
        b: &Builder,
        name: &str,
        gv_src_a: Handle<GlobalVariable>,
        gv_src_b: Handle<GlobalVariable>,
        gv_dst: Handle<GlobalVariable>,
        gv_params: Handle<GlobalVariable>,
        make_result: fn(
            &mut FnBuilder,
            Handle<Expression>,
            Handle<Expression>,
        ) -> Handle<Expression>,
    ) -> Function {
        let mut f = FnBuilder::new(b);
        let gid = f.arg_gid();
        let i = f.vec_x(gid);
        f.label("i", i);
        let params_ptr = f.global(gv_params);
        let len_ptr = f.field(params_ptr, 0);
        let len = f.load(len_ptr);
        let cond = f.binary(BinaryOperator::GreaterEqual, i, len);
        let src_a_ptr = f.global(gv_src_a);
        let a_ptr = f.index(src_a_ptr, i);
        let a_val = f.load(a_ptr);
        let src_b_ptr = f.global(gv_src_b);
        let b_ptr = f.index(src_b_ptr, i);
        let b_val = f.load(b_ptr);
        let result = make_result(&mut f, a_val, b_val);
        let dst_ptr = f.global(gv_dst);
        let dst_elem = f.index(dst_ptr, i);

        f.emit(i, i);
        f.emit(len_ptr, cond);
        f.f.body.push(f.if_return(cond), S);
        f.emit(src_a_ptr, result);
        f.emit(dst_elem, dst_elem);
        f.f.body.push(f.store(dst_elem, result), S);

        let _ = name;
        f.finish()
    }

    let func = binary_ep(
        &b,
        "add",
        gv_src_a,
        gv_src_b,
        gv_dst,
        gv_params,
        |f, a, bv| f.binary(BinaryOperator::Add, a, bv),
    );
    b.entry_point("add", [256, 1, 1], func);

    let func = binary_ep(
        &b,
        "mul",
        gv_src_a,
        gv_src_b,
        gv_dst,
        gv_params,
        |f, a, bv| f.binary(BinaryOperator::Multiply, a, bv),
    );
    b.entry_point("mul", [256, 1, 1], func);

    // greater: select(0.0, 1.0, a > b)
    let func = binary_ep(
        &b,
        "greater",
        gv_src_a,
        gv_src_b,
        gv_dst,
        gv_params,
        |f, a, bv| {
            let cmp = f.binary(BinaryOperator::Greater, a, bv);
            let one = f.literal_f32(1.0);
            let zero = f.literal_f32(0.0);
            f.select(cmp, one, zero)
        },
    );
    b.entry_point("greater", [256, 1, 1], func);

    // swiglu: silu(gate) * up = (gate / (1 + exp(-gate))) * up
    // src_a = gate, src_b = up
    let func = binary_ep(
        &b,
        "swiglu",
        gv_src_a,
        gv_src_b,
        gv_dst,
        gv_params,
        |f, gate, up| {
            let neg_gate = f.unary(UnaryOperator::Negate, gate);
            let exp_neg = f.math1(MathFunction::Exp, neg_gate);
            let one = f.literal_f32(1.0);
            let denom = f.binary(BinaryOperator::Add, one, exp_neg);
            let one2 = f.literal_f32(1.0);
            let sigmoid = f.binary(BinaryOperator::Divide, one2, denom);
            let silu_gate = f.binary(BinaryOperator::Multiply, gate, sigmoid);
            f.binary(BinaryOperator::Multiply, silu_gate, up)
        },
    );
    b.entry_point("swiglu", [256, 1, 1], func);

    b.finish()
}

// ---------------------------------------------------------------------------
// bias_add.wgsl
// ---------------------------------------------------------------------------

fn gen_bias_add() -> Module {
    let mut b = Builder::new();
    let ty_params = b.params_u32x4("Params", &["len", "bias_len", "_pad0", "_pad1"]);
    let gv_src = b.storage_ro("src");
    let gv_bias = b.storage_ro("bias");
    let gv_dst = b.storage_rw("dst");
    let gv_params = b.uniform("params", ty_params);

    let mut f = FnBuilder::new(&b);
    let gid = f.arg_gid();
    let i = f.vec_x(gid);
    f.label("i", i);
    let params_ptr = f.global(gv_params);
    let len_ptr = f.field(params_ptr, 0);
    let len = f.load(len_ptr);
    let cond = f.binary(BinaryOperator::GreaterEqual, i, len);

    let src_ptr = f.global(gv_src);
    let src_elem = f.index(src_ptr, i);
    let src_val = f.load(src_elem);

    let bias_len_ptr = f.field(params_ptr, 1);
    let bias_len = f.load(bias_len_ptr);
    let bias_idx = f.binary(BinaryOperator::Modulo, i, bias_len);
    let bias_ptr = f.global(gv_bias);
    let bias_elem = f.index(bias_ptr, bias_idx);
    let bias_val = f.load(bias_elem);

    let result = f.binary(BinaryOperator::Add, src_val, bias_val);
    let dst_ptr = f.global(gv_dst);
    let dst_elem = f.index(dst_ptr, i);

    f.emit(i, i);
    f.emit(len_ptr, cond);
    f.f.body.push(f.if_return(cond), S);
    f.emit(src_ptr, result);
    f.emit(dst_elem, dst_elem);
    f.f.body.push(f.store(dst_elem, result), S);

    b.entry_point("main", [256, 1, 1], f.finish());
    b.finish()
}

// ---------------------------------------------------------------------------
// sgd.wgsl
// ---------------------------------------------------------------------------

fn gen_sgd() -> Module {
    let mut b = Builder::new();
    let ty_params = b.params_u32_f32_u32_u32("Params", &["len", "lr", "_pad0", "_pad1"]);
    let gv_param = b.storage_ro("param");
    let gv_grad = b.storage_ro("grad");
    let gv_dst = b.storage_rw("dst");
    let gv_params = b.uniform("params", ty_params);

    let mut f = FnBuilder::new(&b);
    let gid = f.arg_gid();
    let i = f.vec_x(gid);
    f.label("i", i);
    let params_ptr = f.global(gv_params);
    let len_ptr = f.field(params_ptr, 0);
    let len = f.load(len_ptr);
    let cond = f.binary(BinaryOperator::GreaterEqual, i, len);

    let lr_ptr = f.field(params_ptr, 1);
    let lr = f.load(lr_ptr);

    let param_ptr = f.global(gv_param);
    let p_elem = f.index(param_ptr, i);
    let p_val = f.load(p_elem);

    let grad_ptr = f.global(gv_grad);
    let g_elem = f.index(grad_ptr, i);
    let g_val = f.load(g_elem);

    // dst[i] = param[i] - lr * grad[i]
    let lr_g = f.binary(BinaryOperator::Multiply, lr, g_val);
    let result = f.binary(BinaryOperator::Subtract, p_val, lr_g);

    let dst_ptr = f.global(gv_dst);
    let dst_elem = f.index(dst_ptr, i);

    f.emit(i, i);
    f.emit(len_ptr, cond);
    f.f.body.push(f.if_return(cond), S);
    f.emit(lr_ptr, result);
    f.emit(dst_elem, dst_elem);
    f.f.body.push(f.store(dst_elem, result), S);

    b.entry_point("main", [256, 1, 1], f.finish());
    b.finish()
}

// ---------------------------------------------------------------------------
// transpose.wgsl
// ---------------------------------------------------------------------------

fn gen_transpose() -> Module {
    let mut b = Builder::new();
    let ty_params = b.params_u32x4("Params", &["m", "n", "_pad0", "_pad1"]);
    let gv_src = b.storage_ro("src");
    let gv_dst = b.storage_rw("dst");
    let gv_params = b.uniform("params", ty_params);

    let mut f = FnBuilder::new(&b);
    let gid = f.arg_gid();
    let col = f.vec_x(gid);
    f.label("col", col);
    let row = f.vec_y(gid);
    f.label("row", row);

    let params_ptr = f.global(gv_params);
    let m_ptr = f.field(params_ptr, 0);
    let m = f.load(m_ptr);
    let n_ptr = f.field(params_ptr, 1);
    let n = f.load(n_ptr);

    let cond_r = f.binary(BinaryOperator::GreaterEqual, row, m);
    let cond_c = f.binary(BinaryOperator::GreaterEqual, col, n);
    let cond = f.binary(BinaryOperator::LogicalOr, cond_r, cond_c);

    // src[row * n + col]
    let row_n = f.binary(BinaryOperator::Multiply, row, n);
    let src_idx = f.binary(BinaryOperator::Add, row_n, col);
    let src_ptr = f.global(gv_src);
    let src_elem = f.index(src_ptr, src_idx);
    let val = f.load(src_elem);

    // dst[col * m + row]
    let col_m = f.binary(BinaryOperator::Multiply, col, m);
    let dst_idx = f.binary(BinaryOperator::Add, col_m, row);
    let dst_ptr = f.global(gv_dst);
    let dst_elem = f.index(dst_ptr, dst_idx);

    f.emit(col, row);
    f.emit(params_ptr, cond);
    f.f.body.push(f.if_return(cond), S);
    f.emit(row_n, dst_elem);
    f.f.body.push(f.store(dst_elem, val), S);

    b.entry_point("main", [16, 16, 1], f.finish());
    b.finish()
}

// ---------------------------------------------------------------------------
// matmul.wgsl — tiled matrix multiply (16×16 tiles)
//
// Workgroup [16, 16, 1], dispatched as [ceil(N/16), ceil(M/16), 1].
// Each thread computes one element of the output, iterating over K in
// tiles of 16 using workgroup shared memory.
// ---------------------------------------------------------------------------

/// Tiled matmul: C = A × B via Naga IR with shared memory.
///
/// Uses 16×16 tiles loaded into workgroup shared memory for data reuse.
/// Each thread computes one element of the output tile.
/// Workgroup [16, 16, 1], dispatched as [ceil(N/16), ceil(M/16), 1].
fn gen_matmul() -> Module {
    gen_tiled_matmul_inner(false)
}

fn gen_tiled_matmul_inner(fused_add: bool) -> Module {
    const TILE: u32 = 16;

    let mut b = Builder::new();
    let ty_params = b.params_u32x4("Params", &["m", "n", "k", "_pad"]);
    let gv_a = b.storage_ro("matrix_a");
    let gv_b = b.storage_ro("matrix_b");
    let gv_c = b.storage_rw("matrix_c");
    let gv_d = if fused_add {
        Some(b.storage_ro("src"))
    } else {
        None
    };
    let gv_params = b.uniform("params", ty_params);
    let gv_sa = b.workgroup_array("shared_a", TILE * TILE);
    let gv_sb = b.workgroup_array("shared_b", TILE * TILE);

    let mut f = FnBuilder::new(&b);
    let wgid = f.arg_wgid();
    let lid = f.arg_lid();

    let wg_x = f.vec_x(wgid);
    let wg_y = f.vec_y(wgid);
    let lx = f.vec_x(lid);
    let ly = f.vec_y(lid);
    f.emit(wg_x, ly);

    // tile_col = wg_x * 16, tile_row = wg_y * 16
    let tile_c = f.literal_u32(TILE);
    let tile_col = f.binary(BinaryOperator::Multiply, wg_x, tile_c);
    let tile_c2 = f.literal_u32(TILE);
    let tile_row = f.binary(BinaryOperator::Multiply, wg_y, tile_c2);
    f.emit(tile_c, tile_row);

    // Load params
    let params_ptr = f.global(gv_params);
    let m_ptr = f.field(params_ptr, 0);
    let pm = f.load(m_ptr);
    let n_ptr = f.field(params_ptr, 1);
    let pn = f.load(n_ptr);
    let k_ptr = f.field(params_ptr, 2);
    let pk = f.load(k_ptr);
    f.emit(params_ptr, pk);

    // var sum = 0.0;
    let zero_f = f.literal_f32(0.0);
    f.emit(zero_f, zero_f);
    let sum_var = f.local_var("sum", b.ty_f32, None);
    let sum_ptr = f.local_ptr(sum_var);
    f.f.body.push(f.store(sum_ptr, zero_f), S);

    // var t = 0u;  (K-tile offset)
    let t_var = f.local_var("t", b.ty_u32, None);
    let zero_u = f.literal_u32(0);
    f.emit(zero_u, zero_u);
    let t_ptr = f.local_ptr(t_var);
    f.f.body.push(f.store(t_ptr, zero_u), S);

    // ===== K-tile loop =====
    let mut loop_body = Block::new();
    {
        let t_val = f.load(t_ptr);
        let break_cond = f.binary(BinaryOperator::GreaterEqual, t_val, pk);
        push_emit(&f.f.expressions, &mut loop_body, t_val, break_cond);
        loop_body.push(FnBuilder::if_break(break_cond), S);

        // --- Load A tile into shared_a ---
        // shared_a[ly * 16 + lx] = A[(tile_row + ly) * k + (t + lx)]  (0 if OOB)
        let sa_ptr = f.global(gv_sa);
        let a_ptr = f.global(gv_a);
        let ly_16 = f.binary(BinaryOperator::Multiply, ly, tile_c);
        let sa_idx = f.binary(BinaryOperator::Add, ly_16, lx);
        let sa_elem = f.index(sa_ptr, sa_idx);
        let a_row = f.binary(BinaryOperator::Add, tile_row, ly);
        let a_col = f.binary(BinaryOperator::Add, t_val, lx);
        let in_m = f.binary(BinaryOperator::Less, a_row, pm);
        let in_k = f.binary(BinaryOperator::Less, a_col, pk);
        let a_in_bounds = f.binary(BinaryOperator::LogicalAnd, in_m, in_k);
        let a_row_k = f.binary(BinaryOperator::Multiply, a_row, pk);
        let a_global = f.binary(BinaryOperator::Add, a_row_k, a_col);
        let a_elem = f.index(a_ptr, a_global);
        let a_val = f.load(a_elem);
        let zero_pad_a = f.literal_f32(0.0);
        let a_selected = f.select(a_in_bounds, a_val, zero_pad_a);
        push_emit(&f.f.expressions, &mut loop_body, sa_ptr, a_selected);
        loop_body.push(f.store(sa_elem, a_selected), S);

        // --- Load B tile into shared_b ---
        // shared_b[ly * 16 + lx] = B[(t + ly) * n + (tile_col + lx)]  (0 if OOB)
        let sb_ptr = f.global(gv_sb);
        let b_ptr = f.global(gv_b);
        let sb_elem = f.index(sb_ptr, sa_idx); // same local index
        let b_row = f.binary(BinaryOperator::Add, t_val, ly);
        let b_col = f.binary(BinaryOperator::Add, tile_col, lx);
        let b_in_k = f.binary(BinaryOperator::Less, b_row, pk);
        let b_in_n = f.binary(BinaryOperator::Less, b_col, pn);
        let b_in_bounds = f.binary(BinaryOperator::LogicalAnd, b_in_k, b_in_n);
        let b_row_n = f.binary(BinaryOperator::Multiply, b_row, pn);
        let b_global = f.binary(BinaryOperator::Add, b_row_n, b_col);
        let b_elem = f.index(b_ptr, b_global);
        let b_val = f.load(b_elem);
        let zero_pad_b = f.literal_f32(0.0);
        let b_selected = f.select(b_in_bounds, b_val, zero_pad_b);
        push_emit(&f.f.expressions, &mut loop_body, sb_ptr, b_selected);
        loop_body.push(f.store(sb_elem, b_selected), S);

        // workgroupBarrier — shared memory is populated
        loop_body.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);

        // --- Accumulate: sum += shared_a[ly*16+j] * shared_b[j*16+lx] for j in 0..16 ---
        // Unroll the inner loop for better performance
        for j in 0..TILE {
            let j_c = f.literal_u32(j);
            let ly_tile = f.binary(BinaryOperator::Multiply, ly, tile_c);
            let sa_j = f.binary(BinaryOperator::Add, ly_tile, j_c);
            let sa_j_elem = f.index(sa_ptr, sa_j);
            let sa_j_val = f.load(sa_j_elem);

            let j_tile = f.binary(BinaryOperator::Multiply, j_c, tile_c);
            let sb_j = f.binary(BinaryOperator::Add, j_tile, lx);
            let sb_j_elem = f.index(sb_ptr, sb_j);
            let sb_j_val = f.load(sb_j_elem);

            let prod = f.binary(BinaryOperator::Multiply, sa_j_val, sb_j_val);
            let old_sum = f.load(sum_ptr);
            let new_sum = f.binary(BinaryOperator::Add, old_sum, prod);
            push_emit(&f.f.expressions, &mut loop_body, j_c, new_sum);
            loop_body.push(f.store(sum_ptr, new_sum), S);
        }

        // workgroupBarrier — before next tile overwrites shared memory
        loop_body.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);

        // t += 16
        let tile_inc = f.literal_u32(TILE);
        let t_val2 = f.load(t_ptr);
        let t_next = f.binary(BinaryOperator::Add, t_val2, tile_inc);
        push_emit(&f.f.expressions, &mut loop_body, tile_inc, t_next);
        loop_body.push(f.store(t_ptr, t_next), S);
    }

    f.f.body.push(
        Statement::Loop {
            body: loop_body,
            continuing: Block::new(),
            break_if: None,
        },
        S,
    );

    // Store result: if (row < m && col < n) { c[row * n + col] = sum [+ d] }
    let row = f.binary(BinaryOperator::Add, tile_row, ly);
    let col = f.binary(BinaryOperator::Add, tile_col, lx);
    let r_lt_m = f.binary(BinaryOperator::Less, row, pm);
    let c_lt_n = f.binary(BinaryOperator::Less, col, pn);
    let store_cond = f.binary(BinaryOperator::LogicalAnd, r_lt_m, c_lt_n);
    let row_n = f.binary(BinaryOperator::Multiply, row, pn);
    let c_idx = f.binary(BinaryOperator::Add, row_n, col);
    let c_ptr = f.global(gv_c);
    let c_elem = f.index(c_ptr, c_idx);
    let final_sum = f.load(sum_ptr);
    f.emit(row, final_sum);

    if let Some(gv_d) = gv_d {
        // Fused add: result = sum + d[idx]
        let d_ptr = f.global(gv_d);
        let d_elem = f.index(d_ptr, c_idx);
        let d_val = f.load(d_elem);
        let fused_val = f.binary(BinaryOperator::Add, final_sum, d_val);
        let mut store_block = Block::new();
        push_emit(&f.f.expressions, &mut store_block, d_ptr, fused_val);
        store_block.push(f.store(c_elem, fused_val), S);
        f.f.body.push(
            Statement::If {
                condition: store_cond,
                accept: store_block,
                reject: Block::new(),
            },
            S,
        );
    } else {
        f.f.body.push(
            Statement::If {
                condition: store_cond,
                accept: Block::from_vec(vec![f.store(c_elem, final_sum)]),
                reject: Block::new(),
            },
            S,
        );
    }

    b.entry_point("main", [16, 16, 1], f.finish());
    b.finish()
}

/// Tiled fused matmul + add: C = A × B + D with shared memory.
///
/// Same as tiled matmul but reads a 4th buffer `addend` and adds it
/// to the result before storing. Eliminates a separate Add dispatch.
fn gen_matmul_add() -> Module {
    gen_tiled_matmul_inner(true)
}

// ---------------------------------------------------------------------------
// matmul_coop.wgsl — cooperative matrix multiply (16×16 tiles)
//
// Uses cooperative matrix operations for hardware-accelerated matrix multiply
// on supported GPUs (VK_KHR_cooperative_matrix on Vulkan, simdgroup_matrix
// on Metal).
//
// Workgroup [8, 8, 1], dispatched as [ceil(M/8), ceil(N/8), 1].
// Each workgroup computes one 8×8 output tile, iterating over K in
// steps of 8.
// ---------------------------------------------------------------------------

/// Cooperative matrix matmul: C = A × B via Naga IR.
///
/// Generates `CooperativeLoad` / `CooperativeMultiplyAdd` / `CooperativeStore`
/// expressions. Hardware-accelerated on Vulkan (VK_KHR_cooperative_matrix)
/// and Metal (simdgroup_matrix).
///
/// Workgroup [8, 8, 1], dispatched as [ceil(M/8), ceil(N/8), 1].
/// Mixed-precision cooperative matmul: C(f32) = A(f16) × B(f16) + C(f32).
///
/// Uses 16×16×16 cooperative matrix tiles with f16 A/B and f32 C/Result,
/// matching AMD RDNA's native MFMA instruction format.
///
/// Data flow per K-tile:
///   1. Each of 64 threads loads 4 f32 elements from A and B
///   2. Converts to f16, stores into workgroup shared memory
///   3. workgroupBarrier
///   4. CooperativeLoad f16 tiles from shared memory
///   5. CooperativeMultiplyAdd (f16 × f16 + f32 → f32)
///
/// Workgroup [64, 1, 1], dispatched as [ceil(M/16), ceil(N/16), 1].
fn gen_matmul_coop() -> Module {
    gen_matmul_coop_inner(false)
}

/// Cooperative matmul with fused addend: C = A × B + D.
/// Same as gen_matmul_coop but loads the accumulator from a 4th buffer.
fn gen_matmul_coop_add() -> Module {
    gen_matmul_coop_inner(true)
}

fn gen_matmul_coop_inner(fused_add: bool) -> Module {
    const TILE: u32 = 16;
    const TILE_ELEMS: u32 = TILE * TILE; // 256
    const WG_SIZE: u32 = 64;
    const ELEMS_PER_THREAD: u32 = TILE_ELEMS / WG_SIZE; // 4
    // 2×2 output tile grid per workgroup: each WG computes 4 coop tiles = 32×32 output.
    // Increases arithmetic intensity 4× vs 1×1, reduces dispatch overhead 4×.
    const OUTPUT_TILE: u32 = TILE * 2; // 32

    let mut b = Builder::new();
    let ty_params = b.params_u32x4("Params", &["m", "n", "k", "_pad"]);
    let gv_a = b.storage_ro("matrix_a");
    let gv_b = b.storage_ro("matrix_b");
    let gv_c = b.storage_rw("matrix_c");
    let gv_addend = if fused_add {
        Some(b.storage_ro("src"))
    } else {
        None
    };
    let gv_params = b.uniform("params", ty_params);

    // f16 shared memory for cooperative matrix staging (256 elements each = 16×16 tile).
    let ty_f16 = b.m.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F16),
        },
        S,
    );
    let ty_shared = b.m.types.insert(
        Type {
            name: None,
            inner: TypeInner::Array {
                base: ty_f16,
                size: ArraySize::Constant(std::num::NonZeroU32::new(TILE_ELEMS).unwrap()),
                stride: 2, // f16 = 2 bytes
            },
        },
        S,
    );

    // Fix #4: stage matrix_b → sa (role=A) and matrix_a → sb (role=B).
    // Col-major load of row-major staged data transposes each tile:
    //   sa = B tile → loaded col-major as A → B.T in role A
    //   sb = A tile → loaded col-major as B → A.T in role B
    //   B.T @ A.T = (A@B).T; col-major store of (A@B).T = A@B row-major ✓
    //
    // 2×2 tile layout (M0=tile_row, N0=tile_col, M1=tile_row+16, N1=tile_col+16):
    //   sa0 = B[:,N0:N0+16] → a_coop0     sb0 = A[M0:M0+16,:] → b_coop0
    //   sa1 = B[:,N1:N1+16] → a_coop1     sb1 = A[M1:M1+16,:] → b_coop1
    //   acc00 += a_coop0 @ b_coop0  →  C[M0:M0+16, N0:N0+16]
    //   acc01 += a_coop1 @ b_coop0  →  C[M0:M0+16, N1:N1+16]
    //   acc10 += a_coop0 @ b_coop1  →  C[M1:M1+16, N0:N0+16]
    //   acc11 += a_coop1 @ b_coop1  →  C[M1:M1+16, N1:N1+16]
    let gv_sa0 = b.m.global_variables.append(
        GlobalVariable {
            name: Some("shared_a0".to_string()),
            space: AddressSpace::WorkGroup,
            binding: None,
            ty: ty_shared,
            init: None,
            memory_decorations: MemoryDecorations::empty(),
        },
        S,
    );
    let gv_sa1 = b.m.global_variables.append(
        GlobalVariable {
            name: Some("shared_a1".to_string()),
            space: AddressSpace::WorkGroup,
            binding: None,
            ty: ty_shared,
            init: None,
            memory_decorations: MemoryDecorations::empty(),
        },
        S,
    );
    let gv_sb0 = b.m.global_variables.append(
        GlobalVariable {
            name: Some("shared_b0".to_string()),
            space: AddressSpace::WorkGroup,
            binding: None,
            ty: ty_shared,
            init: None,
            memory_decorations: MemoryDecorations::empty(),
        },
        S,
    );
    let gv_sb1 = b.m.global_variables.append(
        GlobalVariable {
            name: Some("shared_b1".to_string()),
            space: AddressSpace::WorkGroup,
            binding: None,
            ty: ty_shared,
            init: None,
            memory_decorations: MemoryDecorations::empty(),
        },
        S,
    );

    let mut f = FnBuilder::new(&b);
    let wgid = f.arg_wgid();
    let lid = f.arg_lid();

    let wg_x = f.vec_x(wgid);
    let wg_y = f.vec_y(wgid);
    let tid = f.vec_x(lid); // thread index 0..63
    f.emit(wg_x, tid);

    // tile_row = wg_x * 32, tile_col = wg_y * 32
    let out_tile_c = f.literal_u32(OUTPUT_TILE);
    let tile_row = f.binary(BinaryOperator::Multiply, wg_x, out_tile_c);
    let out_tile_c2 = f.literal_u32(OUTPUT_TILE);
    let tile_col = f.binary(BinaryOperator::Multiply, wg_y, out_tile_c2);
    f.emit(out_tile_c, tile_col);

    // Load params
    let params_ptr = f.global(gv_params);
    let m_ptr = f.field(params_ptr, 0);
    let pm = f.load(m_ptr);
    let n_ptr = f.field(params_ptr, 1);
    let pn = f.load(n_ptr);
    let k_ptr = f.field(params_ptr, 2);
    let pk = f.load(k_ptr);
    f.emit(params_ptr, pk);

    // Compute 4 output tile offsets (tile_row/tile_col × N + col_offset):
    //   c_offset_00 = tile_row * N + tile_col
    //   c_offset_01 = tile_row * N + (tile_col + 16)
    //   c_offset_10 = (tile_row + 16) * N + tile_col
    //   c_offset_11 = (tile_row + 16) * N + (tile_col + 16)
    let c_off_00 = f.binary(BinaryOperator::Multiply, tile_row, pn);
    let c_offset_00 = f.binary(BinaryOperator::Add, c_off_00, tile_col);
    f.emit(c_off_00, c_offset_00);

    let tile16_col = f.literal_u32(TILE);
    let tile_col_16 = f.binary(BinaryOperator::Add, tile_col, tile16_col);
    let c_offset_01 = f.binary(BinaryOperator::Add, c_off_00, tile_col_16);
    f.emit(tile16_col, c_offset_01);

    let tile16_row = f.literal_u32(TILE);
    let tile_row_16 = f.binary(BinaryOperator::Add, tile_row, tile16_row);
    let c_off_10 = f.binary(BinaryOperator::Multiply, tile_row_16, pn);
    let c_offset_10 = f.binary(BinaryOperator::Add, c_off_10, tile_col);
    f.emit(tile16_row, c_offset_10);

    let c_offset_11 = f.binary(BinaryOperator::Add, c_off_10, tile_col_16);
    f.emit(c_offset_11, c_offset_11);

    // Bounds conditions for the secondary (N1, M1) output tiles.
    // tile_col + 16 may exceed pn for edge workgroups: the N1-tile's output positions
    // wrap into the next row of the flat buffer, corrupting valid data.  Guard every
    // load/store of acc_01, acc_10, acc_11 with these flags.
    let cond_n1_valid = f.binary(BinaryOperator::Less, tile_col_16, pn); // tile_col+16 < pn
    f.emit(cond_n1_valid, cond_n1_valid);
    let cond_m1_valid = f.binary(BinaryOperator::Less, tile_row_16, pm); // tile_row+16 < pm
    f.emit(cond_m1_valid, cond_m1_valid);
    let cond_n1m1_valid = f.binary(BinaryOperator::LogicalAnd, cond_n1_valid, cond_m1_valid);
    f.emit(cond_n1m1_valid, cond_n1m1_valid);

    // 4 cooperative matrix accumulators (f32, 16×16 each)
    let ty_coop_c = b.m.types.insert(
        Type {
            name: None,
            inner: TypeInner::CooperativeMatrix {
                columns: CooperativeSize::Sixteen,
                rows: CooperativeSize::Sixteen,
                scalar: Scalar::F32,
                role: CooperativeRole::C,
            },
        },
        S,
    );
    let acc_00_var = f.local_var("acc00", ty_coop_c, None);
    let acc_00_ptr = f.local_ptr(acc_00_var);
    let acc_01_var = f.local_var("acc01", ty_coop_c, None);
    let acc_01_ptr = f.local_ptr(acc_01_var);
    let acc_10_var = f.local_var("acc10", ty_coop_c, None);
    let acc_10_ptr = f.local_ptr(acc_10_var);
    let acc_11_var = f.local_var("acc11", ty_coop_c, None);
    let acc_11_ptr = f.local_ptr(acc_11_var);

    if let Some(gv_add) = gv_addend {
        // Fused add: load 4 addend tiles into accumulators (residual connection).
        let add_src00 = f.global(gv_add);
        let add_elem_00 = f.index(add_src00, c_offset_00);
        let acc_load_00 = f.expr(Expression::CooperativeLoad {
            columns: CooperativeSize::Sixteen,
            rows: CooperativeSize::Sixteen,
            role: CooperativeRole::C,
            data: CooperativeData {
                pointer: add_elem_00,
                stride: pn,
                row_major: false,
            },
        });
        f.emit(add_src00, acc_load_00);
        f.f.body.push(f.store(acc_00_ptr, acc_load_00), S);

        let add_src01 = f.global(gv_add);
        let add_elem_01 = f.index(add_src01, c_offset_01);
        let acc_load_01 = f.expr(Expression::CooperativeLoad {
            columns: CooperativeSize::Sixteen,
            rows: CooperativeSize::Sixteen,
            role: CooperativeRole::C,
            data: CooperativeData {
                pointer: add_elem_01,
                stride: pn,
                row_major: false,
            },
        });
        f.emit(add_src01, acc_load_01);
        f.f.body.push(f.store(acc_01_ptr, acc_load_01), S);

        let add_src10 = f.global(gv_add);
        let add_elem_10 = f.index(add_src10, c_offset_10);
        let acc_load_10 = f.expr(Expression::CooperativeLoad {
            columns: CooperativeSize::Sixteen,
            rows: CooperativeSize::Sixteen,
            role: CooperativeRole::C,
            data: CooperativeData {
                pointer: add_elem_10,
                stride: pn,
                row_major: false,
            },
        });
        f.emit(add_src10, acc_load_10);
        f.f.body.push(f.store(acc_10_ptr, acc_load_10), S);

        let add_src11 = f.global(gv_add);
        let add_elem_11 = f.index(add_src11, c_offset_11);
        let acc_load_11 = f.expr(Expression::CooperativeLoad {
            columns: CooperativeSize::Sixteen,
            rows: CooperativeSize::Sixteen,
            role: CooperativeRole::C,
            data: CooperativeData {
                pointer: add_elem_11,
                stride: pn,
                row_major: false,
            },
        });
        f.emit(add_src11, acc_load_11);
        f.f.body.push(f.store(acc_11_ptr, acc_load_11), S);
    } else {
        // Regular MatMul: all accumulators start at zero.
        let zero_acc = f.expr(Expression::ZeroValue(ty_coop_c));
        f.emit(zero_acc, zero_acc);
        f.f.body.push(f.store(acc_00_ptr, zero_acc), S);
        f.f.body.push(f.store(acc_01_ptr, zero_acc), S);
        f.f.body.push(f.store(acc_10_ptr, zero_acc), S);
        f.f.body.push(f.store(acc_11_ptr, zero_acc), S);
    }

    // var t: u32 = 0
    let t_var = f.local_var("t", b.ty_u32, None);
    let zero_u = f.literal_u32(0);
    f.emit(zero_u, zero_u);
    let t_ptr = f.local_ptr(t_var);
    f.f.body.push(f.store(t_ptr, zero_u), S);

    // Hoist loop-invariant staging index components (depend only on tid and tile offsets).
    //
    // For sa (B staging, role A):
    //   src_col_b = tid & 15  → N col index within tile
    //   base_row_b = tid >> 4 → K row base (varies per e: full row = base + e*4)
    //   cc  = tile_col + src_col_b       → global N col for sa0 (B[:,N0])
    //   cc1 = tile_col + 16 + src_col_b  → global N col for sa1 (B[:,N1])
    //   in_n  = cc  < pn   [fully invariant]
    //   in_n1 = cc1 < pn   [fully invariant]
    //
    // For sb (A staging, role B):
    //   src_col_a = tid & 15  → K col index within tile (same bit as src_col_b)
    //   base_row_a = tid >> 4 → M row base (varies per e: full row = base + e*4)
    //   tile_row_16 = tile_row + 16 → M row base for sb1 (A[M1,:]) [computed above for c_offset_10]
    let mask15 = f.literal_u32(TILE - 1);
    let src_col_a = f.binary(BinaryOperator::And, tid, mask15);
    let shift4 = f.literal_u32(4);
    let base_row_a = f.binary(BinaryOperator::ShiftRight, tid, shift4);
    f.emit(mask15, base_row_a);

    let mask15_b = f.literal_u32(TILE - 1);
    let src_col_b = f.binary(BinaryOperator::And, tid, mask15_b);
    let cc = f.binary(BinaryOperator::Add, tile_col, src_col_b);
    let in_n = f.binary(BinaryOperator::Less, cc, pn);
    let shift4_b = f.literal_u32(4);
    let base_row_b = f.binary(BinaryOperator::ShiftRight, tid, shift4_b);
    f.emit(mask15_b, base_row_b);

    // cc1 = cc + 16 (N col for sa1)
    let tile16_cc1 = f.literal_u32(TILE);
    let cc1 = f.binary(BinaryOperator::Add, cc, tile16_cc1);
    let in_n1 = f.binary(BinaryOperator::Less, cc1, pn);
    f.emit(tile16_cc1, in_n1);

    // ===== K-tile loop =====
    let mut loop_body = Block::new();
    {
        let t_val = f.load(t_ptr);
        let break_cond = f.binary(BinaryOperator::GreaterEqual, t_val, pk);
        push_emit(&f.f.expressions, &mut loop_body, t_val, break_cond);
        loop_body.push(FnBuilder::if_break(break_cond), S);

        // ---- Stage sa0: B[t:t+16, tile_col:tile_col+16] → shared_a0 ----
        let sa0_ptr = f.global(gv_sa0);
        let b_ptr0 = f.global(gv_b);
        let zero_f32_sa0 = f.literal_f32(0.0);
        let zero_f16_sa0 = f.expr(Expression::As {
            expr: zero_f32_sa0,
            kind: ScalarKind::Float,
            convert: Some(2),
        });
        push_emit(&f.f.expressions, &mut loop_body, sa0_ptr, zero_f16_sa0);

        for e in 0..ELEMS_PER_THREAD {
            let offset = f.literal_u32(e * WG_SIZE);
            let flat_idx = f.binary(BinaryOperator::Add, tid, offset);
            let sa0_elem = f.index(sa0_ptr, flat_idx);
            let e_rows_b = f.literal_u32(e * ELEMS_PER_THREAD);
            let src_row = f.binary(BinaryOperator::Add, base_row_b, e_rows_b);
            let tr = f.binary(BinaryOperator::Add, t_val, src_row);
            let in_k_b = f.binary(BinaryOperator::Less, tr, pk);
            let in_bounds = f.binary(BinaryOperator::LogicalAnd, in_k_b, in_n);
            push_emit(&f.f.expressions, &mut loop_body, offset, in_bounds);
            let trn = f.binary(BinaryOperator::Multiply, tr, pn);
            let global_idx = f.binary(BinaryOperator::Add, trn, cc);
            let b_elem = f.index(b_ptr0, global_idx);
            let b_val = f.load(b_elem);
            let b_f16 = f.expr(Expression::As {
                expr: b_val,
                kind: ScalarKind::Float,
                convert: Some(2),
            });
            let mut accept = Block::new();
            push_emit(&f.f.expressions, &mut accept, trn, b_f16);
            accept.push(f.store(sa0_elem, b_f16), S);
            loop_body.push(
                Statement::If {
                    condition: in_bounds,
                    accept,
                    reject: Block::from_vec(vec![f.store(sa0_elem, zero_f16_sa0)]),
                },
                S,
            );
        }

        // ---- Stage sa1: B[t:t+16, tile_col+16:tile_col+32] → shared_a1 ----
        // Same K rows as sa0, next 16 N cols. Uses cc1, in_n1 (hoisted).
        let sa1_ptr = f.global(gv_sa1);
        let b_ptr1 = f.global(gv_b);
        let zero_f32_sa1 = f.literal_f32(0.0);
        let zero_f16_sa1 = f.expr(Expression::As {
            expr: zero_f32_sa1,
            kind: ScalarKind::Float,
            convert: Some(2),
        });
        push_emit(&f.f.expressions, &mut loop_body, sa1_ptr, zero_f16_sa1);

        for e in 0..ELEMS_PER_THREAD {
            let offset1 = f.literal_u32(e * WG_SIZE);
            let flat_idx1 = f.binary(BinaryOperator::Add, tid, offset1);
            let sa1_elem = f.index(sa1_ptr, flat_idx1);
            let e_rows_b1 = f.literal_u32(e * ELEMS_PER_THREAD);
            let src_row1 = f.binary(BinaryOperator::Add, base_row_b, e_rows_b1);
            let tr1 = f.binary(BinaryOperator::Add, t_val, src_row1);
            let in_k_b1 = f.binary(BinaryOperator::Less, tr1, pk);
            let in_bounds1 = f.binary(BinaryOperator::LogicalAnd, in_k_b1, in_n1);
            push_emit(&f.f.expressions, &mut loop_body, offset1, in_bounds1);
            let trn1 = f.binary(BinaryOperator::Multiply, tr1, pn);
            let global_idx1 = f.binary(BinaryOperator::Add, trn1, cc1);
            let b_elem1 = f.index(b_ptr1, global_idx1);
            let b_val1 = f.load(b_elem1);
            let b_f16_1 = f.expr(Expression::As {
                expr: b_val1,
                kind: ScalarKind::Float,
                convert: Some(2),
            });
            let mut accept1 = Block::new();
            push_emit(&f.f.expressions, &mut accept1, trn1, b_f16_1);
            accept1.push(f.store(sa1_elem, b_f16_1), S);
            loop_body.push(
                Statement::If {
                    condition: in_bounds1,
                    accept: accept1,
                    reject: Block::from_vec(vec![f.store(sa1_elem, zero_f16_sa1)]),
                },
                S,
            );
        }

        // ---- Stage sb0: A[tile_row:tile_row+16, t:t+16] → shared_b0 ----
        let sb0_ptr = f.global(gv_sb0);
        let a_ptr0 = f.global(gv_a);
        let zero_f32_sb0 = f.literal_f32(0.0);
        let zero_f16_sb0 = f.expr(Expression::As {
            expr: zero_f32_sb0,
            kind: ScalarKind::Float,
            convert: Some(2),
        });
        // tc_a = t + src_col_a (K col); in_k_a = tc_a < pk. Both shared with sb1.
        let tc_a = f.binary(BinaryOperator::Add, t_val, src_col_a);
        let in_k_a = f.binary(BinaryOperator::Less, tc_a, pk);
        push_emit(&f.f.expressions, &mut loop_body, sb0_ptr, in_k_a);

        for e in 0..ELEMS_PER_THREAD {
            let offset2 = f.literal_u32(e * WG_SIZE);
            let flat_idx2 = f.binary(BinaryOperator::Add, tid, offset2);
            let sb0_elem = f.index(sb0_ptr, flat_idx2);
            let e_rows2 = f.literal_u32(e * ELEMS_PER_THREAD);
            let src_row2 = f.binary(BinaryOperator::Add, base_row_a, e_rows2);
            let gr0 = f.binary(BinaryOperator::Add, tile_row, src_row2);
            let in_m0 = f.binary(BinaryOperator::Less, gr0, pm);
            let in_bounds2 = f.binary(BinaryOperator::LogicalAnd, in_m0, in_k_a);
            push_emit(&f.f.expressions, &mut loop_body, offset2, in_bounds2);
            let gr0k = f.binary(BinaryOperator::Multiply, gr0, pk);
            let global_idx2 = f.binary(BinaryOperator::Add, gr0k, tc_a);
            let a_elem0 = f.index(a_ptr0, global_idx2);
            let a_val0 = f.load(a_elem0);
            let a_f16_0 = f.expr(Expression::As {
                expr: a_val0,
                kind: ScalarKind::Float,
                convert: Some(2),
            });
            let mut accept2 = Block::new();
            push_emit(&f.f.expressions, &mut accept2, gr0k, a_f16_0);
            accept2.push(f.store(sb0_elem, a_f16_0), S);
            loop_body.push(
                Statement::If {
                    condition: in_bounds2,
                    accept: accept2,
                    reject: Block::from_vec(vec![f.store(sb0_elem, zero_f16_sb0)]),
                },
                S,
            );
        }

        // ---- Stage sb1: A[tile_row+16:tile_row+32, t:t+16] → shared_b1 ----
        // Same K cols as sb0. Reuses tc_a, in_k_a. Uses tile_row_16 for M offset.
        let sb1_ptr = f.global(gv_sb1);
        let a_ptr1 = f.global(gv_a);
        let zero_f32_sb1 = f.literal_f32(0.0);
        let zero_f16_sb1 = f.expr(Expression::As {
            expr: zero_f32_sb1,
            kind: ScalarKind::Float,
            convert: Some(2),
        });
        push_emit(&f.f.expressions, &mut loop_body, sb1_ptr, zero_f16_sb1);

        for e in 0..ELEMS_PER_THREAD {
            let offset3 = f.literal_u32(e * WG_SIZE);
            let flat_idx3 = f.binary(BinaryOperator::Add, tid, offset3);
            let sb1_elem = f.index(sb1_ptr, flat_idx3);
            let e_rows3 = f.literal_u32(e * ELEMS_PER_THREAD);
            let src_row3 = f.binary(BinaryOperator::Add, base_row_a, e_rows3);
            let gr1 = f.binary(BinaryOperator::Add, tile_row_16, src_row3);
            let in_m1 = f.binary(BinaryOperator::Less, gr1, pm);
            let in_bounds3 = f.binary(BinaryOperator::LogicalAnd, in_m1, in_k_a);
            push_emit(&f.f.expressions, &mut loop_body, offset3, in_bounds3);
            let gr1k = f.binary(BinaryOperator::Multiply, gr1, pk);
            let global_idx3 = f.binary(BinaryOperator::Add, gr1k, tc_a);
            let a_elem1 = f.index(a_ptr1, global_idx3);
            let a_val1 = f.load(a_elem1);
            let a_f16_1 = f.expr(Expression::As {
                expr: a_val1,
                kind: ScalarKind::Float,
                convert: Some(2),
            });
            let mut accept3 = Block::new();
            push_emit(&f.f.expressions, &mut accept3, gr1k, a_f16_1);
            accept3.push(f.store(sb1_elem, a_f16_1), S);
            loop_body.push(
                Statement::If {
                    condition: in_bounds3,
                    accept: accept3,
                    reject: Block::from_vec(vec![f.store(sb1_elem, zero_f16_sb1)]),
                },
                S,
            );
        }

        // workgroupBarrier — all 4 shared tiles populated
        loop_body.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);

        // ---- CooperativeLoad 4 tiles + 4 CoopMMA ----
        // sa0 → role A (B[:,N0].T); sa1 → role A (B[:,N1].T)
        // sb0 → role B (A[M0,:].T); sb1 → role B (A[M1,:].T)
        let sa0_zero = f.global(gv_sa0);
        let zero_idx_a0 = f.literal_u32(0);
        let sa0_base = f.index(sa0_zero, zero_idx_a0);
        let stride_a0 = f.literal_u32(TILE);
        let a_coop0 = f.expr(Expression::CooperativeLoad {
            columns: CooperativeSize::Sixteen,
            rows: CooperativeSize::Sixteen,
            role: CooperativeRole::A,
            data: CooperativeData {
                pointer: sa0_base,
                stride: stride_a0,
                row_major: false,
            },
        });

        let sa1_zero = f.global(gv_sa1);
        let zero_idx_a1 = f.literal_u32(0);
        let sa1_base = f.index(sa1_zero, zero_idx_a1);
        let stride_a1 = f.literal_u32(TILE);
        let a_coop1 = f.expr(Expression::CooperativeLoad {
            columns: CooperativeSize::Sixteen,
            rows: CooperativeSize::Sixteen,
            role: CooperativeRole::A,
            data: CooperativeData {
                pointer: sa1_base,
                stride: stride_a1,
                row_major: false,
            },
        });

        let sb0_zero = f.global(gv_sb0);
        let zero_idx_b0 = f.literal_u32(0);
        let sb0_base = f.index(sb0_zero, zero_idx_b0);
        let stride_b0 = f.literal_u32(TILE);
        let b_coop0 = f.expr(Expression::CooperativeLoad {
            columns: CooperativeSize::Sixteen,
            rows: CooperativeSize::Sixteen,
            role: CooperativeRole::B,
            data: CooperativeData {
                pointer: sb0_base,
                stride: stride_b0,
                row_major: false,
            },
        });

        let sb1_zero = f.global(gv_sb1);
        let zero_idx_b1 = f.literal_u32(0);
        let sb1_base = f.index(sb1_zero, zero_idx_b1);
        let stride_b1 = f.literal_u32(TILE);
        let b_coop1 = f.expr(Expression::CooperativeLoad {
            columns: CooperativeSize::Sixteen,
            rows: CooperativeSize::Sixteen,
            role: CooperativeRole::B,
            data: CooperativeData {
                pointer: sb1_base,
                stride: stride_b1,
                row_major: false,
            },
        });

        // 4 CoopMMA: acc[ri][ci] += a_coop[ci] @ b_coop[ri]
        let old_acc_00 = f.load(acc_00_ptr);
        let fma_00 = f.expr(Expression::CooperativeMultiplyAdd {
            a: a_coop0,
            b: b_coop0,
            c: old_acc_00,
        });
        let old_acc_01 = f.load(acc_01_ptr);
        let fma_01 = f.expr(Expression::CooperativeMultiplyAdd {
            a: a_coop1,
            b: b_coop0,
            c: old_acc_01,
        });
        let old_acc_10 = f.load(acc_10_ptr);
        let fma_10 = f.expr(Expression::CooperativeMultiplyAdd {
            a: a_coop0,
            b: b_coop1,
            c: old_acc_10,
        });
        let old_acc_11 = f.load(acc_11_ptr);
        let fma_11 = f.expr(Expression::CooperativeMultiplyAdd {
            a: a_coop1,
            b: b_coop1,
            c: old_acc_11,
        });

        push_emit(&f.f.expressions, &mut loop_body, sa0_zero, fma_11);
        loop_body.push(f.store(acc_00_ptr, fma_00), S);
        loop_body.push(f.store(acc_01_ptr, fma_01), S);
        loop_body.push(f.store(acc_10_ptr, fma_10), S);
        loop_body.push(f.store(acc_11_ptr, fma_11), S);

        // workgroupBarrier — before next K-tile overwrites shared memory
        loop_body.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);

        // t += 16
        let tile_inc = f.literal_u32(TILE);
        let t_val2 = f.load(t_ptr);
        let t_next = f.binary(BinaryOperator::Add, t_val2, tile_inc);
        push_emit(&f.f.expressions, &mut loop_body, tile_inc, t_next);
        loop_body.push(f.store(t_ptr, t_next), S);
    }

    f.f.body.push(
        Statement::Loop {
            body: loop_body,
            continuing: Block::new(),
            break_if: None,
        },
        S,
    );

    // Store 4 accumulators.
    // acc_00: always valid (workgroups are only launched when tile_row < pm, tile_col < pn).
    // acc_01/acc_10/acc_11: guarded by the bounds conditions computed above.
    // Without guards the secondary tiles write zeros (or stale residual values) to
    // valid-but-wrong buffer positions (wrapping into the next row), corrupting the
    // output from the primary tile.
    let c_ptr0 = f.global(gv_c);
    let c_elem_00 = f.index(c_ptr0, c_offset_00);
    let final_00 = f.load(acc_00_ptr);
    f.emit(c_ptr0, final_00);
    f.f.body.push(
        Statement::CooperativeStore {
            target: final_00,
            data: CooperativeData {
                pointer: c_elem_00,
                stride: pn,
                row_major: false,
            },
        },
        S,
    );

    // acc_01: only store if N1-tile is within the output column range
    {
        let c_ptr1 = f.global(gv_c);
        let c_elem_01 = f.index(c_ptr1, c_offset_01);
        let final_01 = f.load(acc_01_ptr);
        f.emit(c_ptr1, final_01);
        let mut accept_01 = Block::new();
        accept_01.push(
            Statement::CooperativeStore {
                target: final_01,
                data: CooperativeData {
                    pointer: c_elem_01,
                    stride: pn,
                    row_major: false,
                },
            },
            S,
        );
        f.f.body.push(
            Statement::If {
                condition: cond_n1_valid,
                accept: accept_01,
                reject: Block::new(),
            },
            S,
        );
    }

    // acc_10: only store if M1-tile is within the output row range
    {
        let c_ptr2 = f.global(gv_c);
        let c_elem_10 = f.index(c_ptr2, c_offset_10);
        let final_10 = f.load(acc_10_ptr);
        f.emit(c_ptr2, final_10);
        let mut accept_10 = Block::new();
        accept_10.push(
            Statement::CooperativeStore {
                target: final_10,
                data: CooperativeData {
                    pointer: c_elem_10,
                    stride: pn,
                    row_major: false,
                },
            },
            S,
        );
        f.f.body.push(
            Statement::If {
                condition: cond_m1_valid,
                accept: accept_10,
                reject: Block::new(),
            },
            S,
        );
    }

    // acc_11: only store if both N1- and M1-tiles are within bounds
    {
        let c_ptr3 = f.global(gv_c);
        let c_elem_11 = f.index(c_ptr3, c_offset_11);
        let final_11 = f.load(acc_11_ptr);
        f.emit(c_ptr3, final_11);
        let mut accept_11 = Block::new();
        accept_11.push(
            Statement::CooperativeStore {
                target: final_11,
                data: CooperativeData {
                    pointer: c_elem_11,
                    stride: pn,
                    row_major: false,
                },
            },
            S,
        );
        f.f.body.push(
            Statement::If {
                condition: cond_n1m1_valid,
                accept: accept_11,
                reject: Block::new(),
            },
            S,
        );
    }

    b.entry_point("main", [WG_SIZE, 1, 1], f.finish());
    b.finish()
}

// ---------------------------------------------------------------------------
// reduce.wgsl: sum_all, mean_all
// ---------------------------------------------------------------------------

fn gen_reduce() -> Module {
    let mut b = Builder::new();
    let ty_params = b.params_u32x4("Params", &["len", "_pad0", "_pad1", "_pad2"]);
    let gv_src = b.storage_ro("src");
    let gv_dst = b.storage_rw("dst");
    let gv_params = b.uniform("params", ty_params);
    let gv_wg = b.workgroup_array("wg_data", 256);

    // Helper to build either sum_all or mean_all
    fn gen_reduce_fn(
        b: &Builder,
        is_mean: bool,
        gv_src: Handle<GlobalVariable>,
        gv_dst: Handle<GlobalVariable>,
        gv_params: Handle<GlobalVariable>,
        gv_wg: Handle<GlobalVariable>,
    ) -> Function {
        let mut f = FnBuilder::new(b);
        let gid = f.arg_gid();
        let lid = f.arg_lid();

        let i = f.vec_x(gid);
        f.label("i", i);
        let local_id = f.vec_x(lid);
        f.label("local_id", local_id);
        f.emit(i, local_id);

        let params_ptr = f.global(gv_params);
        let len_ptr = f.field(params_ptr, 0);
        let len = f.load(len_ptr);
        f.emit(params_ptr, len);

        // Load src[i] or 0.0 into wg_data[local_id]
        let i_lt_len = f.binary(BinaryOperator::Less, i, len);
        let src_ptr = f.global(gv_src);
        let src_elem = f.index(src_ptr, i);
        let src_val = f.load(src_elem);
        let zero = f.literal_f32(0.0);
        f.emit(i_lt_len, zero);

        let wg_ptr = f.global(gv_wg);
        let wg_elem = f.index(wg_ptr, local_id);
        f.emit(wg_elem, wg_elem);
        f.f.body.push(
            Statement::If {
                condition: i_lt_len,
                accept: Block::from_vec(vec![f.store(wg_elem, src_val)]),
                reject: Block::from_vec(vec![f.store(wg_elem, zero)]),
            },
            S,
        );
        f.f.body
            .push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);

        // Reduction loop: for stride = 128; stride > 0; stride /= 2
        let stride_var = f.local_var("stride", b.ty_u32, None);
        let stride_ptr = f.local_ptr(stride_var);
        let init_stride = f.literal_u32(128);
        f.emit(init_stride, init_stride);
        f.f.body.push(f.store(stride_ptr, init_stride), S);

        let mut loop_body = Block::new();
        {
            let stride = f.load(stride_ptr);
            let zero_u = f.literal_u32(0);
            let break_cond = f.binary(BinaryOperator::LessEqual, stride, zero_u);

            let cond = f.binary(BinaryOperator::Less, local_id, stride);
            push_emit(&f.f.expressions, &mut loop_body, stride, break_cond);
            loop_body.push(FnBuilder::if_break(break_cond), S);
            push_emit(&f.f.expressions, &mut loop_body, cond, cond);

            // if local_id < stride { wg_data[local_id] += wg_data[local_id + stride] }
            let partner = f.binary(BinaryOperator::Add, local_id, stride);
            let wg_ptr2 = f.global(gv_wg);
            let wg_self = f.index(wg_ptr2, local_id);
            let wg_partner = f.index(wg_ptr2, partner);
            let self_val = f.load(wg_self);
            let partner_val = f.load(wg_partner);
            let new_val = f.binary(BinaryOperator::Add, self_val, partner_val);
            push_emit(&f.f.expressions, &mut loop_body, partner, new_val);

            loop_body.push(
                Statement::If {
                    condition: cond,
                    accept: Block::from_vec(vec![f.store(wg_self, new_val)]),
                    reject: Block::new(),
                },
                S,
            );

            loop_body.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);

            // stride /= 2
            let two = f.literal_u32(2);
            let stride2 = f.load(stride_ptr);
            let next = f.binary(BinaryOperator::Divide, stride2, two);
            push_emit(&f.f.expressions, &mut loop_body, two, next);
            loop_body.push(f.store(stride_ptr, next), S);

            f.f.body.push(
                Statement::Loop {
                    body: loop_body,
                    continuing: Block::new(),
                    break_if: None,
                },
                S,
            );
        }

        // if local_id == 0 { dst[gid.x / WG_SIZE] = wg_data[0]; }
        let zero_u2 = f.literal_u32(0);
        let is_zero = f.binary(BinaryOperator::Equal, local_id, zero_u2);
        let wg_size = f.literal_u32(256);
        let out_idx = f.binary(BinaryOperator::Divide, i, wg_size);
        let wg_ptr3 = f.global(gv_wg);
        let zero_idx = f.literal_u32(0);
        let wg_0 = f.index(wg_ptr3, zero_idx);
        let wg_0_val = f.load(wg_0);

        let dst_ptr = f.global(gv_dst);
        let dst_elem = f.index(dst_ptr, out_idx);

        f.emit(zero_u2, wg_0_val);
        f.emit(dst_elem, dst_elem);

        if is_mean {
            let len_f = f.cast_f32(len);
            let mean_val = f.binary(BinaryOperator::Divide, wg_0_val, len_f);
            f.emit(len_f, mean_val);

            f.f.body.push(
                Statement::If {
                    condition: is_zero,
                    accept: Block::from_vec(vec![f.store(dst_elem, mean_val)]),
                    reject: Block::new(),
                },
                S,
            );
        } else {
            f.f.body.push(
                Statement::If {
                    condition: is_zero,
                    accept: Block::from_vec(vec![f.store(dst_elem, wg_0_val)]),
                    reject: Block::new(),
                },
                S,
            );
        }

        f.finish()
    }

    let sum_fn = gen_reduce_fn(&b, false, gv_src, gv_dst, gv_params, gv_wg);
    b.entry_point("sum_all", [256, 1, 1], sum_fn);

    let mean_fn = gen_reduce_fn(&b, true, gv_src, gv_dst, gv_params, gv_wg);
    b.entry_point("mean_all", [256, 1, 1], mean_fn);

    b.finish()
}

// ---------------------------------------------------------------------------
// softmax.wgsl
// ---------------------------------------------------------------------------

fn gen_softmax() -> Module {
    let mut b = Builder::new();
    let ty_params = b.params_u32x4("Params", &["batch", "features", "_pad0", "_pad1"]);
    let gv_src = b.storage_ro("src");
    let gv_dst = b.storage_rw("dst");
    let gv_params = b.uniform("params", ty_params);

    let mut f = FnBuilder::new(&b);
    let gid = f.arg_gid();
    let row = f.vec_x(gid);
    f.label("row", row);
    f.emit(row, row);

    let params_ptr = f.global(gv_params);
    let batch_ptr = f.field(params_ptr, 0);
    let batch = f.load(batch_ptr);
    let feat_ptr = f.field(params_ptr, 1);
    let features = f.load(feat_ptr);
    f.emit(params_ptr, features);

    let cond = f.binary(BinaryOperator::GreaterEqual, row, batch);
    f.emit(cond, cond);
    f.f.body.push(f.if_return(cond), S);

    // offset = row * features
    let offset = f.named(
        "offset",
        Expression::Binary {
            op: BinaryOperator::Multiply,
            left: row,
            right: features,
        },
    );
    f.emit(offset, offset);

    // Find max
    let src_ptr = f.global(gv_src);
    let src_0 = f.index(src_ptr, offset);
    let src_0_val = f.load(src_0);
    f.emit(src_ptr, src_0_val);

    let max_var = f.local_var("max_val", b.ty_f32, None);
    let max_ptr = f.local_ptr(max_var);
    f.f.body.push(f.store(max_ptr, src_0_val), S);

    // Loop j = 1..features for max
    let j_var = f.local_var("j", b.ty_u32, None);
    let j_ptr = f.local_ptr(j_var);
    let one_u = f.literal_u32(1);
    f.emit(one_u, one_u);
    f.f.body.push(f.store(j_ptr, one_u), S);

    {
        let mut body = Block::new();
        let j = f.load(j_ptr);
        let brk = f.binary(BinaryOperator::GreaterEqual, j, features);
        let idx = f.binary(BinaryOperator::Add, offset, j);
        let src_ptr2 = f.global(gv_src);
        let elem = f.index(src_ptr2, idx);
        let val = f.load(elem);
        let cur_max = f.load(max_ptr);
        let new_max = f.math2(MathFunction::Max, cur_max, val);
        push_emit(&f.f.expressions, &mut body, j, brk);
        body.push(FnBuilder::if_break(brk), S);
        push_emit(&f.f.expressions, &mut body, idx, new_max);
        body.push(f.store(max_ptr, new_max), S);

        let one = f.literal_u32(1);
        let j2 = f.load(j_ptr);
        let j_next = f.binary(BinaryOperator::Add, j2, one);
        push_emit(&f.f.expressions, &mut body, one, j_next);
        body.push(f.store(j_ptr, j_next), S);

        f.f.body.push(
            Statement::Loop {
                body,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    // Compute exp and sum
    let sum_var = f.local_var("sum_exp", b.ty_f32, None);
    let sum_ptr_local = f.local_ptr(sum_var);
    let zero_f = f.literal_f32(0.0);
    f.emit(zero_f, zero_f);
    f.f.body.push(f.store(sum_ptr_local, zero_f), S);

    let j2_var = f.local_var("j2", b.ty_u32, None);
    let j2_ptr = f.local_ptr(j2_var);
    let zero_u = f.literal_u32(0);
    f.emit(zero_u, zero_u);
    f.f.body.push(f.store(j2_ptr, zero_u), S);

    {
        let mut body = Block::new();
        let j = f.load(j2_ptr);
        let brk = f.binary(BinaryOperator::GreaterEqual, j, features);
        let idx = f.binary(BinaryOperator::Add, offset, j);
        let src_ptr2 = f.global(gv_src);
        let elem = f.index(src_ptr2, idx);
        let val = f.load(elem);
        let mx = f.load(max_ptr);
        let diff = f.binary(BinaryOperator::Subtract, val, mx);
        let e = f.math1(MathFunction::Exp, diff);
        push_emit(&f.f.expressions, &mut body, j, brk);
        body.push(FnBuilder::if_break(brk), S);
        push_emit(&f.f.expressions, &mut body, idx, e);

        // dst[offset + j] = e
        let dst_ptr = f.global(gv_dst);
        let dst_elem = f.index(dst_ptr, idx);
        push_emit(&f.f.expressions, &mut body, dst_elem, dst_elem);
        body.push(f.store(dst_elem, e), S);

        // sum_exp += e
        let old_sum = f.load(sum_ptr_local);
        let new_sum = f.binary(BinaryOperator::Add, old_sum, e);
        push_emit(&f.f.expressions, &mut body, old_sum, new_sum);
        body.push(f.store(sum_ptr_local, new_sum), S);

        let one = f.literal_u32(1);
        let j3 = f.load(j2_ptr);
        let j_next = f.binary(BinaryOperator::Add, j3, one);
        push_emit(&f.f.expressions, &mut body, one, j_next);
        body.push(f.store(j2_ptr, j_next), S);

        f.f.body.push(
            Statement::Loop {
                body,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    // Normalize
    let j3_var = f.local_var("j3", b.ty_u32, None);
    let j3_ptr = f.local_ptr(j3_var);
    let zero_u2 = f.literal_u32(0);
    f.emit(zero_u2, zero_u2);
    f.f.body.push(f.store(j3_ptr, zero_u2), S);

    {
        let mut body = Block::new();
        let j = f.load(j3_ptr);
        let brk = f.binary(BinaryOperator::GreaterEqual, j, features);
        let idx = f.binary(BinaryOperator::Add, offset, j);
        let dst_ptr = f.global(gv_dst);
        let dst_elem = f.index(dst_ptr, idx);
        let cur = f.load(dst_elem);
        let sum_val = f.load(sum_ptr_local);
        let normed = f.binary(BinaryOperator::Divide, cur, sum_val);
        push_emit(&f.f.expressions, &mut body, j, brk);
        body.push(FnBuilder::if_break(brk), S);
        push_emit(&f.f.expressions, &mut body, idx, normed);
        body.push(f.store(dst_elem, normed), S);

        let one = f.literal_u32(1);
        let j4 = f.load(j3_ptr);
        let j_next = f.binary(BinaryOperator::Add, j4, one);
        push_emit(&f.f.expressions, &mut body, one, j_next);
        body.push(f.store(j3_ptr, j_next), S);

        f.f.body.push(
            Statement::Loop {
                body,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    b.entry_point("main", [256, 1, 1], f.finish());
    b.finish()
}

// ---------------------------------------------------------------------------
// cross_entropy.wgsl
// ---------------------------------------------------------------------------

fn gen_cross_entropy() -> Module {
    let mut b = Builder::new();
    let ty_params = b.params_u32x4("Params", &["batch", "features", "_pad0", "_pad1"]);
    let gv_logits = b.storage_ro("logits");
    let gv_labels = b.storage_ro("labels");
    let gv_grad = b.storage_rw("grad_out");
    let gv_loss = b.storage_rw("loss_out");
    let gv_params = b.uniform("params", ty_params);

    let mut f = FnBuilder::new(&b);
    let _gid = f.arg_gid();

    let params_ptr = f.global(gv_params);
    let batch_ptr = f.field(params_ptr, 0);
    let batch = f.load(batch_ptr);
    let feat_ptr = f.field(params_ptr, 1);
    let features = f.load(feat_ptr);
    f.emit(params_ptr, features);

    // var total_loss = 0.0;
    let zero_f = f.literal_f32(0.0);
    f.emit(zero_f, zero_f);
    let loss_var = f.local_var("total_loss", b.ty_f32, None);
    let loss_ptr = f.local_ptr(loss_var);
    f.f.body.push(f.store(loss_ptr, zero_f), S);

    // Outer loop: for b = 0..batch
    let b_var = f.local_var("b_idx", b.ty_u32, None);
    let b_ptr = f.local_ptr(b_var);
    let zero_u = f.literal_u32(0);
    f.emit(zero_u, zero_u);
    f.f.body.push(f.store(b_ptr, zero_u), S);

    {
        let mut outer = Block::new();
        let bv = f.load(b_ptr);
        let brk = f.binary(BinaryOperator::GreaterEqual, bv, batch);

        // offset = b * features
        let offset = f.binary(BinaryOperator::Multiply, bv, features);
        push_emit(&f.f.expressions, &mut outer, bv, brk);
        outer.push(FnBuilder::if_break(brk), S);
        push_emit(&f.f.expressions, &mut outer, offset, offset);

        // Find max
        let logits_ptr = f.global(gv_logits);
        let first = f.index(logits_ptr, offset);
        let first_val = f.load(first);
        push_emit(&f.f.expressions, &mut outer, logits_ptr, first_val);

        let max_var = f.local_var("max_val", b.ty_f32, None);
        let max_ptr = f.local_ptr(max_var);
        outer.push(f.store(max_ptr, first_val), S);

        // max loop
        let mj_var = f.local_var("mj", b.ty_u32, None);
        let mj_ptr = f.local_ptr(mj_var);
        let one_u = f.literal_u32(1);
        push_emit(&f.f.expressions, &mut outer, one_u, one_u);
        outer.push(f.store(mj_ptr, one_u), S);

        {
            let mut mbody = Block::new();
            let j = f.load(mj_ptr);
            let mbrk = f.binary(BinaryOperator::GreaterEqual, j, features);
            let idx = f.binary(BinaryOperator::Add, offset, j);
            let lp = f.global(gv_logits);
            let elem = f.index(lp, idx);
            let val = f.load(elem);
            let cur = f.load(max_ptr);
            let nmax = f.math2(MathFunction::Max, cur, val);
            push_emit(&f.f.expressions, &mut mbody, j, mbrk);
            mbody.push(FnBuilder::if_break(mbrk), S);
            push_emit(&f.f.expressions, &mut mbody, idx, nmax);
            mbody.push(f.store(max_ptr, nmax), S);

            let one = f.literal_u32(1);
            let j2 = f.load(mj_ptr);
            let jn = f.binary(BinaryOperator::Add, j2, one);
            push_emit(&f.f.expressions, &mut mbody, one, jn);
            mbody.push(f.store(mj_ptr, jn), S);

            outer.push(
                Statement::Loop {
                    body: mbody,
                    continuing: Block::new(),
                    break_if: None,
                },
                S,
            );
        }

        // Log-sum-exp: sum_exp
        let se_var = f.local_var("sum_exp", b.ty_f32, None);
        let se_ptr = f.local_ptr(se_var);
        let zero_f2 = f.literal_f32(0.0);
        push_emit(&f.f.expressions, &mut outer, zero_f2, zero_f2);
        outer.push(f.store(se_ptr, zero_f2), S);

        let sj_var = f.local_var("sj", b.ty_u32, None);
        let sj_ptr = f.local_ptr(sj_var);
        let zero_u2 = f.literal_u32(0);
        push_emit(&f.f.expressions, &mut outer, zero_u2, zero_u2);
        outer.push(f.store(sj_ptr, zero_u2), S);

        {
            let mut sbody = Block::new();
            let j = f.load(sj_ptr);
            let sbrk = f.binary(BinaryOperator::GreaterEqual, j, features);
            let idx = f.binary(BinaryOperator::Add, offset, j);
            let lp = f.global(gv_logits);
            let elem = f.index(lp, idx);
            let val = f.load(elem);
            let mx = f.load(max_ptr);
            let diff = f.binary(BinaryOperator::Subtract, val, mx);
            let e = f.math1(MathFunction::Exp, diff);
            let old = f.load(se_ptr);
            let nsum = f.binary(BinaryOperator::Add, old, e);
            push_emit(&f.f.expressions, &mut sbody, j, sbrk);
            sbody.push(FnBuilder::if_break(sbrk), S);
            push_emit(&f.f.expressions, &mut sbody, idx, nsum);
            sbody.push(f.store(se_ptr, nsum), S);

            let one = f.literal_u32(1);
            let j2 = f.load(sj_ptr);
            let jn = f.binary(BinaryOperator::Add, j2, one);
            push_emit(&f.f.expressions, &mut sbody, one, jn);
            sbody.push(f.store(sj_ptr, jn), S);

            outer.push(
                Statement::Loop {
                    body: sbody,
                    continuing: Block::new(),
                    break_if: None,
                },
                S,
            );
        }

        // log_sum_exp = log(sum_exp) + max_val
        let se_val = f.load(se_ptr);
        let log_se = f.math1(MathFunction::Log, se_val);
        let mx = f.load(max_ptr);
        let lse = f.binary(BinaryOperator::Add, log_se, mx);
        push_emit(&f.f.expressions, &mut outer, se_val, lse);

        // Loss + gradient loop
        let lj_var = f.local_var("lj", b.ty_u32, None);
        let lj_ptr = f.local_ptr(lj_var);
        let zero_u3 = f.literal_u32(0);
        push_emit(&f.f.expressions, &mut outer, zero_u3, zero_u3);
        outer.push(f.store(lj_ptr, zero_u3), S);

        {
            let mut lbody = Block::new();
            let j = f.load(lj_ptr);
            let lbrk = f.binary(BinaryOperator::GreaterEqual, j, features);
            let idx = f.binary(BinaryOperator::Add, offset, j);

            // log_softmax = logits[idx] - log_sum_exp
            let lp = f.global(gv_logits);
            let elem = f.index(lp, idx);
            let logit_val = f.load(elem);
            let log_softmax = f.binary(BinaryOperator::Subtract, logit_val, lse);
            let softmax_val = f.math1(MathFunction::Exp, log_softmax);

            // label
            let lab_ptr = f.global(gv_labels);
            let lab_elem = f.index(lab_ptr, idx);
            let label_val = f.load(lab_elem);

            // total_loss -= label * log_softmax
            let prod = f.binary(BinaryOperator::Multiply, label_val, log_softmax);
            let old_loss = f.load(loss_ptr);
            let new_loss = f.binary(BinaryOperator::Subtract, old_loss, prod);

            // grad = softmax - label
            let grad_val = f.binary(BinaryOperator::Subtract, softmax_val, label_val);

            push_emit(&f.f.expressions, &mut lbody, j, lbrk);
            lbody.push(FnBuilder::if_break(lbrk), S);
            push_emit(&f.f.expressions, &mut lbody, idx, grad_val);
            lbody.push(f.store(loss_ptr, new_loss), S);

            let grad_ptr = f.global(gv_grad);
            let grad_elem = f.index(grad_ptr, idx);
            push_emit(&f.f.expressions, &mut lbody, grad_elem, grad_elem);
            lbody.push(f.store(grad_elem, grad_val), S);

            let one = f.literal_u32(1);
            let j2 = f.load(lj_ptr);
            let jn = f.binary(BinaryOperator::Add, j2, one);
            push_emit(&f.f.expressions, &mut lbody, one, jn);
            lbody.push(f.store(lj_ptr, jn), S);

            outer.push(
                Statement::Loop {
                    body: lbody,
                    continuing: Block::new(),
                    break_if: None,
                },
                S,
            );
        }

        // b++
        let one_b = f.literal_u32(1);
        let bv2 = f.load(b_ptr);
        let bn = f.binary(BinaryOperator::Add, bv2, one_b);
        push_emit(&f.f.expressions, &mut outer, one_b, bn);
        outer.push(f.store(b_ptr, bn), S);

        f.f.body.push(
            Statement::Loop {
                body: outer,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    // loss_out[0] = total_loss / f32(batch)
    let total = f.load(loss_ptr);
    let batch_f = f.cast_f32(batch);
    let avg = f.binary(BinaryOperator::Divide, total, batch_f);
    let loss_out_ptr = f.global(gv_loss);
    let zero_idx = f.literal_u32(0);
    let loss_elem = f.index(loss_out_ptr, zero_idx);
    f.emit(total, loss_elem);
    f.f.body.push(f.store(loss_elem, avg), S);

    b.entry_point("main", [1, 1, 1], f.finish());
    b.finish()
}

// ---------------------------------------------------------------------------
// rms_norm.wgsl: y[i,j] = x[i,j] / sqrt(mean(x[i,:]²) + eps) * weight[j]
// ---------------------------------------------------------------------------

/// Parallel RMSNorm: one workgroup (256 threads) per row.
///
/// Each thread handles cols/256 elements, then tree-reduces the partial
/// sums-of-squares in shared memory. Much faster than the serial version
/// for wide rows (e.g. 720 columns in SmolVLA).
///
/// Dispatch: [rows, 1, 1].
fn gen_rms_norm() -> Module {
    const WG: u32 = 256;

    let mut b = Builder::new();
    // params: rows, cols, eps_bits, _pad
    let ty_params = b.params_u32x4("Params", &["rows", "cols", "eps_bits", "_pad"]);
    let gv_src = b.storage_ro("src");
    let gv_weight = b.storage_ro("bias"); // named "bias" to match blade binding
    let gv_dst = b.storage_rw("dst");
    let gv_params = b.uniform("params", ty_params);
    let gv_shared = b.workgroup_array("shared", WG);

    let mut f = FnBuilder::new(&b);
    let wgid = f.arg_wgid();
    let lid = f.arg_lid();
    let row = f.vec_x(wgid);
    let tid = f.vec_x(lid);
    f.emit(row, tid);

    // Load params
    let params_ptr = f.global(gv_params);
    let rows_ptr = f.field(params_ptr, 0);
    let rows = f.load(rows_ptr);
    let cols_ptr = f.field(params_ptr, 1);
    let cols = f.load(cols_ptr);
    let eps_ptr = f.field(params_ptr, 2);
    let eps_bits = f.load(eps_ptr);
    let eps = f.expr(Expression::As {
        expr: eps_bits,
        kind: ScalarKind::Float,
        convert: None, // bitcast
    });
    f.emit(params_ptr, eps);

    // Early return for extra workgroups
    let cond = f.binary(BinaryOperator::GreaterEqual, row, rows);
    f.emit(cond, cond);
    f.f.body.push(f.if_return(cond), S);

    // offset = row * cols
    let offset = f.binary(BinaryOperator::Multiply, row, cols);
    f.emit(offset, offset);

    // Phase 1: Each thread accumulates partial sum of squares
    // for (j = tid; j < cols; j += WG) { partial_ss += src[offset+j]² }
    let ss_var = f.local_var("ss", b.ty_f32, None);
    let ss_ptr = f.local_ptr(ss_var);
    let zero_f = f.literal_f32(0.0);
    f.emit(zero_f, zero_f);
    f.f.body.push(f.store(ss_ptr, zero_f), S);

    let j_var = f.local_var("j", b.ty_u32, None);
    let j_ptr = f.local_ptr(j_var);
    f.f.body.push(f.store(j_ptr, tid), S); // j starts at tid

    {
        let mut body = Block::new();
        let j = f.load(j_ptr);
        let brk = f.binary(BinaryOperator::GreaterEqual, j, cols);
        push_emit(&f.f.expressions, &mut body, j, brk);
        body.push(FnBuilder::if_break(brk), S);

        let idx = f.binary(BinaryOperator::Add, offset, j);
        let src_ptr = f.global(gv_src);
        let elem = f.index(src_ptr, idx);
        let val = f.load(elem);
        let sq = f.binary(BinaryOperator::Multiply, val, val);
        let old_ss = f.load(ss_ptr);
        let new_ss = f.binary(BinaryOperator::Add, old_ss, sq);
        push_emit(&f.f.expressions, &mut body, idx, new_ss);
        body.push(f.store(ss_ptr, new_ss), S);

        // j += WG
        let wg_size = f.literal_u32(WG);
        let j2 = f.load(j_ptr);
        let jn = f.binary(BinaryOperator::Add, j2, wg_size);
        push_emit(&f.f.expressions, &mut body, wg_size, jn);
        body.push(f.store(j_ptr, jn), S);

        f.f.body.push(
            Statement::Loop {
                body,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    // Store partial sum to shared memory
    let partial_ss = f.load(ss_ptr);
    let sh_ptr = f.global(gv_shared);
    let sh_elem = f.index(sh_ptr, tid);
    f.emit(partial_ss, sh_elem);
    f.f.body.push(f.store(sh_elem, partial_ss), S);
    f.f.body
        .push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);

    // Phase 2: Tree reduction in shared memory
    // for (stride = 128; stride > 0; stride /= 2)
    let stride_var = f.local_var("stride", b.ty_u32, None);
    let stride_ptr = f.local_ptr(stride_var);
    let init_stride = f.literal_u32(WG / 2);
    f.emit(init_stride, init_stride);
    f.f.body.push(f.store(stride_ptr, init_stride), S);

    {
        let mut body = Block::new();
        let stride = f.load(stride_ptr);
        let zero_s = f.literal_u32(0);
        let brk = f.binary(BinaryOperator::LessEqual, stride, zero_s);
        push_emit(&f.f.expressions, &mut body, stride, brk);
        body.push(FnBuilder::if_break(brk), S);

        // if (tid < stride) { shared[tid] += shared[tid + stride] }
        let cond_r = f.binary(BinaryOperator::Less, tid, stride);
        let partner = f.binary(BinaryOperator::Add, tid, stride);
        let sh_ptr2 = f.global(gv_shared);
        let sh_self = f.index(sh_ptr2, tid);
        let sh_partner = f.index(sh_ptr2, partner);
        let self_val = f.load(sh_self);
        let partner_val = f.load(sh_partner);
        let sum = f.binary(BinaryOperator::Add, self_val, partner_val);
        push_emit(&f.f.expressions, &mut body, cond_r, sum);

        let mut accept = Block::new();
        accept.push(f.store(sh_self, sum), S);
        body.push(
            Statement::If {
                condition: cond_r,
                accept,
                reject: Block::new(),
            },
            S,
        );

        body.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);

        // stride /= 2
        let two = f.literal_u32(2);
        let s2 = f.load(stride_ptr);
        let next = f.binary(BinaryOperator::Divide, s2, two);
        push_emit(&f.f.expressions, &mut body, two, next);
        body.push(f.store(stride_ptr, next), S);

        f.f.body.push(
            Statement::Loop {
                body,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    // Phase 3: Compute rsqrt from shared[0]
    let sh_ptr3 = f.global(gv_shared);
    let zero_idx = f.literal_u32(0);
    let sh_zero = f.index(sh_ptr3, zero_idx);
    let total_ss = f.load(sh_zero);
    let cols_f = f.cast_f32(cols);
    let mean_sq = f.binary(BinaryOperator::Divide, total_ss, cols_f);
    let mean_sq_eps = f.binary(BinaryOperator::Add, mean_sq, eps);
    let rsqrt = f.math1(MathFunction::InverseSqrt, mean_sq_eps);
    f.emit(sh_ptr3, rsqrt);

    // Phase 4: Normalize output
    // for (j = tid; j < cols; j += WG) { dst[offset+j] = src[offset+j] * rsqrt * weight[j] }
    let j2_var = f.local_var("j2", b.ty_u32, None);
    let j2_ptr = f.local_ptr(j2_var);
    f.f.body.push(f.store(j2_ptr, tid), S);

    {
        let mut body = Block::new();
        let j = f.load(j2_ptr);
        let brk = f.binary(BinaryOperator::GreaterEqual, j, cols);
        push_emit(&f.f.expressions, &mut body, j, brk);
        body.push(FnBuilder::if_break(brk), S);

        let idx = f.binary(BinaryOperator::Add, offset, j);
        let src_ptr = f.global(gv_src);
        let elem = f.index(src_ptr, idx);
        let val = f.load(elem);
        let normed = f.binary(BinaryOperator::Multiply, val, rsqrt);
        let w_ptr = f.global(gv_weight);
        let w_elem = f.index(w_ptr, j);
        let w_val = f.load(w_elem);
        let result = f.binary(BinaryOperator::Multiply, normed, w_val);
        push_emit(&f.f.expressions, &mut body, idx, result);

        let dst_ptr = f.global(gv_dst);
        let dst_elem = f.index(dst_ptr, idx);
        push_emit(&f.f.expressions, &mut body, dst_elem, dst_elem);
        body.push(f.store(dst_elem, result), S);

        // j += WG
        let wg_size = f.literal_u32(WG);
        let j3 = f.load(j2_ptr);
        let jn = f.binary(BinaryOperator::Add, j3, wg_size);
        push_emit(&f.f.expressions, &mut body, wg_size, jn);
        body.push(f.store(j2_ptr, jn), S);

        f.f.body.push(
            Statement::Loop {
                body,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    b.entry_point("main", [WG, 1, 1], f.finish());
    b.finish()
}

// ---------------------------------------------------------------------------
// embedding.wgsl: dst[i*hidden+j] = table[indices[i]*hidden+j]
// ---------------------------------------------------------------------------

fn gen_embedding() -> Module {
    let mut b = Builder::new();
    let ty_params = b.params_u32x4("Params", &["seq", "hidden", "_pad0", "_pad1"]);
    let gv_indices = b.storage_ro_u32();
    let gv_table = b.storage_ro("src");
    let gv_dst = b.storage_rw("dst");
    let gv_params = b.uniform("params", ty_params);

    let mut f = FnBuilder::new(&b);
    let gid = f.arg_gid();
    let i = f.vec_x(gid);
    f.label("i", i);
    f.emit(i, i);

    let params_ptr = f.global(gv_params);
    let seq_ptr = f.field(params_ptr, 0);
    let seq = f.load(seq_ptr);
    let hidden_ptr = f.field(params_ptr, 1);
    let hidden = f.load(hidden_ptr);
    f.emit(params_ptr, hidden);

    // total = seq * hidden
    let total = f.binary(BinaryOperator::Multiply, seq, hidden);
    let cond = f.binary(BinaryOperator::GreaterEqual, i, total);
    f.emit(total, cond);
    f.f.body.push(f.if_return(cond), S);

    // row = i / hidden, col = i % hidden
    let row = f.binary(BinaryOperator::Divide, i, hidden);
    let col = f.binary(BinaryOperator::Modulo, i, hidden);

    // token_id = indices[row]
    let idx_ptr = f.global(gv_indices);
    let idx_elem = f.index(idx_ptr, row);
    let token_id = f.load(idx_elem);

    // src_idx = token_id * hidden + col
    let tok_off = f.binary(BinaryOperator::Multiply, token_id, hidden);
    let src_idx = f.binary(BinaryOperator::Add, tok_off, col);
    let tbl_ptr = f.global(gv_table);
    let tbl_elem = f.index(tbl_ptr, src_idx);
    let val = f.load(tbl_elem);

    let dst_ptr = f.global(gv_dst);
    let dst_elem = f.index(dst_ptr, i);

    f.emit(row, val);
    f.emit(dst_elem, dst_elem);
    f.f.body.push(f.store(dst_elem, val), S);

    b.entry_point("main", [256, 1, 1], f.finish());
    b.finish()
}

// ---------------------------------------------------------------------------
// rope.wgsl: Rotary position embeddings
// For each position pos and pair (2i, 2i+1):
//   cos_t = cos(pos * theta^(-2i/dim))
//   sin_t = sin(pos * theta^(-2i/dim))
//   out[2i]   = x[2i]*cos_t - x[2i+1]*sin_t
//   out[2i+1] = x[2i]*sin_t + x[2i+1]*cos_t
// ---------------------------------------------------------------------------

fn gen_rope() -> Module {
    let mut b = Builder::new();
    // params: seq, dim, theta_bits, _pad
    let ty_params = b.params_u32x4("Params", &["seq", "dim", "theta_bits", "_pad"]);
    let gv_src = b.storage_ro("src");
    let gv_dst = b.storage_rw("dst");
    let gv_params = b.uniform("params", ty_params);

    let mut f = FnBuilder::new(&b);
    let gid = f.arg_gid();
    let i = f.vec_x(gid);
    f.label("i", i);
    f.emit(i, i);

    let params_ptr = f.global(gv_params);
    let seq_ptr = f.field(params_ptr, 0);
    let seq = f.load(seq_ptr);
    let dim_ptr = f.field(params_ptr, 1);
    let dim = f.load(dim_ptr);
    let theta_ptr = f.field(params_ptr, 2);
    let theta_bits = f.load(theta_ptr);
    let theta = f.expr(Expression::As {
        expr: theta_bits,
        kind: ScalarKind::Float,
        convert: None, // bitcast
    });
    f.emit(params_ptr, theta);

    // half_dim = dim / 2
    let two_u = f.literal_u32(2);
    let half_dim = f.binary(BinaryOperator::Divide, dim, two_u);
    let total = f.binary(BinaryOperator::Multiply, seq, half_dim);
    let cond = f.binary(BinaryOperator::GreaterEqual, i, total);
    f.emit(two_u, cond);
    f.f.body.push(f.if_return(cond), S);

    // pos = i / half_dim, pair_idx = i % half_dim
    let pos = f.binary(BinaryOperator::Divide, i, half_dim);
    let pair_idx = f.binary(BinaryOperator::Modulo, i, half_dim);

    // freq = pos * pow(theta, -2.0 * pair_idx / dim)
    let two_f = f.literal_f32(2.0);
    let pair_f = f.cast_f32(pair_idx);
    let dim_f = f.cast_f32(dim);
    let exponent = f.binary(BinaryOperator::Multiply, two_f, pair_f);
    let exponent = f.binary(BinaryOperator::Divide, exponent, dim_f);
    let exponent = f.unary(UnaryOperator::Negate, exponent);
    let inv_freq = f.math2(MathFunction::Pow, theta, exponent);
    let pos_f = f.cast_f32(pos);
    let angle = f.binary(BinaryOperator::Multiply, pos_f, inv_freq);

    let cos_val = f.math1(MathFunction::Cos, angle);
    let sin_val = f.math1(MathFunction::Sin, angle);

    // src indices: base = pos * dim + pair_idx * 2
    let pos_dim = f.binary(BinaryOperator::Multiply, pos, dim);
    let pair2 = f.binary(BinaryOperator::Multiply, pair_idx, two_u);
    let idx0 = f.binary(BinaryOperator::Add, pos_dim, pair2);
    let one_u = f.literal_u32(1);
    let idx1 = f.binary(BinaryOperator::Add, idx0, one_u);

    let src_ptr = f.global(gv_src);
    let s0 = f.index(src_ptr, idx0);
    let v0 = f.load(s0);
    let s1 = f.index(src_ptr, idx1);
    let v1 = f.load(s1);

    // out[idx0] = v0 * cos - v1 * sin
    let v0_cos = f.binary(BinaryOperator::Multiply, v0, cos_val);
    let v1_sin = f.binary(BinaryOperator::Multiply, v1, sin_val);
    let r0 = f.binary(BinaryOperator::Subtract, v0_cos, v1_sin);

    // out[idx1] = v0 * sin + v1 * cos
    let v0_sin = f.binary(BinaryOperator::Multiply, v0, sin_val);
    let v1_cos = f.binary(BinaryOperator::Multiply, v1, cos_val);
    let r1 = f.binary(BinaryOperator::Add, v0_sin, v1_cos);

    let dst_ptr = f.global(gv_dst);
    let d0 = f.index(dst_ptr, idx0);
    let d1 = f.index(dst_ptr, idx1);

    f.emit(pos, r1);
    f.emit(d0, d1);
    f.f.body.push(f.store(d0, r0), S);
    f.f.body.push(f.store(d1, r1), S);

    b.entry_point("main", [256, 1, 1], f.finish());
    b.finish()
}

// ---------------------------------------------------------------------------
// causal_attention.wgsl: Fused multi-head causal attention with GQA
// One workgroup per (position, head) pair.
// params: seq, num_heads, num_kv_heads, head_dim
// inputs: q[seq, num_heads*head_dim], k[seq, num_kv_heads*head_dim], v[seq, ...]
// output: [seq, num_heads*head_dim]
// ---------------------------------------------------------------------------

/// Single-pass causal attention with online softmax.
///
/// Computes multi-head causal attention in one pass over key positions,
/// maintaining a running output accumulator. Compared to the 3-pass
/// approach, this reduces compute from O(D²·N) to O(D·N) per (pos, head).
///
/// Algorithm per (pos, head):
///   max_score = -inf, sum_exp = 0, out[d] = 0
///   for t in 0..pos+1:
///     score = Q[pos]·K[t] * scale
///     new_max = max(max_score, score)
///     correction = exp(max_score - new_max)
///     sum_exp = sum_exp * correction + exp(score - new_max)
///     for d: out[d] = out[d] * correction + exp(score - new_max) * V[t,d]
///     max_score = new_max
///   for d: dst[d] = out[d] / sum_exp
fn gen_causal_attention() -> Module {
    gen_attention_parallel(true, false)
}

/// Parallel attention kernel: 64 threads per workgroup, one per head_dim element.
/// Dispatch [q_seq, num_heads, 1] workgroups; workgroup_id gives (pos, head), local_id.x is tid.
///
/// Algorithm: single-pass online softmax across KV positions.
/// - All 64 threads compute partial dot Q[tid]*K[t,tid] → store to wg_dot[tid]
/// - 6-stage parallel reduction gives scalar score = sum_d Q[d]*K[t,d]
/// - Online softmax update (same scalar ops for all threads)
/// - Each thread accumulates its own V dimension: out[tid] += weight * V[t,tid]
/// - Output: dst[q_base + tid] = out[tid] / sum_exp
///
/// is_causal: kv_len = pos+1 (causal mask); else kv_len = kv_seq (all KV positions)
/// is_cross:  params = [q_seq, kv_seq, (nh<<16)|nkv, hd]; else [seq, nh, nkv, hd]
fn gen_attention_parallel(is_causal: bool, is_cross: bool) -> Module {
    const WG: u32 = 64; // one thread per head_dim element (head_dim == 64 for SmolVLA)

    let mut b = Builder::new();
    let ty_params = if is_cross {
        b.params_u32x4("Params", &["q_seq", "kv_seq", "packed_heads", "head_dim"])
    } else {
        b.params_u32x4("Params", &["seq", "num_heads", "num_kv_heads", "head_dim"])
    };
    let gv_q = b.storage_ro("src_a");
    let gv_k = b.storage_ro("src_b");
    let gv_v = b.storage_ro("bias");
    let gv_dst = b.storage_rw("dst");
    let gv_params = b.uniform("params", ty_params);
    // Workgroup shared memory for parallel dot-product reduction
    let gv_wg = b.workgroup_array("wg_dot", WG);

    let mut f = FnBuilder::new(&b);
    // pos = workgroup_id.x, head = workgroup_id.y — same for all 64 threads in workgroup.
    // tid  = local_invocation_id.x — this thread's head_dim element (0..63).
    let wgid = f.arg_wgid();
    let lid = f.arg_lid();
    let pos = f.vec_x(wgid);
    f.label("pos", pos);
    let head = f.vec_y(wgid);
    f.label("head", head);
    let tid = f.vec_x(lid);
    f.label("tid", tid);
    f.emit(pos, tid);

    // Load params[0] = q_seq (always first field)
    let params_ptr = f.global(gv_params);
    let p0 = f.field(params_ptr, 0);
    let q_seq = f.load(p0);
    f.emit(params_ptr, q_seq);

    // Parse remaining params; compute kv_len for this attention type
    let num_heads: Handle<Expression>;
    let num_kv_heads: Handle<Expression>;
    let head_dim: Handle<Expression>;
    let kv_len: Handle<Expression>;
    if is_cross {
        // params: [q_seq, kv_seq, (num_heads<<16)|num_kv_heads, head_dim]
        let p1 = f.field(params_ptr, 1);
        let kv_seq = f.load(p1);
        let p2 = f.field(params_ptr, 2);
        let packed = f.load(p2);
        let p3 = f.field(params_ptr, 3);
        head_dim = f.load(p3);
        let shift16 = f.literal_u32(16);
        num_heads = f.binary(BinaryOperator::ShiftRight, packed, shift16);
        let mask = f.literal_u32(0xFFFF);
        num_kv_heads = f.binary(BinaryOperator::And, packed, mask);
        kv_len = kv_seq;
        f.emit(p1, num_kv_heads);
    } else {
        // params: [seq, num_heads, num_kv_heads, head_dim]
        let p1 = f.field(params_ptr, 1);
        num_heads = f.load(p1);
        let p2 = f.field(params_ptr, 2);
        num_kv_heads = f.load(p2);
        let p3 = f.field(params_ptr, 3);
        head_dim = f.load(p3);
        if is_causal {
            let one = f.literal_u32(1);
            kv_len = f.binary(BinaryOperator::Add, pos, one);
            f.emit(p1, kv_len);
        } else {
            kv_len = q_seq; // full non-causal self-attention: attend to all positions
            f.emit(p1, head_dim);
        }
    }

    // Bounds check — pos/head are workgroup-uniform, so the return is taken by all
    // threads in the workgroup or none (preserves uniform control flow for barriers).
    let cond_pos = f.binary(BinaryOperator::GreaterEqual, pos, q_seq);
    let cond_head = f.binary(BinaryOperator::GreaterEqual, head, num_heads);
    let cond = f.binary(BinaryOperator::LogicalOr, cond_pos, cond_head);
    f.emit(cond_pos, cond);
    f.f.body.push(f.if_return(cond), S);

    // GQA: kv_head = head / (num_heads / num_kv_heads)
    let heads_per_kv = f.binary(BinaryOperator::Divide, num_heads, num_kv_heads);
    let kv_head = f.binary(BinaryOperator::Divide, head, heads_per_kv);
    let q_dim = f.binary(BinaryOperator::Multiply, num_heads, head_dim);
    let kv_dim = f.binary(BinaryOperator::Multiply, num_kv_heads, head_dim);
    // q_base = pos * q_dim + head * head_dim  (base offset for Q and output)
    let pos_q = f.binary(BinaryOperator::Multiply, pos, q_dim);
    let head_off = f.binary(BinaryOperator::Multiply, head, head_dim);
    let q_base = f.binary(BinaryOperator::Add, pos_q, head_off);
    // kv_head_off = kv_head * head_dim  (offset within a KV row for this head group)
    let kv_head_off = f.binary(BinaryOperator::Multiply, kv_head, head_dim);
    let hd_f = f.cast_f32(head_dim);
    let scale = f.math1(MathFunction::InverseSqrt, hd_f);
    f.emit(heads_per_kv, scale);

    // Preload Q[q_base + tid] once — reused for every KV position
    let q_gp = f.global(gv_q);
    let q_idx = f.binary(BinaryOperator::Add, q_base, tid);
    let q_elem = f.index(q_gp, q_idx);
    let q_val = f.load(q_elem);
    f.emit(q_idx, q_val);

    // Spill loop-invariant values to local vars (accessible from inside loop body)
    let kv_dim_var = f.local_var("kv_dim", b.ty_u32, None);
    let kv_dim_ptr = f.local_ptr(kv_dim_var);
    f.f.body.push(f.store(kv_dim_ptr, kv_dim), S);
    let kv_hd_off_var = f.local_var("kv_hd_off", b.ty_u32, None);
    let kv_hd_off_ptr = f.local_ptr(kv_hd_off_var);
    f.f.body.push(f.store(kv_hd_off_ptr, kv_head_off), S);
    let scale_var = f.local_var("scale", b.ty_f32, None);
    let scale_ptr = f.local_ptr(scale_var);
    f.f.body.push(f.store(scale_ptr, scale), S);
    let kv_len_var = f.local_var("kv_len", b.ty_u32, None);
    let kv_len_ptr = f.local_ptr(kv_len_var);
    f.f.body.push(f.store(kv_len_ptr, kv_len), S);

    // Per-thread running state: output accumulator + online softmax scalars
    let my_out_var = f.local_var("my_out", b.ty_f32, None);
    let my_out_ptr = f.local_ptr(my_out_var);
    let max_var = f.local_var("max_score", b.ty_f32, None);
    let max_ptr = f.local_ptr(max_var);
    let sum_var = f.local_var("sum_exp", b.ty_f32, None);
    let sum_ptr = f.local_ptr(sum_var);
    let zero_f = f.literal_f32(0.0);
    let neg_inf = f.literal_f32(-1.0e30);
    f.emit(zero_f, neg_inf);
    f.f.body.push(f.store(my_out_ptr, zero_f), S);
    f.f.body.push(f.store(max_ptr, neg_inf), S);
    f.f.body.push(f.store(sum_ptr, zero_f), S);

    // T-loop: iterate over all KV positions
    let t_var = f.local_var("t", b.ty_u32, None);
    let t_ptr = f.local_ptr(t_var);
    let zero_u = f.literal_u32(0);
    f.emit(zero_u, zero_u);
    f.f.body.push(f.store(t_ptr, zero_u), S);

    {
        let mut body = Block::new();

        // Break if t >= kv_len
        let t = f.load(t_ptr);
        let kl = f.load(kv_len_ptr);
        let brk = f.binary(BinaryOperator::GreaterEqual, t, kl);
        push_emit(&f.f.expressions, &mut body, t, brk);
        body.push(FnBuilder::if_break(brk), S);

        // k_base = t * kv_dim + kv_head_off
        let kvd = f.load(kv_dim_ptr);
        let t_kv = f.binary(BinaryOperator::Multiply, t, kvd);
        let kho = f.load(kv_hd_off_ptr);
        let k_base = f.binary(BinaryOperator::Add, t_kv, kho);
        push_emit(&f.f.expressions, &mut body, kvd, k_base);

        // Each thread loads K[k_base + tid] and writes partial dot-product to wg_dot[tid]
        let k_idx = f.binary(BinaryOperator::Add, k_base, tid);
        let k_gp = f.global(gv_k);
        let k_elem = f.index(k_gp, k_idx);
        let k_val = f.load(k_elem);
        let partial = f.binary(BinaryOperator::Multiply, q_val, k_val);
        push_emit(&f.f.expressions, &mut body, k_idx, partial);

        let wg_ptr = f.global(gv_wg);
        let wg_tid = f.index(wg_ptr, tid);
        push_emit(&f.f.expressions, &mut body, wg_tid, wg_tid);
        body.push(f.store(wg_tid, partial), S);
        body.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);

        // 6-stage parallel reduction: wg_dot[0] = sum_{d=0}^{63} partial[d]
        // Each stage: threads with tid < stride add their partner's value; then barrier.
        // Barriers are unconditional — uniform control flow preserved.
        for stride_val in [32u32, 16, 8, 4, 2, 1] {
            let stride = f.literal_u32(stride_val);
            let cond_s = f.binary(BinaryOperator::Less, tid, stride);
            let partner = f.binary(BinaryOperator::Add, tid, stride);
            let wg_p = f.global(gv_wg);
            let wg_self = f.index(wg_p, tid);
            let wg_part = f.index(wg_p, partner);
            let sv = f.load(wg_self);
            let pv = f.load(wg_part);
            let reduced = f.binary(BinaryOperator::Add, sv, pv);
            push_emit(&f.f.expressions, &mut body, cond_s, reduced);
            body.push(
                Statement::If {
                    condition: cond_s,
                    accept: Block::from_vec(vec![f.store(wg_self, reduced)]),
                    reject: Block::new(),
                },
                S,
            );
            body.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);
        }

        // All threads read scalar dot-product from wg_dot[0] and compute score
        let sc = f.load(scale_ptr);
        let wg_p0 = f.global(gv_wg);
        let z_idx = f.literal_u32(0);
        let wg_0 = f.index(wg_p0, z_idx);
        let dot_sum = f.load(wg_0);
        let score = f.binary(BinaryOperator::Multiply, dot_sum, sc);
        push_emit(&f.f.expressions, &mut body, sc, score);

        // Online softmax update — identical for all 64 threads (scalar ops on registers)
        let old_max = f.load(max_ptr);
        let new_max = f.math2(MathFunction::Max, old_max, score);
        let correction = f.binary(BinaryOperator::Subtract, old_max, new_max);
        let corr_exp = f.math1(MathFunction::Exp, correction);
        let w_shift = f.binary(BinaryOperator::Subtract, score, new_max);
        let weight = f.math1(MathFunction::Exp, w_shift);
        let old_sum = f.load(sum_ptr);
        let sc_sum = f.binary(BinaryOperator::Multiply, old_sum, corr_exp);
        let new_sum = f.binary(BinaryOperator::Add, sc_sum, weight);
        push_emit(&f.f.expressions, &mut body, old_max, new_sum);
        body.push(f.store(sum_ptr, new_sum), S);
        body.push(f.store(max_ptr, new_max), S);

        // V accumulation: thread tid owns output element d=tid, independent across threads
        // my_out = my_out * corr_exp + weight * V[k_base + tid]
        let v_gp = f.global(gv_v);
        let v_idx = f.binary(BinaryOperator::Add, k_base, tid);
        let v_elem = f.index(v_gp, v_idx);
        let v_val = f.load(v_elem);
        let wv = f.binary(BinaryOperator::Multiply, weight, v_val);
        let old_out = f.load(my_out_ptr);
        let sc_out = f.binary(BinaryOperator::Multiply, old_out, corr_exp);
        let new_out = f.binary(BinaryOperator::Add, sc_out, wv);
        push_emit(&f.f.expressions, &mut body, v_idx, new_out);
        body.push(f.store(my_out_ptr, new_out), S);

        // t++
        let one_t = f.literal_u32(1);
        let t2 = f.load(t_ptr);
        let tn = f.binary(BinaryOperator::Add, t2, one_t);
        push_emit(&f.f.expressions, &mut body, one_t, tn);
        body.push(f.store(t_ptr, tn), S);

        f.f.body.push(
            Statement::Loop {
                body,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    // Normalize and write: dst[q_base + tid] = my_out / sum_exp
    let out_val = f.load(my_out_ptr);
    let sum_val = f.load(sum_ptr);
    let normed = f.binary(BinaryOperator::Divide, out_val, sum_val);
    let dst_idx = f.binary(BinaryOperator::Add, q_base, tid);
    let dst_gp = f.global(gv_dst);
    let dst_elem = f.index(dst_gp, dst_idx);
    f.emit(out_val, dst_elem);
    f.f.body.push(f.store(dst_elem, normed), S);

    b.entry_point("main", [WG, 1, 1], f.finish());
    b.finish()
}

// ---------------------------------------------------------------------------
// layer_norm.wgsl: y[i,j] = (x[i,j] - mean) / sqrt(var + eps) * weight[j] + bias[j]
// ---------------------------------------------------------------------------

fn gen_layer_norm() -> Module {
    let mut b = Builder::new();
    // params: rows, cols, eps_bits, _pad
    let ty_params = b.params_u32x4("Params", &["rows", "cols", "eps_bits", "_pad"]);
    let gv_src = b.storage_ro("src");
    let gv_weight = b.storage_ro("src_b"); // weight
    let gv_bias = b.storage_ro("bias"); // bias
    let gv_dst = b.storage_rw("dst");
    let gv_params = b.uniform("params", ty_params);

    let mut f = FnBuilder::new(&b);
    let gid = f.arg_gid();
    let row = f.vec_x(gid);
    f.label("row", row);
    f.emit(row, row);

    let params_ptr = f.global(gv_params);
    let rows_ptr = f.field(params_ptr, 0);
    let rows = f.load(rows_ptr);
    let cols_ptr = f.field(params_ptr, 1);
    let cols = f.load(cols_ptr);
    let eps_ptr = f.field(params_ptr, 2);
    let eps_bits = f.load(eps_ptr);
    let eps = f.expr(Expression::As {
        expr: eps_bits,
        kind: ScalarKind::Float,
        convert: None, // bitcast
    });
    f.emit(params_ptr, eps);

    let cond = f.binary(BinaryOperator::GreaterEqual, row, rows);
    f.emit(cond, cond);
    f.f.body.push(f.if_return(cond), S);

    // offset = row * cols
    let offset = f.binary(BinaryOperator::Multiply, row, cols);
    f.emit(offset, offset);

    // Pass 1: compute mean
    let sum_var = f.local_var("sum", b.ty_f32, None);
    let sum_ptr = f.local_ptr(sum_var);
    let zero_f = f.literal_f32(0.0);
    f.emit(zero_f, zero_f);
    f.f.body.push(f.store(sum_ptr, zero_f), S);

    let j_var = f.local_var("j", b.ty_u32, None);
    let j_ptr = f.local_ptr(j_var);
    let zero_u = f.literal_u32(0);
    f.emit(zero_u, zero_u);
    f.f.body.push(f.store(j_ptr, zero_u), S);

    {
        let mut body = Block::new();
        let j = f.load(j_ptr);
        let brk = f.binary(BinaryOperator::GreaterEqual, j, cols);
        let idx = f.binary(BinaryOperator::Add, offset, j);
        let src_ptr = f.global(gv_src);
        let elem = f.index(src_ptr, idx);
        let val = f.load(elem);
        let old_sum = f.load(sum_ptr);
        let new_sum = f.binary(BinaryOperator::Add, old_sum, val);
        push_emit(&f.f.expressions, &mut body, j, brk);
        body.push(FnBuilder::if_break(brk), S);
        push_emit(&f.f.expressions, &mut body, idx, new_sum);
        body.push(f.store(sum_ptr, new_sum), S);

        let one = f.literal_u32(1);
        let j2 = f.load(j_ptr);
        let jn = f.binary(BinaryOperator::Add, j2, one);
        push_emit(&f.f.expressions, &mut body, one, jn);
        body.push(f.store(j_ptr, jn), S);

        f.f.body.push(
            Statement::Loop {
                body,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    // mean = sum / cols
    let sum_val = f.load(sum_ptr);
    let cols_f = f.cast_f32(cols);
    let mean = f.binary(BinaryOperator::Divide, sum_val, cols_f);
    f.emit(sum_val, mean);

    // Store mean in local var for later use
    let mean_var = f.local_var("mean", b.ty_f32, None);
    let mean_ptr = f.local_ptr(mean_var);
    f.f.body.push(f.store(mean_ptr, mean), S);

    // Pass 2: compute variance
    let var_sum_var = f.local_var("var_sum", b.ty_f32, None);
    let var_sum_ptr = f.local_ptr(var_sum_var);
    let zero_f2 = f.literal_f32(0.0);
    f.emit(zero_f2, zero_f2);
    f.f.body.push(f.store(var_sum_ptr, zero_f2), S);

    let j2_var = f.local_var("j2", b.ty_u32, None);
    let j2_ptr = f.local_ptr(j2_var);
    let zero_u2 = f.literal_u32(0);
    f.emit(zero_u2, zero_u2);
    f.f.body.push(f.store(j2_ptr, zero_u2), S);

    {
        let mut body = Block::new();
        let j = f.load(j2_ptr);
        let brk = f.binary(BinaryOperator::GreaterEqual, j, cols);
        let idx = f.binary(BinaryOperator::Add, offset, j);
        let src_ptr = f.global(gv_src);
        let elem = f.index(src_ptr, idx);
        let val = f.load(elem);
        let m = f.load(mean_ptr);
        let diff = f.binary(BinaryOperator::Subtract, val, m);
        let sq = f.binary(BinaryOperator::Multiply, diff, diff);
        let old_var = f.load(var_sum_ptr);
        let new_var = f.binary(BinaryOperator::Add, old_var, sq);
        push_emit(&f.f.expressions, &mut body, j, brk);
        body.push(FnBuilder::if_break(brk), S);
        push_emit(&f.f.expressions, &mut body, idx, new_var);
        body.push(f.store(var_sum_ptr, new_var), S);

        let one = f.literal_u32(1);
        let j3 = f.load(j2_ptr);
        let jn = f.binary(BinaryOperator::Add, j3, one);
        push_emit(&f.f.expressions, &mut body, one, jn);
        body.push(f.store(j2_ptr, jn), S);

        f.f.body.push(
            Statement::Loop {
                body,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    // rstd = 1.0 / sqrt(var / cols + eps)
    let var_val = f.load(var_sum_ptr);
    let cols_f2 = f.cast_f32(cols);
    let variance = f.binary(BinaryOperator::Divide, var_val, cols_f2);
    let var_eps = f.binary(BinaryOperator::Add, variance, eps);
    let rstd = f.math1(MathFunction::InverseSqrt, var_eps);
    f.emit(var_val, rstd);

    // Pass 3: normalize and apply affine
    let j3_var = f.local_var("j3", b.ty_u32, None);
    let j3_ptr = f.local_ptr(j3_var);
    let zero_u3 = f.literal_u32(0);
    f.emit(zero_u3, zero_u3);
    f.f.body.push(f.store(j3_ptr, zero_u3), S);

    {
        let mut body = Block::new();
        let j = f.load(j3_ptr);
        let brk = f.binary(BinaryOperator::GreaterEqual, j, cols);
        let idx = f.binary(BinaryOperator::Add, offset, j);
        let src_ptr = f.global(gv_src);
        let elem = f.index(src_ptr, idx);
        let val = f.load(elem);
        let m = f.load(mean_ptr);
        let diff = f.binary(BinaryOperator::Subtract, val, m);
        let normed = f.binary(BinaryOperator::Multiply, diff, rstd);
        let w_ptr = f.global(gv_weight);
        let w_elem = f.index(w_ptr, j);
        let w_val = f.load(w_elem);
        let scaled = f.binary(BinaryOperator::Multiply, normed, w_val);
        let b_ptr = f.global(gv_bias);
        let b_elem = f.index(b_ptr, j);
        let b_val = f.load(b_elem);
        let result = f.binary(BinaryOperator::Add, scaled, b_val);
        push_emit(&f.f.expressions, &mut body, j, brk);
        body.push(FnBuilder::if_break(brk), S);
        push_emit(&f.f.expressions, &mut body, idx, result);

        let dst_ptr = f.global(gv_dst);
        let dst_elem = f.index(dst_ptr, idx);
        push_emit(&f.f.expressions, &mut body, dst_elem, dst_elem);
        body.push(f.store(dst_elem, result), S);

        let one = f.literal_u32(1);
        let j4 = f.load(j3_ptr);
        let jn = f.binary(BinaryOperator::Add, j4, one);
        push_emit(&f.f.expressions, &mut body, one, jn);
        body.push(f.store(j3_ptr, jn), S);

        f.f.body.push(
            Statement::Loop {
                body,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    b.entry_point("main", [256, 1, 1], f.finish());
    b.finish()
}

// ---------------------------------------------------------------------------
// full_attention.wgsl: non-causal multi-head attention with GQA
// Same as causal_attention but attends to all positions (no mask).
// ---------------------------------------------------------------------------

fn gen_full_attention() -> Module {
    gen_attention_parallel(false, false)
}

// ---------------------------------------------------------------------------
// cross_attention.wgsl: cross-attention where q and k/v have different seq lengths
// params: q_seq, kv_seq, (num_heads<<16)|num_kv_heads, head_dim
// ---------------------------------------------------------------------------

fn gen_cross_attention() -> Module {
    gen_attention_parallel(false, true)
}

// ---------------------------------------------------------------------------
// multi_head_attn (forward, saves LSE for backward)
// Same as gen_attention_parallel(false, true) but with an extra `lse` binding.
// After normalization, thread 0 writes lse[pos * num_heads + head] = max + log(sum_exp).
// ---------------------------------------------------------------------------

fn gen_mha_forward() -> Module {
    const WG: u32 = 64;

    let mut b = Builder::new();
    // Same param layout as cross-attention: [q_seq, kv_seq, (num_heads<<16)|num_kv_heads, head_dim]
    let ty_params = b.params_u32x4("Params", &["q_seq", "kv_seq", "packed_heads", "head_dim"]);
    let gv_q = b.storage_ro("src_a");
    let gv_k = b.storage_ro("src_b");
    let gv_v = b.storage_ro("bias");
    let gv_dst = b.storage_rw("dst");
    let gv_lse = b.storage_rw("lse"); // EXTRA: LSE output
    let gv_params = b.uniform("params", ty_params);
    let gv_wg = b.workgroup_array("wg_dot", WG);

    let mut f = FnBuilder::new(&b);
    let wgid = f.arg_wgid();
    let lid = f.arg_lid();
    let pos = f.vec_x(wgid);
    f.label("pos", pos);
    let head = f.vec_y(wgid);
    f.label("head", head);
    let tid = f.vec_x(lid);
    f.label("tid", tid);
    f.emit(pos, tid);

    let params_ptr = f.global(gv_params);
    let p0 = f.field(params_ptr, 0);
    let q_seq = f.load(p0);
    f.emit(params_ptr, q_seq);

    // Parse cross-attention style params
    let p1 = f.field(params_ptr, 1);
    let kv_seq = f.load(p1);
    let p2 = f.field(params_ptr, 2);
    let packed = f.load(p2);
    let p3 = f.field(params_ptr, 3);
    let head_dim = f.load(p3);
    let shift16 = f.literal_u32(16);
    let num_heads = f.binary(BinaryOperator::ShiftRight, packed, shift16);
    let mask = f.literal_u32(0xFFFF);
    let num_kv_heads = f.binary(BinaryOperator::And, packed, mask);
    let kv_len = kv_seq;
    f.emit(p1, num_kv_heads);

    let cond_pos = f.binary(BinaryOperator::GreaterEqual, pos, q_seq);
    let cond_head = f.binary(BinaryOperator::GreaterEqual, head, num_heads);
    let cond = f.binary(BinaryOperator::LogicalOr, cond_pos, cond_head);
    f.emit(cond_pos, cond);
    f.f.body.push(f.if_return(cond), S);

    let heads_per_kv = f.binary(BinaryOperator::Divide, num_heads, num_kv_heads);
    let kv_head = f.binary(BinaryOperator::Divide, head, heads_per_kv);
    let q_dim = f.binary(BinaryOperator::Multiply, num_heads, head_dim);
    let kv_dim = f.binary(BinaryOperator::Multiply, num_kv_heads, head_dim);
    let pos_q = f.binary(BinaryOperator::Multiply, pos, q_dim);
    let head_off = f.binary(BinaryOperator::Multiply, head, head_dim);
    let q_base = f.binary(BinaryOperator::Add, pos_q, head_off);
    let kv_head_off = f.binary(BinaryOperator::Multiply, kv_head, head_dim);
    let hd_f = f.cast_f32(head_dim);
    let scale = f.math1(MathFunction::InverseSqrt, hd_f);
    f.emit(heads_per_kv, scale);

    let q_gp = f.global(gv_q);
    let q_idx = f.binary(BinaryOperator::Add, q_base, tid);
    let q_elem = f.index(q_gp, q_idx);
    let q_val = f.load(q_elem);
    f.emit(q_idx, q_val);

    let kv_dim_var = f.local_var("kv_dim", b.ty_u32, None);
    let kv_dim_ptr = f.local_ptr(kv_dim_var);
    f.f.body.push(f.store(kv_dim_ptr, kv_dim), S);
    let kv_hd_off_var = f.local_var("kv_hd_off", b.ty_u32, None);
    let kv_hd_off_ptr = f.local_ptr(kv_hd_off_var);
    f.f.body.push(f.store(kv_hd_off_ptr, kv_head_off), S);
    let scale_var = f.local_var("scale", b.ty_f32, None);
    let scale_ptr = f.local_ptr(scale_var);
    f.f.body.push(f.store(scale_ptr, scale), S);
    let kv_len_var = f.local_var("kv_len", b.ty_u32, None);
    let kv_len_ptr = f.local_ptr(kv_len_var);
    f.f.body.push(f.store(kv_len_ptr, kv_len), S);

    let my_out_var = f.local_var("my_out", b.ty_f32, None);
    let my_out_ptr = f.local_ptr(my_out_var);
    let max_var = f.local_var("max_score", b.ty_f32, None);
    let max_ptr = f.local_ptr(max_var);
    let sum_var = f.local_var("sum_exp", b.ty_f32, None);
    let sum_ptr = f.local_ptr(sum_var);
    let zero_f = f.literal_f32(0.0);
    let neg_inf = f.literal_f32(-1.0e30);
    f.emit(zero_f, neg_inf);
    f.f.body.push(f.store(my_out_ptr, zero_f), S);
    f.f.body.push(f.store(max_ptr, neg_inf), S);
    f.f.body.push(f.store(sum_ptr, zero_f), S);

    let t_var = f.local_var("t", b.ty_u32, None);
    let t_ptr = f.local_ptr(t_var);
    let zero_u = f.literal_u32(0);
    f.emit(zero_u, zero_u);
    f.f.body.push(f.store(t_ptr, zero_u), S);

    {
        let mut body = Block::new();

        let t = f.load(t_ptr);
        let kl = f.load(kv_len_ptr);
        let brk = f.binary(BinaryOperator::GreaterEqual, t, kl);
        push_emit(&f.f.expressions, &mut body, t, brk);
        body.push(FnBuilder::if_break(brk), S);

        let kvd = f.load(kv_dim_ptr);
        let t_kv = f.binary(BinaryOperator::Multiply, t, kvd);
        let kho = f.load(kv_hd_off_ptr);
        let k_base = f.binary(BinaryOperator::Add, t_kv, kho);
        push_emit(&f.f.expressions, &mut body, kvd, k_base);

        let k_idx = f.binary(BinaryOperator::Add, k_base, tid);
        let k_gp = f.global(gv_k);
        let k_elem = f.index(k_gp, k_idx);
        let k_val = f.load(k_elem);
        let partial = f.binary(BinaryOperator::Multiply, q_val, k_val);
        push_emit(&f.f.expressions, &mut body, k_idx, partial);

        let wg_ptr = f.global(gv_wg);
        let wg_tid = f.index(wg_ptr, tid);
        push_emit(&f.f.expressions, &mut body, wg_tid, wg_tid);
        body.push(f.store(wg_tid, partial), S);
        body.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);

        for stride_val in [32u32, 16, 8, 4, 2, 1] {
            let stride = f.literal_u32(stride_val);
            let cond_s = f.binary(BinaryOperator::Less, tid, stride);
            let partner = f.binary(BinaryOperator::Add, tid, stride);
            let wg_p = f.global(gv_wg);
            let wg_self = f.index(wg_p, tid);
            let wg_part = f.index(wg_p, partner);
            let sv = f.load(wg_self);
            let pv = f.load(wg_part);
            let reduced = f.binary(BinaryOperator::Add, sv, pv);
            push_emit(&f.f.expressions, &mut body, cond_s, reduced);
            body.push(
                Statement::If {
                    condition: cond_s,
                    accept: Block::from_vec(vec![f.store(wg_self, reduced)]),
                    reject: Block::new(),
                },
                S,
            );
            body.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);
        }

        let sc = f.load(scale_ptr);
        let wg_p0 = f.global(gv_wg);
        let z_idx = f.literal_u32(0);
        let wg_0 = f.index(wg_p0, z_idx);
        let dot_sum = f.load(wg_0);
        let score = f.binary(BinaryOperator::Multiply, dot_sum, sc);
        push_emit(&f.f.expressions, &mut body, sc, score);

        let old_max = f.load(max_ptr);
        let new_max = f.math2(MathFunction::Max, old_max, score);
        let correction = f.binary(BinaryOperator::Subtract, old_max, new_max);
        let corr_exp = f.math1(MathFunction::Exp, correction);
        let w_shift = f.binary(BinaryOperator::Subtract, score, new_max);
        let weight = f.math1(MathFunction::Exp, w_shift);
        let old_sum = f.load(sum_ptr);
        let sc_sum = f.binary(BinaryOperator::Multiply, old_sum, corr_exp);
        let new_sum = f.binary(BinaryOperator::Add, sc_sum, weight);
        push_emit(&f.f.expressions, &mut body, old_max, new_sum);
        body.push(f.store(sum_ptr, new_sum), S);
        body.push(f.store(max_ptr, new_max), S);

        let v_gp = f.global(gv_v);
        let v_idx = f.binary(BinaryOperator::Add, k_base, tid);
        let v_elem = f.index(v_gp, v_idx);
        let v_val = f.load(v_elem);
        let wv = f.binary(BinaryOperator::Multiply, weight, v_val);
        let old_out = f.load(my_out_ptr);
        let sc_out = f.binary(BinaryOperator::Multiply, old_out, corr_exp);
        let new_out = f.binary(BinaryOperator::Add, sc_out, wv);
        push_emit(&f.f.expressions, &mut body, v_idx, new_out);
        body.push(f.store(my_out_ptr, new_out), S);

        let one_t = f.literal_u32(1);
        let t2 = f.load(t_ptr);
        let tn = f.binary(BinaryOperator::Add, t2, one_t);
        push_emit(&f.f.expressions, &mut body, one_t, tn);
        body.push(f.store(t_ptr, tn), S);

        f.f.body.push(
            Statement::Loop {
                body,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    // Normalize and write: dst[q_base + tid] = my_out / sum_exp
    let out_val = f.load(my_out_ptr);
    let sum_val = f.load(sum_ptr);
    let normed = f.binary(BinaryOperator::Divide, out_val, sum_val);
    let dst_idx = f.binary(BinaryOperator::Add, q_base, tid);
    let dst_gp = f.global(gv_dst);
    let dst_elem = f.index(dst_gp, dst_idx);
    f.emit(out_val, dst_elem);
    f.f.body.push(f.store(dst_elem, normed), S);

    // Write LSE: only thread 0 writes lse[pos * num_heads + head] = max_score + log(sum_exp)
    let max_val = f.load(max_ptr);
    let sum_val_lse = f.load(sum_ptr);
    let log_sum = f.math1(MathFunction::Log, sum_val_lse);
    let lse_val = f.binary(BinaryOperator::Add, max_val, log_sum);
    let lse_pos_off = f.binary(BinaryOperator::Multiply, pos, num_heads);
    let lse_idx = f.binary(BinaryOperator::Add, lse_pos_off, head);
    let lse_gp = f.global(gv_lse);
    let lse_elem = f.index(lse_gp, lse_idx);
    let zero_tid = f.literal_u32(0);
    let tid_is_zero = f.binary(BinaryOperator::Equal, tid, zero_tid);
    f.emit(max_val, tid_is_zero);
    // Only thread 0 writes LSE (all threads computed same max/sum scalars)
    f.f.body.push(
        Statement::If {
            condition: tid_is_zero,
            accept: Block::from_vec(vec![f.store(lse_elem, lse_val)]),
            reject: Block::new(),
        },
        S,
    );

    b.entry_point("main", [WG, 1, 1], f.finish());
    b.finish()
}

// ---------------------------------------------------------------------------
// gen_mha_grad_q: dQ computation for MultiHeadAttn backward
// dispatch [q_seq, num_heads, 1], WG=64
// inputs: [dO, Q, K, V, LSE, O], output: dQ
// ---------------------------------------------------------------------------

fn gen_mha_grad_q() -> Module {
    const WG: u32 = 64;

    let mut b = Builder::new();
    let ty_params = b.params_u32x4("Params", &["q_seq", "kv_seq", "packed_heads", "head_dim"]);
    let gv_d_out = b.storage_ro("d_out"); // dO
    let gv_q = b.storage_ro("src_a"); // Q
    let gv_k = b.storage_ro("src_b"); // K
    let gv_v = b.storage_ro("bias"); // V
    let gv_lse = b.storage_ro("lse"); // LSE (from forward)
    let gv_fwd_o = b.storage_ro("fwd_dst"); // O (from forward)
    let gv_dst = b.storage_rw("dst"); // dQ output
    let gv_params = b.uniform("params", ty_params);
    let gv_wg = b.workgroup_array("wg_dot", WG);

    let mut f = FnBuilder::new(&b);
    let wgid = f.arg_wgid();
    let lid = f.arg_lid();
    let pos = f.vec_x(wgid);
    f.label("pos", pos);
    let head = f.vec_y(wgid);
    f.label("head", head);
    let tid = f.vec_x(lid);
    f.label("tid", tid);
    f.emit(pos, tid);

    let params_ptr = f.global(gv_params);
    let p0 = f.field(params_ptr, 0);
    let q_seq = f.load(p0);
    let p1 = f.field(params_ptr, 1);
    let kv_seq = f.load(p1);
    let p2 = f.field(params_ptr, 2);
    let packed = f.load(p2);
    let p3 = f.field(params_ptr, 3);
    let head_dim = f.load(p3);
    let shift16 = f.literal_u32(16);
    let num_heads = f.binary(BinaryOperator::ShiftRight, packed, shift16);
    let mask = f.literal_u32(0xFFFF);
    let num_kv_heads = f.binary(BinaryOperator::And, packed, mask);
    f.emit(params_ptr, num_kv_heads);

    let cond_pos = f.binary(BinaryOperator::GreaterEqual, pos, q_seq);
    let cond_head = f.binary(BinaryOperator::GreaterEqual, head, num_heads);
    let cond = f.binary(BinaryOperator::LogicalOr, cond_pos, cond_head);
    f.emit(cond_pos, cond);
    f.f.body.push(f.if_return(cond), S);

    // GQA indexing
    let heads_per_kv = f.binary(BinaryOperator::Divide, num_heads, num_kv_heads);
    let kv_head = f.binary(BinaryOperator::Divide, head, heads_per_kv);
    let q_dim = f.binary(BinaryOperator::Multiply, num_heads, head_dim);
    let kv_dim = f.binary(BinaryOperator::Multiply, num_kv_heads, head_dim);
    let pos_q_off = f.binary(BinaryOperator::Multiply, pos, q_dim);
    let head_off = f.binary(BinaryOperator::Multiply, head, head_dim);
    let q_base = f.binary(BinaryOperator::Add, pos_q_off, head_off);
    let kv_head_off = f.binary(BinaryOperator::Multiply, kv_head, head_dim);
    let hd_f = f.cast_f32(head_dim);
    let scale = f.math1(MathFunction::InverseSqrt, hd_f);
    f.emit(heads_per_kv, scale);

    // Spill to local vars
    let kv_dim_var = f.local_var("kv_dim", b.ty_u32, None);
    let kv_dim_ptr = f.local_ptr(kv_dim_var);
    f.f.body.push(f.store(kv_dim_ptr, kv_dim), S);
    let kv_head_off_var = f.local_var("kv_head_off", b.ty_u32, None);
    let kv_head_off_ptr = f.local_ptr(kv_head_off_var);
    f.f.body.push(f.store(kv_head_off_ptr, kv_head_off), S);
    let scale_var = f.local_var("scale", b.ty_f32, None);
    let scale_ptr = f.local_ptr(scale_var);
    f.f.body.push(f.store(scale_ptr, scale), S);
    let q_base_var = f.local_var("q_base", b.ty_u32, None);
    let q_base_ptr = f.local_ptr(q_base_var);
    f.f.body.push(f.store(q_base_ptr, q_base), S);

    // Parallel reduce: row_sum = sum_d dO[q_base+d] * O[q_base+d]
    let d_out_gp = f.global(gv_d_out);
    let fwd_o_gp = f.global(gv_fwd_o);
    let q_base_val = f.load(q_base_ptr);
    let do_idx = f.binary(BinaryOperator::Add, q_base_val, tid);
    let do_elem = f.index(d_out_gp, do_idx);
    let do_val = f.load(do_elem);
    let o_idx = f.binary(BinaryOperator::Add, q_base_val, tid);
    let o_elem = f.index(fwd_o_gp, o_idx);
    let o_val = f.load(o_elem);
    let row_partial = f.binary(BinaryOperator::Multiply, do_val, o_val);
    f.emit(q_base_val, row_partial);

    let wg_ptr = f.global(gv_wg);
    let wg_tid = f.index(wg_ptr, tid);
    f.emit(wg_tid, wg_tid);
    f.f.body.push(f.store(wg_tid, row_partial), S);
    f.f.body.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);
    for stride_val in [32u32, 16, 8, 4, 2, 1] {
        let stride = f.literal_u32(stride_val);
        let cond_s = f.binary(BinaryOperator::Less, tid, stride);
        let partner = f.binary(BinaryOperator::Add, tid, stride);
        let wg_p = f.global(gv_wg);
        let wg_self = f.index(wg_p, tid);
        let wg_part = f.index(wg_p, partner);
        let sv = f.load(wg_self);
        let pv = f.load(wg_part);
        let reduced = f.binary(BinaryOperator::Add, sv, pv);
        push_emit(&f.f.expressions, &mut f.f.body, cond_s, reduced);
        f.f.body.push(
            Statement::If {
                condition: cond_s,
                accept: Block::from_vec(vec![f.store(wg_self, reduced)]),
                reject: Block::new(),
            },
            S,
        );
        f.f.body.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);
    }
    let wg_p0 = f.global(gv_wg);
    let z_idx = f.literal_u32(0);
    let wg_0 = f.index(wg_p0, z_idx);
    let row_sum = f.load(wg_0);
    f.emit(row_sum, row_sum);

    // Store row_sum to local var
    let row_sum_var = f.local_var("row_sum", b.ty_f32, None);
    let row_sum_ptr = f.local_ptr(row_sum_var);
    f.f.body.push(f.store(row_sum_ptr, row_sum), S);

    // Load Q[q_base+tid] for score computation
    let q_gp = f.global(gv_q);
    let q_idx = f.binary(BinaryOperator::Add, q_base, tid);
    let q_elem = f.index(q_gp, q_idx);
    let q_val = f.load(q_elem);
    f.emit(q_idx, q_val);
    let q_val_var = f.local_var("q_val", b.ty_f32, None);
    let q_val_ptr = f.local_ptr(q_val_var);
    f.f.body.push(f.store(q_val_ptr, q_val), S);

    // Load dO[q_base+tid] for use in the loop
    let do_idx2 = f.binary(BinaryOperator::Add, q_base, tid);
    let do_elem2 = f.index(d_out_gp, do_idx2);
    let do_val2 = f.load(do_elem2);
    f.emit(do_idx2, do_val2);
    let do_val_var = f.local_var("do_val", b.ty_f32, None);
    let do_val_ptr = f.local_ptr(do_val_var);
    f.f.body.push(f.store(do_val_ptr, do_val2), S);

    // Load LSE[pos * num_heads + head]
    let lse_gp = f.global(gv_lse);
    let lse_pos_off = f.binary(BinaryOperator::Multiply, pos, num_heads);
    let lse_idx = f.binary(BinaryOperator::Add, lse_pos_off, head);
    let lse_elem = f.index(lse_gp, lse_idx);
    let lse_val = f.load(lse_elem);
    f.emit(lse_pos_off, lse_val);
    let lse_var = f.local_var("lse_val", b.ty_f32, None);
    let lse_ptr = f.local_ptr(lse_var);
    f.f.body.push(f.store(lse_ptr, lse_val), S);

    // my_dq accumulator (per thread owns element d=tid)
    let my_dq_var = f.local_var("my_dq", b.ty_f32, None);
    let my_dq_ptr = f.local_ptr(my_dq_var);
    let zero_f = f.literal_f32(0.0);
    f.emit(zero_f, zero_f);
    f.f.body.push(f.store(my_dq_ptr, zero_f), S);

    // KV loop: for t in 0..kv_seq
    let t_var = f.local_var("t", b.ty_u32, None);
    let t_ptr = f.local_ptr(t_var);
    let zero_u = f.literal_u32(0);
    f.emit(zero_u, zero_u);
    f.f.body.push(f.store(t_ptr, zero_u), S);

    {
        let mut body = Block::new();

        let t = f.load(t_ptr);
        let brk = f.binary(BinaryOperator::GreaterEqual, t, kv_seq);
        push_emit(&f.f.expressions, &mut body, t, brk);
        body.push(FnBuilder::if_break(brk), S);

        // k_base = t * kv_dim + kv_head_off
        let kvd = f.load(kv_dim_ptr);
        let t_kv = f.binary(BinaryOperator::Multiply, t, kvd);
        let kho = f.load(kv_head_off_ptr);
        let k_base = f.binary(BinaryOperator::Add, t_kv, kho);
        push_emit(&f.f.expressions, &mut body, kvd, k_base);

        // Parallel reduce score = sum_d Q[q_base+d] * K[k_base+d] * scale
        let qv = f.load(q_val_ptr);
        let k_gp = f.global(gv_k);
        let k_idx = f.binary(BinaryOperator::Add, k_base, tid);
        let k_elem = f.index(k_gp, k_idx);
        let k_val = f.load(k_elem);
        let qk_partial = f.binary(BinaryOperator::Multiply, qv, k_val);
        push_emit(&f.f.expressions, &mut body, qv, qk_partial);

        let wg_p2 = f.global(gv_wg);
        let wg_tid2 = f.index(wg_p2, tid);
        push_emit(&f.f.expressions, &mut body, wg_tid2, wg_tid2);
        body.push(f.store(wg_tid2, qk_partial), S);
        body.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);
        for stride_val in [32u32, 16, 8, 4, 2, 1] {
            let stride = f.literal_u32(stride_val);
            let cond_s = f.binary(BinaryOperator::Less, tid, stride);
            let partner = f.binary(BinaryOperator::Add, tid, stride);
            let wg_p = f.global(gv_wg);
            let wg_self = f.index(wg_p, tid);
            let wg_part = f.index(wg_p, partner);
            let sv = f.load(wg_self);
            let pv = f.load(wg_part);
            let reduced = f.binary(BinaryOperator::Add, sv, pv);
            push_emit(&f.f.expressions, &mut body, cond_s, reduced);
            body.push(
                Statement::If {
                    condition: cond_s,
                    accept: Block::from_vec(vec![f.store(wg_self, reduced)]),
                    reject: Block::new(),
                },
                S,
            );
            body.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);
        }
        let sc2 = f.load(scale_ptr);
        let wg_p3 = f.global(gv_wg);
        let z_idx2 = f.literal_u32(0);
        let wg_02 = f.index(wg_p3, z_idx2);
        let dot_sum2 = f.load(wg_02);
        let score = f.binary(BinaryOperator::Multiply, dot_sum2, sc2);
        push_emit(&f.f.expressions, &mut body, sc2, score);

        // P_t = exp(score - lse)
        let lse2 = f.load(lse_ptr);
        let score_shifted = f.binary(BinaryOperator::Subtract, score, lse2);
        let p_t = f.math1(MathFunction::Exp, score_shifted);
        push_emit(&f.f.expressions, &mut body, lse2, p_t);

        // Parallel reduce dP_t = sum_d dO[q_base+d] * V[k_base+d]
        let dov = f.load(do_val_ptr);
        let v_gp = f.global(gv_v);
        let v_idx = f.binary(BinaryOperator::Add, k_base, tid);
        let v_elem = f.index(v_gp, v_idx);
        let v_val = f.load(v_elem);
        let dov_v = f.binary(BinaryOperator::Multiply, dov, v_val);
        push_emit(&f.f.expressions, &mut body, dov, dov_v);

        let wg_p4 = f.global(gv_wg);
        let wg_tid4 = f.index(wg_p4, tid);
        push_emit(&f.f.expressions, &mut body, wg_tid4, wg_tid4);
        body.push(f.store(wg_tid4, dov_v), S);
        body.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);
        for stride_val in [32u32, 16, 8, 4, 2, 1] {
            let stride = f.literal_u32(stride_val);
            let cond_s = f.binary(BinaryOperator::Less, tid, stride);
            let partner = f.binary(BinaryOperator::Add, tid, stride);
            let wg_p = f.global(gv_wg);
            let wg_self = f.index(wg_p, tid);
            let wg_part = f.index(wg_p, partner);
            let sv = f.load(wg_self);
            let pv = f.load(wg_part);
            let reduced = f.binary(BinaryOperator::Add, sv, pv);
            push_emit(&f.f.expressions, &mut body, cond_s, reduced);
            body.push(
                Statement::If {
                    condition: cond_s,
                    accept: Block::from_vec(vec![f.store(wg_self, reduced)]),
                    reject: Block::new(),
                },
                S,
            );
            body.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);
        }
        let wg_p5 = f.global(gv_wg);
        let z_idx3 = f.literal_u32(0);
        let wg_03 = f.index(wg_p5, z_idx3);
        let dp_t = f.load(wg_03);
        push_emit(&f.f.expressions, &mut body, dp_t, dp_t);

        // dS_t = P_t * (dP_t - row_sum)
        let rs = f.load(row_sum_ptr);
        let dp_minus_rs = f.binary(BinaryOperator::Subtract, dp_t, rs);
        let ds_t = f.binary(BinaryOperator::Multiply, p_t, dp_minus_rs);
        push_emit(&f.f.expressions, &mut body, rs, ds_t);

        // my_dq += dS_t * scale * K[k_base+tid]
        let sc3 = f.load(scale_ptr);
        let k_gp2 = f.global(gv_k);
        let k_idx2 = f.binary(BinaryOperator::Add, k_base, tid);
        let k_elem2 = f.index(k_gp2, k_idx2);
        let k_val2 = f.load(k_elem2);
        let ds_sc = f.binary(BinaryOperator::Multiply, ds_t, sc3);
        let contrib = f.binary(BinaryOperator::Multiply, ds_sc, k_val2);
        let old_dq = f.load(my_dq_ptr);
        let new_dq = f.binary(BinaryOperator::Add, old_dq, contrib);
        push_emit(&f.f.expressions, &mut body, sc3, new_dq);
        body.push(f.store(my_dq_ptr, new_dq), S);

        // t++
        let one_t = f.literal_u32(1);
        let t2 = f.load(t_ptr);
        let tn = f.binary(BinaryOperator::Add, t2, one_t);
        push_emit(&f.f.expressions, &mut body, one_t, tn);
        body.push(f.store(t_ptr, tn), S);

        f.f.body.push(
            Statement::Loop {
                body,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    // Write dst[q_base+tid] = my_dq
    let dq_val = f.load(my_dq_ptr);
    let dst_idx = f.binary(BinaryOperator::Add, q_base, tid);
    let dst_gp = f.global(gv_dst);
    let dst_elem = f.index(dst_gp, dst_idx);
    f.emit(dq_val, dst_elem);
    f.f.body.push(f.store(dst_elem, dq_val), S);

    b.entry_point("main", [WG, 1, 1], f.finish());
    b.finish()
}

// ---------------------------------------------------------------------------
// gen_mha_grad_k: dK computation for MultiHeadAttn backward
// dispatch [kv_seq, num_kv_heads, 1], WG=64
// ---------------------------------------------------------------------------

fn gen_mha_grad_k() -> Module {
    const WG: u32 = 64;

    let mut b = Builder::new();
    let ty_params = b.params_u32x4("Params", &["q_seq", "kv_seq", "packed_heads", "head_dim"]);
    let gv_d_out = b.storage_ro("d_out");
    let gv_q = b.storage_ro("src_a");
    let gv_k = b.storage_ro("src_b");
    let gv_v = b.storage_ro("bias");
    let gv_lse = b.storage_ro("lse");
    let gv_fwd_o = b.storage_ro("fwd_dst");
    let gv_dst = b.storage_rw("dst"); // dK output
    let gv_params = b.uniform("params", ty_params);
    let gv_wg = b.workgroup_array("wg_dot", WG);

    let mut f = FnBuilder::new(&b);
    let wgid = f.arg_wgid();
    let lid = f.arg_lid();
    let t = f.vec_x(wgid); // kv position
    f.label("t", t);
    let kv_head = f.vec_y(wgid);
    f.label("kv_head", kv_head);
    let tid = f.vec_x(lid);
    f.label("tid", tid);
    f.emit(t, tid);

    let params_ptr = f.global(gv_params);
    let p0 = f.field(params_ptr, 0);
    let q_seq = f.load(p0);
    let p1 = f.field(params_ptr, 1);
    let kv_seq = f.load(p1);
    let p2 = f.field(params_ptr, 2);
    let packed = f.load(p2);
    let p3 = f.field(params_ptr, 3);
    let head_dim = f.load(p3);
    let shift16 = f.literal_u32(16);
    let num_heads = f.binary(BinaryOperator::ShiftRight, packed, shift16);
    let mask = f.literal_u32(0xFFFF);
    let num_kv_heads = f.binary(BinaryOperator::And, packed, mask);
    f.emit(params_ptr, num_kv_heads);

    let cond_t = f.binary(BinaryOperator::GreaterEqual, t, kv_seq);
    let cond_kv = f.binary(BinaryOperator::GreaterEqual, kv_head, num_kv_heads);
    let cond = f.binary(BinaryOperator::LogicalOr, cond_t, cond_kv);
    f.emit(cond_t, cond);
    f.f.body.push(f.if_return(cond), S);

    // kv_base = t * kv_dim + kv_head * head_dim
    let kv_dim = f.binary(BinaryOperator::Multiply, num_kv_heads, head_dim);
    let t_kv = f.binary(BinaryOperator::Multiply, t, kv_dim);
    let kv_head_off = f.binary(BinaryOperator::Multiply, kv_head, head_dim);
    let kv_base = f.binary(BinaryOperator::Add, t_kv, kv_head_off);
    let heads_per_kv = f.binary(BinaryOperator::Divide, num_heads, num_kv_heads);
    let q_dim = f.binary(BinaryOperator::Multiply, num_heads, head_dim);
    let hd_f = f.cast_f32(head_dim);
    let scale = f.math1(MathFunction::InverseSqrt, hd_f);
    f.emit(kv_dim, scale);

    // Spill to local vars
    let kv_base_var = f.local_var("kv_base", b.ty_u32, None);
    let kv_base_ptr = f.local_ptr(kv_base_var);
    f.f.body.push(f.store(kv_base_ptr, kv_base), S);
    let heads_per_kv_var = f.local_var("heads_per_kv", b.ty_u32, None);
    let heads_per_kv_ptr = f.local_ptr(heads_per_kv_var);
    f.f.body.push(f.store(heads_per_kv_ptr, heads_per_kv), S);
    let q_dim_var = f.local_var("q_dim", b.ty_u32, None);
    let q_dim_ptr = f.local_ptr(q_dim_var);
    f.f.body.push(f.store(q_dim_ptr, q_dim), S);
    let scale_var = f.local_var("scale", b.ty_f32, None);
    let scale_ptr = f.local_ptr(scale_var);
    f.f.body.push(f.store(scale_ptr, scale), S);

    // my_dk accumulator
    let my_dk_var = f.local_var("my_dk", b.ty_f32, None);
    let my_dk_ptr = f.local_ptr(my_dk_var);
    let zero_f = f.literal_f32(0.0);
    f.emit(zero_f, zero_f);
    f.f.body.push(f.store(my_dk_ptr, zero_f), S);

    // Outer loop: for pos in 0..q_seq
    let pos_var = f.local_var("pos", b.ty_u32, None);
    let pos_ptr = f.local_ptr(pos_var);
    let zero_u = f.literal_u32(0);
    f.emit(zero_u, zero_u);
    f.f.body.push(f.store(pos_ptr, zero_u), S);

    {
        let mut body_pos = Block::new();

        let pos = f.load(pos_ptr);
        let brk_pos = f.binary(BinaryOperator::GreaterEqual, pos, q_seq);
        push_emit(&f.f.expressions, &mut body_pos, pos, brk_pos);
        body_pos.push(FnBuilder::if_break(brk_pos), S);

        // Inner loop: for head_rel in 0..heads_per_kv
        let head_rel_var = f.local_var("head_rel", b.ty_u32, None);
        let head_rel_ptr = f.local_ptr(head_rel_var);
        let zero_u2 = f.literal_u32(0);
        push_emit(&f.f.expressions, &mut body_pos, zero_u2, zero_u2);
        body_pos.push(f.store(head_rel_ptr, zero_u2), S);

        {
            let mut body_hr = Block::new();

            let head_rel = f.load(head_rel_ptr);
            let hpk = f.load(heads_per_kv_ptr);
            let brk_hr = f.binary(BinaryOperator::GreaterEqual, head_rel, hpk);
            push_emit(&f.f.expressions, &mut body_hr, head_rel, brk_hr);
            body_hr.push(FnBuilder::if_break(brk_hr), S);

            // head = kv_head * heads_per_kv + head_rel
            let kv_h = kv_head;
            let kv_h_hpk = f.binary(BinaryOperator::Multiply, kv_h, hpk);
            let head_cur = f.binary(BinaryOperator::Add, kv_h_hpk, head_rel);
            // q_base = pos * q_dim + head * head_dim
            let qdim = f.load(q_dim_ptr);
            let pos_qdim = f.binary(BinaryOperator::Multiply, pos, qdim);
            let head_hd = f.binary(BinaryOperator::Multiply, head_cur, head_dim);
            let q_base = f.binary(BinaryOperator::Add, pos_qdim, head_hd);
            push_emit(&f.f.expressions, &mut body_hr, kv_h_hpk, q_base);

            // Parallel reduce score = sum_d Q[q_base+d] * K[kv_base+d] * scale
            let q_gp = f.global(gv_q);
            let q_idx = f.binary(BinaryOperator::Add, q_base, tid);
            let q_elem = f.index(q_gp, q_idx);
            let q_val = f.load(q_elem);
            let kvb = f.load(kv_base_ptr);
            let k_gp = f.global(gv_k);
            let k_idx = f.binary(BinaryOperator::Add, kvb, tid);
            let k_elem = f.index(k_gp, k_idx);
            let k_val = f.load(k_elem);
            let qk_partial = f.binary(BinaryOperator::Multiply, q_val, k_val);
            push_emit(&f.f.expressions, &mut body_hr, q_idx, qk_partial);

            let wg_p = f.global(gv_wg);
            let wg_tid = f.index(wg_p, tid);
            push_emit(&f.f.expressions, &mut body_hr, wg_tid, wg_tid);
            body_hr.push(f.store(wg_tid, qk_partial), S);
            body_hr.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);
            for stride_val in [32u32, 16, 8, 4, 2, 1] {
                let stride = f.literal_u32(stride_val);
                let cond_s = f.binary(BinaryOperator::Less, tid, stride);
                let partner = f.binary(BinaryOperator::Add, tid, stride);
                let wg_pp = f.global(gv_wg);
                let wg_self = f.index(wg_pp, tid);
                let wg_part = f.index(wg_pp, partner);
                let sv = f.load(wg_self);
                let pv = f.load(wg_part);
                let reduced = f.binary(BinaryOperator::Add, sv, pv);
                push_emit(&f.f.expressions, &mut body_hr, cond_s, reduced);
                body_hr.push(
                    Statement::If {
                        condition: cond_s,
                        accept: Block::from_vec(vec![f.store(wg_self, reduced)]),
                        reject: Block::new(),
                    },
                    S,
                );
                body_hr.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);
            }
            let sc = f.load(scale_ptr);
            let wg_p0 = f.global(gv_wg);
            let z_idx = f.literal_u32(0);
            let wg_0 = f.index(wg_p0, z_idx);
            let dot_sum = f.load(wg_0);
            let score = f.binary(BinaryOperator::Multiply, dot_sum, sc);
            push_emit(&f.f.expressions, &mut body_hr, sc, score);

            // P_t = exp(score - lse[pos * num_heads + head])
            let lse_gp = f.global(gv_lse);
            let lse_pos_off = f.binary(BinaryOperator::Multiply, pos, num_heads);
            let lse_idx = f.binary(BinaryOperator::Add, lse_pos_off, head_cur);
            let lse_elem = f.index(lse_gp, lse_idx);
            let lse_val = f.load(lse_elem);
            let score_shifted = f.binary(BinaryOperator::Subtract, score, lse_val);
            let p_t = f.math1(MathFunction::Exp, score_shifted);
            push_emit(&f.f.expressions, &mut body_hr, lse_pos_off, p_t);

            // Parallel reduce row_sum = sum_d dO[q_base+d] * O[q_base+d]
            let d_out_gp = f.global(gv_d_out);
            let fwd_o_gp = f.global(gv_fwd_o);
            let do_idx = f.binary(BinaryOperator::Add, q_base, tid);
            let do_elem = f.index(d_out_gp, do_idx);
            let do_val_rs = f.load(do_elem);
            let o_idx = f.binary(BinaryOperator::Add, q_base, tid);
            let o_elem = f.index(fwd_o_gp, o_idx);
            let o_val_rs = f.load(o_elem);
            let rs_partial = f.binary(BinaryOperator::Multiply, do_val_rs, o_val_rs);
            push_emit(&f.f.expressions, &mut body_hr, do_idx, rs_partial);

            let wg_p2 = f.global(gv_wg);
            let wg_tid2 = f.index(wg_p2, tid);
            push_emit(&f.f.expressions, &mut body_hr, wg_tid2, wg_tid2);
            body_hr.push(f.store(wg_tid2, rs_partial), S);
            body_hr.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);
            for stride_val in [32u32, 16, 8, 4, 2, 1] {
                let stride = f.literal_u32(stride_val);
                let cond_s = f.binary(BinaryOperator::Less, tid, stride);
                let partner = f.binary(BinaryOperator::Add, tid, stride);
                let wg_pp = f.global(gv_wg);
                let wg_self = f.index(wg_pp, tid);
                let wg_part = f.index(wg_pp, partner);
                let sv = f.load(wg_self);
                let pv = f.load(wg_part);
                let reduced = f.binary(BinaryOperator::Add, sv, pv);
                push_emit(&f.f.expressions, &mut body_hr, cond_s, reduced);
                body_hr.push(
                    Statement::If {
                        condition: cond_s,
                        accept: Block::from_vec(vec![f.store(wg_self, reduced)]),
                        reject: Block::new(),
                    },
                    S,
                );
                body_hr.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);
            }
            let wg_p3 = f.global(gv_wg);
            let z_idx2 = f.literal_u32(0);
            let wg_02 = f.index(wg_p3, z_idx2);
            let row_sum = f.load(wg_02);
            push_emit(&f.f.expressions, &mut body_hr, row_sum, row_sum);

            // Parallel reduce dP_t = sum_d dO[q_base+d] * V[kv_base+d]
            let do_idx2 = f.binary(BinaryOperator::Add, q_base, tid);
            let do_elem2 = f.index(d_out_gp, do_idx2);
            let do_val2 = f.load(do_elem2);
            let v_gp = f.global(gv_v);
            let v_idx = f.binary(BinaryOperator::Add, kvb, tid);
            let v_elem = f.index(v_gp, v_idx);
            let v_val = f.load(v_elem);
            let dp_partial = f.binary(BinaryOperator::Multiply, do_val2, v_val);
            push_emit(&f.f.expressions, &mut body_hr, do_idx2, dp_partial);

            let wg_p4 = f.global(gv_wg);
            let wg_tid4 = f.index(wg_p4, tid);
            push_emit(&f.f.expressions, &mut body_hr, wg_tid4, wg_tid4);
            body_hr.push(f.store(wg_tid4, dp_partial), S);
            body_hr.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);
            for stride_val in [32u32, 16, 8, 4, 2, 1] {
                let stride = f.literal_u32(stride_val);
                let cond_s = f.binary(BinaryOperator::Less, tid, stride);
                let partner = f.binary(BinaryOperator::Add, tid, stride);
                let wg_pp = f.global(gv_wg);
                let wg_self = f.index(wg_pp, tid);
                let wg_part = f.index(wg_pp, partner);
                let sv = f.load(wg_self);
                let pv = f.load(wg_part);
                let reduced = f.binary(BinaryOperator::Add, sv, pv);
                push_emit(&f.f.expressions, &mut body_hr, cond_s, reduced);
                body_hr.push(
                    Statement::If {
                        condition: cond_s,
                        accept: Block::from_vec(vec![f.store(wg_self, reduced)]),
                        reject: Block::new(),
                    },
                    S,
                );
                body_hr.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);
            }
            let wg_p5 = f.global(gv_wg);
            let z_idx3 = f.literal_u32(0);
            let wg_03 = f.index(wg_p5, z_idx3);
            let dp_t = f.load(wg_03);
            push_emit(&f.f.expressions, &mut body_hr, dp_t, dp_t);

            // dS_t = P_t * (dP_t - row_sum)
            let dp_minus_rs = f.binary(BinaryOperator::Subtract, dp_t, row_sum);
            let ds_t = f.binary(BinaryOperator::Multiply, p_t, dp_minus_rs);
            push_emit(&f.f.expressions, &mut body_hr, dp_minus_rs, ds_t);

            // my_dk += dS_t * scale * Q[q_base+tid]
            let sc2 = f.load(scale_ptr);
            let q_gp2 = f.global(gv_q);
            let q_idx2 = f.binary(BinaryOperator::Add, q_base, tid);
            let q_elem2 = f.index(q_gp2, q_idx2);
            let q_val2 = f.load(q_elem2);
            let ds_sc = f.binary(BinaryOperator::Multiply, ds_t, sc2);
            let contrib = f.binary(BinaryOperator::Multiply, ds_sc, q_val2);
            let old_dk = f.load(my_dk_ptr);
            let new_dk = f.binary(BinaryOperator::Add, old_dk, contrib);
            push_emit(&f.f.expressions, &mut body_hr, sc2, new_dk);
            body_hr.push(f.store(my_dk_ptr, new_dk), S);

            // head_rel++
            let one_hr = f.literal_u32(1);
            let hr2 = f.load(head_rel_ptr);
            let hrn = f.binary(BinaryOperator::Add, hr2, one_hr);
            push_emit(&f.f.expressions, &mut body_hr, one_hr, hrn);
            body_hr.push(f.store(head_rel_ptr, hrn), S);

            body_pos.push(
                Statement::Loop {
                    body: body_hr,
                    continuing: Block::new(),
                    break_if: None,
                },
                S,
            );
        }

        // pos++
        let one_pos = f.literal_u32(1);
        let pos2 = f.load(pos_ptr);
        let posn = f.binary(BinaryOperator::Add, pos2, one_pos);
        push_emit(&f.f.expressions, &mut body_pos, one_pos, posn);
        body_pos.push(f.store(pos_ptr, posn), S);

        f.f.body.push(
            Statement::Loop {
                body: body_pos,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    // Write dst[kv_base+tid] = my_dk
    let dk_val = f.load(my_dk_ptr);
    let kvb_final = f.load(kv_base_ptr);
    let dst_idx = f.binary(BinaryOperator::Add, kvb_final, tid);
    let dst_gp = f.global(gv_dst);
    let dst_elem = f.index(dst_gp, dst_idx);
    f.emit(dk_val, dst_elem);
    f.f.body.push(f.store(dst_elem, dk_val), S);

    b.entry_point("main", [WG, 1, 1], f.finish());
    b.finish()
}

// ---------------------------------------------------------------------------
// gen_mha_grad_v: dV computation for MultiHeadAttn backward
// dispatch [kv_seq, num_kv_heads, 1], WG=64
// ---------------------------------------------------------------------------

fn gen_mha_grad_v() -> Module {
    const WG: u32 = 64;

    let mut b = Builder::new();
    let ty_params = b.params_u32x4("Params", &["q_seq", "kv_seq", "packed_heads", "head_dim"]);
    let gv_d_out = b.storage_ro("d_out");
    let gv_q = b.storage_ro("src_a");
    let gv_k = b.storage_ro("src_b");
    let _gv_v = b.storage_ro("bias"); // V not needed for dV calc but must match binding layout
    let gv_lse = b.storage_ro("lse");
    let _gv_fwd_o = b.storage_ro("fwd_dst"); // O not needed for dV but must match layout
    let gv_dst = b.storage_rw("dst"); // dV output
    let gv_params = b.uniform("params", ty_params);
    let gv_wg = b.workgroup_array("wg_dot", WG);

    let mut f = FnBuilder::new(&b);
    let wgid = f.arg_wgid();
    let lid = f.arg_lid();
    let t = f.vec_x(wgid);
    f.label("t", t);
    let kv_head = f.vec_y(wgid);
    f.label("kv_head", kv_head);
    let tid = f.vec_x(lid);
    f.label("tid", tid);
    f.emit(t, tid);

    let params_ptr = f.global(gv_params);
    let p0 = f.field(params_ptr, 0);
    let q_seq = f.load(p0);
    let p1 = f.field(params_ptr, 1);
    let kv_seq = f.load(p1);
    let p2 = f.field(params_ptr, 2);
    let packed = f.load(p2);
    let p3 = f.field(params_ptr, 3);
    let head_dim = f.load(p3);
    let shift16 = f.literal_u32(16);
    let num_heads = f.binary(BinaryOperator::ShiftRight, packed, shift16);
    let mask = f.literal_u32(0xFFFF);
    let num_kv_heads = f.binary(BinaryOperator::And, packed, mask);
    f.emit(params_ptr, num_kv_heads);

    let cond_t = f.binary(BinaryOperator::GreaterEqual, t, kv_seq);
    let cond_kv = f.binary(BinaryOperator::GreaterEqual, kv_head, num_kv_heads);
    let cond = f.binary(BinaryOperator::LogicalOr, cond_t, cond_kv);
    f.emit(cond_t, cond);
    f.f.body.push(f.if_return(cond), S);

    // kv_base = t * kv_dim + kv_head * head_dim
    let kv_dim = f.binary(BinaryOperator::Multiply, num_kv_heads, head_dim);
    let t_kv = f.binary(BinaryOperator::Multiply, t, kv_dim);
    let kv_head_off = f.binary(BinaryOperator::Multiply, kv_head, head_dim);
    let kv_base = f.binary(BinaryOperator::Add, t_kv, kv_head_off);
    let heads_per_kv = f.binary(BinaryOperator::Divide, num_heads, num_kv_heads);
    let q_dim = f.binary(BinaryOperator::Multiply, num_heads, head_dim);
    let hd_f = f.cast_f32(head_dim);
    let scale = f.math1(MathFunction::InverseSqrt, hd_f);
    f.emit(kv_dim, scale);

    let kv_base_var = f.local_var("kv_base", b.ty_u32, None);
    let kv_base_ptr = f.local_ptr(kv_base_var);
    f.f.body.push(f.store(kv_base_ptr, kv_base), S);
    let heads_per_kv_var = f.local_var("heads_per_kv", b.ty_u32, None);
    let heads_per_kv_ptr = f.local_ptr(heads_per_kv_var);
    f.f.body.push(f.store(heads_per_kv_ptr, heads_per_kv), S);
    let q_dim_var = f.local_var("q_dim", b.ty_u32, None);
    let q_dim_ptr = f.local_ptr(q_dim_var);
    f.f.body.push(f.store(q_dim_ptr, q_dim), S);
    let scale_var = f.local_var("scale", b.ty_f32, None);
    let scale_ptr = f.local_ptr(scale_var);
    f.f.body.push(f.store(scale_ptr, scale), S);

    let my_dv_var = f.local_var("my_dv", b.ty_f32, None);
    let my_dv_ptr = f.local_ptr(my_dv_var);
    let zero_f = f.literal_f32(0.0);
    f.emit(zero_f, zero_f);
    f.f.body.push(f.store(my_dv_ptr, zero_f), S);

    // Outer loop: for pos in 0..q_seq
    let pos_var = f.local_var("pos", b.ty_u32, None);
    let pos_ptr = f.local_ptr(pos_var);
    let zero_u = f.literal_u32(0);
    f.emit(zero_u, zero_u);
    f.f.body.push(f.store(pos_ptr, zero_u), S);

    {
        let mut body_pos = Block::new();

        let pos = f.load(pos_ptr);
        let brk_pos = f.binary(BinaryOperator::GreaterEqual, pos, q_seq);
        push_emit(&f.f.expressions, &mut body_pos, pos, brk_pos);
        body_pos.push(FnBuilder::if_break(brk_pos), S);

        // Inner loop: for head_rel in 0..heads_per_kv
        let head_rel_var = f.local_var("head_rel", b.ty_u32, None);
        let head_rel_ptr = f.local_ptr(head_rel_var);
        let zero_u2 = f.literal_u32(0);
        push_emit(&f.f.expressions, &mut body_pos, zero_u2, zero_u2);
        body_pos.push(f.store(head_rel_ptr, zero_u2), S);

        {
            let mut body_hr = Block::new();

            let head_rel = f.load(head_rel_ptr);
            let hpk = f.load(heads_per_kv_ptr);
            let brk_hr = f.binary(BinaryOperator::GreaterEqual, head_rel, hpk);
            push_emit(&f.f.expressions, &mut body_hr, head_rel, brk_hr);
            body_hr.push(FnBuilder::if_break(brk_hr), S);

            // head = kv_head * heads_per_kv + head_rel
            let kv_h = kv_head;
            let kv_h_hpk = f.binary(BinaryOperator::Multiply, kv_h, hpk);
            let head_cur = f.binary(BinaryOperator::Add, kv_h_hpk, head_rel);
            // q_base = pos * q_dim + head * head_dim
            let qdim = f.load(q_dim_ptr);
            let pos_qdim = f.binary(BinaryOperator::Multiply, pos, qdim);
            let head_hd = f.binary(BinaryOperator::Multiply, head_cur, head_dim);
            let q_base = f.binary(BinaryOperator::Add, pos_qdim, head_hd);
            push_emit(&f.f.expressions, &mut body_hr, kv_h_hpk, q_base);

            // Parallel reduce score = sum_d Q[q_base+d] * K[kv_base+d] * scale
            let q_gp = f.global(gv_q);
            let q_idx = f.binary(BinaryOperator::Add, q_base, tid);
            let q_elem = f.index(q_gp, q_idx);
            let q_val = f.load(q_elem);
            let kvb = f.load(kv_base_ptr);
            let k_gp = f.global(gv_k);
            let k_idx = f.binary(BinaryOperator::Add, kvb, tid);
            let k_elem = f.index(k_gp, k_idx);
            let k_val = f.load(k_elem);
            let qk_partial = f.binary(BinaryOperator::Multiply, q_val, k_val);
            push_emit(&f.f.expressions, &mut body_hr, q_idx, qk_partial);

            let wg_p = f.global(gv_wg);
            let wg_tid = f.index(wg_p, tid);
            push_emit(&f.f.expressions, &mut body_hr, wg_tid, wg_tid);
            body_hr.push(f.store(wg_tid, qk_partial), S);
            body_hr.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);
            for stride_val in [32u32, 16, 8, 4, 2, 1] {
                let stride = f.literal_u32(stride_val);
                let cond_s = f.binary(BinaryOperator::Less, tid, stride);
                let partner = f.binary(BinaryOperator::Add, tid, stride);
                let wg_pp = f.global(gv_wg);
                let wg_self = f.index(wg_pp, tid);
                let wg_part = f.index(wg_pp, partner);
                let sv = f.load(wg_self);
                let pv = f.load(wg_part);
                let reduced = f.binary(BinaryOperator::Add, sv, pv);
                push_emit(&f.f.expressions, &mut body_hr, cond_s, reduced);
                body_hr.push(
                    Statement::If {
                        condition: cond_s,
                        accept: Block::from_vec(vec![f.store(wg_self, reduced)]),
                        reject: Block::new(),
                    },
                    S,
                );
                body_hr.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);
            }
            let sc = f.load(scale_ptr);
            let wg_p0 = f.global(gv_wg);
            let z_idx = f.literal_u32(0);
            let wg_0 = f.index(wg_p0, z_idx);
            let dot_sum = f.load(wg_0);
            let score = f.binary(BinaryOperator::Multiply, dot_sum, sc);
            push_emit(&f.f.expressions, &mut body_hr, sc, score);

            // P_t = exp(score - lse[pos * num_heads + head])
            let lse_gp = f.global(gv_lse);
            let lse_pos_off = f.binary(BinaryOperator::Multiply, pos, num_heads);
            let lse_idx = f.binary(BinaryOperator::Add, lse_pos_off, head_cur);
            let lse_elem = f.index(lse_gp, lse_idx);
            let lse_val = f.load(lse_elem);
            let score_shifted = f.binary(BinaryOperator::Subtract, score, lse_val);
            let p_t = f.math1(MathFunction::Exp, score_shifted);
            push_emit(&f.f.expressions, &mut body_hr, lse_pos_off, p_t);

            // my_dv += P_t * dO[q_base+tid]
            let d_out_gp = f.global(gv_d_out);
            let do_idx = f.binary(BinaryOperator::Add, q_base, tid);
            let do_elem = f.index(d_out_gp, do_idx);
            let do_val = f.load(do_elem);
            let contrib = f.binary(BinaryOperator::Multiply, p_t, do_val);
            let old_dv = f.load(my_dv_ptr);
            let new_dv = f.binary(BinaryOperator::Add, old_dv, contrib);
            push_emit(&f.f.expressions, &mut body_hr, do_idx, new_dv);
            body_hr.push(f.store(my_dv_ptr, new_dv), S);

            // head_rel++
            let one_hr = f.literal_u32(1);
            let hr2 = f.load(head_rel_ptr);
            let hrn = f.binary(BinaryOperator::Add, hr2, one_hr);
            push_emit(&f.f.expressions, &mut body_hr, one_hr, hrn);
            body_hr.push(f.store(head_rel_ptr, hrn), S);

            body_pos.push(
                Statement::Loop {
                    body: body_hr,
                    continuing: Block::new(),
                    break_if: None,
                },
                S,
            );
        }

        // pos++
        let one_pos = f.literal_u32(1);
        let pos2 = f.load(pos_ptr);
        let posn = f.binary(BinaryOperator::Add, pos2, one_pos);
        push_emit(&f.f.expressions, &mut body_pos, one_pos, posn);
        body_pos.push(f.store(pos_ptr, posn), S);

        f.f.body.push(
            Statement::Loop {
                body: body_pos,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    // Write dst[kv_base+tid] = my_dv
    let dv_val = f.load(my_dv_ptr);
    let kvb_final = f.load(kv_base_ptr);
    let dst_idx = f.binary(BinaryOperator::Add, kvb_final, tid);
    let dst_gp = f.global(gv_dst);
    let dst_elem = f.index(dst_gp, dst_idx);
    f.emit(dv_val, dst_elem);
    f.f.body.push(f.store(dst_elem, dv_val), S);

    b.entry_point("main", [WG, 1, 1], f.finish());
    b.finish()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify every shader group generates a valid Naga module.
    #[test]
    fn all_shaders_generate_valid_modules() {
        let groups = [
            (ShaderGroup::Unary, naga::valid::Capabilities::empty()),
            (ShaderGroup::Binary, naga::valid::Capabilities::empty()),
            (ShaderGroup::BiasAdd, naga::valid::Capabilities::empty()),
            (ShaderGroup::Sgd, naga::valid::Capabilities::empty()),
            (ShaderGroup::Transpose, naga::valid::Capabilities::empty()),
            (ShaderGroup::MatMul, naga::valid::Capabilities::empty()),
            (ShaderGroup::MatMulAdd, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::MatMulCoop,
                naga::valid::Capabilities::COOPERATIVE_MATRIX
                    | naga::valid::Capabilities::SHADER_FLOAT16,
            ),
            (
                ShaderGroup::MatMulCoopAdd,
                naga::valid::Capabilities::COOPERATIVE_MATRIX
                    | naga::valid::Capabilities::SHADER_FLOAT16,
            ),
            (ShaderGroup::Reduce, naga::valid::Capabilities::empty()),
            (ShaderGroup::Softmax, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::CrossEntropy,
                naga::valid::Capabilities::empty(),
            ),
            (ShaderGroup::RmsNorm, naga::valid::Capabilities::empty()),
            (ShaderGroup::Embedding, naga::valid::Capabilities::empty()),
            (ShaderGroup::RoPE, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::CausalAttention,
                naga::valid::Capabilities::empty(),
            ),
            (ShaderGroup::LayerNorm, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::FullAttention,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::CrossAttention,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::MultiHeadAttn,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::MultiHeadAttnGradQ,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::MultiHeadAttnGradK,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::MultiHeadAttnGradV,
                naga::valid::Capabilities::empty(),
            ),
        ];

        let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
        for &(group, caps) in &groups {
            let module = generate_module(group);
            naga::valid::Validator::new(flags, caps)
                .validate(&module)
                .unwrap_or_else(|e| {
                    panic!("{group:?}: generated module failed validation: {e:#?}")
                });
        }
    }

    /// Verify the generated modules contain the expected entry points.
    #[test]
    fn entry_points_present() {
        let m = generate_module(ShaderGroup::Unary);
        let names: Vec<&str> = m.entry_points.iter().map(|ep| ep.name.as_str()).collect();
        assert!(names.contains(&"relu"), "missing relu");
        assert!(names.contains(&"sigmoid"), "missing sigmoid");
        assert!(names.contains(&"neg"), "missing neg");
        assert!(names.contains(&"silu"), "missing silu");

        let m = generate_module(ShaderGroup::Binary);
        let names: Vec<&str> = m.entry_points.iter().map(|ep| ep.name.as_str()).collect();
        assert!(names.contains(&"add"));
        assert!(names.contains(&"mul"));
        assert!(names.contains(&"greater"));

        let m = generate_module(ShaderGroup::Reduce);
        let names: Vec<&str> = m.entry_points.iter().map(|ep| ep.name.as_str()).collect();
        assert!(names.contains(&"sum_all"));
        assert!(names.contains(&"mean_all"));
    }

    #[test]
    fn test_rms_norm_wgsl() {
        let _ = generate_wgsl(ShaderGroup::RmsNorm);
    }

    #[test]
    fn test_embedding_wgsl() {
        let _ = generate_wgsl(ShaderGroup::Embedding);
    }

    #[test]
    fn test_rope_wgsl() {
        let _ = generate_wgsl(ShaderGroup::RoPE);
    }

    #[test]
    fn test_causal_attention_wgsl() {
        let _ = generate_wgsl(ShaderGroup::CausalAttention);
    }

    /// Verify every shader group compiles to SPIR-V without panics.
    /// This catches "Expression [N] is not cached!" bugs in hand-built IR.
    #[test]
    fn all_shaders_compile_to_spirv() {
        let empty = naga::valid::Capabilities::empty();
        let coop = naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::SHADER_FLOAT16;
        let groups: &[(ShaderGroup, naga::valid::Capabilities)] = &[
            (ShaderGroup::Unary, empty),
            (ShaderGroup::Binary, empty),
            (ShaderGroup::BiasAdd, empty),
            (ShaderGroup::Sgd, empty),
            (ShaderGroup::Transpose, empty),
            (ShaderGroup::MatMul, empty),
            (ShaderGroup::MatMulAdd, empty),
            (ShaderGroup::MatMulCoop, coop),
            (ShaderGroup::MatMulCoopAdd, coop),
            (ShaderGroup::Reduce, empty),
            (ShaderGroup::Softmax, empty),
            (ShaderGroup::CrossEntropy, empty),
            (ShaderGroup::RmsNorm, empty),
            (ShaderGroup::Embedding, empty),
            (ShaderGroup::RoPE, empty),
            (ShaderGroup::CausalAttention, empty),
            (ShaderGroup::LayerNorm, empty),
            (ShaderGroup::FullAttention, empty),
            (ShaderGroup::CrossAttention, empty),
        ];

        let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
        let options = naga::back::spv::Options {
            lang_version: (1, 0),
            flags: naga::back::spv::WriterFlags::empty(),
            capabilities: None,
            bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
            binding_map: Default::default(),
            ..Default::default()
        };

        let mut failed = Vec::new();
        for &(group, caps) in groups {
            // See note in all_shaders_generate_valid_modules
            if group == ShaderGroup::MatMulCoop || group == ShaderGroup::MatMulCoopAdd {
                continue;
            }
            let module = generate_module(group);
            let info = match naga::valid::Validator::new(flags, caps).validate(&module) {
                Ok(info) => info,
                Err(e) => {
                    failed.push(format!("{group:?}: validation failed: {e}"));
                    continue;
                }
            };
            // Try each entry point
            for ep in &module.entry_points {
                let pipeline_options = naga::back::spv::PipelineOptions {
                    shader_stage: naga::ShaderStage::Compute,
                    entry_point: ep.name.clone(),
                };
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline_options))
                }));
                match result {
                    Ok(Ok(_)) => {}
                    Ok(Err(e)) => failed.push(format!("{group:?}/{}: SPIR-V error: {e}", ep.name)),
                    Err(e) => {
                        let msg = e
                            .downcast_ref::<String>()
                            .map(|s| s.as_str())
                            .or_else(|| e.downcast_ref::<&str>().copied())
                            .unwrap_or("unknown panic");
                        failed.push(format!("{group:?}/{}: SPIR-V panic: {msg}", ep.name));
                    }
                }
            }
        }
        if !failed.is_empty() {
            panic!("SPIR-V compilation failures:\n{}", failed.join("\n"));
        }
    }

    /// Verify that shader global variable names match the runtime ShaderData
    /// struct field names. Blade resolves bindings by name — a mismatch causes
    /// a runtime panic ("Unable to resolve binding for ...").
    #[test]
    fn shader_globals_match_runtime_bindings() {
        use crate::compile::ShaderEntry;
        use std::collections::HashSet;

        // Expected global variable names for each ShaderEntry, derived from
        // the runtime ShaderData structs. Workgroup vars (tile_a, tile_b) and
        // builtin args are not bound by blade and can be ignored.
        fn expected_globals(entry: &ShaderEntry) -> Vec<&'static str> {
            match entry {
                ShaderEntry::MatMul => vec!["matrix_a", "matrix_b", "matrix_c", "params"],
                ShaderEntry::FusedMatMulAdd => {
                    vec!["matrix_a", "matrix_b", "matrix_c", "src", "params"]
                }
                ShaderEntry::Relu
                | ShaderEntry::Sigmoid
                | ShaderEntry::Neg
                | ShaderEntry::Silu
                | ShaderEntry::Gelu
                | ShaderEntry::SumAll
                | ShaderEntry::MeanAll
                | ShaderEntry::RoPE => vec!["src", "dst", "params"],
                ShaderEntry::Add
                | ShaderEntry::Mul
                | ShaderEntry::Greater
                | ShaderEntry::SwiGLU => {
                    vec!["src_a", "src_b", "dst", "params"]
                }
                ShaderEntry::BiasAdd => vec!["src", "bias", "dst", "params"],
                ShaderEntry::SgdUpdate => vec!["param", "grad", "dst", "params"],
                ShaderEntry::Softmax => vec!["src", "dst", "params"],
                ShaderEntry::CrossEntropyLoss => {
                    vec!["logits", "labels", "grad_out", "loss_out", "params"]
                }
                ShaderEntry::Transpose => vec!["src", "dst", "params"],
                ShaderEntry::RmsNorm => vec!["src", "bias", "dst", "params"],
                ShaderEntry::Embedding => vec!["indices", "src", "dst", "params"],
                ShaderEntry::CausalAttention
                | ShaderEntry::FullAttention
                | ShaderEntry::CrossAttention => vec!["src_a", "src_b", "bias", "dst", "params"],
                ShaderEntry::LayerNorm => vec!["src", "src_b", "bias", "dst", "params"],
                ShaderEntry::MultiHeadAttn => {
                    vec!["src_a", "src_b", "bias", "dst", "lse", "params"]
                }
                ShaderEntry::MultiHeadAttnGradQ
                | ShaderEntry::MultiHeadAttnGradK
                | ShaderEntry::MultiHeadAttnGradV => {
                    vec![
                        "d_out", "src_a", "src_b", "bias", "lse", "fwd_dst", "dst", "params",
                    ]
                }
            }
        }

        let entries = [
            ShaderEntry::MatMul,
            ShaderEntry::FusedMatMulAdd,
            ShaderEntry::Relu,
            ShaderEntry::Sigmoid,
            ShaderEntry::Neg,
            ShaderEntry::Add,
            ShaderEntry::Mul,
            ShaderEntry::Greater,
            ShaderEntry::BiasAdd,
            ShaderEntry::SgdUpdate,
            ShaderEntry::SumAll,
            ShaderEntry::MeanAll,
            ShaderEntry::Softmax,
            ShaderEntry::CrossEntropyLoss,
            ShaderEntry::Transpose,
            ShaderEntry::Silu,
            ShaderEntry::RmsNorm,
            ShaderEntry::Embedding,
            ShaderEntry::RoPE,
            ShaderEntry::CausalAttention,
            ShaderEntry::Gelu,
            ShaderEntry::LayerNorm,
            ShaderEntry::FullAttention,
            ShaderEntry::CrossAttention,
            ShaderEntry::MultiHeadAttn,
            ShaderEntry::MultiHeadAttnGradQ,
            ShaderEntry::MultiHeadAttnGradK,
            ShaderEntry::MultiHeadAttnGradV,
        ];

        for entry in &entries {
            let group = entry.shader_group();
            let expected: HashSet<&str> = expected_globals(entry).into_iter().collect();

            let module = generate_module(group);

            let actual: HashSet<&str> = module
                .global_variables
                .iter()
                .filter_map(|(_, gv)| {
                    // Skip workgroup variables — blade doesn't bind those
                    if gv.space == naga::AddressSpace::WorkGroup {
                        return None;
                    }
                    gv.name.as_deref()
                })
                .collect();

            assert_eq!(
                expected, actual,
                "{entry:?} (group {group:?}): shader globals {actual:?} \
                 don't match expected runtime bindings {expected:?}"
            );
        }
    }
}
