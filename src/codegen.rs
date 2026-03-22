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
    MatMulCoop,
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
        ShaderGroup::MatMulCoop => gen_matmul_coop(),
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
    }
}

/// Generate WGSL source for a shader group.
pub fn generate_wgsl(group: ShaderGroup) -> String {
    let module = generate_module(group);
    let capabilities = match group {
        ShaderGroup::MatMulCoop => {
            naga::valid::Capabilities::COOPERATIVE_MATRIX
                | naga::valid::Capabilities::SHADER_FLOAT16
        }
        _ => naga::valid::Capabilities::empty(),
    };
    module_to_wgsl(&module, capabilities)
}

/// Convert a naga Module to WGSL source text.
pub fn module_to_wgsl(module: &Module, capabilities: naga::valid::Capabilities) -> String {
    module_to_wgsl_flags(
        module,
        capabilities,
        naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS,
    )
}

/// Convert a naga Module to WGSL with custom validation flags.
pub fn module_to_wgsl_flags(
    module: &Module,
    capabilities: naga::valid::Capabilities,
    flags: naga::valid::ValidationFlags,
) -> String {
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

/// Naive matmul: C = A × B via Naga IR.
///
/// Each thread computes one element of C by looping over K.
/// Workgroup [16, 16, 1], dispatched as [ceil(N/16), ceil(M/16), 1].
fn gen_matmul() -> Module {
    let mut b = Builder::new();
    let ty_params = b.params_u32x4("Params", &["m", "n", "k", "_pad"]);
    let gv_a = b.storage_ro("matrix_a");
    let gv_b = b.storage_ro("matrix_b");
    let gv_c = b.storage_rw("matrix_c");
    let gv_params = b.uniform("params", ty_params);

    let mut f = FnBuilder::new(&b);
    let gid = f.arg_gid();

    // col = gid.x, row = gid.y — one thread per output element
    let col = f.vec_x(gid);
    let row = f.vec_y(gid);
    f.emit(col, row);

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

    // Loop: for (var i = 0u; i < k; i++)
    let i_var = f.local_var("i", b.ty_u32, None);
    let zero_u = f.literal_u32(0);
    f.emit(zero_u, zero_u);
    let i_ptr = f.local_ptr(i_var);
    f.f.body.push(f.store(i_ptr, zero_u), S);

    let mut loop_body = Block::new();
    {
        // if (i >= k) { break; }
        let i_val = f.load(i_ptr);
        let break_cond = f.binary(BinaryOperator::GreaterEqual, i_val, pk);
        push_emit(&f.f.expressions, &mut loop_body, i_val, break_cond);
        loop_body.push(
            Statement::If {
                condition: break_cond,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            S,
        );

        // sum += a[row * k + i] * b[i * n + col]
        let a_ptr = f.global(gv_a);
        let row_k = f.binary(BinaryOperator::Multiply, row, pk);
        let a_idx = f.binary(BinaryOperator::Add, row_k, i_val);
        let a_elem = f.index(a_ptr, a_idx);
        let a_val = f.load(a_elem);

        let b_ptr = f.global(gv_b);
        let i_n = f.binary(BinaryOperator::Multiply, i_val, pn);
        let b_idx = f.binary(BinaryOperator::Add, i_n, col);
        let b_elem = f.index(b_ptr, b_idx);
        let b_val = f.load(b_elem);

        let prod = f.binary(BinaryOperator::Multiply, a_val, b_val);
        let old_sum = f.load(sum_ptr);
        let new_sum = f.binary(BinaryOperator::Add, old_sum, prod);
        push_emit(&f.f.expressions, &mut loop_body, a_ptr, new_sum);
        loop_body.push(f.store(sum_ptr, new_sum), S);

        // i++
        let one_u = f.literal_u32(1);
        let i_val2 = f.load(i_ptr);
        let i_next = f.binary(BinaryOperator::Add, i_val2, one_u);
        push_emit(&f.f.expressions, &mut loop_body, one_u, i_next);
        loop_body.push(f.store(i_ptr, i_next), S);

        f.f.body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    // if (row < m && col < n) { c[row * n + col] = sum }
    let r_lt_m = f.binary(BinaryOperator::Less, row, pm);
    let c_lt_n = f.binary(BinaryOperator::Less, col, pn);
    let store_cond = f.binary(BinaryOperator::LogicalAnd, r_lt_m, c_lt_n);
    let row_n = f.binary(BinaryOperator::Multiply, row, pn);
    let c_idx = f.binary(BinaryOperator::Add, row_n, col);
    let c_ptr = f.global(gv_c);
    let c_elem = f.index(c_ptr, c_idx);
    let final_val = f.load(sum_ptr);
    f.emit(r_lt_m, final_val);

    f.f.body.push(
        Statement::If {
            condition: store_cond,
            accept: Block::from_vec(vec![f.store(c_elem, final_val)]),
            reject: Block::new(),
        },
        S,
    );

    b.entry_point("main", [16, 16, 1], f.finish());
    b.finish()
}
// ---------------------------------------------------------------------------
// matmul_coop.wgsl — cooperative matrix multiply (8×8 tiles)
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
    const TILE: u32 = 16;
    const TILE_ELEMS: u32 = TILE * TILE; // 256
    const WG_SIZE: u32 = 64;
    const ELEMS_PER_THREAD: u32 = TILE_ELEMS / WG_SIZE; // 4

    let mut b = Builder::new();
    let ty_params = b.params_u32x4("Params", &["m", "n", "k", "_pad"]);
    let gv_a = b.storage_ro("matrix_a");
    let gv_b = b.storage_ro("matrix_b");
    let gv_c = b.storage_rw("matrix_c");
    let gv_params = b.uniform("params", ty_params);

    // Shared memory for f16 staging: array<f32, 128> (packing two f16 per u32 slot,
    // but for simplicity we use f32 array with 128 entries = 256 f16 values)
    // Actually, Naga needs an f16 base type for cooperative load to work.
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
    let gv_sa = b.m.global_variables.append(
        GlobalVariable {
            name: Some("shared_a".to_string()),
            space: AddressSpace::WorkGroup,
            binding: None,
            ty: ty_shared,
            init: None,
            memory_decorations: MemoryDecorations::empty(),
        },
        S,
    );
    let gv_sb = b.m.global_variables.append(
        GlobalVariable {
            name: Some("shared_b".to_string()),
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

    // tile_row = wg.x * 16, tile_col = wg.y * 16
    let tile_c = f.literal_u32(TILE);
    let tile_row = f.binary(BinaryOperator::Multiply, wg_x, tile_c);
    let tile_c2 = f.literal_u32(TILE);
    let tile_col = f.binary(BinaryOperator::Multiply, wg_y, tile_c2);
    f.emit(tile_c, tile_col);

    // Load params
    let params_ptr = f.global(gv_params);
    let m_ptr = f.field(params_ptr, 0);
    let pm = f.load(m_ptr);
    let n_ptr = f.field(params_ptr, 1);
    let pn = f.load(n_ptr);
    let k_ptr = f.field(params_ptr, 2);
    let pk = f.load(k_ptr);
    f.emit(params_ptr, pk);

    // c_offset = tile_row * n + tile_col
    let c_off_base = f.binary(BinaryOperator::Multiply, tile_row, pn);
    let c_offset = f.binary(BinaryOperator::Add, c_off_base, tile_col);
    f.emit(c_off_base, c_offset);

    // Load C tile (f32 cooperative matrix — matches output buffer type)
    let c_ptr = f.global(gv_c);
    let c_elem = f.index(c_ptr, c_offset);
    let acc_load = f.expr(Expression::CooperativeLoad {
        columns: CooperativeSize::Sixteen,
        rows: CooperativeSize::Sixteen,
        role: CooperativeRole::C,
        data: CooperativeData {
            pointer: c_elem,
            stride: pn,
            row_major: false,
        },
    });
    f.emit(c_ptr, acc_load);

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
    let acc_var = f.local_var("acc", ty_coop_c, None);
    let acc_ptr = f.local_ptr(acc_var);
    f.f.body.push(f.store(acc_ptr, acc_load), S);

    // var t: u32 = 0
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

        // --- Stage A tile: f32 global → f16 shared ---
        // Each of 64 threads loads 4 elements (256 total = 16×16 tile)
        let elems = f.literal_u32(ELEMS_PER_THREAD);
        let tile16 = f.literal_u32(TILE);
        let sa_ptr = f.global(gv_sa);
        let a_ptr = f.global(gv_a);
        // Pre-create zero f16 constant for out-of-bounds padding
        let zero_f32_a = f.literal_f32(0.0);
        let zero_f16_a = f.expr(Expression::As {
            expr: zero_f32_a,
            kind: ScalarKind::Float,
            convert: Some(2),
        });
        push_emit(&f.f.expressions, &mut loop_body, elems, zero_f16_a);

        for e in 0..ELEMS_PER_THREAD {
            // flat_idx = tid + e * 64
            let offset = f.literal_u32(e * WG_SIZE);
            let flat_idx = f.binary(BinaryOperator::Add, tid, offset);
            // sa_elem must be created before data loads so it's in the outer scope
            let sa_elem = f.index(sa_ptr, flat_idx);
            // src_row = flat_idx / 16, src_col = flat_idx % 16
            let src_row = f.binary(BinaryOperator::Divide, flat_idx, tile16);
            let src_col = f.binary(BinaryOperator::Modulo, flat_idx, tile16);
            // Bounds: (tile_row + src_row) < m && (t + src_col) < k
            let gr = f.binary(BinaryOperator::Add, tile_row, src_row);
            let tc = f.binary(BinaryOperator::Add, t_val, src_col);
            let in_m = f.binary(BinaryOperator::Less, gr, pm);
            let in_k = f.binary(BinaryOperator::Less, tc, pk);
            let in_bounds = f.binary(BinaryOperator::LogicalAnd, in_m, in_k);
            push_emit(&f.f.expressions, &mut loop_body, offset, in_bounds);
            // Data load + convert only in accept block
            let grk = f.binary(BinaryOperator::Multiply, gr, pk);
            let global_idx = f.binary(BinaryOperator::Add, grk, tc);
            let a_elem = f.index(a_ptr, global_idx);
            let a_val = f.load(a_elem);
            let a_f16 = f.expr(Expression::As {
                expr: a_val,
                kind: ScalarKind::Float,
                convert: Some(2),
            });
            let mut accept = Block::new();
            push_emit(&f.f.expressions, &mut accept, grk, a_f16);
            accept.push(f.store(sa_elem, a_f16), S);
            loop_body.push(
                Statement::If {
                    condition: in_bounds,
                    accept,
                    reject: Block::from_vec(vec![f.store(sa_elem, zero_f16_a)]),
                },
                S,
            );
        }

        // --- Stage B tile: f32 global → f16 shared ---
        let sb_ptr = f.global(gv_sb);
        let b_ptr = f.global(gv_b);
        let zero_f32_b = f.literal_f32(0.0);
        let zero_f16_b = f.expr(Expression::As {
            expr: zero_f32_b,
            kind: ScalarKind::Float,
            convert: Some(2),
        });
        push_emit(&f.f.expressions, &mut loop_body, sb_ptr, zero_f16_b);

        for e in 0..ELEMS_PER_THREAD {
            let offset = f.literal_u32(e * WG_SIZE);
            let flat_idx = f.binary(BinaryOperator::Add, tid, offset);
            let sb_elem = f.index(sb_ptr, flat_idx);
            let src_row = f.binary(BinaryOperator::Divide, flat_idx, tile16);
            let src_col = f.binary(BinaryOperator::Modulo, flat_idx, tile16);
            // Bounds: (t + src_row) < k && (tile_col + src_col) < n
            let tr = f.binary(BinaryOperator::Add, t_val, src_row);
            let cc = f.binary(BinaryOperator::Add, tile_col, src_col);
            let in_k = f.binary(BinaryOperator::Less, tr, pk);
            let in_n = f.binary(BinaryOperator::Less, cc, pn);
            let in_bounds = f.binary(BinaryOperator::LogicalAnd, in_k, in_n);
            push_emit(&f.f.expressions, &mut loop_body, offset, in_bounds);
            // Data load + convert only in accept block
            let trn = f.binary(BinaryOperator::Multiply, tr, pn);
            let global_idx = f.binary(BinaryOperator::Add, trn, cc);
            let b_elem = f.index(b_ptr, global_idx);
            let b_val = f.load(b_elem);
            let b_f16 = f.expr(Expression::As {
                expr: b_val,
                kind: ScalarKind::Float,
                convert: Some(2),
            });
            let mut accept = Block::new();
            push_emit(&f.f.expressions, &mut accept, trn, b_f16);
            accept.push(f.store(sb_elem, b_f16), S);
            loop_body.push(
                Statement::If {
                    condition: in_bounds,
                    accept,
                    reject: Block::from_vec(vec![f.store(sb_elem, zero_f16_b)]),
                },
                S,
            );
        }

        // workgroupBarrier — ensure shared memory is populated
        loop_body.push(Statement::ControlBarrier(Barrier::WORK_GROUP), S);

        // CooperativeLoad f16 tiles from shared memory
        let sa_zero = f.global(gv_sa);
        let zero_idx = f.literal_u32(0);
        let sa_base = f.index(sa_zero, zero_idx);
        let stride16 = f.literal_u32(TILE);
        let a_coop = f.expr(Expression::CooperativeLoad {
            columns: CooperativeSize::Sixteen,
            rows: CooperativeSize::Sixteen,
            role: CooperativeRole::A,
            data: CooperativeData {
                pointer: sa_base,
                stride: stride16,
                row_major: false,
            },
        });

        let sb_zero = f.global(gv_sb);
        let zero_idx2 = f.literal_u32(0);
        let sb_base = f.index(sb_zero, zero_idx2);
        let stride16_2 = f.literal_u32(TILE);
        let b_coop = f.expr(Expression::CooperativeLoad {
            columns: CooperativeSize::Sixteen,
            rows: CooperativeSize::Sixteen,
            role: CooperativeRole::B,
            data: CooperativeData {
                pointer: sb_base,
                stride: stride16_2,
                row_major: false,
            },
        });

        // acc = coopMultiplyAdd(a_f16, b_f16, acc_f32) → f32
        let old_acc = f.load(acc_ptr);
        let fma = f.expr(Expression::CooperativeMultiplyAdd {
            a: a_coop,
            b: b_coop,
            c: old_acc,
        });

        push_emit(&f.f.expressions, &mut loop_body, sa_zero, fma);
        loop_body.push(f.store(acc_ptr, fma), S);

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

    // Store result: coopStore(acc, &matrix_c[c_offset], n)
    let c_ptr2 = f.global(gv_c);
    let c_elem2 = f.index(c_ptr2, c_offset);
    let final_acc = f.load(acc_ptr);
    f.emit(c_ptr2, final_acc);
    f.f.body.push(
        Statement::CooperativeStore {
            target: final_acc,
            data: CooperativeData {
                pointer: c_elem2,
                stride: pn,
                row_major: false,
            },
        },
        S,
    );

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

fn gen_rms_norm() -> Module {
    let mut b = Builder::new();
    // params: rows, cols, eps_bits, _pad
    let ty_params = b.params_u32x4("Params", &["rows", "cols", "eps_bits", "_pad"]);
    let gv_src = b.storage_ro("src");
    let gv_weight = b.storage_ro("bias"); // named "bias" to match blade binding convention
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

    // Compute sum of squares: var ss = 0.0; for j in 0..cols { ss += x[offset+j]² }
    let ss_var = f.local_var("ss", b.ty_f32, None);
    let ss_ptr = f.local_ptr(ss_var);
    let zero_f = f.literal_f32(0.0);
    f.emit(zero_f, zero_f);
    f.f.body.push(f.store(ss_ptr, zero_f), S);

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
        let sq = f.binary(BinaryOperator::Multiply, val, val);
        let old_ss = f.load(ss_ptr);
        let new_ss = f.binary(BinaryOperator::Add, old_ss, sq);
        push_emit(&f.f.expressions, &mut body, j, brk);
        body.push(FnBuilder::if_break(brk), S);
        push_emit(&f.f.expressions, &mut body, idx, new_ss);
        body.push(f.store(ss_ptr, new_ss), S);

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

    // rms = 1.0 / sqrt(ss / cols_f + eps)
    let ss_val = f.load(ss_ptr);
    let cols_f = f.cast_f32(cols);
    let mean_sq = f.binary(BinaryOperator::Divide, ss_val, cols_f);
    let mean_sq_eps = f.binary(BinaryOperator::Add, mean_sq, eps);
    let rsqrt = f.math1(MathFunction::InverseSqrt, mean_sq_eps);
    f.emit(ss_val, rsqrt);

    // Normalize loop: dst[offset+j] = x[offset+j] * rsqrt * weight[j]
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
        let normed = f.binary(BinaryOperator::Multiply, val, rsqrt);
        let w_ptr = f.global(gv_weight);
        let w_elem = f.index(w_ptr, j);
        let w_val = f.load(w_elem);
        let result = f.binary(BinaryOperator::Multiply, normed, w_val);
        push_emit(&f.f.expressions, &mut body, j, brk);
        body.push(FnBuilder::if_break(brk), S);
        push_emit(&f.f.expressions, &mut body, idx, result);

        let dst_ptr = f.global(gv_dst);
        let dst_elem = f.index(dst_ptr, idx);
        push_emit(&f.f.expressions, &mut body, dst_elem, dst_elem);
        body.push(f.store(dst_elem, result), S);

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

    b.entry_point("main", [256, 1, 1], f.finish());
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

fn gen_causal_attention() -> Module {
    let mut b = Builder::new();
    let ty_params = b.params_u32x4("Params", &["seq", "num_heads", "num_kv_heads", "head_dim"]);
    let gv_q = b.storage_ro("src_a"); // q
    let gv_k = b.storage_ro("src_b"); // k
    let gv_v = b.storage_ro("bias"); // v (reusing binding slot name)
    let gv_dst = b.storage_rw("dst");
    let gv_params = b.uniform("params", ty_params);

    let mut f = FnBuilder::new(&b);
    let gid = f.arg_gid();

    // pos = gid.x, head = gid.y
    let pos = f.vec_x(gid);
    f.label("pos", pos);
    let head = f.vec_y(gid);
    f.label("head", head);
    f.emit(pos, head);

    let params_ptr = f.global(gv_params);
    let seq_ptr = f.field(params_ptr, 0);
    let seq = f.load(seq_ptr);
    let nh_ptr = f.field(params_ptr, 1);
    let num_heads = f.load(nh_ptr);
    let nkv_ptr = f.field(params_ptr, 2);
    let num_kv_heads = f.load(nkv_ptr);
    let hd_ptr = f.field(params_ptr, 3);
    let head_dim = f.load(hd_ptr);
    f.emit(params_ptr, head_dim);

    // Bounds check
    let cond_pos = f.binary(BinaryOperator::GreaterEqual, pos, seq);
    let cond_head = f.binary(BinaryOperator::GreaterEqual, head, num_heads);
    let cond = f.binary(BinaryOperator::LogicalOr, cond_pos, cond_head);
    f.emit(cond_pos, cond);
    f.f.body.push(f.if_return(cond), S);

    // GQA: kv_head = head / (num_heads / num_kv_heads)
    let heads_per_kv = f.binary(BinaryOperator::Divide, num_heads, num_kv_heads);
    let kv_head = f.binary(BinaryOperator::Divide, head, heads_per_kv);

    // q_dim = num_heads * head_dim
    let q_dim = f.binary(BinaryOperator::Multiply, num_heads, head_dim);
    // kv_dim = num_kv_heads * head_dim
    let kv_dim = f.binary(BinaryOperator::Multiply, num_kv_heads, head_dim);

    // q_base = pos * q_dim + head * head_dim
    let pos_q = f.binary(BinaryOperator::Multiply, pos, q_dim);
    let head_off = f.binary(BinaryOperator::Multiply, head, head_dim);
    let q_base = f.binary(BinaryOperator::Add, pos_q, head_off);

    // scale = 1.0 / sqrt(head_dim)
    let hd_f = f.cast_f32(head_dim);
    let scale = f.math1(MathFunction::InverseSqrt, hd_f);

    // causal_len = pos + 1
    let one_cl = f.literal_u32(1);
    let causal_len = f.binary(BinaryOperator::Add, pos, one_cl);

    f.emit(heads_per_kv, causal_len);

    // Store key values in local vars so inner scopes can load them fresh
    let q_base_var = f.local_var("q_base", b.ty_u32, None);
    let q_base_ptr = f.local_ptr(q_base_var);
    f.f.body.push(f.store(q_base_ptr, q_base), S);

    let kv_head_var = f.local_var("kv_head", b.ty_u32, None);
    let kv_head_ptr = f.local_ptr(kv_head_var);
    f.f.body.push(f.store(kv_head_ptr, kv_head), S);

    let kv_dim_var = f.local_var("kv_dim", b.ty_u32, None);
    let kv_dim_ptr = f.local_ptr(kv_dim_var);
    f.f.body.push(f.store(kv_dim_ptr, kv_dim), S);

    let head_dim_var = f.local_var("hd", b.ty_u32, None);
    let head_dim_ptr = f.local_ptr(head_dim_var);
    f.f.body.push(f.store(head_dim_ptr, head_dim), S);

    let scale_var = f.local_var("scale", b.ty_f32, None);
    let scale_ptr = f.local_ptr(scale_var);
    f.f.body.push(f.store(scale_ptr, scale), S);

    let causal_var = f.local_var("clen", b.ty_u32, None);
    let causal_ptr = f.local_ptr(causal_var);
    f.f.body.push(f.store(causal_ptr, causal_len), S);

    let pos_q_var = f.local_var("pos_q", b.ty_u32, None);
    let pos_q_ptr = f.local_ptr(pos_q_var);
    f.f.body.push(f.store(pos_q_ptr, pos_q), S);

    let head_off_var = f.local_var("head_off", b.ty_u32, None);
    let head_off_ptr = f.local_ptr(head_off_var);
    f.f.body.push(f.store(head_off_ptr, head_off), S);

    #[allow(clippy::too_many_arguments)]
    fn build_dot_loop(
        f: &mut FnBuilder,
        parent: &mut Block,
        gv_q: Handle<GlobalVariable>,
        gv_k: Handle<GlobalVariable>,
        q_base_ptr: Handle<Expression>,
        head_dim_ptr: Handle<Expression>,
        k_base: Handle<Expression>, // already in scope of parent
        dot_ptr: Handle<Expression>,
        ty_u32: Handle<Type>,
        d_name: &str,
    ) {
        let d_var = f.local_var(d_name, ty_u32, None);
        let d_ptr = f.local_ptr(d_var);
        let zero_d = f.literal_u32(0);
        push_emit(&f.f.expressions, parent, zero_d, zero_d);
        parent.push(f.store(d_ptr, zero_d), S);

        let mut inner = Block::new();
        let d = f.load(d_ptr);
        let hd = f.load(head_dim_ptr);
        let dbrk = f.binary(BinaryOperator::GreaterEqual, d, hd);

        let qb = f.load(q_base_ptr);
        let q_idx = f.binary(BinaryOperator::Add, qb, d);
        let q_gp = f.global(gv_q);
        let q_elem = f.index(q_gp, q_idx);
        let q_val = f.load(q_elem);

        let k_idx = f.binary(BinaryOperator::Add, k_base, d);
        let k_gp = f.global(gv_k);
        let k_elem = f.index(k_gp, k_idx);
        let k_val = f.load(k_elem);

        let prod = f.binary(BinaryOperator::Multiply, q_val, k_val);
        let old_dot = f.load(dot_ptr);
        let new_dot = f.binary(BinaryOperator::Add, old_dot, prod);
        push_emit(&f.f.expressions, &mut inner, d, dbrk);
        inner.push(FnBuilder::if_break(dbrk), S);
        push_emit(&f.f.expressions, &mut inner, qb, new_dot);
        inner.push(f.store(dot_ptr, new_dot), S);

        let one_d = f.literal_u32(1);
        let d2 = f.load(d_ptr);
        let dn = f.binary(BinaryOperator::Add, d2, one_d);
        push_emit(&f.f.expressions, &mut inner, one_d, dn);
        inner.push(f.store(d_ptr, dn), S);

        parent.push(
            Statement::Loop {
                body: inner,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    // Helper: compute k_base = t * kv_dim + kv_head * head_dim, emit into block
    fn compute_k_base(
        f: &mut FnBuilder,
        block: &mut Block,
        t: Handle<Expression>,
        kv_dim_ptr: Handle<Expression>,
        kv_head_ptr: Handle<Expression>,
        head_dim_ptr: Handle<Expression>,
    ) -> Handle<Expression> {
        let kvd = f.load(kv_dim_ptr);
        let t_kv = f.binary(BinaryOperator::Multiply, t, kvd);
        let kvh = f.load(kv_head_ptr);
        let hd = f.load(head_dim_ptr);
        let kvh_off = f.binary(BinaryOperator::Multiply, kvh, hd);
        let k_base = f.binary(BinaryOperator::Add, t_kv, kvh_off);
        push_emit(&f.f.expressions, block, kvd, k_base);
        k_base
    }

    // === Pass 1: find max score ===
    let max_var = f.local_var("max_score", b.ty_f32, None);
    let max_ptr = f.local_ptr(max_var);
    let neg_inf = f.literal_f32(-1.0e30);
    f.emit(neg_inf, neg_inf);
    f.f.body.push(f.store(max_ptr, neg_inf), S);

    let t_var = f.local_var("t", b.ty_u32, None);
    let t_ptr = f.local_ptr(t_var);
    let zero_u = f.literal_u32(0);
    f.emit(zero_u, zero_u);
    f.f.body.push(f.store(t_ptr, zero_u), S);

    {
        let mut body = Block::new();
        let t = f.load(t_ptr);
        let cl = f.load(causal_ptr);
        let brk = f.binary(BinaryOperator::GreaterEqual, t, cl);

        let dot_var = f.local_var("dot", b.ty_f32, None);
        let dot_ptr = f.local_ptr(dot_var);
        let zero_f = f.literal_f32(0.0);
        push_emit(&f.f.expressions, &mut body, t, brk);
        body.push(FnBuilder::if_break(brk), S);
        push_emit(&f.f.expressions, &mut body, dot_ptr, zero_f);
        body.push(f.store(dot_ptr, zero_f), S);

        let k_base = compute_k_base(&mut f, &mut body, t, kv_dim_ptr, kv_head_ptr, head_dim_ptr);

        build_dot_loop(
            &mut f,
            &mut body,
            gv_q,
            gv_k,
            q_base_ptr,
            head_dim_ptr,
            k_base,
            dot_ptr,
            b.ty_u32,
            "d1",
        );

        // score = dot * scale; max_score = max(max_score, score)
        let dot_val = f.load(dot_ptr);
        let sc = f.load(scale_ptr);
        let score = f.binary(BinaryOperator::Multiply, dot_val, sc);
        let old_max = f.load(max_ptr);
        let new_max = f.math2(MathFunction::Max, old_max, score);
        push_emit(&f.f.expressions, &mut body, dot_val, new_max);
        body.push(f.store(max_ptr, new_max), S);

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

    // === Pass 2: compute exp sum ===
    let sum_var = f.local_var("sum_exp", b.ty_f32, None);
    let sum_ptr_local = f.local_ptr(sum_var);
    let zero_f2 = f.literal_f32(0.0);
    f.emit(zero_f2, zero_f2);
    f.f.body.push(f.store(sum_ptr_local, zero_f2), S);

    let t2_var = f.local_var("t2", b.ty_u32, None);
    let t2_ptr = f.local_ptr(t2_var);
    let zero_t2 = f.literal_u32(0);
    f.emit(zero_t2, zero_t2);
    f.f.body.push(f.store(t2_ptr, zero_t2), S);

    {
        let mut body = Block::new();
        let t = f.load(t2_ptr);
        let cl = f.load(causal_ptr);
        let brk = f.binary(BinaryOperator::GreaterEqual, t, cl);

        let dot_var2 = f.local_var("dot2", b.ty_f32, None);
        let dot_ptr2 = f.local_ptr(dot_var2);
        let zero_f3 = f.literal_f32(0.0);
        push_emit(&f.f.expressions, &mut body, t, brk);
        body.push(FnBuilder::if_break(brk), S);
        push_emit(&f.f.expressions, &mut body, dot_ptr2, zero_f3);
        body.push(f.store(dot_ptr2, zero_f3), S);

        let k_base = compute_k_base(&mut f, &mut body, t, kv_dim_ptr, kv_head_ptr, head_dim_ptr);

        build_dot_loop(
            &mut f,
            &mut body,
            gv_q,
            gv_k,
            q_base_ptr,
            head_dim_ptr,
            k_base,
            dot_ptr2,
            b.ty_u32,
            "d2",
        );

        let dot_val = f.load(dot_ptr2);
        let sc = f.load(scale_ptr);
        let score = f.binary(BinaryOperator::Multiply, dot_val, sc);
        let mx = f.load(max_ptr);
        let shifted = f.binary(BinaryOperator::Subtract, score, mx);
        let e = f.math1(MathFunction::Exp, shifted);
        let old_sum = f.load(sum_ptr_local);
        let new_sum = f.binary(BinaryOperator::Add, old_sum, e);
        push_emit(&f.f.expressions, &mut body, dot_val, new_sum);
        body.push(f.store(sum_ptr_local, new_sum), S);

        let one_t = f.literal_u32(1);
        let t3 = f.load(t2_ptr);
        let tn = f.binary(BinaryOperator::Add, t3, one_t);
        push_emit(&f.f.expressions, &mut body, one_t, tn);
        body.push(f.store(t2_ptr, tn), S);

        f.f.body.push(
            Statement::Loop {
                body,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    // === Pass 3: weighted sum of values ===
    let d3_var = f.local_var("d3", b.ty_u32, None);
    let d3_ptr = f.local_ptr(d3_var);
    let zero_d3 = f.literal_u32(0);
    f.emit(zero_d3, zero_d3);
    f.f.body.push(f.store(d3_ptr, zero_d3), S);

    {
        let mut d_body = Block::new();
        let d = f.load(d3_ptr);
        let hd = f.load(head_dim_ptr);
        let d_brk = f.binary(BinaryOperator::GreaterEqual, d, hd);

        let acc_var = f.local_var("acc", b.ty_f32, None);
        let acc_ptr = f.local_ptr(acc_var);
        let zero_acc = f.literal_f32(0.0);
        push_emit(&f.f.expressions, &mut d_body, d, d_brk);
        d_body.push(FnBuilder::if_break(d_brk), S);
        push_emit(&f.f.expressions, &mut d_body, acc_ptr, zero_acc);
        d_body.push(f.store(acc_ptr, zero_acc), S);

        // Pre-compute output base index (pos_q + head_off) so we can use it after the inner loop
        // without a wide emit range spanning inner-loop expressions
        let pq = f.load(pos_q_ptr);
        let ho = f.load(head_off_ptr);
        let out_base = f.binary(BinaryOperator::Add, pq, ho);
        // Store in local var to avoid cross-scope emit issues
        let out_base_var = f.local_var("out_base", b.ty_u32, None);
        let out_base_ptr = f.local_ptr(out_base_var);
        push_emit(&f.f.expressions, &mut d_body, pq, out_base);
        d_body.push(f.store(out_base_ptr, out_base), S);

        let t3_var = f.local_var("t3", b.ty_u32, None);
        let t3_iptr = f.local_ptr(t3_var);
        let zero_t3 = f.literal_u32(0);
        push_emit(&f.f.expressions, &mut d_body, zero_t3, zero_t3);
        d_body.push(f.store(t3_iptr, zero_t3), S);

        {
            let mut t_body = Block::new();
            let t = f.load(t3_iptr);
            let cl = f.load(causal_ptr);
            let t_brk = f.binary(BinaryOperator::GreaterEqual, t, cl);

            // Recompute dot product for attention weight
            let dot_var3 = f.local_var("dot3", b.ty_f32, None);
            let dot_ptr3 = f.local_ptr(dot_var3);
            let zero_dot = f.literal_f32(0.0);
            push_emit(&f.f.expressions, &mut t_body, t, t_brk);
            t_body.push(FnBuilder::if_break(t_brk), S);
            push_emit(&f.f.expressions, &mut t_body, dot_ptr3, zero_dot);
            t_body.push(f.store(dot_ptr3, zero_dot), S);

            let k_base = compute_k_base(
                &mut f,
                &mut t_body,
                t,
                kv_dim_ptr,
                kv_head_ptr,
                head_dim_ptr,
            );

            build_dot_loop(
                &mut f,
                &mut t_body,
                gv_q,
                gv_k,
                q_base_ptr,
                head_dim_ptr,
                k_base,
                dot_ptr3,
                b.ty_u32,
                "d4",
            );

            // attn_weight = exp(score * scale - max) / sum_exp
            let dot_val = f.load(dot_ptr3);
            let sc = f.load(scale_ptr);
            let score = f.binary(BinaryOperator::Multiply, dot_val, sc);
            let mx = f.load(max_ptr);
            let shifted = f.binary(BinaryOperator::Subtract, score, mx);
            let e = f.math1(MathFunction::Exp, shifted);
            let sum_val = f.load(sum_ptr_local);
            let weight = f.binary(BinaryOperator::Divide, e, sum_val);

            // v_val = v[k_base + d]
            let v_idx = f.binary(BinaryOperator::Add, k_base, d);
            let v_gp = f.global(gv_v);
            let v_elem = f.index(v_gp, v_idx);
            let v_val = f.load(v_elem);

            let contrib = f.binary(BinaryOperator::Multiply, weight, v_val);
            let old_acc = f.load(acc_ptr);
            let new_acc = f.binary(BinaryOperator::Add, old_acc, contrib);
            push_emit(&f.f.expressions, &mut t_body, dot_val, new_acc);
            t_body.push(f.store(acc_ptr, new_acc), S);

            let one_t = f.literal_u32(1);
            let t4 = f.load(t3_iptr);
            let tn = f.binary(BinaryOperator::Add, t4, one_t);
            push_emit(&f.f.expressions, &mut t_body, one_t, tn);
            t_body.push(f.store(t3_iptr, tn), S);

            d_body.push(
                Statement::Loop {
                    body: t_body,
                    continuing: Block::new(),
                    break_if: None,
                },
                S,
            );
        }

        // dst[out_base + d] = acc
        let ob = f.load(out_base_ptr);
        let out_idx = f.binary(BinaryOperator::Add, ob, d);
        let dst_gp = f.global(gv_dst);
        let dst_elem = f.index(dst_gp, out_idx);
        let acc_val = f.load(acc_ptr);
        // Single emit range covers ob through acc_val (dst_gp is pre-emit/skipped)
        push_emit(&f.f.expressions, &mut d_body, ob, acc_val);
        d_body.push(f.store(dst_elem, acc_val), S);

        let one_d = f.literal_u32(1);
        let d5 = f.load(d3_ptr);
        let dn = f.binary(BinaryOperator::Add, d5, one_d);
        push_emit(&f.f.expressions, &mut d_body, one_d, dn);
        d_body.push(f.store(d3_ptr, dn), S);

        f.f.body.push(
            Statement::Loop {
                body: d_body,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    b.entry_point("main", [1, 1, 1], f.finish());
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
    gen_attention_common(false)
}

// ---------------------------------------------------------------------------
// cross_attention.wgsl: cross-attention where q and k/v have different seq lengths
// params: q_seq, kv_seq, (num_heads<<16)|num_kv_heads, head_dim
// ---------------------------------------------------------------------------

fn gen_cross_attention() -> Module {
    gen_attention_common(true)
}

/// Shared attention generator for both full self-attention and cross-attention.
/// When `is_cross` is true, q_seq and kv_seq may differ (params layout differs).
/// When `is_cross` is false, it's non-causal self-attention (q_seq == kv_seq).
fn gen_attention_common(is_cross: bool) -> Module {
    let mut b = Builder::new();
    let ty_params = if is_cross {
        // params: q_seq, kv_seq, (num_heads<<16)|num_kv_heads, head_dim
        b.params_u32x4("Params", &["q_seq", "kv_seq", "packed_heads", "head_dim"])
    } else {
        // params: seq, num_heads, num_kv_heads, head_dim
        b.params_u32x4("Params", &["seq", "num_heads", "num_kv_heads", "head_dim"])
    };
    let gv_q = b.storage_ro("src_a"); // q
    let gv_k = b.storage_ro("src_b"); // k
    let gv_v = b.storage_ro("bias"); // v (reusing binding slot name)
    let gv_dst = b.storage_rw("dst");
    let gv_params = b.uniform("params", ty_params);

    let mut f = FnBuilder::new(&b);
    let gid = f.arg_gid();

    // pos = gid.x, head = gid.y
    let pos = f.vec_x(gid);
    f.label("pos", pos);
    let head = f.vec_y(gid);
    f.label("head", head);
    f.emit(pos, head);

    let params_ptr = f.global(gv_params);

    // Parse params based on attention type
    let (q_seq, kv_seq, num_heads, num_kv_heads, head_dim);
    if is_cross {
        let p0 = f.field(params_ptr, 0);
        q_seq = f.load(p0);
        let p1 = f.field(params_ptr, 1);
        kv_seq = f.load(p1);
        let p2 = f.field(params_ptr, 2);
        let packed = f.load(p2);
        let p3 = f.field(params_ptr, 3);
        head_dim = f.load(p3);
        // Unpack: num_heads = packed >> 16, num_kv_heads = packed & 0xFFFF
        let shift = f.literal_u32(16);
        num_heads = f.binary(BinaryOperator::ShiftRight, packed, shift);
        let mask = f.literal_u32(0xFFFF);
        num_kv_heads = f.binary(BinaryOperator::And, packed, mask);
        f.emit(params_ptr, num_kv_heads);
    } else {
        let p0 = f.field(params_ptr, 0);
        q_seq = f.load(p0);
        kv_seq = q_seq; // same for self-attention
        let p1 = f.field(params_ptr, 1);
        num_heads = f.load(p1);
        let p2 = f.field(params_ptr, 2);
        num_kv_heads = f.load(p2);
        let p3 = f.field(params_ptr, 3);
        head_dim = f.load(p3);
        f.emit(params_ptr, head_dim);
    }

    // Bounds check
    let cond_pos = f.binary(BinaryOperator::GreaterEqual, pos, q_seq);
    let cond_head = f.binary(BinaryOperator::GreaterEqual, head, num_heads);
    let cond = f.binary(BinaryOperator::LogicalOr, cond_pos, cond_head);
    f.emit(cond_pos, cond);
    f.f.body.push(f.if_return(cond), S);

    // GQA: kv_head = head / (num_heads / num_kv_heads)
    let heads_per_kv = f.binary(BinaryOperator::Divide, num_heads, num_kv_heads);
    let kv_head = f.binary(BinaryOperator::Divide, head, heads_per_kv);

    // q_dim = num_heads * head_dim
    let q_dim = f.binary(BinaryOperator::Multiply, num_heads, head_dim);
    // kv_dim = num_kv_heads * head_dim
    let kv_dim = f.binary(BinaryOperator::Multiply, num_kv_heads, head_dim);

    // q_base = pos * q_dim + head * head_dim
    let pos_q = f.binary(BinaryOperator::Multiply, pos, q_dim);
    let head_off = f.binary(BinaryOperator::Multiply, head, head_dim);
    let q_base = f.binary(BinaryOperator::Add, pos_q, head_off);

    // scale = 1.0 / sqrt(head_dim)
    let hd_f = f.cast_f32(head_dim);
    let scale = f.math1(MathFunction::InverseSqrt, hd_f);

    f.emit(heads_per_kv, scale);

    // Store needed values in local vars
    let q_base_var = f.local_var("q_base", b.ty_u32, None);
    let q_base_ptr = f.local_ptr(q_base_var);
    f.f.body.push(f.store(q_base_ptr, q_base), S);

    let kv_head_var = f.local_var("kv_head", b.ty_u32, None);
    let kv_head_ptr = f.local_ptr(kv_head_var);
    f.f.body.push(f.store(kv_head_ptr, kv_head), S);

    let kv_dim_var = f.local_var("kv_dim", b.ty_u32, None);
    let kv_dim_ptr = f.local_ptr(kv_dim_var);
    f.f.body.push(f.store(kv_dim_ptr, kv_dim), S);

    let head_dim_var = f.local_var("hd", b.ty_u32, None);
    let head_dim_ptr = f.local_ptr(head_dim_var);
    f.f.body.push(f.store(head_dim_ptr, head_dim), S);

    let scale_var = f.local_var("scale", b.ty_f32, None);
    let scale_ptr = f.local_ptr(scale_var);
    f.f.body.push(f.store(scale_ptr, scale), S);

    let kv_seq_var = f.local_var("kv_len", b.ty_u32, None);
    let kv_seq_ptr = f.local_ptr(kv_seq_var);
    f.f.body.push(f.store(kv_seq_ptr, kv_seq), S);

    let pos_q_var = f.local_var("pos_q", b.ty_u32, None);
    let pos_q_ptr = f.local_ptr(pos_q_var);
    f.f.body.push(f.store(pos_q_ptr, pos_q), S);

    let head_off_var = f.local_var("head_off", b.ty_u32, None);
    let head_off_ptr = f.local_ptr(head_off_var);
    f.f.body.push(f.store(head_off_ptr, head_off), S);

    #[allow(clippy::too_many_arguments)]
    fn build_dot_loop_attn(
        f: &mut FnBuilder,
        parent: &mut Block,
        gv_q: Handle<GlobalVariable>,
        gv_k: Handle<GlobalVariable>,
        q_base_ptr: Handle<Expression>,
        head_dim_ptr: Handle<Expression>,
        k_base: Handle<Expression>,
        dot_ptr: Handle<Expression>,
        ty_u32: Handle<Type>,
        d_name: &str,
    ) {
        let d_var = f.local_var(d_name, ty_u32, None);
        let d_ptr = f.local_ptr(d_var);
        let zero_d = f.literal_u32(0);
        push_emit(&f.f.expressions, parent, zero_d, zero_d);
        parent.push(f.store(d_ptr, zero_d), S);

        let mut inner = Block::new();
        let d = f.load(d_ptr);
        let hd = f.load(head_dim_ptr);
        let dbrk = f.binary(BinaryOperator::GreaterEqual, d, hd);

        let qb = f.load(q_base_ptr);
        let q_idx = f.binary(BinaryOperator::Add, qb, d);
        let q_gp = f.global(gv_q);
        let q_elem = f.index(q_gp, q_idx);
        let q_val = f.load(q_elem);

        let k_idx = f.binary(BinaryOperator::Add, k_base, d);
        let k_gp = f.global(gv_k);
        let k_elem = f.index(k_gp, k_idx);
        let k_val = f.load(k_elem);

        let prod = f.binary(BinaryOperator::Multiply, q_val, k_val);
        let old_dot = f.load(dot_ptr);
        let new_dot = f.binary(BinaryOperator::Add, old_dot, prod);
        push_emit(&f.f.expressions, &mut inner, d, dbrk);
        inner.push(FnBuilder::if_break(dbrk), S);
        push_emit(&f.f.expressions, &mut inner, qb, new_dot);
        inner.push(f.store(dot_ptr, new_dot), S);

        let one_d = f.literal_u32(1);
        let d2 = f.load(d_ptr);
        let dn = f.binary(BinaryOperator::Add, d2, one_d);
        push_emit(&f.f.expressions, &mut inner, one_d, dn);
        inner.push(f.store(d_ptr, dn), S);

        parent.push(
            Statement::Loop {
                body: inner,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    fn compute_k_base_attn(
        f: &mut FnBuilder,
        block: &mut Block,
        t: Handle<Expression>,
        kv_dim_ptr: Handle<Expression>,
        kv_head_ptr: Handle<Expression>,
        head_dim_ptr: Handle<Expression>,
    ) -> Handle<Expression> {
        let kvd = f.load(kv_dim_ptr);
        let t_kv = f.binary(BinaryOperator::Multiply, t, kvd);
        let kvh = f.load(kv_head_ptr);
        let hd = f.load(head_dim_ptr);
        let kvh_off = f.binary(BinaryOperator::Multiply, kvh, hd);
        let k_base = f.binary(BinaryOperator::Add, t_kv, kvh_off);
        push_emit(&f.f.expressions, block, kvd, k_base);
        k_base
    }

    // === Pass 1: find max score ===
    let max_var = f.local_var("max_score", b.ty_f32, None);
    let max_ptr = f.local_ptr(max_var);
    let neg_inf = f.literal_f32(-1.0e30);
    f.emit(neg_inf, neg_inf);
    f.f.body.push(f.store(max_ptr, neg_inf), S);

    let t_var = f.local_var("t", b.ty_u32, None);
    let t_ptr = f.local_ptr(t_var);
    let zero_u = f.literal_u32(0);
    f.emit(zero_u, zero_u);
    f.f.body.push(f.store(t_ptr, zero_u), S);

    {
        let mut body = Block::new();
        let t = f.load(t_ptr);
        let kv_len = f.load(kv_seq_ptr);
        let brk = f.binary(BinaryOperator::GreaterEqual, t, kv_len);

        let dot_var = f.local_var("dot", b.ty_f32, None);
        let dot_ptr = f.local_ptr(dot_var);
        let zero_f = f.literal_f32(0.0);
        push_emit(&f.f.expressions, &mut body, t, brk);
        body.push(FnBuilder::if_break(brk), S);
        push_emit(&f.f.expressions, &mut body, dot_ptr, zero_f);
        body.push(f.store(dot_ptr, zero_f), S);

        let k_base =
            compute_k_base_attn(&mut f, &mut body, t, kv_dim_ptr, kv_head_ptr, head_dim_ptr);

        build_dot_loop_attn(
            &mut f,
            &mut body,
            gv_q,
            gv_k,
            q_base_ptr,
            head_dim_ptr,
            k_base,
            dot_ptr,
            b.ty_u32,
            "d1",
        );

        let dot_val = f.load(dot_ptr);
        let sc = f.load(scale_ptr);
        let score = f.binary(BinaryOperator::Multiply, dot_val, sc);
        let old_max = f.load(max_ptr);
        let new_max = f.math2(MathFunction::Max, old_max, score);
        push_emit(&f.f.expressions, &mut body, dot_val, new_max);
        body.push(f.store(max_ptr, new_max), S);

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

    // === Pass 2: compute exp sum ===
    let sum_var = f.local_var("sum_exp", b.ty_f32, None);
    let sum_ptr_local = f.local_ptr(sum_var);
    let zero_f2 = f.literal_f32(0.0);
    f.emit(zero_f2, zero_f2);
    f.f.body.push(f.store(sum_ptr_local, zero_f2), S);

    let t2_var = f.local_var("t2", b.ty_u32, None);
    let t2_ptr = f.local_ptr(t2_var);
    let zero_t2 = f.literal_u32(0);
    f.emit(zero_t2, zero_t2);
    f.f.body.push(f.store(t2_ptr, zero_t2), S);

    {
        let mut body = Block::new();
        let t = f.load(t2_ptr);
        let kv_len = f.load(kv_seq_ptr);
        let brk = f.binary(BinaryOperator::GreaterEqual, t, kv_len);

        let dot_var2 = f.local_var("dot2", b.ty_f32, None);
        let dot_ptr2 = f.local_ptr(dot_var2);
        let zero_f3 = f.literal_f32(0.0);
        push_emit(&f.f.expressions, &mut body, t, brk);
        body.push(FnBuilder::if_break(brk), S);
        push_emit(&f.f.expressions, &mut body, dot_ptr2, zero_f3);
        body.push(f.store(dot_ptr2, zero_f3), S);

        let k_base =
            compute_k_base_attn(&mut f, &mut body, t, kv_dim_ptr, kv_head_ptr, head_dim_ptr);

        build_dot_loop_attn(
            &mut f,
            &mut body,
            gv_q,
            gv_k,
            q_base_ptr,
            head_dim_ptr,
            k_base,
            dot_ptr2,
            b.ty_u32,
            "d2",
        );

        let dot_val = f.load(dot_ptr2);
        let sc = f.load(scale_ptr);
        let score = f.binary(BinaryOperator::Multiply, dot_val, sc);
        let mx = f.load(max_ptr);
        let shifted = f.binary(BinaryOperator::Subtract, score, mx);
        let e = f.math1(MathFunction::Exp, shifted);
        let old_sum = f.load(sum_ptr_local);
        let new_sum = f.binary(BinaryOperator::Add, old_sum, e);
        push_emit(&f.f.expressions, &mut body, dot_val, new_sum);
        body.push(f.store(sum_ptr_local, new_sum), S);

        let one_t = f.literal_u32(1);
        let t3 = f.load(t2_ptr);
        let tn = f.binary(BinaryOperator::Add, t3, one_t);
        push_emit(&f.f.expressions, &mut body, one_t, tn);
        body.push(f.store(t2_ptr, tn), S);

        f.f.body.push(
            Statement::Loop {
                body,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    // === Pass 3: weighted sum of values ===
    let d3_var = f.local_var("d3", b.ty_u32, None);
    let d3_ptr = f.local_ptr(d3_var);
    let zero_d3 = f.literal_u32(0);
    f.emit(zero_d3, zero_d3);
    f.f.body.push(f.store(d3_ptr, zero_d3), S);

    {
        let mut d_body = Block::new();
        let d = f.load(d3_ptr);
        let hd = f.load(head_dim_ptr);
        let d_brk = f.binary(BinaryOperator::GreaterEqual, d, hd);

        let acc_var = f.local_var("acc", b.ty_f32, None);
        let acc_ptr = f.local_ptr(acc_var);
        let zero_acc = f.literal_f32(0.0);
        push_emit(&f.f.expressions, &mut d_body, d, d_brk);
        d_body.push(FnBuilder::if_break(d_brk), S);
        push_emit(&f.f.expressions, &mut d_body, acc_ptr, zero_acc);
        d_body.push(f.store(acc_ptr, zero_acc), S);

        let pq = f.load(pos_q_ptr);
        let ho = f.load(head_off_ptr);
        let out_base = f.binary(BinaryOperator::Add, pq, ho);
        let out_base_var = f.local_var("out_base", b.ty_u32, None);
        let out_base_ptr = f.local_ptr(out_base_var);
        push_emit(&f.f.expressions, &mut d_body, pq, out_base);
        d_body.push(f.store(out_base_ptr, out_base), S);

        let t3_var = f.local_var("t3", b.ty_u32, None);
        let t3_iptr = f.local_ptr(t3_var);
        let zero_t3 = f.literal_u32(0);
        push_emit(&f.f.expressions, &mut d_body, zero_t3, zero_t3);
        d_body.push(f.store(t3_iptr, zero_t3), S);

        {
            let mut t_body = Block::new();
            let t = f.load(t3_iptr);
            let kv_len = f.load(kv_seq_ptr);
            let t_brk = f.binary(BinaryOperator::GreaterEqual, t, kv_len);

            let dot_var3 = f.local_var("dot3", b.ty_f32, None);
            let dot_ptr3 = f.local_ptr(dot_var3);
            let zero_dot = f.literal_f32(0.0);
            push_emit(&f.f.expressions, &mut t_body, t, t_brk);
            t_body.push(FnBuilder::if_break(t_brk), S);
            push_emit(&f.f.expressions, &mut t_body, dot_ptr3, zero_dot);
            t_body.push(f.store(dot_ptr3, zero_dot), S);

            let k_base = compute_k_base_attn(
                &mut f,
                &mut t_body,
                t,
                kv_dim_ptr,
                kv_head_ptr,
                head_dim_ptr,
            );

            build_dot_loop_attn(
                &mut f,
                &mut t_body,
                gv_q,
                gv_k,
                q_base_ptr,
                head_dim_ptr,
                k_base,
                dot_ptr3,
                b.ty_u32,
                "d4",
            );

            let dot_val = f.load(dot_ptr3);
            let sc = f.load(scale_ptr);
            let score = f.binary(BinaryOperator::Multiply, dot_val, sc);
            let mx = f.load(max_ptr);
            let shifted = f.binary(BinaryOperator::Subtract, score, mx);
            let e = f.math1(MathFunction::Exp, shifted);
            let sum_val = f.load(sum_ptr_local);
            let weight = f.binary(BinaryOperator::Divide, e, sum_val);

            let v_idx = f.binary(BinaryOperator::Add, k_base, d);
            let v_gp = f.global(gv_v);
            let v_elem = f.index(v_gp, v_idx);
            let v_val = f.load(v_elem);

            let contrib = f.binary(BinaryOperator::Multiply, weight, v_val);
            let old_acc = f.load(acc_ptr);
            let new_acc = f.binary(BinaryOperator::Add, old_acc, contrib);
            push_emit(&f.f.expressions, &mut t_body, dot_val, new_acc);
            t_body.push(f.store(acc_ptr, new_acc), S);

            let one_t = f.literal_u32(1);
            let t4 = f.load(t3_iptr);
            let tn = f.binary(BinaryOperator::Add, t4, one_t);
            push_emit(&f.f.expressions, &mut t_body, one_t, tn);
            t_body.push(f.store(t3_iptr, tn), S);

            d_body.push(
                Statement::Loop {
                    body: t_body,
                    continuing: Block::new(),
                    break_if: None,
                },
                S,
            );
        }

        let ob = f.load(out_base_ptr);
        let out_idx = f.binary(BinaryOperator::Add, ob, d);
        let dst_gp = f.global(gv_dst);
        let dst_elem = f.index(dst_gp, out_idx);
        let acc_val = f.load(acc_ptr);
        push_emit(&f.f.expressions, &mut d_body, ob, acc_val);
        d_body.push(f.store(dst_elem, acc_val), S);

        let one_d = f.literal_u32(1);
        let d5 = f.load(d3_ptr);
        let dn = f.binary(BinaryOperator::Add, d5, one_d);
        push_emit(&f.f.expressions, &mut d_body, one_d, dn);
        d_body.push(f.store(d3_ptr, dn), S);

        f.f.body.push(
            Statement::Loop {
                body: d_body,
                continuing: Block::new(),
                break_if: None,
            },
            S,
        );
    }

    b.entry_point("main", [1, 1, 1], f.finish());
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
            (
                ShaderGroup::MatMulCoop,
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
            (ShaderGroup::MatMulCoop, coop),
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
            if group == ShaderGroup::MatMulCoop {
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
                ShaderEntry::Relu
                | ShaderEntry::Sigmoid
                | ShaderEntry::Neg
                | ShaderEntry::Silu
                | ShaderEntry::Gelu
                | ShaderEntry::SumAll
                | ShaderEntry::MeanAll
                | ShaderEntry::RoPE => vec!["src", "dst", "params"],
                ShaderEntry::Add | ShaderEntry::Mul | ShaderEntry::Greater => {
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
            }
        }

        let entries = [
            ShaderEntry::MatMul,
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
