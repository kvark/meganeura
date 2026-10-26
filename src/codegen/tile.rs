//! Convert an already selected compute implementation to a callable tile.

use naga::{
    AddressSpace, ArraySize, BinaryOperator, Binding, Block, BuiltIn, Expression, Function,
    FunctionArgument, Handle, Literal, LocalVariable, Module, Range, Scalar, ShaderStage, Span,
    Statement, Type, TypeInner,
};
use std::collections::HashMap;

const PREFIX: &str = "persistent_template_";

#[derive(Clone)]
pub(crate) struct Tile {
    pub source: String,
    pub call: String,
    pub workgroup_size: [u32; 3],
    pub shared_bytes: u64,
    pub subgroups: bool,
    pub cooperative: Option<naga::CooperativeScope>,
    pub shared_aligned: bool,
}

pub(super) fn validate(module: &Module) -> Result<naga::valid::ModuleInfo, String> {
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all() - naga::valid::ValidationFlags::BINDINGS,
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map_err(|error| format!("callable tile validation: {error:?}"))
}

/// Every physical lane reaches the original workgroup barriers. Only the
/// original lanes modify shared/storage memory; inactive subgroups also skip
/// cooperative products assigned to local accumulators. The caller supplies a
/// subgroup-uniform activity and scratch-group arguments, and the original
/// logical geometry.
pub(crate) fn mask_extra_lanes(mut tile: Tile, physical_threads: u32) -> Result<Tile, String> {
    let logical_threads = tile.workgroup_size.into_iter().product::<u32>();
    if physical_threads == logical_threads {
        return Ok(tile);
    }
    if !physical_threads.is_multiple_of(logical_threads) {
        return Err("physical worker size must be a multiple of each logical workgroup".into());
    }
    if tile.cooperative == Some(naga::CooperativeScope::Workgroup) {
        return Err("workgroup cooperative tiles require their physical workgroup size".into());
    }
    let source = tile.source.replace("$PREFIX", PREFIX);
    let mut module =
        naga::front::wgsl::parse_str(&source).map_err(|error| error.emit_to_string(&source))?;
    let info = validate(&module)?;
    let bool_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::BOOL),
        },
        Span::UNDEFINED,
    );
    let group_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::U32),
        },
        Span::UNDEFINED,
    );
    for (handle, function) in module.functions.iter_mut() {
        let index = function.arguments.len() as u32;
        function.arguments.push(FunctionArgument {
            name: Some(format!("{PREFIX}active_lane")),
            ty: bool_ty,
            binding: None,
        });
        let condition = function
            .expressions
            .append(Expression::FunctionArgument(index), Span::UNDEFINED);
        function.arguments.push(FunctionArgument {
            name: Some(format!("{PREFIX}tile_group_argument")),
            ty: group_ty,
            binding: None,
        });
        let group = function
            .expressions
            .append(Expression::FunctionArgument(index + 1), Span::UNDEFINED);
        append_call_arguments(&mut function.body, &[condition, group]);
        mask_writes(&mut function.body, condition, &info[handle], &module.types)?;
        if tile.cooperative == Some(naga::CooperativeScope::Subgroup) {
            let mut products = HashMap::new();
            collect_products(&function.body, &function.expressions, &mut products);
            let mut products = products.into_iter().collect::<Vec<_>>();
            products.sort_by_key(|&(value, _)| value);
            let products = products
                .into_iter()
                .map(|(value, pointer)| {
                    let product = function
                        .expressions
                        .append(function.expressions[value].clone(), Span::UNDEFINED);
                    function.expressions[value] = Expression::Load { pointer };
                    (value, (pointer, product))
                })
                .collect();
            mask_products(&mut function.body, condition, &products);
        }
    }
    let end = tile.call.rfind(')').ok_or("missing tile call arguments")?;
    tile.call
        .insert_str(end, ", $PREFIXactive_lane, $PREFIXtile_group");
    let mut names = HashMap::new();
    for (index, (_, var)) in module.global_variables.iter_mut().enumerate() {
        let original = var.name.clone().ok_or("unnamed tile global")?;
        let temporary = format!("{PREFIX}adapted_global_{index}_value");
        names.insert(temporary.clone(), original);
        var.name = Some(temporary);
    }
    let info = validate(&module)?;
    let source =
        naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
            .map_err(|error| format!("worker lane adaptation: {error}"))?;
    tile.source = rename_tokens(&source, &names).replace(PREFIX, "$PREFIX");
    Ok(tile)
}

// Cooperative products assigned back to a local accumulator can be skipped by
// whole subgroups. The original result becomes a load of that accumulator, so
// every SSA value used outside the conditional remains in scope.
fn collect_products(
    block: &Block,
    expressions: &naga::Arena<Expression>,
    products: &mut HashMap<Handle<Expression>, Handle<Expression>>,
) {
    let mut last_emitted = None;
    for statement in block {
        match *statement {
            Statement::Store { pointer, value }
                if matches!(
                    expressions[value],
                    Expression::CooperativeMultiplyAdd { .. }
                ) && matches!(expressions[pointer], Expression::LocalVariable(_))
                    && last_emitted == Some(value) =>
            {
                products.insert(value, pointer);
            }
            Statement::Block(ref body) => collect_products(body, expressions, products),
            Statement::If {
                ref accept,
                ref reject,
                ..
            } => {
                collect_products(accept, expressions, products);
                collect_products(reject, expressions, products);
            }
            Statement::Switch { ref cases, .. } => {
                for case in cases {
                    collect_products(&case.body, expressions, products);
                }
            }
            Statement::Loop {
                ref body,
                ref continuing,
                ..
            } => {
                collect_products(body, expressions, products);
                collect_products(continuing, expressions, products);
            }
            _ => {}
        }
        last_emitted = match *statement {
            Statement::Emit(ref range) => range.clone().last(),
            _ => None,
        };
    }
}

fn mask_products(
    block: &mut Block,
    condition: Handle<Expression>,
    products: &HashMap<Handle<Expression>, (Handle<Expression>, Handle<Expression>)>,
) {
    let mut output = Block::new();
    for (mut statement, span) in std::mem::take(block).span_into_iter() {
        match statement {
            Statement::Emit(ref range) => {
                for value in range.clone() {
                    if let Some(&(pointer, product)) = products.get(&value) {
                        let mut accept = Block::new();
                        emit(&mut accept, product, product);
                        accept.push(
                            Statement::Store {
                                pointer,
                                value: product,
                            },
                            span,
                        );
                        output.push(
                            Statement::If {
                                condition,
                                accept,
                                reject: Block::new(),
                            },
                            span,
                        );
                    }
                    emit(&mut output, value, value);
                }
                continue;
            }
            Statement::Block(ref mut body) => mask_products(body, condition, products),
            Statement::If {
                ref mut accept,
                ref mut reject,
                ..
            } => {
                mask_products(accept, condition, products);
                mask_products(reject, condition, products);
            }
            Statement::Switch { ref mut cases, .. } => {
                for case in cases {
                    mask_products(&mut case.body, condition, products);
                }
            }
            Statement::Loop {
                ref mut body,
                ref mut continuing,
                ..
            } => {
                mask_products(body, condition, products);
                mask_products(continuing, condition, products);
            }
            _ => {}
        }
        output.push(statement, span);
    }
    *block = output;
}

pub(crate) fn rename_tokens(source: &str, names: &HashMap<String, String>) -> String {
    let mut renamed = String::with_capacity(source.len());
    let mut start = 0;
    for (index, ch) in source.char_indices() {
        if !ch.is_ascii_alphanumeric() && ch != '_' {
            let token = &source[start..index];
            renamed.push_str(names.get(token).map_or(token, String::as_str));
            renamed.push(ch);
            start = index + ch.len_utf8();
        }
    }
    let token = &source[start..];
    renamed.push_str(names.get(token).map_or(token, String::as_str));
    renamed
}

/// Pass node uniforms as values whose caller obtains with workgroupUniformLoad.
/// This retains uniform-control-flow proofs for barriers inside tile helpers.
pub(crate) fn lift_uniforms(mut tile: Tile) -> Result<Tile, String> {
    let source = tile.source.replace("$PREFIX", PREFIX);
    let mut module =
        naga::front::wgsl::parse_str(&source).map_err(|e| e.emit_to_string(&source))?;
    let storage = module
        .global_variables
        .iter()
        .filter(|&(_, var)| matches!(var.space, AddressSpace::Storage { .. }))
        .map(|(_, var)| var.name.clone().unwrap())
        .collect::<Vec<_>>();
    let index_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Vector {
                size: naga::VectorSize::Bi,
                scalar: Scalar::U32,
            },
        },
        Span::UNDEFINED,
    );
    for name in storage {
        module.global_variables.append(
            naga::GlobalVariable {
                name: Some(format!("{name}_binding_index")),
                space: AddressSpace::Uniform,
                binding: None,
                ty: index_ty,
                init: None,
                memory_decorations: Default::default(),
            },
            Span::UNDEFINED,
        );
    }
    let uniforms: Vec<_> = module
        .global_variables
        .iter()
        .filter(|&(_, var)| var.space == AddressSpace::Uniform)
        .map(|(handle, var)| (handle, var.ty, var.name.clone().unwrap()))
        .collect();
    for (_, function) in module.functions.iter_mut() {
        let first_argument = function.arguments.len() as u32;
        let mut roots = HashMap::new();
        for (handle, expression) in function.expressions.iter() {
            let root = match *expression {
                Expression::GlobalVariable(global) => uniforms.iter().position(|u| u.0 == global),
                Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                    roots.get(&base).copied()
                }
                _ => None,
            };
            if let Some(root) = root {
                roots.insert(handle, root);
            }
        }
        let mut replacements = Vec::new();
        for (handle, expression) in function.expressions.iter() {
            match *expression {
                Expression::GlobalVariable(_) if roots.contains_key(&handle) => {
                    replacements.push((
                        handle,
                        Expression::FunctionArgument(first_argument + roots[&handle] as u32),
                    ));
                }
                Expression::Load { pointer } if roots.contains_key(&pointer) => {
                    let value = match function.expressions[pointer] {
                        Expression::GlobalVariable(_) => {
                            Expression::FunctionArgument(first_argument + roots[&pointer] as u32)
                        }
                        ref access => access.clone(),
                    };
                    replacements.push((handle, value));
                }
                _ => {}
            }
        }
        for (handle, replacement) in replacements {
            function.expressions[handle] = replacement;
        }
        let mut arguments = Vec::new();
        for (i, &(_, ty, ref name)) in uniforms.iter().enumerate() {
            function.arguments.push(FunctionArgument {
                name: Some(format!("{name}_argument")),
                ty,
                binding: None,
            });
            arguments.push(function.expressions.append(
                Expression::FunctionArgument(first_argument + i as u32),
                Span::UNDEFINED,
            ));
        }
        append_call_arguments(&mut function.body, &arguments);
    }
    let mut names = HashMap::new();
    for (index, (_, var)) in module.global_variables.iter_mut().enumerate() {
        let original = var.name.clone().ok_or("unnamed tile global")?;
        let temporary = format!("{PREFIX}lifted_global_{index}_value");
        names.insert(temporary.clone(), original);
        var.name = Some(temporary);
    }
    let info = validate(&module)?;
    let source =
        naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
            .map_err(|e| e.to_string())?;
    tile.source = rename_tokens(&source, &names).replace(PREFIX, "$PREFIX");
    if !uniforms.is_empty() {
        let end = tile.call.rfind(')').ok_or("missing tile call arguments")?;
        let values = uniforms
            .iter()
            .map(|u| u.2.replace(PREFIX, "$PREFIX"))
            .collect::<Vec<_>>()
            .join(", ");
        tile.call.insert_str(end, &format!(", {values}"));
    }
    Ok(tile)
}

fn append_call_arguments(block: &mut Block, values: &[Handle<Expression>]) {
    for statement in block.iter_mut() {
        match *statement {
            Statement::Call {
                ref mut arguments, ..
            } => arguments.extend_from_slice(values),
            Statement::Block(ref mut body) => append_call_arguments(body, values),
            Statement::If {
                ref mut accept,
                ref mut reject,
                ..
            } => {
                append_call_arguments(accept, values);
                append_call_arguments(reject, values);
            }
            Statement::Switch { ref mut cases, .. } => {
                for case in cases {
                    append_call_arguments(&mut case.body, values);
                }
            }
            Statement::Loop {
                ref mut body,
                ref mut continuing,
                ..
            } => {
                append_call_arguments(body, values);
                append_call_arguments(continuing, values);
            }
            _ => {}
        }
    }
}

fn mask_writes(
    block: &mut Block,
    condition: Handle<Expression>,
    info: &naga::valid::FunctionInfo,
    types: &naga::UniqueArena<Type>,
) -> Result<(), String> {
    for statement in block.iter_mut() {
        let mask = match *statement {
            Statement::Block(ref mut inner) => {
                mask_writes(inner, condition, info, types)?;
                false
            }
            Statement::If {
                ref mut accept,
                ref mut reject,
                ..
            } => {
                mask_writes(accept, condition, info, types)?;
                mask_writes(reject, condition, info, types)?;
                false
            }
            Statement::Switch { ref mut cases, .. } => {
                for case in cases {
                    mask_writes(&mut case.body, condition, info, types)?;
                }
                false
            }
            Statement::Loop {
                ref mut body,
                ref mut continuing,
                ..
            } => {
                mask_writes(body, condition, info, types)?;
                mask_writes(continuing, condition, info, types)?;
                false
            }
            Statement::Store { pointer, .. } => matches!(
                *info[pointer].ty.inner_with(types),
                TypeInner::Pointer {
                    space: AddressSpace::Storage { .. } | AddressSpace::WorkGroup,
                    ..
                } | TypeInner::ValuePointer {
                    space: AddressSpace::Storage { .. } | AddressSpace::WorkGroup,
                    ..
                }
            ),
            Statement::CooperativeStore { .. } => true,
            Statement::Atomic { .. }
            | Statement::ImageAtomic { .. }
            | Statement::ImageStore { .. } => {
                return Err(
                    "atomic/image tile side effects require an exact physical workgroup".into(),
                );
            }
            _ => false,
        };
        if mask {
            let write = std::mem::replace(statement, Statement::Break);
            *statement = Statement::If {
                condition,
                accept: Block::from_vec(vec![write]),
                reject: Block::new(),
            };
        }
    }
    Ok(())
}

fn zero_type(module: &mut Module, ty: Handle<Type>) -> Result<Handle<Type>, String> {
    match module.types[ty].inner {
        TypeInner::Scalar(Scalar {
            kind: naga::ScalarKind::Bool,
            ..
        })
        | TypeInner::Vector {
            scalar:
                Scalar {
                    kind: naga::ScalarKind::Bool,
                    ..
                },
            ..
        } => Err("boolean tile scratch requires backend layout accounting".into()),
        TypeInner::Atomic(scalar) => Ok(module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(scalar),
            },
            Span::UNDEFINED,
        )),
        TypeInner::Scalar(_) | TypeInner::Vector { .. } | TypeInner::Matrix { .. } => Ok(ty),
        _ => Err("tile scratch requires a scalar, vector, matrix or flat array".into()),
    }
}

fn emit(block: &mut Block, first: Handle<Expression>, last: Handle<Expression>) {
    block.push(
        Statement::Emit(Range::new_from_bounds(first, last)),
        Span::UNDEFINED,
    );
}

fn omit_argument_emits(block: &mut Block, expressions: &naga::Arena<Expression>) {
    let mut output = Block::new();
    for (mut statement, span) in std::mem::take(block).span_into_iter() {
        match statement {
            Statement::Emit(ref range) => {
                for value in range.clone() {
                    if !matches!(expressions[value], Expression::FunctionArgument(_)) {
                        emit(&mut output, value, value);
                    }
                }
                continue;
            }
            Statement::Block(ref mut body) => omit_argument_emits(body, expressions),
            Statement::If {
                ref mut accept,
                ref mut reject,
                ..
            } => {
                omit_argument_emits(accept, expressions);
                omit_argument_emits(reject, expressions);
            }
            Statement::Switch { ref mut cases, .. } => {
                for case in cases {
                    omit_argument_emits(&mut case.body, expressions);
                }
            }
            Statement::Loop {
                ref mut body,
                ref mut continuing,
                ..
            } => {
                omit_argument_emits(body, expressions);
                omit_argument_emits(continuing, expressions);
            }
            _ => {}
        }
        output.push(statement, span);
    }
    *block = output;
}

/// Ordinary dispatches initialize shared storage once per workgroup. A reused
/// worker must repeat that initialization for each logical tile, before its body.
fn reset_scratch(
    module: &mut Module,
    function: &mut Function,
    threads: u32,
) -> Result<(u64, bool), String> {
    let u32_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::U32),
        },
        Span::UNDEFINED,
    );
    let lane = function.expressions.append(
        Expression::FunctionArgument(function.arguments.len() as u32),
        Span::UNDEFINED,
    );
    function.arguments.push(FunctionArgument {
        name: Some("tile_lane".into()),
        ty: u32_ty,
        binding: None,
    });
    let mut layouter = naga::proc::Layouter::default();
    layouter
        .update(module.to_ctx())
        .map_err(|error| error.to_string())?;
    let globals: Vec<_> = module
        .global_variables
        .iter()
        .filter_map(|(handle, var)| {
            (var.space == AddressSpace::WorkGroup).then_some((handle, var.ty, var.init))
        })
        .collect();
    let mut bytes = 0;
    let mut aligned = true;
    let mut prefix = Block::new();
    let mut leader = Block::new();
    for (global, ty, init) in globals {
        if init.is_some() {
            return Err("initialized workgroup globals require explicit tile reset".into());
        }
        let layout = layouter[ty];
        aligned &= layout.size.is_multiple_of(16);
        // Include worst-case inter-global alignment padding, independent of the
        // backend's placement of globals from other handlers in this region.
        bytes += u64::from(layout.size) + u64::from(layout.alignment.round_up(1)) - 1;
        let pointer = function
            .expressions
            .append(Expression::GlobalVariable(global), Span::UNDEFINED);
        if let TypeInner::Array {
            base,
            size: ArraySize::Constant(count),
            ..
        } = module.types[ty].inner
        {
            let zero_ty = zero_type(module, base)?;
            let zero = function
                .expressions
                .append(Expression::ZeroValue(zero_ty), Span::UNDEFINED);
            let count = function.expressions.append(
                Expression::Literal(Literal::U32(count.get())),
                Span::UNDEFINED,
            );
            let cursor = function.local_variables.append(
                LocalVariable {
                    name: Some("tile_reset_index".into()),
                    ty: u32_ty,
                    init: None,
                },
                Span::UNDEFINED,
            );
            let cursor = function
                .expressions
                .append(Expression::LocalVariable(cursor), Span::UNDEFINED);
            prefix.push(
                Statement::Store {
                    pointer: cursor,
                    value: lane,
                },
                Span::UNDEFINED,
            );
            let mut body = Block::new();
            let index = function
                .expressions
                .append(Expression::Load { pointer: cursor }, Span::UNDEFINED);
            let done = function.expressions.append(
                Expression::Binary {
                    op: BinaryOperator::GreaterEqual,
                    left: index,
                    right: count,
                },
                Span::UNDEFINED,
            );
            emit(&mut body, index, done);
            body.push(
                Statement::If {
                    condition: done,
                    accept: Block::from_vec(vec![Statement::Break]),
                    reject: Block::new(),
                },
                Span::UNDEFINED,
            );
            let element = function.expressions.append(
                Expression::Access {
                    base: pointer,
                    index,
                },
                Span::UNDEFINED,
            );
            emit(&mut body, element, element);
            body.push(
                Statement::Store {
                    pointer: element,
                    value: zero,
                },
                Span::UNDEFINED,
            );
            let mut continuing = Block::new();
            let index = function
                .expressions
                .append(Expression::Load { pointer: cursor }, Span::UNDEFINED);
            let stride = function
                .expressions
                .append(Expression::Literal(Literal::U32(threads)), Span::UNDEFINED);
            let next = function.expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: index,
                    right: stride,
                },
                Span::UNDEFINED,
            );
            // Literal expressions are always in scope and must not be emitted.
            emit(&mut continuing, index, index);
            emit(&mut continuing, next, next);
            continuing.push(
                Statement::Store {
                    pointer: cursor,
                    value: next,
                },
                Span::UNDEFINED,
            );
            prefix.push(
                Statement::Loop {
                    body,
                    continuing,
                    break_if: None,
                },
                Span::UNDEFINED,
            );
        } else {
            let zero_ty = zero_type(module, ty)?;
            let zero = function
                .expressions
                .append(Expression::ZeroValue(zero_ty), Span::UNDEFINED);
            leader.push(
                Statement::Store {
                    pointer,
                    value: zero,
                },
                Span::UNDEFINED,
            );
        }
    }
    if !leader.is_empty() {
        let zero = function
            .expressions
            .append(Expression::Literal(Literal::U32(0)), Span::UNDEFINED);
        let condition = function.expressions.append(
            Expression::Binary {
                op: BinaryOperator::Equal,
                left: lane,
                right: zero,
            },
            Span::UNDEFINED,
        );
        emit(&mut prefix, condition, condition);
        prefix.push(
            Statement::If {
                condition,
                accept: leader,
                reject: Block::new(),
            },
            Span::UNDEFINED,
        );
    }
    if bytes != 0 {
        prefix.push(
            Statement::ControlBarrier(naga::Barrier::WORK_GROUP),
            Span::UNDEFINED,
        );
    }
    prefix.extend_block(std::mem::take(&mut function.body));
    function.body = prefix;
    Ok((bytes, aligned))
}

pub(crate) fn from_module(source: &str, entry_point: &str) -> Result<Tile, String> {
    if source.contains(PREFIX) {
        return Err("tile namespace already occurs in source".into());
    }
    let mut module =
        naga::front::wgsl::parse_str(source).map_err(|error| error.emit_to_string(source))?;
    module
        .entry_points
        .retain(|entry| entry.name == entry_point);
    if module.entry_points.len() != 1 || !module.overrides.is_empty() {
        return Err("tile requires one selected compute entry and resolved overrides".into());
    }
    validate(&module)?;
    naga::compact::compact(&mut module, naga::compact::KeepUnused::No);
    if module
        .global_variables
        .iter()
        .any(|(_, var)| var.space == AddressSpace::Private)
    {
        return Err("private globals require explicit per-tile initialization".into());
    }
    if module.types.iter().any(|(_, ty)| {
        matches!(
            ty.inner,
            TypeInner::Pointer {
                space: AddressSpace::Storage { .. },
                ..
            }
        )
    }) {
        return Err("storage pointer helpers require coherent-provenance qualification".into());
    }
    let entry = module.entry_points.pop().unwrap();
    if entry.stage != ShaderStage::Compute
        || entry.function.result.is_some()
        || entry.workgroup_size_overrides.is_some()
    {
        return Err("tile requires a compute entry with fixed workgroup geometry".into());
    }
    let geometry = entry.workgroup_size;
    let shape = format!(
        "vec3<u32>({}u, {}u, {}u)",
        geometry[0], geometry[1], geometry[2]
    );
    let mut subgroups = false;
    let mut arguments = Vec::new();
    let mut function = entry.function;
    let mut workgroup_argument = None;
    for (index, argument) in function.arguments.iter_mut().enumerate() {
        let value = match argument.binding.take() {
            Some(Binding::BuiltIn(BuiltIn::WorkGroupId)) => {
                workgroup_argument = Some(index as u32);
                "logical".into()
            }
            Some(Binding::BuiltIn(BuiltIn::LocalInvocationId)) => "lid".into(),
            Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)) => "lane".into(),
            Some(Binding::BuiltIn(BuiltIn::GlobalInvocationId)) => {
                format!("(logical * {shape} + lid)")
            }
            Some(Binding::BuiltIn(BuiltIn::WorkGroupSize)) => shape.clone(),
            Some(Binding::BuiltIn(BuiltIn::NumWorkGroups)) => "grid".into(),
            Some(Binding::BuiltIn(builtin)) => {
                subgroups = true;
                match builtin {
                    BuiltIn::SubgroupId => "persistent_subgroup_id".into(),
                    BuiltIn::SubgroupInvocationId => "persistent_subgroup_lane".into(),
                    BuiltIn::SubgroupSize => "persistent_subgroup_size".into(),
                    BuiltIn::NumSubgroups => "persistent_num_subgroups".into(),
                    _ => return Err(format!("unsupported tile builtin {builtin:?}")),
                }
            }
            _ => return Err("tile entry arguments must be direct compute builtins".into()),
        };
        arguments.push(value);
    }
    if let Some(argument) = workgroup_argument
        && !subgroups
        && !module
            .types
            .iter()
            .any(|(_, ty)| matches!(ty.inner, TypeInner::CooperativeMatrix { .. }))
    {
        let scalar = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::UNDEFINED,
        );
        let first = function.arguments.len() as u32;
        for axis in ["x", "y", "z"] {
            function.arguments.push(FunctionArgument {
                name: Some(format!("{PREFIX}workgroup_{axis}")),
                ty: scalar,
                binding: None,
            });
            arguments.push(format!("logical.{axis}"));
        }
        // Separate coordinate arguments let barrier analysis distinguish
        // query-dependent loops from independent heads in the same worker.
        // Whole-vector uses retain their conservative dependency on all axes.
        let components = function
            .expressions
            .iter()
            .filter_map(|(handle, expression)| match *expression {
                Expression::AccessIndex { base, index }
                    if matches!(function.expressions[base], Expression::FunctionArgument(i) if i == argument) =>
                {
                    Some((handle, index))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        for (handle, component) in components {
            function.expressions[handle] = Expression::FunctionArgument(first + component);
        }
        omit_argument_emits(&mut function.body, &function.expressions);
    }
    let (shared_bytes, shared_aligned) =
        reset_scratch(&mut module, &mut function, geometry.into_iter().product())?;
    arguments.push("lane".into());
    let named_types: Vec<_> = module
        .types
        .iter()
        .filter(|&(handle, _)| {
            !module
                .special_types
                .predeclared_types
                .values()
                .any(|&special| special == handle)
        })
        .filter_map(|(handle, ty)| {
            ty.name.as_ref().map(|name| {
                let mut ty = ty.clone();
                ty.name = Some(format!("{PREFIX}internal_{name}"));
                (handle, ty)
            })
        })
        .collect();
    for (handle, ty) in named_types {
        module.types.replace(handle, ty);
    }
    for (_, constant) in module.constants.iter_mut() {
        if let Some(ref name) = constant.name {
            constant.name = Some(format!("{PREFIX}internal_{name}"));
        }
    }
    for (_, other) in module.functions.iter_mut() {
        if let Some(ref name) = other.name {
            other.name = Some(format!("{PREFIX}internal_{name}"));
        }
    }
    let mut global_names = HashMap::new();
    for (index, (_, var)) in module.global_variables.iter_mut().enumerate() {
        let name = var.name.as_deref().ok_or("unnamed tile global")?;
        if name == "entry" {
            return Err("tile global collides with callable entry".into());
        }
        // Naga's writer sanitizes names ending in digits. Use unique temporary
        // identifiers ending in a letter, then restore the binding vocabulary
        // in one token pass so names cannot collide through partial replacement.
        let temporary = format!("{PREFIX}global_{index}_value");
        global_names.insert(temporary.clone(), format!("{PREFIX}{name}"));
        var.name = Some(temporary);
    }
    function.name = Some(format!("{PREFIX}entry"));
    subgroups |= std::iter::once(&function)
        .chain(module.functions.iter().map(|(_, function)| function))
        .any(|function| {
            function.expressions.iter().any(|(_, expression)| {
                matches!(
                    *expression,
                    Expression::SubgroupBallotResult | Expression::SubgroupOperationResult { .. }
                )
            })
        });
    module.functions.append(function, Span::UNDEFINED);
    let info = validate(&module)?;
    let source =
        naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
            .map_err(|error| format!("callable tile emission: {error}"))?;
    let renamed = rename_tokens(&source, &global_names);
    Ok(Tile {
        source: renamed.replace(PREFIX, "$PREFIX"),
        call: format!("$PREFIXentry({})", arguments.join(", ")),
        workgroup_size: geometry,
        shared_bytes,
        subgroups,
        cooperative: module
            .types
            .iter()
            .filter_map(|(_, ty)| match ty.inner {
                TypeInner::CooperativeMatrix { scope, .. } => Some(scope),
                _ => None,
            })
            .max_by_key(|scope| match *scope {
                naga::CooperativeScope::Subgroup => 0,
                naga::CooperativeScope::Workgroup => 1,
            }),
        shared_aligned,
    })
}
