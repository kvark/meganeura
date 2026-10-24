//! Convert an already selected compute implementation to a callable tile.

use naga::{
    AddressSpace, ArraySize, BinaryOperator, Binding, Block, BuiltIn, Expression, Function,
    FunctionArgument, Handle, Literal, LocalVariable, Module, Range, Scalar, ShaderStage, Span,
    Statement, Type, TypeInner,
};
use std::collections::HashMap;

const PREFIX: &str = "persistent_template_";

pub(crate) struct Tile {
    pub source: String,
    pub call: String,
    pub workgroup_size: [u32; 3],
    pub shared_bytes: u64,
    pub subgroups: bool,
    pub cooperative: Option<naga::CooperativeScope>,
    pub shared_aligned: bool,
}

fn validate(module: &Module) -> Result<naga::valid::ModuleInfo, String> {
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all() - naga::valid::ValidationFlags::BINDINGS,
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map_err(|error| format!("callable tile validation: {error:?}"))
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
    for argument in &mut function.arguments {
        let value = match argument.binding.take() {
            Some(Binding::BuiltIn(BuiltIn::WorkGroupId)) => "logical".into(),
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
    module.functions.append(function, Span::UNDEFINED);
    let info = validate(&module)?;
    let source =
        naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
            .map_err(|error| format!("callable tile emission: {error}"))?;
    let mut renamed = String::with_capacity(source.len());
    let mut start = 0;
    for (index, ch) in source.char_indices() {
        if !ch.is_ascii_alphanumeric() && ch != '_' {
            let token = &source[start..index];
            renamed.push_str(global_names.get(token).map_or(token, String::as_str));
            renamed.push(ch);
            start = index + ch.len_utf8();
        }
    }
    let token = &source[start..];
    renamed.push_str(global_names.get(token).map_or(token, String::as_str));
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
