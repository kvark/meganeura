//! Share resource tables and scratch across the selected whole-graph handlers.
use super::tile;
use blade_graphics as bg;
use naga::{AddressSpace, ArraySize, Handle, ScalarKind, Type, TypeInner};
use std::{
    collections::{BTreeSet, HashMap},
    fmt::Write,
};

pub(crate) struct BufferBinding {
    pub slot: usize,
    pub stride: u32,
    pub write: bool,
}

pub(crate) struct Kernel {
    pub buffers: Vec<BufferBinding>,
    pub source: String,
    pub case: String,
    pub subgroups: bool,
    pub subgroup_threads: u32,
    pub workgroup_cooperative: bool,
    pub batch_axis: Option<usize>,
    pub batch_size: u32,
}

fn packing_width(maximum: u32, extent: u32, subgroups: bool) -> u32 {
    if extent == 0 {
        return 1;
    }
    [4, 3, 2, 1]
        .into_iter()
        .find(|&width| {
            width <= maximum && (width != 3 || !subgroups) && extent.is_multiple_of(width)
        })
        .unwrap()
}

impl Kernel {
    pub fn tiles_per_claim(&self, grid: [u32; 3]) -> u32 {
        match self.batch_axis {
            Some(axis) if !grid.contains(&0) => {
                packing_width(self.batch_size, grid[axis], self.subgroups)
            }
            None if !grid.contains(&0) => self.batch_size,
            _ => 1,
        }
    }
}

pub(crate) struct View {
    pub name: String,
    ty: String,
}

struct Scratch {
    ty: String,
    slot: usize,
    count: u32,
    stride: u32,
}

pub(crate) struct Builder {
    pub threads: u32,
    pub parameter_words: usize,
    pub views: Vec<View>,
    scratch: Vec<Scratch>,
    pub batch_limit: u32,
}

impl Builder {
    pub fn new(threads: u32) -> Self {
        Self {
            threads,
            parameter_words: 0,
            views: Vec::new(),
            scratch: Vec::new(),
            batch_limit: 4,
        }
    }

    pub fn shared_bytes(&self) -> u64 {
        64 + (self.parameter_words.max(4) * 4) as u64
            + self
                .scratch
                .iter()
                .map(|s| u64::from(s.count * s.stride))
                .sum::<u64>()
    }

    pub fn add(
        &mut self,
        body: tile::Tile,
        layout: &bg::ShaderDataLayout,
        index: usize,
        grids: &[[u32; 3]],
        sequential: bool,
    ) -> Result<Kernel, String> {
        let logical_threads = body.workgroup_size.into_iter().product::<u32>();
        // Amortize queue publication only for large, independent optimizer grids.
        // Smaller grids retain the straight-line handler and its parallelism.
        if sequential
            && logical_threads == self.threads
            && grids.iter().any(|grid| {
                grid.iter()
                    .fold(1u64, |count, &n| count.saturating_mul(u64::from(n)))
                    >= 1024
            })
        {
            return self.add_inner(body, layout, index, None, 4);
        }
        let subgroups = body.subgroups || body.cooperative.is_some();
        let batch_size = self.batch_limit.min(self.threads / logical_threads);
        if batch_size > 1 {
            let mut axes = [0, 1, 2];
            axes.sort_by_key(|&axis| {
                grids.iter().fold(0u64, |cost, grid| {
                    let count = grid
                        .iter()
                        .fold(1u64, |count, &n| count.saturating_mul(u64::from(n)));
                    cost.saturating_add(
                        count / u64::from(packing_width(batch_size, grid[axis], subgroups)),
                    )
                })
            });
            for axis in axes {
                let batch_size = grids
                    .iter()
                    .filter(|grid| !grid.contains(&0))
                    .map(|grid| packing_width(batch_size, grid[axis], subgroups))
                    .max()
                    .unwrap_or(if grids.is_empty() { batch_size } else { 1 });
                if batch_size == 1 {
                    continue;
                }
                let mut candidate = Self::new(self.threads);
                let Ok(kernel) =
                    candidate.add_inner(body.clone(), layout, index, Some(axis), batch_size)
                else {
                    continue;
                };
                let mut source = candidate.source(std::slice::from_ref(&kernel), 1);
                // Admission only: apply the scope-aware cooperative checker
                // to barrier control, including helper arguments and locals.
                for barrier in [
                    "workgroupBarrier();",
                    "storageBarrier();",
                    "textureBarrier();",
                ] {
                    source =
                        source.replace(barrier, &format!("graph_packing_sync_probe(); {barrier}"));
                }
                if !source.contains("enable wgpu_cooperative_matrix;") {
                    source.insert_str(0, "enable wgpu_cooperative_matrix;\n");
                }
                source.push_str(include_str!("../shaders/graph_packing_probe.wgsl"));
                let Ok(module) = naga::front::wgsl::parse_str(&source) else {
                    continue;
                };
                if tile::validate(&module).is_ok() {
                    return self.add_inner(body, layout, index, Some(axis), batch_size);
                }
            }
        }
        self.add_inner(body, layout, index, None, 1)
    }

    fn add_inner(
        &mut self,
        body: tile::Tile,
        layout: &bg::ShaderDataLayout,
        index: usize,
        batch_axis: Option<usize>,
        batch_size: u32,
    ) -> Result<Kernel, String> {
        let packing_size = if batch_axis.is_some() { batch_size } else { 1 };
        if packing_size > 1
            && (body.source.contains("workgroupUniformLoad(") || body.source.contains("switch"))
        {
            return Err(
                "packed tiles require qualified barrier control and no workgroup broadcasts".into(),
            );
        }
        let logical_threads = body.workgroup_size.into_iter().product::<u32>();
        let body = tile::mask_extra_lanes(body, self.threads)?;
        let body = tile::lift_uniforms(body)?;
        self.parameter_words = self.parameter_words.max(
            2 * layout.bindings.len()
                + layout
                    .bindings
                    .iter()
                    .map(|b| match b.1 {
                        bg::ShaderBinding::Plain { size } => size.div_ceil(4) as usize,
                        _ => 0,
                    })
                    .sum::<usize>(),
        );
        self.parameter_words = self.parameter_words.next_multiple_of(4);
        let prefix = format!("graph_handler_{index}_");
        let source = body.source.replace("$PREFIX", &prefix);
        let module =
            naga::front::wgsl::parse_str(&source).map_err(|e| e.emit_to_string(&source))?;
        tile::validate(&module)?;
        let mut layouter = naga::proc::Layouter::default();
        layouter
            .update(module.to_ctx())
            .map_err(|e| e.to_string())?;
        let mut replacements = Vec::new();
        let mut names = HashMap::new();
        let mut buffers = Vec::new();
        let mut parameters = String::new();
        let mut shared_slots = HashMap::new();
        let mut shared_offsets = HashMap::new();
        let mut buffer_offsets = HashMap::new();
        for (handle, global) in module.global_variables.iter() {
            let name = global.name.as_deref().ok_or("unnamed graph global")?;
            let range = module
                .global_variables
                .get_span(handle)
                .to_range()
                .ok_or("missing graph declaration span")?;
            match global.space {
                AddressSpace::Storage { access } => {
                    let original = name
                        .strip_prefix(&prefix)
                        .ok_or("handler binding has no namespace")?;
                    let slot = layout
                        .bindings
                        .iter()
                        .position(|b| b.0 == original)
                        .ok_or_else(|| format!("missing binding {original}"))?;
                    let ty = type_name(&module.types, global.ty)?;
                    let stride = match module.types[global.ty].inner {
                        TypeInner::Array { stride, .. } => stride,
                        _ => {
                            return Err(
                                "whole-graph storage requires directly indexed arrays".into()
                            );
                        }
                    };
                    let view = self
                        .views
                        .iter()
                        .position(|v| v.ty == ty)
                        .unwrap_or_else(|| {
                            let i = self.views.len();
                            self.views.push(View {
                                name: format!("graph_buffers_{i}"),
                                ty,
                            });
                            i
                        });
                    names.insert(
                        name.to_owned(),
                        format!(
                            "{}[{name}_binding_index_argument.x].data",
                            self.views[view].name
                        ),
                    );
                    buffer_offsets
                        .insert(name.to_owned(), format!("{name}_binding_index_argument.y"));
                    buffers.push(BufferBinding {
                        slot,
                        stride,
                        write: access.contains(naga::StorageAccess::STORE),
                    });
                    replacements.push((range, String::new()));
                }
                AddressSpace::Uniform => {
                    let original = name
                        .strip_prefix(&prefix)
                        .ok_or("uniform has no namespace")?;
                    let binding_index = original.strip_suffix("_binding_index");
                    let slot = layout
                        .bindings
                        .iter()
                        .position(|b| b.0 == binding_index.unwrap_or(original))
                        .ok_or_else(|| format!("missing uniform {original}"))?;
                    let ty = type_name(&module.types, global.ty)?;
                    let base = 2 * layout.bindings.len()
                        + layout.bindings[..slot]
                            .iter()
                            .map(|b| match b.1 {
                                bg::ShaderBinding::Plain { size } => size.div_ceil(4) as usize,
                                _ => 0,
                            })
                            .sum::<usize>();
                    let value = if binding_index.is_some() {
                        format!(
                            "vec2<u32>(graph_parameter_values[{}u], graph_parameter_values[{}u])",
                            2 * slot,
                            2 * slot + 1
                        )
                    } else {
                        uniform_value(&module.types, global.ty, base, 0)?
                    };
                    writeln!(parameters, "let {name}: {ty} = {value};").unwrap();
                    replacements.push((range, String::new()));
                }
                AddressSpace::WorkGroup => {
                    if batch_axis.is_none() && batch_size > 1 {
                        return Err("sequential batches require scratch-free handlers".into());
                    }
                    let (base, count, stride, array) = match module.types[global.ty].inner {
                        TypeInner::Array {
                            base,
                            size: ArraySize::Constant(count),
                            stride,
                        } => (base, count.get(), stride, true),
                        TypeInner::Array { .. } => return Err("dynamic workgroup scratch".into()),
                        _ => (global.ty, 1, layouter[global.ty].to_stride(), false),
                    };
                    let ty = type_name(&module.types, base)?;
                    let slot = shared_slots.entry(ty.clone()).or_insert(0);
                    let count = (count * stride).next_multiple_of(16).div_ceil(stride);
                    let bank = if let Some(i) = self
                        .scratch
                        .iter()
                        .position(|s| s.ty == ty && s.slot == *slot)
                    {
                        self.scratch[i].count = self.scratch[i].count.max(count * packing_size);
                        i
                    } else {
                        let i = self.scratch.len();
                        self.scratch.push(Scratch {
                            ty,
                            slot: *slot,
                            count: count * packing_size,
                            stride,
                        });
                        i
                    };
                    *slot += 1;
                    let offset = if packing_size == 1 {
                        "0u".to_owned()
                    } else {
                        format!("{count}u * {prefix}tile_group_argument")
                    };
                    if array && packing_size > 1 {
                        shared_offsets.insert(name.to_owned(), count);
                    }
                    names.insert(
                        name.to_owned(),
                        format!(
                            "graph_scratch_{bank}{}",
                            if array {
                                String::new()
                            } else {
                                format!("[{offset}]")
                            }
                        ),
                    );
                    replacements.push((range, String::new()));
                }
                AddressSpace::Private => {}
                _ => return Err("whole-graph handlers require buffer resources".into()),
            }
        }
        // Naga uniquifies argument names across helpers. Resolve each resource
        // through the descriptor argument of the function that actually uses it.
        for (handle, function) in module.functions.iter() {
            let range = module
                .functions
                .get_span(handle)
                .to_range()
                .ok_or("missing graph function span")?;
            let mut local_names = names.clone();
            let mut local_offsets = buffer_offsets.clone();
            let mut function_source = source[range.clone()].to_owned();
            if packing_size > 1 {
                let argument = function
                    .arguments
                    .iter()
                    .filter_map(|a| a.name.as_deref())
                    .find(|name| name.starts_with(&format!("{prefix}tile_group_argument")))
                    .ok_or("missing tile group argument")?;
                for name in local_names.values_mut() {
                    *name = name.replace(&format!("{prefix}tile_group_argument"), argument);
                }
                function_source = offset_scratch(&function_source, &shared_offsets, argument)?;
            }
            for (_, global) in module.global_variables.iter() {
                if !matches!(global.space, AddressSpace::Storage { .. }) {
                    continue;
                }
                let name = global.name.as_deref().unwrap();
                let prefix = format!("{name}_binding_index_argument");
                let argument = function
                    .arguments
                    .iter()
                    .filter_map(|a| a.name.as_deref())
                    .find(|name| name.starts_with(&prefix))
                    .ok_or("missing descriptor argument")?;
                let replacement = local_names.get_mut(name).unwrap();
                *replacement = replacement.replace(&prefix, argument);
                let offset = local_offsets.get_mut(name).unwrap();
                *offset = offset.replace(&prefix, argument);
            }
            function_source = offset_arrays(&function_source, &local_offsets)?;
            replacements.push((
                range.clone(),
                tile::rename_tokens(&function_source, &local_names),
            ));
        }
        let mut source = source;
        replacements.sort_by_key(|r| r.0.start);
        for (range, replacement) in replacements.into_iter().rev() {
            source.replace_range(range, &replacement);
        }
        source = tile::rename_tokens(&source, &names);
        let lane = if body.subgroups || body.cooperative.is_some() {
            format!(
                "((graph_subgroup_id % ({logical_threads}u / graph_subgroup_size)) * graph_subgroup_size + graph_subgroup_lane)"
            )
        } else {
            format!("(physical_lane % {logical_threads}u)")
        };
        let subgroups = if body.subgroups {
            format!(
                "let persistent_subgroup_id = graph_subgroup_id % ({logical_threads}u / graph_subgroup_size);\nlet persistent_subgroup_lane = graph_subgroup_lane;\nlet persistent_subgroup_size = graph_subgroup_size;\nlet persistent_num_subgroups = {logical_threads}u / graph_subgroup_size;"
            )
        } else {
            String::new()
        };
        let group = if body.subgroups || body.cooperative.is_some() {
            format!("graph_subgroup_id / ({logical_threads}u / graph_subgroup_size)")
        } else {
            format!("physical_lane / {logical_threads}u")
        };
        let sequential = batch_axis.is_none() && batch_size > 1;
        let tile = if sequential {
            "(ticket.y + graph_repeat)"
        } else {
            "ticket.y"
        };
        let logical = if let Some(axis) = batch_axis {
            let coordinate = ["x", "y", "z"][axis];
            let mut parts = [String::new(), String::new(), String::new()];
            parts[axis] = format!(
                "ticket.y % grid.{coordinate} + min(graph_tile_group, (node.grid.w >> 16u) - 1u)"
            );
            let remaining = (0..3).filter(|&i| i != axis).collect::<Vec<_>>();
            parts[remaining[0]] = format!(
                "(ticket.y / grid.{coordinate}) % grid.{}",
                ["x", "y", "z"][remaining[0]]
            );
            parts[remaining[1]] = format!(
                "ticket.y / (grid.{coordinate} * grid.{})",
                ["x", "y", "z"][remaining[0]]
            );
            parts
        } else {
            [
                format!("{tile} % grid.x"),
                format!("({tile} / grid.x) % grid.y"),
                format!("{tile} / (grid.x * grid.y)"),
            ]
        };
        let (repeat_begin, repeat_end) = if sequential {
            include_str!("../shaders/graph_repeat.wgsl")
                .split_once("$BODY")
                .unwrap()
        } else {
            ("", "")
        };
        let case = include_str!("../shaders/graph_handler.wgsl")
            .replace("$INDEX", &index.to_string())
            .replace("$LANE", &lane)
            .replace("$X", &body.workgroup_size[0].to_string())
            .replace("$Y", &body.workgroup_size[1].to_string())
            .replace("$SUBGROUPS", &subgroups)
            .replace("$GROUP", &group)
            .replace("$LAST_GROUP", &(packing_size - 1).to_string())
            .replace(
                "$ACTIVE_GROUPS",
                if sequential {
                    "1u"
                } else {
                    "(node.grid.w >> 16u)"
                },
            )
            .replace("$REPEAT_BEGIN", repeat_begin)
            .replace("$REPEAT_END", repeat_end)
            .replace("$LOGICAL_X", &logical[0])
            .replace("$LOGICAL_Y", &logical[1])
            .replace("$LOGICAL_Z", &logical[2])
            .replace("$PARAMETERS", &parameters)
            .replace(
                "$CALL",
                &body
                    .call
                    .replace("logical.x", &format!("({})", logical[0]))
                    .replace("logical.y", &format!("({})", logical[1]))
                    .replace("logical.z", &format!("({})", logical[2]))
                    .replace("$PREFIXactive_lane", "graph_active_lane")
                    .replace("$PREFIXtile_group", "graph_tile_group")
                    .replace("$PREFIX", &prefix),
            );
        Ok(Kernel {
            buffers,
            source,
            case,
            subgroups: body.subgroups || body.cooperative.is_some(),
            subgroup_threads: if body.subgroups || body.cooperative.is_some() {
                logical_threads
            } else {
                0
            },
            workgroup_cooperative: body.cooperative == Some(naga::CooperativeScope::Workgroup),
            batch_axis,
            batch_size,
        })
    }

    pub fn source(&self, kernels: &[Kernel], count: usize) -> String {
        assert!(count > 0);
        let mut enables = BTreeSet::new();
        let mut handlers = String::new();
        for kernel in kernels {
            for line in kernel.source.lines() {
                if line.trim_start().starts_with("enable ") {
                    if !line.contains("wgpu_memory_fence") && !line.contains("wgpu_binding_array") {
                        enables.insert(line.trim());
                    }
                } else {
                    writeln!(handlers, "{line}").unwrap();
                }
            }
        }
        let mut buffers = String::new();
        for (i, view) in self.views.iter().enumerate() {
            // All typed views map to the same descriptor binding. The backend
            // preserves NonPrivatePointer when any of those aliases is written.
            // Availability/visibility comes from the tile-boundary fences.
            writeln!(buffers, "struct GraphBuffer{i} {{ data: {} }}\nvar<storage, read_write> {}: binding_array<GraphBuffer{i}, {count}>;", view.ty, view.name).unwrap();
        }
        let mut shared = String::new();
        for (i, s) in self.scratch.iter().enumerate() {
            writeln!(
                shared,
                "var<workgroup> graph_scratch_{i}: array<{}, {}>;",
                s.ty, s.count
            )
            .unwrap();
        }
        include_str!("../shaders/graph_worker.wgsl")
            .replace(
                "$ENABLES",
                &enables.into_iter().collect::<Vec<_>>().join("\n"),
            )
            .replace("$HANDLERS", &handlers)
            .replace("$BUFFERS", &buffers)
            .replace("$SHARED", &shared)
            .replace(
                "$GEOMETRY",
                if kernels.iter().any(|k| k.subgroups) {
                    include_str!("../shaders/graph_subgroups.wgsl")
                } else {
                    ""
                },
            )
            .replace(
                "$SUBGROUP_THREADS",
                &kernels
                    .iter()
                    .filter(|k| k.subgroups)
                    .map(|k| k.subgroup_threads)
                    .fold(self.threads, |mut a, mut b| {
                        while b != 0 {
                            let r = a % b;
                            a = b;
                            b = r;
                        }
                        a
                    })
                    .to_string(),
            )
            .replace("$THREADS", &self.threads.to_string())
            .replace("$PARAMETER_WORDS", &self.parameter_words.max(4).to_string())
            .replace(
                "$SUBGROUPS",
                &if kernels.iter().any(|k| k.subgroups) {
                    include_str!("../shaders/persistent_subgroups.wgsl")
                        .replace("persistent_", "graph_")
                } else {
                    String::new()
                },
            )
            .replace(
                "$CASES",
                &kernels
                    .iter()
                    .map(|k| k.case.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
    }
}

fn offset_scratch(
    source: &str,
    arrays: &HashMap<String, u32>,
    group: &str,
) -> Result<String, String> {
    let offsets = arrays
        .iter()
        .map(|(name, count)| (name.clone(), format!("{count}u * {group}")))
        .collect();
    offset_arrays(source, &offsets)
}

fn offset_arrays(source: &str, arrays: &HashMap<String, String>) -> Result<String, String> {
    let mut insertions = Vec::new();
    let identifier = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    for (name, offset) in arrays {
        for (start, _) in source.match_indices(name) {
            let end = start + name.len();
            if start != 0 && identifier(source.as_bytes()[start - 1])
                || end < source.len() && identifier(source.as_bytes()[end])
            {
                continue;
            }
            let opening = end + source[end..].len() - source[end..].trim_start().len();
            if source.as_bytes().get(opening) != Some(&b'[') {
                return Err(format!("graph array {name} must be directly indexed"));
            }
            let mut depth = 0;
            let closing = source[opening..]
                .char_indices()
                .find_map(|(i, c)| {
                    match c {
                        '[' => depth += 1,
                        ']' => depth -= 1,
                        _ => {}
                    }
                    (depth == 0).then_some(opening + i)
                })
                .ok_or("unterminated array index")?;
            insertions.push((opening + 1, format!("{offset} + u32(")));
            insertions.push((closing, ")".to_owned()));
        }
    }
    let mut output = source.to_owned();
    insertions.sort_by_key(|&(position, _)| position);
    for (position, text) in insertions.into_iter().rev() {
        output.insert_str(position, &text);
    }
    Ok(output)
}

fn type_name(types: &naga::UniqueArena<Type>, ty: Handle<Type>) -> Result<String, String> {
    let scalar = |s: naga::Scalar| match (s.kind, s.width) {
        (ScalarKind::Float, 4) => Ok("f32"),
        (ScalarKind::Float, 2) => Ok("f16"),
        (ScalarKind::Uint, 4) => Ok("u32"),
        (ScalarKind::Sint, 4) => Ok("i32"),
        (ScalarKind::Bool, 1) => Ok("bool"),
        _ => Err("unsupported graph scalar type".to_owned()),
    };
    Ok(match types[ty].inner {
        TypeInner::Scalar(s) => scalar(s)?.into(),
        TypeInner::Atomic(s) => format!("atomic<{}>", scalar(s)?),
        TypeInner::Vector { size, scalar: s } => format!("vec{}<{}>", size as u8, scalar(s)?),
        TypeInner::Matrix {
            columns,
            rows,
            scalar: s,
        } => format!("mat{}x{}<{}>", columns as u8, rows as u8, scalar(s)?),
        TypeInner::Array { base, size, .. } => {
            let suffix = match size {
                ArraySize::Constant(n) => format!(", {}", n.get()),
                ArraySize::Dynamic => String::new(),
                _ => return Err("unresolved graph array size".into()),
            };
            format!("array<{}{suffix}>", type_name(types, base)?)
        }
        TypeInner::Struct { .. } => types[ty].name.clone().ok_or("unnamed graph struct")?,
        _ => return Err("unsupported graph resource or scratch type".into()),
    })
}

fn uniform_value(
    types: &naga::UniqueArena<Type>,
    ty: Handle<Type>,
    base_word: usize,
    offset: u32,
) -> Result<String, String> {
    let scalar = |s: naga::Scalar, offset: u32| {
        let word = format!(
            "graph_parameter_values[{}u]",
            base_word + offset as usize / 4
        );
        match (s.kind, s.width) {
            (ScalarKind::Uint, 4) => Ok(word),
            (ScalarKind::Sint, 4) => Ok(format!("bitcast<i32>({word})")),
            (ScalarKind::Float, 4) => Ok(format!("bitcast<f32>({word})")),
            (ScalarKind::Float, 2) => Ok(format!(
                "f16(unpack2x16float({word})[{}u])",
                (offset % 4) / 2
            )),
            _ => Err("unsupported uniform scalar".to_owned()),
        }
    };
    let fields = match types[ty].inner {
        TypeInner::Scalar(s) => return scalar(s, offset),
        TypeInner::Vector { size, scalar: s } => (0..size as u32)
            .map(|i| scalar(s, offset + i * u32::from(s.width)))
            .collect::<Result<Vec<_>, _>>()?,
        TypeInner::Struct { ref members, .. } => members
            .iter()
            .map(|m| uniform_value(types, m.ty, base_word, offset + m.offset))
            .collect::<Result<Vec<_>, _>>()?,
        TypeInner::Array {
            base,
            size: ArraySize::Constant(n),
            stride,
        } => (0..n.get())
            .map(|i| uniform_value(types, base, base_word, offset + i * stride))
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err("unsupported graph uniform aggregate".into()),
    };
    Ok(format!("{}({})", type_name(types, ty)?, fields.join(", ")))
}

#[cfg(test)]
mod tests {
    use super::{Builder, tile};

    #[test]
    fn packing_rejects_query_dependent_barrier_loops() {
        let attention = crate::codegen::generate_flash_attention_coop_module(64);
        let body =
            tile::from_module(&attention.source, &attention.module.entry_points[0].name).unwrap();
        let mut builder = Builder::new(256);
        let kernel = builder
            .add(
                body,
                &crate::runtime::shader_data_layout(
                    &crate::compile::ShaderEntry::FlashAttentionCoop,
                ),
                0,
                &[],
                false,
            )
            .unwrap();
        assert_eq!(kernel.batch_axis, None);
        assert_eq!(kernel.batch_size, 1);
    }

    #[test]
    fn scalar_attention_packs_heads_with_uniform_barrier_loops() {
        for (part, source) in [
            (
                crate::kernels::attention_grad::Part::Q,
                include_str!("../shaders/mha_grad_q.wgsl"),
            ),
            (
                crate::kernels::attention_grad::Part::KV,
                include_str!("../shaders/mha_grad_kv.wgsl"),
            ),
        ] {
            let body = tile::from_module(source, "main").unwrap();
            let layout =
                crate::runtime::shader_data_layout(&crate::compile::ShaderEntry::AttentionGrad(
                    crate::kernels::attention_grad::AttentionGrad {
                        part,
                        path: crate::kernels::attention_grad::Path::Rowwise,
                    },
                ));
            let mut builder = Builder::new(256);
            let kernel = builder
                .add(body.clone(), &layout, 0, &[[4, 1, 1]], false)
                .unwrap();
            assert_eq!(kernel.batch_axis, None);
            let kernel = builder
                .add(body.clone(), &layout, 1, &[[7, 15, 1], [9, 6, 1]], false)
                .unwrap();
            assert_eq!(kernel.batch_axis, Some(1));
            assert_eq!(kernel.batch_size, 3);
            for (heads, packed) in [(1, 1), (2, 2), (3, 3), (4, 2), (5, 1), (6, 3), (15, 3)] {
                assert_eq!(kernel.tiles_per_claim([7, heads, 1]), packed);
            }
            let kernel = builder
                .add(
                    body,
                    &layout,
                    2,
                    &[[7, 15, 1], [9, 4, 1], [11, 2, 1], [13, 1, 1]],
                    false,
                )
                .unwrap();
            assert_eq!(kernel.batch_axis, Some(1));
            assert_eq!(kernel.batch_size, 4);
            for (heads, packed) in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 1), (6, 3), (15, 3)] {
                assert_eq!(kernel.tiles_per_claim([7, heads, 1]), packed);
            }
        }
    }

    #[test]
    fn mixed_geometry_graph_preserves_barrier_uniformity() {
        let mut builder = Builder::new(256);
        let mut kernels = Vec::new();
        for (source, entry, shader) in [
            (
                include_str!("../shaders/layer_norm.wgsl"),
                "main",
                crate::compile::ShaderEntry::LayerNorm,
            ),
            (
                include_str!("../shaders/global_avg_pool.wgsl"),
                "global_avg_pool",
                crate::compile::ShaderEntry::GlobalAvgPool,
            ),
        ] {
            let body = tile::from_module(source, entry).unwrap();
            let layout = crate::runtime::shader_data_layout(&shader);
            kernels.push(
                builder
                    .add(body, &layout, kernels.len(), &[], false)
                    .unwrap(),
            );
        }
        assert_eq!(builder.views.len(), 1);
        let cooperative = crate::codegen::generate_module_coop(
            crate::codegen::ShaderGroup::MatMul,
            &crate::codegen::CoopConfig {
                tile_size: 16,
                use_f16_input: true,
                compensated: false,
            },
        );
        let body = tile::from_module(
            &cooperative.source,
            &cooperative.module.entry_points[0].name,
        )
        .unwrap();
        kernels.push(
            builder
                .add(
                    body,
                    &crate::runtime::shader_data_layout(&crate::compile::ShaderEntry::MatMul),
                    kernels.len(),
                    &[],
                    false,
                )
                .unwrap(),
        );
        assert_eq!(kernels.last().unwrap().batch_size, 4);
        assert!(builder.views.len() > 1);
        let source = builder.source(&kernels, 6);
        assert!(!source.contains("graph_packing_sync_probe"));
        let mut module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        let mut binding_map = naga::back::spv::BindingMap::default();
        for (handle, global) in module.global_variables.iter_mut() {
            if matches!(
                global.space,
                naga::AddressSpace::Storage { .. } | naga::AddressSpace::Uniform
            ) {
                let binding = naga::ResourceBinding {
                    group: 0,
                    binding: handle.index() as u32,
                };
                global.binding = Some(binding);
                if builder
                    .views
                    .iter()
                    .any(|view| global.name.as_deref() == Some(&view.name))
                {
                    binding_map.insert(
                        binding,
                        naga::back::spv::BindingInfo {
                            descriptor_set: 0,
                            binding: 999,
                            binding_array_size: Some(6),
                        },
                    );
                }
            }
        }
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(module.entry_points.len(), 1);
        assert_eq!(module.entry_points[0].workgroup_size, [256, 1, 1]);
        let words = naga::back::spv::write_vec(
            &module,
            &info,
            &naga::back::spv::Options {
                lang_version: (1, 6),
                flags: naga::back::spv::WriterFlags::DEBUG,
                use_vulkan_memory_model: true,
                binding_map,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        let mut roots = std::collections::HashMap::new();
        let mut names = std::collections::HashMap::new();
        let mut loads = std::collections::HashMap::<String, usize>::new();
        let mut stores = 0;
        let mut aliases = 0;
        let mut cursor = 5;
        while cursor < words.len() {
            let count = (words[cursor] >> 16) as usize;
            let opcode = words[cursor] & 0xffff;
            let operands = &words[cursor + 1..cursor + count];
            match opcode {
                71 if operands[1] == 20 => aliases += 1, // OpDecorate Aliased
                5 => {
                    // OpName
                    let bytes = operands[1..]
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .take_while(|&b| b != 0)
                        .collect::<Vec<_>>();
                    names.insert(operands[0], String::from_utf8(bytes).unwrap());
                }
                65..=67 => {
                    // Pointer access chains
                    roots.insert(
                        operands[1],
                        *roots.get(&operands[2]).unwrap_or(&operands[2]),
                    );
                }
                61 | 62 => {
                    // OpLoad / OpStore
                    let pointer = operands[if opcode == 61 { 2 } else { 0 }];
                    let root = *roots.get(&pointer).unwrap_or(&pointer);
                    if let Some(view) = names
                        .get(&root)
                        .and_then(|name| builder.views.iter().find(|v| v.name == *name))
                    {
                        let access = operands[if opcode == 61 { 3 } else { 2 }];
                        assert_ne!(
                            access & 32,
                            0,
                            "every tensor access must be NonPrivatePointer"
                        );
                        assert_eq!(access & 24, 0, "typed aliases use the publication fences");
                        if opcode == 61 {
                            *loads.entry(view.name.clone()).or_default() += 1;
                        } else {
                            stores += 1;
                        }
                    }
                }
                _ => {}
            }
            cursor += count;
        }
        assert!(
            builder
                .views
                .iter()
                .all(|view| loads.contains_key(&view.name))
                && stores > 0
        );
        assert_eq!(aliases, builder.views.len());
    }
}
