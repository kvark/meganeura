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
    pub view: usize,
    pub write: bool,
}

pub(crate) struct Kernel {
    pub buffers: Vec<BufferBinding>,
    pub source: String,
    pub case: String,
    pub subgroups: bool,
    pub subgroup_threads: u32,
    pub workgroup_cooperative: bool,
}

pub(crate) struct View {
    pub name: String,
    ty: String,
    written: bool,
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
}

impl Builder {
    pub fn new(threads: u32) -> Self {
        Self {
            threads,
            parameter_words: 0,
            views: Vec::new(),
            scratch: Vec::new(),
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
    ) -> Result<Kernel, String> {
        let logical_threads = body.workgroup_size.into_iter().product::<u32>();
        let body = tile::mask_extra_lanes(body, self.threads)?;
        let body = tile::lift_uniforms(body)?;
        self.parameter_words = self.parameter_words.max(
            layout.bindings.len()
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
        let info = tile::validate(&module)?;
        let mut layouter = naga::proc::Layouter::default();
        layouter
            .update(module.to_ctx())
            .map_err(|e| e.to_string())?;
        let mut replacements = Vec::new();
        let mut names = HashMap::new();
        let mut buffers = Vec::new();
        let mut parameters = String::new();
        let mut shared_slots = HashMap::new();
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
                    let view = self
                        .views
                        .iter()
                        .position(|v| v.ty == ty)
                        .unwrap_or_else(|| {
                            let i = self.views.len();
                            self.views.push(View {
                                name: format!("graph_buffers_{i}"),
                                ty,
                                written: false,
                            });
                            i
                        });
                    self.views[view].written |= module.functions.iter().any(|(function, _)| {
                        info[function][handle].contains(naga::valid::GlobalUse::WRITE)
                    });
                    names.insert(
                        name.to_owned(),
                        format!(
                            "{}[{name}_binding_index_argument].data",
                            self.views[view].name
                        ),
                    );
                    buffers.push(BufferBinding {
                        slot,
                        view,
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
                    let base = layout.bindings.len()
                        + layout.bindings[..slot]
                            .iter()
                            .map(|b| match b.1 {
                                bg::ShaderBinding::Plain { size } => size.div_ceil(4) as usize,
                                _ => 0,
                            })
                            .sum::<usize>();
                    let value = if binding_index.is_some() {
                        format!("graph_parameter_values[{slot}u]")
                    } else {
                        uniform_value(&module.types, global.ty, base, 0)?
                    };
                    writeln!(parameters, "let {name}: {ty} = {value};").unwrap();
                    replacements.push((range, String::new()));
                }
                AddressSpace::WorkGroup => {
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
                        self.scratch[i].count = self.scratch[i].count.max(count);
                        i
                    } else {
                        let i = self.scratch.len();
                        self.scratch.push(Scratch {
                            ty,
                            slot: *slot,
                            count,
                            stride,
                        });
                        i
                    };
                    *slot += 1;
                    names.insert(
                        name.to_owned(),
                        format!("graph_scratch_{bank}{}", if array { "" } else { "[0]" }),
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
            }
            replacements.push((
                range.clone(),
                tile::rename_tokens(&source[range], &local_names),
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
        let active = if logical_threads == self.threads {
            String::new()
        } else {
            let predicate = if body.subgroups || body.cooperative.is_some() {
                format!("graph_subgroup_id < {logical_threads}u / graph_subgroup_size")
            } else {
                format!("physical_lane < {logical_threads}u")
            };
            format!("let {prefix}active_lane = {predicate};")
        };
        let case = include_str!("../shaders/graph_handler.wgsl")
            .replace("$INDEX", &index.to_string())
            .replace("$LANE", &lane)
            .replace("$X", &body.workgroup_size[0].to_string())
            .replace("$Y", &body.workgroup_size[1].to_string())
            .replace("$SUBGROUPS", &subgroups)
            .replace("$ACTIVE", &active)
            .replace("$PARAMETERS", &parameters)
            .replace("$CALL", &body.call.replace("$PREFIX", &prefix));
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
        })
    }

    pub fn source(&self, kernels: &[Kernel], view_counts: &[usize]) -> String {
        assert_eq!(view_counts.len(), self.views.len());
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
        for (i, (view, &count)) in self.views.iter().zip(view_counts).enumerate() {
            assert!(count > 0);
            // Written globals retain NonPrivatePointer on every access. Their
            // availability/visibility is supplied by the tile-boundary fences.
            // Read-only typed aliases still need the explicit decoration to
            // prevent Naga's immutable-global private-access optimization.
            let decoration = if view.written { "" } else { "@coherent " };
            writeln!(buffers, "struct GraphBuffer{i} {{ data: {} }}\n{decoration}var<storage, read_write> {}: binding_array<GraphBuffer{i}, {count}>;", view.ty, view.name).unwrap();
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
            kernels.push(builder.add(body, &layout, kernels.len()).unwrap());
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
                )
                .unwrap(),
        );
        assert!(builder.views.iter().any(|view| view.written));
        assert!(builder.views.iter().any(|view| !view.written));
        let source = builder.source(&kernels, &vec![6; builder.views.len()]);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all() - naga::valid::ValidationFlags::BINDINGS,
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
                ..Default::default()
            },
            None,
        )
        .unwrap();
        let mut roots = std::collections::HashMap::new();
        let mut names = std::collections::HashMap::new();
        let mut loads = [0; 2];
        let mut stores = 0;
        let mut cursor = 5;
        while cursor < words.len() {
            let count = (words[cursor] >> 16) as usize;
            let opcode = words[cursor] & 0xffff;
            let operands = &words[cursor + 1..cursor + count];
            match opcode {
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
                        assert_eq!(
                            access & 24 != 0,
                            !view.written,
                            "only immutable typed aliases need per-access propagation"
                        );
                        if opcode == 61 {
                            loads[usize::from(view.written)] += 1;
                        } else {
                            stores += 1;
                        }
                    }
                }
                _ => {}
            }
            cursor += count;
        }
        assert!(loads.iter().all(|&count| count > 0) && stores > 0);
    }
}
