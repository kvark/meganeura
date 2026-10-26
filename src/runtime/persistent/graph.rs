//! One dispatch owns the graph, including the optimizer commands of this step.
use super::{
    AppendedDispatch, Gpu, PersistentOptions, Session, SharedPipeline, SharedPipelineKey,
    SharedPipelines, tile,
};
use crate::{
    codegen,
    compile::{Dispatch, ExecutionPlan, Kernel, ShaderEntry, WeightFormat},
};
use blade_graphics as bg;
use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    ops::Range,
    sync::{Arc, LazyLock, Mutex},
};

struct Access {
    piece: bg::BufferPiece,
    write: bool,
    range: Range<u64>,
}

fn output_bytes(dispatch: &Dispatch, name: &str) -> Option<u64> {
    if dispatch.pointwise_storage.bf16_output
        || dispatch.pointwise_storage.bf16_inputs != 0
        || dispatch.matmul_epilogue.is_some()
        || dispatch.matmul_prologue.is_some()
    {
        return None;
    }
    match dispatch.shader {
        ShaderEntry::MatMulAT
            if matches!(
                dispatch.kernel,
                Kernel::Default | Kernel::SmallTile | Kernel::ScalarMatmul(_)
            ) && dispatch.weight_format == WeightFormat::F32
                && (name == "matrix_c"
                    || name
                        .strip_prefix("matrix_c")
                        .and_then(|s| s.parse::<u32>().ok())
                        .is_some_and(|i| i < dispatch.horizontal_batch)) =>
        {
            // The scalar template guards row < M and col < N, stores
            // C[row*N + col], and does not read C. Horizontal outputs each
            // use this same extent; split-K and cooperative kernels stay out.
            let (m, n, _) = dispatch.mnk()?;
            u64::from(m).checked_mul(u64::from(n))?.checked_mul(4)
        }
        ShaderEntry::SumRows if dispatch.kernel == Kernel::Default && name == "dst" => {
            // Both layouts guard col < N. Parallel rows additionally store
            // at workgroup_id.y*N; serial rows store at col only.
            let n = *dispatch.params.get(1)?;
            let splits = if dispatch.params.get(2).copied().unwrap_or(0) == 0 {
                dispatch.workgroups[1]
            } else {
                1
            };
            u64::from(n).checked_mul(u64::from(splits))?.checked_mul(4)
        }
        _ => None,
    }
    .filter(|&bytes| bytes != 0)
}

#[derive(Default)]
pub(in crate::runtime) struct Bindings {
    buffers: Vec<Option<bg::BufferPiece>>,
    plain: Vec<Option<Vec<u8>>>,
}

impl bg::ShaderDataVisitor for Bindings {
    fn buffer(&mut self, index: u32, piece: bg::BufferPiece) {
        self.buffers
            .resize(self.buffers.len().max(index as usize + 1), None);
        self.buffers[index as usize] = Some(piece);
    }
    fn plain(&mut self, index: u32, bytes: &[u8]) {
        self.plain
            .resize(self.plain.len().max(index as usize + 1), None);
        self.plain[index as usize] = Some(bytes.to_vec());
    }
}

impl bg::traits::PipelineEncoder for Bindings {
    fn bind<D: bg::ShaderData>(&mut self, _group: u32, data: &D) {
        data.visit(self);
    }
}

pub(in crate::runtime) struct Command {
    entry: ShaderEntry,
    grid: [u32; 3],
    label: &'static str,
    bindings: Bindings,
}

impl Command {
    pub fn new<D: bg::ShaderData>(
        entry: ShaderEntry,
        label: &'static str,
        grid: [u32; 3],
        data: &D,
    ) -> Self {
        let mut bindings = Bindings::default();
        data.visit(&mut bindings);
        Self {
            entry,
            grid,
            label,
            bindings,
        }
    }
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Config {
    nodes: u32,
    ready_words: u32,
    counters: u32,
    worker_counts: u32,
    workers: u32,
    track_workers: u32,
    padding: [u32; 2],
}

struct Allocation {
    gpu: Arc<Gpu>,
    buffer: bg::Buffer,
}

impl Allocation {
    fn new(gpu: &Arc<Gpu>, name: &str, words: &[u32], device: bool) -> Self {
        let size = (words.len().max(4) * 4) as u64;
        let buffer = gpu.create_buffer(bg::BufferDesc {
            name,
            size,
            memory: if device {
                bg::Memory::Device
            } else {
                bg::Memory::Upload
            },
        });
        if !device {
            unsafe {
                std::ptr::write_bytes(buffer.data(), 0, size as usize);
                std::ptr::copy_nonoverlapping(words.as_ptr(), buffer.data().cast(), words.len());
            }
            gpu.sync_buffer(buffer.into(), size, bg::BufferTarget::Data);
        }
        Self {
            gpu: Arc::clone(gpu),
            buffer,
        }
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        self.gpu.destroy_buffer(self.buffer);
    }
}

struct Frame {
    nodes: Allocation,
    edges: Allocation,
    parameters: Allocation,
    initial_queue: Allocation,
    queue: Allocation,
    buffers: Vec<bg::BufferPiece>,
    config: Config,
    tasks: Vec<u32>,
    graph_nodes: usize,
    pipeline: Arc<SharedPipeline>,
    key: String,
    bindings: Vec<(usize, [u32; 3], Vec<Option<bg::BufferPiece>>)>,
    parameter_words: Vec<u32>,
    kernel_count: usize,
    epoch: u64,
}

impl bg::ShaderData for Frame {
    fn layout() -> bg::ShaderDataLayout {
        bg::ShaderDataLayout::default()
    }
    fn fill(&self, mut context: bg::PipelineContext) {
        use bg::ShaderBindable;
        self.nodes.buffer.at(0).bind_to(&mut context, 0);
        self.edges.buffer.at(0).bind_to(&mut context, 1);
        self.parameters.buffer.at(0).bind_to(&mut context, 2);
        self.queue.buffer.at(0).bind_to(&mut context, 3);
        self.config.bind_to(&mut context, 4);
        context.bind_buffer_array(5, &self.buffers);
    }
}

pub(in crate::runtime) struct Graph {
    gpu: Arc<Gpu>,
    options: PersistentOptions,
    builder: codegen::graph::Builder,
    kernels: Vec<codegen::graph::Kernel>,
    layouts: Vec<bg::ShaderDataLayout>,
    handlers: Vec<usize>,
    optimizer_handlers: HashMap<ShaderEntry, usize>,
    dump_dir: Option<String>,
    // Each external recording keeps its own immutable parameters and queue.
    // Session::wait retires earlier frames only after their submission completes.
    current: Option<Frame>,
    retired: Vec<Frame>,
    idle: bool,
    epoch: u64,
    submitted_epoch: u64,
}

fn binding_name(name: &str) -> &'static str {
    static NAMES: LazyLock<Mutex<HashMap<String, &'static str>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    let mut names = NAMES.lock().unwrap();
    names
        .entry(name.to_owned())
        .or_insert_with(|| Box::leak(name.to_owned().into_boxed_str()))
}

impl Graph {
    pub fn new(
        gpu: &Arc<Gpu>,
        plan: &ExecutionPlan,
        options: PersistentOptions,
        dump_dir: Option<&str>,
        coop: Option<&codegen::CoopConfig>,
    ) -> Result<Self, String> {
        let caps = gpu.capabilities();
        if !caps.device_memory_fence || !caps.binding_array {
            return Err(
                "whole-graph execution requires native device fences and buffer binding arrays"
                    .into(),
            );
        }
        let mut selected = Vec::new();
        let mut unique = HashMap::new();
        let mut handlers = Vec::new();
        let mut threads = 256;
        for dispatch in &plan.dispatches {
            let handler = tile(gpu, plan, dispatch, coop)?;
            threads = threads.max(handler.body.workgroup_size.into_iter().product());
            let signature = format!(
                "{}\n{}\n{:?}",
                handler.body.source, handler.body.call, handler.layout
            );
            let index = *unique.entry(signature).or_insert_with(|| {
                let index = selected.len();
                selected.push(handler);
                index
            });
            handlers.push(index);
        }
        if threads > 1024 {
            return Err("whole-graph worker exceeds 1024 invocations".into());
        }
        let (builder, kernels, layouts) = [4, 2, 1]
            .into_iter()
            .find_map(|batch_limit| {
                let mut builder = codegen::graph::Builder::new(threads);
                builder.batch_limit = batch_limit;
                let mut kernels = Vec::new();
                let mut layouts = Vec::new();
                for (index, handler) in selected.iter().enumerate() {
                    let grids = plan
                        .dispatches
                        .iter()
                        .zip(&handlers)
                        .filter_map(|(dispatch, &handler_index)| {
                            (handler_index == index).then_some(dispatch.workgroups)
                        })
                        .collect::<Vec<_>>();
                    match builder.add(
                        handler.body.clone(),
                        &handler.layout,
                        kernels.len(),
                        &grids,
                        false,
                    ) {
                        Ok(kernel) => kernels.push(kernel),
                        Err(error) => return Some(Err(error)),
                    }
                    layouts.push(handler.layout.clone());
                }
                let reserved = if kernels.iter().any(|kernel| kernel.workgroup_cooperative) {
                    u64::from(caps.cooperative_matrix.workgroup.reserved_shared_memory)
                } else {
                    0
                };
                if builder.shared_bytes() + reserved
                    <= u64::from(caps.max_compute_shared_memory_size)
                    || batch_limit == 1
                {
                    Some(Ok((builder, kernels, layouts)))
                } else {
                    None
                }
            })
            .unwrap()?;
        Ok(Self {
            gpu: Arc::clone(gpu),
            options,
            builder,
            kernels,
            layouts,
            handlers,
            optimizer_handlers: HashMap::new(),
            dump_dir: dump_dir.map(str::to_owned),
            current: None,
            retired: Vec::new(),
            idle: true,
            epoch: 0,
            submitted_epoch: 0,
        })
    }

    pub fn threads(&self) -> u32 {
        self.builder.threads
    }
    pub fn has_graph(&self) -> bool {
        self.current
            .as_ref()
            .is_none_or(|frame| frame.graph_nodes != 0)
    }
    pub fn pipeline(&self) -> Option<&Arc<SharedPipeline>> {
        self.current.as_ref().map(|f| &f.pipeline)
    }
    pub fn pipeline_key(&self) -> Option<&str> {
        self.current.as_ref().map(|f| f.key.as_str())
    }
    pub fn queue(&self) -> Option<bg::Buffer> {
        self.current.as_ref().map(|f| f.queue.buffer)
    }
    pub fn idle(&mut self) {
        self.retired
            .retain(|frame| frame.epoch > self.submitted_epoch);
        self.idle = self
            .current
            .as_ref()
            .is_none_or(|frame| frame.epoch <= self.submitted_epoch);
    }
    pub fn submitted(&mut self) {
        self.submitted_epoch = self.epoch;
    }

    pub fn prepare(
        &mut self,
        plan: &ExecutionPlan,
        buffers: &[bg::BufferPiece],
        commands: Vec<Command>,
        include_graph: bool,
    ) -> Result<Vec<AppendedDispatch>, String> {
        self.epoch = self
            .epoch
            .checked_add(1)
            .expect("graph recording epoch overflow");
        let mut work = Vec::new();
        if include_graph {
            for (index, dispatch) in plan.dispatches.iter().enumerate() {
                let mut bindings = Bindings::default();
                Session::bind_dispatch_at(buffers, dispatch, &mut bindings, 0);
                work.push((self.handlers[index], dispatch.workgroups, bindings));
            }
        }
        let graph_nodes = work.len();
        let mut records = Vec::new();
        for command in commands {
            let index = if let Some(&index) = self.optimizer_handlers.get(&command.entry) {
                index
            } else {
                let dispatch = Dispatch {
                    shader: command.entry.clone(),
                    ..Default::default()
                };
                let handler = tile(&self.gpu, plan, &dispatch, None)?;
                let index = self.kernels.len();
                self.kernels.push(self.builder.add(
                    handler.body,
                    &handler.layout,
                    index,
                    &[command.grid],
                    command.entry == ShaderEntry::SgdUpdate,
                )?);
                self.layouts.push(handler.layout);
                self.optimizer_handlers.insert(command.entry.clone(), index);
                index
            };
            let mut record =
                AppendedDispatch::ordinary(&command.entry, command.label, command.grid);
            record.workers = self.options.workers;
            record.device_scheduled = true;
            record.workgroup_shape = Some([self.threads(), 1, 1]);
            records.push(record);
            work.push((index, command.grid, command.bindings));
        }
        if work.is_empty() {
            if let Some(old) = self.current.take() {
                if !self.idle {
                    self.retired.push(old);
                }
            }
            return Ok(records);
        }
        // A completed frame with unchanged bindings has the same physical
        // dependencies and descriptors. Only values such as Adam's step vary.
        if self.idle {
            if let Some(ref mut frame) = self.current {
                let compatible = frame.kernel_count == self.kernels.len()
                    && frame.graph_nodes == graph_nodes
                    && frame.bindings.len() == work.len()
                    && frame.bindings.iter().zip(&work).all(
                        |(&(handler, grid, ref buffers), &(next_handler, next_grid, ref next))| {
                            handler == next_handler
                                && grid == next_grid
                                && buffers.len() == next.buffers.len()
                                && buffers
                                    .iter()
                                    .zip(&next.buffers)
                                    .all(|(a, b)| match (*a, *b) {
                                        (None, None) => true,
                                        (Some(a), Some(b)) => {
                                            a.buffer == b.buffer && a.offset == b.offset
                                        }
                                        _ => false,
                                    })
                        },
                    );
                if compatible {
                    for (node, work_item) in work.iter().enumerate() {
                        let bindings = &work_item.2;
                        let base = node * self.builder.parameter_words.max(4);
                        for (slot, bytes) in bindings.plain.iter().enumerate() {
                            if let Some(ref bytes) = *bytes {
                                let offset = base + frame.parameter_words[base + 2 * slot] as usize;
                                let target = bytemuck::cast_slice_mut::<u32, u8>(
                                    &mut frame.parameter_words[offset..],
                                );
                                target[..bytes.len()].copy_from_slice(bytes);
                            }
                        }
                    }
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            frame.parameter_words.as_ptr(),
                            frame.parameters.buffer.data().cast(),
                            frame.parameter_words.len(),
                        );
                    }
                    self.gpu.sync_buffer(
                        frame.parameters.buffer.into(),
                        frame.parameters.buffer.size(),
                        bg::BufferTarget::Data,
                    );
                    for record in &mut records {
                        record.pipeline_key = frame.key.clone();
                    }
                    self.idle = false;
                    frame.epoch = self.epoch;
                    return Ok(records);
                }
            }
        }
        let bindings = work
            .iter()
            .map(|&(handler, grid, ref bindings)| (handler, grid, bindings.buffers.clone()))
            .collect();
        let nodes = u32::try_from(work.len()).map_err(|_| "too many graph nodes")?;
        let ready_words = nodes.div_ceil(32);
        let counters = 2 + ready_words;
        let worker_counts = counters
            .checked_add(nodes.checked_mul(3).ok_or("queue size overflow")?)
            .ok_or("queue size overflow")?;
        let queue_words = worker_counts
            .checked_add(if self.options.track_workers {
                nodes
                    .checked_mul(self.options.workers)
                    .ok_or("worker counter overflow")?
            } else {
                0
            })
            .ok_or("queue size overflow")?;
        let config = Config {
            nodes,
            ready_words,
            counters,
            worker_counts,
            workers: self.options.workers,
            track_workers: u32::from(self.options.track_workers),
            padding: [0; 2],
        };
        let mut queue = vec![0u32; queue_words as usize];
        queue[0] = nodes;
        let mut parameters = Vec::new();
        let mut metadata = Vec::new();
        let mut buffers: Vec<bg::BufferPiece> = Vec::new();
        let mut accesses: Vec<Vec<Access>> = Vec::new();
        let mut ancestors: Vec<Vec<u64>> = Vec::new();
        let mut successors = vec![Vec::new(); nodes as usize];
        let mut tasks = Vec::new();
        for (node, (handler, grid, bindings)) in work.into_iter().enumerate() {
            let count = grid
                .into_iter()
                .try_fold(1u32, |a, b| a.checked_mul(b))
                .ok_or("tile count overflow")?;
            if handler > u16::MAX as usize {
                return Err("too many whole-graph handlers".into());
            }
            let kernel = &self.kernels[handler];
            let batch = kernel.tiles_per_claim(grid);
            if u64::from(count) + u64::from(batch) * u64::from(self.options.workers) + 1
                > u64::from(u32::MAX)
            {
                return Err("tile claims may overflow".into());
            }
            tasks.push(count);
            let base = u32::try_from(parameters.len()).map_err(|_| "parameter offset overflow")?;
            let layout = &self.layouts[handler];
            let mut words = vec![0; 2 * layout.bindings.len()];
            for (slot, bytes) in bindings.plain.iter().enumerate() {
                if let Some(ref bytes) = *bytes {
                    words[2 * slot] = words.len() as u32;
                    for chunk in bytes.chunks(4) {
                        let mut word = [0u8; 4];
                        word[..chunk.len()].copy_from_slice(chunk);
                        words.push(u32::from_ne_bytes(word));
                    }
                }
            }
            let mut node_accesses = Vec::new();
            for binding in &self.kernels[handler].buffers {
                let piece = bindings.buffers[binding.slot].ok_or("missing node buffer")?;
                if piece.offset >= piece.buffer.size()
                    || !piece.offset.is_multiple_of(u64::from(binding.stride))
                {
                    return Err(
                        "graph buffer offset is out of range or not a whole storage element".into(),
                    );
                }
                let index = buffers
                    .iter()
                    .position(|p| p.buffer == piece.buffer)
                    .unwrap_or_else(|| {
                        let index = buffers.len();
                        buffers.push(piece.buffer.at(0));
                        index
                    });
                words[2 * binding.slot] = index as u32;
                words[2 * binding.slot + 1] =
                    u32::try_from(piece.offset / u64::from(binding.stride))
                        .map_err(|_| "graph buffer element offset exceeds u32")?;
                let range = (node < graph_nodes && binding.write && binding.stride == 4)
                    .then(|| output_bytes(&plan.dispatches[node], layout.bindings[binding.slot].0))
                    .flatten()
                    .and_then(|bytes| piece.offset.checked_add(bytes))
                    .filter(|&end| end <= piece.buffer.size())
                    .map_or(0..piece.buffer.size(), |end| piece.offset..end);
                node_accesses.push(Access {
                    piece,
                    write: binding.write,
                    range,
                });
            }
            if words.len() > self.builder.parameter_words {
                return Err("node uniforms exceed shared parameter capacity".into());
            }
            words.resize(self.builder.parameter_words.max(4), 0);
            parameters.extend(words);
            let mut pending = 0;
            let mut ordered = vec![0u64; (nodes as usize).div_ceil(64)];
            for (prior, prior_accesses) in accesses.iter().enumerate().rev() {
                if ordered[prior / 64] & (1u64 << (prior % 64)) != 0 {
                    continue;
                }
                // Unknown accesses cover the whole allocation. Proven output
                // ranges can be disjoint; reuse and overlapping aliases still
                // preserve every physical RAW/WAR/WAW relation.
                if prior_accesses.iter().any(|a| {
                    node_accesses.iter().any(|b| {
                        a.piece.buffer == b.piece.buffer
                            && (a.write || b.write)
                            && a.range.start < b.range.end
                            && b.range.start < a.range.end
                    })
                }) {
                    successors[prior].push(node as u32);
                    pending += 1;
                    // An edge from a later predecessor already orders all of its
                    // ancestors. Keep the same happens-before relation without
                    // making the completing worker publish redundant edges.
                    ordered[prior / 64] |= 1u64 << (prior % 64);
                    for (word, &inherited) in ordered.iter_mut().zip(&ancestors[prior]) {
                        *word |= inherited;
                    }
                }
            }
            ancestors.push(ordered);
            accesses.push(node_accesses);
            let counter = counters as usize + 3 * node;
            queue[counter + 1] = count.max(1);
            queue[counter + 2] = pending;
            if pending == 0 {
                queue[2 + node / 32] |= 1 << (node % 32);
            }
            metadata.extend([
                grid[0],
                grid[1],
                grid[2],
                handler as u32 | (batch << 16),
                base,
                count,
                0,
                0,
            ]);
        }
        log::debug!(
            "packed graph: shared={} nodes={} packed_nodes={} packed_tiles={}",
            self.builder.shared_bytes(),
            nodes,
            metadata
                .as_chunks::<8>()
                .0
                .iter()
                .filter(|node| node[3] >> 16 > 1)
                .count(),
            metadata
                .as_chunks::<8>()
                .0
                .iter()
                .filter(|node| node[3] >> 16 > 1)
                .map(|node| u64::from(node[5]))
                .sum::<u64>()
        );
        let mut edges = Vec::new();
        if self.options.track_workers
            && tasks.iter().map(|&count| u64::from(count)).sum::<u64>() > u64::from(u32::MAX)
        {
            return Err("aggregate worker diagnostics may overflow".into());
        }
        for (node, next) in successors.into_iter().enumerate() {
            metadata[node * 8 + 6] =
                u32::try_from(edges.len()).map_err(|_| "dependency offset overflow")?;
            metadata[node * 8 + 7] = next.len() as u32;
            edges.extend(next);
        }
        if buffers.is_empty() {
            return Err("graph has no buffers".into());
        }
        let caps = self.gpu.capabilities();
        let limits = caps
            .compute_binding_limits
            .ok_or("device has no binding limits")?;
        let storage = 4 + buffers.len();
        if storage > limits.max_storage_buffers as usize
            || storage + 1 > limits.max_buffer_bindings as usize
            || self.builder.views.len() + 5 > bg::limits::RESOURCES_IN_GROUP as usize
        {
            return Err(format!(
                "whole graph needs {storage} storage descriptors; device limits {limits:?}"
            ));
        }
        let shared = self.builder.shared_bytes()
            + if self.kernels.iter().any(|k| k.workgroup_cooperative) {
                u64::from(caps.cooperative_matrix.workgroup.reserved_shared_memory)
            } else {
                0
            };
        if shared > u64::from(caps.max_compute_shared_memory_size) {
            return Err(format!(
                "whole graph needs {shared} shared bytes; device limit {}",
                caps.max_compute_shared_memory_size
            ));
        }
        for words in [&metadata, &edges, &parameters, &queue] {
            if words.len().max(4) as u64 * 4 > limits.max_storage_buffer_range {
                return Err("graph metadata exceeds storage range".into());
            }
        }
        for piece in &buffers {
            if piece.buffer.size() - piece.offset > limits.max_storage_buffer_range {
                return Err("graph buffer exceeds storage range".into());
            }
        }
        let mut layout = bg::ShaderDataLayout {
            bindings: vec![
                ("graph_nodes", bg::ShaderBinding::Buffer),
                ("graph_edges", bg::ShaderBinding::Buffer),
                ("graph_parameters", bg::ShaderBinding::Buffer),
                ("graph_queue", bg::ShaderBinding::Buffer),
                (
                    "graph_config",
                    bg::ShaderBinding::Plain {
                        size: std::mem::size_of::<Config>() as u32,
                    },
                ),
            ],
        };
        for (index, view) in self.builder.views.iter().enumerate() {
            layout.bindings.push((
                binding_name(&view.name),
                if index == 0 {
                    bg::ShaderBinding::BufferArray {
                        count: buffers.len() as u32,
                    }
                } else {
                    bg::ShaderBinding::BufferAlias { binding: 5 }
                },
            ));
        }
        let source = self.builder.source(&self.kernels, buffers.len());
        let key = SharedPipelineKey {
            context: Arc::as_ptr(&self.gpu) as usize,
            entry_point: "graph_main",
            layout: format!("{layout:?}"),
            source,
        };
        let mut hasher = DefaultHasher::new();
        key.source.hash(&mut hasher);
        key.layout.hash(&mut hasher);
        let pipeline_key = format!("whole-graph:{:016x}", hasher.finish());
        let pipeline = if let Some(pipeline) = SharedPipelines::get(&key) {
            pipeline
        } else {
            let module = codegen::ShaderModule::with_hint(key.source.clone(), "whole_graph");
            if let Some(ref dir) = self.dump_dir {
                module.dump(dir);
            }
            let shader = super::super::create_gen_shader(&self.gpu, module)?;
            let pipeline = Arc::new(SharedPipeline {
                gpu: Arc::clone(&self.gpu),
                raw: self.gpu.create_compute_pipeline(bg::ComputePipelineDesc {
                    name: "whole graph",
                    data_layouts: &[&layout],
                    compute: shader.at("graph_main"),
                }),
            });
            SharedPipelines::insert(key, &pipeline);
            pipeline
        };
        let bytes = (metadata.len() + edges.len() + parameters.len() + 2 * queue.len()) * 4;
        super::super::ensure_device_memory_budget(&self.gpu, bytes, "whole-graph scheduler");
        let frame = Frame {
            nodes: Allocation::new(&self.gpu, "graph nodes", &metadata, false),
            edges: Allocation::new(&self.gpu, "graph edges", &edges, false),
            parameters: Allocation::new(&self.gpu, "graph parameters", &parameters, false),
            initial_queue: Allocation::new(&self.gpu, "graph reset", &queue, false),
            queue: Allocation::new(&self.gpu, "graph queue", &queue, true),
            buffers,
            config,
            tasks,
            graph_nodes,
            pipeline,
            key: pipeline_key.clone(),
            bindings,
            parameter_words: parameters,
            kernel_count: self.kernels.len(),
            epoch: self.epoch,
        };
        if let Some(old) = self.current.replace(frame) {
            if !self.idle {
                self.retired.push(old);
            }
        }
        self.idle = false;
        for record in &mut records {
            record.pipeline_key = pipeline_key.clone();
        }
        Ok(records)
    }

    pub fn record(&self, encoder: &mut bg::CommandEncoder) {
        if let Some(ref frame) = self.current {
            encoder
                .transfer("reset whole-graph queue")
                .copy_buffer_to_buffer(
                    frame.initial_queue.buffer.into(),
                    frame.queue.buffer.into(),
                    frame.queue.buffer.size(),
                );
            let mut pass = encoder.compute("whole graph");
            let mut pc = pass.with(&frame.pipeline.raw);
            pc.bind(0, frame);
            pc.dispatch([self.options.workers, 1, 1]);
        }
    }

    pub fn bytes(&self) -> usize {
        self.current
            .iter()
            .chain(&self.retired)
            .map(|f| {
                [
                    &f.nodes,
                    &f.edges,
                    &f.parameters,
                    &f.initial_queue,
                    &f.queue,
                ]
                .iter()
                .map(|a| a.buffer.size() as usize)
                .sum::<usize>()
            })
            .sum()
    }

    pub fn check_completion(&self, words: &[u32]) -> Result<(), String> {
        let frame = self.current.as_ref().ok_or("graph has not run")?;
        if words[..frame.config.counters as usize]
            .iter()
            .any(|&w| w != 0)
        {
            return Err(format!(
                "whole graph incomplete: remaining={}, diagnostics={}",
                words[0], words[1]
            ));
        }
        for (node, &tasks) in frame.tasks.iter().enumerate() {
            let counter = frame.config.counters as usize + 3 * node;
            if words[counter] < tasks.max(1) || words[counter + 1] != 0 || words[counter + 2] != 0 {
                return Err(format!(
                    "whole-graph node {node} incomplete: {:?}",
                    &words[counter..counter + 3]
                ));
            }
            if self.options.track_workers {
                let row = self.worker_row(words, node);
                if row.iter().map(|&v| u64::from(v)).sum::<u64>() != u64::from(tasks) {
                    return Err(format!("whole-graph node {node} task count mismatch"));
                }
            }
        }
        Ok(())
    }

    fn worker_row<'a>(&self, words: &'a [u32], node: usize) -> &'a [u32] {
        let frame = self.current.as_ref().unwrap();
        let start = frame.config.worker_counts as usize + node * self.options.workers as usize;
        &words[start..start + self.options.workers as usize]
    }

    pub fn worker_tasks(&self, words: &[u32], appended: bool) -> Vec<Vec<u32>> {
        let frame = self.current.as_ref().unwrap();
        if appended {
            (frame.graph_nodes..frame.tasks.len())
                .map(|n| self.worker_row(words, n).to_vec())
                .collect()
        } else if frame.graph_nodes == 0 {
            Vec::new()
        } else {
            let mut row = vec![0; self.options.workers as usize];
            for node in 0..frame.graph_nodes {
                for (sum, count) in row.iter_mut().zip(self.worker_row(words, node)) {
                    *sum += count;
                }
            }
            vec![row]
        }
    }
}
