use blade_graphics as bg;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Node {
    extent: u32,
    data_offset: u32,
    first_task: u32,
    task_count: u32,
    first_pred: u32,
    pred_count: u32,
    first_succ: u32,
    succ_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    node_count: u32,
    task_count: u32,
    word_count: u32,
    seed: u32,
    bitmap_offset: u32,
    pending_offset: u32,
    tiles_offset: u32,
    counts_offset: u32,
    workers_offset: u32,
    worker_count: u32,
    padding: [u32; 2],
}

#[derive(blade_macros::ShaderData)]
struct Bindings {
    payload: bg::BufferPiece,
    nodes: bg::BufferPiece,
    edges: bg::BufferPiece,
    tasks: bg::BufferPiece,
    state: bg::BufferPiece,
    params: Params,
}

fn check_graph(gpu: &bg::Context, pipeline: &bg::ComputePipeline, graph: &[(u32, Vec<u32>)]) {
    let mut nodes = Vec::new();
    let mut tasks = Vec::new();
    let mut edges = Vec::new();
    let mut data_len = 0;
    for (index, &(extent, ref predecessors)) in graph.iter().enumerate() {
        assert!(extent != 0 && predecessors.iter().all(|&p| p < index as u32));
        let first_pred = edges.len() as u32;
        edges.extend_from_slice(predecessors);
        let first_succ = edges.len() as u32;
        for (successor, candidate) in graph.iter().enumerate() {
            if candidate.1.contains(&(index as u32)) {
                edges.push(successor as u32);
            }
        }
        let node = Node {
            extent,
            data_offset: data_len,
            first_task: tasks.len() as u32,
            task_count: extent.div_ceil(64),
            first_pred,
            pred_count: predecessors.len() as u32,
            first_succ,
            succ_count: edges.len() as u32 - first_succ,
        };
        tasks.extend((0..node.task_count).map(|tile| [index as u32, tile]));
        data_len += extent;
        nodes.push(node);
    }
    let mut buffers = Vec::new();
    let mut allocate = |name, size, memory| {
        let buffer = gpu.create_buffer(bg::BufferDesc { name, size, memory });
        buffers.push(buffer);
        buffer
    };
    let payload_bytes = u64::from(data_len.max(1).next_multiple_of(4)) * 4;
    let payload = allocate("protocol payload", payload_bytes, bg::Memory::Device);
    let mut metadata = Vec::new();
    for bytes in [
        bytemuck::cast_slice(&nodes),
        bytemuck::cast_slice(&edges),
        bytemuck::cast_slice(&tasks),
    ] {
        let buffer = allocate(
            "protocol metadata",
            bytes.len().max(32) as u64,
            bg::Memory::Upload,
        );
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.data(), bytes.len());
        }
        gpu.sync_buffer(
            buffer.into(),
            bytes.len().max(32) as u64,
            bg::BufferTarget::Data,
        );
        metadata.push(buffer);
    }
    let word_count = (tasks.len() as u32).div_ceil(32).max(1);
    let pending_offset = 4 + word_count;
    let tiles_offset = pending_offset + nodes.len() as u32;
    let counts_offset = tiles_offset + nodes.len() as u32;
    let workers_offset = counts_offset + tasks.len() as u32;
    let state_bytes = u64::from(workers_offset + 4096) * 4;
    let state = allocate("protocol state", state_bytes, bg::Memory::Device);
    let reset = allocate("protocol reset", state_bytes, bg::Memory::Upload);
    let download = allocate(
        "protocol readback",
        payload_bytes + state_bytes,
        bg::Memory::Download,
    );
    let mut initial = vec![0u32; state_bytes as usize / 4];
    initial[0] = tasks.len() as u32;
    for (index, node) in nodes.iter().enumerate() {
        initial[pending_offset as usize + index] = node.pred_count;
        initial[tiles_offset as usize + index] = node.task_count;
        if node.pred_count == 0 {
            for task in node.first_task..node.first_task + node.task_count {
                initial[4 + task as usize / 32] |= 1 << (task % 32);
            }
        }
    }
    unsafe {
        std::ptr::copy_nonoverlapping(initial.as_ptr(), reset.data().cast(), initial.len());
    }
    gpu.sync_buffer(reset.into(), state_bytes, bg::BufferTarget::Data);
    let mut encoder = gpu.create_command_encoder(bg::CommandEncoderDesc {
        name: "bounded device protocol",
        buffer_count: 1,
        manual_barriers: false,
    });
    for workers in [1, 3, 64, 4096] {
        for epoch in 0..16 {
            let seed = 71 + epoch * 97;
            let mut expected: Vec<u32> = Vec::new();
            for (index, node) in nodes.iter().enumerate() {
                for element in 0..node.extent {
                    let mut value = seed + (index as u32 + 1) * 1009 + element;
                    for &predecessor in &graph[index].1 {
                        let input = nodes[predecessor as usize];
                        let source = (element * 17 + index as u32 * 13 + seed) % input.extent;
                        value = value.wrapping_add(expected[(input.data_offset + source) as usize]);
                    }
                    for _ in 0..(index % 5) * 3 {
                        value = value.wrapping_mul(1664525).wrapping_add(1013904223);
                    }
                    expected.push(value);
                }
            }
            let bindings = Bindings {
                payload: payload.into(),
                nodes: metadata[0].into(),
                edges: metadata[1].into(),
                tasks: metadata[2].into(),
                state: state.into(),
                params: Params {
                    node_count: nodes.len() as u32,
                    task_count: tasks.len() as u32,
                    word_count,
                    seed,
                    bitmap_offset: 4,
                    pending_offset,
                    tiles_offset,
                    counts_offset,
                    workers_offset,
                    worker_count: workers,
                    padding: [0; 2],
                },
            };
            encoder.start();
            {
                let mut transfer = encoder.transfer("reset protocol epoch");
                transfer.fill_buffer(payload.into(), payload_bytes, 0xcd);
                transfer.copy_buffer_to_buffer(reset.into(), state.into(), state_bytes);
            }
            {
                let mut pass = encoder.compute("device graph protocol");
                let mut dispatch = pass.with(pipeline);
                dispatch.bind(0, &bindings);
                dispatch.dispatch([workers, 1, 1]);
            }
            {
                let mut transfer = encoder.transfer("read protocol results");
                transfer.copy_buffer_to_buffer(payload.into(), download.into(), payload_bytes);
                transfer.copy_buffer_to_buffer(
                    state.into(),
                    download.at(payload_bytes),
                    state_bytes,
                );
            }
            let sync = gpu.submit(&mut encoder);
            assert!(
                gpu.wait_for(&sync, 30_000).unwrap(),
                "bounded protocol timed out"
            );
            let actual = unsafe {
                std::slice::from_raw_parts(download.data().cast::<u32>(), expected.len())
            };
            assert_eq!(actual, expected, "workers={workers}, epoch={epoch}");
            let state = unsafe {
                std::slice::from_raw_parts(
                    download.data().add(payload_bytes as usize).cast::<u32>(),
                    initial.len(),
                )
            };
            assert_eq!(
                state[0], 0,
                "unfinished tasks, workers={workers}, epoch={epoch}"
            );
            assert_eq!(state[1], 0, "protocol diagnostic failure");
            assert!(state[4..counts_offset as usize].iter().all(|&v| v == 0));
            assert!(
                state[counts_offset as usize..workers_offset as usize]
                    .iter()
                    .all(|&v| v == 1)
            );
            assert_eq!(
                state[workers_offset as usize..].iter().sum::<u32>(),
                tasks.len() as u32
            );
        }
    }
    gpu.destroy_command_encoder(&mut encoder);
    for buffer in buffers {
        gpu.destroy_buffer(buffer);
    }
}

#[test]
fn all_lane_fences_publish_device_graphs_and_reuse_ready_slots() {
    let gpu = crate::support::gpu::gpu();
    if !gpu.capabilities().device_memory_fence {
        eprintln!("device graph fence contract is unsupported on this backend");
        return;
    }
    let shader = gpu
        .try_create_shader(bg::ShaderDesc {
            source: include_str!("shaders/device_protocol.wgsl"),
            naga_module: None,
        })
        .unwrap();
    let mut layout = <Bindings as bg::ShaderData>::layout();
    layout
        .bindings
        .push(("incoming", bg::ShaderBinding::BufferAlias { binding: 0 }));
    let mut pipeline = gpu.create_compute_pipeline(bg::ComputePipelineDesc {
        name: "device graph protocol",
        data_layouts: &[&layout],
        compute: shader.at("main"),
    });
    let chain: Vec<_> = (0..64)
        .map(|i| {
            (
                [1, 63, 64, 65, 129][i as usize % 5],
                if i == 0 { Vec::new() } else { vec![i - 1] },
            )
        })
        .collect();
    let mut fan = vec![(129, Vec::new()); 65];
    fan.push((513, (0..65).collect()));
    fan.extend((0..8).map(|_| (257, vec![65])));
    let layered: Vec<_> = (0..64)
        .map(|i| {
            let predecessors = if i < 8 {
                Vec::new()
            } else {
                ((i / 8 - 1) * 8..(i / 8) * 8).collect()
            };
            (257, predecessors)
        })
        .collect();
    for graph in [
        &[][..],
        chain.as_slice(),
        fan.as_slice(),
        layered.as_slice(),
    ] {
        check_graph(&gpu, &pipeline, graph);
    }
    gpu.destroy_compute_pipeline(&mut pipeline);
    eprintln!("qualified 4 DAG shapes × 4 worker pools × 16 reset epochs");
}
