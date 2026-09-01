use meganeura::{Graph, Mode, SessionConfig};

#[test]
fn runtime_prefix_updates_only_the_selected_leading_rows() {
    const ROWS: usize = 1024;
    const WIDTH: usize = 8;
    const TABLE_ROWS: usize = 64;

    let mut graph = Graph::new();
    let indices = graph.input_u32("indices", &[ROWS]);
    let factors = graph.input("factors", &[ROWS, WIDTH]);
    let table = graph.parameter("table", &[TABLE_ROWS, WIDTH]);
    let gathered = graph.embedding(indices, table);
    let weighted = graph.mul(gathered, factors);
    let output = graph.sum_inner(weighted);
    graph.set_outputs(vec![output]);

    let (mut session, _) = meganeura::build(
        &graph,
        SessionConfig {
            mode: Mode::Inference,
            ..SessionConfig::default()
        },
    );
    let info = session
        .configure_runtime_prefix(&["indices", "factors"], ROWS as u32)
        .unwrap();
    assert!(info.dispatches > 0);
    assert!(info.alignment < ROWS as u32);

    let indices: Vec<u32> = (0..ROWS)
        .map(|row| (row * 17 % TABLE_ROWS) as u32)
        .collect();
    let table: Vec<f32> = (0..TABLE_ROWS * WIDTH)
        .map(|index| index as f32 * 0.001 - 0.2)
        .collect();
    let first_factors = vec![1.0_f32; ROWS * WIDTH];
    session.set_input_u32("indices", &indices);
    session.set_input("factors", &first_factors);
    session.set_parameter("table", &table);
    session.step();
    session.wait();
    let first = session.read_output(ROWS);

    let active_rows = info.alignment;
    session.set_runtime_prefix_rows(active_rows).unwrap();
    session.set_input("factors", &vec![2.0_f32; ROWS * WIDTH]);
    session.step();
    session.wait();
    let second = session.read_output(ROWS);

    for row in 0..ROWS {
        let expected = if row < active_rows as usize {
            first[row] * 2.0
        } else {
            first[row]
        };
        assert_eq!(
            second[row].to_bits(),
            expected.to_bits(),
            "row {row}: actual={}, expected={expected}",
            second[row]
        );
    }
}

#[test]
fn runtime_prefix_training_matches_zero_padded_static_scatter() {
    const ROWS: usize = 4096;
    const WIDTH: usize = 8;
    const TABLE_ROWS: usize = 257;
    const PIXELS: usize = 257;

    fn make_graph() -> Graph {
        let mut graph = Graph::new();
        let indices = graph.input_u32("indices", &[ROWS]);
        let pixel_indices = graph.input_u32("pixel_indices", &[ROWS]);
        let factors = graph.input("factors", &[ROWS, WIDTH]);
        let table = graph.parameter("table", &[TABLE_ROWS, WIDTH]);
        let gathered = graph.embedding(indices, table);
        let weighted = graph.mul(gathered, factors);
        let row_values = graph.sum_inner(weighted);
        let pixels = graph.scatter_add(pixel_indices, row_values, PIXELS);
        let squared = graph.mul(pixels, pixels);
        let loss = graph.mean_all(squared);
        graph.set_outputs(vec![loss]);
        graph
    }

    let graph = make_graph();
    let (mut compact, _) = meganeura::build(&graph, SessionConfig::default());
    let (mut padded, _) = meganeura::build(&graph, SessionConfig::default());
    let info = compact
        .configure_runtime_prefix(&["indices", "pixel_indices", "factors"], ROWS as u32)
        .unwrap();
    assert!(info.dispatches >= 4, "configured {info:?}");
    assert!(info.alignment < ROWS as u32, "configured {info:?}");
    compact.set_runtime_prefix_rows(info.alignment).unwrap();

    let active_rows = info.alignment as usize;
    let indices: Vec<u32> = (0..ROWS)
        .map(|row| (row * 29 % TABLE_ROWS) as u32)
        .collect();
    let pixel_indices: Vec<u32> = (0..ROWS).map(|row| (row * 11 % PIXELS) as u32).collect();
    let mut factors = vec![0.0_f32; ROWS * WIDTH];
    for (index, factor) in factors[..active_rows * WIDTH].iter_mut().enumerate() {
        *factor = (index * 13 % 31) as f32 * 0.003 - 0.04;
    }
    let table: Vec<f32> = (0..TABLE_ROWS * WIDTH)
        .map(|index| (index * 7 % 101) as f32 * 0.002 - 0.1)
        .collect();

    for session in [&mut compact, &mut padded] {
        session.set_input_u32("indices", &indices);
        session.set_input_u32("pixel_indices", &pixel_indices);
        session.set_input("factors", &factors);
        session.set_parameter("table", &table);
        session.set_learning_rate(0.01);
        session.step();
        session.wait();
    }

    let compact_loss = compact.read_output(1)[0];
    let padded_loss = padded.read_output(1)[0];
    assert_eq!(compact_loss.to_bits(), padded_loss.to_bits());

    let mut compact_table = vec![0.0_f32; table.len()];
    let mut padded_table = vec![0.0_f32; table.len()];
    compact.read_param("table", &mut compact_table);
    padded.read_param("table", &mut padded_table);
    for (index, (&compact_value, &padded_value)) in
        compact_table.iter().zip(&padded_table).enumerate()
    {
        assert!(
            (compact_value - padded_value).abs() <= 1.0e-7,
            "table[{index}]: compact={compact_value}, padded={padded_value}"
        );
    }
}

#[cfg(target_os = "linux")]
#[test]
fn gpu_count_drives_runtime_prefix_without_host_readback() {
    const ROWS: usize = 1024;
    const WIDTH: usize = 8;
    const TABLE_ROWS: usize = 64;

    let gpu = std::sync::Arc::new(
        meganeura::init_gpu_context_with(meganeura::GpuOptions::from_env()).expect("GPU context"),
    );
    let count = gpu.create_buffer(blade_graphics::BufferDesc {
        name: "runtime-prefix-test-count",
        size: 4,
        memory: blade_graphics::Memory::External(blade_graphics::ExternalMemorySource::Fd(None)),
    });
    let count_upload = gpu.create_buffer(blade_graphics::BufferDesc {
        name: "runtime-prefix-test-count-upload",
        size: 4,
        memory: blade_graphics::Memory::Upload,
    });
    let source = gpu
        .get_external_buffer_source(count)
        .expect("external count source");

    let mut graph = Graph::new();
    let indices = graph.input_u32("indices", &[ROWS]);
    let factors = graph.input("factors", &[ROWS, WIDTH]);
    let table = graph.parameter("table", &[TABLE_ROWS, WIDTH]);
    let gathered = graph.embedding(indices, table);
    let weighted = graph.mul(gathered, factors);
    let output = graph.sum_inner(weighted);
    graph.set_outputs(vec![output]);
    let (mut session, _) = meganeura::build(
        &graph,
        SessionConfig {
            mode: Mode::Inference,
            gpu: Some(gpu.clone()),
            ..SessionConfig::default()
        },
    );
    let info = session
        .configure_runtime_prefix(&["indices", "factors"], ROWS as u32)
        .unwrap();
    session.bind_runtime_prefix_count(source, 4).unwrap();

    let indices: Vec<u32> = (0..ROWS)
        .map(|row| (row * 17 % TABLE_ROWS) as u32)
        .collect();
    let table: Vec<f32> = (0..TABLE_ROWS * WIDTH)
        .map(|index| index as f32 * 0.001 - 0.2)
        .collect();
    session.set_input_u32("indices", &indices);
    session.set_input("factors", &vec![1.0_f32; ROWS * WIDTH]);
    session.set_parameter("table", &table);

    let mut encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
        name: "runtime-prefix-test-producer",
        buffer_count: 1,
        manual_barriers: false,
    });
    let write_count = |value: u32,
                       encoder: &mut blade_graphics::CommandEncoder,
                       gpu: &blade_graphics::Context| {
        unsafe {
            *(count_upload.data() as *mut u32) = value;
        }
        encoder.start();
        {
            let mut transfer = encoder.transfer("runtime-prefix-test-count");
            transfer.copy_buffer_to_buffer(count_upload.at(0), count.at(0), 4);
        }
        let _ = gpu.submit(encoder);
    };

    write_count(ROWS as u32, &mut encoder, &gpu);
    session.step();
    session.wait();
    let first = session.read_output(ROWS);

    let active_rows = info.alignment;
    write_count(active_rows, &mut encoder, &gpu);
    session.set_input("factors", &vec![2.0_f32; ROWS * WIDTH]);
    session.step();
    session.wait();
    let second = session.read_output(ROWS);
    for row in 0..ROWS {
        let expected = if row < active_rows as usize {
            first[row] * 2.0
        } else {
            first[row]
        };
        assert_eq!(second[row].to_bits(), expected.to_bits(), "row {row}");
    }

    drop(session);
    gpu.destroy_command_encoder(&mut encoder);
    gpu.destroy_buffer(count_upload);
    gpu.destroy_buffer(count);
}
