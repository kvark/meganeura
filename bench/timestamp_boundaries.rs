//! Source-only diagnostic: compare isolated and neighboring pass intervals.
use blade_graphics::{self as bg, ShaderData};
use std::time::Instant;

const WORDS: usize = 65_536;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Params {
    steps: u32,
    padding: [u32; 3],
}

#[derive(blade_macros::ShaderData)]
struct Data {
    values: bg::BufferPiece,
    params: Params,
}

fn main() {
    env_logger::init();
    let gpu = unsafe {
        bg::Context::init(bg::ContextDesc {
            timing: true,
            validation: cfg!(debug_assertions),
            ..Default::default()
        })
        .unwrap()
    };
    assert!(gpu.capabilities().timing);
    let shader = gpu.create_shader(bg::ShaderDesc {
        source: "
            struct Params { steps: u32, pad0: u32, pad1: u32, pad2: u32 }
            var<uniform> params: Params;
            var<storage, read_write> values: array<u32>;
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                var x = values[id.x];
                for (var i = 0u; i < params.steps; i++) {
                    x = x * 1664525u + 1013904223u;
                }
                values[id.x] = x;
            }
        ",
        naga_module: None,
    });
    let mut pipeline = gpu.create_compute_pipeline(bg::ComputePipelineDesc {
        name: "lcg",
        data_layouts: &[&Data::layout()],
        compute: shader.at("main"),
    });
    let buffer = gpu.create_buffer(bg::BufferDesc {
        name: "values",
        size: (WORDS * 4) as u64,
        memory: bg::Memory::Shared,
    });
    let mut encoder = gpu.create_command_encoder(bg::CommandEncoderDesc {
        name: "pass_boundaries",
        buffer_count: 2,
        manual_barriers: false,
    });
    let cases: &[&[u32]] = &[&[4096], &[1], &[4096, 1], &[1, 4096], &[4096, 1, 4096]];
    let mut records = Vec::new();
    for &steps in cases {
        let mut scale = 1u32;
        let mut offset = 0u32;
        for _ in 0..steps.iter().sum::<u32>() {
            scale = scale.wrapping_mul(1664525);
            offset = offset.wrapping_mul(1664525).wrapping_add(1013904223);
        }
        for sample in 0..13 {
            for i in 0..WORDS {
                unsafe { buffer.data().cast::<u32>().add(i).write(i as u32) };
            }
            let before = Instant::now();
            encoder.start();
            for &count in steps {
                let mut pass = encoder.compute(if count == 1 { "light" } else { "heavy" });
                let mut pc = pass.with(&pipeline);
                pc.bind(
                    0,
                    &Data {
                        values: buffer.at(0),
                        params: Params {
                            steps: count,
                            padding: [0; 3],
                        },
                    },
                );
                pc.dispatch([WORDS as u32 / 256, 1, 1]);
            }
            let sync = gpu.submit(&mut encoder);
            assert!(gpu.wait_for(&sync, !0).unwrap());
            let after = Instant::now();
            let times = encoder.last_timing();
            assert_eq!(times.passes.len(), steps.len());
            let intervals: Vec<_> = times
                .passes
                .iter()
                .enumerate()
                .map(|(i, &(_, start))| {
                    let end = times.passes.get(i + 1).map_or(times.done, |p| p.1);
                    end.duration_since(start).as_secs_f64() * 1e6
                })
                .collect();
            assert!(times.passes[0].1 >= before);
            assert!(times.done <= after);
            for i in 0..WORDS {
                let actual = unsafe { buffer.data().cast::<u32>().add(i).read_volatile() };
                assert_eq!(actual, scale.wrapping_mul(i as u32).wrapping_add(offset));
            }
            if sample >= 3 {
                records.push(serde_json::json!({
                    "steps": steps, "intervals_us": intervals,
                    "wall_us": after.duration_since(before).as_secs_f64() * 1e6,
                }));
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "device": gpu.device_information().device_name,
            "records": records,
        })
    );
    gpu.destroy_command_encoder(&mut encoder);
    gpu.destroy_compute_pipeline(&mut pipeline);
    gpu.destroy_buffer(buffer);
}
