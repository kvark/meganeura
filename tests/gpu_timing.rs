//! Explicitly selected GPU regression; no monitoring library or background work.

use meganeura::{CoopPolicy, GpuOptions, Graph, SessionConfig, SessionOptions};
use std::sync::Arc;

fn check(timing: bool) {
    let device_name =
        std::env::var("MEGANEURA_TIMING_TEST_DEVICE").expect("declare expected device");
    let driver = std::env::var("MEGANEURA_TIMING_TEST_DRIVER").expect("declare expected driver");
    let options = GpuOptions {
        timing,
        ..GpuOptions::from_env()
    };
    assert!(options.device_id.is_some(), "select a physical device");
    let gpu = Arc::new(meganeura::init_gpu_context_with(options).unwrap());
    let info = gpu.device_information();
    assert_eq!(info.device_name, device_name);
    assert_eq!(info.driver_info, driver);
    assert!(!info.is_software_emulated);
    eprintln!("timing-device enabled={timing} {info:?}");

    let buffer = gpu.create_buffer(blade_graphics::BufferDesc {
        name: "timing-regression",
        size: 256,
        memory: blade_graphics::Memory::Shared,
    });
    let mut encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
        name: "timing-regression",
        buffer_count: 1,
        manual_barriers: false,
    });
    assert_eq!(encoder.timing_enabled(), timing);
    for (name, value) in [("first", 7), ("second", 19)] {
        encoder.start();
        encoder.transfer(name).fill_buffer(buffer.at(0), 256, value);
        let sync = gpu.submit(&mut encoder);
        assert!(gpu.wait_for(&sync, !0).unwrap());
        let bytes = unsafe { std::slice::from_raw_parts(buffer.data(), 256) };
        assert!(bytes.iter().all(|&byte| byte == value));
        if timing {
            let actual = encoder.last_timing();
            assert_eq!(actual.passes.len(), 1);
            assert_eq!(actual.passes[0].0, name, "must resolve this submission");
            assert!(actual.done > actual.passes[0].1);
            let duration = actual.pass_durations().next().unwrap().1;
            eprintln!(
                "timing-submission name={name} duration_ns={}",
                duration.as_nanos()
            );
            assert_eq!(
                encoder.last_timing(),
                actual,
                "reading timings is repeatable"
            );
        }
    }
    gpu.destroy_command_encoder(&mut encoder);
    gpu.destroy_buffer(buffer);

    let mut graph = Graph::new();
    let x = graph.input("x", &[1, 256]);
    let weight = graph.parameter("weight", &[1, 256]);
    let output = graph.mul(x, weight);
    graph.set_outputs(vec![output]);
    let config = SessionConfig {
        mode: meganeura::Mode::Inference,
        gpu: Some(gpu),
        runtime: SessionOptions {
            coop: CoopPolicy::Disabled,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut session = meganeura::build(&graph, config).0;
    session.set_parameter("weight", &[2.0; 256]);
    session.set_profiling(true);
    for value in [3.0, 5.0] {
        session.set_input("x", &[value; 256]);
        session.step();
        session.wait();
        assert_eq!(session.read_output(256), vec![2.0 * value; 256]);
        let timings = session.gpu_timings();
        assert_eq!(!timings.is_empty(), timing);
        if timing {
            assert!(
                timings
                    .iter()
                    .all(|(name, duration)| !name.is_empty() && !duration.is_zero())
            );
        }
        eprintln!("timing-session input={value} passes={}", timings.len());
    }
    eprintln!("timing-complete enabled={timing} submissions=2 steps=2");
}

#[test]
#[ignore = "requires a separately declared GPU device and host-only guard"]
fn timing_disabled_preserves_execution() {
    check(false);
}

#[test]
#[ignore = "requires a separately declared GPU device and host-only guard"]
fn timing_resolves_the_completed_submission() {
    check(true);
}
