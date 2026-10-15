use blade_graphics as bg;
use meganeura::codegen::WorkgroupMatmulShape as Tile;
use meganeura::{CoopPolicy, Graph, Session, SessionConfig};
use serde_json::json;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) * 0.5
}

fn batch(gpu: &bg::Context, encoder: &mut bg::CommandEncoder, session: &mut Session) -> f64 {
    let start = Instant::now();
    encoder.start();
    for _ in 0..64 {
        session.record(encoder).unwrap();
    }
    let sync = gpu.submit(encoder);
    session.track_submission(sync);
    session.wait();
    start.elapsed().as_secs_f64() * 1e3 / 64.0
}

fn session(
    gpu: &Arc<bg::Context>,
    graph: &Graph,
    tile: Option<Tile>,
    a: &[f32],
    b: &[f32],
) -> Session {
    let mut config = SessionConfig::inference_from_env_on(gpu.clone());
    config.tune = false;
    config.runtime.coop = CoopPolicy::AllowF16;
    config.runtime.poison = true;
    config.runtime.workgroup_matmul = tile;
    let (mut session, _) = meganeura::build(graph, config);
    session.set_input("a", a);
    session.set_parameter("b", b);
    session
}

fn main() {
    let gpu = meganeura::reference::gpu::shared_context();
    eprintln!("capabilities {:?}", gpu.capabilities().cooperative_matrix);
    let sweep = std::env::var_os("SWEEP").is_some();
    let smoke = std::env::var_os("SMOKE").is_some();
    let case_filter = std::env::var("CASE").ok();
    let reverse = std::env::var_os("REVERSE").is_some();
    let tiles: Vec<_> = if let Ok(tile) = std::env::var("TILE") {
        let values: Vec<u32> = tile.split(',').map(|s| s.parse().unwrap()).collect();
        assert_eq!(values.len(), 4);
        vec![Tile {
            rows: values[0],
            columns: values[1],
            k_stage: values[2],
            threads: values[3],
        }]
    } else if sweep {
        [
            (32, 64, 32, 32),
            (64, 64, 32, 64),
            (64, 64, 32, 128),
            (64, 64, 64, 128),
            (64, 128, 32, 128),
            (64, 128, 64, 128),
            (128, 64, 64, 128),
            (128, 128, 32, 128),
            (128, 128, 64, 128),
            (128, 128, 64, 256),
        ]
        .map(|(rows, columns, k_stage, threads)| Tile {
            rows,
            columns,
            k_stage,
            threads,
        })
        .to_vec()
    } else {
        vec![Tile {
            rows: 64,
            columns: 64,
            k_stage: 32,
            threads: 64,
        }]
    };
    let shapes = if smoke {
        vec![(128, 128, 128)]
    } else if sweep {
        vec![(512, 512, 512), (1024, 1024, 1024)]
    } else {
        vec![
            (128, 576, 576),
            (512, 512, 512),
            (1024, 1024, 1024),
            (128, 1536, 576),
            (257, 512, 65),
        ]
    };
    let mut encoder = gpu.create_command_encoder(bg::CommandEncoderDesc {
        name: "workgroup matrix paired probe",
        buffer_count: 2,
        manual_barriers: false,
    });
    for (m, n, k) in shapes {
        for bf16 in [false, true] {
            for layout in ["nn", "bt", "at"] {
                let name = format!("{m}x{n}x{k}-{bf16}-{layout}");
                if case_filter
                    .as_ref()
                    .is_some_and(|filter| !name.contains(filter))
                {
                    continue;
                }
                let mut graph = Graph::new();
                let a = graph.input("a", &if layout == "at" { [k, m] } else { [m, k] });
                let b = graph.parameter("b", &if layout == "bt" { [n, k] } else { [k, n] });
                let (aa, bb) = if bf16 {
                    (graph.to_bf16(a), graph.to_bf16(b))
                } else {
                    (a, b)
                };
                let y = match layout {
                    "bt" => graph.matmul_bt(aa, bb),
                    "at" => graph.matmul_at(aa, bb),
                    _ => graph.matmul(aa, bb),
                };
                graph.set_outputs(vec![y]);
                let a: Vec<f32> = (0..m * k)
                    .map(|i| ((i * 17 % 61) as f32 - 30.) / 64.)
                    .collect();
                let b: Vec<f32> = (0..n * k)
                    .map(|i| ((i * 31 % 59) as f32 - 29.) / 64.)
                    .collect();
                let mut expected = vec![0.; m * n];
                for r in 0..m {
                    for c in 0..n {
                        expected[r * n + c] = (0..k)
                            .map(|i| {
                                f64::from(a[if layout == "at" { i * m + r } else { r * k + i }])
                                    * f64::from(
                                        b[if layout == "bt" { c * k + i } else { i * n + c }],
                                    )
                            })
                            .sum::<f64>();
                    }
                }
                let mut baseline = session(&gpu, &graph, None, &a, &b);
                baseline.step();
                baseline.wait();
                let baseline_out = baseline.read_output(m * n);
                assert!(
                    baseline_out
                        .iter()
                        .zip(&expected)
                        .all(|(&x, &y)| x.is_finite()
                            && (f64::from(x) - y).abs() <= 1e-5 * (1. + y.abs()))
                );
                for &tile in &tiles {
                    let mut candidate = session(&gpu, &graph, Some(tile), &a, &b);
                    candidate.step();
                    candidate.wait();
                    let out = candidate.read_output(m * n);
                    let mut max_error = 0f64;
                    for (&x, &y) in out.iter().zip(&expected) {
                        let error = (f64::from(x) - y).abs();
                        assert!(
                            x.is_finite() && error <= 1e-5 * (1. + y.abs()),
                            "{name} {tile:?}: {x} != {y}"
                        );
                        max_error = max_error.max(error);
                    }
                    let selected = candidate
                        .dispatch_pipeline_keys()
                        .iter()
                        .any(|s| s.contains(":workgroup-"));
                    assert_eq!(
                        selected,
                        tile.supports(&gpu.capabilities(), bf16)
                            && tile.fits(m as u32, n as u32, k as u32),
                        "{name} {tile:?}"
                    );
                    let mut sessions = [&mut baseline, &mut candidate];
                    for s in &mut sessions {
                        let start = Instant::now();
                        while start.elapsed() < Duration::from_millis(if sweep { 150 } else { 500 })
                        {
                            batch(&gpu, &mut encoder, s);
                        }
                    }
                    let mut samples = [Vec::new(), Vec::new()];
                    for round in 0..if sweep { 4 } else { 10 } {
                        let order = if (round % 2 == 0) ^ reverse {
                            [0, 1, 1, 0]
                        } else {
                            [1, 0, 0, 1]
                        };
                        for index in order {
                            samples[index].push(batch(&gpu, &mut encoder, sessions[index]));
                        }
                    }
                    let times = samples.each_ref().map(|v| median(v));
                    println!(
                        "{}",
                        json!({"case":name,"shape":[m,n,k],"bf16":bf16,"layout":layout,"tile":tile,"selected":selected,"max_error":max_error,"values_checked":out.len(),"replays":64,"median_ms":times,"speedup":times[0]/times[1],"samples_ms":samples,"pipelines":sessions.each_ref().map(|s|s.dispatch_pipeline_keys()),"device":gpu.device_information().device_name})
                    );
                }
            }
        }
    }
    gpu.destroy_command_encoder(&mut encoder);
}
