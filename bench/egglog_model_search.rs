//! CPU survey, or whole-model search with a full independent CPU reference.
//! Usage: egglog_model_search MODEL [REFERENCE.f32|optimized] [fast] [baseline] [--static]
use meganeura::{
    Graph,
    models::{smolvla, whisper},
    optimize::search,
};
use std::time::{Duration, Instant};

fn initialize(session: &mut meganeura::Session, model: &str) {
    for (name, buffer) in session.plan().param_buffers.clone() {
        let seed_name = if model == "Whisper-tiny" {
            name.strip_prefix("model.encoder.")
                .unwrap_or(&name)
                .replace("fused_bias", "bias")
        } else {
            name.clone()
        };
        let seed = seed_name
            .bytes()
            .fold(0u32, |hash, c| hash.wrapping_mul(31).wrapping_add(c as u32))
            % 10000;
        let len = session.plan().buffers[buffer.0 as usize] / 4;
        let values: Vec<_> = (0..len)
            .map(|i| (i as f32 * 0.01 + seed as f32).sin() * 0.02)
            .collect();
        session.set_parameter(&name, &values);
    }
    for (name, buffer) in session.plan().input_buffers.clone() {
        if !session
            .plan()
            .dispatches
            .iter()
            .any(|dispatch| dispatch.input_buffers.contains(&buffer))
        {
            continue;
        }
        let len = session.plan().buffers[buffer.0 as usize] / 4;
        let values: Vec<_> = (0..len)
            .map(|i| match name.as_str() {
                "mel" => (i as f32 * 0.001).sin(),
                "noisy_actions" => (i as f32 * 0.01).sin(),
                "timestep" => (i as f32 * 0.005).sin(),
                other if other.starts_with("vlm_kv_layer_") => (i as f32 * 0.01).cos(),
                other => panic!("unexpected model input {other}"),
            })
            .collect();
        session.set_input(&name, &values);
    }
}

fn check(session: &mut meganeura::Session, reference: &[f32]) -> Result<(f64, f64, f64), String> {
    session.step();
    session.wait();
    let output = session.read_output(reference.len());
    let mut squared_error = 0.0;
    let mut max_error = 0.0f64;
    let mut norm = 0.0;
    let mut actual_norm = 0.0;
    for (&actual, &expected) in output.iter().zip(reference) {
        if !actual.is_finite() || !expected.is_finite() {
            return Err("non-finite model output".into());
        }
        let diff = f64::from(actual) - f64::from(expected);
        squared_error += diff * diff;
        max_error = max_error.max(diff.abs());
        norm += f64::from(expected).powi(2);
        actual_norm += f64::from(actual).powi(2);
    }
    let relative_l2 = (squared_error / norm.max(1e-24)).sqrt();
    let loss_error = (actual_norm - norm).abs() / norm.max(1e-24);
    // Inferena's fixed forward gate, applied to every element here rather than
    // its cross-framework sample. This experiment does not validate training.
    if relative_l2 >= 0.01 || loss_error >= 0.01 {
        return Err(format!(
            "full output rel-L2={relative_l2}, loss error={loss_error}, max={max_error}"
        ));
    }
    Ok((relative_l2, loss_error, max_error))
}

fn measure(model: &str, graph: Graph, reference: &[f32], fast: bool, baseline: bool) {
    use meganeura::{SessionOptions, codegen, compile};
    let output_len = graph.node(graph.outputs()[0]).ty.num_elements();
    assert_eq!(output_len, reference.len());
    let graph = meganeura::optimize::optimize(&graph);
    let gpu = std::sync::Arc::new(meganeura::runtime::init_gpu_context().unwrap());
    let caps = gpu.capabilities().cooperative_matrix;
    let caps = codegen::CoopCaps {
        f16_tile: if fast { caps.f16_tile } else { 0 },
        f32_tile: caps.f32_tile,
    };
    let mut forms = vec![search::Candidate {
        graph: graph.deep_clone(),
        expression: "greedy control".into(),
    }];
    let mut extraction_truncated = false;
    let extraction_start = Instant::now();
    if !baseline {
        for region in meganeura::outline::detect_repeated_regions(&graph)
            .into_iter()
            .take(1)
        {
            let space = search::repeated_candidates(&graph, region, 8).unwrap();
            extraction_truncated |= space.truncated;
            forms.extend(space.candidates);
        }
    }
    let extraction_ms = extraction_start.elapsed().as_secs_f64() * 1000.0;
    let mut programs = Vec::new();
    for (index, form) in forms.into_iter().enumerate() {
        let variants: &[(bool, u32)] = if baseline {
            &[(true, 1)]
        } else {
            &[(true, 1), (false, 1), (false, 8)]
        };
        for &(fuse_dispatches, splits) in variants {
            let options = compile::CompileOptions {
                fuse_dispatches,
                flash_forward_coop: fast,
                ..Default::default()
            };
            let mut plan = compile::compile_with_caps(&form.graph, &options, caps);
            let mut split_products = 0;
            if splits > 1 {
                for i in (0..plan.dispatches.len()).rev() {
                    let shape = codegen::ScalarMatmulShape {
                        tile_size: 64,
                        k_stage: 8,
                        interleave_columns: false,
                    };
                    if plan
                        .split_matmul(i, shape, splits, 64 * 1024 * 1024)
                        .is_ok()
                    {
                        split_products += 1;
                    }
                }
                if split_products == 0 {
                    continue;
                }
            }
            programs.push(search::measure::Program {
                description: format!("form={index}, dispatch_fusion={fuse_dispatches}, split_products={split_products}; {}", form.expression),
                plan,
            });
        }
    }
    let (mut session, report) = search::measure::select(
        programs,
        gpu,
        SessionOptions {
            coop: if fast {
                meganeura::CoopPolicy::Auto
            } else {
                meganeura::CoopPolicy::NativeF32
            },
            ..Default::default()
        },
        search::measure::Options {
            tuning: meganeura::TuneOptions {
                max_time: if std::env::args().any(|arg| arg == "--static") {
                    Duration::ZERO
                } else {
                    Duration::from_secs(2)
                },
                max_classes: 32,
                sample_pairs: 12,
                min_improvement: 0.02,
                ..Default::default()
            },
            max_time: Duration::from_secs(180),
            max_programs: 32,
            max_plan_bytes: 3 * 1024 * 1024 * 1024,
        },
        |session| {
            initialize(session, model);
            Ok(())
        },
        |session| check(session, reference).map(|_| ()),
    )
    .unwrap();
    let mut samples = Vec::new();
    for i in 0..35 {
        let start = Instant::now();
        session.step();
        session.wait();
        if i >= 5 {
            samples.push(start.elapsed().as_secs_f64() * 1000.0)
        }
    }
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    let errors = check(&mut session, reference).unwrap();
    println!(
        "{}",
        serde_json::json!({"model": model, "device": session.device_information().device_name,
            "fast": fast, "extraction_ms": extraction_ms, "extraction_truncated": extraction_truncated,
            "report": report, "held_out_ms": samples, "median_ms": sorted[sorted.len()/2],
            "relative_l2_error": errors.0, "loss_relative_error": errors.1, "max_abs_error": errors.2,
            "dispatches": session.plan().dispatches.len(), "groups": session.num_groups(),
            "allocated_buffer_bytes": session.memory_summary().allocated_buffer_bytes,
        })
    );
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let model = std::env::args().nth(1).expect("SmolVLA or Whisper-tiny");
    let mut graph = Graph::new();
    let output = match model.as_str() {
        "SmolVLA" => {
            smolvla::build_action_expert(&mut graph, &smolvla::Config::smolvla_base(), 50, 16)
        }
        "Whisper-tiny" => {
            whisper::build_encoder(&mut graph, &whisper::Config::whisper_tiny(), 1, 3000)
        }
        _ => panic!("SmolVLA or Whisper-tiny"),
    };
    graph.set_outputs(vec![output]);
    if let Some(reference) = std::env::args().nth(2).filter(|arg| arg != "optimized") {
        let bytes = std::fs::read(reference).unwrap();
        assert_eq!(bytes.len() % 4, 0);
        let values: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&word| f32::from_le_bytes(word))
            .collect();
        measure(
            &model,
            graph,
            &values,
            std::env::args().nth(3).is_some_and(|s| s == "fast"),
            std::env::args().nth(4).is_some_and(|s| s == "baseline"),
        );
        return;
    }
    if std::env::args()
        .nth(2)
        .is_some_and(|arg| arg == "optimized")
    {
        graph = meganeura::optimize::optimize(&graph);
    }
    for node in graph.nodes() {
        log::debug!(
            "{}: {:?} {:?} {:?}",
            node.id,
            node.op,
            node.ty.shape,
            node.inputs
        );
    }
    let regions = meganeura::outline::detect_repeated_regions(&graph);
    let mut results = Vec::new();
    for region in regions {
        let start = Instant::now();
        let space = search::repeated_candidates(&graph, region, 8).unwrap();
        results.push(serde_json::json!({
            "start": region.start, "nodes": region.period, "instances": region.count,
            "search_ms": start.elapsed().as_secs_f64() * 1000.0,
            "truncated": space.truncated,
            "expressions": space.candidates.iter().map(|c| &c.expression).collect::<Vec<_>>(),
            "candidate_nodes": space.candidates.iter().map(|c| c.graph.nodes().len()).collect::<Vec<_>>(),
        }));
    }
    println!(
        "{}",
        serde_json::json!({"model": model, "source_nodes": graph.nodes().len(), "regions": results})
    );
}
