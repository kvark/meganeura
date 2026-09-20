//! CPU survey, or whole-model search with a full independent CPU reference.
//! Usage: egglog_model_search MODEL [REFERENCE.f32|optimized] [fast] [baseline]
//! Options: --static, --confirm, --reverse, --profile, --optimized-graph,
//! --attention-tiles, --cooperative-split,
//! --program=N (one generated plan, for attribution), --serial-sums,
//! --seconds=N, --warmup=N (whole-program warmup pairs; default 32).
use meganeura::{
    Graph,
    models::{smolvla, whisper},
    optimize::search,
};
use std::time::{Duration, Instant};

fn initialize(
    session: &mut meganeura::Session,
    mut incumbent: Option<&mut meganeura::Session>,
    model: &str,
    parameters: &mut std::collections::HashMap<(String, usize), Vec<f32>>,
) {
    // A complete matching interface can share prepared derived weights too.
    // Otherwise initialize dependent sources normally so transforms are rebuilt.
    let mut shareable = std::collections::HashSet::new();
    if let Some(source) = incumbent.as_ref() {
        let target = session.plan();
        let origin = source.plan();
        for (name, buffer) in &target.param_buffers {
            let matches = origin
                .param_buffers
                .iter()
                .find(|p| &p.0 == name)
                .is_some_and(|p| {
                    let transform = |plan: &meganeura::compile::ExecutionPlan, b| {
                        plan.derived_params
                            .iter()
                            .find(|d| d.0 == b)
                            .map(|d| (d.1.clone(), d.2.clone()))
                    };
                    target.param_types.contains_key(buffer)
                        && target.param_types.get(buffer) == origin.param_types.get(&p.1)
                        && target.weight_buffers.get(buffer) == origin.weight_buffers.get(&p.1)
                        && transform(target, *buffer) == transform(origin, p.1)
                });
            if matches {
                shareable.insert(name.clone());
            }
        }
        if shareable.len() != target.param_buffers.len() {
            for (name, buffer) in &target.param_buffers {
                if target
                    .derived_params
                    .iter()
                    .any(|d| d.0 == *buffer || d.1.iter().any(|source| source.0 == *name))
                {
                    shareable.remove(name);
                }
            }
        }
    }
    for (name, buffer) in session.plan().param_buffers.clone() {
        if shareable.contains(&name) {
            session
                .share_parameter_from(incumbent.as_deref_mut().unwrap(), &name)
                .unwrap();
            continue;
        }
        let len = session.plan().buffers[buffer.0 as usize] / 4;
        let values = parameters.entry((name.clone(), len)).or_insert_with(|| {
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
            (0..len)
                .map(|i| (i as f32 * 0.01 + seed as f32).sin() * 0.02)
                .collect()
        });
        session.set_parameter(&name, values);
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
    verify(session, reference)
}

fn verify(session: &meganeura::Session, reference: &[f32]) -> Result<(f64, f64, f64), String> {
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
    let original = graph;
    let graph = meganeura::optimize::optimize(&original);
    let original_graph = !std::env::args().any(|arg| arg == "--optimized-graph");
    let search_source = if original_graph { &original } else { &graph };
    let profile = std::env::args().any(|arg| arg == "--profile");
    let gpu = std::sync::Arc::new(
        meganeura::runtime::init_gpu_context_with(meganeura::runtime::GpuOptions {
            timing: profile,
            capture: profile,
            ..Default::default()
        })
        .unwrap(),
    );
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
        for region in meganeura::outline::detect_repeated_regions(search_source)
            .into_iter()
            .take(1)
        {
            let space = search::repeated_candidates(search_source, region, 8).unwrap();
            extraction_truncated |= space.truncated;
            forms.extend(space.candidates);
        }
    }
    let extraction_ms = extraction_start.elapsed().as_secs_f64() * 1000.0;
    let mut programs: Vec<search::measure::Program> = Vec::new();
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
            // These two plans came from the same graph. If dispatch fusion did
            // not change its lowering, do not rebuild and retune it a second time.
            if !fuse_dispatches
                && splits == 1
                && programs.last().is_some_and(|p| {
                    p.plan.dispatches == plan.dispatches && p.plan.buffers == plan.buffers
                })
            {
                continue;
            }
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
                description: format!("form={index}, dispatch_fusion={fuse_dispatches}, split_products={split_products}"),
                plan,
            });
        }
    }
    if std::env::args().any(|arg| arg == "--cooperative-split")
        && fast
        && caps.f16_tile == 16
        && gpu.capabilities().fixed_compute_subgroup_size == Some(32)
    {
        let mut variants = Vec::new();
        for program in &programs {
            let mut plan = program.plan.clone();
            let changed = (0..plan.dispatches.len())
                .filter(|&i| plan.cooperative_split_matmul(i).is_ok())
                .count();
            if changed > 0 {
                variants.push(search::measure::Program {
                    description: format!("cooperative_split={changed}; {}", program.description),
                    plan,
                });
            }
        }
        programs.extend(variants);
    }
    if std::env::args().any(|arg| arg == "--attention-tiles")
        && gpu.capabilities().fixed_compute_subgroup_size == Some(32)
    {
        let mut variants = Vec::new();
        for program in &programs {
            for query_tiles in [1, 2, 4] {
                let mut plan = program.plan.clone();
                let mut changed = 0;
                for d in &mut plan.dispatches {
                    if d.shader == compile::ShaderEntry::FlashAttentionCoop
                        && codegen::cooperative_attention_tile_is_legal(d.params[3], query_tiles)
                    {
                        d.kernel = compile::Kernel::CooperativeAttention { query_tiles };
                        d.workgroups[0] = d.params[0].div_ceil(16 * query_tiles);
                        changed += 1;
                    }
                }
                if changed > 0 {
                    variants.push(search::measure::Program {
                        description: format!("query_tiles={query_tiles}; {}", program.description),
                        plan,
                    });
                }
            }
        }
        programs.extend(variants);
    }
    if std::env::args().any(|arg| arg == "--serial-sums") {
        let mut expanded = Vec::new();
        for program in programs {
            let mut variants = Vec::new();
            for workgroup_size in [64, 128, 256] {
                let mut plan = program.plan.clone();
                let mut changed = 0;
                for d in &mut plan.dispatches {
                    if d.shader == compile::ShaderEntry::SumRows {
                        d.kernel = compile::Kernel::SumRowsSerial { workgroup_size };
                        d.workgroups[0] = d.params[1].div_ceil(workgroup_size);
                        changed += 1;
                    }
                }
                if changed > 0 {
                    variants.push(search::measure::Program {
                        description: format!(
                            "serial_sums={workgroup_size}; {}",
                            program.description
                        ),
                        plan,
                    });
                }
            }
            expanded.push(program);
            expanded.extend(variants);
        }
        programs = expanded;
    }
    if let Some(index) = std::env::args().find_map(|arg| {
        arg.strip_prefix("--program=")
            .map(|s| s.parse::<usize>().unwrap())
    }) {
        programs = vec![programs.remove(index)];
    }
    let fixed_subgroup_size = gpu.capabilities().fixed_compute_subgroup_size;
    let runtime = SessionOptions {
        gpu_timing: profile,
        wgsl_dump_dir: std::env::var("MEGANEURA_DUMP_WGSL").ok(),
        skip_parameter_zero: true,
        reuse_upload_staging: true,
        coop: if fast {
            meganeura::CoopPolicy::Auto
        } else {
            meganeura::CoopPolicy::NativeF32
        },
        ..Default::default()
    };
    let tuning = meganeura::TuneOptions {
        max_time: if std::env::args().any(|arg| arg == "--static") {
            Duration::ZERO
        } else {
            Duration::from_secs(2)
        },
        max_classes: 32,
        sample_pairs: 12,
        min_improvement: 0.02,
        ..Default::default()
    };
    let control_plan = std::env::args()
        .any(|arg| arg == "--confirm")
        .then(|| programs[0].plan.clone());
    if std::env::args().any(|arg| arg == "--reverse") {
        programs[1..].reverse();
    }
    let search_seconds = std::env::args()
        .find_map(|arg| {
            arg.strip_prefix("--seconds=")
                .map(|s| s.parse::<u64>().unwrap())
        })
        .unwrap_or(180);
    let mut parameters = std::collections::HashMap::new();
    let warmup_runs = std::env::args()
        .find_map(|arg| {
            arg.strip_prefix("--warmup=")
                .map(|n| n.parse::<u32>().unwrap())
        })
        .unwrap_or(32);
    let (mut session, report) = search::measure::select(
        programs,
        gpu.clone(),
        runtime.clone(),
        search::measure::Options {
            tuning: tuning.clone(),
            warmup_runs,
            max_time: Duration::from_secs(search_seconds),
            max_programs: 64,
            max_plan_bytes: 3 * 1024 * 1024 * 1024,
        },
        |session, incumbent| {
            initialize(session, incumbent, model, &mut parameters);
            Ok(())
        },
        |session| verify(session, reference).map(|_| ()),
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
    let confirmation = control_plan.map(|plan| {
        let mut control = meganeura::Session::with_context_opts(plan, gpu, runtime);
        initialize(&mut control, None, model, &mut parameters);
        check(&mut control, reference).unwrap();
        let control_tuning = control.tune_with(tuning).unwrap();
        check(&mut control, reference).unwrap();
        let mut times = [Vec::new(), Vec::new()];
        for pair in 0..70 {
            for index in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
                let target = if index == 0 { &mut control } else { &mut session };
                let start = Instant::now();
                target.step();
                target.wait();
                if pair >= 30 { times[index].push(start.elapsed().as_secs_f64() * 1000.0); }
            }
        }
        check(&mut control, reference).unwrap();
        check(&mut session, reference).unwrap();
        serde_json::json!({"control_ms": times[0], "selected_ms": times[1], "control_tuning": control_tuning})
    });
    let profile = profile.then(|| {
        let result = meganeura::profiler::capture_session_profile(
            &mut session,
            |_| {},
            meganeura::profiler::CaptureOptions {
                samples: 5,
                ..Default::default()
            },
        )
        .unwrap();
        check(&mut session, reference).unwrap();
        result
    });
    println!(
        "{}",
        serde_json::json!({"model": model, "device": session.device_information().device_name,
            "fast": fast, "original_graph": original_graph,
            "fixed_compute_subgroup_size": fixed_subgroup_size,
            "extraction_ms": extraction_ms, "extraction_truncated": extraction_truncated,
            "report": report, "held_out_ms": samples, "median_ms": sorted[sorted.len()/2],
            "relative_l2_error": errors.0, "loss_relative_error": errors.1, "max_abs_error": errors.2,
            "dispatches": session.plan().dispatches.len(), "groups": session.num_groups(),
            "allocated_buffer_bytes": session.memory_summary().allocated_buffer_bytes,
            "confirmation": confirmation,
            "profile": profile,
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
