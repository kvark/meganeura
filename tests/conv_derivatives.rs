//! Full f64 oracle for convolution derivatives; no finite-difference cancellation.
use meganeura::{
    CoopPolicy, Graph, Session, SessionOptions,
    compile::{BufferRef, ExecutionPlan, ShaderEntry},
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
struct Shape {
    batch: u32,
    ci: u32,
    h: u32,
    w: u32,
    co: u32,
    kh: u32,
    kw: u32,
    stride: u32,
    ph: u32,
    pw: u32,
}

impl Shape {
    fn output(self) -> (u32, u32) {
        (
            (self.h + 2 * self.ph - self.kh) / self.stride + 1,
            (self.w + 2 * self.pw - self.kw) / self.stride + 1,
        )
    }

    fn sizes(self) -> [usize; 3] {
        let (oh, ow) = self.output();
        [
            (self.batch * self.ci * self.h * self.w) as usize,
            (self.co * self.ci * self.kh * self.kw) as usize,
            (self.batch * self.co * oh * ow) as usize,
        ]
    }
}

fn gpu() -> Arc<blade_graphics::Context> {
    let gpu =
        Arc::new(meganeura::init_gpu_context_with(meganeura::GpuOptions::from_env()).unwrap());
    eprintln!("GPU: {}", gpu.device_information().device_name);
    if let Ok(expected) = std::env::var("KINDLE_EXPECT_DEVICE_NAME") {
        assert_eq!(gpu.device_information().device_name, expected);
        assert!(!gpu.device_information().is_software_emulated);
        let memory = gpu.memory_stats();
        assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
    }
    gpu
}

fn data(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16777216.0 - 0.5) * scale
        })
        .collect()
}

fn reference(s: Shape, x: &[f32], w: &[f32], dy: &[f32]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut output = vec![0.0; dy.len()];
    let mut dx = vec![0.0; x.len()];
    let mut dw = vec![0.0; w.len()];
    let (out_h, out_w) = s.output();
    // Scatter the forward cross-correlation's contributions. This independent
    // indexing does not use either GPU kernel's implicit-GEMM/gather formula.
    for n in 0..s.batch {
        for co in 0..s.co {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let yi = (((n * s.co + co) * out_h + oh) * out_w + ow) as usize;
                    for ci in 0..s.ci {
                        for kh in 0..s.kh {
                            for kw in 0..s.kw {
                                let ih = (oh * s.stride + kh) as i32 - s.ph as i32;
                                let iw = (ow * s.stride + kw) as i32 - s.pw as i32;
                                if ih < 0 || iw < 0 || ih >= s.h as i32 || iw >= s.w as i32 {
                                    continue;
                                }
                                let xi = (((n * s.ci + ci) * s.h + ih as u32) * s.w + iw as u32)
                                    as usize;
                                let wi = (((co * s.ci + ci) * s.kh + kh) * s.kw + kw) as usize;
                                output[yi] += f64::from(x[xi]) * f64::from(w[wi]);
                                dx[xi] += f64::from(dy[yi]) * f64::from(w[wi]);
                                dw[wi] += f64::from(dy[yi]) * f64::from(x[xi]);
                            }
                        }
                    }
                }
            }
        }
    }
    (output, dx, dw)
}

fn check(label: &str, actual: &[f32], expected: &[f64], scale: f32) {
    qualification(label, actual, expected, scale).unwrap();
}

fn qualification(label: &str, actual: &[f32], expected: &[f64], scale: f32) -> Result<(), String> {
    assert_eq!(actual.len(), expected.len());
    let mut error = 0.0;
    let mut norm = 0.0;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let difference = (f64::from(a) - b).abs();
        if !(a.is_finite()
            && b.is_finite()
            && difference <= f64::from(scale) * 1e-5 + b.abs() * 2e-4)
        {
            return Err(format!("{label}[{i}] = {a:e}, reference {b:e}"));
        }
        error += difference * difference;
        norm += b * b;
    }
    if !(norm > 0.0 && (error / norm).sqrt() <= 2e-4) {
        return Err(format!(
            "{label}: reference norm {norm:e}, relative L2 {}",
            (error / norm).sqrt()
        ));
    }
    Ok(())
}

fn run(
    s: Shape,
    tile: u32,
    gpu: &Arc<blade_graphics::Context>,
    policy: CoopPolicy,
    tune: bool,
) -> Session {
    let (session, rejected) = run_split(s, tile, gpu, policy, tune, 1);
    assert!(rejected.is_empty());
    session
}

fn plan(s: Shape, tile: u32, splits: u32) -> (ExecutionPlan, Option<BufferRef>) {
    let [nx, nw, ny] = s.sizes();
    let mut graph = Graph::new();
    let x = graph.parameter("x", &[nx]);
    let w = graph.parameter("w", &[nw]);
    let y = graph.conv2d_hw(
        x, w, s.batch, s.ci, s.h, s.w, s.co, s.kh, s.kw, s.stride, s.ph, s.pw,
    );
    let dy = graph.input("dy", &[ny]);
    let weighted = graph.mul(y, dy);
    let loss = graph.sum_all(weighted);
    graph.set_outputs(vec![loss]);
    let mut plan = meganeura::compile::compile(&meganeura::autodiff::differentiate(&graph));
    let mut forced = 0;
    for d in &mut plan.dispatches {
        match d.shader {
            ShaderEntry::Conv2dGemm | ShaderEntry::Conv2dGemmSmall | ShaderEntry::Conv2dGemm16 => {
                d.shader = match tile {
                    16 => ShaderEntry::Conv2dGemm16,
                    32 => ShaderEntry::Conv2dGemmSmall,
                    _ => ShaderEntry::Conv2dGemm,
                };
                let (oh, ow) = s.output();
                d.workgroups = [(oh * ow).div_ceil(tile), s.co.div_ceil(tile), s.batch];
                forced += 1;
            }
            ShaderEntry::Conv2dGradInputGemm
            | ShaderEntry::Conv2dGradInputGemmSmall
            | ShaderEntry::Conv2dGradInputGemm16 => {
                d.shader = match tile {
                    16 => ShaderEntry::Conv2dGradInputGemm16,
                    32 => ShaderEntry::Conv2dGradInputGemmSmall,
                    _ => ShaderEntry::Conv2dGradInputGemm,
                };
                d.workgroups = [(s.h * s.w).div_ceil(tile), s.ci.div_ceil(tile), s.batch];
                forced += 1;
            }
            ShaderEntry::Conv2dGradWeightGemm
            | ShaderEntry::Conv2dGradWeightGemmSmall
            | ShaderEntry::Conv2dGradWeightGemm16 => {
                d.shader = match tile {
                    16 => ShaderEntry::Conv2dGradWeightGemm16,
                    32 => ShaderEntry::Conv2dGradWeightGemmSmall,
                    _ => ShaderEntry::Conv2dGradWeightGemm,
                };
                d.workgroups = [(s.ci * s.kh * s.kw).div_ceil(tile), s.co.div_ceil(tile), 1];
                forced += 1;
            }
            _ => {}
        }
    }
    assert_eq!(forced, 3);
    let partial = if splits > 1 {
        let index = plan
            .dispatches
            .iter()
            .position(|d| {
                matches!(
                    d.shader,
                    ShaderEntry::Conv2dGradWeightGemm
                        | ShaderEntry::Conv2dGradWeightGemmSmall
                        | ShaderEntry::Conv2dGradWeightGemm16
                )
            })
            .unwrap();
        let buffer = BufferRef(plan.buffers.len() as u32);
        let bytes = nw * splits as usize * 4;
        assert_eq!(
            plan.split_conv_weight_gradients(&[(index, splits)], bytes),
            Ok(bytes)
        );
        Some(buffer)
    } else {
        None
    };
    (plan, partial)
}

fn run_split(
    s: Shape,
    tile: u32,
    gpu: &Arc<blade_graphics::Context>,
    policy: CoopPolicy,
    tune: bool,
    splits: u32,
) -> (Session, Vec<String>) {
    let [nx, nw, ny] = s.sizes();
    let (plan, partial) = plan(s, tile, splits);
    let mut session = Session::with_context_opts(
        plan,
        Arc::clone(gpu),
        SessionOptions {
            coop: policy,
            no_alias: true, // Keep forward output available for the independent full oracle.
            ..Default::default()
        },
    );
    let cooperative = policy != CoopPolicy::Disabled;
    let half_inputs = cooperative && meganeura::runtime::auto_tune(gpu, 0).coop_caps.f32_tile == 0;
    if cooperative {
        assert!(
            session
                .plan()
                .dispatches
                .iter()
                .any(|d| d.use_coop()
                    && matches!(d.shader, ShaderEntry::Conv2dGradInputGemmCoopGen(..))),
            "generated dX kernel must actually execute"
        );
    }
    let values = |n, seed, scale| {
        let mut values = data(n, seed, scale);
        if half_inputs {
            for value in &mut values {
                *value = half::f16::from_f32(*value).to_f32();
            }
        }
        values
    };
    let x = values(nx, 7, 1.0);
    let w = values(nw, 19, 1.0);
    session.set_parameter("x", &x);
    session.set_parameter("w", &w);
    let mut searched = false;
    let mut rejected = Vec::new();
    for scale in [1.0, 1e-12] {
        if half_inputs && scale < 1.0 {
            continue;
        }
        let dy = values(ny, 37, scale);
        session.set_input("dy", &dy);
        session.step();
        session.wait();
        if tune && !searched {
            let state = |s: &Session| {
                let mut values = vec![s.adam_step_count()];
                for (i, bytes) in s.plan().buffers.iter().enumerate() {
                    let mut data = vec![0.0; bytes / 4];
                    s.read_buffer(meganeura::compile::BufferRef(i as u32), &mut data);
                    values.extend(data.into_iter().map(f32::to_bits));
                }
                values
            };
            let before = state(&session);
            let keys = session.dispatch_pipeline_keys();
            for mut options in [
                meganeura::TuneOptions {
                    max_scratch_bytes: 0,
                    ..Default::default()
                },
                meganeura::TuneOptions {
                    max_time: std::time::Duration::ZERO,
                    ..Default::default()
                },
            ] {
                options.scope = meganeura::TuneScope::Convolution;
                let report = session.tune_with(options).unwrap();
                assert_eq!(report.eligible_classes, 3, "{s:?}");
                assert_eq!(session.dispatch_pipeline_keys(), keys);
                assert_eq!(state(&session), before);
                assert_eq!(report.scratch.unwrap().retained_staging_bytes, 0);
            }
            let report = session
                .tune_with(meganeura::TuneOptions {
                    scope: meganeura::TuneScope::Convolution,
                    max_time: std::time::Duration::from_secs(60),
                    ..Default::default()
                })
                .unwrap();
            assert_eq!(report.eligible_classes, 3, "{s:?}: {report:?}");
            assert_eq!(report.outcomes.len(), 24, "{s:?}: {report:?}");
            assert!(
                report.outcomes.iter().all(|o| o.qualified
                    && o.class.conv2d.is_some()
                    && matches!(
                        o.decision,
                        meganeura::TuneDecision::KeepBaseline
                            | meganeura::TuneDecision::FasterCandidate
                    )),
                "{report:?}"
            );
            assert_eq!(report.scratch.unwrap().retained_staging_bytes, 0);
            assert_eq!(state(&session), before);
            session.step();
            session.wait();
            searched = true;
        }
        let (output, dx, dw) = reference(s, &x, &w, &dy);
        if let Some(buffer) = partial {
            let mut actual = vec![f32::NAN; nw * splits as usize];
            session.read_buffer(buffer, &mut actual);
            let (oh, ow) = s.output();
            let spatial = oh * ow;
            let tiles = (s.batch * spatial).div_ceil(16);
            for split in 0..splits {
                let first = split * (tiles / splits) + split.min(tiles % splits);
                let last = first + tiles / splits + u32::from(split < tiles % splits);
                let mut masked = dy.clone();
                for n in 0..s.batch {
                    for co in 0..s.co {
                        for hw in 0..spatial {
                            if !(first * 16..last * 16).contains(&(n * spatial + hw)) {
                                masked[((n * s.co + co) * spatial + hw) as usize] = 0.0;
                            }
                        }
                    }
                }
                let expected = reference(s, &x, &w, &masked).2;
                let offset = split as usize * nw;
                if let Err(error) = qualification(
                    &format!("{s:?}, tile={tile}, scale={scale:e}, partial {split}/{splits}"),
                    &actual[offset..offset + nw],
                    &expected,
                    scale,
                ) {
                    rejected.push(error);
                }
            }
        }
        let forward = session
            .plan()
            .dispatches
            .iter()
            .find(|d| {
                matches!(
                    d.shader,
                    ShaderEntry::Conv2dGemm
                        | ShaderEntry::Conv2dGemmSmall
                        | ShaderEntry::Conv2dGemm16
                        | ShaderEntry::Conv2dGemmCoopGen(..)
                )
            })
            .unwrap();
        assert_eq!(forward.workgroups[2], s.batch);
        let mut actual = vec![f32::NAN; ny];
        session.read_buffer(forward.output_buffer, &mut actual);
        check(&format!("{s:?}, forward"), &actual, &output, 1.0);
        for (name, expected) in [("x", dx), ("w", dw)] {
            let mut actual = vec![f32::NAN; expected.len()];
            session.read_param_grad(name, &mut actual);
            check(
                &format!("{s:?}, tile={tile}, d{name}, scale={scale:e}"),
                &actual,
                &expected,
                scale,
            );
        }
    }
    (session, rejected)
}

#[test]
fn split_weight_gradients_and_every_partial_match_full_f64_oracles() {
    let gpu = gpu();
    for (batch, ci, h, w, co, kh, kw, stride, ph, pw) in [
        (3, 5, 7, 9, 7, 2, 3, 1, 1, 0),
        (2, 17, 9, 11, 19, 3, 3, 2, 0, 1),
        (2, 65, 5, 13, 33, 2, 2, 1, 1, 0),
        (2, 3, 1, 41, 5, 1, 1, 1, 0, 0),
    ] {
        let s = Shape {
            batch,
            ci,
            h,
            w,
            co,
            kh,
            kw,
            stride,
            ph,
            pw,
        };
        let (oh, ow) = s.output();
        for tile in [32, 64] {
            run(s, tile, &gpu, CoopPolicy::Disabled, false);
            for splits in [2, 3, 7, 16]
                .into_iter()
                .filter(|&count| count <= (batch * oh * ow).div_ceil(16))
            {
                let (_, rejected) = run_split(s, tile, &gpu, CoopPolicy::Disabled, false, splits);
                assert!(rejected.is_empty(), "{rejected:?}");
            }
        }
    }
}

#[test]
fn long_split_weight_gradients_report_partial_rejections_without_relaxing_the_gate() {
    let gpu = gpu();
    let s = Shape {
        batch: 2,
        ci: 3,
        h: 1,
        w: 32771,
        co: 5,
        kh: 1,
        kw: 1,
        stride: 1,
        ph: 0,
        pw: 0,
    };
    for tile in [32, 64] {
        run(s, tile, &gpu, CoopPolicy::Disabled, false);
        let mut qualified = 0;
        for splits in [2, 3, 7, 16] {
            let (_, rejected) = run_split(s, tile, &gpu, CoopPolicy::Disabled, false, splits);
            qualified += usize::from(rejected.is_empty());
            eprintln!(
                "long split-K tile={tile}, splits={splits}, qualified={}, rejections={rejected:?}",
                rejected.is_empty()
            );
        }
        assert!(
            qualified > 0,
            "no fully qualified partition count for tile {tile}"
        );
    }
}

#[test]
fn observed_tiny_partial_error_is_rejected_by_the_original_gate() {
    assert!(
        qualification(
            "observed partial",
            &[-8.967522e-14],
            &[-8.963817608922598e-14],
            1e-12
        )
        .is_err()
    );
    for (actual, expected, scale) in [
        (f32::NAN, 1.0, 1.0),
        (1.0, f64::INFINITY, 1.0),
        (1.0, 1.0, f32::NAN),
        (0.0, 0.0, 1.0),
        (0.0, 1e-12, 1e-12),
    ] {
        assert!(qualification("gate", &[actual], &[expected], scale).is_err());
    }
}

#[test]
#[ignore = "large RGB gradients: independent f64/f32 oracles and isolated paired timing"]
fn rgb_weight_splits_against_independent_accumulation() {
    let gpu = gpu();
    let counts = [1, 64, 1024];
    for (ci, co) in [(3, 8), (8, 3)] {
        let s = Shape {
            batch: 128,
            ci,
            h: 64,
            w: 64,
            co,
            kh: 5,
            kw: 5,
            stride: 1,
            ph: 2,
            pw: 2,
        };
        let [nx, nw, ny] = s.sizes();
        let x = data(nx, 7, 1.0);
        let mut graph = Graph::new();
        let input = graph.input("x", &[nx]);
        let dy = graph.input("dy", &[ny]);
        let dw = graph.conv2d_grad_weight(dy, input, ci, s.h, s.w, co, 5, 5, 1, 2, 2);
        graph.set_outputs(vec![dw]);
        let base = meganeura::compile::compile(&graph);
        assert_eq!(base.dispatches.len(), 1);
        let mut sessions = counts.map(|splits| {
            let mut plan = base.clone();
            let partial = if splits > 1 {
                let buffer = BufferRef(plan.buffers.len() as u32);
                plan.split_conv_weight_gradients(&[(0, splits)], 64 << 20)
                    .unwrap();
                Some(buffer)
            } else {
                None
            };
            let mut session = Session::with_context_opts(
                plan,
                Arc::clone(&gpu),
                SessionOptions {
                    coop: CoopPolicy::Disabled,
                    ..Default::default()
                },
            );
            session.set_input("x", &x);
            (session, partial)
        });
        let mut qualified = [true; 3];
        for scale in [1.0, 1e-12] {
            let dy = data(ny, 37, scale);
            let mut exact = vec![0.0f64; nw];
            let mut separate = vec![0.0f32; nw];
            let mut fused = separate.clone();
            let mut partials = counts.map(|splits| vec![0.0f64; nw * splits as usize]);
            let k = (s.batch * s.h * s.w) as usize;
            for oc in 0..co as usize {
                for ic in 0..ci as usize {
                    for ky in 0..5 {
                        for kx in 0..5 {
                            let wi = ((oc * ci as usize + ic) * 5 + ky) * 5 + kx;
                            for batch in 0..128 {
                                for y in 0..64 {
                                    for z in 0..64 {
                                        let (iy, ix) =
                                            (y as i32 + ky as i32 - 2, z as i32 + kx as i32 - 2);
                                        if !(0..64).contains(&iy) || !(0..64).contains(&ix) {
                                            continue;
                                        }
                                        let a = dy[((batch * co as usize + oc) * 64 + y) * 64 + z];
                                        let b = x[((batch * ci as usize + ic) * 64 + iy as usize)
                                            * 64
                                            + ix as usize];
                                        let value = f64::from(a) * f64::from(b);
                                        exact[wi] += value;
                                        separate[wi] += a * b;
                                        fused[wi] = a.mul_add(b, fused[wi]);
                                        let position = (batch * 64 + y) * 64 + z;
                                        for variant in 1..3 {
                                            let split = position / (k / counts[variant] as usize);
                                            partials[variant][split * nw + wi] += value;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for (variant, (session, partial)) in sessions.iter_mut().enumerate() {
                session.set_input("dy", &dy);
                session.step();
                session.wait();
                let output = session.read_output(nw);
                let final_error = qualification("final", &output, &exact, scale).err();
                let mut partial_errors = Vec::new();
                if let Some(buffer) = partial {
                    let mut actual = vec![0.0; nw * counts[variant] as usize];
                    session.read_buffer(*buffer, &mut actual);
                    for split in 0..counts[variant] as usize {
                        let span = split * nw..(split + 1) * nw;
                        if let Err(error) = qualification(
                            "partial",
                            &actual[span.clone()],
                            &partials[variant][span],
                            scale,
                        ) {
                            partial_errors.push(error);
                        }
                    }
                } else {
                    for i in 0..nw {
                        assert!(
                            output[i] == separate[i] || output[i] == fused[i],
                            "{i}: GPU={}, f32 separate={}, f32 FMA={}",
                            output[i],
                            separate[i],
                            fused[i]
                        );
                    }
                }
                qualified[variant] &= final_error.is_none() && partial_errors.is_empty();
                let error2 = output
                    .iter()
                    .zip(&exact)
                    .map(|(&a, &b)| (f64::from(a) - b).powi(2))
                    .sum::<f64>();
                let norm2 = exact.iter().map(|x| x * x).sum::<f64>();
                eprintln!(
                    "{}",
                    serde_json::json!({"channels": [ci,co], "splits": counts[variant], "scale": scale,
                    "qualified": final_error.is_none() && partial_errors.is_empty(), "relative_l2": (error2 / norm2).sqrt(),
                    "final_error": final_error, "partial_rejections": partial_errors.len(), "first_partial_error": partial_errors.first(),
                    "unsplit_matches_sequential_f32": variant == 0})
                );
            }
        }
        assert!(
            qualified[1..].iter().any(|&q| q),
            "no qualified split count"
        );
        for (session, _) in &mut sessions {
            session.set_input("dy", &data(ny, 37, 1.0));
            session.step();
            session.wait();
        }
        let mut timings = [Vec::new(), Vec::new(), Vec::new()];
        for pair in 0..8 {
            for i in 0..3 {
                let variant = if pair % 2 == 0 { i } else { 2 - i };
                let start = std::time::Instant::now();
                sessions[variant].0.step();
                sessions[variant].0.wait();
                timings[variant].push(start.elapsed().as_secs_f64() * 1000.0);
            }
        }
        eprintln!(
            "{}",
            serde_json::json!({"channels": [ci,co], "counts": counts, "qualified": qualified, "wall_ms": timings})
        );
        let memory = gpu.memory_stats();
        assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
    }
}

fn training_state(session: &Session) -> Vec<(String, Vec<f32>)> {
    let loss = session.plan().loss_buffer.unwrap();
    let mut partials = vec![f32::NAN; session.plan().buffers[loss.0 as usize] / 4];
    session.read_buffer(loss, &mut partials);
    let mut state = vec![
        ("loss".into(), vec![session.read_loss()]),
        ("loss_partials".into(), partials),
    ];
    for name in ["x", "w"] {
        let n = session.param_size(name).unwrap();
        let mut parameter = vec![f32::NAN; n];
        let mut gradient = vec![f32::NAN; n];
        session.read_param(name, &mut parameter);
        session.read_param_grad(name, &mut gradient);
        state.push((format!("parameter.{name}"), parameter));
        state.push((format!("gradient.{name}"), gradient));
        if session.memory_summary().adam_state_bytes != 0 {
            let mut m = vec![f32::NAN; n];
            let mut v = vec![f32::NAN; n];
            session.read_adam_m(name, &mut m);
            session.read_adam_v(name, &mut v);
            state.push((format!("adam_m.{name}"), m));
            state.push((format!("adam_v.{name}"), v));
        }
    }
    state
}

#[test]
fn split_weight_gradients_preserve_optimizer_updates_with_and_without_aliasing() {
    optimizer_updates(false, 32);
}

#[test]
#[ignore = "GPU isolated sequence qualification/timing; requires idle device"]
fn split_sequence_measurement_preserves_state_budgets_and_subsequent_updates() {
    for tile in [16, 32, 64] {
        optimizer_updates(true, tile);
    }
}

fn optimizer_updates(measure: bool, tile: u32) {
    let gpu = gpu();
    let s = Shape {
        batch: 3,
        ci: 5,
        h: 7,
        w: 9,
        co: 7,
        kh: 2,
        kw: 3,
        stride: 1,
        ph: 1,
        pw: 0,
    };
    let [nx, nw, ny] = s.sizes();
    for adam in [false, true] {
        let mut sessions: Vec<_> = [(1, true), (3, true), (3, false)]
            .into_iter()
            .map(|(splits, no_alias)| {
                let mut session = Session::with_context_opts(
                    plan(s, tile, splits).0,
                    Arc::clone(&gpu),
                    SessionOptions {
                        coop: CoopPolicy::Disabled,
                        no_alias,
                        ..Default::default()
                    },
                );
                session.set_parameter("x", &data(nx, 7, 1.0));
                session.set_parameter("w", &data(nw, 19, 1.0));
                if adam {
                    session.set_adam(1e-4, 0.9, 0.999, 1e-8);
                } else {
                    session.set_learning_rate(1e-3);
                }
                session.set_grad_clip_norm(1.0);
                session
            })
            .collect();
        for step in 1..=4 {
            for session in &mut sessions {
                session.set_input("dy", &data(ny, 37 + step, 1.0));
                session.step();
                session.wait();
                assert_eq!(session.adam_step_count(), if adam { step } else { 0 });
                assert_eq!(
                    session.memory_summary().adam_state_bytes,
                    // Moments follow the parameter arena's 256-byte slots.
                    if adam {
                        [nx, nw]
                            .map(|n| (n * 4).next_multiple_of(256))
                            .iter()
                            .sum::<usize>()
                            * 2
                    } else {
                        0
                    }
                );
                for (name, n, seed) in [("x", nx, 7), ("w", nw, 19)] {
                    let mut actual = vec![f32::NAN; n];
                    session.read_param(name, &mut actual);
                    assert_ne!(actual, data(n, seed, 1.0), "{name} must update");
                }
            }
            let control = training_state(&sessions[0]);
            if measure && step == 2 {
                let session = &mut sessions[0];
                let index = session
                    .plan()
                    .dispatches
                    .iter()
                    .position(|d| {
                        matches!(
                            d.shader,
                            ShaderEntry::Conv2dGradWeightGemm
                                | ShaderEntry::Conv2dGradWeightGemmSmall
                                | ShaderEntry::Conv2dGradWeightGemm16
                        )
                    })
                    .unwrap();
                let keys = session.dispatch_pipeline_keys();
                let bytes = session.memory_summary().total_allocated_bytes();
                for counts in [
                    vec![],
                    vec![1],
                    vec![2, 2],
                    vec![2, 3, 4, 7, 8],
                    vec![2, u32::MAX],
                ] {
                    assert!(
                        session
                            .measure_conv_weight_splits(index, &counts, Default::default())
                            .is_err()
                    );
                }
                for options in [
                    meganeura::TuneOptions {
                        max_time: std::time::Duration::ZERO,
                        ..Default::default()
                    },
                    meganeura::TuneOptions {
                        max_classes: 0,
                        ..Default::default()
                    },
                    meganeura::TuneOptions {
                        max_scratch_bytes: 0,
                        ..Default::default()
                    },
                    meganeura::TuneOptions {
                        max_scratch_bytes: (ny + nx + nw + 2 * nw) * 4 + ny.max(nx).max(2 * nw) * 4
                            - 1,
                        ..Default::default()
                    },
                ] {
                    let report = session
                        .measure_conv_weight_splits(index, &[2, 3], options)
                        .unwrap();
                    assert!(
                        report
                            .outcomes
                            .iter()
                            .all(|o| o.decision == meganeura::TuneDecision::ScratchLimit)
                    );
                    assert_eq!(report.scratch.unwrap().peak_bytes, 0);
                }
                let report = session
                    .measure_conv_weight_splits(
                        index,
                        &[2, 3, 7, 8],
                        meganeura::TuneOptions {
                            max_time: std::time::Duration::from_secs(60),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                assert_eq!(report.outcomes.len(), 4);
                for (outcome, splits) in report.outcomes.iter().zip([2, 3, 7, 8]) {
                    assert!(outcome.qualified, "{outcome:?}");
                    assert_eq!(outcome.candidate_split_k, Some(splits));
                    assert_eq!(outcome.initial, outcome.candidate);
                    assert_eq!(outcome.selected, outcome.initial);
                    assert_eq!(outcome.baseline_ms.len(), 6);
                    assert_eq!(outcome.candidate_ms.len(), 6);
                    let scratch = outcome.scratch.as_ref().unwrap();
                    assert_eq!(
                        scratch.binding_bytes,
                        [ny * 4, nx * 4, nw * 4, nw * splits as usize * 4]
                    );
                    assert_eq!(
                        scratch.staging_bytes,
                        *scratch.binding_bytes.iter().max().unwrap()
                    );
                }
                let scratch = report.scratch.unwrap();
                assert_eq!(scratch.staging_allocations, scratch.staging_releases);
                assert_eq!(scratch.retained_staging_bytes, 0);
                assert_eq!(scratch.staging_allocations, 3);
                assert_eq!(scratch.staging_reuses, 1);
                assert_eq!(
                    scratch.peak_bytes,
                    (ny + nx + nw + 8 * nw) * 4 + (8 * nw * 4).max(ny * 4).max(nx * 4)
                );
                assert_eq!(session.dispatch_pipeline_keys(), keys);
                assert_eq!(session.memory_summary().total_allocated_bytes(), bytes);
                assert_eq!(training_state(session), control);
                assert_eq!(session.adam_step_count(), if adam { step } else { 0 });
            }
            for session in &mut sessions[1..] {
                let before = training_state(session);
                assert_eq!(before.len(), control.len());
                for ((name, actual), (other, expected)) in before.iter().zip(&control) {
                    assert_eq!(name, other);
                    check(
                        &format!("Adam={adam}, step={step}, {name}"),
                        actual,
                        &expected.iter().copied().map(f64::from).collect::<Vec<_>>(),
                        1.0,
                    );
                }
                let keys = session.dispatch_pipeline_keys();
                let report = session
                    .tune_with(meganeura::TuneOptions {
                        scope: meganeura::TuneScope::ConvDerivatives,
                        max_time: std::time::Duration::ZERO,
                        ..Default::default()
                    })
                    .unwrap();
                assert_eq!(
                    report.eligible_classes, 1,
                    "only unsplit dX is tile-tunable"
                );
                assert_eq!(session.dispatch_pipeline_keys(), keys);
                assert_eq!(training_state(session), before);
            }
            let keys: Vec<_> = sessions
                .iter()
                .map(Session::dispatch_pipeline_keys)
                .collect();
            let before: Vec<_> = sessions.iter().map(training_state).collect();
            let (control, split) = sessions.split_at_mut(1);
            for session in split {
                assert!(control[0].swap_tuning_with(session).is_err());
            }
            for (i, session) in sessions.iter().enumerate() {
                assert_eq!(session.dispatch_pipeline_keys(), keys[i]);
                assert_eq!(training_state(session), before[i]);
                assert_eq!(session.adam_step_count(), if adam { step } else { 0 });
            }
        }
        assert!(
            sessions[2].memory_summary().allocated_buffer_bytes
                < sessions[1].memory_summary().allocated_buffer_bytes
        );
    }
}

#[test]
fn scalar_conv_derivatives_match_full_oracle_across_padding_stride_and_tile_edges() {
    scalar_oracles(false);
}

#[test]
fn scalar_conv_indexing_matches_full_oracle_at_reciprocal_boundaries() {
    reciprocal_boundary_oracles(false);
}

#[test]
#[ignore = "GPU scalar convolution tuning qualification; requires idle device"]
fn tuned_conv_indexing_matches_full_oracle_at_reciprocal_boundaries() {
    reciprocal_boundary_oracles(true);
}

fn reciprocal_boundary_oracles(tune: bool) {
    let gpu = gpu();
    for divisor in [41, 47, 55] {
        // Old f32 reciprocal multiplication maps divisor/divisor to zero.
        // Cover spatial/batch boundaries, then kernel/channel decomposition.
        for (h, w, kh, kw, ph, pw) in [
            (1, divisor, 1, 1, 0, 0),
            (3, divisor + 6, 2, divisor, 1, divisor / 2),
        ] {
            for tile in [32, 64] {
                run(
                    Shape {
                        batch: 2,
                        ci: 3,
                        h,
                        w,
                        co: 5,
                        kh,
                        kw,
                        stride: 1,
                        ph,
                        pw,
                    },
                    tile,
                    &gpu,
                    CoopPolicy::Disabled,
                    tune,
                );
            }
        }
    }
}

#[test]
#[ignore = "GPU scalar convolution tuning qualification; requires idle device"]
fn tuned_conv_derivatives_match_full_oracle_and_preserve_state_and_budgets() {
    scalar_oracles(true);
}

fn scalar_oracles(tune: bool) {
    let gpu = gpu();
    for (batch, ci, h, w, co, kh, kw, stride, ph, pw) in [
        (2, 3, 5, 7, 5, 3, 3, 1, 0, 0),
        (2, 3, 7, 9, 5, 2, 4, 1, 0, 1),
        (2, 5, 5, 9, 7, 3, 2, 1, 2, 0),
        (2, 3, 7, 9, 5, 3, 2, 2, 0, 1),
        (1, 17, 9, 11, 19, 3, 3, 1, 1, 1),
        (2, 65, 5, 13, 33, 2, 2, 1, 1, 0),
        (3, 5, 7, 9, 3, 1, 1, 1, 0, 0),
        (2, 3, 9, 7, 5, 1, 3, 2, 0, 1),
    ] {
        for tile in [32, 64] {
            run(
                Shape {
                    batch,
                    ci,
                    h,
                    w,
                    co,
                    kh,
                    kw,
                    stride,
                    ph,
                    pw,
                },
                tile,
                &gpu,
                CoopPolicy::Disabled,
                tune,
            );
        }
    }
}

#[test]
#[ignore = "Requires cooperative hardware; verifies actual generated dX execution"]
fn generated_conv_derivatives_match_oracle_without_assuming_same_padding() {
    let gpu = gpu();
    let policy = cooperative_policy(&gpu);
    for (kh, kw, stride, ph, pw) in [(3, 3, 1, 0, 0), (2, 4, 1, 0, 1), (3, 2, 2, 0, 1)] {
        run(
            Shape {
                batch: 2,
                ci: 64,
                h: 8,
                w: 16,
                co: 5,
                kh,
                kw,
                stride,
                ph,
                pw,
            },
            64,
            &gpu,
            policy,
            false,
        );
    }
}

fn cooperative_policy(gpu: &blade_graphics::Context) -> CoopPolicy {
    assert!(gpu.capabilities().cooperative_matrix.is_supported());
    if meganeura::runtime::auto_tune(gpu, 0).coop_caps.f32_tile > 0 {
        CoopPolicy::Auto
    } else {
        // Exactly representable bounded operands isolate indexing on f16-only
        // hardware. This is not qualification of tiny f32 derivatives on f16.
        CoopPolicy::AllowF16
    }
}

#[test]
#[ignore = "Requires cooperative hardware; verifies generated dX and admitted forward execution"]
fn generated_conv_indexing_matches_full_oracle_at_reciprocal_boundaries() {
    let gpu = gpu();
    let policy = cooperative_policy(&gpu);
    for width in [41, 47, 55] {
        let session = run(
            Shape {
                batch: 2,
                ci: 64,
                h: 16,
                w: width,
                co: 128,
                kh: 1,
                kw: 1,
                stride: 1,
                ph: 0,
                pw: 0,
            },
            64,
            &gpu,
            policy,
            false,
        );
        // The native tile-8 policy deliberately keeps forward convolution scalar.
        if !gpu
            .capabilities()
            .cooperative_matrix
            .f32_shapes
            .contains(&[8, 8, 8])
        {
            assert!(
                session
                    .plan()
                    .dispatches
                    .iter()
                    .any(|d| d.use_coop() && matches!(d.shader, ShaderEntry::Conv2dGemmCoopGen(..)))
            );
        }
    }
}
