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
            ShaderEntry::Conv2dGemm | ShaderEntry::Conv2dGemmSmall => {
                d.shader = if tile == 32 {
                    ShaderEntry::Conv2dGemmSmall
                } else {
                    ShaderEntry::Conv2dGemm
                };
                let (oh, ow) = s.output();
                d.workgroups = [(oh * ow).div_ceil(tile), s.co.div_ceil(tile), s.batch];
                forced += 1;
            }
            ShaderEntry::Conv2dGradInputGemm | ShaderEntry::Conv2dGradInputGemmSmall => {
                d.shader = if tile == 32 {
                    ShaderEntry::Conv2dGradInputGemmSmall
                } else {
                    ShaderEntry::Conv2dGradInputGemm
                };
                d.workgroups = [(s.h * s.w).div_ceil(tile), s.ci.div_ceil(tile), s.batch];
                forced += 1;
            }
            ShaderEntry::Conv2dGradWeightGemm | ShaderEntry::Conv2dGradWeightGemmSmall => {
                d.shader = if tile == 32 {
                    ShaderEntry::Conv2dGradWeightGemmSmall
                } else {
                    ShaderEntry::Conv2dGradWeightGemm
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
                    ShaderEntry::Conv2dGradWeightGemm | ShaderEntry::Conv2dGradWeightGemmSmall
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
    let half_inputs = cooperative && gpu.capabilities().cooperative_matrix.f32_tile == 0;
    if cooperative {
        assert!(
            session
                .plan()
                .dispatches
                .iter()
                .any(|d| d.use_coop
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
                options.scope = meganeura::TuneScope::ConvDerivatives;
                let report = session.tune_with(options).unwrap();
                assert_eq!(report.eligible_classes, 2, "{s:?}");
                assert_eq!(session.dispatch_pipeline_keys(), keys);
                assert_eq!(state(&session), before);
                assert_eq!(report.scratch.unwrap().retained_staging_bytes, 0);
            }
            let report = session
                .tune_with(meganeura::TuneOptions {
                    scope: meganeura::TuneScope::ConvDerivatives,
                    max_time: std::time::Duration::from_secs(60),
                    ..Default::default()
                })
                .unwrap();
            assert_eq!(report.outcomes.len(), 2, "{s:?}: {report:?}");
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
    let gpu = Arc::new(meganeura::init_gpu_context().unwrap());
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
    let gpu = Arc::new(meganeura::init_gpu_context().unwrap());
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
    let gpu = Arc::new(meganeura::init_gpu_context().unwrap());
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
                    plan(s, 32, splits).0,
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
                    if adam { (nx + nw) * 8 } else { 0 }
                );
                for (name, n, seed) in [("x", nx, 7), ("w", nw, 19)] {
                    let mut actual = vec![f32::NAN; n];
                    session.read_param(name, &mut actual);
                    assert_ne!(actual, data(n, seed, 1.0), "{name} must update");
                }
            }
            let control = training_state(&sessions[0]);
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
    let gpu = Arc::new(meganeura::init_gpu_context().unwrap());
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
    let gpu = Arc::new(meganeura::init_gpu_context().unwrap());
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
    let gpu = Arc::new(meganeura::init_gpu_context().unwrap());
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
    if gpu.capabilities().cooperative_matrix.f32_tile > 0 {
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
    let gpu = Arc::new(meganeura::init_gpu_context().unwrap());
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
        if gpu.capabilities().cooperative_matrix.f32_tile != 8 {
            assert!(
                session
                    .plan()
                    .dispatches
                    .iter()
                    .any(|d| d.use_coop && matches!(d.shader, ShaderEntry::Conv2dGemmCoopGen(..)))
            );
        }
    }
}
