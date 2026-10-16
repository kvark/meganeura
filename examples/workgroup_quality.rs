use meganeura::codegen::WorkgroupMatmulShape;
use meganeura::{CoopPolicy, Graph, SessionConfig};
use serde_json::json;

fn main() {
    let gpu = meganeura::reference::gpu::shared_context();
    let large = std::env::args().any(|arg| arg == "--large-tile");
    for bf16 in [false, true] {
        let tile = if large {
            WorkgroupMatmulShape {
                rows: 128,
                columns: 256,
                k_stage: if bf16 { 64 } else { 32 },
                threads: 256,
            }
        } else {
            WorkgroupMatmulShape {
                rows: 64,
                columns: 64,
                k_stage: 32,
                threads: 64,
            }
        };
        let shapes = if large {
            [
                (128, 256, 64),
                (256, 512, 128),
                (129, 256, 64),
                (128, 257, 64),
            ]
        } else {
            [(64, 64, 32), (128, 576, 576), (192, 320, 96), (65, 96, 33)]
        };
        for layout in ["nn", "at", "bt"] {
            for (m, n, k) in shapes {
                let mut graph = Graph::new();
                let a = graph.input("a", &if layout == "at" { [k, m] } else { [m, k] });
                let b = graph.parameter("b", &if layout == "bt" { [n, k] } else { [k, n] });
                let (aa, bb) = if bf16 {
                    (graph.to_bf16(a), graph.to_bf16(b))
                } else {
                    (a, b)
                };
                let y = match layout {
                    "at" => graph.matmul_at(aa, bb),
                    "bt" => graph.matmul_bt(aa, bb),
                    _ => graph.matmul(aa, bb),
                };
                graph.set_outputs(vec![y]);
                for policy in [CoopPolicy::AllowF16, CoopPolicy::NativeF32] {
                    let mut config = SessionConfig::inference_from_env_on(gpu.clone());
                    config.tune = false;
                    config.runtime.coop = policy;
                    config.runtime.poison = true;
                    config.runtime.workgroup_matmul = Some(tile);
                    let (mut session, _) = meganeura::build(&graph, config);
                    let keys = session.dispatch_pipeline_keys();
                    let selected = keys.iter().any(|key| key.contains(":workgroup-"));
                    assert_eq!(
                        selected,
                        policy == CoopPolicy::AllowF16
                            && tile.supports(
                                &gpu.capabilities(),
                                if bf16 {
                                    meganeura::codegen::MatrixTypes::BF16
                                } else {
                                    meganeura::codegen::MatrixTypes::F32_AS_F16
                                }
                            )
                            && tile.fits(m as u32, n as u32, k as u32)
                    );
                    let f16 = !bf16
                        && keys
                            .iter()
                            .any(|key| key.contains(":workgroup-") || key.contains(":cooperative"));
                    for seed in [17u32, 8123] {
                        let mut state = seed;
                        let mut sample = |i: usize| {
                            state ^= state << 13;
                            state ^= state >> 17;
                            state ^= state << 5;
                            let value = (state as f64 / u32::MAX as f64 * 2. - 1.) as f32;
                            if bf16 {
                                value * 10f32.powi((i % 7) as i32 * 3 - 9)
                            } else {
                                value
                            }
                        };
                        let a: Vec<f32> = (0..m * k).map(&mut sample).collect();
                        let b: Vec<f32> = (0..n * k).map(&mut sample).collect();
                        session.set_input("a", &a);
                        session.set_parameter("b", &b);
                        session.step();
                        session.wait();
                        let actual = session.read_output(m * n);
                        let round = |x: f32| {
                            f64::from(if bf16 {
                                half::bf16::from_f32(x).to_f32()
                            } else if f16 {
                                half::f16::from_f32(x).to_f32()
                            } else {
                                x
                            })
                        };
                        let mut squared_error = 0.;
                        let mut squared_ref = 0.;
                        let mut max_scaled_error = 0f64;
                        for r in 0..m {
                            for c in 0..n {
                                let mut expected = 0.;
                                let mut magnitude = 0.;
                                for t in 0..k {
                                    let product = round(
                                        a[if layout == "at" { t * m + r } else { r * k + t }],
                                    ) * round(
                                        b[if layout == "bt" { c * k + t } else { t * n + c }],
                                    );
                                    expected += product;
                                    magnitude += product.abs();
                                }
                                let x = f64::from(actual[r * n + c]);
                                let error = (x - expected).abs();
                                assert!(
                                    x.is_finite() && error <= 8e-6 * magnitude + 1e-5,
                                    "{bf16} {layout} {m} {n} {k} {policy:?}: {x} != {expected}, magnitude {magnitude}"
                                );
                                squared_error += error * error;
                                squared_ref += expected * expected;
                                max_scaled_error =
                                    max_scaled_error.max(error / (magnitude + 1e-30));
                            }
                        }
                        println!(
                            "{}",
                            json!({"shape":[m,n,k],"bf16":bf16,"layout":layout,"policy":format!("{policy:?}"),"seed":seed,"selected":selected,"f16_rounding":f16,"values_checked":actual.len(),"relative_l2":(squared_error/squared_ref).sqrt(),"max_scaled_error":max_scaled_error,"pipelines":keys,"device":gpu.device_information().device_name})
                        );
                    }
                }
            }
        }
    }
}
