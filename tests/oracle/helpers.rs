//! The building blocks of `graph::helpers` against direct loops written
//! from the kernels a dedicated Lyria port used: each helper computes what
//! its single-purpose shader did, from primitives alone.

use meganeura::graph::{Nchw, t5_bucket};
use meganeura::reference::{Feeds, Rng, gpu};
use meganeura::{Graph, Mode, NodeId, SessionConfig};

/// Run `graph` on the GPU and return its first output.
fn run(graph: &Graph, feeds: &[(&str, Vec<f32>)], u32_feeds: &[(&str, Vec<u32>)]) -> Vec<f32> {
    let mut config = SessionConfig::from_env();
    config.mode = Mode::Inference;
    config.gpu = Some(gpu::shared_context());
    let (mut session, _) = meganeura::build(graph, config);
    for (name, data) in feeds {
        session.set_parameter(name, data);
    }
    for (name, data) in u32_feeds {
        session.set_input_u32(name, data);
    }
    session.step();
    session.wait();
    let mut out = vec![0.0; graph.node(graph.outputs()[0]).ty.num_elements()];
    session.read_output_by_index(0, &mut out);
    out
}

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.uniform(-2.0, 2.0)).collect()
}

/// The graph also agrees with the reference interpreter on the GPU.
fn check_reference(graph: &Graph, feeds: &[(&str, Vec<f32>)], u32_feeds: &[(&str, Vec<u32>)]) {
    let mut f = Feeds::new();
    for (name, data) in feeds {
        f.set(name, data);
    }
    for (name, data) in u32_feeds {
        f.set_u32(name, data);
    }
    gpu::check_inference(graph, &f, &gpu::Options::default())
        .unwrap()
        .assert_passed("helper vs reference");
}

#[track_caller]
fn assert_close(got: &[f32], want: &[f32], tolerance: f32) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tolerance * (1.0 + w.abs()),
            "element {i}: {g} vs {w}"
        );
    }
}

fn single(dims: usize, build: impl FnOnce(&mut Graph, NodeId) -> NodeId) -> Graph {
    let mut g = Graph::new();
    let x = g.parameter("x", &[dims]);
    let y = build(&mut g, x);
    g.set_outputs(vec![y]);
    g
}

const D: Nchw = Nchw {
    batch: 2,
    channels: 4,
    h: 3,
    w: 5,
};

fn at(d: Nchw, n: u32, c: u32, h: u32, w: u32) -> usize {
    (((n * d.channels + c) * d.h + h) * d.w + w) as usize
}

#[test]
fn elu() {
    let x = values(64, 1);
    let g = single(64, |g, x| g.elu(x));
    let want: Vec<f32> = x
        .iter()
        .map(|&v| if v > 0.0 { v } else { v.exp() - 1.0 })
        .collect();
    assert_close(&run(&g, &[("x", x.clone())], &[]), &want, 1e-5);
    check_reference(&g, &[("x", x)], &[]);
}

#[test]
fn data_movement() {
    let x = values(D.len(), 2);
    // crop_2d: slice2d.wgsl
    let g = single(D.len(), |g, x| g.crop_2d(x, D, [1, 0], [1, 2]));
    let mut want = Vec::new();
    for n in 0..D.batch {
        for c in 0..D.channels {
            for h in 1..D.h {
                for w in 1..D.w - 2 {
                    want.push(x[at(D, n, c, h, w)]);
                }
            }
        }
    }
    assert_eq!(run(&g, &[("x", x.clone())], &[]), want, "crop");
    check_reference(&g, &[("x", x.clone())], &[]);

    // pixel_shuffle_w: out[b, c, h, f·w + k] = x[b, k·C/f + c, h, w]
    let f = 2;
    let g = single(D.len(), |g, x| g.pixel_shuffle_w(x, D, f));
    let out_c = D.channels / f;
    let mut want = Vec::new();
    for b in 0..D.batch {
        for c in 0..out_c {
            for h in 0..D.h {
                for ow in 0..D.w * f {
                    want.push(x[at(D, b, (ow % f) * out_c + c, h, ow / f)]);
                }
            }
        }
    }
    assert_eq!(run(&g, &[("x", x.clone())], &[]), want, "pixel shuffle");

    // dilate_w / dilate_h: dilate_zeros_{w,h}.wgsl
    for (stride, along_w) in [(3, true), (2, false)] {
        let g = single(D.len(), |g, x| {
            if along_w {
                g.dilate_w(x, D, stride)
            } else {
                g.dilate_h(x, D, stride)
            }
        });
        let (oh, ow) = if along_w {
            (D.h, D.w * stride - (stride - 1))
        } else {
            (D.h * stride - (stride - 1), D.w)
        };
        let mut want = Vec::new();
        for n in 0..D.batch {
            for c in 0..D.channels {
                for h in 0..oh {
                    for w in 0..ow {
                        let (hit, ih, iw) = if along_w {
                            (w % stride == 0, h, w / stride)
                        } else {
                            (h % stride == 0, h / stride, w)
                        };
                        want.push(if hit { x[at(D, n, c, ih, iw)] } else { 0.0 });
                    }
                }
            }
        }
        assert_eq!(
            run(&g, &[("x", x.clone())], &[]),
            want,
            "dilate w={along_w}"
        );
        check_reference(&g, &[("x", x.clone())], &[]);
    }

    // upsample_nearest: upsample_nearest.wgsl
    let (sh, sw) = (2, 3);
    let g = single(D.len(), |g, x| g.upsample_nearest(x, D, sh, sw));
    let mut want = Vec::new();
    for n in 0..D.batch {
        for c in 0..D.channels {
            for h in 0..D.h * sh {
                for w in 0..D.w * sw {
                    want.push(x[at(D, n, c, h / sh, w / sw)]);
                }
            }
        }
    }
    assert_eq!(run(&g, &[("x", x.clone())], &[]), want, "upsample");
}

/// PyTorch's ConvTranspose2d by its definition: every input pixel scatters
/// the kernel into the output.
fn conv_transpose_direct(
    x: &[f32],
    k: &[f32],
    d: Nchw,
    co: u32,
    [kh, kw]: [u32; 2],
    [sh, sw]: [u32; 2],
    [ph, pw]: [u32; 2],
) -> (Vec<f32>, Nchw) {
    let out = Nchw {
        batch: d.batch,
        channels: co,
        h: (d.h - 1) * sh + kh - 2 * ph,
        w: (d.w - 1) * sw + kw - 2 * pw,
    };
    let mut y = vec![0.0; out.len()];
    for n in 0..d.batch {
        for ci in 0..d.channels {
            for ih in 0..d.h {
                for iw in 0..d.w {
                    let v = x[at(d, n, ci, ih, iw)];
                    for o in 0..co {
                        for a in 0..kh {
                            for b in 0..kw {
                                let oh = (ih * sh + a) as i64 - ph as i64;
                                let ow = (iw * sw + b) as i64 - pw as i64;
                                if (0..out.h as i64).contains(&oh)
                                    && (0..out.w as i64).contains(&ow)
                                {
                                    let kv = k[(((ci * co + o) * kh + a) * kw + b) as usize];
                                    y[at(out, n, o, oh as u32, ow as u32)] += v * kv;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (y, out)
}

#[test]
fn conv_transpose() {
    for (kernel, stride, padding) in [
        ([3, 4], [1, 2], [1, 1]),
        ([2, 3], [2, 3], [0, 0]),
        ([3, 3], [2, 1], [3, 2]),
    ] {
        let co = 3;
        let x = values(D.len(), 3);
        let k = values((D.channels * co * kernel[0] * kernel[1]) as usize, 4);
        let (want, out) = conv_transpose_direct(&x, &k, D, co, kernel, stride, padding);
        let mut g = Graph::new();
        let xn = g.parameter("x", &[D.len()]);
        let kn = g.parameter("k", &[k.len()]);
        let y = g.conv_transpose_2d(xn, kn, D, co, kernel, stride, padding);
        assert_eq!(g.node(y).ty.num_elements(), out.len());
        g.set_outputs(vec![y]);
        let got = run(&g, &[("x", x.clone()), ("k", k.clone())], &[]);
        assert_close(&got, &want, 1e-4);
        check_reference(&g, &[("x", x), ("k", k)], &[]);
    }
}

/// Buckets of flaxformer's relative position bias: 32 buckets out to 128
/// positions, both directions.
#[test]
fn t5_buckets() {
    let bucket = |n| t5_bucket(n, 32, 128, true);
    // 16 per direction, 8 exact: 0..8 as is, then log-spaced.
    assert_eq!(bucket(0), 0);
    assert_eq!(bucket(7), 7);
    assert_eq!(bucket(8), 8);
    // 8 + floor(ln(20/8) / ln(128/8) · 8) = 8 + floor(2.64)
    assert_eq!(bucket(20), 10);
    assert_eq!(bucket(1000), 15);
    // Keys after the query take the upper half.
    assert_eq!(bucket(-1), 17);
    assert_eq!(bucket(-20), 26);
    // One direction: all 32 buckets, 16 exact; future keys count as
    // distance zero. 16 + floor(ln(20/16) / ln(128/16) · 16) = 16 + 1.
    assert_eq!(t5_bucket(-5, 32, 128, false), 0);
    assert_eq!(t5_bucket(20, 32, 128, false), 17);
}

#[test]
fn t5_relative_bias() {
    let (heads, buckets, q_len, kv_len) = (3, 8, 5, 7);
    let table = values(heads * buckets, 5);
    let mut g = Graph::new();
    let t = g.parameter("table", &[heads, buckets]);
    let bias = g.t5_relative_bias(t, [q_len, kv_len], 16, true);
    g.set_outputs(vec![bias]);
    let mut want = Vec::new();
    for h in 0..heads {
        for i in 0..q_len {
            for j in 0..kv_len {
                let b = t5_bucket(i as i32 - j as i32, buckets as u32, 16, true) as usize;
                want.push(table[h * buckets + b]);
            }
        }
    }
    assert_eq!(run(&g, &[("table", table.clone())], &[]), want);
    check_reference(&g, &[("table", table.clone())], &[]);

    // The cached row of a query at position 4 is that position's row of the
    // full bias.
    let max_seq = 9;
    let mut g = Graph::new();
    let t = g.parameter("table", &[heads, buckets]);
    let pos = g.input_u32("pos", &[1]);
    let row = g.t5_relative_bias_cached(t, pos, max_seq, 16, false);
    g.set_outputs(vec![row]);
    let mut want = Vec::new();
    for h in 0..heads {
        for j in 0..max_seq {
            let b = t5_bucket(4 - j as i32, buckets as u32, 16, false) as usize;
            want.push(table[h * buckets + b]);
        }
    }
    let feeds = [("table", table)];
    let pos = [("pos", vec![4u32])];
    assert_eq!(run(&g, &feeds, &pos), want);
    check_reference(&g, &feeds, &pos);
}

/// A T5-style encoder layer and decode step, written with the helpers,
/// compile to the fused biased-attention kernel and nothing slower, and
/// their decompositions build the same plans.
#[test]
fn t5_blocks_take_the_fused_path() {
    use meganeura::compile::ShaderEntry;
    use meganeura::{CompileOptions, OptimizeConfig, compile_plan};
    let (rows, width, heads, dim, buckets, max_seq) = (6, 32, 4, 8, 8, 16);
    let plan = |g: &Graph| {
        compile_plan(
            g,
            Mode::Inference,
            OptimizeConfig::default(),
            &CompileOptions::default(),
        )
    };
    let check = |what: &str, g: &Graph| {
        let p = plan(g);
        let shaders: Vec<_> = p.dispatches.iter().map(|d| d.shader.clone()).collect();
        assert!(
            shaders.contains(&ShaderEntry::BiasedAttention),
            "{what}: {shaders:?}"
        );
        assert!(
            !shaders
                .iter()
                .any(|s| matches!(s, ShaderEntry::BatchMatMulBT | ShaderEntry::Permute)),
            "{what} fell back to primitive attention: {shaders:?}"
        );
        let decomposed = plan(&g.decompose());
        assert_eq!(
            p.dispatch_inventory(),
            decomposed.dispatch_inventory(),
            "{what}"
        );
        assert_eq!(p.dataflow_digest(), decomposed.dataflow_digest(), "{what}");
    };

    let mut g = Graph::new();
    let x = g.input("x", &[rows, width]);
    let norm = g.parameter("norm", &[width]);
    let h = g.rms_norm(x, norm, 1e-6);
    let [q, k, v] = ["q", "k", "v"].map(|n| {
        let w = g.parameter(n, &[width, heads * dim]);
        g.matmul(h, w)
    });
    let table = g.parameter("rel", &[heads, buckets]);
    let bias = g.t5_relative_bias(table, [rows, rows], 32, true);
    let a = g.biased_attention(
        [q, k, v, bias],
        heads as u32,
        heads as u32,
        dim as u32,
        1.0,
        false,
    );
    let o = g.parameter("o", &[heads * dim, width]);
    let y = g.matmul(a, o);
    let y = g.add(x, y);
    g.set_outputs(vec![y]);
    check("encoder layer", &g);

    let mut g = Graph::new();
    let x = g.input("x", &[1, width]);
    let pos = g.input_u32("pos", &[1]);
    let [q, k, v] = ["q", "k", "v"].map(|n| {
        let w = g.parameter(n, &[width, heads * dim]);
        g.matmul(x, w)
    });
    let k_cache = g.parameter("k_cache", &[max_seq, heads * dim]);
    let v_cache = g.parameter("v_cache", &[max_seq, heads * dim]);
    let k_cache = g.cache_write(k, k_cache, pos);
    let v_cache = g.cache_write(v, v_cache, pos);
    let table = g.parameter("rel", &[heads, buckets]);
    let bias = g.t5_relative_bias_cached(table, pos, max_seq, 32, false);
    let a = g.biased_cached_attention(
        [q, k_cache, v_cache, pos, bias],
        heads as u32,
        heads as u32,
        dim as u32,
        1.0,
    );
    g.set_outputs(vec![a]);
    check("decode step", &g);
}
