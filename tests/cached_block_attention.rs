use meganeura::{Graph, SessionConfig};

#[test]
fn cached_blocks_match_cpu_and_reset_excludes_stale_tokens() {
    const BLOCK: usize = 3;
    const HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const D: usize = 64;
    const FRAMES: usize = 4;
    const Q_DIM: usize = HEADS * D;
    const KV_DIM: usize = KV_HEADS * D;
    let mut g = Graph::new();
    let q = g.input("q", &[BLOCK, Q_DIM]);
    let k = g.input("k", &[1, BLOCK * KV_DIM]);
    let v = g.input("v", &[1, BLOCK * KV_DIM]);
    let frame = g.input_u32("frame", &[1]);
    let last_token = g.input_u32("last_token", &[1]);
    let k_cache = g.parameter("k_cache", &[FRAMES, BLOCK * KV_DIM]);
    let v_cache = g.parameter("v_cache", &[FRAMES, BLOCK * KV_DIM]);
    let k = g.cache_write(k, k_cache, frame);
    let v = g.cache_write(v, v_cache, frame);
    let k = g.reshape(k, &[FRAMES * BLOCK, KV_DIM]);
    let v = g.reshape(v, &[FRAMES * BLOCK, KV_DIM]);
    let out = g.cached_attention(q, k, v, last_token, HEADS as u32, KV_HEADS as u32, D as u32);
    g.set_outputs(vec![out]);
    let mut session = meganeura::build(&g, SessionConfig::inference_from_env()).0;
    let mut keys = vec![0.0; FRAMES * BLOCK * KV_DIM];
    let mut values = keys.clone();
    session.set_parameter("k_cache", &keys);
    session.set_parameter("v_cache", &values);
    for step in 0..FRAMES + 2 {
        let frame = step % FRAMES;
        let q: Vec<f32> = (0..BLOCK * Q_DIM)
            .map(|i| ((i + step * 13) as f32 * 0.021).sin())
            .collect();
        let k: Vec<f32> = (0..BLOCK * KV_DIM)
            .map(|i| ((i + step * 17) as f32 * 0.037).cos())
            .collect();
        let v: Vec<f32> = (0..BLOCK * KV_DIM)
            .map(|i| ((i + step * 19) as f32 * 0.041).sin())
            .collect();
        let offset = frame * BLOCK * KV_DIM;
        keys[offset..offset + k.len()].copy_from_slice(&k);
        values[offset..offset + v.len()].copy_from_slice(&v);
        session.set_input("q", &q);
        session.set_input("k", &k);
        session.set_input("v", &v);
        session.set_input_u32("frame", &[frame as u32]);
        session.set_input_u32("last_token", &[((frame + 1) * BLOCK - 1) as u32]);
        session.step();
        session.wait();
        let actual = session.read_output(BLOCK * Q_DIM);
        for query in 0..BLOCK {
            for head in 0..HEADS {
                let kv_head = head / (HEADS / KV_HEADS);
                let q_base = query * Q_DIM + head * D;
                let scores: Vec<f64> = (0..(frame + 1) * BLOCK)
                    .map(|token| {
                        let k_base = token * KV_DIM + kv_head * D;
                        (0..D)
                            .map(|d| f64::from(q[q_base + d]) * f64::from(keys[k_base + d]))
                            .sum::<f64>()
                            / (D as f64).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|score| (score - max).exp()).collect();
                let sum: f64 = weights.iter().sum();
                for d in 0..D {
                    let expected = weights
                        .iter()
                        .enumerate()
                        .map(|(token, weight)| {
                            weight * f64::from(values[token * KV_DIM + kv_head * D + d])
                        })
                        .sum::<f64>()
                        / sum;
                    let got = f64::from(actual[q_base + d]);
                    assert!(
                        (got - expected).abs() < 2e-5,
                        "step {step}, query {query}, head {head}, d {d}: {got} != {expected}"
                    );
                }
            }
        }
    }
}
