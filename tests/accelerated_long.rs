//! Scalar flash attention checked against the independent F64 oracle.
use meganeura::compile::ShaderEntry;
use meganeura::kernels::attention_grad::{AttentionGrad, Part, Path};
use meganeura::reference::{self, gpu};
use meganeura::{CoopPolicy, Graph};

fn accelerated() -> gpu::Options {
    // The measured accelerated search ships the scalar flash kernel for this
    // sequence. Cooperative forward is a separate candidate and is not the
    // selected Whisper program.
    let compile = meganeura::CompileOptions {
        flash_forward_coop: false,
        flash_backward_coop: false,
        // Accelerated search opts into f16 backward tiles. The default
        // build keeps derivative tiles in f32.
        flash_f16_tiles: true,
        ..meganeura::CompileOptions::default()
    };
    gpu::Options {
        coop: CoopPolicy::Auto,
        compile,
        tolerance: reference::Tolerance::default(),
        ..gpu::Options::default()
    }
}

fn skip_backprop() -> bool {
    // CI Metal and lavapipe skip training checks. MHA backward gradients
    // diverge on those runners.
    if std::env::var("MEGANEURA_SKIP_BACKPROP").unwrap_or_default() == "1" {
        eprintln!("MEGANEURA_SKIP_BACKPROP set — skipping");
        true
    } else {
        false
    }
}

fn software_device() -> bool {
    gpu::shared_context()
        .device_information()
        .device_name
        .to_ascii_lowercase()
        .contains("llvmpipe")
}

fn values(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 17 + 3) % 200) as f32 / 100.0 - 1.0)
        .map(|v| v * scale)
        .collect()
}

#[test]
#[ignore = "GPU: three Whisper-scale F64 backward oracles; run explicitly"]
fn accelerated_whisper_attention_matches_oracle() {
    if skip_backprop() || software_device() {
        eprintln!("skipping long accelerated oracle on a software Vulkan device");
        return;
    }
    let options = accelerated();

    let seq = 1500usize;
    let width = 6 * 64;
    let mut attention = Graph::new();
    let q = attention.parameter("q", &[seq, width]);
    let k = attention.parameter("k", &[seq, width]);
    let v = attention.parameter("v", &[seq, width]);
    let hidden = attention.full_attention(q, k, v, 6, 6, 64);
    let weights = attention.input("weights", &[seq, width]);
    let weighted = attention.mul(hidden, weights);
    let loss = attention.sum_all(weighted);
    attention.set_outputs(vec![loss]);
    let mut feeds = reference::Feeds::new();
    feeds.set("q", &values(seq * width, 0.05));
    feeds.set("k", &values(seq * width, 0.05));
    feeds.set("v", &values(seq * width, 0.05));
    feeds.set("weights", &values(seq * width, 0.02));
    let attention_report = gpu::check_training(&attention, &feeds, &options).unwrap();
    attention_report.assert_passed("whisper-scale attention");
    assert!(
        attention_report
            .shaders
            .contains(&ShaderEntry::FlashAttention),
        "auto path did not build scalar flash attention: {:?}",
        attention_report.shaders
    );
    for part in [Part::Q, Part::KV] {
        let want = ShaderEntry::AttentionGrad(AttentionGrad::new(part, Path::Flash));
        assert!(
            attention_report.shaders.contains(&want),
            "auto path did not build {want:?}: {:?}",
            attention_report.shaders
        );
    }

    // The flash shape's workgroup width is also the backward group. 128
    // threads is a measured search candidate, so it has to match the oracle.
    let mut narrow = options.clone();
    narrow.compile.knobs.flash.threads = 128;
    let narrow_report = gpu::check_training(&attention, &feeds, &narrow).unwrap();
    narrow_report.assert_passed("whisper-scale attention, 128-thread flash group");

    // The accelerated Whisper step's upstream attention gradient is near
    // 1e-5, under the f16 normal minimum. dO has to stay f32 or v_proj is zero.
    let mut small = narrow.clone();
    small.compile.knobs.flash_ept_cap = 16;
    small.compile.knobs.flash_grad_q_ept_cap = 16;
    small.compile.knobs.flash_grad_kv_ept_cap = 16;
    small.compile.knobs.flash.keys = 8;
    let mut small_feeds = reference::Feeds::new();
    small_feeds.set("q", &values(seq * width, 0.05));
    small_feeds.set("k", &values(seq * width, 0.05));
    small_feeds.set("v", &values(seq * width, 0.05));
    small_feeds.set("weights", &values(seq * width, 1.0e-5));
    let small_report = gpu::check_training(&attention, &small_feeds, &small).unwrap();
    small_report.assert_passed("whisper-scale attention, small upstream gradient");
}

#[test]
fn accelerated_peaked_unmasked_attention_matches_oracle() {
    if skip_backprop() || software_device() {
        eprintln!("skipping peaked attention oracle on a software Vulkan device");
        return;
    }
    // One key carries the softmax, and its V sits halfway between two f16
    // values. Storing that V tile as f16 misses the strict oracle. The
    // shipped forward path keeps V in f32.
    let seq = 128usize;
    let heads = 4u32;
    let head_dim = 64u32;
    let width = (heads * head_dim) as usize;
    let mut attention = Graph::new();
    let q = attention.parameter("q", &[seq, width]);
    let k = attention.parameter("k", &[seq, width]);
    let v = attention.parameter("v", &[seq, width]);
    let hidden = attention.full_attention(q, k, v, heads, heads, head_dim);
    let weights = attention.input("weights", &[seq, width]);
    let weighted = attention.mul(hidden, weights);
    let loss = attention.sum_all(weighted);
    attention.set_outputs(vec![loss]);
    let mut q_values = vec![1.0f32; seq * width];
    let mut k_values = vec![-1.0f32; seq * width];
    for lane in 0..width {
        k_values[lane] = 1.0;
    }
    // Halfway between two f16 values near 1, so a one-key row exposes V's
    // rounding instead of landing on a representable number.
    let v_values = vec![1.0 + 0.5 / 1024.0; seq * width];
    // Keep q from being an unused binding warning by a tiny per-row tilt
    // that does not move the winning key.
    for row in 0..seq {
        q_values[row * width] += 0.01;
    }
    let mut feeds = reference::Feeds::new();
    feeds.set("q", &q_values);
    feeds.set("k", &k_values);
    feeds.set("v", &v_values);
    feeds.set("weights", &vec![1.0f32; seq * width]);
    // f16 backward tiles miss this gradient tolerance: a one-key row makes
    // the dK residual about 0.5. Search may still select them when the whole
    // program qualifies. The default build keeps derivative tiles in f32.
    let mut options = accelerated();
    options.compile.flash_f16_tiles = false;
    let report = gpu::check_training(&attention, &feeds, &options).unwrap();
    assert!(
        report.shaders.contains(&ShaderEntry::FlashAttention),
        "peaked case did not build scalar flash attention: {:?}",
        report.shaders
    );
    report.assert_passed("peaked unmasked attention");
}
