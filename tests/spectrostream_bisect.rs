//! Bisection debug for the SpectroStream all-zeros output. Each test isolates
//! a stage of the decoder pipeline and checks for non-zero values.

use meganeura::data::safetensors::SafeTensorsModel;
use meganeura::{build_inference_session, Graph};
use std::path::Path;

const DUMP: &str = "magenta_rt_codec_dump/weights_spectrostream.safetensors";

fn nonzero_count(v: &[f32]) -> (usize, f32, f32) {
    let nz = v.iter().filter(|&&x| x != 0.0 && x.is_finite()).count();
    let nan = v.iter().filter(|x| x.is_nan()).count();
    let inf = v.iter().filter(|x| x.is_infinite()).count();
    let max_abs = v.iter().filter(|x| x.is_finite()).fold(0.0_f32, |a, &x| a.max(x.abs()));
    let mean_abs = v.iter().filter(|x| x.is_finite()).map(|x| x.abs()).sum::<f32>()
        / v.iter().filter(|x| x.is_finite()).count().max(1) as f32;
    if nan > 0 || inf > 0 {
        eprintln!("  (nan={nan}, inf={inf})");
    }
    (nz, max_abs, mean_abs)
}

#[test]
#[ignore]
fn s01_input_passthrough() {
    // Sanity: input -> output as Identity, verify GPU I/O is wired right.
    let mut g = Graph::new();
    let x = g.input("x", &[100]);
    // Use add(x, x) and then halve to make sure dispatch happens.
    let two_x = g.add(x, x);
    g.set_outputs(vec![two_x]);
    let mut s = build_inference_session(&g);
    let inp: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01).collect();
    s.set_input("x", &inp);
    s.step();
    s.wait();
    let out = s.read_output(100);
    let (nz, max_abs, mean_abs) = nonzero_count(&out);
    println!("[s01] nz={nz} max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}");
    assert!(nz > 50, "input + add roundtrip should give >50 non-zero values, got {nz}");
}

#[test]
#[ignore]
fn s02_elu_roundtrip() {
    // input -> elu -> output. ELU of small non-zero values stays non-zero.
    let mut g = Graph::new();
    let x = g.input("x", &[100]);
    let y = g.elu(x);
    g.set_outputs(vec![y]);
    let mut s = build_inference_session(&g);
    let inp: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01).collect();
    s.set_input("x", &inp);
    s.step();
    s.wait();
    let out = s.read_output(100);
    let (nz, max_abs, mean_abs) = nonzero_count(&out);
    println!("[s02] nz={nz} max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}");
    assert!(nz > 50, "ELU output should be mostly non-zero, got {nz}");
}

#[test]
#[ignore]
fn s03_weights_actually_nonzero() {
    // Sanity: do the safetensors weights themselves have non-zero values?
    if !Path::new(DUMP).exists() {
        eprintln!("skipping: {DUMP} not found");
        return;
    }
    let model = SafeTensorsModel::load(DUMP.into()).unwrap();
    let names = [
        "decoder.input_layer.conv2d_3x3_a.weight_norm.rescaled.kernel",
        "decoder.input_layer.conv2d_3x3_a.weight_norm.bias",
        "decoder.decoder_0.conv2dtranspose_4x3.weight_norm.rescaled.kernel",
        "decoder.input_layer.base_conv_last.conv.kernel",
        "decoder.input_layer.base_conv_last.conv.bias",
    ];
    for name in names {
        let data = model.tensor_f32_auto(name).expect("read tensor");
        let (nz, max_abs, mean_abs) = nonzero_count(&data);
        let shape = &model.tensor_info()[name].shape;
        println!("[s03] {name:80} shape={shape:?} nz={nz}/{} max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}",
                 data.len());
        assert!(nz > 0, "{name} is all zeros");
    }
}

fn permute_3201(data: &[f32], dims: [usize; 4]) -> Vec<f32> {
    let mut out = vec![0.0_f32; data.len()];
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            for k in 0..dims[2] {
                for l in 0..dims[3] {
                    let src = ((i * dims[1] + j) * dims[2] + k) * dims[3] + l;
                    let dst = ((l * dims[2] + k) * dims[0] + i) * dims[1] + j;
                    out[dst] = data[src];
                }
            }
        }
    }
    out
}

#[test]
#[ignore]
fn s05_pixel_shuffle_w() {
    // PixelShuffleW with known input - factor=2 on [1, 4, 1, 2] -> [1, 2, 1, 4].
    let mut g = Graph::new();
    let x = g.input("x", &[8]);
    let y = g.pixel_shuffle_w(x, 1, 4, 1, 2, 2);
    g.set_outputs(vec![y]);
    let mut s = build_inference_session(&g);
    // Input channels: 0=[1,2], 1=[3,4], 2=[5,6], 3=[7,8]
    // Output [B, C/2=2, H=1, W*2=4]
    // out[b, c_out, h, factor*w + k] = in[b, k*(C/factor)+c_out=k*2+c_out, h, w]
    // For c_out=0: out[0,0,0,0]=in[0,0,0,0]=1; out[0,0,0,1]=in[0,2,0,0]=5; out[0,0,0,2]=in[0,0,0,1]=2; out[0,0,0,3]=in[0,2,0,1]=6
    // For c_out=1: out[0,1,0,0]=in[0,1,0,0]=3; out[0,1,0,1]=in[0,3,0,0]=7; out[0,1,0,2]=in[0,1,0,1]=4; out[0,1,0,3]=in[0,3,0,1]=8
    let inp = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0_f32];
    s.set_input("x", &inp);
    s.step();
    s.wait();
    let out = s.read_output(8);
    println!("[s05] PixelShuffleW out = {:?}", out);
    // Expected: c=0 row [1, 5, 2, 6], c=1 row [3, 7, 4, 8]
    assert_eq!(out, vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]);
}

#[test]
#[ignore]
fn s06_conv_transpose_hw_with_loaded_weights() {
    use meganeura::models::magenta_rt::spectrostream::SpectroStreamConfig;
    if !Path::new(DUMP).exists() {
        eprintln!("skipping: {DUMP} not found");
        return;
    }
    let model = SafeTensorsModel::load(DUMP.into()).unwrap();
    let _cfg = SpectroStreamConfig::default();

    // decoder_0's conv-transpose: TF kernel [4, 3, 1024, 512] (out_c=1024, in_c=512).
    // After permute_3201: [512, 1024, 4, 3] = PyTorch [in_c, out_c, kH, kW].
    let mut g = Graph::new();
    let in_h = 5u32; let in_w = 4u32;
    let in_c = 512u32; let out_c = 1024u32;
    let kh = 4u32; let kw = 3u32;
    let stride_h = 1u32; let stride_w = 2u32;
    let in_size = (in_c * in_h * in_w) as usize;
    let k_size = (in_c * out_c * kh * kw) as usize;
    let out_h = (in_h - 1) * stride_h + kh;
    let out_w = (in_w - 1) * stride_w + kw;
    let out_size = (out_c * out_h * out_w) as usize;

    let x = g.input("x", &[in_size]);
    let k = g.parameter(
        "decoder.decoder_0.conv2dtranspose_4x3.weight_norm.rescaled.kernel",
        &[k_size],
    );
    let b = g.parameter("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.bias", &[out_c as usize]);
    let conv = g.conv_transpose_2d_hw(x, k, 1, in_c, in_h, in_w, out_c, kh, kw, stride_h, stride_w, 0, 0);
    let with_bias = g.add_per_channel(conv, b, out_c, out_h * out_w);
    g.set_outputs(vec![with_bias]);

    let mut s = build_inference_session(&g);

    let kernel_raw = model
        .tensor_f32_auto("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.rescaled.kernel")
        .unwrap();
    let kernel = permute_3201(&kernel_raw, [kh as usize, kw as usize, out_c as usize, in_c as usize]);
    let bias = model
        .tensor_f32_auto("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.bias")
        .unwrap();
    s.set_parameter("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.rescaled.kernel", &kernel);
    s.set_parameter("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.bias", &bias);

    let inp: Vec<f32> = (0..in_size).map(|i| (i as f32 * 1e-3).sin()).collect();
    s.set_input("x", &inp);
    s.step();
    s.wait();
    let out = s.read_output(out_size);
    let (nz, max_abs, mean_abs) = nonzero_count(&out);
    println!("[s06] conv_transpose_hw nz={nz}/{} max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}", out.len());
    assert!(nz > out_size / 10, "expected mostly-nonzero output from conv_transpose_hw");
}

#[test]
#[ignore]
fn s07_full_input_layer_block() {
    use meganeura::models::magenta_rt::spectrostream::SpectroStreamConfig;
    if !Path::new(DUMP).exists() {
        eprintln!("skipping: {DUMP} not found");
        return;
    }
    let model = SafeTensorsModel::load(DUMP.into()).unwrap();
    let cfg = SpectroStreamConfig::default();
    let h_padded = 14u32; // small for testing
    let in_size = (cfg.initial_channels * h_padded * cfg.initial_freq_bins) as usize;

    let mut g = Graph::new();
    let x_input = g.input("x", &[in_size]);
    let k_a = g.parameter(
        "decoder.input_layer.conv2d_3x3_a.weight_norm.rescaled.kernel",
        &[3 * 3 * 512 * 512],
    );
    let b_a = g.parameter("decoder.input_layer.conv2d_3x3_a.weight_norm.bias", &[512]);
    let k_b = g.parameter(
        "decoder.input_layer.conv2d_3x3.weight_norm.rescaled.kernel",
        &[3 * 3 * 512 * 512],
    );
    let b_b = g.parameter("decoder.input_layer.conv2d_3x3.weight_norm.bias", &[512]);

    // Replicate input_layer's residual block.
    let h1 = g.elu(x_input);
    // conv2d_3x3_a: 512 → 512, padding_h=0, padding_w=1
    let conv_a = g.conv2d_hw(h1, k_a, 1, 512, h_padded, 5, 512, 3, 3, 1, 0, 1);
    let h_a = g.add_per_channel(conv_a, b_a, 512, (h_padded - 2) * 5);
    let h2 = g.elu(h_a);
    let conv_b = g.conv2d_hw(h2, k_b, 1, 512, h_padded - 2, 5, 512, 3, 3, 1, 0, 1);
    let h_b = g.add_per_channel(conv_b, b_b, 512, (h_padded - 4) * 5);
    // Residual: slice input by (2, 2, 0, 0)
    let residual = g.slice_2d(x_input, 1, 512, h_padded, 5, 2, 2, 0, 0);
    let out_node = g.add(h_b, residual);
    g.set_outputs(vec![out_node]);

    let mut s = build_inference_session(&g);

    for (name, dims_perm) in [
        ("decoder.input_layer.conv2d_3x3_a.weight_norm.rescaled.kernel", Some([3, 3, 512, 512])),
        ("decoder.input_layer.conv2d_3x3.weight_norm.rescaled.kernel", Some([3, 3, 512, 512])),
        ("decoder.input_layer.conv2d_3x3_a.weight_norm.bias", None),
        ("decoder.input_layer.conv2d_3x3.weight_norm.bias", None),
    ] {
        let raw = model.tensor_f32_auto(name).unwrap();
        let data = if let Some(d) = dims_perm { permute_3201(&raw, d) } else { raw };
        s.set_parameter(name, &data);
    }

    let input: Vec<f32> = (0..in_size).map(|i| (i as f32 * 1e-3).sin()).collect();
    s.set_input("x", &input);
    s.step();
    s.wait();
    let out_size = (512 * (h_padded - 4) * 5) as usize;
    let out = s.read_output(out_size);
    let (nz, max_abs, mean_abs) = nonzero_count(&out);
    println!("[s07] full input_layer nz={nz}/{} max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}", out.len());
    println!("[s07] first 8: {:?}", &out[..8]);
    assert!(nz > out_size / 10, "expected mostly-nonzero output from input_layer block");
}

#[test]
#[ignore]
fn s10_decoder_stage_sweep() {
    use meganeura::models::magenta_rt::spectrostream::{
        build_decoder_graph_through, load_decoder_weights, DecoderStage, SpectroStreamConfig,
    };
    if !Path::new(DUMP).exists() {
        eprintln!("skipping: {DUMP} not found");
        return;
    }
    let model = SafeTensorsModel::load(DUMP.into()).unwrap();
    let cfg = SpectroStreamConfig::default();
    let num_frames = 50;

    let stages = [
        ("InputLayer", DecoderStage::InputLayer),
        ("Block(0)",  DecoderStage::Block(0)),
        ("Block(1)",  DecoderStage::Block(1)),
        ("Block(2)",  DecoderStage::Block(2)),
        ("Block(3)",  DecoderStage::Block(3)),
        ("Block(4)",  DecoderStage::Block(4)),
        ("Block(5)",  DecoderStage::Block(5)),
        ("Block(6)",  DecoderStage::Block(6)),
        ("Output",    DecoderStage::Output),
    ];
    for (name, stage) in stages {
        let mut g = Graph::new();
        let out = build_decoder_graph_through(&mut g, &cfg, num_frames, stage);
        g.set_outputs(vec![out]);
        let mut s = build_inference_session(&g);
        load_decoder_weights(&model, &mut s).unwrap();
        let h_padded = num_frames + cfg.temporal_pad;
        let in_size = (cfg.initial_channels * h_padded * cfg.initial_freq_bins) as usize;
        // Use much smaller input so values stay in a normal float range.
        let input: Vec<f32> = (0..in_size).map(|i| (i as f32 * 1e-7).sin()).collect();
        s.set_input("decoder_input_preprocessed", &input);
        s.step();
        s.wait();
        let out_shape = &g.node(out).ty.shape;
        let expected = out_shape.iter().product::<usize>();
        let data = s.read_output(expected);
        let (nz, max_abs, mean_abs) = nonzero_count(&data);
        println!("[s10] {name:12} shape={out_shape:?} nz={nz}/{} max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}",
                 data.len());
    }
}

#[test]
#[ignore]
fn s11_conv_transpose_then_slice2d() {
    use meganeura::models::magenta_rt::spectrostream::SpectroStreamConfig;
    if !Path::new(DUMP).exists() {
        eprintln!("skipping: {DUMP} not found");
        return;
    }
    let model = SafeTensorsModel::load(DUMP.into()).unwrap();
    let _cfg = SpectroStreamConfig::default();

    let mut g = Graph::new();
    let in_h = 70u32; let in_w = 5u32;
    let in_c = 512u32; let out_c = 1024u32;
    let in_size = (in_c * in_h * in_w) as usize;

    let x = g.input("x", &[in_size]);
    let k = g.parameter(
        "decoder.decoder_0.conv2dtranspose_4x3.weight_norm.rescaled.kernel",
        &[4 * 3 * 1024 * 512],
    );
    let b = g.parameter("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.bias", &[1024]);
    let conv = g.conv_transpose_2d_hw(x, k, 1, in_c, in_h, in_w, out_c, 4, 3, 1, 2, 0, 0);
    let conv_h = 73u32;
    let conv_w = 11u32;
    let conv_b = g.add_per_channel(conv, b, out_c, conv_h * conv_w);
    // Slice (1, 2, 0, 1): H 73→70, W 11→10
    let sliced = g.slice_2d(conv_b, 1, out_c, conv_h, conv_w, 1, 2, 0, 1);
    g.set_outputs(vec![sliced]);

    let mut s = build_inference_session(&g);
    let kernel_raw = model
        .tensor_f32_auto("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.rescaled.kernel")
        .unwrap();
    let kernel = permute_3201(&kernel_raw, [4, 3, 1024, 512]);
    let bias = model
        .tensor_f32_auto("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.bias")
        .unwrap();
    s.set_parameter("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.rescaled.kernel", &kernel);
    s.set_parameter("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.bias", &bias);

    let inp: Vec<f32> = (0..in_size).map(|i| (i as f32 * 1e-3).sin()).collect();
    s.set_input("x", &inp);
    s.step();
    s.wait();
    let out_size = (out_c * 70 * 10) as usize;
    let out = s.read_output(out_size);
    let (nz, max_abs, mean_abs) = nonzero_count(&out);
    println!("[s11] convT→bias→slice2d nz={nz}/{} max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}",
             out.len());
    println!("[s11] first 8: {:?}", &out[..8]);
    assert!(nz > out_size / 100, "expected non-zero output from conv_transpose + slice2d");
}

#[test]
#[ignore]
fn s12_decoder_0_main_path_only() {
    // Main path of decoder_0: ELU → ConvT → slice → ELU → conv2d_3x3.
    use meganeura::models::magenta_rt::spectrostream::SpectroStreamConfig;
    if !Path::new(DUMP).exists() {
        eprintln!("skipping: {DUMP} not found");
        return;
    }
    let model = SafeTensorsModel::load(DUMP.into()).unwrap();
    let _cfg = SpectroStreamConfig::default();

    let mut g = Graph::new();
    let in_h = 70u32; let in_w = 5u32;
    let in_c = 512u32; let mid_c = 1024u32; let out_c = 1024u32;
    let in_size = (in_c * in_h * in_w) as usize;
    let x = g.input("x", &[in_size]);

    let kt = g.parameter("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.rescaled.kernel", &[4 * 3 * mid_c as usize * in_c as usize]);
    let kt_b = g.parameter("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.bias", &[mid_c as usize]);
    let k3 = g.parameter("decoder.decoder_0.conv2d_3x3.weight_norm.rescaled.kernel", &[3 * 3 * mid_c as usize * out_c as usize]);
    let k3_b = g.parameter("decoder.decoder_0.conv2d_3x3.weight_norm.bias", &[out_c as usize]);

    let h = g.elu(x);
    let conv = g.conv_transpose_2d_hw(h, kt, 1, in_c, in_h, in_w, mid_c, 4, 3, 1, 2, 0, 0);
    let conv_h = 73u32; let conv_w = 11u32;
    let conv_with_bias = g.add_per_channel(conv, kt_b, mid_c, conv_h * conv_w);
    let sliced = g.slice_2d(conv_with_bias, 1, mid_c, conv_h, conv_w, 1, 2, 0, 1);
    // After slice: (1, 1024, 70, 10)
    let act = g.elu(sliced);
    // conv2d_3x3 (padding_w=1): H 70→68, W stays at 10
    let c3 = g.conv2d_hw(act, k3, 1, mid_c, 70, 10, out_c, 3, 3, 1, 0, 1);
    let c3_b = g.add_per_channel(c3, k3_b, out_c, 68 * 10);
    g.set_outputs(vec![c3_b]);

    let mut s = build_inference_session(&g);
    for (name, dims) in [
        ("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.rescaled.kernel", Some([4, 3, 1024, 512])),
        ("decoder.decoder_0.conv2d_3x3.weight_norm.rescaled.kernel", Some([3, 3, 1024, 1024])),
        ("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.bias", None),
        ("decoder.decoder_0.conv2d_3x3.weight_norm.bias", None),
    ] {
        let raw = model.tensor_f32_auto(name).unwrap();
        let data = if let Some(d) = dims { permute_3201(&raw, d) } else { raw };
        s.set_parameter(name, &data);
    }

    let inp: Vec<f32> = (0..in_size).map(|i| (i as f32 * 1e-3).sin()).collect();
    s.set_input("x", &inp);
    s.step();
    s.wait();
    let out_size = (out_c * 68 * 10) as usize;
    let out = s.read_output(out_size);
    let (nz, max_abs, mean_abs) = nonzero_count(&out);
    println!("[s12] decoder_0 main path nz={nz}/{} max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}",
             out.len());
    println!("[s12] first 8: {:?}", &out[..8]);
}

#[test]
#[ignore]
fn s13_decoder_0_shortcut_path_only() {
    if !Path::new(DUMP).exists() {
        eprintln!("skipping: {DUMP} not found");
        return;
    }
    let model = SafeTensorsModel::load(DUMP.into()).unwrap();
    let mut g = Graph::new();
    let in_h = 70u32; let in_w = 5u32;
    let in_c = 512u32; let out_c = 1024u32;
    let in_size = (in_c * in_h * in_w) as usize;
    let x = g.input("x", &[in_size]);

    let sc = g.parameter("decoder.decoder_0.shortcut.conv1x1.weight_norm.rescaled.kernel",
                        &[1 * 1 * in_c as usize * out_c as usize]);
    let sc_b = g.parameter("decoder.decoder_0.shortcut.conv1x1.weight_norm.bias", &[out_c as usize]);
    // conv1x1: 512 → 1024
    let conv = g.conv2d_hw(x, sc, 1, in_c, in_h, in_w, out_c, 1, 1, 1, 0, 0);
    let conv_b = g.add_per_channel(conv, sc_b, out_c, in_h * in_w);
    // upsample_w by 2: W 5 → 10
    let up = g.upsample_nearest(conv_b, 1, out_c, in_h, in_w, 1, 2);
    // slice_2d (H 1, 1, 0, 0): H 70 → 68
    let sliced = g.slice_2d(up, 1, out_c, in_h, in_w * 2, 1, 1, 0, 0);
    g.set_outputs(vec![sliced]);

    let mut s = build_inference_session(&g);
    let raw = model.tensor_f32_auto("decoder.decoder_0.shortcut.conv1x1.weight_norm.rescaled.kernel").unwrap();
    let kernel = permute_3201(&raw, [1, 1, out_c as usize, in_c as usize]);
    let bias = model.tensor_f32_auto("decoder.decoder_0.shortcut.conv1x1.weight_norm.bias").unwrap();
    s.set_parameter("decoder.decoder_0.shortcut.conv1x1.weight_norm.rescaled.kernel", &kernel);
    s.set_parameter("decoder.decoder_0.shortcut.conv1x1.weight_norm.bias", &bias);

    let inp: Vec<f32> = (0..in_size).map(|i| (i as f32 * 1e-3).sin()).collect();
    s.set_input("x", &inp);
    s.step();
    s.wait();
    let out_size = (out_c * 68 * 10) as usize;
    let out = s.read_output(out_size);
    let (nz, max_abs, mean_abs) = nonzero_count(&out);
    println!("[s13] shortcut path nz={nz}/{} max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}", out.len());
    println!("[s13] first 8: {:?}", &out[..8]);
}

#[test]
#[ignore]
fn s14_decoder_0_full_block_via_helpers() {
    // Call the actual build_decoder_graph_through with stop_after=Block(0).
    // This is identical to s10's Block(0) row — so it should reproduce zero
    // output. Then we tweak: just isolate main path, just shortcut, just add,
    // to narrow down which part of the combined decoder_block destroys data.
    use meganeura::models::magenta_rt::spectrostream::{
        build_decoder_graph_through, load_decoder_weights, DecoderStage, SpectroStreamConfig,
    };
    if !Path::new(DUMP).exists() {
        eprintln!("skipping: {DUMP} not found");
        return;
    }
    let model = SafeTensorsModel::load(DUMP.into()).unwrap();
    let cfg = SpectroStreamConfig::default();
    let num_frames = 50;

    let mut g = Graph::new();
    let out = build_decoder_graph_through(&mut g, &cfg, num_frames, DecoderStage::Block(0));
    g.set_outputs(vec![out]);
    let mut s = build_inference_session(&g);
    load_decoder_weights(&model, &mut s).unwrap();
    let h_padded = num_frames + cfg.temporal_pad;
    let in_size = (cfg.initial_channels * h_padded * cfg.initial_freq_bins) as usize;
    let input: Vec<f32> = (0..in_size).map(|i| (i as f32 * 1e-4).sin()).collect();
    s.set_input("decoder_input_preprocessed", &input);
    s.step();
    s.wait();
    let data = s.read_output(696320);
    let (nz, max_abs, mean_abs) = nonzero_count(&data);
    println!("[s14] Block(0) full nz={nz}/{} max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}",
             data.len());
    println!("[s14] first 8: {:?}", &data[..8]);
}

#[test]
#[ignore]
fn s15_chained_add_after_main_and_shortcut() {
    // Reproduce decoder_0 BUT in a single test (so we eliminate possible
    // build_decoder_graph_through bugs). Replicate exactly what decoder_block does.
    if !Path::new(DUMP).exists() {
        eprintln!("skipping: {DUMP} not found");
        return;
    }
    let model = SafeTensorsModel::load(DUMP.into()).unwrap();

    let mut g = Graph::new();
    let in_h = 70u32; let in_w = 5u32;
    let in_c = 512u32; let mid_c = 1024u32; let out_c = 1024u32;
    let in_size = (in_c * in_h * in_w) as usize;
    let x = g.input("x", &[in_size]);

    let kt = g.parameter("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.rescaled.kernel", &[4 * 3 * mid_c as usize * in_c as usize]);
    let kt_b = g.parameter("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.bias", &[mid_c as usize]);
    let k3 = g.parameter("decoder.decoder_0.conv2d_3x3.weight_norm.rescaled.kernel", &[3 * 3 * mid_c as usize * out_c as usize]);
    let k3_b = g.parameter("decoder.decoder_0.conv2d_3x3.weight_norm.bias", &[out_c as usize]);
    let sc = g.parameter("decoder.decoder_0.shortcut.conv1x1.weight_norm.rescaled.kernel", &[1 * 1 * in_c as usize * out_c as usize]);
    let sc_b = g.parameter("decoder.decoder_0.shortcut.conv1x1.weight_norm.bias", &[out_c as usize]);

    // Main
    let main_h = g.elu(x);
    let main_conv = g.conv_transpose_2d_hw(main_h, kt, 1, in_c, in_h, in_w, mid_c, 4, 3, 1, 2, 0, 0);
    let main_b = g.add_per_channel(main_conv, kt_b, mid_c, 73 * 11);
    let main_sl = g.slice_2d(main_b, 1, mid_c, 73, 11, 1, 2, 0, 1);
    let main_act = g.elu(main_sl);
    let main_c3 = g.conv2d_hw(main_act, k3, 1, mid_c, 70, 10, out_c, 3, 3, 1, 0, 1);
    let main = g.add_per_channel(main_c3, k3_b, out_c, 68 * 10);

    // Shortcut
    let sc_conv = g.conv2d_hw(x, sc, 1, in_c, in_h, in_w, out_c, 1, 1, 1, 0, 0);
    let sc_post = g.add_per_channel(sc_conv, sc_b, out_c, in_h * in_w);
    let sc_up = g.upsample_nearest(sc_post, 1, out_c, in_h, in_w, 1, 2);
    let sc_sliced = g.slice_2d(sc_up, 1, out_c, in_h, in_w * 2, 1, 1, 0, 0);

    // Add main + shortcut
    let out = g.add(main, sc_sliced);
    g.set_outputs(vec![out]);

    let mut s = build_inference_session(&g);
    for (name, dims) in [
        ("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.rescaled.kernel", Some([4, 3, 1024, 512])),
        ("decoder.decoder_0.conv2d_3x3.weight_norm.rescaled.kernel", Some([3, 3, 1024, 1024])),
        ("decoder.decoder_0.shortcut.conv1x1.weight_norm.rescaled.kernel", Some([1, 1, 1024, 512])),
        ("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.bias", None),
        ("decoder.decoder_0.conv2d_3x3.weight_norm.bias", None),
        ("decoder.decoder_0.shortcut.conv1x1.weight_norm.bias", None),
    ] {
        let raw = model.tensor_f32_auto(name).unwrap();
        let data = if let Some(d) = dims { permute_3201(&raw, d) } else { raw };
        s.set_parameter(name, &data);
    }
    let inp: Vec<f32> = (0..in_size).map(|i| (i as f32 * 1e-3).sin()).collect();
    s.set_input("x", &inp);
    s.step();
    s.wait();
    let out = s.read_output((out_c * 68 * 10) as usize);
    let (nz, max_abs, mean_abs) = nonzero_count(&out);
    println!("[s15] explicit decoder_0 (main+short) nz={nz}/{} max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}",
             out.len());
    println!("[s15] first 8: {:?}", &out[..8]);
}

#[test]
#[ignore]
fn s04_one_conv2d_with_loaded_weights() {
    use meganeura::models::magenta_rt::spectrostream::SpectroStreamConfig;
    if !Path::new(DUMP).exists() {
        eprintln!("skipping: {DUMP} not found");
        return;
    }
    let model = SafeTensorsModel::load(DUMP.into()).unwrap();
    let cfg = SpectroStreamConfig::default();

    // Build a graph with just one conv2d_3x3 using input_layer's loaded weights.
    let mut g = Graph::new();
    // NCHW input: [1, 512, 12, 5] — small but nonzero spatial.
    let h = 12u32;
    let w = 5u32;
    let in_size = (1 * cfg.initial_channels * h * w) as usize;
    let x = g.input("x", &[in_size]);
    let k = g.parameter(
        "decoder.input_layer.conv2d_3x3_a.weight_norm.rescaled.kernel",
        &[3 * 3 * 512 * 512],
    );
    let b = g.parameter("decoder.input_layer.conv2d_3x3_a.weight_norm.bias", &[512]);
    // Conv2d: 512 → 512 with padding_w=1 (W stays at 5), padding_h=0 (H shrinks by 2 → 10).
    let conv = g.conv2d_hw(x, k, 1, 512, h, w, 512, 3, 3, 1, 0, 1);
    let out_h = h - 2;
    let out_w = w;
    let with_bias = g.add_per_channel(conv, b, 512, out_h * out_w);
    g.set_outputs(vec![with_bias]);

    let mut s = build_inference_session(&g);

    // Load just the two params we declared.
    let kernel_raw = model
        .tensor_f32_auto("decoder.input_layer.conv2d_3x3_a.weight_norm.rescaled.kernel")
        .unwrap();
    // TF [3, 3, 512, 512] -> meganeura [512, 512, 3, 3] via [3, 2, 0, 1] perm.
    let kernel = {
        let dims = [3, 3, 512, 512];
        let mut out = vec![0.0_f32; kernel_raw.len()];
        for i in 0..dims[0] {
            for j in 0..dims[1] {
                for kk in 0..dims[2] {
                    for l in 0..dims[3] {
                        let src = ((i * dims[1] + j) * dims[2] + kk) * dims[3] + l;
                        let dst = ((l * dims[2] + kk) * dims[0] + i) * dims[1] + j;
                        out[dst] = kernel_raw[src];
                    }
                }
            }
        }
        out
    };
    let bias = model
        .tensor_f32_auto("decoder.input_layer.conv2d_3x3_a.weight_norm.bias")
        .unwrap();
    s.set_parameter("decoder.input_layer.conv2d_3x3_a.weight_norm.rescaled.kernel", &kernel);
    s.set_parameter("decoder.input_layer.conv2d_3x3_a.weight_norm.bias", &bias);

    let input: Vec<f32> = (0..in_size).map(|i| (i as f32 * 1e-3).sin()).collect();
    s.set_input("x", &input);
    s.step();
    s.wait();
    let out_size = (512 * out_h * out_w) as usize;
    let out = s.read_output(out_size);
    let (nz, max_abs, mean_abs) = nonzero_count(&out);
    println!("[s04] nz={nz}/{} max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}", out.len());
    println!("[s04] first 8: {:?}", &out[..8]);
    assert!(nz > out_size / 10, "expected mostly-nonzero output from one conv layer");
}
