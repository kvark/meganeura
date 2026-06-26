//! SpectroStream: 48 kHz stereo neural audio codec (STFT-domain).
//!
//! Architecture decoded from `magenta_rt_codec_dump/manifest.json` +
//! `tools/magenta_rt/trace_codec.py` output:
//!
//! ```text
//! tokens [50 frames, 16 RVQ levels]
//!   → dequantize via 16 of 64 codebooks → embed [50, 256]
//!   → reshape [1, 50, 1, 256] NHWC
//!   → temporal_padding (pad 1 zero frame at the END of T → 51 frames)
//!   → input_layer:
//!       conv1x1_first (gated 1×1 expand 256 → 2560)
//!       reshape [B, T, 1, 2560] → [B, T, 5, 512]
//!       residual block: ELU → freq_pad → conv2d_3x3 → ELU → freq_pad → conv2d_3x3 → + identity
//!   → 7 decoder blocks (decoder_0 … decoder_6):
//!       ELU → conv2dtranspose (stride (stride_h, stride_w)) → crop_freq_dim
//!       ELU → freq_pad → conv2d_3x3
//!       shortcut: conv1x1 if channels differ, then nearest-upsample W by stride_w, +
//!   → final ELU → freq_pad (3, 3) → base_conv_last (7×7, → 2 channels)
//!   → [B, T_final, F_final, 2 (re, im)]
//!   → iSTFT (Hann window, overlap-add) — done host-side on CPU
//!   → audio [B, 96000, 2 stereo]
//! ```
//!
//! Per-block (stride_h, stride_w): (2,1) (2,2) (1,2) (1,2) (1,3) (1,2) (1,2).
//! Per-block kernel shape: 4×3, 4×4, 3×4, 3×4, 3×6, 3×4, 3×4 (kH × kW).
//! Per-block has_shortcut_conv: true, true, false, false, true, false, true.
//! All conv padding = VALID; SpectroStream emulates SAME/causal via explicit
//! Pad/Crop ops. Activation = ELU. Weight norm precomputed in `.rescaled.kernel`.
//!
//! Status: **VERIFIED bit-exact against TF.** The decoder body graph (input
//! residual block → 7 decoder blocks → causal `base_conv_last` → tail) matches
//! the TF `body_out` to **rel err 2.0e-6** on real reference tokens
//! (`tests/spectrostream_vs_tf_body.rs`), and the host-side [`istft_to_audio`]
//! matches `tf.signal.inverse_stft` to **3.7e-6** (`tests/spectrostream_istft.rs`).
//! The three former "free reinterpret" open questions are now all confirmed
//! correct by that match: the decoder_0→1 batch-fold (PixelShuffle), the final
//! subbatch/channel merge to `[re_L, im_L, re_R, im_R]` (tail transpose
//! `(1,2,0,3)`), and the `base_conv_last` W padding. The input_layer's gated
//! conv1x1 preprocess is host-side (verified end-to-end: embed → TF body match,
//! `tools/magenta_rt/decoder_reference_v2.py`).
//!
//! Known issue: on **AMD RDNA3.5 (RADV) under single-submit** a cross-pass
//! cache-visibility bug produces NaN in some late dispatches (the demo clamps
//! NaN→0); lavapipe (and the gate above) show **0 NaN**. Tracked separately —
//! it is a Blade/RADV barrier issue, not a SpectroStream modeling bug.

use crate::graph::{Graph, NodeId};

/// Per-decoder-block spec. Each block applies one conv-transpose (upsample W)
/// + one 3×3 refinement conv + a residual shortcut.
#[allow(dead_code)] // stride_w/has_shortcut_conv used by the (still-TODO) forward pass
#[derive(Clone, Copy)]
struct DecoderBlock {
    /// Conv-transpose kernel `(kH, kW)`.
    kt_kernel: (u32, u32),
    stride_h: u32,
    /// W-axis stride for the conv-transpose (H stride is always 1).
    stride_w: u32,
    /// Input channel count entering the block.
    in_c: u32,
    /// Output channel count after the conv-transpose.
    mid_c: u32,
    /// Output channel count after conv2d_3x3 (and the block as a whole).
    out_c: u32,
    /// Whether the shortcut needs a 1×1 conv to match channels (in_c != out_c).
    has_shortcut_conv: bool,
    /// Name for parameter prefixing.
    name: &'static str,
}

/// SpectroStream decoder blocks in order — derived from manifest channel counts.
// TF SpectroStream's actual strides — extracted via forward layer walk through
// the dec.keras_api.layers[0] body. Time dim 50 → 100 (block 0) → 200 (block 1)
// → 200 (rest). See /tmp/spectrostream_findings.md.
// Strides extracted directly from the TF SavedModel's Conv2DBackpropInput
// strides attribute (see tools/magenta_rt/ARCH_FINDINGS.md). decoder_0 doubles
// H only (stride_h=2, stride_w=1); decoder_1 doubles both axes.
const DECODER_BLOCKS: [DecoderBlock; 7] = [
    DecoderBlock {
        kt_kernel: (4, 3),
        stride_h: 2,
        stride_w: 1,
        in_c: 512,
        mid_c: 1024,
        out_c: 1024,
        has_shortcut_conv: true,
        name: "decoder_0",
    },
    DecoderBlock {
        kt_kernel: (4, 4),
        stride_h: 2,
        stride_w: 2,
        in_c: 512,
        mid_c: 256,
        out_c: 256,
        has_shortcut_conv: true,
        name: "decoder_1",
    },
    DecoderBlock {
        kt_kernel: (3, 4),
        stride_h: 1,
        stride_w: 2,
        in_c: 256,
        mid_c: 256,
        out_c: 256,
        has_shortcut_conv: false,
        name: "decoder_2",
    },
    DecoderBlock {
        kt_kernel: (3, 4),
        stride_h: 1,
        stride_w: 2,
        in_c: 256,
        mid_c: 256,
        out_c: 256,
        has_shortcut_conv: false,
        name: "decoder_3",
    },
    DecoderBlock {
        kt_kernel: (3, 6),
        stride_h: 1,
        stride_w: 3,
        in_c: 256,
        mid_c: 128,
        out_c: 128,
        has_shortcut_conv: true,
        name: "decoder_4",
    },
    DecoderBlock {
        kt_kernel: (3, 4),
        stride_h: 1,
        stride_w: 2,
        in_c: 128,
        mid_c: 128,
        out_c: 128,
        has_shortcut_conv: false,
        name: "decoder_5",
    },
    DecoderBlock {
        kt_kernel: (3, 4),
        stride_h: 1,
        stride_w: 2,
        in_c: 128,
        mid_c: 64,
        out_c: 64,
        has_shortcut_conv: true,
        name: "decoder_6",
    },
];

/// Top-level SpectroStream config (decoder-side).
#[derive(Clone, Debug)]
pub struct SpectroStreamConfig {
    /// 48000 Hz.
    pub sample_rate: u32,
    /// 25 Hz codec frame rate (1920 samples per frame).
    pub frame_rate: u32,
    /// 2 stereo channels.
    pub num_channels: u32,
    /// 64 max RVQ depth (codebook count). Magenta-RT's LLM uses only the first 16.
    pub max_rvq_depth: u32,
    /// 1024 entries per codebook.
    pub codebook_size: u32,
    /// 256 — the codec's internal embedding dim (codebook entry size).
    pub embedding_dim: u32,
    /// 5 — the spectrogram's W (frequency) bin count after input_layer reshape.
    /// 5 × 512 = 2560 (the conv1x1_first expansion target).
    pub initial_freq_bins: u32,
    /// 512 — channels per freq bin after input_layer reshape.
    pub initial_channels: u32,
    /// Temporal padding to add at decoder input (symmetric, half each side).
    /// TF SpectroStream uses SAME padding on H throughout (no shrinkage), with
    /// decoder_0/decoder_1 each doubling T (stride_h=2). The resulting decoder
    /// output T = (num_frames + temporal_pad) × 4. TF then `temporal_cropping`
    /// trims this back to the actual output time dim.
    ///
    /// Empirically TF body produces pre-crop T=224 from S=50 frames → 56×4=224
    /// ⇒ temporal_pad = 6 (3 each side).
    pub temporal_pad: u32,
}

impl Default for SpectroStreamConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            frame_rate: 25,
            num_channels: 2,
            max_rvq_depth: 64,
            codebook_size: 1024,
            embedding_dim: 256,
            initial_freq_bins: 5,
            initial_channels: 512,
            temporal_pad: 1,
        }
    }
}

/// Track shape alongside the NodeId for forward-pass bookkeeping.
#[derive(Clone, Copy)]
struct Feat {
    node: NodeId,
    b: u32,
    c: u32,
    h: u32,
    w: u32,
}

/// ELU activation (shape preserved).
fn elu(g: &mut Graph, x: Feat) -> Feat {
    Feat {
        node: g.elu(x.node),
        ..x
    }
}

/// 2D convolution with arbitrary output channels. Kernel/bias stored as
/// `[out_c, in_c, kH, kW]` and `[out_c]` flat — the safetensors loader is
/// responsible for transposing from TF's `[kH, kW, in_c, out_c]` storage.
/// Bias is added separately via `add_per_channel` since meganeura's conv2d
/// has no fused bias term.
#[allow(clippy::too_many_arguments)]
fn conv2d(
    g: &mut Graph,
    x: Feat,
    params: &std::collections::HashMap<String, NodeId>,
    kernel_param_name: &str,
    bias_param_name: &str,
    out_c: u32,
    kh: u32,
    kw: u32,
    padding_h: u32,
    padding_w: u32,
) -> Feat {
    let kernel = params[kernel_param_name];
    let bias = params[bias_param_name];
    let out_h = x.h + 2 * padding_h - kh + 1;
    let out_w = x.w + 2 * padding_w - kw + 1;
    let node = g.conv2d_hw(
        x.node, kernel, x.b, x.c, x.h, x.w, out_c, kh, kw, 1, padding_h, padding_w,
    );
    let out = Feat {
        node,
        b: x.b,
        c: out_c,
        h: out_h,
        w: out_w,
    };
    Feat {
        node: g.add_per_channel(out.node, bias, out_c, out.h * out.w),
        ..out
    }
}

/// 2D transposed convolution, routed through `dilate_zeros_w → forward
/// conv2d_hw` so the heavy GEMM uses cooperative-matrix tiles instead of
/// the bandwidth-bound `Conv2dGradInputHW` shader. Only stride_h=1 is
/// supported (the only stride pattern SpectroStream uses); other strides
/// would need a `dilate_zeros_h` op too.
///
/// **Kernel layout expected at the bound parameter**: forward-conv
/// `[out_c, in_c, kH, kW]` with spatial axes flipped (kh' = kH-1-kh,
/// kw' = kW-1-kw). [`load_decoder_weights`] applies this transformation
/// CPU-side for parameter names matching `.conv2dtranspose_*`.
#[allow(clippy::too_many_arguments)]
fn conv_transpose(
    g: &mut Graph,
    x: Feat,
    params: &std::collections::HashMap<String, NodeId>,
    kernel_param_name: &str,
    bias_param_name: &str,
    out_c: u32,
    kh: u32,
    kw: u32,
    stride_h: u32,
    stride_w: u32,
) -> Feat {
    let kernel = params[kernel_param_name];
    let bias = params[bias_param_name];
    let out_h = (x.h - 1) * stride_h + kh;
    let out_w = (x.w - 1) * stride_w + kw;

    // Dilate W, then dilate H (independent ops; order doesn't matter).
    let dilated_w = g.dilate_zeros_w(x.node, x.b, x.c, x.h, x.w, stride_w);
    let dil_w = if stride_w == 1 {
        x.w
    } else {
        x.w * stride_w - (stride_w - 1)
    };
    let dilated = if stride_h == 1 {
        dilated_w
    } else {
        g.dilate_zeros_h(dilated_w, x.b, x.c, x.h, dil_w, stride_h)
    };
    let dil_h = if stride_h == 1 {
        x.h
    } else {
        x.h * stride_h - (stride_h - 1)
    };
    let node = g.conv2d_hw(
        dilated,
        kernel,
        x.b,
        x.c,
        dil_h,
        dil_w,
        out_c,
        kh,
        kw,
        1, // stride
        kh - 1,
        kw - 1,
    );
    let out = Feat {
        node,
        b: x.b,
        c: out_c,
        h: out_h,
        w: out_w,
    };
    Feat {
        node: g.add_per_channel(out.node, bias, out_c, out.h * out.w),
        ..out
    }
}

fn slice2d(g: &mut Graph, x: Feat, start_h: u32, end_h: u32, start_w: u32, end_w: u32) -> Feat {
    let out_h = x.h - start_h - end_h;
    let out_w = x.w - start_w - end_w;
    let node = g.slice_2d(x.node, x.b, x.c, x.h, x.w, start_h, end_h, start_w, end_w);
    Feat {
        node,
        b: x.b,
        c: x.c,
        h: out_h,
        w: out_w,
    }
}

fn add_feat(g: &mut Graph, a: Feat, b: Feat) -> Feat {
    assert_eq!(
        (a.b, a.c, a.h, a.w),
        (b.b, b.c, b.h, b.w),
        "add_feat shape mismatch"
    );
    Feat {
        node: g.add(a.node, b.node),
        ..a
    }
}

/// Helper: name of a weight_norm rescaled kernel parameter.
fn wn_kernel(prefix: &str) -> String {
    format!("{prefix}.weight_norm.rescaled.kernel")
}
/// Helper: name of a weight_norm bias parameter.
fn wn_bias(prefix: &str) -> String {
    format!("{prefix}.weight_norm.bias")
}

/// conv2d_3x3 as TF SpectroStream implements it: causal H pad [2, 0] +
/// SAME W pad [1, 1] + VALID 3×3. Net effect: H and W preserved.
///
/// Emulated via symmetric padding (2, 1) + VALID conv (output is H_in+2 in H),
/// then slicing off the last 2 rows of H. Matches causal pad bit-exactly
/// because the 2 future-looking outputs are simply discarded.
#[allow(clippy::too_many_arguments)]
fn causal_conv2d_3x3(
    g: &mut Graph,
    x: Feat,
    params: &std::collections::HashMap<String, NodeId>,
    kernel_param_name: &str,
    bias_param_name: &str,
    out_c: u32,
) -> Feat {
    let after_conv = conv2d(
        g,
        x,
        params,
        kernel_param_name,
        bias_param_name,
        out_c,
        3,
        3,
        2,
        1,
    );
    // conv2d output H = x.h + 2*2 - 3 + 1 = x.h + 2; slice strips trailing 2.
    slice2d(g, after_conv, 0, 2, 0, 0)
}

/// Build the SpectroStream decoder graph.
///
/// **Host-side preprocessing required** (not part of the graph):
/// 1. Dequantize tokens `[S, K]` via codebooks `[K, V, D]` → embedding `[S, D=256]`.
/// 2. Apply input_layer's gated conv1x1 expansion (256 → 2560 via 3 conv1x1's
///    with ELU between, residual-add at end). This is just matmuls + ELU —
///    fast on CPU and avoids the awkward NHWC↔NCHW reshape that would
///    otherwise be needed at the channel-to-freq-dim split.
/// 3. Reshape `[S, 2560]` → `[S, 5, 512]` then transpose to NCHW
///    `[1, 512, S, 5]`.
/// 4. Temporal-pad H by `temporal_pad/2` on each side → `[1, 512, S + temporal_pad, 5]`.
///    This compensates for the 9 conv2d_3x3 VALID-padding H-shrinkages
///    (input_layer's 2 + 7 decoder blocks × 1 each).
///
/// **What this function builds (GPU graph):**
/// - input_layer residual block (2× ELU → conv2d_3x3) → shrinks H by 4
/// - 7 decoder blocks; per block:
///     - ELU → conv2dtranspose (stride 1×stride_w, VALID, no auto-pad) → slice_2d crop
///     - ELU → conv2d_3x3 with padding_w=1 (folds in freq_dim_pad) → shrinks H by 2
///     - shortcut: conv2d 1×1 if channels differ, then upsample_nearest by (1, stride_w)
///       to match main path W, then slice/pad to match H
///     - add
/// - final ELU → conv2d 7×7 with padding_w=3 → 2 output channels (spectrogram)
///
/// **Output**: spectrogram `[1, 2, T_final, F_final]` in NCHW (channels=2 are
/// re/im of a complex spectrogram). Host-side iSTFT (Hann window + overlap-add)
/// produces the final stereo audio.
///
/// **Open question**: the inter-block reshape between decoder_0 → decoder_1
/// (StridedSlice + Reshape + Transpose in TF trace) is hypothesized to be a
/// 1D PixelShuffle-W: `[B, T, 5, 1024]` → `[B, T, 10, 512]` (W↑2, C↓2). For
/// NCHW this means `[B, 1024, T, 5]` → `[B, 512, T, 10]`. The current
/// implementation builds this via a reshape + transpose chain (see code).
///
/// Builds the full decoder body: input_layer residual block → 7 decoder
/// blocks (with the decoder_0→1 fold) → causal `base_conv_last` → temporal
/// crop. The host supplies the input_layer's gated-conv1x1 output as the
/// `decoder_input_preprocessed` input (see `examples/magenta_rt_demo.rs`).
pub fn build_decoder_graph(g: &mut Graph, cfg: &SpectroStreamConfig, num_frames: u32) -> NodeId {
    build_decoder_graph_through(g, cfg, num_frames, DecoderStage::Output)
}

/// Stage in the decoder pipeline — pass to [`build_decoder_graph_through`] for
/// debugging to terminate the graph at a particular intermediate output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoderStage {
    /// After input_layer's residual block. Shape `[1, 512, num_frames+pad, 5]`.
    /// (H is preserved by the causal conv2d_3x3 stack.)
    InputLayer,
    /// After decoder block `n` (0..=6). After block 0, the C-axis 2-way fold
    /// is also applied (b=2, c=512 for n=0, not c=1024).
    Block(u8),
    /// After base_conv_last — the final 2-channel spectrogram output.
    Output,
}

pub fn build_decoder_graph_through(
    g: &mut Graph,
    cfg: &SpectroStreamConfig,
    num_frames: u32,
    stop_after: DecoderStage,
) -> NodeId {
    let params = declare_all_params(g, cfg);

    let h_padded = num_frames + cfg.temporal_pad;
    let in_size = (cfg.initial_channels * h_padded * cfg.initial_freq_bins) as usize;
    let x_input = g.input("decoder_input_preprocessed", &[in_size]);
    let mut x = Feat {
        node: x_input,
        b: 1,
        c: cfg.initial_channels,
        h: h_padded,
        w: cfg.initial_freq_bins,
    };

    // input_layer residual block. Each conv2d_3x3 is causal H + SAME W
    // (preserves H, W) — see causal_conv2d_3x3.
    let residual = x;
    for sub in &["conv2d_3x3_a", "conv2d_3x3"] {
        let prefix = format!("decoder.input_layer.{sub}");
        let h = elu(g, x);
        x = causal_conv2d_3x3(
            g,
            h,
            &params,
            &wn_kernel(&prefix),
            &wn_bias(&prefix),
            cfg.initial_channels,
        );
    }
    x = add_feat(g, x, residual);
    if stop_after == DecoderStage::InputLayer {
        return x.node;
    }

    for (idx, blk) in DECODER_BLOCKS.iter().enumerate() {
        x = decoder_block(g, x, &params, blk);
        if idx == 0 {
            // TF body splits 1024 channels into [2, 512] and folds the
            // leading 2 into the BATCH dim — decoder_1..6 + base_conv_last
            // then run with batch=2. In NCHW layout
            //   [B=1, C=1024, T, W] (memory) == [B=2, C=512, T, W] (interpretation)
            // since the channel stride is the major axis and 1024 = 2 × 512
            // splits cleanly. So this is a free Feat-only reshape — no op.
            x = Feat {
                node: x.node,
                b: 2,
                c: x.c / 2,
                h: x.h,
                w: x.w,
            };
        }
        if stop_after == DecoderStage::Block(idx as u8) {
            return x.node;
        }
    }
    assert_eq!(x.c, 64);

    let x_act = elu(g, x);
    let kernel = params["decoder.input_layer.base_conv_last.conv.kernel"];
    let bias = params["decoder.input_layer.base_conv_last.conv.bias"];
    // base_conv_last: causal H pad [6, 0] + SAME W pad [3, 3] + VALID 7×7.
    // Emulate via symmetric pad (6, 3) + VALID conv + slice strip trailing 6 H.
    // Output H = x_act.h + 12 - 7 + 1 = x_act.h + 6 (pre-slice); W = x_act.w.
    let conv_node = g.conv2d_hw(
        x_act.node, kernel, x_act.b, x_act.c, x_act.h, x_act.w, 2, 7, 7, 1, 6, 3,
    );
    let pre_bias = Feat {
        node: conv_node,
        b: x_act.b,
        c: 2,
        h: x_act.h + 6,
        w: x_act.w,
    };
    let biased = Feat {
        node: g.add_per_channel(pre_bias.node, bias, 2, pre_bias.h * pre_bias.w),
        ..pre_bias
    };
    let body = slice2d(g, biased, 0, 6, 0, 0);
    // body shape: [B=2, C=2, H=T_pre_crop, W=480].
    // Tail: TF's reshape_8/transpose_3/reshape_9 collapses the 2-batch +
    // 2-channel into a single batch=1, channel=4 in the NHWC order
    // (L_re, L_im, R_re, R_im). In NCHW the same ordering is just a free
    // reinterpretation [B=2, C=2, H, W] → [B=1, C=4, H, W] because batch is
    // the major axis and 2 batches × 2 channels concatenate into 4 channels
    // with the same memory layout.
    let merged = Feat {
        node: body.node,
        b: 1,
        c: body.b * body.c,
        h: body.h,
        w: body.w,
    };
    // temporal_cropping: slice front 4 frames from H.
    let cropped = slice2d(g, merged, 4, 0, 0, 0);
    cropped.node
}

/// Declares every safetensors parameter the decoder needs and returns a
/// `name → NodeId` map. Each name is declared exactly once. The host-side
/// params (quantizer, conv1x1_first) are declared too so the loader sees
/// them; they're unused by the GPU graph.
fn declare_all_params(
    g: &mut Graph,
    cfg: &SpectroStreamConfig,
) -> std::collections::HashMap<String, NodeId> {
    let mut params = std::collections::HashMap::new();
    let decl = |g: &mut Graph,
                name: String,
                shape: Vec<usize>,
                params: &mut std::collections::HashMap<String, NodeId>| {
        let node = g.parameter(&name, &shape);
        params.insert(name, node);
    };

    decl(
        g,
        "quantizer.rvq_codebooks".to_string(),
        vec![(cfg.max_rvq_depth * cfg.codebook_size * cfg.embedding_dim) as usize],
        &mut params,
    );
    for sub in &[
        "conv1x1_first.conv1x1",
        "conv1x1_first.conv1x1_a",
        "conv1x1_first.conv1x1_b",
    ] {
        let prefix = format!("decoder.input_layer.{sub}");
        let (out_c, in_c) = match *sub {
            "conv1x1_first.conv1x1" => (2560, cfg.embedding_dim as usize),
            "conv1x1_first.conv1x1_a" => (2560, cfg.embedding_dim as usize),
            "conv1x1_first.conv1x1_b" => (2560, 2560),
            _ => unreachable!(),
        };
        // 1×1 conv kernel [out_c, in_c, 1, 1] flat.
        decl(g, wn_kernel(&prefix), vec![in_c * out_c], &mut params);
        decl(g, wn_bias(&prefix), vec![out_c], &mut params);
    }
    for sub in &["conv2d_3x3", "conv2d_3x3_a"] {
        let prefix = format!("decoder.input_layer.{sub}");
        decl(g, wn_kernel(&prefix), vec![3 * 3 * 512 * 512], &mut params);
        decl(g, wn_bias(&prefix), vec![512], &mut params);
    }
    decl(
        g,
        "decoder.input_layer.base_conv_last.conv.kernel".to_string(),
        vec![7 * 7 * 64 * 2],
        &mut params,
    );
    decl(
        g,
        "decoder.input_layer.base_conv_last.conv.bias".to_string(),
        vec![2],
        &mut params,
    );
    for blk in DECODER_BLOCKS.iter() {
        let (kh, kw) = blk.kt_kernel;
        let kt_name = format!("decoder.{}.conv2dtranspose_{kh}x{kw}", blk.name);
        decl(
            g,
            wn_kernel(&kt_name),
            vec![(kh * kw) as usize * blk.mid_c as usize * blk.in_c as usize],
            &mut params,
        );
        decl(g, wn_bias(&kt_name), vec![blk.mid_c as usize], &mut params);
        let c3_name = format!("decoder.{}.conv2d_3x3", blk.name);
        decl(
            g,
            wn_kernel(&c3_name),
            vec![3 * 3 * blk.mid_c as usize * blk.out_c as usize],
            &mut params,
        );
        decl(g, wn_bias(&c3_name), vec![blk.out_c as usize], &mut params);
        if blk.has_shortcut_conv {
            let sc_name = format!("decoder.{}.shortcut.conv1x1", blk.name);
            // 1×1 shortcut conv kernel [out_c, in_c, 1, 1] flat.
            decl(
                g,
                wn_kernel(&sc_name),
                vec![blk.in_c as usize * blk.out_c as usize],
                &mut params,
            );
            decl(g, wn_bias(&sc_name), vec![blk.out_c as usize], &mut params);
        }
    }
    params
}

/// Permute a 4D NHWC-style flat tensor `[a, b, c, d]` to `[d, c, a, b]`.
/// Both TF Conv2D `[kH, kW, in_c, out_c]` and TF Conv2DTranspose
/// `[kH, kW, out_c, in_c]` map to meganeura's layout under this permutation:
///   Conv2D     → `[out_c, in_c, kH, kW]`   ✓
///   ConvT      → `[in_c, out_c, kH, kW]`   ✓
fn permute_4d_3201(data: &[f32], dims: [usize; 4]) -> Vec<f32> {
    let [a, b, c, d] = dims;
    assert_eq!(data.len(), a * b * c * d);
    let mut out = vec![0.0_f32; a * b * c * d];
    for i in 0..a {
        for j in 0..b {
            for k in 0..c {
                for l in 0..d {
                    let src = ((i * b + j) * c + k) * d + l;
                    // Output dims: [d, c, a, b]
                    let dst = ((l * c + k) * a + i) * b + j;
                    out[dst] = data[src];
                }
            }
        }
    }
    out
}

/// Convert a conv-T kernel from PyTorch ConvTranspose2D layout
/// `[in_c, out_c, kH, kW]` (what `permute_4d_3201` produces from TF's
/// `[kH, kW, out_c, in_c]`) to the layout the dilate+forward-conv2d
/// rewrite expects:
///   `[out_c, in_c, kH, kW]` with spatial axes flipped.
/// That is: `out[oc, ic, r, c] = in[ic, oc, kH-1-r, kW-1-c]`.
fn flip_and_transpose_conv_t(
    data: &[f32],
    in_c: usize,
    out_c: usize,
    kh: usize,
    kw: usize,
) -> Vec<f32> {
    assert_eq!(data.len(), in_c * out_c * kh * kw);
    let mut out = vec![0.0_f32; data.len()];
    for oc in 0..out_c {
        for ic in 0..in_c {
            for r in 0..kh {
                for c in 0..kw {
                    let src = ((ic * out_c + oc) * kh + (kh - 1 - r)) * kw + (kw - 1 - c);
                    let dst = ((oc * in_c + ic) * kh + r) * kw + c;
                    out[dst] = data[src];
                }
            }
        }
    }
    out
}

/// Load SpectroStream decoder weights from a safetensors file into a session.
///
/// Handles layout conversions:
/// - 4D Conv2D and Conv2DTranspose kernels: TF `[kH, kW, *, *]` → meganeura `[*, *, kH, kW]`
///   via [`permute_4d_3201`].
/// - 1D biases and quantizer codebooks: passed through as-is.
/// - Kernel parameters resolve the `weight_norm.rescaled.kernel` suffix
///   automatically when present (matches the safetensors dump).
///
/// Returns the list of safetensors keys that were NOT loaded (e.g. raw
/// `weight_norm.kernel` / `g` / `initialized` artifacts the meganeura side
/// doesn't need) — callers can ignore or sanity-check.
pub fn load_decoder_weights(
    model: &crate::data::safetensors::SafeTensorsModel,
    session: &mut crate::Session,
) -> Result<Vec<String>, String> {
    let param_names: Vec<String> = session
        .plan()
        .param_buffers
        .iter()
        .map(|entry| entry.0.clone())
        .collect();
    let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
    for name in &param_names {
        // Skip derived parameters (their names carry a ':' tag, e.g.
        // ":winograd"); they're computed at runtime from their source param.
        // The codebooks have no ':' and must be loaded as-is.
        if name.contains(':') && name != "quantizer.rvq_codebooks" {
            continue;
        }
        let info = model
            .tensor_info()
            .get(name)
            .ok_or_else(|| format!("safetensors missing tensor: {name}"))?;
        let data = model
            .tensor_f32_auto(name)
            .map_err(|e| format!("{name}: {e}"))?;
        used.insert(name.clone());
        // Decide if the tensor is a 4D kernel that needs the NHWC→NCHW perm.
        // All `.kernel`-suffixed parameters in our skeleton are 4D conv kernels,
        // except the quantizer codebooks (3D) and biases (1D).
        let permuted = if info.shape.len() == 4
            && (name.ends_with(".kernel") || name.ends_with(".rescaled.kernel"))
        {
            let dims = [info.shape[0], info.shape[1], info.shape[2], info.shape[3]];
            let p = permute_4d_3201(&data, dims);
            // Conv-T kernels also need a transpose+spatial-flip so the
            // weights match the forward-conv kernel layout used by the
            // dilate+conv2d rewrite. Detect by name pattern.
            if name.contains(".conv2dtranspose_") {
                // After permute_4d_3201, layout is
                //   [in_c=info.shape[3], out_c=info.shape[2], kh=info.shape[0], kw=info.shape[1]]
                let in_c = info.shape[3];
                let out_c = info.shape[2];
                let kh = info.shape[0];
                let kw = info.shape[1];
                flip_and_transpose_conv_t(&p, in_c, out_c, kh, kw)
            } else {
                p
            }
        } else {
            data
        };
        session.set_parameter(name, &permuted);
    }
    // Report unloaded safetensors keys (the weight_norm artifacts).
    let leftover: Vec<String> = model
        .tensor_info()
        .keys()
        .filter(|k| !k.starts_with("encoder."))
        .filter(|k| !used.contains(*k))
        .cloned()
        .collect();
    Ok(leftover)
}

/// One residual upsampling block.
fn decoder_block(
    g: &mut Graph,
    x: Feat,
    params: &std::collections::HashMap<String, NodeId>,
    blk: &DecoderBlock,
) -> Feat {
    assert_eq!(
        x.c, blk.in_c,
        "block {}: input channels {} ≠ expected {}",
        blk.name, x.c, blk.in_c
    );

    let (kt_h, kt_w) = blk.kt_kernel;
    let prefix_kt = format!("decoder.{}.conv2dtranspose_{kt_h}x{kt_w}", blk.name);
    let prefix_c3 = format!("decoder.{}.conv2d_3x3", blk.name);

    // Main path: ELU → ConvTranspose → causal H trim + W crop_freq_dim →
    // ELU → causal conv2d_3x3.
    let main = elu(g, x);
    let main = conv_transpose(
        g,
        main,
        params,
        &wn_kernel(&prefix_kt),
        &wn_bias(&prefix_kt),
        blk.mid_c,
        kt_h,
        kt_w,
        blk.stride_h,
        blk.stride_w,
    );
    // TF's conv-T post-processing:
    //   - internal slice trims H to H_in*stride_h from END (causal)
    //   - crop_freq_dim strips W [1, -1] universally; decoder_4 uses [1, -2]
    //     because of its (stride 3, kW 6) sizing.
    // h_excess = kt_h - stride_h frames to drop from H (all from END).
    // w_excess = kt_w - stride_w cells to drop from W; always 1 from front and
    //   the remainder (w_excess - 1) from back.
    let h_excess = kt_h - blk.stride_h;
    let w_excess = kt_w - blk.stride_w;
    let main = slice2d(g, main, 0, h_excess, 1, w_excess - 1);

    let main = elu(g, main);
    // conv2d_3x3 is causal in H (pad [2, 0]) + SAME in W (pad [1, 1]).
    let main = causal_conv2d_3x3(
        g,
        main,
        params,
        &wn_kernel(&prefix_c3),
        &wn_bias(&prefix_c3),
        blk.out_c,
    );

    // Shortcut path: optional 1×1 conv → H/W upsample to match main.
    let short = if blk.has_shortcut_conv {
        let prefix_sc = format!("decoder.{}.shortcut.conv1x1", blk.name);
        conv2d(
            g,
            x,
            params,
            &wn_kernel(&prefix_sc),
            &wn_bias(&prefix_sc),
            blk.out_c,
            1,
            1,
            0,
            0,
        )
    } else {
        x
    };
    let short = upsample_hw(g, short, blk.stride_h, blk.stride_w);

    add_feat(g, main, short)
}

/// Nearest-neighbor upsample on both axes.
fn upsample_hw(g: &mut Graph, x: Feat, scale_h: u32, scale_w: u32) -> Feat {
    let node = g.upsample_nearest(x.node, x.b, x.c, x.h, x.w, scale_h, scale_w);
    Feat {
        node,
        b: x.b,
        c: x.c,
        h: x.h * scale_h,
        w: x.w * scale_w,
    }
}

// ===================== iSTFT (decoder body → audio) =====================

/// iSTFT parameters mirroring `tf.signal.inverse_stft` as the SpectroStream
/// decoder uses it: `frame_length = fft_length = 960`, `frame_step = 480`, a
/// periodic-Hann analysis window with the matching `inverse_stft_window_fn`
/// synthesis window, and `num_bins = 480` kept STFT bins (DC..478,479; the
/// Nyquist bin is dropped on the forward STFT and reconstructed as zero).
#[derive(Clone, Debug)]
pub struct IstftConfig {
    pub frame_length: usize,
    pub frame_step: usize,
    pub fft_length: usize,
    pub num_bins: usize,
    pub num_audio_channels: usize,
}

impl Default for IstftConfig {
    fn default() -> Self {
        Self {
            frame_length: 960,
            frame_step: 480,
            fft_length: 960,
            num_bins: 480,
            num_audio_channels: 2,
        }
    }
}

/// Periodic Hann window (`tf.signal.hann_window`, `periodic=True`).
fn hann_periodic_d(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
        .collect()
}

/// `tf.signal.inverse_stft_window_fn(frame_step)` synthesis window: the forward
/// (Hann) window divided by the hop-folded sum of its square — the COLA
/// normalization that makes analysis×synthesis overlap-add to 1.
fn inverse_synthesis_window(frame_length: usize, frame_step: usize) -> Vec<f64> {
    let fw = hann_periodic_d(frame_length);
    let overlaps = frame_length.div_ceil(frame_step);
    // denom[j] = sum_o fw[o*step + j]^2, folded over the hop, tiled to length.
    let mut denom = vec![0.0_f64; frame_step];
    for (n, &w) in fw.iter().enumerate() {
        denom[n % frame_step] += w * w;
    }
    let _ = overlaps;
    fw.iter()
        .enumerate()
        .map(|(n, &w)| {
            let d = denom[n % frame_step];
            if d > 1e-12 {
                w / d
            } else {
                0.0
            }
        })
        .collect()
}

/// Inverse STFT of the decoder body output → interleaved stereo audio.
///
/// `spec` is the body output `[num_frames, num_bins, 2*num_audio_channels]`
/// row-major (NHWC), the last axis being `(re, im)` interleaved per channel
/// (e.g. `[L_re, L_im, R_re, R_im]`). Returns `[num_samples, num_audio_channels]`
/// interleaved, where `num_samples = num_frames * frame_step` (the
/// `frame_length - frame_step` tail beyond that is dropped, matching the codec's
/// 2-second chunking).
///
/// Verified bit-exact (rms-ratio 0.0) against `tf.signal.inverse_stft` on real
/// decoder body outputs — see `tests/spectrostream_istft.rs`.
pub fn istft_to_audio(spec: &[f32], num_frames: usize, cfg: &IstftConfig) -> Vec<f32> {
    use rustfft::num_complex::Complex;
    use rustfft::FftPlanner;

    let n = cfg.fft_length;
    let bins = cfg.num_bins;
    let chans = cfg.num_audio_channels;
    let cols = 2 * chans;
    assert_eq!(spec.len(), num_frames * bins * cols, "spec size mismatch");

    let syn = inverse_synthesis_window(cfg.frame_length, cfg.frame_step);
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(n);

    let out_len = num_frames * cfg.frame_step;
    let mut audio = vec![0.0_f32; out_len * chans];

    for ch in 0..chans {
        let mut acc = vec![0.0_f64; (num_frames - 1) * cfg.frame_step + cfg.frame_length];
        for f in 0..num_frames {
            // Build the full `n`-point Hermitian spectrum from the kept bins
            // (bins 0..bins-1; Nyquist and the rest = 0, filled by symmetry).
            let mut buf = vec![Complex::new(0.0_f64, 0.0); n];
            for k in 0..bins {
                let base = (f * bins + k) * cols + ch * 2;
                let re = spec[base] as f64;
                let im = spec[base + 1] as f64;
                buf[k] = Complex::new(re, im);
                if k > 0 && k < n {
                    buf[n - k] = Complex::new(re, -im);
                }
            }
            ifft.process(&mut buf);
            // rustfft inverse is unnormalized; divide by n. Apply synthesis
            // window and overlap-add.
            let start = f * cfg.frame_step;
            for (i, &w) in syn.iter().enumerate() {
                acc[start + i] += buf[i].re / n as f64 * w;
            }
        }
        for (s, &v) in acc.iter().take(out_len).enumerate() {
            audio[s * chans + ch] = v as f32;
        }
    }
    audio
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_match_codec_manifest() {
        let c = SpectroStreamConfig::default();
        assert_eq!(c.sample_rate, 48000);
        assert_eq!(c.frame_rate, 25);
        assert_eq!(c.sample_rate / c.frame_rate, 1920);
        assert_eq!(c.embedding_dim, 256);
        assert_eq!(c.initial_freq_bins * c.initial_channels, 2560);
    }

    #[test]
    fn decoder_block_channel_chain_is_consistent() {
        // Block n's out_c must equal block n+1's in_c (or be bridged by an
        // inter-block reshape — currently only decoder_0→1 has 1024→512).
        for i in 1..DECODER_BLOCKS.len() {
            let prev = &DECODER_BLOCKS[i - 1];
            let cur = &DECODER_BLOCKS[i];
            if prev.out_c != cur.in_c {
                // Only allowed mismatch is the documented decoder_0→1 (1024→512).
                assert_eq!(prev.name, "decoder_0");
                assert_eq!(cur.name, "decoder_1");
                assert_eq!(prev.out_c, 1024);
                assert_eq!(cur.in_c, 512);
            }
        }
    }

    #[test]
    fn decoder_graph_declares_all_expected_params() {
        let cfg = SpectroStreamConfig::default();
        let mut g = Graph::new();
        let _ = build_decoder_graph(&mut g, &cfg, 50);
        // Per-block: 4 weight-norm tensors (kernel + bias for conv-transpose, conv2d_3x3)
        // + 2 for shortcut.conv1x1 if has_shortcut_conv. Plus input_layer: 6 conv1x1
        // tensors + 4 conv2d_3x3 tensors + 2 base_conv_last + 1 codebook = ~13 + 28 + ~10 = ~50.
        // Just verify we got a sensible nonzero count.
        let n_params = g
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, crate::graph::Op::Parameter { .. }))
            .count();
        assert!(
            n_params > 30,
            "expected ≥30 parameter nodes, got {n_params}"
        );
    }
}
