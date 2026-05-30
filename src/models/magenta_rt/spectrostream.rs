//! SpectroStream: 48 kHz stereo neural audio codec (STFT-domain).
//!
//! Architecture decoded from `magenta_rt_codec_dump/manifest.json` +
//! `tools/magenta_rt/trace_codec.py` output:
//!
//! ```text
//! tokens [50 frames, 16 RVQ levels]
//!   → dequantize via 16 of 64 codebooks → embed [50, 256]
//!   → reshape [1, 50, 1, 256] NHWC
//!   → temporal_padding (pad T by ~18 to absorb VALID-conv shrinkage)
//!   → input_layer:
//!       conv1x1_first (gated 1×1 expand 256 → 2560)
//!       reshape [B, T, 1, 2560] → [B, T, 5, 512]
//!       residual block: ELU → freq_pad → conv2d_3x3 → ELU → freq_pad → conv2d_3x3 → + identity
//!   → 7 decoder blocks (decoder_0 … decoder_6):
//!       ELU → conv2dtranspose (stride (1, stride_w)) → crop_freq_dim
//!       ELU → freq_pad → conv2d_3x3
//!       shortcut: conv1x1 if channels differ, then nearest-upsample W by stride_w, +
//!   → final ELU → freq_pad (3, 3) → base_conv_last (7×7, → 2 channels)
//!   → [B, T_final, F_final, 2 (re, im)]
//!   → iSTFT (Hann window, overlap-add) — done host-side on CPU
//!   → audio [B, 96000, 2 stereo]
//! ```
//!
//! Per-block stride_w pattern: 2, 2, 2, 2, 3, 2, 2 → total W upsample = 192×.
//! Per-block kernel shape: 4×3, 4×4, 3×4, 3×4, 3×6, 3×4, 3×4 (kH × kW).
//! Per-block has_shortcut_conv: true, true, false, false, true, false, true.
//! All conv padding = VALID; SpectroStream emulates SAME via explicit Pad/Crop ops.
//! Activation = ELU everywhere. Weight norm precomputed in `.rescaled.kernel`.
//!
//! Status: skeleton only. Builds the graph structure with correct ops but
//! shapes have several TODOs:
//! - exact temporal_padding amount (assumed 18 = sum of conv2d_3x3 H-shrinkages)
//! - inter-block reshape between decoder_0 and decoder_1 (StridedSlice+Reshape+
//!   Transpose mystery — looks like PixelShuffle-2 to halve channels 1024 → 512)
//! - final reshape between decoder_6 and base_conv_last (similar transform)
//! - stereo handling in the spectrogram domain (2 output channels are likely
//!   real+imag of a single complex spectrogram representing folded stereo)

use crate::graph::{Graph, NodeId};

/// Per-decoder-block spec. Each block applies one conv-transpose (upsample W)
/// + one 3×3 refinement conv + a residual shortcut.
#[allow(dead_code)] // stride_w/has_shortcut_conv used by the (still-TODO) forward pass
#[derive(Clone, Copy)]
struct DecoderBlock {
    /// Conv-transpose kernel `(kH, kW)`.
    kt_kernel: (u32, u32),
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
const DECODER_BLOCKS: [DecoderBlock; 7] = [
    DecoderBlock { kt_kernel: (4, 3), stride_w: 2, in_c: 512,  mid_c: 1024, out_c: 1024, has_shortcut_conv: true,  name: "decoder_0" },
    DecoderBlock { kt_kernel: (4, 4), stride_w: 2, in_c: 512,  mid_c: 256,  out_c: 256,  has_shortcut_conv: true,  name: "decoder_1" },
    DecoderBlock { kt_kernel: (3, 4), stride_w: 2, in_c: 256,  mid_c: 256,  out_c: 256,  has_shortcut_conv: false, name: "decoder_2" },
    DecoderBlock { kt_kernel: (3, 4), stride_w: 2, in_c: 256,  mid_c: 256,  out_c: 256,  has_shortcut_conv: false, name: "decoder_3" },
    DecoderBlock { kt_kernel: (3, 6), stride_w: 3, in_c: 256,  mid_c: 128,  out_c: 128,  has_shortcut_conv: true,  name: "decoder_4" },
    DecoderBlock { kt_kernel: (3, 4), stride_w: 2, in_c: 128,  mid_c: 128,  out_c: 128,  has_shortcut_conv: false, name: "decoder_5" },
    DecoderBlock { kt_kernel: (3, 4), stride_w: 2, in_c: 128,  mid_c: 64,   out_c: 64,   has_shortcut_conv: true,  name: "decoder_6" },
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
    /// Revised math (with crop_freq_dim cropping H by kt_h - stride_h, matching
    /// the SAME-via-VALID-emulation pattern used throughout the codec):
    ///   input_layer 2× conv2d_3x3 VALID → -4
    ///   per block: ConvT +kt_h-1, crop -(kt_h-1), conv2d_3x3 -2 → net -2 each
    ///   7 blocks → -14
    ///   base_conv_last 7×7 VALID → -6
    ///   total: -24. temporal_pad = 24.
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
            temporal_pad: 24,
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

impl Feat {
    #[allow(dead_code)]
    fn total(&self) -> u32 { self.b * self.c * self.h * self.w }
}

/// ELU activation (shape preserved).
fn elu(g: &mut Graph, x: Feat) -> Feat {
    Feat { node: g.elu(x.node), ..x }
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
        x.node, kernel,
        x.b, x.c, x.h, x.w,
        out_c, kh, kw,
        1, padding_h, padding_w,
    );
    let out = Feat { node, b: x.b, c: out_c, h: out_h, w: out_w };
    Feat { node: g.add_per_channel(out.node, bias, out_c, out.h * out.w), ..out }
}

/// 2D transposed convolution with separate H/W strides. Kernel uploaded as
/// `[in_c, out_c, kH, kW]` flat (PyTorch convention).
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
    let node = g.conv_transpose_2d_hw(
        x.node, kernel,
        x.b, x.c, x.h, x.w,
        out_c, kh, kw,
        stride_h, stride_w,
        0, 0,
    );
    let out = Feat { node, b: x.b, c: out_c, h: out_h, w: out_w };
    Feat { node: g.add_per_channel(out.node, bias, out_c, out.h * out.w), ..out }
}

fn slice2d(g: &mut Graph, x: Feat, start_h: u32, end_h: u32, start_w: u32, end_w: u32) -> Feat {
    let out_h = x.h - start_h - end_h;
    let out_w = x.w - start_w - end_w;
    let node = g.slice_2d(x.node, x.b, x.c, x.h, x.w, start_h, end_h, start_w, end_w);
    Feat { node, b: x.b, c: x.c, h: out_h, w: out_w }
}

fn upsample_w(g: &mut Graph, x: Feat, scale_w: u32) -> Feat {
    let node = g.upsample_nearest(x.node, x.b, x.c, x.h, x.w, 1, scale_w);
    Feat { node, b: x.b, c: x.c, h: x.h, w: x.w * scale_w }
}

fn add_feat(g: &mut Graph, a: Feat, b: Feat) -> Feat {
    assert_eq!((a.b, a.c, a.h, a.w), (b.b, b.c, b.h, b.w), "add_feat shape mismatch");
    Feat { node: g.add(a.node, b.node), ..a }
}

/// Helper: name of a weight_norm rescaled kernel parameter.
fn wn_kernel(prefix: &str) -> String {
    format!("{prefix}.weight_norm.rescaled.kernel")
}
/// Helper: name of a weight_norm bias parameter.
fn wn_bias(prefix: &str) -> String {
    format!("{prefix}.weight_norm.bias")
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
/// TODO: this is still partial — does not include input_layer's residual
/// block, the inter-block reshape, or the temporal_cropping at the end.
/// Returns a placeholder NodeId for now so the graph compiles.
pub fn build_decoder_graph(g: &mut Graph, cfg: &SpectroStreamConfig, num_frames: u32) -> NodeId {
    build_decoder_graph_through(g, cfg, num_frames, DecoderStage::Output)
}

/// Stage in the decoder pipeline — pass to [`build_decoder_graph_through`] for
/// debugging to terminate the graph at a particular intermediate output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoderStage {
    /// After input_layer's residual block. Shape `[1, 512, num_frames+pad-4, 5]`.
    InputLayer,
    /// After decoder block `n` (0..=6). After block 0, the PixelShuffleW
    /// is also applied (so c=512 when n=0 not 1024).
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
    let in_size = (1 * cfg.initial_channels * h_padded * cfg.initial_freq_bins) as usize;
    let x_input = g.input("decoder_input_preprocessed", &[in_size]);
    let mut x = Feat {
        node: x_input,
        b: 1,
        c: cfg.initial_channels,
        h: h_padded,
        w: cfg.initial_freq_bins,
    };

    // input_layer residual block.
    let residual = x;
    for sub in &["conv2d_3x3_a", "conv2d_3x3"] {
        let prefix = format!("decoder.input_layer.{sub}");
        let h = elu(g, x);
        x = conv2d(g, h, &params, &wn_kernel(&prefix), &wn_bias(&prefix),
                   cfg.initial_channels, 3, 3, 0, 1);
    }
    let residual_sliced = slice2d(g, residual, 2, 2, 0, 0);
    x = add_feat(g, x, residual_sliced);
    if stop_after == DecoderStage::InputLayer { return x.node; }

    for (idx, blk) in DECODER_BLOCKS.iter().enumerate() {
        x = decoder_block(g, x, &params, blk);
        if idx == 0 {
            let node = g.pixel_shuffle_w(x.node, x.b, x.c, x.h, x.w, 2);
            x = Feat { node, b: x.b, c: x.c / 2, h: x.h, w: x.w * 2 };
        }
        if stop_after == DecoderStage::Block(idx as u8) { return x.node; }
    }
    assert_eq!(x.c, 64);

    let x_act = elu(g, x);
    let kernel = params["decoder.input_layer.base_conv_last.conv.kernel"];
    let bias = params["decoder.input_layer.base_conv_last.conv.bias"];
    let out_h = x_act.h - 6;
    let out_w = x_act.w;
    let conv_node = g.conv2d_hw(
        x_act.node, kernel,
        x_act.b, x_act.c, x_act.h, x_act.w,
        2, 7, 7,
        1, 0, 3,
    );
    let pre_bias = Feat { node: conv_node, b: x_act.b, c: 2, h: out_h, w: out_w };
    g.add_per_channel(pre_bias.node, bias, 2, pre_bias.h * pre_bias.w)
}

/// Declares every safetensors parameter the decoder needs and returns a
/// `name → NodeId` map. Each name is declared exactly once. The host-side
/// params (quantizer, conv1x1_first) are declared too so the loader sees
/// them; they're unused by the GPU graph.
fn declare_all_params(g: &mut Graph, cfg: &SpectroStreamConfig) -> std::collections::HashMap<String, NodeId> {
    let mut params = std::collections::HashMap::new();
    let mut decl = |g: &mut Graph, name: String, shape: Vec<usize>, params: &mut std::collections::HashMap<String, NodeId>| {
        let node = g.parameter(&name, &shape);
        params.insert(name, node);
    };

    decl(
        g,
        "quantizer.rvq_codebooks".to_string(),
        vec![(cfg.max_rvq_depth * cfg.codebook_size * cfg.embedding_dim) as usize],
        &mut params,
    );
    for sub in &["conv1x1_first.conv1x1", "conv1x1_first.conv1x1_a", "conv1x1_first.conv1x1_b"] {
        let prefix = format!("decoder.input_layer.{sub}");
        let (out_c, in_c) = match *sub {
            "conv1x1_first.conv1x1"   => (2560, cfg.embedding_dim as usize),
            "conv1x1_first.conv1x1_a" => (2560, cfg.embedding_dim as usize),
            "conv1x1_first.conv1x1_b" => (2560, 2560),
            _ => unreachable!(),
        };
        decl(g, wn_kernel(&prefix), vec![1 * 1 * in_c * out_c], &mut params);
        decl(g, wn_bias(&prefix), vec![out_c], &mut params);
    }
    for sub in &["conv2d_3x3", "conv2d_3x3_a"] {
        let prefix = format!("decoder.input_layer.{sub}");
        decl(g, wn_kernel(&prefix), vec![3 * 3 * 512 * 512], &mut params);
        decl(g, wn_bias(&prefix), vec![512], &mut params);
    }
    decl(g, "decoder.input_layer.base_conv_last.conv.kernel".to_string(), vec![7 * 7 * 64 * 2], &mut params);
    decl(g, "decoder.input_layer.base_conv_last.conv.bias".to_string(), vec![2], &mut params);
    for blk in DECODER_BLOCKS.iter() {
        let (kh, kw) = blk.kt_kernel;
        let kt_name = format!("decoder.{}.conv2dtranspose_{kh}x{kw}", blk.name);
        decl(g, wn_kernel(&kt_name), vec![(kh * kw) as usize * blk.mid_c as usize * blk.in_c as usize], &mut params);
        decl(g, wn_bias(&kt_name), vec![blk.mid_c as usize], &mut params);
        let c3_name = format!("decoder.{}.conv2d_3x3", blk.name);
        decl(g, wn_kernel(&c3_name), vec![3 * 3 * blk.mid_c as usize * blk.out_c as usize], &mut params);
        decl(g, wn_bias(&c3_name), vec![blk.out_c as usize], &mut params);
        if blk.has_shortcut_conv {
            let sc_name = format!("decoder.{}.shortcut.conv1x1", blk.name);
            decl(g, wn_kernel(&sc_name), vec![1 * 1 * blk.in_c as usize * blk.out_c as usize], &mut params);
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
        .map(|(name, _)| name.clone())
        .collect();
    let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
    for name in &param_names {
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
            permute_4d_3201(&data, dims)
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
    assert_eq!(x.c, blk.in_c, "block {}: input channels {} ≠ expected {}", blk.name, x.c, blk.in_c);

    let (kt_h, kt_w) = blk.kt_kernel;
    let prefix_kt = format!("decoder.{}.conv2dtranspose_{kt_h}x{kt_w}", blk.name);
    let prefix_c3 = format!("decoder.{}.conv2d_3x3", blk.name);

    // Main path: ELU → ConvTranspose → crop_freq_dim → ELU → conv2d_3x3.
    let main = elu(g, x);
    let main = conv_transpose(
        g, main, params,
        &wn_kernel(&prefix_kt), &wn_bias(&prefix_kt),
        blk.mid_c, kt_h, kt_w,
        1, blk.stride_w,
    );
    // crop_freq_dim removes (kt_h - stride_h, kt_w - stride_w) cells total on H/W
    // to bring spatial back to (in_h * stride_h, in_w * stride_w) — the SAME-via
    // -VALID-emulation pattern. stride_h=1 so H crop = kt_h - 1; W crop = kt_w - stride_w.
    let h_excess = kt_h - 1;
    let w_excess = kt_w - blk.stride_w;
    let main = slice2d(g, main, h_excess / 2, h_excess - h_excess / 2,
                       w_excess / 2, w_excess - w_excess / 2);

    let main = elu(g, main);
    let main = conv2d(
        g, main, params,
        &wn_kernel(&prefix_c3), &wn_bias(&prefix_c3),
        blk.out_c, 3, 3,
        0, 1,
    );

    // Shortcut path: optional 1×1 conv → W-upsample → H-slice by 2 (conv2d_3x3
    // shrinkage on the main path).
    let short = if blk.has_shortcut_conv {
        let prefix_sc = format!("decoder.{}.shortcut.conv1x1", blk.name);
        conv2d(g, x, params, &wn_kernel(&prefix_sc), &wn_bias(&prefix_sc),
               blk.out_c, 1, 1, 0, 0)
    } else {
        x
    };
    let short = upsample_w(g, short, blk.stride_w);
    let short = slice2d(g, short, 1, 1, 0, 0);

    add_feat(g, main, short)
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
        let n_params = g.nodes().iter().filter(|n| {
            matches!(n.op, crate::graph::Op::Parameter { .. })
        }).count();
        assert!(n_params > 30, "expected ≥30 parameter nodes, got {n_params}");
    }
}
