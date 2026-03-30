//! High-level neural network building blocks.
//!
//! Each struct holds parameter [`NodeId`]s and provides a `forward()` method
//! that appends operations to the [`Graph`]. These are thin wrappers over
//! the low-level graph API — no trait hierarchy, no dynamic dispatch.
//!
//! # Example
//! ```ignore
//! let mut g = Graph::new();
//! let x = g.input("x", &[batch, 784]);
//! let labels = g.input("labels", &[batch, 10]);
//!
//! let fc1 = nn::Linear::new(&mut g, "fc1", 784, 128);
//! let fc2 = nn::Linear::new(&mut g, "fc2", 128, 10);
//!
//! let h = fc1.forward(&mut g, x);
//! let h = g.relu(h);
//! let logits = fc2.forward(&mut g, h);
//! let loss = g.cross_entropy_loss(logits, labels);
//! ```

use crate::graph::{Graph, NodeId};

/// Fully connected linear layer: `y = x @ weight + bias`.
pub struct Linear {
    pub weight: NodeId,
    pub bias: Option<NodeId>,
}

impl Linear {
    pub fn new(g: &mut Graph, name: &str, in_features: usize, out_features: usize) -> Self {
        let weight = g.parameter(&format!("{name}.weight"), &[in_features, out_features]);
        let bias = Some(g.parameter(&format!("{name}.bias"), &[out_features]));
        Self { weight, bias }
    }

    pub fn no_bias(g: &mut Graph, name: &str, in_features: usize, out_features: usize) -> Self {
        let weight = g.parameter(&format!("{name}.weight"), &[in_features, out_features]);
        Self { weight, bias: None }
    }

    pub fn forward(&self, g: &mut Graph, x: NodeId) -> NodeId {
        let mm = g.matmul(x, self.weight);
        match self.bias {
            Some(b) => g.bias_add(mm, b),
            None => mm,
        }
    }
}

/// Token embedding lookup table.
pub struct Embedding {
    pub weight: NodeId,
}

impl Embedding {
    pub fn new(g: &mut Graph, name: &str, vocab_size: usize, embed_dim: usize) -> Self {
        let weight = g.parameter(name, &[vocab_size, embed_dim]);
        Self { weight }
    }

    pub fn forward(&self, g: &mut Graph, indices: NodeId) -> NodeId {
        g.embedding(indices, self.weight)
    }
}

/// RMS normalization layer.
pub struct RmsNorm {
    pub weight: NodeId,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(g: &mut Graph, name: &str, dim: usize, eps: f32) -> Self {
        let weight = g.parameter(name, &[dim]);
        Self { weight, eps }
    }

    pub fn forward(&self, g: &mut Graph, x: NodeId) -> NodeId {
        g.rms_norm(x, self.weight, self.eps)
    }
}

/// Layer normalization with weight and bias.
pub struct LayerNorm {
    pub weight: NodeId,
    pub bias: NodeId,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(g: &mut Graph, name: &str, dim: usize, eps: f32) -> Self {
        let weight = g.parameter(&format!("{name}.weight"), &[dim]);
        let bias = g.parameter(&format!("{name}.bias"), &[dim]);
        Self { weight, bias, eps }
    }

    pub fn forward(&self, g: &mut Graph, x: NodeId) -> NodeId {
        g.layer_norm(x, self.weight, self.bias, self.eps)
    }
}

/// SwiGLU feed-forward network: `silu(gate(x)) * up(x)` then down-project.
pub struct SwiGluFfn {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

impl SwiGluFfn {
    pub fn new(g: &mut Graph, name: &str, hidden: usize, intermediate: usize) -> Self {
        Self {
            gate: Linear::no_bias(g, &format!("{name}.gate_proj"), hidden, intermediate),
            up: Linear::no_bias(g, &format!("{name}.up_proj"), hidden, intermediate),
            down: Linear::no_bias(g, &format!("{name}.down_proj"), intermediate, hidden),
        }
    }

    pub fn forward(&self, g: &mut Graph, x: NodeId) -> NodeId {
        let gate = self.gate.forward(g, x);
        let up = self.up.forward(g, x);
        let h = g.swiglu(gate, up);
        self.down.forward(g, h)
    }
}

/// Standard MLP: `linear2(act(linear1(x)))`.
pub struct Mlp {
    pub fc1: Linear,
    pub fc2: Linear,
    pub activation: Activation,
}

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Relu,
    Gelu,
    Silu,
    Sigmoid,
}

impl Mlp {
    pub fn new(
        g: &mut Graph,
        name: &str,
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        activation: Activation,
    ) -> Self {
        Self {
            fc1: Linear::new(g, &format!("{name}.fc1"), in_dim, hidden_dim),
            fc2: Linear::new(g, &format!("{name}.fc2"), hidden_dim, out_dim),
            activation,
        }
    }

    pub fn forward(&self, g: &mut Graph, x: NodeId) -> NodeId {
        let h = self.fc1.forward(g, x);
        let h = match self.activation {
            Activation::Relu => g.relu(h),
            Activation::Gelu => g.gelu(h),
            Activation::Silu => g.silu(h),
            Activation::Sigmoid => g.sigmoid(h),
        };
        self.fc2.forward(g, h)
    }
}

/// 2D convolution layer: `y = conv2d(x, weight) + bias`.
///
/// Input and output are flat 1D tensors in NCHW layout.
pub struct Conv2d {
    pub weight: NodeId,
    pub bias: Option<NodeId>,
    pub in_channels: u32,
    pub in_h: u32,
    pub in_w: u32,
    pub out_channels: u32,
    pub kernel_h: u32,
    pub kernel_w: u32,
    pub stride: u32,
    pub padding: u32,
}

impl Conv2d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        g: &mut Graph,
        name: &str,
        in_channels: u32,
        out_channels: u32,
        kernel_size: u32,
        in_h: u32,
        in_w: u32,
        stride: u32,
        padding: u32,
    ) -> Self {
        let weight = g.parameter(
            &format!("{name}.weight"),
            &[out_channels as usize
                * in_channels as usize
                * kernel_size as usize
                * kernel_size as usize],
        );
        Self {
            weight,
            bias: None,
            in_channels,
            in_h,
            in_w,
            out_channels,
            kernel_h: kernel_size,
            kernel_w: kernel_size,
            stride,
            padding,
        }
    }

    pub fn forward(&self, g: &mut Graph, x: NodeId, batch: u32) -> NodeId {
        g.conv2d(
            x,
            self.weight,
            batch,
            self.in_channels,
            self.in_h,
            self.in_w,
            self.out_channels,
            self.kernel_h,
            self.kernel_w,
            self.stride,
            self.padding,
        )
    }
}

/// Configuration for [`CausalSelfAttention`].
pub struct AttentionConfig {
    pub hidden: usize,
    pub kv_dim: usize,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub rope_theta: f32,
}

/// Causal self-attention with grouped-query attention and RoPE.
pub struct CausalSelfAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub rope_theta: f32,
}

impl CausalSelfAttention {
    pub fn new(g: &mut Graph, name: &str, cfg: &AttentionConfig) -> Self {
        Self {
            q_proj: Linear::no_bias(g, &format!("{name}.q_proj"), cfg.hidden, cfg.hidden),
            k_proj: Linear::no_bias(g, &format!("{name}.k_proj"), cfg.hidden, cfg.kv_dim),
            v_proj: Linear::no_bias(g, &format!("{name}.v_proj"), cfg.hidden, cfg.kv_dim),
            o_proj: Linear::no_bias(g, &format!("{name}.o_proj"), cfg.hidden, cfg.hidden),
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
        }
    }

    pub fn forward(&self, g: &mut Graph, x: NodeId) -> NodeId {
        let q = self.q_proj.forward(g, x);
        let k = self.k_proj.forward(g, x);
        let v = self.v_proj.forward(g, x);
        let q = g.rope(q, self.rope_theta, self.head_dim);
        let k = g.rope(k, self.rope_theta, self.head_dim);
        let attn = g.causal_attention(q, k, v, self.num_heads, self.num_kv_heads, self.head_dim);
        self.o_proj.forward(g, attn)
    }
}

/// Configuration for [`TransformerBlock`].
pub struct TransformerBlockConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub kv_dim: usize,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub rms_eps: f32,
    pub rope_theta: f32,
}

/// A single transformer decoder block: attention + FFN with residual connections.
pub struct TransformerBlock {
    pub attn_norm: RmsNorm,
    pub attn: CausalSelfAttention,
    pub ffn_norm: RmsNorm,
    pub ffn: SwiGluFfn,
}

impl TransformerBlock {
    pub fn new(g: &mut Graph, name: &str, cfg: &TransformerBlockConfig) -> Self {
        Self {
            attn_norm: RmsNorm::new(
                g,
                &format!("{name}.input_layernorm.weight"),
                cfg.hidden,
                cfg.rms_eps,
            ),
            attn: CausalSelfAttention::new(
                g,
                &format!("{name}.self_attn"),
                &AttentionConfig {
                    hidden: cfg.hidden,
                    kv_dim: cfg.kv_dim,
                    num_heads: cfg.num_heads,
                    num_kv_heads: cfg.num_kv_heads,
                    head_dim: cfg.head_dim,
                    rope_theta: cfg.rope_theta,
                },
            ),
            ffn_norm: RmsNorm::new(
                g,
                &format!("{name}.post_attention_layernorm.weight"),
                cfg.hidden,
                cfg.rms_eps,
            ),
            ffn: SwiGluFfn::new(g, &format!("{name}.mlp"), cfg.hidden, cfg.intermediate),
        }
    }

    pub fn forward(&self, g: &mut Graph, x: NodeId) -> NodeId {
        let h = self.attn_norm.forward(g, x);
        let h = self.attn.forward(g, h);
        let x = g.add(x, h);
        let h = self.ffn_norm.forward(g, x);
        let h = self.ffn.forward(g, h);
        g.add(x, h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Graph;

    #[test]
    fn linear_builds_graph() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let fc = Linear::new(&mut g, "fc", 8, 3);
        let y = fc.forward(&mut g, x);
        assert_eq!(g.node(y).ty.shape, vec![4, 3]);
    }

    #[test]
    fn linear_no_bias() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let fc = Linear::no_bias(&mut g, "fc", 8, 3);
        let y = fc.forward(&mut g, x);
        assert_eq!(g.node(y).ty.shape, vec![4, 3]);
    }

    #[test]
    fn mlp_builds_graph() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let mlp = Mlp::new(&mut g, "mlp", 8, 16, 3, Activation::Relu);
        let y = mlp.forward(&mut g, x);
        assert_eq!(g.node(y).ty.shape, vec![4, 3]);
    }

    #[test]
    fn transformer_block_builds_graph() {
        let mut g = Graph::new();
        let x = g.input("x", &[16, 64]);
        let cfg = TransformerBlockConfig {
            hidden: 64,
            intermediate: 128,
            kv_dim: 32,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            rms_eps: 1e-5,
            rope_theta: 10000.0,
        };
        let block = TransformerBlock::new(&mut g, "model.layers.0", &cfg);
        let y = block.forward(&mut g, x);
        assert_eq!(g.node(y).ty.shape, vec![16, 64]);
    }
}
