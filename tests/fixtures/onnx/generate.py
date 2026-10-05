"""Generate transformer-layer ONNX fixtures as `torch.onnx.export` writes them.

Each model is spelled node for node the way PyTorch's exporter emits a
Hugging Face layer with static shapes: linear layers as MatMul + Add,
attention heads through Reshape and Transpose, decomposed normalizations,
and activations as elementary math. The expected output comes from ONNX's
own reference evaluator, independent of Meganeura.

    pip install onnx numpy
    python tests/fixtures/onnx/generate.py

writes `<name>.onnx`, `<name>.input.bin` and `<name>.expected.bin` (raw
little-endian f32) next to this script.
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ, HIDDEN, HEADS, FFN = 8, 32, 4, 64
HEAD = HIDDEN // HEADS


class Builder:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.nodes, self.inits, self.count = [], [], 0

    def name(self, hint):
        self.count += 1
        return f"{hint}_{self.count}"

    def const(self, hint, value, dtype=np.float32):
        name = self.name(hint)
        self.inits.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def weight(self, hint, shape, scale):
        return self.const(hint, self.rng.normal(0.0, scale, shape))

    def op(self, op_type, *inputs, **attrs):
        out = self.name(op_type.lower())
        self.nodes.append(helper.make_node(op_type, list(inputs), [out], **attrs))
        return out

    def linear(self, x, n_in, n_out, bias=True):
        y = self.op("MatMul", x, self.weight("weight", (n_in, n_out), n_in**-0.5))
        if bias:
            y = self.op("Add", y, self.weight("bias", (n_out,), 0.1))
        return y

    def heads(self, x, perm=(0, 2, 1, 3)):
        shape = self.const("shape", [1, SEQ, HEADS, HEAD], np.int64)
        return self.op("Transpose", self.op("Reshape", x, shape), perm=list(perm))

    def merge_heads(self, x):
        x = self.op("Transpose", x, perm=[0, 2, 1, 3])
        return self.op("Reshape", x, self.const("shape", [1, SEQ, HIDDEN], np.int64))

    def layer_norm(self, x):
        # Decomposed as in opsets before LayerNormalization (17).
        mean = self.op("ReduceMean", x, axes=[-1], keepdims=1)
        centered = self.op("Sub", x, mean)
        square = self.op("Pow", centered, self.const("two", 2.0))
        variance = self.op("ReduceMean", square, axes=[-1], keepdims=1)
        shifted = self.op("Add", variance, self.const("eps", 1e-12))
        normalized = self.op("Div", centered, self.op("Sqrt", shifted))
        scaled = self.op("Mul", normalized, self.weight("gamma", (HIDDEN,), 0.1))
        return self.op("Add", scaled, self.weight("beta", (HIDDEN,), 0.1))

    def rms_norm(self, x):
        # LlamaRMSNorm: weight * (x * rsqrt(mean(x²) + eps)).
        square = self.op("Pow", x, self.const("two", 2.0))
        mean = self.op("ReduceMean", square, axes=[-1], keepdims=1)
        root = self.op("Sqrt", self.op("Add", mean, self.const("eps", 1e-6)))
        normalized = self.op("Mul", x, self.op("Reciprocal", root))
        gamma = self.const("gamma", 1.0 + self.rng.normal(0.0, 0.1, (HIDDEN,)))
        return self.op("Mul", gamma, normalized)

    def finish(self, name, x_name, y, opset):
        graph = helper.make_graph(
            self.nodes,
            name,
            [helper.make_tensor_value_info(x_name, TensorProto.FLOAT, [1, SEQ, HIDDEN])],
            [helper.make_tensor_value_info(y, TensorProto.FLOAT, [1, SEQ, HIDDEN])],
            self.inits,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        x = self.rng.normal(0.0, 1.0, (1, SEQ, HIDDEN)).astype(np.float32)
        (expected,) = ReferenceEvaluator(model).run(None, {x_name: x})
        onnx.save(model, os.path.join(HERE, f"{name}.onnx"))
        x.tofile(os.path.join(HERE, f"{name}.input.bin"))
        expected.astype(np.float32).tofile(os.path.join(HERE, f"{name}.expected.bin"))
        print(name, len(self.nodes), "nodes")


def bert_layer():
    """BertLayer: post-norm attention and GELU (erf) feed-forward."""
    b = Builder(1)
    x = "hidden_states"
    q = b.heads(b.linear(x, HIDDEN, HIDDEN))
    k = b.heads(b.linear(x, HIDDEN, HIDDEN))
    v = b.heads(b.linear(x, HIDDEN, HIDDEN))
    k_t = b.op("Transpose", k, perm=[0, 1, 3, 2])
    scores = b.op("Div", b.op("MatMul", q, k_t), b.const("scale", np.sqrt(HEAD)))
    # The extended attention mask of an all-ones mask: zeros over [1, 1, 1, S].
    scores = b.op("Add", scores, b.const("mask", np.zeros((1, 1, 1, SEQ))))
    probs = b.op("Softmax", scores, axis=-1)
    context = b.merge_heads(b.op("MatMul", probs, v))
    attn = b.linear(context, HIDDEN, HIDDEN)
    h = b.layer_norm(b.op("Add", attn, x))
    up = b.linear(h, HIDDEN, FFN)
    erf = b.op("Erf", b.op("Div", up, b.const("sqrt2", np.sqrt(2.0))))
    gelu = b.op("Mul", b.op("Mul", up, b.op("Add", erf, b.const("one", 1.0))), b.const("half", 0.5))
    down = b.linear(gelu, FFN, HIDDEN)
    y = b.layer_norm(b.op("Add", down, h))
    b.finish("bert_layer", x, y, 13)


def llama_layer():
    """LlamaDecoderLayer: pre-norm causal attention with rotary embeddings
    and a SwiGLU feed-forward."""
    b = Builder(2)
    x = "hidden_states"
    h = b.rms_norm(x)
    q = b.heads(b.linear(h, HIDDEN, HIDDEN, bias=False))
    k = b.heads(b.linear(h, HIDDEN, HIDDEN, bias=False))
    v = b.heads(b.linear(h, HIDDEN, HIDDEN, bias=False))
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, HEAD, 2) / HEAD))
    angles = np.outer(np.arange(SEQ), inv_freq)
    angles = np.concatenate([angles, angles], axis=-1)[None, None]
    cos, sin = b.const("cos", np.cos(angles)), b.const("sin", np.sin(angles))

    def rotate_half(t):
        axes = b.const("axes", [3], np.int64)
        first = b.op("Slice", t, b.const("start", [0], np.int64), b.const("end", [HEAD // 2], np.int64), axes)
        second = b.op("Slice", t, b.const("start", [HEAD // 2], np.int64), b.const("end", [HEAD], np.int64), axes)
        return b.op("Concat", b.op("Neg", second), first, axis=-1)

    def rope(t):
        return b.op("Add", b.op("Mul", t, cos), b.op("Mul", rotate_half(t), sin))

    q, k = rope(q), rope(k)
    k_t = b.op("Transpose", k, perm=[0, 1, 3, 2])
    scores = b.op("Div", b.op("MatMul", q, k_t), b.const("scale", np.sqrt(HEAD)))
    causal = np.triu(np.full((SEQ, SEQ), np.finfo(np.float32).min), 1)[None, None]
    scores = b.op("Add", scores, b.const("mask", causal))
    probs = b.op("Softmax", scores, axis=-1)
    context = b.merge_heads(b.op("MatMul", probs, v))
    h = b.op("Add", x, b.linear(context, HIDDEN, HIDDEN, bias=False))
    n = b.rms_norm(h)
    gate = b.linear(n, HIDDEN, FFN, bias=False)
    silu = b.op("Mul", gate, b.op("Sigmoid", gate))
    up = b.linear(n, HIDDEN, FFN, bias=False)
    down = b.linear(b.op("Mul", silu, up), FFN, HIDDEN, bias=False)
    y = b.op("Add", h, down)
    b.finish("llama_layer", x, y, 17)


def llama_gqa_dynamic_layer():
    """LlamaDecoderLayer with grouped-query attention, exported with dynamic
    batch and sequence axes: reshape targets come from Shape, Gather,
    Unsqueeze and Concat, and KV heads repeat through Unsqueeze, Expand and
    Reshape (`repeat_kv`)."""
    kv_heads = HEADS // 2
    b = Builder(3)
    x = "hidden_states"
    shape = b.op("Shape", x)
    batch = b.op("Unsqueeze", b.op("Gather", shape, b.const("zero", 0, np.int64), axis=0), b.const("axes", [0], np.int64))
    seq = b.op("Unsqueeze", b.op("Gather", shape, b.const("one", 1, np.int64), axis=0), b.const("axes", [0], np.int64))

    def dims(*parts):
        return b.op("Concat", *[p if isinstance(p, str) else b.const("dim", [p], np.int64) for p in parts], axis=0)

    def heads(t, n):
        t = b.op("Reshape", t, dims(batch, seq, n, HEAD))
        return b.op("Transpose", t, perm=[0, 2, 1, 3])

    def repeat_kv(t):
        t = b.op("Unsqueeze", t, b.const("axes", [2], np.int64))
        t = b.op("Expand", t, dims(batch, kv_heads, HEADS // kv_heads, seq, HEAD))
        return b.op("Reshape", t, dims(batch, HEADS, seq, HEAD))

    h = b.rms_norm(x)
    q = heads(b.linear(h, HIDDEN, HIDDEN, bias=False), HEADS)
    k = repeat_kv(heads(b.linear(h, HIDDEN, kv_heads * HEAD, bias=False), kv_heads))
    v = repeat_kv(heads(b.linear(h, HIDDEN, kv_heads * HEAD, bias=False), kv_heads))
    scores = b.op("Div", b.op("MatMul", q, b.op("Transpose", k, perm=[0, 1, 3, 2])), b.const("scale", np.sqrt(HEAD)))
    causal = np.triu(np.full((SEQ, SEQ), np.finfo(np.float32).min), 1)[None, None]
    probs = b.op("Softmax", b.op("Add", scores, b.const("mask", causal)), axis=-1)
    context = b.op("Transpose", b.op("MatMul", probs, v), perm=[0, 2, 1, 3])
    context = b.op("Reshape", context, dims(batch, seq, HIDDEN))
    h = b.op("Add", x, b.linear(context, HIDDEN, HIDDEN, bias=False))
    n = b.rms_norm(h)
    gate = b.linear(n, HIDDEN, FFN, bias=False)
    silu = b.op("Mul", gate, b.op("Sigmoid", gate))
    down = b.linear(b.op("Mul", silu, b.linear(n, HIDDEN, FFN, bias=False)), FFN, HIDDEN, bias=False)
    b.finish("llama_gqa_dynamic_layer", x, b.op("Add", h, down), 17)


if __name__ == "__main__":
    bert_layer()
    llama_layer()
    llama_gqa_dynamic_layer()
