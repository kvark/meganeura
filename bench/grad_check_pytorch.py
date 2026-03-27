#!/usr/bin/env python3
"""Compare meganeura gradients against PyTorch for the SmolVLA action expert.

Reads meganeura's JSON from stdin (output of grad_check example), runs an
equivalent PyTorch forward+backward with IDENTICAL weights and inputs, and
reports per-parameter cosine similarity and relative gradient norm error.

Usage:
    cargo run --release --example grad_check [-- --small] | python bench/grad_check_pytorch.py
    # or:
    cargo run --release --example grad_check > /tmp/mega_grads.json
    python bench/grad_check_pytorch.py /tmp/mega_grads.json

Exit code: 0 if all checks pass, 1 if any parameter fails the threshold.

Weight layout convention:
  meganeura stores matmul weights as [in, out] (W in y = x @ W).
  PyTorch nn.Linear stores weight as [out, in] (weight.T in y = x @ weight.T).
  So for nn.Linear params:   pytorch_weight = meganeura_weight.T
                              pytorch_grad   = meganeura_grad.T
  For 1-D params (biases, RMSNorm weights): no transposition.
"""

import json
import math
import sys

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model (must match build_action_expert_training exactly)
# ---------------------------------------------------------------------------

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class GQAAttention(torch.nn.Module):
    def __init__(self, hidden, num_heads, num_kv_heads, head_dim, kv_in, is_cross):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.is_cross = is_cross
        self.heads_per_kv = num_heads // num_kv_heads
        self.q_proj = torch.nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(kv_in, num_kv_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(kv_in, num_kv_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, hidden, bias=False)

    def forward(self, x, kv=None):
        B, q_seq, _ = x.shape
        src = kv if (self.is_cross and kv is not None) else x
        kv_seq = src.shape[1]
        q = self.q_proj(x).view(B, q_seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(src).view(B, kv_seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(src).view(B, kv_seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.heads_per_kv, dim=1)
        v = v.repeat_interleave(self.heads_per_kv, dim=1)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(B, q_seq, self.num_heads * self.head_dim)
        return self.o_proj(out)


class ExpertLayer(torch.nn.Module):
    def __init__(self, hidden, num_heads, num_kv_heads, head_dim, kv_dim,
                 intermediate, eps, is_cross, layer_idx, self_attn_every_n):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden, eps)
        kv_in = kv_dim if is_cross else hidden
        self.self_attn = GQAAttention(hidden, num_heads, num_kv_heads, head_dim, kv_in, is_cross)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.mlp = SwiGLU(hidden, intermediate)

    def forward(self, x, kv=None):
        h = self.input_layernorm(x)
        x = x + self.self_attn(h, kv)
        h = self.post_attention_layernorm(x)
        x = x + self.mlp(h)
        return x


class SwiGLU(torch.nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ActionExpert(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg["hidden_size"]
        n_layers = cfg["num_layers"]
        n_heads = cfg["num_heads"]
        n_kv = cfg["num_kv_heads"]
        hd = cfg["head_dim"]
        kv_dim = n_kv * hd
        inter = cfg["intermediate_size"]
        ad = cfg["action_dim"]
        eps = cfg["rms_norm_eps"]
        every_n = cfg["self_attn_every_n"]

        self.action_in_proj = torch.nn.Linear(ad, h, bias=True)
        self.action_time_mlp_in = torch.nn.Linear(h * 2, h, bias=True)
        self.action_time_mlp_out = torch.nn.Linear(h, h, bias=True)
        self.layers = torch.nn.ModuleList([
            ExpertLayer(h, n_heads, n_kv, hd, kv_dim, inter, eps,
                        is_cross=(i % every_n != 0),
                        layer_idx=i, self_attn_every_n=every_n)
            for i in range(n_layers)
        ])
        self.action_out_proj = torch.nn.Linear(h, ad, bias=True)
        self.self_attn_every_n = every_n

    def forward(self, noisy_actions, timestep, vlm_kv_per_layer):
        # noisy_actions: [1, chunk, ad]; timestep: [1, 1, h*2]; vlm_kv_per_layer: dict i→tensor
        x = self.action_in_proj(noisy_actions)
        t = F.silu(self.action_time_mlp_in(timestep))
        t = self.action_time_mlp_out(t)
        x = x + t
        for i, layer in enumerate(self.layers):
            kv = vlm_kv_per_layer.get(i)
            x = layer(x, kv)
        return self.action_out_proj(x)


# ---------------------------------------------------------------------------
# Parameter name mapping: meganeura → PyTorch
# ---------------------------------------------------------------------------

def mega_to_pt_name(mega_name):
    """Map a meganeura parameter name to the corresponding PyTorch parameter name."""
    prefix = "model.vlm_with_expert.lm_expert.layers."
    if mega_name.startswith(prefix):
        rest = mega_name[len(prefix):]
        return "layers." + rest
    for p in ("model.action_in_proj.", "model.action_time_mlp_in.",
              "model.action_time_mlp_out.", "model.action_out_proj."):
        if mega_name.startswith(p):
            return mega_name[len("model."):]
    return mega_name  # fallback


def is_linear_weight(mega_name, mega_shape):
    """True if this parameter is a 2-D linear weight requiring transposition."""
    return len(mega_shape) == 2 and mega_name.endswith(".weight")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def cosine_sim(a, b):
    na, nb = a.norm(), b.norm()
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return (a @ b / (na * nb)).item()


def rel_norm_err(mega_norm, pt_norm):
    denom = max(mega_norm, pt_norm, 1e-12)
    return abs(mega_norm - pt_norm) / denom


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else None
    if src:
        with open(src) as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)

    cfg = data["config"]
    mega_loss = data["loss"]
    mega_grads = data["param_grads"]

    print(f"config: hidden={cfg['hidden_size']} layers={cfg['num_layers']} "
          f"heads={cfg['num_heads']}/{cfg['num_kv_heads']} head_dim={cfg['head_dim']} "
          f"chunk={cfg['chunk_size']} vlm_seq={cfg['vlm_seq_len']}")
    print(f"meganeura loss: {mega_loss:.8f}")

    device = "cpu"
    dtype = torch.float32

    # Build model
    expert = ActionExpert(cfg).to(device=device, dtype=dtype)

    # -----------------------------------------------------------------------
    # Initialize PyTorch weights to match meganeura exactly.
    #
    # meganeura init: data[i] = sin(i * 0.01 + 1.0) * 0.1  (per-parameter)
    # For Linear weights: meganeura shape is [in, out], PyTorch is [out, in].
    # We build the [in, out] matrix from the sin pattern and transpose into PT.
    # -----------------------------------------------------------------------
    param_dict = {n: p for n, p in expert.named_parameters()}

    for mega_name, info in mega_grads.items():
        shape = info["shape"]  # meganeura shape [in, out] or [n]
        n = 1
        for s in shape:
            n *= s
        flat = torch.tensor(
            [(math.sin(i * 0.01 + 1.0) * 0.1) for i in range(n)], dtype=dtype
        )
        pt_name = mega_to_pt_name(mega_name)
        if pt_name not in param_dict:
            print(f"  WARNING: no PT param for {mega_name!r} (→ {pt_name!r})")
            continue
        p = param_dict[pt_name]
        with torch.no_grad():
            if is_linear_weight(mega_name, shape):
                # flat → [in, out].T = [out, in] which is PyTorch's layout
                p.copy_(flat.reshape(shape[0], shape[1]).T.contiguous())
            else:
                p.copy_(flat.reshape(shape))

    # -----------------------------------------------------------------------
    # Build inputs matching meganeura's sin/cos patterns
    # -----------------------------------------------------------------------
    h = cfg["hidden_size"]
    ad = cfg["action_dim"]
    chunk = cfg["chunk_size"]
    vlm_seq = cfg["vlm_seq_len"]
    kv_dim = cfg["num_kv_heads"] * cfg["head_dim"]
    every_n = cfg["self_attn_every_n"]

    def make_input(n_elems, fn):
        return torch.tensor([fn(i) for i in range(n_elems)], dtype=dtype)

    noisy_actions = make_input(chunk * ad, lambda i: math.sin(i * 0.01) * 0.1)
    noisy_actions = noisy_actions.reshape(1, chunk, ad)

    timestep = make_input(h * 2, lambda i: math.cos(i * 0.005) * 0.1)
    timestep = timestep.reshape(1, 1, h * 2)

    vlm_kv_flat = make_input(vlm_seq * kv_dim, lambda i: math.sin(i * 0.002) * 0.05)
    vlm_kv_tensor = vlm_kv_flat.reshape(1, vlm_seq, kv_dim)

    # Cross-attention layers need vlm_kv; self-attention layers get None
    vlm_kv_per_layer = {
        i: vlm_kv_tensor
        for i in range(cfg["num_layers"])
        if i % every_n != 0
    }

    target = torch.zeros(1, chunk, ad, dtype=dtype)

    # -----------------------------------------------------------------------
    # Forward + backward
    # -----------------------------------------------------------------------
    expert.zero_grad()
    out = expert(noisy_actions, timestep, vlm_kv_per_layer)
    loss = F.mse_loss(out, target)
    loss.backward()

    pt_loss = loss.item()
    print(f"pytorch    loss: {pt_loss:.8f}")
    loss_rel_err = abs(mega_loss - pt_loss) / max(abs(mega_loss), abs(pt_loss), 1e-12)
    print(f"loss rel-err:    {loss_rel_err:.4e}  {'OK' if loss_rel_err < 1e-4 else 'FAIL'}\n")

    # -----------------------------------------------------------------------
    # Compare gradients
    # -----------------------------------------------------------------------
    COSINE_THRESHOLD = 0.99   # must be nearly identical direction
    NORM_REL_THRESHOLD = 0.05  # gradient norms within 5%

    all_ok = loss_rel_err < 1e-4
    n_params = 0
    n_failed = 0

    print(f"{'Parameter':<65} {'cos_sim':>8} {'norm_err':>9} {'status':>6}")
    print("-" * 92)

    for mega_name, info in mega_grads.items():
        shape = info["shape"]
        mega_norm = info["norm"]
        mega_sample = torch.tensor(info["sample"], dtype=dtype)

        pt_name = mega_to_pt_name(mega_name)
        if pt_name not in param_dict:
            continue

        p = param_dict[pt_name]
        if p.grad is None:
            print(f"  {mega_name}: no PyTorch gradient")
            continue

        n_params += 1

        # Align PyTorch gradient to meganeura's layout for comparison
        pt_grad = p.grad.detach()
        if is_linear_weight(mega_name, shape):
            # PT grad is [out, in]; meganeura grad is [in, out] = PT.T
            pt_grad_aligned = pt_grad.T.contiguous().flatten()
        else:
            pt_grad_aligned = pt_grad.flatten()

        pt_norm = pt_grad_aligned.norm().item()

        # Cosine similarity on the sampled prefix
        n_sample = len(mega_sample)
        pt_sample = pt_grad_aligned[:n_sample]
        cos = cosine_sim(mega_sample, pt_sample)

        # Norm relative error
        nre = rel_norm_err(mega_norm, pt_norm)

        ok = (math.isnan(cos) or cos > COSINE_THRESHOLD) and nre < NORM_REL_THRESHOLD
        if not ok:
            n_failed += 1
            all_ok = False

        label = "OK" if ok else "FAIL"
        cos_str = f"{cos:.5f}" if not math.isnan(cos) else "  nan  "
        print(f"  {mega_name:<63} {cos_str:>8} {nre:>9.4e} {label:>6}")

    print()
    print(f"Result: {n_params - n_failed}/{n_params} parameters OK")
    if all_ok:
        print("PASS: meganeura gradients match PyTorch within tolerance")
    else:
        print(f"FAIL: {n_failed} parameter(s) outside tolerance "
              f"(cos_sim>{COSINE_THRESHOLD}, norm_err<{NORM_REL_THRESHOLD})")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
