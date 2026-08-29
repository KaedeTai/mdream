"""Qwen3-VL-8B decoder in MLX — the 36 layers HiDream-O1 runs its sequence through.

Three details the reference pins down that a port gets wrong by default:

* **q_norm / k_norm are per-head and applied BEFORE rope.** They are RMSNorm over
  head_dim (128), not over hidden_size, and `rms_norm_add` is False for this
  config so there is no `+1` on the weight.
* **MRoPE is interleaved.** Starting from the T-axis frequencies, the H axis
  overwrites `slice(1, 3*rope_dims[1], 3)` and W overwrites
  `slice(2, 3*rope_dims[2], 3)`. With dims [24, 20, 20] that leaves indices
  0,3,..,57 plus 60..63 on T — 24 of them, which is what rope_dims[0] says.
* **GQA is 32 query heads over 8 KV heads**, expanded by repeat_interleave, not
  by tiling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class TextConfig:
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    rope_dims: List[int] = field(default_factory=lambda: [24, 20, 20])
    interleaved_mrope: bool = True
    rms_norm_add: bool = False


def rms_norm(x: mx.array, w: mx.array, eps: float, add: bool = False) -> mx.array:
    if add:
        w = w + 1.0
    xf = x.astype(mx.float32)
    out = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (out * w.astype(mx.float32)).astype(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, add: bool = False):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps
        self.add = add

    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x, self.weight, self.eps, self.add)


def precompute_freqs_cis(head_dim: int, position_ids: mx.array, theta: float,
                         rope_dims: Optional[List[int]] = None,
                         interleaved_mrope: bool = False):
    """position_ids: (3, T) for MRoPE, or (1, T). Returns (cos, sin, nsin)."""
    num = mx.arange(0, head_dim, 2, dtype=mx.float32)
    inv_freq = 1.0 / (theta ** (num / head_dim))                     # (head_dim/2,)
    pos = position_ids.astype(mx.float32)                            # (A, T)
    freqs = inv_freq[None, :, None] * pos[:, None, :]                # (A, D/2, T)
    freqs = freqs.transpose(0, 2, 1)                                 # (A, T, D/2)

    if rope_dims is not None and position_ids.shape[0] > 1 and interleaved_mrope:
        # MLX has no in-place index_put, so each axis is merged with a boolean
        # selector over the last dim instead of assigning into a slice.
        inter = freqs[0]
        d = inter.shape[-1]
        cols = mx.arange(d)
        for axis_idx, offset in ((1, 1), (2, 2)):
            length = rope_dims[axis_idx] * 3
            sel = (cols >= offset) & (cols < length) & (((cols - offset) % 3) == 0)
            inter = mx.where(sel, freqs[axis_idx], inter)
        emb = mx.concatenate([inter, inter], axis=-1)                # (T, D)
        cos = mx.expand_dims(mx.cos(emb), 0)
        sin = mx.expand_dims(mx.sin(emb), 0)
    else:
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.expand_dims(mx.cos(emb), 1)
        sin = mx.expand_dims(mx.sin(emb), 1)

    half = sin.shape[-1] // 2
    return cos, sin[..., :half], -sin[..., half:]


def apply_rope(xq: mx.array, xk: mx.array, freqs_cis) -> Tuple[mx.array, mx.array]:
    cos, sin, nsin = freqs_cis
    org = xq.dtype

    def rope(x):
        e = x * cos
        h = e.shape[-1] // 2
        lo = e[..., :h] + x[..., h:] * nsin
        hi = e[..., h:] + x[..., :h] * sin
        return mx.concatenate([lo, hi], axis=-1)

    return rope(xq).astype(org), rope(xk).astype(org)


class Attention(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        inner = self.n_heads * self.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, inner, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(inner, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps, cfg.rms_norm_add)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps, cfg.rms_norm_add)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array, freqs_cis, mask=None) -> mx.array:
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)

        q = self.q_norm(q)          # per-head, before rope
        k = self.k_norm(k)
        q, k = apply_rope(q, k, freqs_cis)

        rep = self.n_heads // self.n_kv
        if rep > 1:                 # repeat_interleave, not tile
            k = mx.repeat(k, rep, axis=1)
            v = mx.repeat(v, rep, axis=1)

        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(o)


class MLP(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def __call__(self, x: mx.array, freqs_cis, mask=None) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), freqs_cis, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


LAYER_KEYS = [
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
]


def load_layer(block: TransformerBlock, weights: dict, prefix: str, dtype=mx.float32) -> None:
    """Copy one layer's 11 tensors out of the checkpoint dict into an MLX block."""
    def g(name):
        return weights[f"{prefix}{name}"].astype(dtype)
    block.input_layernorm.weight = g("input_layernorm.weight")
    block.post_attention_layernorm.weight = g("post_attention_layernorm.weight")
    a = block.self_attn
    a.q_proj.weight = g("self_attn.q_proj.weight")
    a.k_proj.weight = g("self_attn.k_proj.weight")
    a.v_proj.weight = g("self_attn.v_proj.weight")
    a.o_proj.weight = g("self_attn.o_proj.weight")
    a.q_norm.weight = g("self_attn.q_norm.weight")
    a.k_norm.weight = g("self_attn.k_norm.weight")
    m = block.mlp
    m.gate_proj.weight = g("mlp.gate_proj.weight")
    m.up_proj.weight = g("mlp.up_proj.weight")
    m.down_proj.weight = g("mlp.down_proj.weight")


def two_pass_mask(ar_len: int, T: int, dtype=mx.float32) -> Optional[mx.array]:
    """Additive mask for HiDream's split sequence.

    Positions [0, ar_len) are autoregressive and see only themselves and earlier;
    positions [ar_len, T) are the generation half and see everything. ComfyUI
    avoids materialising this by splitting Q at the boundary and running two
    attention calls — worth doing here too at real sequence lengths, but for
    correctness testing an explicit mask is the unambiguous statement of intent.
    """
    if ar_len <= 0:
        return None
    if ar_len >= T:
        return "causal"
    rows = mx.arange(T)[:, None]
    cols = mx.arange(T)[None, :]
    allowed = (rows >= ar_len) | (cols <= rows)      # gen rows see all; ar rows causal
    neg = mx.array(-1e9, dtype=dtype)
    return mx.where(allowed, mx.zeros((T, T), dtype=dtype), neg)[None, None]


class Decoder(nn.Module):
    """The 36-layer Qwen3-VL stack. Embeddings are looked up outside, because
    HiDream scatters image embeds and the timestep embedding into them first."""

    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = [TransformerBlock(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def __call__(self, h: mx.array, freqs_cis, mask=None) -> mx.array:
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        return self.norm(h)


def load_decoder(dec: Decoder, weights: dict, dtype=mx.bfloat16,
                 prefix: str = "model.language_model.") -> None:
    for i, layer in enumerate(dec.layers):
        load_layer(layer, weights, f"{prefix}layers.{i}.", dtype)
    dec.norm.weight = weights[f"{prefix}norm.weight"].astype(dtype)
