"""Pixel shims: the small modules that bolt raw RGB onto the Qwen3-VL sequence.

Five tensors of the checkpoint's 758, but they sit on both ends of the forward
pass, so getting the patch ordering wrong here corrupts everything downstream in
a way that looks like a model bug. The ordering is pinned by the reference's
einops strings:

    unpatch:  B (H W) (C p1 p2) -> B C (H p1) (W p2)
    patchify: the exact inverse

i.e. inside a flat patch vector the channel index is slowest and p2 fastest.
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

PATCH_SIZE = 32


def timestep_embedding(t: mx.array, dim: int = 256, max_period: float = 10000.0) -> mx.array:
    """Sinusoidal embedding. Note the reference concatenates COS FIRST, then sin."""
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t.astype(mx.float32)[:, None] * freqs[None]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
    return emb


class TimestepEmbedder(nn.Module):
    """model.t_embedder1 — Linear(256, 4096) / SiLU / Linear(4096, 4096)."""

    def __init__(self, hidden_size: int = 4096, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp_0 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.mlp_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, t: mx.array) -> mx.array:
        h = timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp_2(nn.silu(self.mlp_0(h)))


class BottleneckPatchEmbed(nn.Module):
    """model.x_embedder — 3072 -> 1024 (no bias) -> 4096 (bias)."""

    def __init__(self, patch_size: int = PATCH_SIZE, in_chans: int = 3,
                 pca_dim: int = 1024, embed_dim: int = 4096):
        super().__init__()
        self.proj1 = nn.Linear(patch_size * patch_size * in_chans, pca_dim, bias=False)
        self.proj2 = nn.Linear(pca_dim, embed_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj2(self.proj1(x))


class FinalLayer(nn.Module):
    """model.final_layer2 — 4096 -> 3072 (bias)."""

    def __init__(self, hidden_size: int = 4096, patch_size: int = PATCH_SIZE, out_channels: int = 3):
        super().__init__()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


def patchify(x: mx.array, p: int = PATCH_SIZE) -> mx.array:
    """(B, C, H, W) -> (B, H/p * W/p, C*p*p), channel slowest inside the patch."""
    B, C, H, W = x.shape
    if H % p or W % p:
        raise ValueError(f"{H}x{W} is not a multiple of patch size {p}")
    hp, wp = H // p, W // p
    x = x.reshape(B, C, hp, p, wp, p)          # B C hp p1 wp p2
    x = x.transpose(0, 2, 4, 1, 3, 5)          # B hp wp C p1 p2
    return x.reshape(B, hp * wp, C * p * p)


def unpatchify(x: mx.array, hp: int, wp: int, p: int = PATCH_SIZE, C: int = 3) -> mx.array:
    """(B, hp*wp, C*p*p) -> (B, C, hp*p, wp*p). Inverse of patchify."""
    B = x.shape[0]
    x = x.reshape(B, hp, wp, C, p, p)          # B hp wp C p1 p2
    x = x.transpose(0, 3, 1, 4, 2, 5)          # B C hp p1 wp p2
    return x.reshape(B, C, hp * p, wp * p)


# checkpoint key -> module attribute, for the three shim modules
SHIM_KEYS = {
    "model.x_embedder.proj1.weight": ("patch_embed", "proj1.weight"),
    "model.x_embedder.proj2.weight": ("patch_embed", "proj2.weight"),
    "model.x_embedder.proj2.bias":   ("patch_embed", "proj2.bias"),
    "model.final_layer2.linear.weight": ("final_layer", "linear.weight"),
    "model.final_layer2.linear.bias":   ("final_layer", "linear.bias"),
    "model.t_embedder1.mlp.0.weight": ("t_embedder", "mlp_0.weight"),
    "model.t_embedder1.mlp.0.bias":   ("t_embedder", "mlp_0.bias"),
    "model.t_embedder1.mlp.2.weight": ("t_embedder", "mlp_2.weight"),
    "model.t_embedder1.mlp.2.bias":   ("t_embedder", "mlp_2.bias"),
}
