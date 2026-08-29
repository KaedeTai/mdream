"""Qwen3-VL vision tower — the reference-image path.

Only needed for editing: text-to-image never calls it. 27 blocks, 1152 wide,
and deliberately unlike the decoder next door — LayerNorm rather than RMSNorm,
fused QKV *with* bias, GELU rather than SwiGLU, plain 2-D rope rather than
interleaved MRoPE. Porting it by analogy with the decoder is how you get a
tower that runs and returns garbage.

Four things that are easy to get wrong and are pinned here:

- **The Conv3d is a linear layer.** kernel == stride == the whole patch, so
  `Conv3d(3, 1152, (2,16,16), stride=(2,16,16))` on a single patch is exactly
  `x @ W.reshape(1152, 1536).T + b`, with the flat axis ordered (C, T, Ph, Pw).
  Writing it as a convolution in MLX would be slower and no more correct.

- **Two different GELUs.** The block MLP uses the tanh approximation
  (`approximate="tanh"`); the patch merger uses the exact erf one. The
  reference writes them differently one screen apart and means it.

- **rope is 2-D and half-width.** head_dim is 72, the rope table is 36 wide
  (h and w contribute 18 each), and it is duplicated to 72 before cos/sin — so
  this is plain rotate-half, not the decoder's 3-axis interleave.

- **deepstack is dead.** The checkpoint carries three `deepstack_merger_list`
  entries, 230 MiB, that inference never reads. They are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class VisionConfig:
    hidden_size: int = 1152
    num_heads: int = 16
    intermediate_size: int = 4304
    depth: int = 27
    patch_size: int = 16
    temporal_patch_size: int = 2
    in_channels: int = 3
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    out_hidden_size: int = 4096
    eps: float = 1e-6
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size ** 2


class PatchEmbed(nn.Module):
    """The Conv3d, as the linear layer it actually is."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.patch_dim, cfg.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


class VisionMLP(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        # tanh approximation here; the merger below uses the exact one
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(x)))


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """rotate-half rope. x is (N, heads, head_dim); cos/sin are (N, 1, head_dim)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = mx.concatenate([-x2, x1], axis=-1)
    return x * cos + rot * sin


class VisionAttention(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(cfg.hidden_size, cfg.hidden_size * 3, bias=True)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array,
                 lengths: Sequence[int]) -> mx.array:
        n = x.shape[0]
        qkv = self.qkv(x).reshape(n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        outs = []
        off = 0
        for length in lengths:
            sl = slice(off, off + length)
            # (1, heads, len, head_dim)
            qs = mx.expand_dims(q[sl].transpose(1, 0, 2), 0)
            ks = mx.expand_dims(k[sl].transpose(1, 0, 2), 0)
            vs = mx.expand_dims(v[sl].transpose(1, 0, 2), 0)
            o = mx.fast.scaled_dot_product_attention(qs, ks, vs, scale=self.scale)
            outs.append(o[0].transpose(1, 0, 2).reshape(length, -1))
            off += length
        return self.proj(mx.concatenate(outs, axis=0))


class VisionBlock(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.eps)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.eps)
        self.attn = VisionAttention(cfg)
        self.mlp = VisionMLP(cfg)

    def __call__(self, x, cos, sin, lengths):
        x = x + self.attn(self.norm1(x), cos, sin, lengths)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.merge_dim = cfg.hidden_size * cfg.spatial_merge_size ** 2
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.eps)
        self.linear_fc1 = nn.Linear(self.merge_dim, self.merge_dim, bias=True)
        self.linear_fc2 = nn.Linear(self.merge_dim, cfg.out_hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x).reshape(-1, self.merge_dim)
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))   # exact gelu


def rope_tables(cfg: VisionConfig, grid_thw: np.ndarray) -> Tuple[mx.array, mx.array]:
    """cos / sin of shape (N, 1, head_dim), matching the reference's layout."""
    dim = cfg.head_dim // 2                       # 36
    inv = 1.0 / (cfg.rope_theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    max_hw = int(max(max(h, w) for _, h, w in grid_thw))
    table = np.outer(np.arange(max_hw, dtype=np.float32), inv)   # (max_hw, dim/2)

    merge = cfg.spatial_merge_size
    rows, cols = [], []
    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)
        mh, mw = h // merge, w // merge
        br = np.arange(mh)[:, None, None, None] * merge + np.arange(merge)[None, None, :, None]
        bc = np.arange(mw)[None, :, None, None] * merge + np.arange(merge)[None, None, None, :]
        br = np.broadcast_to(br, (mh, mw, merge, merge)).reshape(-1)
        bc = np.broadcast_to(bc, (mh, mw, merge, merge)).reshape(-1)
        if t > 1:
            br, bc = np.tile(br, t), np.tile(bc, t)
        rows.append(br)
        cols.append(bc)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    freqs = np.concatenate([table[r], table[c]], axis=-1)        # (N, dim)
    emb = np.concatenate([freqs, freqs], axis=-1)               # (N, head_dim)
    cos = np.cos(emb)[:, None, :]
    sin = np.sin(emb)[:, None, :]
    return mx.array(cos.astype(np.float32)), mx.array(sin.astype(np.float32))


def pos_embed_indices(cfg: VisionConfig, grid_thw: np.ndarray):
    """Bilinear interpolation of the 48x48 learned grid onto this image's grid.

    Returns the four index arrays and their weights, exactly as the reference
    builds them, so the embedding lookup itself stays a single gather.
    """
    side = int(cfg.num_position_embeddings ** 0.5)
    idx = [[] for _ in range(4)]
    wts = [[] for _ in range(4)]
    for _, h, w in grid_thw:
        h, w = int(h), int(w)
        hi = np.linspace(0, side - 1, h, dtype=np.float64).astype(np.float32)
        wi = np.linspace(0, side - 1, w, dtype=np.float64).astype(np.float32)
        hf, wf = hi.astype(np.int32), wi.astype(np.int32)
        hc = np.clip(hf + 1, None, side - 1)
        wc = np.clip(wf + 1, None, side - 1)
        dh, dw = hi - hf, wi - wf
        bh, bhc = hf * side, hc * side
        for j, (a, b) in enumerate(((bh, wf), (bh, wc), (bhc, wf), (bhc, wc))):
            idx[j].append((a[:, None] + b[None, :]).reshape(-1))
        for j, (a, b) in enumerate((((1 - dh), (1 - dw)), ((1 - dh), dw),
                                    (dh, (1 - dw)), (dh, dw))):
            wts[j].append((a[:, None] * b[None, :]).reshape(-1))
    return (np.stack([np.concatenate(v) for v in idx]).astype(np.int64),
            np.stack([np.concatenate(v) for v in wts]).astype(np.float32))


class VisionTower(nn.Module):
    def __init__(self, cfg: Optional[VisionConfig] = None):
        super().__init__()
        cfg = cfg or VisionConfig()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.pos_embed = nn.Embedding(cfg.num_position_embeddings, cfg.hidden_size)
        self.blocks = [VisionBlock(cfg) for _ in range(cfg.depth)]
        self.merger = PatchMerger(cfg)

    def interpolated_pos_embed(self, grid_thw: np.ndarray) -> mx.array:
        cfg = self.cfg
        idx, wts = pos_embed_indices(cfg, grid_thw)
        e = self.pos_embed(mx.array(idx)) * mx.array(wts)[:, :, None]
        pe = e[0] + e[1] + e[2] + e[3]

        merge = cfg.spatial_merge_size
        out, off = [], 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            chunk = pe[off:off + h * w]
            off += h * w
            if t > 1:
                chunk = mx.concatenate([chunk] * t, axis=0)
            chunk = chunk.reshape(t, h // merge, merge, w // merge, merge, -1)
            chunk = chunk.transpose(0, 1, 3, 2, 4, 5).reshape(-1, cfg.hidden_size)
            out.append(chunk)
        return mx.concatenate(out, axis=0)

    def __call__(self, pixel_values: mx.array, grid_thw: np.ndarray,
                 compute_dtype=mx.bfloat16) -> mx.array:
        grid_thw = np.asarray(grid_thw).reshape(-1, 3)
        x = self.patch_embed(pixel_values.astype(compute_dtype))
        x = x + self.interpolated_pos_embed(grid_thw).astype(x.dtype)
        cos, sin = rope_tables(self.cfg, grid_thw)
        cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)
        lengths = [int(t) * int(h) * int(w) for t, h, w in grid_thw]
        for blk in self.blocks:
            x = blk(x, cos, sin, lengths)
        return self.merger(x)


def load_vision(tower: VisionTower, weights: dict, dtype=mx.bfloat16,
                prefix: str = "model.visual.") -> None:
    cfg = tower.cfg
    w = weights[f"{prefix}patch_embed.proj.weight"]
    tower.patch_embed.proj.weight = w.reshape(cfg.hidden_size, cfg.patch_dim).astype(dtype)
    tower.patch_embed.proj.bias = weights[f"{prefix}patch_embed.proj.bias"].astype(dtype)
    tower.pos_embed.weight = weights[f"{prefix}pos_embed.weight"].astype(dtype)
    for i, blk in enumerate(tower.blocks):
        p = f"{prefix}blocks.{i}."
        for dst, src in ((blk.norm1, "norm1"), (blk.norm2, "norm2")):
            dst.weight = weights[p + src + ".weight"].astype(dtype)
            dst.bias = weights[p + src + ".bias"].astype(dtype)
        blk.attn.qkv.weight = weights[p + "attn.qkv.weight"].astype(dtype)
        blk.attn.qkv.bias = weights[p + "attn.qkv.bias"].astype(dtype)
        blk.attn.proj.weight = weights[p + "attn.proj.weight"].astype(dtype)
        blk.attn.proj.bias = weights[p + "attn.proj.bias"].astype(dtype)
        blk.mlp.linear_fc1.weight = weights[p + "mlp.linear_fc1.weight"].astype(dtype)
        blk.mlp.linear_fc1.bias = weights[p + "mlp.linear_fc1.bias"].astype(dtype)
        blk.mlp.linear_fc2.weight = weights[p + "mlp.linear_fc2.weight"].astype(dtype)
        blk.mlp.linear_fc2.bias = weights[p + "mlp.linear_fc2.bias"].astype(dtype)
    m = f"{prefix}merger."
    tower.merger.norm.weight = weights[m + "norm.weight"].astype(dtype)
    tower.merger.norm.bias = weights[m + "norm.bias"].astype(dtype)
    tower.merger.linear_fc1.weight = weights[m + "linear_fc1.weight"].astype(dtype)
    tower.merger.linear_fc1.bias = weights[m + "linear_fc1.bias"].astype(dtype)
    tower.merger.linear_fc2.weight = weights[m + "linear_fc2.weight"].astype(dtype)
    tower.merger.linear_fc2.bias = weights[m + "linear_fc2.bias"].astype(dtype)
