"""HiDream-O1 forward pass (text-to-image path).

Assembly order is taken from the reference and matters:

    z            = patchify(x)                       raw RGB -> flat patches
    inputs_embeds= embed_tokens(input_ids)           text only
    t_emb        = t_embedder1((1 - sigma) * 1000)   note: 1 - sigma, not sigma
    inputs_embeds[input_ids == tms] = t_emb          timestep rides on one token
    inputs_embeds= cat(inputs_embeds, x_embedder(z)) text then pixels
    hidden       = decoder(inputs_embeds, mrope, two_pass_mask(ar_len))
    target       = hidden[vinput_mask][:tgt_image_len]
    x_pred       = unpatchify(final_layer2(target))
    return (x - x_pred) / max(sigma, 1e-3)           in fp32, deliberately

The last line is fp32 on purpose: the reference carries a comment that bf16
there noticeably degrades samples, so it stays fp32 here regardless of the
compute dtype.
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

import numpy as np

from .conditioning import IMAGE_TOKEN_ID, TMS_TOKEN_ID
from .decoder import Decoder, TextConfig, precompute_freqs_cis, two_pass_mask
from .layers import BottleneckPatchEmbed, FinalLayer, TimestepEmbedder, patchify, unpatchify

PATCH_SIZE = 32


def scatter_rows(dst: mx.array, mask: mx.array, rows: mx.array) -> mx.array:
    """Write `rows` into `dst` (1, T, D) at the True positions of `mask` (T,).

    MLX has no in-place index assignment, so this is done as a gather plus a
    select: `cumsum(mask) - 1` gives, at every position, the index of the row
    that belongs there, and `where` keeps it only where the mask says so. The
    True positions are consumed in order, which is what the reference's
    `inputs_embeds[:, image_idx] = image_embeds` does.
    """
    order = mx.maximum(mx.cumsum(mask.astype(mx.int32)) - 1, 0)
    return mx.where(mask[None, :, None], mx.take(rows, order, axis=0)[None], dst)


class HiDreamO1(nn.Module):
    def __init__(self, cfg: Optional[TextConfig] = None, patch_size: int = PATCH_SIZE,
                 in_channels: int = 3):
        super().__init__()
        cfg = cfg or TextConfig()
        self.cfg = cfg
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.tms_token_id = TMS_TOKEN_ID

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.decoder = Decoder(cfg)
        self.t_embedder = TimestepEmbedder(cfg.hidden_size)
        # the reference ties pca_dim to hidden_size // 4
        self.x_embedder = BottleneckPatchEmbed(patch_size, in_channels,
                                               cfg.hidden_size // 4, cfg.hidden_size)
        self.final_layer = FinalLayer(cfg.hidden_size, patch_size, in_channels)

    def prepare(self, position_ids: mx.array, vinput_mask: mx.array, ar_len: int,
                tgt_len: int, seq_len: int, compute_dtype=mx.bfloat16) -> dict:
        """Everything that depends only on the geometry, not on x or sigma.

        The rope frequencies, the two-pass mask and the target indices are the
        same at every sampling step; recomputing them per step costs a Python
        loop over the whole sequence and a mask build, which is pure waste over
        28 steps. The sampler builds this once.
        """
        freqs = precompute_freqs_cis(self.cfg.head_dim, position_ids, self.cfg.rope_theta,
                                     self.cfg.rope_dims, self.cfg.interleaved_mrope)
        return {
            "freqs": tuple(t.astype(compute_dtype) for t in freqs),
            "mask": two_pass_mask(ar_len, seq_len, compute_dtype),
            "idx": mx.array([i for i, v in enumerate(vinput_mask.tolist()[0]) if v][:tgt_len]),
        }

    def __call__(self, x: mx.array, timesteps: mx.array, input_ids: mx.array,
                 position_ids: mx.array, vinput_mask: mx.array, ar_len: int,
                 compute_dtype=mx.bfloat16, cache: Optional[dict] = None,
                 ref_patches: Optional[mx.array] = None,
                 image_embeds: Optional[mx.array] = None) -> mx.array:
        """`ref_patches` and `image_embeds` are the edit path.

        A reference image enters twice and in two different currencies: as
        32x32 raw patches appended to the noised target's patch stream, and as
        vision-tower tokens scattered into the text sequence at the
        <|image_pad|> positions. The pixel half is longer than the target as a
        result, which is why the target slice below takes only the first
        `tgt_len` vision positions and drops the rest.
        """
        B, _, H, W = x.shape
        h_p, w_p = H // self.patch_size, W // self.patch_size
        tgt_len = h_p * w_p

        z = patchify(x, self.patch_size).astype(compute_dtype)
        if ref_patches is not None:
            z = mx.concatenate([z, ref_patches.astype(compute_dtype)], axis=1)
        emb = self.embed_tokens(input_ids).astype(compute_dtype)

        if image_embeds is not None:
            is_pad = (input_ids[0] == IMAGE_TOKEN_ID)
            n_pad = int(np.asarray(is_pad).sum())
            if n_pad != image_embeds.shape[0]:
                raise ValueError(
                    f"image_pad count {n_pad} != vision tower output "
                    f"{image_embeds.shape[0]}; preprocessing is misaligned")
            emb = scatter_rows(emb, is_pad, image_embeds.astype(compute_dtype))

        sigma = timesteps.astype(mx.float32) / 1000.0
        t_emb = self.t_embedder(((1.0 - sigma) * 1000.0)).astype(compute_dtype)
        is_tms = (input_ids == self.tms_token_id)[..., None]
        emb = mx.where(is_tms, mx.expand_dims(t_emb, 1), emb)

        h = mx.concatenate([emb, self.x_embedder(z)], axis=1)
        T = h.shape[1]

        if cache is None:
            cache = self.prepare(position_ids, vinput_mask, ar_len, tgt_len, T,
                                 compute_dtype)
        h = self.decoder(h, cache["freqs"], cache["mask"])

        # target positions are the pixel half; take the first tgt_len of them
        target = mx.take(h, cache["idx"], axis=1)

        x_pred = unpatchify(self.final_layer(target).astype(mx.float32),
                            h_p, w_p, self.patch_size, self.in_channels)
        s = mx.maximum(sigma, 1e-3).reshape(B, 1, 1, 1)
        return (x.astype(mx.float32) - x_pred) / s


def load_model(model: HiDreamO1, weights: dict, dtype=mx.bfloat16,
               prefix: str = "model.") -> None:
    from .decoder import load_decoder
    model.embed_tokens.weight = weights[f"{prefix}language_model.embed_tokens.weight"].astype(dtype)
    load_decoder(model.decoder, weights, dtype, f"{prefix}language_model.")
    model.t_embedder.mlp_0.weight = weights[f"{prefix}t_embedder1.mlp.0.weight"].astype(dtype)
    model.t_embedder.mlp_0.bias = weights[f"{prefix}t_embedder1.mlp.0.bias"].astype(dtype)
    model.t_embedder.mlp_2.weight = weights[f"{prefix}t_embedder1.mlp.2.weight"].astype(dtype)
    model.t_embedder.mlp_2.bias = weights[f"{prefix}t_embedder1.mlp.2.bias"].astype(dtype)
    model.x_embedder.proj1.weight = weights[f"{prefix}x_embedder.proj1.weight"].astype(dtype)
    model.x_embedder.proj2.weight = weights[f"{prefix}x_embedder.proj2.weight"].astype(dtype)
    model.x_embedder.proj2.bias = weights[f"{prefix}x_embedder.proj2.bias"].astype(dtype)
    model.final_layer.linear.weight = weights[f"{prefix}final_layer2.linear.weight"].astype(dtype)
    model.final_layer.linear.bias = weights[f"{prefix}final_layer2.linear.bias"].astype(dtype)
