"""Patch-seam smoothing: cancel the 32x32 grid the model leaves behind.

HiDream-O1 writes its output one 32x32 patch at a time, and the patches do not
quite agree at their borders. On a photograph of skin the result is a visible
grid. Measured as the ratio of gradient energy on the patch boundaries to
everywhere else (1.0 = no grid):

    real photograph          1.02
    mdream bf16              1.23     <- the model's own seam
    ComfyUI bf16             1.25
    mdream 6-bit             1.63     <- quantisation roughly triples it

The trick, taken from ComfyUI's `HiDreamO1PatchSeamSmoothing`: over the last
fraction of sampling, run the model again on an `x` that has been rolled by
half a patch, roll the prediction back, and average. The two runs put their
seams in different places, so averaging cancels them. `ramp_2_4` uses two
offsets early in the gated range and four later, where seams matter most.

Rolling wraps, so a strip one patch wide at each border is contaminated by the
opposite edge. That strip keeps the unshifted prediction, with a 4px feather.
"""
from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple

import mlx.core as mx

PATCH_SIZE = 32
EDGE_FEATHER = 4

SHIFTS_BY_PATTERN = {
    ("single_shift", 2): [(0, 0), (16, 16)],
    ("single_shift", 4): [(0, 0), (16, 0), (0, 16), (16, 16)],
    ("single_shift", 8): [(0, 0), (16, 0), (0, 16), (16, 16),
                          (8, 8), (24, 8), (8, 24), (24, 24)],
    ("symmetric", 2): [(-8, -8), (8, 8)],
    ("symmetric", 4): [(-8, -8), (8, -8), (-8, 8), (8, 8)],
    ("symmetric", 8): [(-12, -12), (4, -12), (-12, 4), (4, 4),
                       (-4, -4), (12, -4), (-4, 12), (12, 12)],
}
RAMP_LEVELS = {
    "2": [2], "4": [4], "ramp_2_4": [2, 4], "ramp_2_4_8": [2, 4, 8],
}


def roll2d(x: mx.array, sy: int, sx: int) -> mx.array:
    """torch.roll over the last two axes. MLX has no roll, so this is two
    concatenates -- which is what roll is anyway."""
    h, w = x.shape[-2], x.shape[-1]
    sy, sx = sy % h, sx % w
    if sy:
        x = mx.concatenate([x[..., h - sy:, :], x[..., :h - sy, :]], axis=-2)
    if sx:
        x = mx.concatenate([x[..., :, w - sx:], x[..., :, :w - sx]], axis=-1)
    return x


def edge_ramp(h: int, w: int, dtype=mx.float32) -> mx.array:
    """1 in the interior, 0 within one patch of the border, feathered between."""
    ys = mx.minimum(mx.arange(h), (h - 1) - mx.arange(h)).astype(mx.float32)
    xs = mx.minimum(mx.arange(w), (w - 1) - mx.arange(w)).astype(mx.float32)
    ym = mx.clip((ys - PATCH_SIZE) / EDGE_FEATHER, 0.0, 1.0)
    xm = mx.clip((xs - PATCH_SIZE) / EDGE_FEATHER, 0.0, 1.0)
    return (ym[:, None] * xm[None, :]).astype(dtype)


class SeamSmoother:
    """Wraps a `net(x, sigma, i) -> velocity` callable.

    Gating is on sigma, matching ComfyUI: `percent_to_sigma` for a CONST flow
    model with shift s is `time_snr_shift(s, 1 - percent)`.
    """

    def __init__(self, shift: float = 3.0, start_percent: float = 0.8,
                 end_percent: float = 1.0, pattern: str = "single_shift",
                 passes: str = "ramp_2_4", blend: str = "average",
                 strength: float = 1.0):
        assert blend in ("average", "median"), blend
        self.levels: List[Sequence[Tuple[int, int]]] = [
            SHIFTS_BY_PATTERN[(pattern, n)] for n in RAMP_LEVELS[passes]
        ]
        self.blend = blend
        self.strength = strength
        self.enabled = strength > 0.0 and end_percent > start_percent

        def pct_to_sigma(p: float) -> float:
            if p <= 0.0:
                return 1.0
            if p >= 1.0:
                return 0.0
            t = 1.0 - p
            return shift * t / (1 + (shift - 1) * t)

        self.start_sigma = pct_to_sigma(start_percent)
        self.end_sigma = pct_to_sigma(end_percent)
        self.extra_calls = 0

    def active(self, sigma: float) -> bool:
        return self.enabled and self.end_sigma <= sigma <= self.start_sigma

    def __call__(self, net: Callable, x: mx.array, sigma: float, i: int) -> mx.array:
        pred = net(x, sigma, i)
        if not self.active(sigma):
            return pred

        if len(self.levels) == 1:
            shifts = self.levels[0]
        else:
            span = max(self.start_sigma - self.end_sigma, 1e-8)
            phase = (self.start_sigma - sigma) / span
            shifts = self.levels[min(int(phase * len(self.levels)), len(self.levels) - 1)]

        preds = []
        for sy, sx in shifts:
            if sy == 0 and sx == 0:
                preds.append(pred)
                continue
            rolled = net(roll2d(x, sy, sx), sigma, i)
            preds.append(roll2d(rolled, -sy, -sx))
            self.extra_calls += 1

        stacked = mx.stack(preds, axis=0)
        avg = mx.median(stacked, axis=0) if self.blend == "median" \
            else mx.mean(stacked, axis=0)

        h, w = pred.shape[-2], pred.shape[-1]
        mask = edge_ramp(h, w, pred.dtype) * self.strength
        return pred * (1.0 - mask) + avg * mask
