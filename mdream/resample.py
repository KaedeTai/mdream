"""Image resamplers that match torch's, because the edit path resizes three
times before the model ever sees a pixel.

Reference images go through: 2x2 average pooling while very large, then a
**bicubic** fit-and-crop, then a **PIL LANCZOS** resize for the ViT branch, then
a **bilinear** resize inside the Qwen2-VL processor. Any of those being
"close enough" changes the model's input, and there is no later stage that can
tell you it happened.

PIL LANCZOS is free: ComfyUI's `lanczos()` round-trips through uint8 PIL
itself, so calling PIL the same way is exact by construction. The other two are
`torch.nn.functional.interpolate` with `align_corners=False` and no antialias,
which is a well-defined separable kernel:

    src = (dst + 0.5) * (in / out) - 0.5

bilinear clamps that to >= 0 and takes two taps; bicubic does not clamp and
takes four, with the usual A = -0.75 cubic convolution. Both are checked
against torch in tests/test_resample.py.
"""
from __future__ import annotations

import numpy as np

# "numpy" accumulates in float64 and is the more accurate answer; "torch"
# calls torch and reproduces the reference bit for bit. Which you want is not
# obvious, and the difference is not academic: a reference image is quantised
# to uint8 partway through preprocessing, so being *more* accurate here tips
# ~7% of pixels to the other side of a rounding boundary, and the vision tower
# turns that into a 5% change in the embeddings (README, milestone 8c). Use
# "torch" when the goal is to reproduce a ComfyUI result exactly.
_BACKEND = "numpy"


def set_backend(name: str) -> None:
    global _BACKEND
    assert name in ("numpy", "torch"), name
    _BACKEND = name


def get_backend() -> str:
    return _BACKEND


def _torch_resize(x: np.ndarray, out_h: int, out_w: int, mode: str) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
    return F.interpolate(t, size=(out_h, out_w), mode=mode,
                         align_corners=False)[0].numpy()


def avg_pool2x2(x: np.ndarray) -> np.ndarray:
    """(C, H, W) -> (C, H//2, W//2), matching F.avg_pool2d(kernel=2, stride=2)."""
    c, h, w = x.shape
    h2, w2 = h // 2, w // 2
    return x[:, :h2 * 2, :w2 * 2].reshape(c, h2, 2, w2, 2).mean(axis=(2, 4))


def _src_index(out: int, scale: float, clamp: bool) -> np.ndarray:
    src = (np.arange(out, dtype=np.float64) + 0.5) * scale - 0.5
    return np.maximum(src, 0.0) if clamp else src


def _bilinear_weights(inp: int, out: int):
    scale = inp / out
    src = _src_index(out, scale, clamp=True)
    i0 = np.floor(src).astype(np.int64)
    lam = src - i0
    i1 = np.minimum(i0 + 1, inp - 1)
    i0 = np.clip(i0, 0, inp - 1)
    return (i0, i1), (1.0 - lam, lam)


def _cubic_weights(t: np.ndarray, a: float = -0.75):
    """torch's cubic convolution taps for offsets t-(-1), t-0, t-1, t-2."""
    def w0(x):   # |x| <= 1
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0

    def w1(x):   # 1 < |x| < 2
        return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
    return np.stack([w1(t + 1.0), w0(t), w0(1.0 - t), w1(2.0 - t)])


def _bicubic_weights(inp: int, out: int):
    scale = inp / out
    src = _src_index(out, scale, clamp=False)
    i = np.floor(src).astype(np.int64)
    t = src - i
    w = _cubic_weights(t)                       # (4, out)
    idx = np.clip(i[None, :] + np.array([-1, 0, 1, 2])[:, None], 0, inp - 1)
    return idx, w


def _apply_axis(x: np.ndarray, idx, w, axis: int) -> np.ndarray:
    """Sum_k w[k] * x[..., idx[k], ...] along `axis`."""
    out = None
    for k in range(len(idx)):
        term = np.take(x, idx[k], axis=axis)
        shape = [1] * x.ndim
        shape[axis] = -1
        term = term * w[k].reshape(shape)
        out = term if out is None else out + term
    return out


def resize_bilinear(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """(C, H, W) float -> (C, out_h, out_w). F.interpolate(mode='bilinear')."""
    if _BACKEND == "torch":
        return _torch_resize(x, out_h, out_w, "bilinear")
    x = x.astype(np.float64)
    (i0, i1), (w0, w1) = _bilinear_weights(x.shape[-2], out_h)
    x = _apply_axis(x, (i0, i1), (w0, w1), axis=-2)
    (j0, j1), (v0, v1) = _bilinear_weights(x.shape[-1], out_w)
    x = _apply_axis(x, (j0, j1), (v0, v1), axis=-1)
    return x.astype(np.float32)


def resize_bicubic(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """(C, H, W) float -> (C, out_h, out_w). F.interpolate(mode='bicubic').

    No clamping of the result: torch's bicubic overshoots and the reference
    does not clip it either, so neither does this.
    """
    if _BACKEND == "torch":
        return _torch_resize(x, out_h, out_w, "bicubic")
    x = x.astype(np.float64)
    idx, w = _bicubic_weights(x.shape[-2], out_h)
    x = _apply_axis(x, idx, w, axis=-2)
    idx, w = _bicubic_weights(x.shape[-1], out_w)
    x = _apply_axis(x, idx, w, axis=-1)
    return x.astype(np.float32)


def resize_lanczos(x: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """(C, H, W) float in [0, 1] -> same, via PIL, the way ComfyUI does it.

    Note the uint8 round trip: ComfyUI's `lanczos()` quantises to 8 bits before
    resizing and back to float after. That is lossy, and reproducing it is the
    point -- doing it "better" in float would not match.
    """
    from PIL import Image
    img = np.clip(255.0 * x.transpose(1, 2, 0), 0, 255).astype(np.uint8)
    out = Image.fromarray(img).resize((out_w, out_h), resample=Image.Resampling.LANCZOS)
    return (np.array(out).astype(np.float32) / 255.0).transpose(2, 0, 1)
