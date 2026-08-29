"""Milestone 8b: the resamplers, against torch and PIL.

Preprocessing is not glamorous and it is exactly where a port silently drifts,
so each resampler is checked directly rather than inferred from the final
image. Sizes here are the awkward ones -- upscale, downscale, non-integer
ratios, odd dimensions -- because the easy ratios agree by accident.

The comparison is against torch run in **float64**, not float32. mdream's
resamplers accumulate in float64 and land within 3e-8 of that; torch in
float32 is 2.9e-5 away from its own float64 answer. Comparing float32 to
float32 would have meant asserting a 1e-4 tolerance and never knowing whether
the gap was the algorithm or the accumulation. The float32 column is printed
so the size of that rounding stays visible.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mdream.resample import (avg_pool2x2, resize_bicubic, resize_bilinear,  # noqa: E402
                             resize_lanczos)

CASES = [((3, 64, 64), 32, 32), ((3, 37, 53), 64, 96), ((3, 100, 60), 40, 96),
         ((3, 512, 288), 384, 216), ((3, 17, 19), 33, 5)]


def main() -> int:
    print("milestone 8b - resamplers vs torch / PIL\n")
    rs = np.random.RandomState(5)
    ok = True

    print("  bilinear (align_corners=False)")
    for shape, oh, ow in CASES:
        x = rs.rand(*shape).astype(np.float32)
        a = resize_bilinear(x, oh, ow)
        b64 = F.interpolate(torch.from_numpy(x.astype(np.float64))[None], size=(oh, ow),
                            mode="bilinear", align_corners=False)[0].numpy()
        b32 = F.interpolate(torch.from_numpy(x)[None], size=(oh, ow),
                            mode="bilinear", align_corners=False)[0].numpy()
        e = float(np.abs(a - b64).max())
        ok &= e < 1e-6
        print(f"    {shape[1]}x{shape[2]} -> {oh}x{ow}   vs torch f64 {e:.3e}"
              f"   (torch f32 is {float(np.abs(b32 - b64).max()):.3e} from f64)")

    print("  bicubic (align_corners=False)")
    for shape, oh, ow in CASES:
        x = rs.rand(*shape).astype(np.float32)
        a = resize_bicubic(x, oh, ow)
        b64 = F.interpolate(torch.from_numpy(x.astype(np.float64))[None], size=(oh, ow),
                            mode="bicubic", align_corners=False)[0].numpy()
        b32 = F.interpolate(torch.from_numpy(x)[None], size=(oh, ow),
                            mode="bicubic", align_corners=False)[0].numpy()
        e = float(np.abs(a - b64).max())
        ok &= e < 1e-6
        print(f"    {shape[1]}x{shape[2]} -> {oh}x{ow}   vs torch f64 {e:.3e}"
              f"   (torch f32 is {float(np.abs(b32 - b64).max()):.3e} from f64)")

    print("  lanczos (ComfyUI's uint8 PIL round trip)")
    sys.path.insert(0, str(Path(os.environ.get("MDREAM_COMFYUI",
                                            Path.home() / "ComfyUI")).expanduser()))
    import comfy.utils
    for shape, oh, ow in CASES:
        x = rs.rand(*shape).astype(np.float32)
        a = resize_lanczos(x, ow, oh)
        b = comfy.utils.common_upscale(torch.from_numpy(x)[None], ow, oh,
                                       "lanczos", "disabled")[0].numpy()
        e = float(np.abs(a - b).max())
        ok &= e == 0.0
        print(f"    {shape[1]}x{shape[2]} -> {oh}x{ow}   "
              f"{'exact' if e == 0 else f'max diff {e:.3e}'}")

    print("  2x2 average pooling")
    x = rs.rand(3, 64, 90).astype(np.float32)
    a = avg_pool2x2(x)
    b = F.avg_pool2d(torch.from_numpy(x)[None], 2, 2)[0].numpy()
    e = float(np.abs(a - b).max())
    ok &= e < 1e-6
    print(f"    64x90 -> {a.shape[1]}x{a.shape[2]}   max diff {e:.3e}")

    print("\n  " + ("PASS - resamplers match" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
