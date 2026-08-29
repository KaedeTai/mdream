#!/usr/bin/env python3
"""mdream text-to-image CLI.

    python3 scripts/generate.py "a red fox in fresh snow" -o fox.png \
        --width 768 --height 1024 --steps 28 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from mdream.generate import DEFAULT_CKPT, Generator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("-o", "--out", default="out.png")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--sigmas", default=None,
                    help="npy file of sigmas, to reproduce another sampler exactly")
    ap.add_argument("--noise", default=None,
                    help="npy file of (1,3,H,W) unit noise, to match another run")
    ap.add_argument("--save-latent", default=None)
    ap.add_argument("--bits", type=int, default=None, choices=[2, 3, 4, 5, 6, 8],
                    help="quantise the decoder to this many bits")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--quantize-embed", action="store_true")
    ap.add_argument("--seam", default=None,
                    choices=["2", "4", "ramp_2_4", "ramp_2_4_8"],
                    help="patch-seam smoothing; off by default for text-to-image")
    ap.add_argument("--save-quantized", default=None,
                    help="write the quantised weights here and exit")
    ap.add_argument("--hi-bits", type=int, default=None,
                    help="bits for the modules named by --hi-modules")
    ap.add_argument("--hi-modules", default="down_proj",
                    help="comma-separated path substrings kept at --hi-bits")
    a = ap.parse_args()

    overrides = ({m: {"bits": a.hi_bits} for m in a.hi_modules.split(",")}
                 if a.hi_bits else None)
    g = Generator(Path(a.ckpt), dtype=getattr(mx, a.dtype), bits=a.bits,
                  group_size=a.group_size, quantize_embed=a.quantize_embed,
                  overrides=overrides)
    if a.save_quantized:
        from mdream.quantize import save_quantized
        assert a.bits, "--save-quantized needs --bits"
        save_quantized(g.model, a.save_quantized, a.bits, a.group_size,
                       a.quantize_embed, overrides, tower=g.tower())
        print(f"  wrote {a.save_quantized}  {Path(a.save_quantized).stat().st_size / 2**30:.2f} GiB")
        return 0

    img, latent = g.generate(
        a.prompt, width=a.width, height=a.height, steps=a.steps, seed=a.seed,
        seam=a.seam,
        sigmas=np.load(a.sigmas) if a.sigmas else None,
        noise=np.load(a.noise) if a.noise else None,
        return_latent=True,
    )

    from PIL import Image
    Image.fromarray(img).save(a.out)
    print(f"  wrote {a.out}  {img.shape[1]}x{img.shape[0]}")
    if a.save_latent:
        np.save(a.save_latent, latent)
        print(f"  wrote {a.save_latent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
