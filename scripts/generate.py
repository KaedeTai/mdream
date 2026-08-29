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
    a = ap.parse_args()

    g = Generator(Path(a.ckpt), dtype=getattr(mx, a.dtype))
    img, latent = g.generate(
        a.prompt, width=a.width, height=a.height, steps=a.steps, seed=a.seed,
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
