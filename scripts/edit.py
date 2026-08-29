#!/usr/bin/env python3
"""mdream reference-image editing.

    python3 scripts/edit.py "change the sweater to dark green" -i photo.png \
        -o out.png --width 1152 --height 1536 --cfg 5.0

Two settings are not really optional:

  --cfg must be > 1. At cfg 1.0 the edit path returns noise -- and so does
  ComfyUI, so this is the model, not the port.

  the canvas must be >= ~1.7 MP. HiDream-O1 was trained at ~4 MP and the edit
  path does not merely degrade below that, it collapses. 768x1024 is noise;
  1152x1536 is clean.

  use the BASE checkpoint for anything with skin in it. The distilled `dev`
  variant returns faces covered in dark speckles with a crazed texture. This
  script defaults to base for that reason.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from mdream.generate import Generator  # noqa: E402
from mdream.paths import DEFAULT_EDIT_CKPT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    # prompt first, then -i for each reference image: a greedy nargs="+" for
    # the images would swallow the prompt.
    ap.add_argument("prompt")
    ap.add_argument("-i", "--image", action="append", required=True,
                    help="reference image (repeat for more than one)")
    ap.add_argument("-o", "--out", default="edit.png")
    ap.add_argument("--width", type=int, default=1152)
    ap.add_argument("--height", type=int, default=1536)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--negative", default="")
    ap.add_argument("--sampler", default="dpmpp_2m", choices=["dpmpp_2m", "euler"])
    ap.add_argument("--ckpt", default=str(DEFAULT_EDIT_CKPT),
                    help="defaults to the BASE checkpoint: the distilled dev one "
                         "wrecks skin (see README)")
    ap.add_argument("--match-comfy", action="store_true",
                    help="route the preprocessing resizes through torch, "
                         "bit-identical to ComfyUI")
    a = ap.parse_args()

    refs = [np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            for p in a.image]
    g = Generator(Path(a.ckpt))
    img = g.edit(a.prompt, refs, width=a.width, height=a.height, steps=a.steps,
                 seed=a.seed, cfg=a.cfg, negative_prompt=a.negative,
                 sampler=a.sampler, match_comfy=a.match_comfy)
    Image.fromarray(img).save(a.out)
    print(f"  wrote {a.out}  {img.shape[1]}x{img.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
