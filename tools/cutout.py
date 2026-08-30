#!/usr/bin/env python3
"""Background removal by segmentation, for comparison with the edit path.

This is not generation. It predicts an alpha mask and keeps the original
pixels underneath, so the subject comes out byte-for-byte unchanged -- which
is exactly what HiDream-O1's edit path cannot do, because it redraws the whole
canvas from the reference rather than compositing.

Runs u2net_human_seg directly through onnxruntime; no rembg install needed,
the weights are already cached in ~/.u2net.

    cutout photo.png --white
    cutout photo.png -o mask.png --model u2net_human_seg
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODELS = Path.home() / ".u2net"

# Normalisation is per-model and getting it wrong is silent: the mask comes
# back plausible but worse. These match rembg's session classes.
CONFIG = {
    "u2net":             dict(size=(320, 320),   mean=(0.485, 0.456, 0.406),
                              std=(0.229, 0.224, 0.225)),
    "u2net_human_seg":   dict(size=(320, 320),   mean=(0.485, 0.456, 0.406),
                              std=(0.229, 0.224, 0.225)),
    "isnet-general-use": dict(size=(1024, 1024), mean=(0.5, 0.5, 0.5),
                              std=(1.0, 1.0, 1.0)),
}
# u2net_cloth_seg is deliberately absent: it emits 4 class channels (upper /
# lower / full body), not an alpha, so it needs different post-processing.


def predict_alpha(img: Image.Image, model: str = "isnet-general-use") -> Image.Image:
    cfg = CONFIG[model]
    sess = ort.InferenceSession(str(MODELS / f"{model}.onnx"),
                                providers=["CPUExecutionProvider"])
    mean = np.array(cfg["mean"], dtype=np.float32)
    std = np.array(cfg["std"], dtype=np.float32)

    small = np.array(img.convert("RGB").resize(cfg["size"], Image.LANCZOS),
                     dtype=np.float32)
    small = small / max(small.max(), 1e-6)
    x = ((small - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)

    pred = sess.run(None, {sess.get_inputs()[0].name: x})[0][:, 0, :, :]
    lo, hi = pred.min(), pred.max()
    pred = (pred - lo) / max(hi - lo, 1e-6)
    return pred[0]


def stretch(pred: np.ndarray, lo: float = 0.05, hi: float = 0.95) -> np.ndarray:
    """Saturate the confident parts of the mask, keep a soft transition band.

    IS-Net's probabilities rarely reach 1.0 -- straight out of the model only
    0.1% of pixels land on alpha 255, so compositing washes the whole subject
    toward the background. Stretching fixes that and restores the property that
    matters: inside the subject, the output pixels are the input pixels.
    """
    return np.clip((pred - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default=None,
                    help="RGBA output (default: <input>_cutout.png beside the input)")
    ap.add_argument("--white", nargs="?", const="__auto__", default=None,
                    help="also write a white-backed version; bare flag names it "
                         "<input>_white.png")
    ap.add_argument("--model", default="isnet-general-use",
                    choices=list(CONFIG),
                    help="isnet runs at 1024x1024 and gives much crisper edges; "
                         "u2net_human_seg is 320x320 and 3x faster")
    ap.add_argument("--raw-alpha", action="store_true",
                    help="skip the alpha stretch and use the model's raw probabilities")
    ap.add_argument("--lo", type=float, default=0.05)
    ap.add_argument("--hi", type=float, default=0.95)
    a = ap.parse_args()

    src = Path(a.image)
    # Derived defaults, so `cutout photo.png --white` needs no other arguments.
    if a.out is None:
        a.out = str(src.with_name(src.stem + "_cutout.png"))
    if a.white == "__auto__":
        a.white = str(src.with_name(src.stem + "_white.png"))

    img = Image.open(a.image).convert("RGB")
    t0 = time.time()
    pred = predict_alpha(img, a.model)
    if not a.raw_alpha:
        pred = stretch(pred, a.lo, a.hi)
    alpha = Image.fromarray((pred * 255).astype(np.uint8)).resize(
        img.size, Image.LANCZOS)
    dt = time.time() - t0

    rgba = img.copy().convert("RGBA")
    rgba.putalpha(alpha)
    rgba.save(a.out)

    if a.white:
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, (0, 0), alpha)
        bg.save(a.white)

    al = np.array(alpha)
    print(f"  {a.model}  {img.size[0]}x{img.size[1]}  {dt:.2f}s  "
          f"subject {al.mean() / 255 * 100:.1f}% of frame, "
          f"{(al == 255).mean() * 100:.1f}% fully opaque, "
          f"{((al > 10) & (al < 245)).mean() * 100:.2f}% soft edge")
    print(f"  wrote {a.out}" + (f" and {a.white}" if a.white else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
