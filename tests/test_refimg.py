"""Milestone 8c: reference-image preprocessing and the edit-path sequence.

Compared against ComfyUI's `build_extra_conds` on a real photograph, because
the whole chain is a sequence of roundings -- area fits, patch alignment,
floor-vs-round candidate selection, three resizes -- and synthetic inputs pick
the easy branches.

The integer results (token ids, position ids, masks, ar_len) are compared
*exactly*; there is no tolerance that makes sense for them. The float results
carry the resamplers' float64-vs-float32 difference, which
tests/test_resample.py already measured at ~3e-5.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(os.environ.get("MDREAM_COMFYUI",
                                            Path.home() / "ComfyUI")).expanduser()))

from mdream.refimg import build_edit_conds  # noqa: E402
from mdream.tokenizer import PromptTokenizer  # noqa: E402

from comfy.ldm.hidream_o1.conditioning import build_extra_conds  # noqa: E402

# Any photograph works; point MDREAM_REF_IMAGE at one. The chain being
# tested is a sequence of roundings that depends on the image's dimensions,
# not its content, so use something that is not square and not tiny.
IMG = Path(os.environ["MDREAM_REF_IMAGE"]).expanduser() \
    if "MDREAM_REF_IMAGE" in os.environ else None
PROMPT = "make the sweater dark green and keep everything else identical"
H, W = 1024, 768


def load(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def main() -> int:
    print("milestone 8c - ref-image preprocessing vs ComfyUI\n")
    if IMG is None or not IMG.exists():
        print("  set MDREAM_REF_IMAGE to a photograph to run this test")
        return 0
    img = load(IMG)
    ids = PromptTokenizer().encode(PROMPT)
    print(f"  reference image {img.shape[1]}x{img.shape[0]}, target {W}x{H}, "
          f"{ids.shape[1]} text tokens")

    mine = build_edit_conds(ids, H, W, [img], match_comfy=True)

    noise = torch.zeros(1, 3, H, W)
    ref_t = [torch.from_numpy(img)[None]]
    ref = build_extra_conds(torch.from_numpy(ids), noise, ref_images=ref_t,
                            target_patch_size=32)

    ok = True

    def exact(name, a, b):
        nonlocal ok
        a, b = np.asarray(a), b.numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
        same = a.shape == b.shape and np.array_equal(a, b)
        ok &= same
        print(f"  {'OK ' if same else 'BAD'} {name:<24} {str(a.shape):<18}"
              f"{'exact' if same else f'shapes {a.shape} vs {b.shape}'}")

    def close(name, a, b, tol):
        nonlocal ok
        a = np.asarray(a, dtype=np.float64)
        b = (b.numpy() if isinstance(b, torch.Tensor) else np.asarray(b)).astype(np.float64)
        if a.shape != b.shape:
            ok = False
            print(f"  BAD {name:<24} shape {a.shape} vs {b.shape}")
            return
        e = float(np.abs(a - b).max())
        ok &= e < tol
        print(f"  {'OK ' if e < tol else 'BAD'} {name:<24} {str(a.shape):<18}"
              f"max diff {e:.3e}  (tol {tol:.0e})")

    exact("input_ids", mine["input_ids"], ref["input_ids"])
    exact("position_ids", mine["position_ids"], ref["position_ids"][0])
    exact("vinput_mask", mine["vinput_mask"], ref["vinput_mask"])
    exact("token_types", mine["token_types"], ref["token_types"])
    same_ar = mine["ar_len"] == int(ref["ar_len"])
    ok &= same_ar
    print(f"  {'OK ' if same_ar else 'BAD'} {'ar_len':<24} "
          f"{mine['ar_len']} vs {int(ref['ar_len'])}")
    exact("ref_image_grid_thw", mine["ref_image_grid_thw"], ref["ref_image_grid_thw"][0])

    # match_comfy=True routes the two interpolations through torch, so these
    # are expected to be bit-identical, not merely close.
    close("ref_patches", mine["ref_patches"], ref["ref_patches"], 1e-12)
    close("ref_pixel_values", mine["ref_pixel_values"], ref["ref_pixel_values"][0], 1e-12)

    n_pad = int((mine["input_ids"][0] == 151655).sum())
    g = mine["ref_image_grid_thw"][0]
    n_vit = (int(g[1]) // 2) * (int(g[2]) // 2)
    match = n_pad == n_vit
    ok &= match
    print(f"  {'OK ' if match else 'BAD'} {'image_pad == ViT count':<24} "
          f"{n_pad} vs {n_vit}   <- the reference raises if these differ")

    print("\n  " + ("PASS - the edit-path conditioning matches" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
