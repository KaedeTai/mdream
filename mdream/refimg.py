"""Reference-image preprocessing and the edit-path sequence.

Every reference image travels two roads at once and they do not agree with
each other about anything:

  32-patch road   fit to a patch-aligned size, map to [-1, 1], cut into 32x32
                  patches, and append to the noised target's patch stream.
  ViT road        resize *again* (lanczos) to a ~384px-area box, normalise with
                  mean/std 0.5, cut into 16x16 patches, run the vision tower,
                  and scatter the merged tokens into input_ids at the
                  <|image_pad|> positions.

So a ref image is resized three times with three different kernels before the
model sees it. `mdream/resample.py` matches all three against torch and PIL;
this file is the geometry and the token bookkeeping on top.

The one thing worth flagging: the number of <|image_pad|> tokens spliced into
input_ids has to equal the vision tower's output count exactly. The reference
raises if they disagree, and that check is the real test of whether the whole
chain of roundings above was ported correctly.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from .conditioning import (IMAGE_TOKEN_ID, VISION_END_ID, VISION_START_ID,
                           get_rope_index_fix_point)
from .resample import avg_pool2x2, resize_bicubic, resize_bilinear, resize_lanczos

PATCH_SIZE = 32
CONDITION_IMAGE_SIZE = 384
VIT_PATCH = 16
VIT_MERGE = 2
VIT_MEAN = 0.5
VIT_STD = 0.5


def ref_max_size(target_max_dim: int, k: int) -> int:
    if k == 1:
        return target_max_dim
    if k == 2:
        return target_max_dim * 48 // 64
    if k <= 4:
        return target_max_dim // 2
    if k <= 8:
        return target_max_dim * 24 // 64
    return target_max_dim // 4


def cond_image_size(k: int) -> int:
    if k <= 4:
        return CONDITION_IMAGE_SIZE
    if k <= 8:
        return CONDITION_IMAGE_SIZE * 48 // 64
    return CONDITION_IMAGE_SIZE // 2


def calculate_dimensions(max_size: int, ratio: float) -> Tuple[int, int]:
    width = math.sqrt(max_size * max_size * ratio)
    height = width / ratio
    return int(width / 32) * 32, int(height / 32) * 32


def resize_tensor(img: np.ndarray, image_size: int, patch_size: int = 16) -> np.ndarray:
    """(3, H, W) in [0, 1] -> patch-aligned, area-fitted, centre-cropped."""
    while min(img.shape[-2], img.shape[-1]) >= 2 * image_size:
        img = avg_pool2x2(img)

    height, width = img.shape[-2], img.shape[-1]
    m = patch_size
    s_max = image_size * image_size
    scale = math.sqrt(s_max / (width * height))

    candidates = [
        (round(width * scale) // m * m, round(height * scale) // m * m),
        (round(width * scale) // m * m, math.floor(height * scale) // m * m),
        (math.floor(width * scale) // m * m, round(height * scale) // m * m),
        (math.floor(width * scale) // m * m, math.floor(height * scale) // m * m),
    ]
    candidates = sorted(candidates, key=lambda x: x[0] * x[1], reverse=True)
    new_size = candidates[-1]
    for c in candidates:
        if c[0] * c[1] <= s_max:
            new_size = c
            break

    new_w, new_h = new_size
    s1, s2 = width / new_w, height / new_h
    if s1 < s2:
        resize_w, resize_h = new_w, round(height / s1)
    else:
        resize_w, resize_h = round(width / s2), new_h
    img = resize_bicubic(img, resize_h, resize_w)
    top, left = (resize_h - new_h) // 2, (resize_w - new_w) // 2
    return img[..., top:top + new_h, left:left + new_w]


def process_vit_image(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Port of process_qwen2vl_images for the single-image, no-pixel-cap case.

    Returns (flat_patches (N, 1536), grid_thw (3,)).
    """
    _, height, width = img.shape
    factor = VIT_PATCH * VIT_MERGE                     # 32
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    resized = resize_bilinear(img, h_bar, w_bar)
    normalized = (resized - VIT_MEAN) / VIT_STD

    grid_h, grid_w = h_bar // VIT_PATCH, w_bar // VIT_PATCH
    # temporal patch size 2: the still image is duplicated, not interpolated
    pv = np.stack([normalized, normalized])            # (2, 3, H, W)
    patches = pv.reshape(1, 2, 3,
                         grid_h // VIT_MERGE, VIT_MERGE, VIT_PATCH,
                         grid_w // VIT_MERGE, VIT_MERGE, VIT_PATCH)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = patches.reshape(grid_h * grid_w, 3 * 2 * VIT_PATCH * VIT_PATCH)
    return flat.astype(np.float32), np.array([1, grid_h, grid_w], dtype=np.int64)


def prepare_ref_images(refs: Sequence[np.ndarray], target_h: int, target_w: int,
                       match_comfy: bool = False) -> dict:
    """refs are (H, W, 3) float arrays in [0, 1], as loaded from disk.

    `match_comfy` routes the two interpolations through torch so the result is
    bit-identical to ComfyUI's. Off by default: it is only worth a torch
    dependency when you are reproducing a specific ComfyUI output.
    """
    from . import resample
    prev = resample.get_backend()
    resample.set_backend("torch" if match_comfy else "numpy")
    try:
        return _prepare_ref_images(refs, target_h, target_w)
    finally:
        resample.set_backend(prev)


def _prepare_ref_images(refs: Sequence[np.ndarray], target_h: int, target_w: int) -> dict:
    k = len(refs)
    if k == 0:
        return None
    max_size = ref_max_size(max(target_h, target_w), k)
    cis = cond_image_size(k)

    imgs = [np.clip(r, 0.0, 1.0).transpose(2, 0, 1).astype(np.float32) for r in refs]
    imgs = [resize_tensor(t, max_size, PATCH_SIZE) for t in imgs]

    patches_per, patch_grids = [], []
    for t in imgs:
        t_norm = (t - 0.5) / 0.5
        h_p, w_p = t_norm.shape[-2] // PATCH_SIZE, t_norm.shape[-1] // PATCH_SIZE
        patch_grids.append((h_p, w_p))
        p = t_norm.reshape(3, h_p, PATCH_SIZE, w_p, PATCH_SIZE)
        p = p.transpose(1, 3, 0, 2, 4).reshape(h_p * w_p, 3 * PATCH_SIZE * PATCH_SIZE)
        patches_per.append(p)

    pv_list, grid_list, vit_tokens = [], [], []
    for t in imgs:
        _, h, w = t.shape
        cond_w, cond_h = calculate_dimensions(cis, w / h)
        cond_w = max(cond_w, VIT_PATCH * VIT_MERGE)
        cond_h = max(cond_h, VIT_PATCH * VIT_MERGE)
        t_v = resize_lanczos(t, cond_w, cond_h)
        pv, grid = process_vit_image(t_v)
        pv_list.append(pv)
        grid_list.append(grid)
        vit_tokens.append((int(grid[1]) // VIT_MERGE) * (int(grid[2]) // VIT_MERGE))

    return {
        "ref_patches": np.concatenate(patches_per, axis=0)[None],
        "ref_pixel_values": np.concatenate(pv_list, axis=0),
        "ref_image_grid_thw": np.stack(grid_list, axis=0),
        "per_ref_vit_tokens": vit_tokens,
        "per_ref_patch_grids": patch_grids,
    }


def build_ref_input_ids(text_input_ids: np.ndarray, per_ref_vit_tokens: List[int]) -> np.ndarray:
    """Splice [vision_start, image_pad * n, vision_end] in after the three-token
    chat prefix [im_start, user, newline] -- which is where the original chat
    template puts the image, and why the constant 3 is not arbitrary."""
    ids = text_input_ids[0].tolist()
    inserted: List[int] = []
    for n_pad in per_ref_vit_tokens:
        inserted += [VISION_START_ID] + [IMAGE_TOKEN_ID] * n_pad + [VISION_END_ID]
    return np.array([ids[:3] + inserted + ids[3:]], dtype=np.int64)


def build_edit_conds(text_input_ids: np.ndarray, height: int, width: int,
                     refs: Sequence[np.ndarray], patch_size: int = PATCH_SIZE,
                     match_comfy: bool = False, ref: dict = None) -> dict:
    """The edit-path counterpart of conditioning.build_t2i_conds.

    `ref` lets a caller pass in an already-preprocessed reference bundle. CFG
    needs two sequences (positive and negative prompt) over the *same* images,
    and preprocessing them twice would be both slow and, if any rounding
    differed, wrong.
    """
    if text_input_ids.ndim == 1:
        text_input_ids = text_input_ids[None]
    h_p, w_p = height // patch_size, width // patch_size
    image_len = h_p * w_p

    if ref is None:
        ref = prepare_ref_images(refs, height, width, match_comfy=match_comfy)
    ids = build_ref_input_ids(text_input_ids, ref["per_ref_vit_tokens"])
    txt_len = ids.shape[1]

    ref_lengths = [hp * wp for hp, wp in ref["per_ref_patch_grids"]]
    tgt_vision = np.full((1, image_len), IMAGE_TOKEN_ID, dtype=np.int64)
    tgt_vision[:, 0] = VISION_START_ID
    blocks = [tgt_vision]
    for rl in ref_lengths:
        blk = np.full((1, rl), IMAGE_TOKEN_ID, dtype=np.int64)
        blk[:, 0] = VISION_START_ID
        blocks.append(blk)
    input_ids_pad = np.concatenate([ids] + blocks, axis=1)

    k = len(refs)
    igthw_cond = ref["ref_image_grid_thw"].copy()
    igthw_cond[:, 1] //= 2
    igthw_cond[:, 2] //= 2
    igthw_all = np.concatenate([
        igthw_cond,
        np.array([[1, h_p, w_p]], dtype=np.int64),
        np.array([[1, hp, wp] for hp, wp in ref["per_ref_patch_grids"]], dtype=np.int64),
    ], axis=0)
    position_ids = get_rope_index_fix_point(
        input_ids_pad, igthw_all,
        skip_vision_start_token=[0] * k + [1] + [1] * k,
        spatial_merge_size=1, fix_point=4096)

    total_len = txt_len + image_len + sum(ref_lengths)
    ar_len = txt_len - 1
    vinput_mask = np.zeros((1, total_len), dtype=bool)
    vinput_mask[:, txt_len:] = True
    token_types = np.zeros((1, total_len), dtype=np.int64)
    token_types[:, ar_len:] = 1

    return {
        "input_ids": ids,
        "input_ids_pad": input_ids_pad,
        "position_ids": position_ids[:, 0],
        "vinput_mask": vinput_mask,
        "token_types": token_types,
        "ar_len": ar_len,
        "grid": (h_p, w_p),
        "image_len": image_len,
        "ref_patches": ref["ref_patches"],
        "ref_pixel_values": ref["ref_pixel_values"],
        "ref_image_grid_thw": ref["ref_image_grid_thw"],
    }
