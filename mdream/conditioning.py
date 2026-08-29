"""Sequence assembly for the text-to-image path.

Everything here is integer index arithmetic, so it is written in numpy and
compared against the reference exactly rather than to a tolerance. This is the
part a port gets wrong quietly: the position ids are 3-axis MRoPE with a
"fix point" that jumps the image block to a fixed coordinate, and ar_len — the
causal/full attention boundary — is derived here.

Layout for T2I:

    [ text tokens .................. ][ VISION_START, IMAGE_PAD x (N-1) ]
      ^ ar_len = len(text) - 1                ^ vinput_mask starts here

ar_len sits one before the end of the text because the last text token is the
<|tms_token|> whose embedding is replaced by the timestep embedding, and it
belongs to the generation half.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

VISION_START_ID = 151652
VISION_END_ID = 151653
IMAGE_TOKEN_ID = 151655
TMS_TOKEN_ID = 151673
PATCH_SIZE = 32


def get_rope_index_fix_point(input_ids: np.ndarray, image_grid_thw: np.ndarray,
                             skip_vision_start_token, spatial_merge_size: int = 1,
                             fix_point: int = 4096) -> np.ndarray:
    """(3, B, T) MRoPE position ids. Port of the reference's function of the same
    name; only the batch-of-1, no-attention-mask path is exercised here."""
    B, T = input_ids.shape
    position_ids = np.ones((3, B, T), dtype=np.int64)

    for i in range(B):
        ids = input_ids[i]
        fp = fix_point
        image_index = 0
        vision_start_indices = np.argwhere(ids == VISION_START_ID).squeeze(1)
        vision_tokens = ids[vision_start_indices + 1]
        image_nums = int((vision_tokens == IMAGE_TOKEN_ID).sum())
        tokens = ids.tolist()
        chunks = []
        st = 0
        remain = image_nums
        for _ in range(image_nums):
            ed = tokens.index(IMAGE_TOKEN_ID, st) if (IMAGE_TOKEN_ID in tokens and remain > 0) \
                else len(tokens) + 1
            t, h, w = (int(v) for v in image_grid_thw[image_index])
            image_index += 1
            remain -= 1
            gt, gh, gw = t, h // spatial_merge_size, w // spatial_merge_size

            text_len = max(0, (ed - st) - skip_vision_start_token[image_index - 1])
            st_idx = int(chunks[-1].max()) + 1 if chunks else 0
            chunks.append(np.tile(np.arange(text_len), (3, 1)) + st_idx)

            t_index = np.repeat(np.arange(gt), gh * gw)
            h_index = np.tile(np.repeat(np.arange(gh), gw), gt)
            w_index = np.tile(np.arange(gw), gt * gh)
            block = np.stack([t_index, h_index, w_index])

            if skip_vision_start_token[image_index - 1]:
                if fp > 0:
                    fp = fp - st_idx
                chunks.append(block + fp + st_idx)
                fp = 0
            else:
                chunks.append(block + text_len + st_idx)
            st = ed + gt * gh * gw

        if st < len(tokens):
            st_idx = int(chunks[-1].max()) + 1 if chunks else 0
            chunks.append(np.tile(np.arange(len(tokens) - st), (3, 1)) + st_idx)

        position_ids[:, i, :] = np.concatenate(chunks, axis=1).reshape(3, -1)
    return position_ids


def build_t2i_conds(text_input_ids: np.ndarray, height: int, width: int,
                    patch_size: int = PATCH_SIZE) -> dict:
    """Assemble input_ids, position_ids, vinput_mask and ar_len for T2I."""
    if text_input_ids.ndim == 1:
        text_input_ids = text_input_ids[None]
    B, txt_len = text_input_ids.shape
    h_p, w_p = height // patch_size, width // patch_size
    image_len = h_p * w_p

    vision = np.full((B, image_len), IMAGE_TOKEN_ID, dtype=np.int64)
    vision[:, 0] = VISION_START_ID
    input_ids_pad = np.concatenate([text_input_ids, vision], axis=-1)

    grid = np.array([[1, h_p, w_p]], dtype=np.int64)
    position_ids = get_rope_index_fix_point(input_ids_pad, grid, skip_vision_start_token=[1])

    total_len = txt_len + image_len
    ar_len = txt_len - 1
    vinput_mask = np.zeros((B, total_len), dtype=bool)
    vinput_mask[:, txt_len:] = True
    token_types = np.zeros((B, total_len), dtype=np.int64)
    token_types[:, ar_len:] = 1

    return {
        "input_ids": text_input_ids,
        "input_ids_pad": input_ids_pad,
        "position_ids": position_ids[:, 0],      # (3, T)
        "vinput_mask": vinput_mask,
        "token_types": token_types,
        "ar_len": ar_len,
        "grid": (h_p, w_p),
        "image_len": image_len,
    }
