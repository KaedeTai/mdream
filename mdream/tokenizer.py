"""Prompt -> input_ids, in HiDream-O1's chat-template form.

The "text encoder" in this model is not a text encoder at all: the Qwen3-VL
backbone inside the diffusion model does the encoding every step, so the only
job here is to produce integer token ids. The template is fixed:

    <|im_start|>user\\n {prompt} <|im_end|>\\n <|im_start|>assistant\\n <|boi|><|tms|>

The last token is <|tms|>, whose embedding is overwritten by the timestep
embedding in the forward pass -- which is why `ar_len` is len(ids) - 1: the tms
token belongs to the generation half, not the autoregressive prefix.

Vocabulary is plain Qwen2 BPE. Verified against ComfyUI's HiDreamO1Tokenizer:
its output is exactly prefix + Qwen2Tokenizer(prompt) + suffix, with no weight
parsing and no padding.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np

IM_START_ID = 151644
IM_END_ID = 151645
ASSISTANT_ID = 77091
USER_ID = 872
NEWLINE_ID = 198
BOI_TOKEN_ID = 151669
TMS_TOKEN_ID = 151673

PREFIX = [IM_START_ID, USER_ID, NEWLINE_ID]
SUFFIX = [IM_END_ID, NEWLINE_ID, IM_START_ID, ASSISTANT_ID, NEWLINE_ID,
          BOI_TOKEN_ID, TMS_TOKEN_ID]

_SEARCH = [
    Path.home() / "ComfyUI/comfy/text_encoders/qwen25_tokenizer",
    Path(__file__).resolve().parent / "qwen25_tokenizer",
]


def default_tokenizer_path() -> Path:
    for p in _SEARCH:
        if (p / "vocab.json").exists():
            return p
    raise FileNotFoundError(
        "no Qwen2 tokenizer found; looked in " + ", ".join(str(p) for p in _SEARCH)
    )


class PromptTokenizer:
    def __init__(self, path: Optional[os.PathLike] = None):
        from transformers import Qwen2Tokenizer  # imported late: heavy
        self.path = Path(path) if path is not None else default_tokenizer_path()
        self.tok = Qwen2Tokenizer.from_pretrained(str(self.path))

    def encode(self, prompt: str) -> np.ndarray:
        body: List[int] = list(self.tok(prompt)["input_ids"])
        return np.array([PREFIX + body + SUFFIX], dtype=np.int64)
