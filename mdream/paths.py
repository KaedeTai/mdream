"""Where things live, with environment overrides.

Defaults match a typical ComfyUI install on macOS. Nothing here is required to
be in these places; set the environment variables instead:

    MDREAM_CKPT      path to the HiDream-O1 checkpoint (.safetensors)
    MDREAM_COMFYUI   path to a ComfyUI checkout -- only needed to run the
                     tests, which compare against ComfyUI's own code, and for
                     the Qwen2 tokenizer files it ships
    MDREAM_TOKENIZER path to a directory with vocab.json / merges.txt, if you
                     do not have ComfyUI
"""
from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else default


COMFYUI = _env("MDREAM_COMFYUI", Path.home() / "ComfyUI")
DEFAULT_CKPT = _env(
    "MDREAM_CKPT",
    Path.home() / "models/HiDream-O1-Image/checkpoints/hidream_o1_image_dev_bf16.safetensors",
)
TOKENIZER_DIRS = [
    p for p in (
        _env("MDREAM_TOKENIZER", COMFYUI / "comfy/text_encoders/qwen25_tokenizer"),
        COMFYUI / "comfy/text_encoders/qwen25_tokenizer",
        Path(__file__).resolve().parent / "qwen25_tokenizer",
    )
]
