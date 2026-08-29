"""End-to-end text-to-image generation.

Ties together the pieces that were each verified on their own: the tokenizer
(exact vs ComfyUI), the conditioning (exact), the forward pass (8.9e-8 on
synthetic weights) and the sampler (exact Euler, schedule within 2 ULP).

Nothing here does any maths of its own -- that is the point. If the image is
wrong, it is wrong in a piece that has its own test.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

from . import sampling as S
from .conditioning import build_t2i_conds
from .decoder import TextConfig
from .model import HiDreamO1, load_model
from .tokenizer import PromptTokenizer

DEFAULT_CKPT = Path.home() / "models/HiDream-O1-Image/checkpoints/hidream_o1_image_dev_bf16.safetensors"


def torch_style_noise(shape, seed: int) -> np.ndarray:
    """ComfyUI's `prepare_noise`: torch.randn on CPU from torch.manual_seed.

    Reproduced through torch when it is importable, so that a generation can be
    compared against ComfyUI at the same seed. Without torch, falls back to
    MLX's generator -- fine for making images, useless for comparing them.
    """
    try:
        import torch
        g = torch.manual_seed(seed)
        return torch.randn(shape, dtype=torch.float32, generator=g,
                           device="cpu").numpy()
    except ImportError:
        return np.array(mx.random.normal(shape, key=mx.random.key(seed)),
                        dtype=np.float32)


class Generator:
    def __init__(self, ckpt: Optional[Path] = None, dtype=mx.bfloat16,
                 tokenizer_path=None, verbose: bool = True):
        self.dtype = dtype
        self.verbose = verbose
        ckpt = Path(ckpt) if ckpt is not None else DEFAULT_CKPT
        t0 = time.time()
        weights = mx.load(str(ckpt))
        self.model = HiDreamO1(TextConfig())
        load_model(self.model, weights, dtype=dtype)
        mx.eval(self.model.parameters())
        del weights
        self.tokenizer = PromptTokenizer(tokenizer_path)
        self.model_sampling = S.ModelSamplingDiscreteFlow()
        if verbose:
            print(f"  loaded {ckpt.name} in {time.time() - t0:.1f}s")

    def generate(self, prompt: str, width: int = 1024, height: int = 1024,
                 steps: int = 28, seed: int = 0,
                 sigmas: Optional[np.ndarray] = None,
                 noise: Optional[np.ndarray] = None,
                 return_latent: bool = False):
        assert width % 32 == 0 and height % 32 == 0, "patch size is 32"

        ids = self.tokenizer.encode(prompt)
        conds = build_t2i_conds(ids, height, width)
        if sigmas is None:
            sigmas = S.normal_schedule(self.model_sampling, steps)
        if noise is None:
            noise = torch_style_noise((1, 3, height, width), seed)

        x = S.initial_noise_scaling(float(sigmas[0]), noise,
                                    np.zeros_like(noise),
                                    self.model_sampling.noise_scale)
        x = mx.array(x, dtype=mx.float32)

        ids_mx = mx.array(conds["input_ids"])
        pos_mx = mx.array(conds["position_ids"])
        mask_np = conds["vinput_mask"]
        seq_len = ids.shape[1] + conds["image_len"]
        cache = self.model.prepare(pos_mx, mx.array(mask_np), conds["ar_len"],
                                   conds["image_len"], seq_len, self.dtype)

        if self.verbose:
            print(f"  {width}x{height}  {conds['image_len']} image tokens + "
                  f"{ids.shape[1]} text = {seq_len} sequence   "
                  f"{len(sigmas) - 1} steps  seed {seed}")

        t_start = time.time()
        step_times = []

        def net(xc, sigma, i):
            t = mx.array([sigma * self.model_sampling.multiplier], dtype=mx.float32)
            v = self.model(xc, t, ids_mx, pos_mx, None, conds["ar_len"],
                           compute_dtype=self.dtype, cache=cache)
            mx.eval(v)
            return v

        def cb(i, sigma, xc, denoised):
            mx.eval(xc)
            now = time.time()
            step_times.append(now)
            if self.verbose:
                prev = step_times[-2] if len(step_times) > 1 else t_start
                print(f"    step {i + 1:>3}/{len(sigmas) - 1}  sigma {sigma:7.4f}  "
                      f"{now - prev:5.2f}s", flush=True)

        x = S.sample_euler(net, x, sigmas, callback=cb)
        mx.eval(x)
        total = time.time() - t_start
        if self.verbose:
            print(f"  {total:.1f}s total, {total / (len(sigmas) - 1):.2f}s/step")

        latent = np.array(x, dtype=np.float32)
        img = S.to_image(latent)[0]
        return (img, latent) if return_latent else img
