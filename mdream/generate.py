"""End-to-end text-to-image generation.

Ties together the pieces that were each verified on their own: the tokenizer
(exact vs ComfyUI), the conditioning (exact), the forward pass (8.9e-8 on
synthetic weights) and the sampler (exact Euler, schedule within 2 ULP).

Nothing here does any maths of its own -- that is the point. If the image is
wrong, it is wrong in a piece that has its own test.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import sampling as S
from .conditioning import build_t2i_conds
from .refimg import build_edit_conds
from .decoder import TextConfig
from .model import HiDreamO1, load_model
from .vision import VisionTower, load_vision
from .tokenizer import PromptTokenizer

from .paths import DEFAULT_CKPT  # noqa: F401  (re-exported)


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
                 tokenizer_path=None, verbose: bool = True,
                 bits: Optional[int] = None, group_size: int = 64,
                 quantize_embed: bool = False,
                 overrides: Optional[dict] = None):
        self.dtype = dtype
        self.verbose = verbose
        ckpt = Path(ckpt) if ckpt is not None else DEFAULT_CKPT
        t0 = time.time()
        from .quantize import (parameter_bytes, quant_config, quantize_model,
                               make_predicate, PIXEL_SHIMS)
        qc = quant_config(ckpt)
        if qc is not None:
            # already-quantised checkpoint: build the same shapes, then load
            self.model = HiDreamO1(TextConfig())
            nn.quantize(self.model, group_size=qc["group_size"], bits=qc["bits"],
                        class_predicate=make_predicate(qc["quantize_embed"],
                                                       qc.get("skip", PIXEL_SHIMS),
                                                       qc["group_size"],
                                                       qc.get("overrides")))
            w = mx.load(str(ckpt))
            self.model.load_weights([(k, v) for k, v in w.items()
                                     if not k.startswith("visual.")])
            mx.eval(self.model.parameters())
            if qc.get("has_vision_tower"):
                tw = VisionTower()
                tw.load_weights([(k[len("visual."):], v) for k, v in w.items()
                                 if k.startswith("visual.")])
                mx.eval(tw.parameters())
                self._packed_tower = tw
            del w
            self.param_bytes = parameter_bytes(self.model)
            self.quant = (qc["bits"], qc["group_size"], qc["quantize_embed"])
            self.tokenizer = PromptTokenizer(tokenizer_path)
            self.model_sampling = S.ModelSamplingDiscreteFlow()
            self.ckpt = ckpt
            self._tower = getattr(self, "_packed_tower", None)
            if verbose:
                print(f"  loaded {ckpt.name} ({qc['bits']}-bit g{qc['group_size']}, "
                      f"{self.param_bytes / 2**30:.2f} GiB) in {time.time() - t0:.1f}s")
            return
        weights = mx.load(str(ckpt))
        self.model = HiDreamO1(TextConfig())
        load_model(self.model, weights, dtype=dtype)
        mx.eval(self.model.parameters())
        del weights
        before = parameter_bytes(self.model)
        if bits is not None:
            quantize_model(self.model, bits=bits, group_size=group_size,
                           quantize_embed=quantize_embed, overrides=overrides)
            after = parameter_bytes(self.model)
            if verbose:
                extra = " +embed" if quantize_embed else ""
                if overrides:
                    extra += " " + ",".join(f"{k}@{v['bits']}"
                                            for k, v in overrides.items())
                print(f"  quantised to {bits}-bit g{group_size}{extra}: "
                      f"{before / 2**30:.2f} -> {after / 2**30:.2f} GiB "
                      f"({before / after:.2f}x)")
        self.param_bytes = parameter_bytes(self.model)
        self.quant = (bits, group_size, quantize_embed) if bits else None
        self.tokenizer = PromptTokenizer(tokenizer_path)
        self.model_sampling = S.ModelSamplingDiscreteFlow()
        self.ckpt = ckpt
        self._tower = None
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


    def tower(self) -> VisionTower:
        """The vision tower, loaded on first use.

        Text-to-image never touches it, and it is a separate 0.9 GiB, so it
        stays out of the way until an edit asks for it.
        """
        if self._tower is None:
            t0 = time.time()
            w = mx.load(str(self.ckpt))
            tw = VisionTower()
            load_vision(tw, w, dtype=self.dtype)
            mx.eval(tw.parameters())
            del w
            self._tower = tw
            if self.verbose:
                print(f"  vision tower loaded in {time.time() - t0:.1f}s")
        return self._tower

    def edit(self, prompt: str, refs, width: int = 768, height: int = 1024,
             steps: int = 28, seed: int = 0, cfg: float = 5.0,
             negative_prompt: str = "",
             sampler: str = "dpmpp_2m",
             sigmas: Optional[np.ndarray] = None,
             noise: Optional[np.ndarray] = None,
             match_comfy: bool = False,
             return_latent: bool = False):
        """Reference-image editing.

        `refs` are (H, W, 3) float arrays in [0, 1]. The vision tower runs once
        for the whole generation, not once per step -- its output does not
        depend on sigma, and recomputing it 28 times would roughly double the
        cost of an edit.
        """
        assert width % 32 == 0 and height % 32 == 0, "patch size is 32"
        if "dev" in Path(self.ckpt).name and self.verbose:
            print("  WARNING: this is the dev (distilled) checkpoint. On anything "
                  "involving skin it returns speckled, crazed faces -- and often a "
                  "before/after diptych. Use the base checkpoint for portraits; "
                  "dev is fine for clothing, background and objects.")
        if width * height < 1_700_000 and self.verbose:
            print(f"  WARNING: {width}x{height} is {width * height / 1e6:.2f} MP. "
                  "The edit path collapses to noise below ~1.7 MP -- ComfyUI does "
                  "the same, at every sampler, cfg and checkpoint. Use 1152x1536 "
                  "or larger.")
        from .refimg import prepare_ref_images
        ids = self.tokenizer.encode(prompt)
        ref = prepare_ref_images(refs, height, width, match_comfy=match_comfy)
        conds = build_edit_conds(ids, height, width, refs, match_comfy=match_comfy,
                                 ref=ref)
        # cfg 1.0 is not a usable setting for editing: ComfyUI itself returns
        # pure noise there. The uncond pass is built whenever cfg != 1.
        unc = None
        if not math.isclose(cfg, 1.0):
            unc = build_edit_conds(self.tokenizer.encode(negative_prompt),
                                   height, width, refs,
                                   match_comfy=match_comfy, ref=ref)
        if sigmas is None:
            sigmas = S.normal_schedule(self.model_sampling, steps)
        if noise is None:
            noise = torch_style_noise((1, 3, height, width), seed)

        x = mx.array(S.initial_noise_scaling(float(sigmas[0]), noise,
                                             np.zeros_like(noise),
                                             self.model_sampling.noise_scale),
                     dtype=mx.float32)

        t_vit = time.time()
        image_embeds = self.tower()(mx.array(conds["ref_pixel_values"]),
                                    conds["ref_image_grid_thw"],
                                    compute_dtype=self.dtype)
        mx.eval(image_embeds)
        t_vit = time.time() - t_vit

        ids_mx = mx.array(conds["input_ids"])
        pos_mx = mx.array(conds["position_ids"])
        ref_patches = mx.array(conds["ref_patches"])
        seq_len = conds["input_ids_pad"].shape[1]
        cache = self.model.prepare(pos_mx, mx.array(conds["vinput_mask"]),
                                   conds["ar_len"], conds["image_len"], seq_len,
                                   self.dtype)
        if self.verbose:
            print(f"  {width}x{height}  {seq_len} sequence "
                  f"({conds['image_len']} target + {ref_patches.shape[1]} ref patches"
                  f" + {image_embeds.shape[0]} ViT tokens)   "
                  f"{len(sigmas) - 1} steps  seed {seed}   ViT {t_vit:.1f}s")

        t_start = time.time()

        if unc is not None:
            unc_ids = mx.array(unc["input_ids"])
            unc_pos = mx.array(unc["position_ids"])
            unc_cache = self.model.prepare(
                unc_pos, mx.array(unc["vinput_mask"]), unc["ar_len"],
                unc["image_len"], unc["input_ids_pad"].shape[1], self.dtype)

        def net(xc, sigma, i):
            t = mx.array([sigma * self.model_sampling.multiplier], dtype=mx.float32)
            v = self.model(xc, t, ids_mx, pos_mx, None, conds["ar_len"],
                           compute_dtype=self.dtype, cache=cache,
                           ref_patches=ref_patches, image_embeds=image_embeds)
            if unc is not None:
                vu = self.model(xc, t, unc_ids, unc_pos, None, unc["ar_len"],
                                compute_dtype=self.dtype, cache=unc_cache,
                                ref_patches=ref_patches, image_embeds=image_embeds)
                # denoised = x - v * sigma is affine in v, so combining the
                # velocities is the same as ComfyUI combining the denoised
                # predictions -- one fewer place to get a sign wrong.
                v = vu + (v - vu) * cfg
            mx.eval(v)
            return v

        x = S.SAMPLERS[sampler](net, x, sigmas)
        mx.eval(x)
        if self.verbose:
            total = time.time() - t_start
            print(f"  {total:.1f}s total, {total / (len(sigmas) - 1):.2f}s/step")

        latent = np.array(x, dtype=np.float32)
        img = S.to_image(latent)[0]
        return (img, latent) if return_latent else img
