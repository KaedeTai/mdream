"""Quantisation — the reason this implementation exists.

ComfyUI on this machine cannot quantise HiDream-O1 at all: the fp8 checkpoint
dies on MPS with `Undefined type Float8_e4m3fn`, because PyTorch's Metal
backend has no fp8 kernels. MLX quantises natively, so this is the one place
where the port buys something a wrapper could not.

What is quantised and what is not:

  quantised   the 36 decoder layers -- q/k/v/o and gate/up/down. 14.2 of the
              15.2 GiB, and 100% of the per-step compute.
  optional    embed_tokens, 1.24 GiB. It only ever sees ~24 text tokens, so it
              is cheap to quantise and cheap to keep; off by default because
              it is a lookup, not a matmul, so it costs bandwidth once.
  never       the pixel shims -- x_embedder, t_embedder, final_layer. 60 MiB
              between them, and final_layer writes the image directly: every
              bit of error there lands in the output rather than being
              averaged over 36 residual additions.

The norms stay in fp32 regardless (they already do -- see decoder.rms_norm).
"""
from __future__ import annotations

from typing import Iterable, Optional

import mlx.core as mx
import mlx.nn as nn

# never quantised: small, and directly on the pixel path
PIXEL_SHIMS = ("x_embedder", "t_embedder", "final_layer")


def make_predicate(quantize_embed: bool = False,
                   skip: Iterable[str] = PIXEL_SHIMS,
                   group_size: int = 64,
                   overrides: Optional[dict] = None):
    """`overrides` maps a substring of the module path to {"bits", "group_size"}.

    Mixed precision is not a nicety here. Flat 4-bit affine visibly darkens
    every image (see README); keeping the layers that carry the outlier
    channels at 8 bits is what makes 4-bit usable at all.
    """
    skip = tuple(skip)
    overrides = overrides or {}

    def predicate(path: str, module: nn.Module):
        if any(path == s or path.startswith(s + ".") for s in skip):
            return False
        if isinstance(module, nn.Embedding):
            return quantize_embed
        if not isinstance(module, nn.Linear):
            return False
        for key, cfg in overrides.items():
            if key in path:
                gs = cfg.get("group_size", group_size)
                if module.weight.shape[-1] % gs != 0:
                    return False
                return dict(cfg)
        # MLX needs the contracted dimension to be a multiple of group_size
        if module.weight.shape[-1] % group_size != 0:
            return False
        return True

    return predicate


def quantize_model(model: nn.Module, bits: int = 4, group_size: int = 64,
                   quantize_embed: bool = False,
                   skip: Optional[Iterable[str]] = None,
                   overrides: Optional[dict] = None) -> nn.Module:
    nn.quantize(model, group_size=group_size, bits=bits,
                class_predicate=make_predicate(quantize_embed,
                                               PIXEL_SHIMS if skip is None else skip,
                                               group_size, overrides))
    mx.eval(model.parameters())
    return model


def parameter_bytes(model: nn.Module) -> int:
    from mlx.utils import tree_flatten
    return sum(v.nbytes for _, v in tree_flatten(model.parameters())
               if isinstance(v, mx.array))


def save_quantized(model: nn.Module, path, bits: int, group_size: int,
                   quantize_embed: bool = False,
                   overrides: Optional[dict] = None) -> None:
    """Write a quantised checkpoint plus the metadata needed to rebuild it.

    The quantisation config has to travel with the weights: `nn.quantize` has
    to run with the *same* predicate before `load_weights`, or the shapes will
    not line up and the failure is a confusing shape error rather than
    "you loaded a 4-bit file as 8-bit".
    """
    import json

    from mlx.utils import tree_flatten
    flat = {k: v for k, v in tree_flatten(model.parameters())
            if isinstance(v, mx.array)}
    meta = {"mdream_quant": json.dumps({
        "bits": bits, "group_size": group_size,
        "quantize_embed": quantize_embed, "skip": list(PIXEL_SHIMS),
        "overrides": overrides or {},
    })}
    mx.save_safetensors(str(path), flat, metadata=meta)


def quant_config(path) -> Optional[dict]:
    """Read the quantisation config out of a checkpoint, or None if it is a
    plain one. Cheap: reads the safetensors header only."""
    import json
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    meta = hdr.get("__metadata__", {})
    return json.loads(meta["mdream_quant"]) if "mdream_quant" in meta else None
