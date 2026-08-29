"""Milestone 8: the Qwen3-VL vision tower, against ComfyUI's own.

The tower is ~455M parameters, small enough to run both stacks in fp32, so
this test does not have to reason about a bf16 envelope the way the 8B decoder
tests did. It runs the same rule anyway: CPU at an fp32 tolerance proves the
arithmetic, GPU is reported beside it.

Stage by stage, because a tower that is wrong in the rope or the pos-embed
interpolation still produces plausible-looking numbers at the output.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(os.environ.get("MDREAM_COMFYUI",
                                            Path.home() / "ComfyUI")).expanduser()))

from mdream.vision import VisionConfig, VisionTower, load_vision, rope_tables  # noqa: E402

import comfy.ops  # noqa: E402
from comfy.ldm.hidream_o1.model import QWEN3VL_VISION_DEFAULTS  # noqa: E402
from comfy.text_encoders.qwen35 import Qwen35VisionModel  # noqa: E402

CKPT = Path(os.environ.get(
    "MDREAM_CKPT",
    Path.home() / "models/HiDream-O1-Image/checkpoints/hidream_o1_image_dev_bf16.safetensors")).expanduser()
GRID = np.array([[1, 32, 44]], dtype=np.int64)          # 1408 patches -> 352 tokens


def to_np(a):
    return np.array(a.astype(mx.float32), copy=False) if isinstance(a, mx.array) \
        else a.detach().float().cpu().numpy()


def rel(a, b) -> float:
    a, b = to_np(a).astype(np.float64), to_np(b).astype(np.float64)
    assert a.shape == b.shape, f"{a.shape} != {b.shape}"
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def build_reference():
    cfg = dict(QWEN3VL_VISION_DEFAULTS)
    ref = Qwen35VisionModel(cfg, device="cpu", dtype=torch.float32,
                            ops=comfy.ops.manual_cast).eval()
    return ref


def main() -> int:
    print("milestone 8 - Qwen3-VL vision tower, fp32\n")
    # Standing rule from milestone 3: CPU at an fp32 tolerance proves the
    # arithmetic, because MLX's CPU GEMM is accurate and logic errors cannot
    # hide there. Metal's fp32 GEMM is ~300x looser (notes/precision.md), so
    # the GPU number is reported beside it, not asserted against fp32.
    mx.set_default_device(mx.cpu)
    n = int(GRID[0, 0] * GRID[0, 1] * GRID[0, 2])
    rs = np.random.RandomState(11)
    px = (rs.randn(n, VisionConfig().patch_dim) * 0.5).astype(np.float32)

    W = mx.load(str(CKPT))
    mine = VisionTower()
    load_vision(mine, W, dtype=mx.float32)
    mx.eval(mine.parameters())

    ref = build_reference()
    sd = {k[len("model.visual."):]:
          torch.from_numpy(np.array(v.astype(mx.float32), copy=False))
          for k, v in W.items() if k.startswith("model.visual.")}
    missing, unexpected = ref.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected if "deepstack" not in k]
    print(f"  reference load: missing={len(missing)} unexpected(non-deepstack)={len(unexpected)}")
    if missing or unexpected:
        print(f"    {missing[:4]} {unexpected[:4]}")
    del W

    px_t = torch.from_numpy(px)
    grid_t = torch.from_numpy(GRID)
    ok = True
    tol = 2e-5

    # --- stage 1: patch embed (the Conv3d-as-linear claim) -------------------
    a = mine.patch_embed(mx.array(px), )
    with torch.no_grad():
        b = ref.patch_embed(px_t)
    e = rel(a, b)
    ok &= e < tol
    print(f"  {'OK ' if e < tol else 'BAD'} patch embed (Conv3d as a linear)   {e:.3e}")

    # --- stage 2: interpolated position embedding ---------------------------
    a = mine.interpolated_pos_embed(GRID)
    with torch.no_grad():
        b = ref.fast_pos_embed_interpolate(grid_t)
    e = rel(a, b)
    ok &= e < tol
    print(f"  {'OK ' if e < tol else 'BAD'} pos embed interpolation           {e:.3e}")

    # --- stage 3: rope tables ----------------------------------------------
    cos, sin = rope_tables(VisionConfig(), GRID)
    with torch.no_grad():
        rpe = ref.rot_pos_emb(grid_t).reshape(n, -1)
        emb = torch.cat((rpe, rpe), dim=-1)
        bcos = emb.cos().unsqueeze(-2)
        bsin = emb.sin().unsqueeze(-2)
    e = max(rel(cos, bcos), rel(sin, bsin))
    ok &= e < tol
    print(f"  {'OK ' if e < tol else 'BAD'} rope cos/sin                      {e:.3e}")

    # --- stage 4: one block -------------------------------------------------
    x0 = mx.array((rs.randn(n, 1152) * 0.4).astype(np.float32))
    lengths = [n]
    a = mine.blocks[0](x0, cos, sin, lengths)
    cu = torch.tensor([0, n], dtype=torch.int32)
    sin_t = bsin
    pe = (bcos, sin_t[..., :sin_t.shape[-1] // 2], -sin_t[..., sin_t.shape[-1] // 2:])
    from comfy.ldm.modules.attention import optimized_attention_for_device
    oa = optimized_attention_for_device(torch.device("cpu"), mask=False, small_input=True)
    with torch.no_grad():
        b = ref.blocks[0](torch.from_numpy(to_np(x0)), cu_seqlens=cu,
                          position_embeddings=pe, optimized_attention=oa)
    e = rel(a, b)
    ok &= e < tol
    print(f"  {'OK ' if e < tol else 'BAD'} block 0                           {e:.3e}")

    # --- stage 5: the whole tower ------------------------------------------
    a_cpu = mine(mx.array(px), GRID, compute_dtype=mx.float32)
    mx.eval(a_cpu)
    with torch.no_grad():
        b = ref(px_t, grid_t)
    tol_full = 2e-4      # 27 blocks of accumulation, still 4x inside Metal fp32
    e_cpu = rel(a_cpu, b)
    ok &= e_cpu < tol_full
    print(f"  {'OK ' if e_cpu < tol_full else 'BAD'} full tower, CPU fp32             "
          f"{e_cpu:.3e}   (tol {tol_full:.0e}, shape {tuple(a_cpu.shape)})")

    mx.set_default_device(mx.gpu)
    mine_gpu = VisionTower()
    W2 = mx.load(str(CKPT))
    load_vision(mine_gpu, W2, dtype=mx.float32)
    mx.eval(mine_gpu.parameters())
    del W2
    a_gpu = mine_gpu(mx.array(px), GRID, compute_dtype=mx.float32)
    mx.eval(a_gpu)
    print(f"      full tower, GPU fp32             {rel(a_gpu, b):.3e}"
          f"   <- Metal GEMM, see notes/precision.md")
    a_bf = mine_gpu(mx.array(px), GRID, compute_dtype=mx.bfloat16)
    mx.eval(a_bf)
    print(f"      full tower, GPU bf16             {rel(a_bf, b):.3e}"
          f"   <- what the shipped path runs at")

    print("\n  " + ("PASS - the vision tower matches the reference"
                    if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
