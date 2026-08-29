"""Milestone 2: the pixel shims match PyTorch on the real weights.

This is also where the parity harness itself gets built, so later milestones
only have to supply a module and an input. Both frameworks run on the same
random input in fp32 with the checkpoint's own weights.

Tolerances come from notes/precision.md, which measured the floors on this
machine first: bf16 round-trip is 2.0e-3 and MLX's fp32 matmul is 8.0e-4, so
"matches" means "differs by less than the precision the model actually runs
at", not "matches torch bit for bit". Each result prints its actual error so a
number creeping upward is visible rather than hidden behind a threshold.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mdream import layers as L  # noqa: E402

CKPT = Path(os.environ.get(
    "MDREAM_CKPT",
    Path.home() / "models/HiDream-O1-Image/checkpoints/hidream_o1_image_dev_bf16.safetensors")).expanduser()
WANT = list(L.SHIM_KEYS)


def load_shim_weights():
    w = mx.load(str(CKPT))
    out = {k: w[k] for k in WANT}
    mx.eval(list(out.values()))
    return out


def to_np(a) -> np.ndarray:
    if isinstance(a, mx.array):
        return np.array(a.astype(mx.float32), copy=False)
    return a.detach().float().cpu().numpy()


def rel_err(a, b) -> float:
    a, b = to_np(a).astype(np.float64), to_np(b).astype(np.float64)
    if a.shape != b.shape:
        raise AssertionError(f"shape {a.shape} != {b.shape}")
    denom = max(np.abs(b).max(), 1e-12)
    return float(np.abs(a - b).max() / denom)


BF16_FLOOR = 2.0e-3   # measured: see notes/precision.md


def report(name: str, err: float, tol: float, fails: list) -> None:
    ok = err <= tol
    frac = err / BF16_FLOOR
    print(f"  {'OK ' if ok else 'FAIL'} {name:34s} max rel err {err:.3e}  "
          f"({frac:6.2%} of the bf16 floor, tol {tol:.0e})")
    if not ok:
        fails.append(name)


def main() -> int:
    torch.manual_seed(0)
    W = load_shim_weights()
    tw = {k: torch.from_numpy(to_np(v)) for k, v in W.items()}
    fails: list = []

    print("milestone 2 — pixel shims\n")

    # --- patch embed ---
    m = L.BottleneckPatchEmbed()
    m.proj1.weight = W["model.x_embedder.proj1.weight"].astype(mx.float32)
    m.proj2.weight = W["model.x_embedder.proj2.weight"].astype(mx.float32)
    m.proj2.bias = W["model.x_embedder.proj2.bias"].astype(mx.float32)
    x = np.random.RandomState(1).randn(2, 7, 3072).astype(np.float32)
    got = m(mx.array(x))
    ref = torch.nn.functional.linear(
        torch.nn.functional.linear(torch.from_numpy(x), tw["model.x_embedder.proj1.weight"]),
        tw["model.x_embedder.proj2.weight"], tw["model.x_embedder.proj2.bias"])
    report("BottleneckPatchEmbed", rel_err(got, ref), 2e-3, fails)

    # --- final layer ---
    f = L.FinalLayer()
    f.linear.weight = W["model.final_layer2.linear.weight"].astype(mx.float32)
    f.linear.bias = W["model.final_layer2.linear.bias"].astype(mx.float32)
    h = np.random.RandomState(2).randn(2, 5, 4096).astype(np.float32)
    got = f(mx.array(h))
    ref = torch.nn.functional.linear(torch.from_numpy(h), tw["model.final_layer2.linear.weight"],
                                     tw["model.final_layer2.linear.bias"])
    report("FinalLayer", rel_err(got, ref), 2e-3, fails)

    # --- timestep embedding (the cos-then-sin order) ---
    t = np.array([0.0, 1.0, 137.5, 999.0], dtype=np.float32)
    got = L.timestep_embedding(mx.array(t), 256)
    half = 128
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = torch.from_numpy(t)[:, None] * freqs[None]
    ref = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    report("timestep_embedding", rel_err(got, ref), 2e-3, fails)

    te = L.TimestepEmbedder()
    te.mlp_0.weight = W["model.t_embedder1.mlp.0.weight"].astype(mx.float32)
    te.mlp_0.bias = W["model.t_embedder1.mlp.0.bias"].astype(mx.float32)
    te.mlp_2.weight = W["model.t_embedder1.mlp.2.weight"].astype(mx.float32)
    te.mlp_2.bias = W["model.t_embedder1.mlp.2.bias"].astype(mx.float32)
    got = te(mx.array(t))
    r = torch.nn.functional.linear(ref, tw["model.t_embedder1.mlp.0.weight"], tw["model.t_embedder1.mlp.0.bias"])
    r = torch.nn.functional.silu(r)
    r = torch.nn.functional.linear(r, tw["model.t_embedder1.mlp.2.weight"], tw["model.t_embedder1.mlp.2.bias"])
    report("TimestepEmbedder", rel_err(got, r), 2e-3, fails)

    # --- patch ordering, checked against einops directly ---
    import einops
    img = np.random.RandomState(3).randn(2, 3, 64, 96).astype(np.float32)
    got = L.patchify(mx.array(img))
    ref = einops.rearrange(torch.from_numpy(img), "B C (H p1) (W p2) -> B (H W) (C p1 p2)", p1=32, p2=32)
    report("patchify vs einops", rel_err(got, ref), 0.0, fails)

    flat = np.random.RandomState(4).randn(2, 2 * 3, 3072).astype(np.float32)
    got = L.unpatchify(mx.array(flat), hp=2, wp=3)
    ref = einops.rearrange(torch.from_numpy(flat), "B (H W) (C p1 p2) -> B C (H p1) (W p2)",
                           H=2, W=3, p1=32, p2=32)
    report("unpatchify vs einops", rel_err(got, ref), 0.0, fails)

    rt = L.unpatchify(L.patchify(mx.array(img)), hp=2, wp=3)
    report("patchify round trip", rel_err(rt, torch.from_numpy(img)), 0.0, fails)

    print(f"\n  bf16 round-trip floor on this machine: {BF16_FLOOR:.1e}")
    if fails:
        print(f"  FAIL — {len(fails)} check(s): {fails}")
        return 1
    print("  PASS — pixel shims match the PyTorch reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
