"""Milestone 4a: the two-pass attention boundary.

ar_len splits the sequence: [0, ar_len) is causal, [ar_len, T) attends
everything. Off by one here leaks future tokens into the autoregressive prefix,
which does not crash and does not obviously look wrong — so it gets its own test
on a small config, against ComfyUI's own two-pass callable.
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

from mdream import decoder as D  # noqa: E402
from comfy.ldm.hidream_o1.attention import make_two_pass_attention  # noqa: E402


def to_np(a):
    return np.array(a.astype(mx.float32), copy=False) if isinstance(a, mx.array) \
        else a.detach().float().cpu().numpy()


def rel_err(a, b) -> float:
    a, b = to_np(a).astype(np.float64), to_np(b).astype(np.float64)
    assert a.shape == b.shape, f"{a.shape} != {b.shape}"
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def main() -> int:
    mx.set_default_device(mx.cpu)          # logic check: accurate GEMM
    fails = []
    B, H, T, Dh = 1, 4, 16, 32
    rs = np.random.RandomState(11)
    q = rs.randn(B, H, T, Dh).astype(np.float32)
    k = rs.randn(B, H, T, Dh).astype(np.float32)
    v = rs.randn(B, H, T, Dh).astype(np.float32)
    scale = Dh ** -0.5

    print("milestone 4a — two-pass attention boundary\n")
    for ar_len in (0, 1, 5, 15, 16):
        ref_fn = make_two_pass_attention(ar_len)
        with torch.no_grad():
            r = ref_fn(torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v), H)
        r = r.reshape(B, T, H, Dh).transpose(1, 2)      # back to (B,H,T,D) for comparison

        m = D.two_pass_mask(ar_len, T)
        got = mx.fast.scaled_dot_product_attention(
            mx.array(q), mx.array(k), mx.array(v), scale=scale, mask=m)
        mx.eval(got)
        err = rel_err(got, r)
        ok = err <= 1e-5
        print(f"  {'OK ' if ok else 'FAIL'} ar_len={ar_len:3d} / T={T}   max rel err {err:.3e}")
        if not ok:
            fails.append(f"ar_len={ar_len}")

    # A leak would show up as the prefix depending on tokens after it.
    ar_len = 5
    m = D.two_pass_mask(ar_len, T)
    v2 = v.copy()
    v2[:, :, ar_len:] += 100.0                      # perturb only the gen half
    a = mx.fast.scaled_dot_product_attention(mx.array(q), mx.array(k), mx.array(v), scale=scale, mask=m)
    b = mx.fast.scaled_dot_product_attention(mx.array(q), mx.array(k), mx.array(v2), scale=scale, mask=m)
    mx.eval(a, b)
    leak = float(np.abs(to_np(a)[:, :, :ar_len] - to_np(b)[:, :, :ar_len]).max())
    ok = leak == 0.0
    print(f"  {'OK ' if ok else 'FAIL'} prefix isolation            "
          f"max change in [0,{ar_len}) when the gen half moves: {leak:.3e}")
    if not ok:
        fails.append("prefix leak")

    print()
    if fails:
        print(f"  FAIL — {fails}")
        return 1
    print("  PASS — boundary matches, and the prefix cannot see the gen half")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
