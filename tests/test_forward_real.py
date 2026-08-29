"""Milestone 6b: the same forward, with the real 8B weights.

Milestone 6 proved the wiring on a small synthetic model in fp32. This proves
`load_model` puts the real checkpoint's 758 tensors in the right places, which
the synthetic test cannot: it runs both stacks in bf16 (16 GB each) on a small
canvas and compares against the bf16 envelope, since fp32 would need ~60 GB.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path.home() / "ComfyUI"))

from mdream import conditioning as C, decoder as D  # noqa: E402
from mdream.model import HiDreamO1, load_model  # noqa: E402
import comfy.ops  # noqa: E402
from comfy.ldm.hidream_o1.model import HiDreamO1Transformer  # noqa: E402

CKPT = Path.home() / "models/HiDream-O1-Image/checkpoints/hidream_o1_image_dev_bf16.safetensors"


def to_np(a):
    return np.array(a.astype(mx.float32), copy=False) if isinstance(a, mx.array) \
        else a.detach().float().cpu().numpy()


def rel_err(a, b) -> float:
    a, b = to_np(a).astype(np.float64), to_np(b).astype(np.float64)
    assert a.shape == b.shape, f"{a.shape} != {b.shape}"
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def main() -> int:
    H, W, txt_len = 64, 96, 11
    rs = np.random.RandomState(9)
    ids = rs.randint(1000, 150000, size=(1, txt_len)).astype(np.int64)
    ids[0, -1] = C.TMS_TOKEN_ID
    conds = C.build_t2i_conds(ids, H, W)
    x = (rs.randn(1, 3, H, W) * 0.5).astype(np.float32)
    t = np.array([700.0], dtype=np.float32)

    print("milestone 6b — full forward, real 8B weights, bf16\n")
    W_ = mx.load(str(CKPT))
    mine = HiDreamO1(D.TextConfig())
    t0 = time.time()
    load_model(mine, W_, dtype=mx.bfloat16)
    mx.eval(mine.parameters())
    print(f"  mdream loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    got = mine(mx.array(x), mx.array(t), mx.array(conds["input_ids"]),
               mx.array(conds["position_ids"]), mx.array(conds["vinput_mask"]),
               conds["ar_len"], compute_dtype=mx.bfloat16)
    mx.eval(got)
    print(f"  mdream forward   {time.time() - t0:.2f}s   "
          f"range [{to_np(got).min():+.3f}, {to_np(got).max():+.3f}]   "
          f"finite: {bool(np.isfinite(to_np(got)).all())}")

    ref = HiDreamO1Transformer(dtype=torch.bfloat16, device="cpu",
                               operations=comfy.ops.manual_cast).eval()
    sd = {k[len("model."):]: torch.from_numpy(
              np.array(v.astype(mx.float32), copy=False)).bfloat16()
          for k, v in W_.items() if k.startswith("model.")}
    missing, unexpected = ref.load_state_dict(sd, strict=False)
    print(f"  reference loaded, missing={len(missing)} unexpected={len(unexpected)}")

    t0 = time.time()
    with torch.no_grad():
        r = ref._forward(torch.from_numpy(x).bfloat16(), torch.from_numpy(t),
                         context=None, transformer_options={},
                         input_ids=torch.from_numpy(conds["input_ids"]),
                         position_ids=torch.from_numpy(conds["position_ids"])[None],
                         vinput_mask=torch.from_numpy(conds["vinput_mask"]),
                         ar_len=conds["ar_len"])
    print(f"  reference forward {time.time() - t0:.2f}s   "
          f"range [{to_np(r).min():+.3f}, {to_np(r).max():+.3f}]")

    err = rel_err(got, r)
    ok = err <= 0.15
    print(f"\n  {'OK ' if ok else 'FAIL'} velocity, mdream bf16 vs reference bf16: {err:.3e}")
    print("       (both in bf16 through 36 layers, where the reference's own bf16")
    print("        drifts >100% from its fp32 self — so this is a loose check by")
    print("        nature; milestone 6 is what proves the wiring.)")
    print("\n  PASS — real weights load into the right places" if ok else "\n  FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
