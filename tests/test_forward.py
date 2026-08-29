"""Milestone 6: the whole text-to-image forward, on a small synthetic model.

Running the real 8B twice would need ~60 GB and would only say pass or fail.
A small model with random weights exercises exactly the same wiring — embedding
lookup, the timestep substitution on the tms token, text-then-pixels
concatenation, MRoPE, the two-pass boundary, target slicing, unpatch and the
fp32 tail — and it does it in fp32 on CPU where MLX's GEMM is accurate, so any
wiring mistake shows up as a large error instead of hiding under bf16 noise.

The real weights are exercised separately by the milestones that loaded them.
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

from mdream import conditioning as C, decoder as D  # noqa: E402
from mdream.model import HiDreamO1  # noqa: E402
import comfy.ops  # noqa: E402
from comfy.ldm.hidream_o1.model import HiDreamO1Transformer  # noqa: E402

SMALL = dict(hidden_size=256, intermediate_size=512, num_hidden_layers=4,
             num_attention_heads=8, num_key_value_heads=2, head_dim=32,
             rope_dims=[6, 5, 5], vocab_size=152000)


def to_np(a):
    return np.array(a.astype(mx.float32), copy=False) if isinstance(a, mx.array) \
        else a.detach().float().cpu().numpy()


def rel_err(a, b) -> float:
    a, b = to_np(a).astype(np.float64), to_np(b).astype(np.float64)
    assert a.shape == b.shape, f"{a.shape} != {b.shape}"
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def main() -> int:
    mx.set_default_device(mx.cpu)
    torch.manual_seed(3)
    print("milestone 6 — full T2I forward (synthetic weights)\n")

    ref = HiDreamO1Transformer(
        dtype=torch.float32, device="cpu", operations=comfy.ops.manual_cast,
        text_config_overrides=SMALL,
        vision_config_overrides=dict(hidden_size=64, num_heads=2, intermediate_size=128,
                                     depth=1, out_hidden_size=SMALL["hidden_size"]),
    ).eval()
    for p in ref.parameters():
        with torch.no_grad():
            p.normal_(0, 0.02)

    cfg = D.TextConfig(**{k: v for k, v in SMALL.items()})
    mine = HiDreamO1(cfg)
    sd = {k: v for k, v in ref.state_dict().items()}
    w = {}
    for k, v in sd.items():
        if k.startswith("visual."):
            continue
        w["model." + k] = mx.array(v.detach().float().numpy())
    from mdream.model import load_model
    load_model(mine, w, dtype=mx.float32)

    H, W, txt_len = 64, 96, 11
    rs = np.random.RandomState(9)
    ids = rs.randint(1000, 150000, size=(1, txt_len)).astype(np.int64)
    ids[0, -1] = C.TMS_TOKEN_ID
    conds = C.build_t2i_conds(ids, H, W)
    x = (rs.randn(1, 3, H, W) * 0.5).astype(np.float32)
    t = np.array([700.0], dtype=np.float32)

    got = mine(mx.array(x), mx.array(t), mx.array(conds["input_ids"]),
               mx.array(conds["position_ids"]), mx.array(conds["vinput_mask"]),
               conds["ar_len"], compute_dtype=mx.float32)
    mx.eval(got)

    with torch.no_grad():
        r = ref._forward(
            torch.from_numpy(x), torch.from_numpy(t), context=None, transformer_options={},
            input_ids=torch.from_numpy(conds["input_ids"]),
            position_ids=torch.from_numpy(conds["position_ids"])[None],
            vinput_mask=torch.from_numpy(conds["vinput_mask"]),
            ar_len=conds["ar_len"],
        )

    err = rel_err(got, r)
    ok = err <= 1e-4
    print(f"  {'OK ' if ok else 'FAIL'} velocity prediction   shape {tuple(got.shape)}   "
          f"max rel err {err:.3e}  (tol 1e-04)")
    print(f"       output range mdream [{to_np(got).min():+.3f}, {to_np(got).max():+.3f}]  "
          f"reference [{to_np(r).min():+.3f}, {to_np(r).max():+.3f}]")
    print("\n  PASS — the full forward is wired correctly" if ok else "\n  FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
