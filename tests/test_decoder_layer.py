"""Milestone 3: one decoder layer matches ComfyUI's own TransformerBlock.

The reference is imported and run directly rather than reimplemented, so this
compares against the code that actually produced the images on this machine —
no chance of a reimplementation quietly agreeing with a wrong port.

Checked in order of how easy each is to get wrong:
  1. interleaved MRoPE frequencies
  2. rope application
  3. the whole block, real layer-0 weights
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path.home() / "ComfyUI"))

from mdream import decoder as D  # noqa: E402
import comfy.ops  # noqa: E402
from comfy.text_encoders.llama import (  # noqa: E402
    Llama2Config, TransformerBlock as RefBlock, precompute_freqs_cis as ref_freqs,
    apply_rope as ref_apply_rope,
)

CKPT = Path.home() / "models/HiDream-O1-Image/checkpoints/hidream_o1_image_dev_bf16.safetensors"
PREFIX = "model.language_model.layers.0."
BF16_FLOOR = 2.0e-3


def to_np(a):
    if isinstance(a, mx.array):
        return np.array(a.astype(mx.float32), copy=False)
    return a.detach().float().cpu().numpy()


def rel_err(a, b) -> float:
    a, b = to_np(a).astype(np.float64), to_np(b).astype(np.float64)
    assert a.shape == b.shape, f"shape {a.shape} != {b.shape}"
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def report(name, err, tol, fails):
    ok = err <= tol
    print(f"  {'OK ' if ok else 'FAIL'} {name:32s} max rel err {err:.3e}  "
          f"({err / BF16_FLOOR:6.2%} of the bf16 floor, tol {tol:.0e})")
    if not ok:
        fails.append(name)


def hidream_ref_config():
    """The text config HiDreamO1TextConfig sets, expressed as a Llama2Config."""
    c = Llama2Config()
    c.vocab_size, c.hidden_size, c.intermediate_size = 151936, 4096, 12288
    c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads = 36, 32, 8
    c.head_dim, c.rms_norm_eps, c.rope_theta = 128, 1e-6, 5000000.0
    c.qkv_bias, c.rms_norm_add, c.mlp_activation = False, False, "silu"
    c.q_norm = c.k_norm = "gemma3"
    return c


def main() -> int:
    fails: list = []
    cfg = D.TextConfig()
    rcfg = hidream_ref_config()
    T = 12
    print("milestone 3 — one decoder layer\n")

    # --- 1. interleaved MRoPE -------------------------------------------------
    pos_np = np.stack([np.arange(T), np.arange(T) + 3, np.arange(T) + 7]).astype(np.int64)
    mine = D.precompute_freqs_cis(cfg.head_dim, mx.array(pos_np), cfg.rope_theta,
                                  cfg.rope_dims, cfg.interleaved_mrope)
    ref = ref_freqs(rcfg.head_dim, torch.from_numpy(pos_np), rcfg.rope_theta,
                    rope_dims=[24, 20, 20], interleaved_mrope=True)
    for i, nm in enumerate(("cos", "sin", "nsin")):
        report(f"MRoPE {nm}", rel_err(mine[i], ref[i]), 1e-5, fails)

    # --- 2. rope application --------------------------------------------------
    rs = np.random.RandomState(7)
    q = rs.randn(1, 32, T, 128).astype(np.float32)
    k = rs.randn(1, 8, T, 128).astype(np.float32)
    gq, gk = D.apply_rope(mx.array(q), mx.array(k), mine)
    rq, rk = ref_apply_rope(torch.from_numpy(q.copy()), torch.from_numpy(k.copy()), ref)
    report("apply_rope q", rel_err(gq, rq), 1e-5, fails)
    report("apply_rope k", rel_err(gk, rk), 1e-5, fails)

    # --- 3. the whole block on real weights -----------------------------------
    W = mx.load(str(CKPT))
    need = {f"{PREFIX}{k_}": W[f"{PREFIX}{k_}"].astype(mx.float32) for k_ in D.LAYER_KEYS}
    mx.eval(list(need.values()))

    ref_blk = RefBlock(rcfg, 0, device="cpu", dtype=torch.float32, ops=comfy.ops.manual_cast)
    sd = {k_.replace(PREFIX, ""): torch.from_numpy(to_np(v)) for k_, v in need.items()}
    missing, unexpected = ref_blk.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  note: reference state_dict missing={list(missing)} unexpected={list(unexpected)}")

    x = rs.randn(1, T, 4096).astype(np.float32) * 0.02
    from comfy.ldm.modules.attention import optimized_attention
    with torch.no_grad():
        r, _ = ref_blk(torch.from_numpy(x), attention_mask=None, freqs_cis=ref,
                       optimized_attention=optimized_attention, past_key_value=None)

    # Two separate questions, so two separate checks.
    #
    # CPU answers "is the arithmetic right?" — MLX's CPU GEMM is accurate, so any
    # real difference in logic shows up here at fp32 scale and cannot hide behind
    # framework noise.
    #
    # GPU answers "does the path we actually ship stay inside the precision the
    # model runs at?" — and rather than hardcode that, the envelope is measured
    # here: the same reference block is run in bf16 and compared with itself in
    # fp32. That is exactly the error deployment already tolerates, so the bar
    # for the GPU path is to come in under it.
    ref_bf = RefBlock(rcfg, 0, device="cpu", dtype=torch.bfloat16, ops=comfy.ops.manual_cast)
    ref_bf.load_state_dict({k_: v.bfloat16() for k_, v in sd.items()}, strict=False)
    with torch.no_grad():
        r_bf, _ = ref_bf(torch.from_numpy(x).bfloat16(), attention_mask=None,
                         freqs_cis=tuple(t.bfloat16() for t in ref),
                         optimized_attention=optimized_attention, past_key_value=None)
    envelope = rel_err(r_bf, r)
    print(f"  --  reference bf16 vs its own fp32: {envelope:.3e}  "
          f"<- the precision one layer actually runs at")

    for dev, dname, tol in ((mx.cpu, "cpu  (logic)", 1e-5),
                            (mx.gpu, "gpu  (shipped path)", envelope)):
        mx.set_default_device(dev)
        f = D.precompute_freqs_cis(cfg.head_dim, mx.array(pos_np), cfg.rope_theta,
                                   cfg.rope_dims, cfg.interleaved_mrope)
        b = D.TransformerBlock(cfg)
        D.load_layer(b, need, PREFIX)
        got = b(mx.array(x), f, mask=None)
        mx.eval(got)
        report(f"TransformerBlock L0 {dname}", rel_err(got, r), tol, fails)
    mx.set_default_device(mx.gpu)

    print(f"\n  bf16 round-trip floor on this machine: {BF16_FLOOR:.1e}")
    if fails:
        print(f"  FAIL — {len(fails)} check(s): {fails}")
        return 1
    print("  PASS — decoder layer matches ComfyUI's own implementation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
