"""Milestone 4b: all 36 layers, one at a time.

Holding both stacks in memory at once would cost ~30 GB and, worse, would only
tell you the final number. This streams layer by layer instead: each MLX layer
is compared against the same reference layer on the same input, so a divergence
is attributed to a specific layer rather than to "somewhere in the model".

Three streams over the stack:
  synced        — both get the reference's hidden state, isolating each layer
  free-run      — mdream carries its own state forward, showing accumulation
  reference bf16 — the same reference layers in bf16, also free-running

The third stream is the bar. Accumulated rounding over 36 layers has to be
judged against something, and the honest something is what the reference itself
loses when run at the precision it actually ships at — measured here rather than
guessed at.
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
    Llama2Config, TransformerBlock as RefBlock, RMSNorm as RefRMSNorm,
    precompute_freqs_cis as ref_freqs,
)
from comfy.ldm.modules.attention import optimized_attention  # noqa: E402

CKPT = Path.home() / "models/HiDream-O1-Image/checkpoints/hidream_o1_image_dev_bf16.safetensors"
PRE = "model.language_model."


def to_np(a):
    return np.array(a.astype(mx.float32), copy=False) if isinstance(a, mx.array) \
        else a.detach().float().cpu().numpy()


def rel_err(a, b) -> float:
    a, b = to_np(a).astype(np.float64), to_np(b).astype(np.float64)
    assert a.shape == b.shape, f"{a.shape} != {b.shape}"
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def ref_config():
    c = Llama2Config()
    c.vocab_size, c.hidden_size, c.intermediate_size = 151936, 4096, 12288
    c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads = 36, 32, 8
    c.head_dim, c.rms_norm_eps, c.rope_theta = 128, 1e-6, 5000000.0
    c.qkv_bias, c.rms_norm_add, c.mlp_activation = False, False, "silu"
    c.q_norm = c.k_norm = "gemma3"
    return c


def main() -> int:
    mx.set_default_device(mx.cpu)           # logic check
    cfg, rcfg = D.TextConfig(), ref_config()
    T, ar_len = 24, 9
    rs = np.random.RandomState(21)
    pos = np.stack([np.arange(T), np.arange(T) + 2, np.arange(T) + 5]).astype(np.int64)

    W = mx.load(str(CKPT))
    fm = D.precompute_freqs_cis(cfg.head_dim, mx.array(pos), cfg.rope_theta,
                                cfg.rope_dims, cfg.interleaved_mrope)
    fr = ref_freqs(rcfg.head_dim, torch.from_numpy(pos), rcfg.rope_theta,
                   rope_dims=[24, 20, 20], interleaved_mrope=True)
    mask_m = D.two_pass_mask(ar_len, T)
    ref_attn = __import__("comfy.ldm.hidream_o1.attention", fromlist=["x"]).make_two_pass_attention(ar_len)

    x = (rs.randn(1, T, 4096) * 0.02).astype(np.float32)
    h_ref = torch.from_numpy(x)
    h_free = mx.array(x)
    h_bf = torch.from_numpy(x).bfloat16()
    fr_bf = tuple(t.bfloat16() for t in fr)
    worst_sync, worst_at = 0.0, -1

    print(f"milestone 4b — 36 layers, T={T}, ar_len={ar_len}\n")
    for i in range(cfg.num_hidden_layers):
        prefix = f"{PRE}layers.{i}."
        need = {f"{prefix}{k}": W[f"{prefix}{k}"].astype(mx.float32) for k in D.LAYER_KEYS}
        mx.eval(list(need.values()))

        blk = D.TransformerBlock(cfg)
        D.load_layer(blk, need, prefix)
        rblk = RefBlock(rcfg, i, device="cpu", dtype=torch.float32, ops=comfy.ops.manual_cast)
        rblk.load_state_dict({k.replace(prefix, ""): torch.from_numpy(to_np(v))
                              for k, v in need.items()}, strict=False)

        synced = blk(mx.array(to_np(h_ref)), fm, mask_m)
        h_free = blk(h_free, fm, mask_m)
        mx.eval(synced, h_free)
        rbf = RefBlock(rcfg, i, device="cpu", dtype=torch.bfloat16, ops=comfy.ops.manual_cast)
        rbf.load_state_dict({k.replace(prefix, ""): torch.from_numpy(to_np(v)).bfloat16()
                             for k, v in need.items()}, strict=False)
        with torch.no_grad():
            h_ref, _ = rblk(h_ref, attention_mask=None, freqs_cis=fr,
                            optimized_attention=ref_attn, past_key_value=None)
            h_bf, _ = rbf(h_bf, attention_mask=None, freqs_cis=fr_bf,
                          optimized_attention=ref_attn, past_key_value=None)

        e = rel_err(synced, h_ref)
        if e > worst_sync:
            worst_sync, worst_at = e, i
        if i % 6 == 0 or i == cfg.num_hidden_layers - 1:
            print(f"  layer {i:2d}  synced {e:.3e}   free-run {rel_err(h_free, h_ref):.3e}"
                  f"   reference bf16 {rel_err(h_bf.float(), h_ref):.3e}")
        del blk, rblk, rbf, need
        mx.clear_cache()

    nm = D.RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    nm.weight = W[f"{PRE}norm.weight"].astype(mx.float32)
    out = nm(h_free)
    rn = RefRMSNorm(4096, eps=1e-6, device="cpu", dtype=torch.float32)
    rn.load_state_dict({"weight": torch.from_numpy(to_np(W[f"{PRE}norm.weight"]))})
    with torch.no_grad():
        rout = rn(h_ref)
    mx.eval(out)

    with torch.no_grad():
        rout_bf = rn(h_bf.float())
    final = rel_err(out, rout)
    envelope = rel_err(rout_bf, rout)
    print(f"\n  worst single layer (synced): {worst_sync:.3e} at layer {worst_at}")
    print(f"  mdream, full stack:          {final:.3e}")
    print(f"  reference bf16, full stack:  {envelope:.3e}  <- the precision the stack runs at")
    ok = worst_sync <= 1e-5 and final <= envelope
    if ok:
        print(f"\n  PASS — every layer matches, and 36-layer accumulation is "
              f"{envelope / max(final, 1e-12):.0f}x inside the bf16 envelope")
    else:
        print("\n  FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
