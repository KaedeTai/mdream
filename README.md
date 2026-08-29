# mdream

MLX implementation of **HiDream-O1-Image** for Apple Silicon.

Status: **day 1 — architecture mapped, weight audit passing, nothing generates yet.**

```
$ python -m mdream.weights
758 tensors, 15.24 GiB
  decoder.layer               396   36 instances
  vision.block                324   27 instances
  vision.deepstack(DROPPED)    18    3 instances   230 MiB, unused
  vision.merger                 6
  vision.other / decoder.*      6
  patch_embed / final_layer / timestep_embed   9
PASS - every tensor is accounted for
```

## Why this exists

HiDream-O1 runs on this machine today through ComfyUI on torch/MPS. That path
works but leaves two things on the table:

1. **No quantisation.** `hidream_o1_image_dev_mxfp8.safetensors` fails on MPS with
   `Undefined type Float8_e4m3fn` — PyTorch's Metal backend has no fp8 kernels.
   On a unified-memory machine the binding constraint is usually memory
   bandwidth, so being unable to shrink 16 GB of bf16 weights is the single
   biggest lost opportunity. MLX quantises to 4/8-bit natively.
2. **MPS overhead.** Measured baseline to beat, warm, 768x1024, one edit:
   **40.1 s** (dev bf16, 28 steps, cfg 1.0) and **75.1 s** (dev bf16, 28 steps,
   cfg 5.0). Anything slower than that is not worth shipping.

mflux was the obvious host and it is the wrong one: mflux is built around FLUX
(DiT + T5/CLIP text encoders + VAE) and HiDream-O1 has none of that shape.

## What HiDream-O1 actually is

Not a latent DiT. It is **Qwen3-VL-8B with pixel patch shims bolted on**, run as
a flow model in pixel space. There is no VAE and no frozen text encoder; pixels,
text and task conditions share one token sequence.

```
raw RGB  --32x32 patch-->  BottleneckPatchEmbed  3072 -> 1024 -> 4096
ref imgs --------------->  Qwen3-VL vision tower (27 blocks, 1152 -> 4096)
text     --------------->  Qwen3-VL embeddings
                               |
                    concat into one sequence
                               |
                   Qwen3-VL-8B decoder, 36 layers
                   (interleaved MRoPE, gemma3-style q/k norm)
                   two-pass attention: [0, ar_len) causal, [ar_len, T) full
                               |
                        FinalLayer 4096 -> 3072
                               |
                    unpatch -> (x - x_pred) / sigma
```

Config recovered from the ComfyUI implementation:

| | |
|---|---|
| text decoder | Qwen3-VL-8B: 36 layers, hidden 4096, ffn 12288, 32 heads / 8 KV, head_dim 128 |
| rope | theta 5e6, dims [24, 20, 20], **interleaved MRoPE** |
| norms | RMSNorm eps 1e-6, q_norm/k_norm gemma3-style, no qkv bias |
| vision | 27 blocks, hidden 1152, 16 heads, ffn 4304, patch 16, spatial_merge 2, out 4096 |
| patch | 32x32 RGB, PCA bottleneck 1024 |
| tokens | image_pad 151655, tms_token 151673 |
| deepstack | mergers present in the checkpoint, **unused** — dropped at load |

The whole thing is ~870 lines of PyTorch in ComfyUI. That is the entire porting
surface, and a numerically-correct reference sits on the same disk.

## What MLX gives us for free, and what it does not

`mlx_lm.models.qwen3_vl` is 57 lines — a text-only wrapper over qwen3 with **no
vision tower and no MRoPE**. So the reusable part is smaller than it looks:

| piece | status |
|---|---|
| Qwen3 decoder block, RMSNorm, SwiGLU, GQA | reusable from mlx_lm |
| interleaved MRoPE, rope_dims [24,20,20] | **write** |
| gemma3-style q/k norm | **write** |
| Qwen3-VL vision tower (27 blocks) | **write** |
| two-pass attention (causal prefix + full gen) | **write** |
| patch embed / final layer / timestep embed | **write** (trivial) |
| flow sampler, sigma schedule, conditioning | **write** |
| tokenizer / processor | reuse HF |

Parameter inventory recovered by the audit, which is what the port implements against:

**decoder layer** (11 tensors, no biases anywhere)

```
input_layernorm.weight            (4096,)
self_attn.q_proj.weight           (4096, 4096)      32 heads
self_attn.k_proj.weight           (1024, 4096)       8 KV heads -> GQA confirmed
self_attn.v_proj.weight           (1024, 4096)
self_attn.q_norm.weight            (128,)           per-head-dim, gemma3 style
self_attn.k_norm.weight            (128,)
self_attn.o_proj.weight           (4096, 4096)
post_attention_layernorm.weight   (4096,)
mlp.gate_proj.weight             (12288, 4096)      SwiGLU
mlp.up_proj.weight               (12288, 4096)
mlp.down_proj.weight              (4096, 12288)
```

**vision block** (12 tensors — note it is unlike the decoder: fused QKV, biases
everywhere, and `norm1`/`norm2` carry both weight and bias, so **LayerNorm, not
RMSNorm**)

```
norm1.weight / norm1.bias          (1152,)
attn.qkv.weight / bias        (3456, 1152)   fused 3x1152
attn.proj.weight / bias       (1152, 1152)
norm2.weight / norm2.bias          (1152,)
mlp.linear_fc1.weight / bias  (4304, 1152)
mlp.linear_fc2.weight / bias  (1152, 4304)
```

## Verification strategy

The rule that made the MTP work in this repo's sibling project succeed, applied
again: **never build the whole thing and then debug it.** Every stage is checked
numerically against ComfyUI on the same machine, same weights, same inputs,
before the next stage is written.

1. weight audit — every one of the 758 tensors assigned to a module (**done**)
2. patch embed + final layer — exact match on random input
3. one decoder layer — match hidden states
4. full decoder, no vision — match hidden states at every layer
5. vision tower — match image embeds
6. full forward at one timestep — match the velocity prediction
7. sampler — match the image bit-for-bit at cfg 1.0, fixed seed
8. only then: quantise, and re-measure against the 40.1 s baseline

## Layout

```
mdream/weights.py    checkpoint inspection and key -> module mapping
notes/reference.md   where the PyTorch reference lives, and what to read
tests/              numeric parity checks against ComfyUI
```

## Reference

The authoritative reference for this checkpoint format is ComfyUI's own
implementation, not the upstream repo (upstream wants the HF multi-folder
layout and a CUDA GPU):

```
~/ComfyUI/comfy/ldm/hidream_o1/model.py          306 lines  transformer + forward
~/ComfyUI/comfy/ldm/hidream_o1/conditioning.py   230 lines  prompt/ref/latent assembly
~/ComfyUI/comfy/ldm/hidream_o1/utils.py          173 lines
~/ComfyUI/comfy/ldm/hidream_o1/attention.py       41 lines  two-pass attention
~/ComfyUI/comfy/text_encoders/hidream_o1.py      119 lines  tokenizer shim
~/ComfyUI/comfy/text_encoders/qwen35.py                     vision tower
~/ComfyUI/comfy/text_encoders/llama.py                      Llama2_ decoder
```

Weights: `~/models/HiDream-O1-Image/checkpoints/hidream_o1_image_dev_bf16.safetensors`
(dev/distilled, 16.4 GB — the base variant is beside it and needs cfg 5.0).
