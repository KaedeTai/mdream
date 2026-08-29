# mdream

MLX implementation of **HiDream-O1-Image** for Apple Silicon.

Status: **text-to-image works and matches ComfyUI.** 768x1024, 28 steps, euler,
cfg 1.0 — `mdream fp32` vs `ComfyUI fp32` is 46.3 dB PSNR on the final latent,
16.2 dB tighter than the reference's own bf16-vs-fp32 envelope. The vision
tower (reference-image editing) and quantisation are not done yet.

```
$ python3 scripts/generate.py "a red fox in fresh snow, golden hour" \
      -o fox.png --width 768 --height 1024 --steps 28 --seed 42
  loaded hidream_o1_image_dev_bf16.safetensors in 1.7s
  768x1024  768 image tokens + 24 text = 792 sequence   28 steps  seed 42
  8.0s total, 0.29s/step
```

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
2. ~~**MPS overhead.**~~ This was the second reason and it turned out to be
   mostly wrong, so it is left here rather than quietly deleted. The number it
   was based on — 40.1 s, warm, 768x1024, dev bf16, 28 steps, cfg 1.0 — is the
   **image-edit** path, which carries a reference image through the ViT and
   roughly doubles the sequence. The comparable text-to-image number is 8.4 s,
   and mdream does the identical job in 8.0 s. **In bf16 MLX buys about 5%.**

   The gap only opens in fp32 — ComfyUI 39.9 s, mdream 14.2 s, **2.8x** — which
   is real but not what anyone ships. So reason 1 is the whole case: the reason
   to have an MLX implementation is that it can be quantised on this machine
   and the torch/MPS one cannot.

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
| Qwen3 decoder block, RMSNorm, SwiGLU, GQA | **done** — 1.09e-6 on CPU |
| interleaved MRoPE, rope_dims [24,20,20] | **done** — 9.4e-7 |
| gemma3-style q/k norm | **done** |
| Qwen3-VL vision tower (27 blocks) | **write** |
| two-pass attention (causal prefix + full gen) | **done** — 1.2e-7, prefix isolation exact |
| patch embed / final layer / timestep embed | **write** (trivial) |
| conditioning: input_ids, MRoPE position ids, ar_len | **done** — exact |
| flow sampler, sigma schedule | **write** |
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
2. patch embed + final layer — match on random input (**done**)
3. one decoder layer — match hidden states (**done**, 1.09e-6 on CPU)
4. full decoder, no vision — match hidden states at every layer (**done**, worst layer 2.8e-6)
5. T2I conditioning — position ids, masks, ar_len (**done**, exact)
6. full forward at one timestep, text only — match the velocity prediction (**done**, 8.9e-8)
7. sampler — match the image at cfg 1.0, fixed seed (**done**, 46.3 dB fp32)
8. vision tower — match image embeds (needed for the edit path, not for T2I)
9. only then: quantise, and re-measure against the 8.4 s T2I baseline

Order changed at milestone 5: the vision tower is only needed for reference-image
editing, and everything else — conditioning, forward, sampler — can be finished
and verified on the text-only path first. That gets a running generator sooner
and leaves the tower as one self-contained piece rather than a blocker.

## Precision: what "matches" means

Measured before any tolerance was written (`notes/precision.md`): the bf16
round-trip floor on this machine is **2.0e-3**, and in bf16 MLX and torch matmul
are *identical* (both 3.968e-3 vs float64). MLX's fp32 matmul is 8.0e-4 — 300x
looser than torch's 2.4e-6, but still 4x tighter than the bf16 floor the real
weights carry.

So the bar is: **the port must differ from the reference by less than the
precision the model actually runs at.** Milestone 2 lands at 3–46% of that floor:

```
BottleneckPatchEmbed   9.162e-04   45.81% of the bf16 floor
FinalLayer             3.670e-04   18.35%
TimestepEmbedder       7.768e-04   38.84%
timestep_embedding     6.056e-05    3.03%
patchify / unpatchify  0.000e+00    exact, checked against einops
```

The patch ordering is the one that had to be exact rather than close, since a
wrong ordering corrupts everything downstream while looking like a model bug.

Milestone 3 forced a better testing rule. The block first came in at 2.3e-3 —
above the single-tensor floor — and the useful question was whether that was a
bug or the framework. Running the identical MLX code on **CPU** answered it:
**1.09e-6**. MLX's CPU GEMM is accurate, so logic errors cannot hide there; the
GPU gap is purely Metal's fp32 matmul. And the bar for the GPU path is not the
single-tensor floor either — it is what the reference itself loses in bf16 on
the same block, measured inside the test at **7.8e-3**. mdream's GPU path sits
3.4x inside that.

So every milestone from here runs twice: **CPU at fp32 tolerance to prove the
arithmetic, GPU against a measured bf16 envelope to prove the shipped path.**

Milestone 4 showed how wide that envelope really is. Streaming all 36 layers and
carrying three hidden states — mdream fp32, reference fp32, reference bf16 —
gives:

```
worst single layer, synced      2.756e-06
mdream free-running, 36 layers  5.567e-04
reference bf16, 36 layers       1.173e+00   <- what the stack actually runs at
```

The reference's own bf16 hidden states diverge from its fp32 self by more than
100% relative by layer 36. The model is evidently robust to that — it makes good
images — but it means **hidden-state parity is only meaningful against fp32**.
Later milestones, and especially quantisation, have to be judged on the output
image, not on matching activations.

## Milestone 7: the sampler, and what "matches" means for an image

The sampler itself is bookkeeping, so it is checked exactly
(`tests/test_sampling.py`): tokenisation is byte-identical to ComfyUI's
`HiDreamO1Tokenizer` on four prompts including Chinese, the 8x initial noise
scaling is identical, and a 12-step Euler trajectory driven by a fake network
is bit-identical to `comfy.k_diffusion.sampling.sample_euler`. Only the sigma
schedule is not exact, by 2 float32 ULPs, because torch's `linspace` uses a
vectorised path no scalar numpy formula reproduces. The end-to-end test removes
even that by importing ComfyUI's sigmas.

The image is the hard part. A 28-step trajectory in bf16 is chaotic enough that
the reference does not agree with *itself* across precisions, so a single
mdream-vs-ComfyUI number would be unreadable. Measuring the envelope first
makes it readable:

```
768x1024, 28 steps euler, cfg 1.0, seed 42, ComfyUI's sigmas, same noise

  the envelope (how far the reference moves from itself)
    comfy bf16  vs comfy fp32     mean|d| 2.361e-02   PSNR 30.10 dB
    mdream bf16 vs mdream fp32    mean|d| 2.351e-02   PSNR 29.13 dB

  mdream against the reference
    mdream fp32 vs comfy fp32     mean|d| 2.551e-03   PSNR 46.34 dB
    mdream bf16 vs comfy bf16     mean|d| 3.169e-02   PSNR 26.50 dB
```

Three things fall out of that table:

- **The fp32 row is the proof.** 2.55e-3 is 9.3x tighter than the envelope the
  model actually runs in. The arithmetic is right.
- **mdream's bf16 sensitivity equals the reference's** (2.351e-2 vs 2.361e-2).
  A port that had, say, an fp32 accumulation the reference does not have would
  show up here as a *smaller* number, which would look like a win and be a
  divergence.
- **The bf16-vs-bf16 row is not a failure.** Two independent bf16 trajectories
  should diverge by roughly the quadrature sum of their own bf16 errors:
  sqrt(2.361^2 + 2.351^2) = 3.33e-2 predicted, 3.17e-2 measured. It lands where
  the other two rows say it must.

The images are visually indistinguishable — same composition, same pose, same
light; the difference is in fur and twig texture, which is where a chaotic
trajectory puts it.

## Layout

```
mdream/weights.py     checkpoint inspection and key -> module mapping
mdream/layers.py      pixel shims: patch embed, final layer, timestep embed
mdream/decoder.py     Qwen3-VL decoder: MRoPE, GQA attention, SwiGLU, two-pass mask
mdream/conditioning.py  T2I sequence assembly and MRoPE position ids
mdream/model.py       the assembled forward pass
mdream/tokenizer.py   prompt -> input_ids, HiDream-O1's chat template
mdream/sampling.py    flow schedule, 8x noise scaling, Euler
mdream/generate.py    the three tied together
scripts/generate.py         CLI
scripts/compare_vs_comfy.py milestone 7 harness (drives a running ComfyUI)
notes/reference.md    where the PyTorch reference lives, and what to read
notes/precision.md    measured precision floors; where tolerances come from
tests/test_shims.py          milestone 2 parity check
tests/test_decoder_layer.py  milestone 3, against ComfyUI's own TransformerBlock
tests/test_two_pass.py       milestone 4a, attention boundary and prefix isolation
tests/test_decoder_full.py   milestone 4b, all 36 layers streamed one at a time
tests/test_conditioning.py   milestone 5, exact match on sequence assembly
tests/test_forward.py        milestone 6, whole forward on a small synthetic model
tests/test_forward_real.py   milestone 6b, real 8B weights load where they belong
tests/test_sampling.py       milestone 7a, schedule/tokenizer/Euler vs ComfyUI
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
