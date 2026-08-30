# mdream

MLX implementation of **HiDream-O1-Image** for Apple Silicon.

Status: **complete for text-to-image and reference-image editing, quantised,
and matched against ComfyUI at every stage.**

Text-to-image: `mdream fp32` vs `ComfyUI fp32` is 46.3 dB PSNR on the final
latent, 16.2 dB tighter than the reference's own bf16-vs-fp32 envelope.
Editing: matches ComfyUI at 1152x1536, and reproduces its failure at 768x1024,
which turns out to be the model rather than either implementation.
Quantisation: works (6-bit is visually indistinguishable at 6.49 GiB against
14.17) but is **not recommended** — it is slower than bf16 at every resolution
and the memory it saves is only worth having on a machine that cannot hold
bf16. See milestone 9.

Multi-reference works: two images (a person and a garment) fuse correctly, and
the K=2 conditioning is bit-identical to ComfyUI's.

Deliberately not done: the SDE samplers and the prefix KV cache. Both are
argued below rather than left as a TODO.

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

## Getting it running

Apple Silicon only — this is an MLX implementation.

```bash
pip install mlx numpy pillow
# torch is optional: needed to run the tests (they compare against ComfyUI's
# own code) and for --match-comfy, which reproduces ComfyUI's preprocessing
# bit-for-bit. Generating images does not need it.
```

Weights are the ComfyUI single-file repack, not the HF multi-folder layout:
[Comfy-Org/HiDream-O1-Image](https://huggingface.co/Comfy-Org/HiDream-O1-Image/tree/main/checkpoints)
— `hidream_o1_image_dev_bf16.safetensors` (distilled, 16.4 GB) is the one used
throughout; the `base` variant beside it needs cfg 5.0 even for text-to-image.

Paths come from the environment, with macOS-ish defaults:

```
MDREAM_CKPT        the checkpoint            default ~/models/HiDream-O1-Image/checkpoints/hidream_o1_image_dev_bf16.safetensors
MDREAM_COMFYUI     a ComfyUI checkout        default ~/ComfyUI
MDREAM_TOKENIZER   vocab.json + merges.txt   default <ComfyUI>/comfy/text_encoders/qwen25_tokenizer
```

ComfyUI is needed for two things and neither is generation: the Qwen2 tokenizer
files it ships, and running the tests, all of which import ComfyUI's modules
and compare against them directly rather than reimplementing the reference.

```bash
python3 scripts/generate.py "a red fox in fresh snow, golden hour" -o fox.png \
    --width 768 --height 1024 --steps 28 --seed 42

python3 scripts/edit.py "change the sweater to dark green" -i photo.png \
    -o out.png --width 1152 --height 1536 --cfg 5.0
```

Quantisation exists (`--bits 6 --save-quantized q6.safetensors`) but bf16 is
the better default on any machine that fits it, and there is no published
quantised checkpoint on purpose — see milestone 9.

### The five rules, all of them the model's and none of them this port's

Every one of these was found by running ComfyUI as a control and watching it
fail the same way. They are the difference between this model being useless and
being good.

| | |
|---|---|
| **`base`, not `dev`, for anything with skin** | `dev` returns speckled, crazed faces. Nothing rescues it — not 4 MP, not lower cfg, not seam smoothing. `dev` is fine for text-to-image below ~2 MP. |
| **cfg > 1 when editing** | At cfg 1.0 the edit path returns pure noise. `base` wants cfg 5 even for text-to-image. |
| **≥ ~1.7 MP when editing** | Below that the edit path does not degrade, it collapses. 768x1024 is noise; 1152x1536 is clean. |
| **source ≥ canvas when editing** | The model reproduces the source's detail at the new scale rather than inventing more, so a small source stretched to a big canvas comes out flat. |
| **seam smoothing on for skin** | The 32px patch grid is visible on faces and flat sky. On by default for editing. |

`scripts/edit.py` defaults to base with seam smoothing, and warns below 1.7 MP.

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
   is real but not what anyone ships.

   And reason 1 turned out weaker than it looked too, once quantisation was
   measured rather than assumed: 6-bit is *slower* than bf16 here, and the
   memory it saves only matters on a machine that cannot hold bf16 at all.

   So the honest case for this implementation is narrower than the one it was
   started on: it is a dependency-free MLX path that matches ComfyUI, it wins
   in fp32, and it makes quantisation *possible* for the machines that need it
   — not a speed-up for machines that do not.

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
8. vision tower and the edit path — image embeds, preprocessing, CFG, and a
   working reference-image edit (**done**)

Steps 8 and 9 swapped order: quantisation is the whole reason this port exists
and it only needs the text-to-image path, so it went first.
9. quantise (**done** — 6-bit is visually identical at 2.18x smaller)

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

## Milestone 9: quantisation, and the one layer that breaks it

`--bits 6` produces images indistinguishable from bf16 at 6.49 GiB instead of
14.17. That is the headline; the interesting part is 4-bit.

Flat 4-bit affine does not produce noise or artefacts — it produces a
*systematically darker* image. Three unrelated prompts, same seed, mean pixel
value of the output:

```
                    portrait   market   still life      weights
  bf16                0.3103   0.2587       0.3028     14.17 GiB
  8-bit g64           0.3105   0.2582       0.3023      8.10
  6-bit g64           0.3155   0.2576       0.3101      6.49
  4-bit g64                -        -       0.1311      4.87   <- half as bright
  4-bit g32                -        -       0.1973      5.27
  4-bit g64 + down_proj@8   0.3030   0.2229  0.2993     5.71   <- fixed
```

The weight-error diagnostic says nothing is special: quantise/dequantise every
projection in all 36 layers and the relative error is 0.099-0.103 across the
board, `down_proj` only marginally worst. But keeping *only* `down_proj` at 8
bits restores the brightness completely, at a cost of 0.84 GiB.

That matches something ComfyUI's config already says out loud:

    # fp16 not supported: LM MLP down_proj activations fp16 overflow, causing NaNs

`down_proj` is where the large-magnitude activations live. Weight error is the
wrong thing to look at for it; what matters is that its *inputs* are large, so
the same relative weight error produces a much larger absolute output error,
and a systematic one. In a language model the next softmax hides that. Here the
residual stream is projected straight to pixels, so it shows up as gain.

So the honest recommendation is not 4-bit:

- **6-bit g64, 6.49 GiB** — visually identical to bf16 across all three test
  prompts, and the best of the quantised options. Use it only if bf16 does not
  fit; it is slower.
- **8-bit g64, 8.10 GiB** — identical, if you want no argument at all.
- **4-bit + `down_proj` at 8-bit, 5.71 GiB** — correct exposure, but a
  different and consistently *simpler* sample: fewer objects, less background
  detail. Usable, not equivalent.
- **flat 4-bit, 4.87 GiB** — do not.

```
python3 scripts/generate.py x --bits 6 --save-quantized q6.safetensors
python3 scripts/generate.py "..." --ckpt q6.safetensors -o out.png
```

The quantisation config travels in the safetensors metadata, so a quantised
checkpoint reloads without being told what it is, and the round-trip is
bit-identical to quantising in process.

### What quantisation does *not* buy: speed

Measured properly — warm, best of two, both precisions in the same session, and
for the edit path interleaved round by round so drift cannot favour either:

```
text-to-image, 28 steps, cfg 1.0            bf16 s/step   6-bit s/step
   768x1024   0.79 MP    768 tokens             0.266        0.438
  1024x1024   1.05 MP   1024 tokens             0.365        0.578
  1152x1536   1.77 MP   1728 tokens             0.627        0.963
  1536x2048   3.15 MP   3072 tokens             1.301        1.972
  2048x2048   4.19 MP   4096 tokens             2.011        2.928
  peak memory                                  15.6 GiB     8.8 GiB

edit 1152x1536, cfg 5.0, 4158 tokens, two forwards per step
                                               6.21         6.71
```

**6-bit is 1.5x slower on text-to-image and 1.08x slower on editing. It never
wins on speed.** What it wins is peak memory, 15.6 GiB against 8.8.

The reason is the shape of the work, and it is worth stating because the LLM
intuition points the wrong way. Decoding one token from an LLM is
bandwidth-bound: every weight is read to produce one column, so halving the
weights nearly halves the time. This model pushes 768–4096 tokens through 8B
parameters on every step, which is a compute-bound GEMM — the weights are read
once and reused across the whole batch, so shrinking them saves nothing and
dequantising them costs.

An earlier version of this file said quantisation "starts paying at the edit
path's 4158 tokens", from a 113.6 s 6-bit run against a 160.3 s bf16 run. Those
two were forty minutes apart with different things on the machine. Interleaved,
the ordering reverses. The lesson is the one this repo keeps relearning: two
numbers measured at different times are not a comparison.

### `dev` cannot edit skin

The distilled `dev` checkpoint returns portraits covered in dark speckles with
a crazed, reptilian skin texture, and frequently paints a before/after diptych
instead of one image. One variable changed at a time, everything else identical
(1152×1536, dpmpp_2m, 28 steps, cfg 5.0, seed 7):

```
  dev                      speckled, crazed skin, diptych
  dev + seam smoothing     identical failure
  dev at 1728x2304 (4 MP)  identical failure
  dev at cfg 3.0           identical failure
  dev in bf16 and in 6-bit identical failure
  base                     clean, natural, single portrait
```

ComfyUI reproduces the `dev` failure exactly, so it is upstream, not this port
— the third time in this project that running the control first was worth more
than reading the code. Nothing rescues `dev`: not resolution, not cfg, not the
seam-smoothing node written for patch artefacts.

Skin is the only high-frequency region in a portrait — hair, clothing and
background come out fine either way — which is why the damage lands precisely
on the face. Distillation evidently costs the top of the frequency range, and
a pixel-space model has nowhere to hide it.

`dev` remains the default for text-to-image, where it is fine and cheaper.

### The patch grid, and removing it

HiDream-O1 writes its output one 32x32 patch at a time and the patches do not
quite agree at their borders, so photographic skin comes out with a visible
grid. Measured as gradient energy on the patch boundaries over gradient energy
everywhere else — 1.0 would be no grid at all:

```
  a real photograph                        1.019 / 0.993
  mdream bf16, no smoothing                1.232 / 1.169
  ComfyUI bf16, no smoothing               1.254 / 1.185   <- the model's own seam
  mdream bf16 + seam smoothing ramp_2_4    1.087 / 1.024
  mdream 6-bit, no smoothing               1.634 / 1.384   <- quantisation triples it
```

Two things fall out. mdream's grid is the model's, not the port's — it sits
marginally *below* ComfyUI's. And **6-bit roughly triples it**: the excess over
1.0 goes from +0.23 to +0.63. That is mechanical rather than mysterious. The
output is produced per token, one patch each, so a per-token error lands
exactly on patch boundaries. `final_layer` is never quantised, but the decoder
hidden states feeding it are.

`mdream/seam.py` ports ComfyUI's `HiDreamO1PatchSeamSmoothing`: over the last
fifth of sampling, run the model again on an `x` rolled by half a patch, roll
the prediction back, and average, so the two runs put their seams in different
places and cancel. `ramp_2_4` uses two offsets early in that window and four
near the end. Rolling wraps, so one patch-width at each border keeps the
unshifted prediction behind a 4px feather.

Averaging is not free: it removes the grid by smoothing, and smoothing also
costs the micro-texture that makes an image read as a photograph. The whole
trade-off, same seed, measured as grid ratio / patch-interior detail /
high-frequency energy:

```
  no seam                        1.200   2.884   3.82    +0 forwards
  --seam 2                       1.076   2.633   3.36    +6
  --seam ramp_2_4 --start 0.8    1.056   2.531   3.14    +12
  --seam ramp_2_4 --start 0.9    1.061   2.611   3.22    +6    <- default
  a real photograph              1.006   3.162   8.10
```

Starting the window at 0.9 rather than ComfyUI's 0.8 removes essentially the
same grid, keeps more texture, and costs half the passes. **On by default for
editing** (`--seam off` to disable); off by default for text-to-image, whose
grid is milder, where it is available as `--seam ramp_2_4`.

Note the last row with care — it is not what it looks like, and an earlier
version of this file drew the wrong conclusion from it. Comparing per-pixel
high-frequency energy across *different resolutions* is meaningless: the
"photograph" is 768x1024 and the output is 1152x1536, and merely upscaling the
photograph to the output's size drops it from 8.10 to 4.25. Against that
baseline the model's 3.82 is ordinary, not a deficit.

Measured properly, the model has no trouble with fine texture at all:

```
  text-to-image, 768x1024 native      21.25    (ComfyUI 21.31)
  edit from a 1152x1536 source         9.36    source itself: 1.91
  edit from a 768x1024 source          3.82    source upscaled to fit: 4.25
```

The middle row is the informative one: given a source at the target's own
resolution, the model *adds* texture — 1.91 in, 9.36 out. It is only when the
source is smaller than the target that output looks flat, because the model
reproduces the source's detail at the new scale rather than inventing more.

So the practical rule for editing is **give it a source at least as large as
the canvas you ask for**. With the edit path also needing ~1.7 MP, a 768x1024
photograph is in the worst position available: too small to edit at its own
size, and stretched 1.5x if you go bigger.

### Multi-reference

`prepare_ref_images` takes K images and both `ref_max_size` and
`cond_image_size` change with K, so every per-image grid has to line up
independently — the kind of code that is written once and quietly wrong.
`tests/test_refimg.py` takes `MDREAM_REF_IMAGE2` and checks K=2 the same way it
checks K=1: input_ids, position_ids, masks and `ar_len` exact, `ref_patches`
and `ref_pixel_values` bit-identical to ComfyUI, and the image-pad count
matching the vision tower's output across both images.

End to end, a person plus a garment fuses correctly — face, cap, background and
framing preserved, the garment transferred with its collar, pockets and
buttons. 1152x1536, two references, 4683 tokens, cfg 5, 28 steps: 407 s.

### Two things deliberately left out

**The SDE samplers.** ComfyUI's `dpmpp_2m_sde_gpu` draws its noise from a
`BrownianTreeNoiseSampler`, which needs `torchsde`. A port could implement a
correct SDE sampler, but not *that* sampler — the Brownian tree's sequence is
not reproducible — so it would be the one piece of this repo that could never
be checked against the reference numerically, only by eye. Against that, it was
measured **worse** than `dpmpp_2m` at 1152x1536 (it returned pure noise where
`dpmpp_2m` gave a clean subject). Its value is confined to the ~4 MP workflows
it was tuned for. Not worth trading the verification discipline for.

### The prefix KV cache, and why it is not built

The reference caches the autoregressive prefix's K/V across sampling steps,
keyed on input_ids, position_ids and the reference images. It is the obvious
missing optimisation here, and measuring what it could save is what stopped it
being written:

```
  text-to-image  768x1024      24 prefix tokens of   792   3.0%
  edit           1152x1536    175 prefix tokens of  4220   4.1%
  edit, 2 refs   1152x1536    388 prefix tokens of  4683   8.3%
```

The prefix is a small share of the sequence, and caching it only removes the
q/k/v projections for those positions — attention still runs over everything.
Call it 2-4% of wall clock. Recorded rather than implemented.

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
mdream/quantize.py    4/6/8-bit, mixed precision, quantised checkpoints
mdream/vision.py      Qwen3-VL vision tower, for the reference-image path
mdream/resample.py    bicubic / bilinear / lanczos, matched to torch and PIL
mdream/refimg.py      reference-image preprocessing and the edit sequence
scripts/generate.py         CLI
scripts/edit.py             reference-image editing CLI
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
tests/test_vision.py         milestone 8, the vision tower, stage by stage
tests/test_resample.py       milestone 8b, resamplers vs torch and PIL
tests/test_refimg.py         milestone 8c, ref preprocessing, bit-identical
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

## License

MIT — see `LICENSE`. This covers the code in this repository only. The
HiDream-O1-Image weights are licensed separately by HiDream-ai, and the
reference implementation this was checked against is ComfyUI's, under its own
license.
