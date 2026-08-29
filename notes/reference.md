# Reading order for the PyTorch reference

1. `~/ComfyUI/comfy/ldm/hidream_o1/model.py` — start at `HiDreamO1Transformer.forward`.
   The last 60 lines carry the whole contract: timestep -> sigma, tms token
   substitution, concat of text and pixel embeddings, MRoPE freqs, two-pass
   attention over 36 layers, slice target positions, final projection, unpatch,
   and the fp32 `(x - x_pred) / sigma` at the end (the comment says bf16 there
   noticeably degrades samples — keep that in fp32 in MLX too).

2. `attention.py` — 41 lines. `ar_len` splits the sequence: `[0, ar_len)` is the
   autoregressive prefix and gets causal attention, `[ar_len, T)` attends
   everything. ComfyUI splits Q at the boundary purely to avoid materialising a
   (B, 1, T, T) additive mask (~500 MB at T~16K). In MLX the same split is worth
   keeping for the same reason.

3. `conditioning.py` — how input_ids, position_ids, ref pixel values and the
   vinput mask are built. This is what a port gets wrong first.

4. `comfy/text_encoders/llama.py` `Llama2_` — the decoder layer, including the
   `past_key_value` contract the KV cache uses.

5. `comfy/text_encoders/qwen35.py` `Qwen35VisionModel` — 27 blocks, and note
   the deepstack visual indexes (8, 16, 24) whose merger weights exist in the
   checkpoint but are dropped.

## Things the reference calls out that a port will get wrong

- **Final subtraction must be fp32.** Explicit comment in model.py.
- **`ar_len` boundary.** Get it off by one and the prefix leaks future tokens.
- **MRoPE is interleaved** with dims [24, 20, 20], not a plain 1-D rope.
- **q_norm / k_norm are gemma3-style**, i.e. applied per head before rope.
- **Deepstack mergers are dead weight** — 3 sets of merger tensors in the
  checkpoint are never used. Do not wire them up.
- **image_pad token count must equal ViT output count** — the reference raises
  on mismatch, which is the tokenizer/processor alignment check.
