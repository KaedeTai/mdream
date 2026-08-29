"""Milestone 7a: the sampler, checked against ComfyUI without the 8B model.

Everything the sampler does apart from calling the network is deterministic
integer/float bookkeeping, so it can be made *exact* rather than approximate.
This test does that first, so that when the real generation is run any
difference in the image is known to come from the network, not the schedule.

Three checks:
  1. sigma schedule, "normal" scheduler, shift 3.0 -- within 2 float32 ULP.
     Not exact, and the reason is worth recording: torch's `linspace` switches
     to a SIMD path above ~32 elements and no scalar numpy formula reproduces
     it bit-for-bit. The residual is 2 ULP on the *timestep* ramp, i.e. ~2e-7
     relative on sigma. The end-to-end image comparison sidesteps it entirely
     by importing ComfyUI's sigmas, so it never confounds that measurement.
  2. prompt tokenisation -- exact against ComfyUI's HiDreamO1Tokenizer
  3. the Euler loop driven by a fake network -- exact against
     comfy.k_diffusion.sampling.sample_euler wrapped the way ComfyUI wraps it
     (CONST.calculate_denoised inside, to_d outside)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path.home() / "ComfyUI"))

from mdream import sampling as S             # noqa: E402
from mdream.tokenizer import PromptTokenizer  # noqa: E402

import comfy.model_sampling as CMS            # noqa: E402
import comfy.samplers                         # noqa: E402
import comfy.supported_models as SM           # noqa: E402
import comfy.k_diffusion.sampling as KS       # noqa: E402
from comfy.text_encoders.hidream_o1 import HiDreamO1Tokenizer  # noqa: E402

PROMPTS = [
    "a photograph of a red fox sitting in fresh snow, golden hour light",
    "",
    "一只橘貓坐在窗台上，清晨的阳光",
    "close-up macro of a dew-covered spiderweb, 85mm, f/1.4, bokeh",
]


def comfy_model_sampling():
    cfg = SM.HiDreamO1({"image_model": "hidream_o1"})

    class MS(CMS.ModelSamplingDiscreteFlow, CMS.CONST):
        pass
    return MS(cfg)


def check_schedule() -> bool:
    ms_ref = comfy_model_sampling()
    ms_mine = S.ModelSamplingDiscreteFlow()
    ok = True
    print("  1. sigma schedule (normal, shift 3.0)")
    print(f"     sigma_max ref {ms_ref.sigma_max:.10f}  mine {ms_mine.sigma_max:.10f}")
    print(f"     sigma_min ref {ms_ref.sigma_min:.10f}  mine {ms_mine.sigma_min:.10f}")
    for steps in (1, 4, 8, 20, 28, 40, 50):
        ref = comfy.samplers.normal_scheduler(ms_ref, steps).numpy()
        mine = S.normal_schedule(ms_mine, steps)
        assert ref.shape == mine.shape, f"{ref.shape} != {mine.shape}"
        # distance in float32 ULPs: torch computes the timestep ramp with a
        # vectorised float32 linspace, mdream in float64, so a couple of ULPs
        # apart is the floor, not a disagreement about the schedule.
        ulps = int(np.abs(ref.view(np.int32) - mine.view(np.int32)).max())
        good = ulps <= 2
        ok &= good
        print(f"     steps={steps:<3} n={len(mine):<3} "
              f"{'exact' if ulps == 0 else f'{ulps} ULP'}"
              f"{'' if good else '  TOO FAR'}"
              f"   first={mine[0]:.6f} last-1={mine[-2]:.6f}")
    return ok


def check_tokenizer() -> bool:
    ref = HiDreamO1Tokenizer()
    mine = PromptTokenizer()
    ok = True
    print("  2. tokenisation")
    for p in PROMPTS:
        want = [int(t[0]) for t in ref.tokenize_with_weights(p)["hidream_o1"][0]]
        got = mine.encode(p)[0].tolist()
        same = want == got
        ok &= same
        label = (p[:34] + "...") if len(p) > 34 else (p or "<empty>")
        print(f"     {'exact' if same else 'MISMATCH'}  n={len(got):<4} {label}")
    return ok


def check_euler() -> bool:
    """A fake network with memory, so any reordering or reuse of x shows up."""
    print("  3. Euler loop")
    rs = np.random.RandomState(3)
    A = rs.randn(1, 3, 8, 8).astype(np.float32) * 0.3
    B = rs.randn(1, 3, 8, 8).astype(np.float32) * 0.1

    def velocity(x, sigma):
        return x * (0.7 + 0.2 * sigma) + A * np.sin(sigma * 3.0) + B

    ms_ref = comfy_model_sampling()
    sigmas_t = comfy.samplers.normal_scheduler(ms_ref, 12)
    sigmas = sigmas_t.numpy()

    noise = rs.randn(1, 3, 8, 8).astype(np.float32)
    x0 = S.initial_noise_scaling(float(sigmas[0]), noise, np.zeros_like(noise))
    x0_ref = ms_ref.noise_scaling(sigmas_t[0], torch.from_numpy(noise),
                                  torch.zeros(1, 3, 8, 8)).numpy()
    scale_ok = np.array_equal(x0, x0_ref)
    print(f"     initial noise scaling  {'exact' if scale_ok else 'MISMATCH'}"
          f"   |x0|max {np.abs(x0).max():.4f}  (noise_scale {ms_ref.noise_scale})")

    class Wrapped:
        """Stands in for ComfyUI's CFGGuider: network -> denoised."""
        def __call__(self, x, sigma_vec, **kw):
            sigma = float(sigma_vec.reshape(-1)[0])
            v = velocity(x.numpy(), sigma)
            return ms_ref.calculate_denoised(torch.tensor(sigma),
                                             torch.from_numpy(v), x)

    ref = KS.sample_euler(Wrapped(), torch.from_numpy(x0_ref), sigmas_t,
                          disable=True).numpy()
    mine = S.sample_euler(lambda x, sigma, i: velocity(x, sigma), x0, sigmas)
    err = float(np.abs(ref - mine).max())
    loop_ok = err == 0.0
    print(f"     12-step trajectory     {'exact' if loop_ok else 'diff %.3e' % err}"
          f"   |x|max {np.abs(mine).max():.4f}")
    return scale_ok and loop_ok


def main() -> int:
    print("milestone 7a - sampler bookkeeping vs ComfyUI\n")
    ok = check_schedule()
    ok &= check_tokenizer()
    ok &= check_euler()
    print("\n  " + ("PASS - tokenizer and Euler loop exact, schedule within 2 ULP"
                    if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
