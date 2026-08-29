"""Flow-matching sampling for HiDream-O1.

Three pieces, all ported from ComfyUI so they can be checked against it rather
than argued about:

  schedule   ModelSamplingDiscreteFlow with shift=3.0, "normal" scheduler
  noising    CONST.noise_scaling with noise_scale=8.0  ->  x0 = 8 * noise
  step       Euler on the velocity

The 8x noise scale is the surprising one. It is not a stylistic knob: it is in
the model config (`sampling_settings.noise_scale`) and ComfyUI multiplies the
initial noise by it. Sampling with unit noise gives washed-out mush.

The Euler step reduces to something trivial here and it is worth writing down
why, because it is easy to accidentally apply the conversion twice:

    model returns   v      = (x - x_pred) / sigma
    ComfyUI forms   denoised = x - v * sigma          (CONST.calculate_denoised)
    then            d        = (x - denoised) / sigma  (to_d)

so d == v exactly. The two conversions are kept explicit below anyway: they
cost nothing and they are the contract, so if the model's output convention
ever changes the sampler still does the right thing.
"""
from __future__ import annotations

import math
from typing import Callable, List

import numpy as np

SHIFT = 3.0
NOISE_SCALE = 8.0
MULTIPLIER = 1000.0


def time_snr_shift(alpha: float, t: float) -> float:
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


class ModelSamplingDiscreteFlow:
    """Port of comfy.model_sampling.ModelSamplingDiscreteFlow."""

    def __init__(self, shift: float = SHIFT, multiplier: float = MULTIPLIER,
                 timesteps: int = 1000, noise_scale: float = NOISE_SCALE):
        self.shift = shift
        self.multiplier = multiplier
        self.noise_scale = noise_scale
        ts = np.arange(1, timesteps + 1, dtype=np.float64) / timesteps * multiplier
        self.sigmas = np.array([self.sigma(t) for t in ts], dtype=np.float64)

    def sigma(self, timestep: float) -> float:
        return time_snr_shift(self.shift, float(timestep) / self.multiplier)

    def timestep(self, sigma: float) -> float:
        return float(sigma) * self.multiplier

    @property
    def sigma_min(self) -> float:
        return float(self.sigmas[0])

    @property
    def sigma_max(self) -> float:
        return float(self.sigmas[-1])


def normal_schedule(model_sampling: ModelSamplingDiscreteFlow, steps: int,
                    sgm: bool = False) -> np.ndarray:
    """Port of comfy.samplers.normal_scheduler (the "normal" scheduler).

    The timestep ramp is computed in float64 here and in float32 by torch, so
    individual sigmas can land one float32 ULP apart from ComfyUI's. That is
    torch's vectorised `linspace`, not a disagreement about the schedule: the
    float64 values computed here are the exact ones, and the test measures the
    gap in ULPs rather than pretending it is zero.
    """
    s = model_sampling
    start = s.timestep(s.sigma_max)
    end = s.timestep(s.sigma_min)

    append_zero = True
    if sgm:
        timesteps = np.linspace(start, end, steps + 1)[:-1]
    else:
        if math.isclose(s.sigma(end), 0.0, abs_tol=1e-5):
            steps += 1
            append_zero = False
        timesteps = np.linspace(start, end, steps)

    sigs: List[float] = [float(np.float32(s.sigma(t))) for t in timesteps]
    if append_zero:
        sigs.append(0.0)
    return np.array(sigs, dtype=np.float32)


def initial_noise_scaling(sigma: float, noise: np.ndarray, latent: np.ndarray,
                          noise_scale: float = NOISE_SCALE) -> np.ndarray:
    """CONST.noise_scaling. For text-to-image `latent` is zeros, so this is
    just `noise_scale * noise` at sigma = 1."""
    return sigma * (noise_scale * noise) + (1.0 - sigma) * latent


def calculate_denoised(sigma: float, model_output, model_input):
    return model_input - model_output * sigma


def to_d(x, sigma: float, denoised):
    return (x - denoised) / sigma


def sample_euler(model: Callable, x, sigmas: np.ndarray, callback=None):
    """comfy.k_diffusion.sampling.sample_euler with s_churn = 0.

    `dt` is formed by a float32 subtraction before being widened, because that
    is what torch does; subtracting in float64 puts the trajectory ~4e-8 away
    from ComfyUI's for no reason. Everything else is dtype-agnostic on purpose,
    so `x` can be an mx.array and stay on the GPU for the whole loop.
    """
    sigmas = np.asarray(sigmas, dtype=np.float32)
    for i in range(len(sigmas) - 1):
        sigma = float(sigmas[i])
        dt = float(sigmas[i + 1] - sigmas[i])   # float32 subtraction, then widened
        v = model(x, sigma, i)
        denoised = calculate_denoised(sigma, v, x)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(i, sigma, x, denoised)
        x = x + d * dt
    return x


def to_image(x: np.ndarray) -> np.ndarray:
    """Pixel-space "VAE" decode: identity, then ComfyUI's process_output
    (-1..1 -> 0..1) and to uint8. x is (B, 3, H, W)."""
    img = np.clip((x.astype(np.float32) + 1.0) / 2.0, 0.0, 1.0)
    return (img.transpose(0, 2, 3, 1) * 255.0 + 0.5).astype(np.uint8)
