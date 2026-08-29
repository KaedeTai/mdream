#!/usr/bin/env python3
"""Milestone 7 verification: mdream's image vs ComfyUI's, same seed, same sigmas.

Run it once per precision against a matching ComfyUI server, then print the
table:

    # server started with no precision flags
    python3 scripts/compare_vs_comfy.py --precision bf16
    # server restarted with --fp32-unet
    python3 scripts/compare_vs_comfy.py --precision fp32
    python3 scripts/compare_vs_comfy.py --table

Why both precisions matter. A single number ("mdream differs from ComfyUI by
X") is meaningless for a 28-step chaotic trajectory in bf16 -- the reference
does not agree with *itself* across precisions. So the table measures the
envelope first (ComfyUI bf16 vs ComfyUI fp32) and judges mdream against it.
The fp32-vs-fp32 row is the one that proves the arithmetic; the bf16-vs-bf16
row is the shipped path and is expected to land near the quadrature sum of the
two independent bf16 trajectories.

The sigmas come from ComfyUI's own scheduler so that the 2-ULP schedule
difference (see tests/test_sampling.py) cannot contaminate the measurement.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path.home() / "ComfyUI"))

PROMPT = ("a photograph of a red fox sitting in fresh snow at the edge of a "
          "birch forest, golden hour light, shallow depth of field")
CKPT_NAME = "hidream_o1_image_dev_bf16.safetensors"


def comfy_sigmas(steps: int) -> np.ndarray:
    import comfy.samplers
    import comfy.model_sampling as CMS
    import comfy.supported_models as SM
    cfg = SM.HiDreamO1({"image_model": "hidream_o1"})

    class MS(CMS.ModelSamplingDiscreteFlow, CMS.CONST):
        pass
    return comfy.samplers.normal_scheduler(MS(cfg), steps).numpy()


def comfy_run(url: str, prefix: str, prompt: str, w: int, h: int, steps: int,
              seed: int) -> tuple[float, np.ndarray]:
    g = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": CKPT_NAME}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": prompt}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": ""}},
        "4": {"class_type": "EmptyHiDreamO1LatentImage",
              "inputs": {"width": w, "height": h, "batch_size": 1}},
        "5": {"class_type": "BasicScheduler",
              "inputs": {"model": ["1", 0], "scheduler": "normal", "steps": steps,
                         "denoise": 1.0}},
        "6": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "7": {"class_type": "SamplerCustom",
              "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                         "sampler": ["6", 0], "sigmas": ["5", 0], "latent_image": ["4", 0],
                         "add_noise": True, "noise_seed": seed, "cfg": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0],
                                                    "filename_prefix": prefix}},
        "10": {"class_type": "SaveLatent", "inputs": {"samples": ["7", 0],
                                                      "filename_prefix": "latents/" + prefix}},
    }
    req = urllib.request.Request(url + "/prompt", data=json.dumps({"prompt": g}).encode(),
                                 headers={"Content-Type": "application/json"})
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    t0 = time.time()
    while True:
        hist = json.load(urllib.request.urlopen(url + "/history/" + pid))
        if pid in hist and hist[pid].get("status", {}).get("completed") is not None:
            break
        time.sleep(1.0)
    elapsed = time.time() - t0
    name = hist[pid]["outputs"]["10"]["latents"][0]["filename"]
    import safetensors.torch as st
    path = Path.home() / "ComfyUI/output/latents" / name
    return elapsed, st.load_file(str(path))["latent_tensor"].float().numpy()


def compare(name, x, y):
    d = np.abs(x - y)
    ix, iy = np.clip((x + 1) / 2, 0, 1), np.clip((y + 1) / 2, 0, 1)
    mse = float(((ix - iy) ** 2).mean())
    psnr = 10 * np.log10(1.0 / max(mse, 1e-20))
    print(f"  {name:<32} mean|d| {d.mean():.3e}   max {d.max():.3e}   "
          f"PSNR {psnr:6.2f} dB")
    return psnr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["bf16", "fp32"])
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8189")
    ap.add_argument("--cache", default="/tmp/mdream_m7")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)

    if a.table:
        try:
            lat = {k: np.load(cache / f"{k}.npy")
                   for k in ("comfy_bf16", "comfy_fp32", "mdream_bf16", "mdream_fp32")}
        except FileNotFoundError as e:
            print(f"missing {e.filename}; run --precision bf16 and --precision fp32 first")
            return 1
        print(f"milestone 7 - final latent, {a.width}x{a.height}, {a.steps} steps "
              f"euler, cfg 1.0, seed {a.seed}\n")
        print("  the envelope (how far the reference moves from itself):")
        env = compare("comfy bf16  vs comfy fp32", lat["comfy_bf16"], lat["comfy_fp32"])
        compare("mdream bf16 vs mdream fp32", lat["mdream_bf16"], lat["mdream_fp32"])
        print("\n  mdream against the reference:")
        arith = compare("mdream fp32 vs comfy fp32", lat["mdream_fp32"], lat["comfy_fp32"])
        compare("mdream bf16 vs comfy bf16", lat["mdream_bf16"], lat["comfy_bf16"])
        ok = arith > env + 6.0     # at least 2x tighter than the bf16 envelope
        print(f"\n  {'PASS' if ok else 'FAIL'} - the fp32 path is "
              f"{arith - env:+.1f} dB relative to the bf16 envelope")
        return 0 if ok else 1

    if not a.precision:
        ap.error("--precision or --table required")

    import mlx.core as mx
    from mdream.generate import Generator

    sig = comfy_sigmas(a.steps)
    np.save(cache / "sigmas.npy", sig)

    print(f"ComfyUI ({a.precision}) ...", flush=True)
    t_ref, ref = comfy_run(a.comfy_url, f"m7_{a.precision}", a.prompt,
                           a.width, a.height, a.steps, a.seed)
    np.save(cache / f"comfy_{a.precision}.npy", ref)
    print(f"  {t_ref:.1f}s")

    print(f"mdream ({a.precision}) ...", flush=True)
    g = Generator(dtype=mx.bfloat16 if a.precision == "bf16" else mx.float32,
                  verbose=False)
    t0 = time.time()
    _, mine = g.generate(a.prompt, width=a.width, height=a.height, steps=a.steps,
                         seed=a.seed, sigmas=sig, return_latent=True)
    t_mine = time.time() - t0
    np.save(cache / f"mdream_{a.precision}.npy", mine)
    print(f"  {t_mine:.1f}s")

    print()
    compare(f"mdream vs comfy ({a.precision})", mine, ref)
    print(f"\n  wall clock: ComfyUI {t_ref:.1f}s   mdream {t_mine:.1f}s   "
          f"({t_ref / t_mine:.2f}x)")
    print("  (not a fair speed comparison unless both ran alone and warm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
