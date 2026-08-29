"""Milestone 5: T2I sequence assembly matches the reference exactly.

Integer arithmetic, so the bar is equality, not a tolerance. The position ids
carry the 'fix point' jump that puts the image block at a fixed MRoPE
coordinate; getting it wrong shifts every image token's rope and would look like
a subtly wrong model rather than a conditioning bug.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(os.environ.get("MDREAM_COMFYUI",
                                            Path.home() / "ComfyUI")).expanduser()))

from mdream import conditioning as C  # noqa: E402
from comfy.ldm.hidream_o1.conditioning import build_extra_conds  # noqa: E402


def main() -> int:
    fails = []
    print("milestone 5 — T2I conditioning\n")
    rs = np.random.RandomState(5)
    for txt_len, H, W in ((17, 128, 128), (33, 256, 192), (8, 96, 64)):
        ids = rs.randint(1000, 150000, size=(1, txt_len)).astype(np.int64)
        ids[0, -1] = C.TMS_TOKEN_ID
        noise = torch.zeros(1, 3, H, W)
        ref = build_extra_conds(torch.from_numpy(ids), noise, ref_images=None,
                                target_patch_size=C.PATCH_SIZE)
        got = C.build_t2i_conds(ids, H, W)

        checks = [
            ("position_ids", got["position_ids"], ref["position_ids"][0].numpy()),
            ("vinput_mask", got["vinput_mask"], ref["vinput_mask"].numpy()),
            ("token_types", got["token_types"], ref["token_types"].numpy()),
        ]
        tag = f"{txt_len} txt + {H}x{W}"
        for name, a, b in checks:
            ok = a.shape == b.shape and bool(np.array_equal(a, b))
            print(f"  {'OK ' if ok else 'FAIL'} {tag:18s} {name:14s} shape {tuple(a.shape)}")
            if not ok:
                fails.append(f"{tag}/{name}")
                if a.shape == b.shape:
                    d = np.argwhere(a != b)
                    print(f"       first mismatch at {d[0].tolist()}: {a[tuple(d[0])]} vs {b[tuple(d[0])]}")
        ok = got["ar_len"] == ref["ar_len"]
        print(f"  {'OK ' if ok else 'FAIL'} {tag:18s} {'ar_len':14s} {got['ar_len']}")
        if not ok:
            fails.append(f"{tag}/ar_len")

    print()
    if fails:
        print(f"  FAIL — {fails}")
        return 1
    print("  PASS — sequence assembly is identical to the reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
