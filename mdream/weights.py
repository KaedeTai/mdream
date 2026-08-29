"""Checkpoint audit: assign every tensor in the HiDream-O1 checkpoint to a module.

This is milestone 1 and it exists to make one claim checkable before any model
code is written: that we understand the whole file. If a tensor cannot be
assigned, the architecture map in README.md is wrong somewhere, and it is much
cheaper to find that out now than after writing 900 lines of MLX.

Run:  python -m mdream.weights [checkpoint.safetensors]
"""
from __future__ import annotations

import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .paths import DEFAULT_CKPT  # noqa: F401  (re-exported)

# What the ComfyUI implementation says must be there.
EXPECT = {
    "lm_layers": 36,
    "vision_blocks": 27,
    "hidden_size": 4096,
    "vocab_size": 151936,
    "vision_hidden": 1152,
    "patch_size": 32,
    "pca_dim": 1024,
}

# Prefix -> the module it belongs to in the port. Order matters: first match wins.
ROUTES = [
    (re.compile(r"^model\.language_model\.layers\.(\d+)\."), "decoder.layer"),
    (re.compile(r"^model\.language_model\.embed_tokens\."),  "decoder.embed"),
    (re.compile(r"^model\.language_model\.norm\."),          "decoder.final_norm"),
    (re.compile(r"^model\.language_model\."),                "decoder.other"),
    (re.compile(r"^model\.visual\.blocks\.(\d+)\."),         "vision.block"),
    (re.compile(r"^model\.visual\.merger\."),                "vision.merger"),
    (re.compile(r"^model\.visual\.deepstack_merger_list\.(\d+)\."), "vision.deepstack(DROPPED)"),
    (re.compile(r"^model\.visual\."),                        "vision.other"),
    (re.compile(r"^model\.x_embedder\."),                    "patch_embed"),
    (re.compile(r"^model\.final_layer2\."),                  "final_layer"),
    (re.compile(r"^model\.t_embedder1\."),                   "timestep_embed"),
]


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def audit(path: Path) -> int:
    hdr = read_header(path)
    groups: dict[str, list[str]] = defaultdict(list)
    indices: dict[str, set[int]] = defaultdict(set)
    unassigned: list[str] = []

    for key in hdr:
        for pat, name in ROUTES:
            m = pat.match(key)
            if m:
                groups[name].append(key)
                if m.groups():
                    indices[name].add(int(m.group(1)))
                break
        else:
            unassigned.append(key)

    total_bytes = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in hdr.values())
    print(f"{path.name}")
    print(f"  {len(hdr)} tensors, {total_bytes / 2**30:.2f} GiB\n")

    print(f"  {'module':28s} {'tensors':>8s}  {'instances':>9s}")
    for name in sorted(groups):
        n_inst = f"{len(indices[name])}" if indices[name] else "-"
        print(f"  {name:28s} {len(groups[name]):8d}  {n_inst:>9s}")

    problems = []
    if unassigned:
        problems.append(f"{len(unassigned)} tensors matched no module: {unassigned[:5]}")
    if len(indices["decoder.layer"]) != EXPECT["lm_layers"]:
        problems.append(f"decoder layers {len(indices['decoder.layer'])} != {EXPECT['lm_layers']}")
    if len(indices["vision.block"]) != EXPECT["vision_blocks"]:
        problems.append(f"vision blocks {len(indices['vision.block'])} != {EXPECT['vision_blocks']}")

    # Shape spot-checks against the config recovered from the reference.
    def shape(k):
        return tuple(hdr[k]["shape"]) if k in hdr else None
    checks = [
        ("model.language_model.embed_tokens.weight", (EXPECT["vocab_size"], EXPECT["hidden_size"])),
        ("model.x_embedder.proj1.weight", (EXPECT["pca_dim"], EXPECT["patch_size"] ** 2 * 3)),
        ("model.x_embedder.proj2.weight", (EXPECT["hidden_size"], EXPECT["pca_dim"])),
        ("model.final_layer2.linear.weight", (EXPECT["patch_size"] ** 2 * 3, EXPECT["hidden_size"])),
    ]
    print()
    for k, want in checks:
        got = shape(k)
        ok = got == want
        print(f"  {'OK ' if ok else 'BAD'} {k:44s} {got} expected {want}")
        if not ok:
            problems.append(f"{k}: {got} != {want}")

    print("\n  one decoder layer (layer 0):")
    for k in sorted(x for x in groups["decoder.layer"] if ".layers.0." in x):
        print(f"    {k[len('model.language_model.layers.0.'):]:38s} {tuple(hdr[k]['shape'])}")

    print("\n  one vision block (block 0):")
    for k in sorted(x for x in groups["vision.block"] if ".blocks.0." in x):
        print(f"    {k[len('model.visual.blocks.0.'):]:38s} {tuple(hdr[k]['shape'])}")

    dead = groups.get("vision.deepstack(DROPPED)", [])
    dead_bytes = sum(hdr[k]["data_offsets"][1] - hdr[k]["data_offsets"][0] for k in dead)
    print(f"\n  deepstack mergers: {len(dead)} tensors, {dead_bytes / 2**20:.0f} MiB — "
          f"present but unused by the reference, do not implement")

    dtypes = Counter(v["dtype"] for v in hdr.values())
    print(f"  dtypes: {dict(dtypes)}")

    if problems:
        print("\n  FAIL")
        for p in problems:
            print(f"    - {p}")
        return 1
    print("\n  PASS — every tensor is accounted for")
    return 0


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CKPT
    raise SystemExit(audit(p))
