#!/usr/bin/env python3
"""Merge the time-boxed per-batch-size L6a tune results into the seed config.

  merge_l6a.py SEED.json OUT.json TUNED_DIR...   (each TUNED_DIR holds the tuner's CFG_NAME file)

Tuned batch keys replace the seed's; keys the time box did not reach keep the seed (B200) entry, so
nearest-key lookup still has an entry for every size up to 8192. Prints which keys came from where.
"""
import json
import sys
from pathlib import Path

CFG = "E=512,N=320,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[64,64].json"
seed_p, out_p, *dirs = sys.argv[1:]
merged = json.loads(Path(seed_p).read_text())
tv = merged.pop("triton_version", None)
src = {k: "seed" for k in merged}
for d in dirs:
    f = Path(d) / CFG
    if not f.exists():
        print(f"skip {d}: no tuned file")
        continue
    t = json.loads(f.read_text())
    tv = t.pop("triton_version", tv)
    for k, v in t.items():
        merged[k] = v
        src[k] = f"tuned({Path(d).name})"
out = {k: merged[k] for k in sorted(merged, key=int)}
if tv is not None:
    out = {"triton_version": tv, **out}
Path(out_p).write_text(json.dumps(out, indent=4) + "\n")
for k in sorted(src, key=int):
    print(k, src[k], json.dumps(merged[k]))
