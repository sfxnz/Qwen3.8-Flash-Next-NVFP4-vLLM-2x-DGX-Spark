#!/usr/bin/env python3
"""GB10 device facts for the kernel track (plan §4: 48 SMs, 24 MB L2, ~99 KB opt-in smem).

Reads CUDA driver attributes through ctypes (libcuda.so.1), so it needs no
deviceQuery binary; torch properties and the NCCL version ride along when the
image has them.

  python3 -S device_facts.py --json OUT
"""
from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# CUdevice_attribute enum values (cuda.h).
ATTRS = {
    "sm_count": 16,
    "clock_khz": 13,
    "mem_clock_khz": 36,
    "mem_bus_width_bits": 37,
    "l2_bytes": 38,
    "max_threads_per_sm": 39,
    "cc_major": 75,
    "cc_minor": 76,
    "smem_per_block": 8,
    "smem_per_sm": 81,
    "smem_per_block_optin": 97,
    "max_persisting_l2_bytes": 108,
    "regs_per_sm": 82,
}
EXPECT = {"sm_count": 48, "l2_bytes": 24 << 20, "smem_per_block_optin": 99 << 10}


def driver_attrs() -> dict:
    cu = ctypes.CDLL("libcuda.so.1")
    if cu.cuInit(0) != 0:
        raise RuntimeError("cuInit failed")
    dev = ctypes.c_int()
    if cu.cuDeviceGet(ctypes.byref(dev), 0) != 0:
        raise RuntimeError("cuDeviceGet failed")
    out = {}
    for name, code in ATTRS.items():
        v = ctypes.c_int()
        rc = cu.cuDeviceGetAttribute(ctypes.byref(v), code, dev)
        out[name] = v.value if rc == 0 else None
    buf = ctypes.create_string_buffer(256)
    cu.cuDeviceGetName(buf, 256, dev)
    out["name"] = buf.value.decode()
    return out


def check(facts: dict) -> dict:
    """Plan facts vs measured: {fact: [expected, measured, ok]}; smem within 2 KB."""
    res = {}
    for k, want in EXPECT.items():
        got = facts.get(k)
        ok = got is not None and (abs(got - want) <= 2048 if k == "smem_per_block_optin" else got == want)
        res[k] = [want, got, ok]
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    res = {"bench": "device", "meta": common.meta()}
    try:
        res["driver"] = driver_attrs()
        res["check"] = check(res["driver"])
    except Exception as e:  # noqa: BLE001
        res["driver_error"] = f"{type(e).__name__}: {e}"
    try:
        import torch

        p = torch.cuda.get_device_properties(0)
        res["torch"] = {k: getattr(p, k) for k in ("name", "multi_processor_count", "L2_cache_size",
                                                     "total_memory", "major", "minor",
                                                     "shared_memory_per_block_optin") if hasattr(p, k)}
        res["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
    except Exception as e:  # noqa: BLE001
        res["torch_error"] = f"{type(e).__name__}: {e}"
    common.write_json(args.json, res)
    for k, (want, got, ok) in res.get("check", {}).items():
        print(f"{k:<22} expect {want:<10} got {got!s:<10} {'ok' if ok else 'MISMATCH'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
