#!/usr/bin/env python3
"""Two-node bf16 all-reduce, single rail vs dual rail, serve down (plan 0-9, S1.1, F29).

Adapted from the DeepSeek-V4.1 sibling's tools/nccl_allreduce_sweep.py (stand-in
for nccl-tests: the image has no nccl-tests or MPI; same libnccl as the serve).
Sizes: 8 B .. 256 KiB doubling (the decode small-message penalty), the 42 MB
prefill all-reduce (8192-token chunk x 2560 x bf16 = 41,943,040 B), and 32/64/128 MiB.
busbw = algbw * 2(n-1)/n, as in nccl-tests.

Arms:
  single  the serve env from run.sh (NCCL_IB_HCA=rocep1s0f1, CROSS_NIC=1)
  dual    both ACTIVE twins, NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1, CROSS_NIC=0,
          IB_MERGE_NICS=0 (F29: merge measured +30% small-message latency)

  run:     python3 -S nccl_allreduce_sweep.py run --rank R --master HOST:PORT --json OUT
  arm-env: python3 nccl_allreduce_sweep.py arm-env ARM
  compare: python3 nccl_allreduce_sweep.py compare single.json dual.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PREFILL_AR = 8192 * common.HIDDEN * common.BF16  # 41,943,040
LAT_SIZES = tuple(1 << i for i in range(3, 19))  # 8 B .. 256 KiB
BW_SIZES = (32 << 20, PREFILL_AR, 64 << 20, 128 << 20)
ARMS = {
    "single": {},
    "dual": {"NCCL_IB_HCA": f"{common.HCA},{common.HCA_TWIN}", "NCCL_CROSS_NIC": "0",
             "NCCL_IB_MERGE_NICS": "0"},
}


def arm_env(name: str) -> dict:
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; known: {', '.join(ARMS)}")
    env = dict(common.NCCL_BASE_ENV)
    env.update(ARMS[name])
    return env


def sizes() -> list[int]:
    return sorted(set(LAT_SIZES) | set(BW_SIZES))


def busbw(nbytes: int, seconds: float, world: int = 2) -> float:
    return nbytes / seconds / 1e9 * 2 * (world - 1) / world


def iters_for(nbytes: int) -> int:
    return 200 if nbytes <= (1 << 20) else 50 if nbytes <= (32 << 20) else 20


def run(args) -> int:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=f"tcp://{args.master}", rank=args.rank, world_size=2)
    rows = []
    for n in sizes():
        x = torch.ones(max(1, n // 2), dtype=torch.bfloat16, device="cuda")
        it = iters_for(n)
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(it):
            dist.all_reduce(x)
        end.record()
        torch.cuda.synchronize()
        sec = start.elapsed_time(end) / 1e3 / it
        rows.append({"bytes": n, "us": round(sec * 1e6, 2), "busbw_GBps": round(busbw(n, sec), 3), "iters": it})
        if args.rank == 0:
            print(f"{n:>12d} {sec * 1e6:10.1f} us {rows[-1]['busbw_GBps']:8.2f} GB/s", flush=True)
    dist.destroy_process_group()
    if args.rank == 0 and args.json:
        Path(args.json).write_text(json.dumps({"arm": args.arm, "rows": rows}, indent=1) + "\n")
    return 0


def compare(base: dict, cand: dict) -> dict:
    """cand vs base: prefill time ratio and worst small-message latency change."""
    b = {r["bytes"]: r for r in base["rows"]}
    c = {r["bytes"]: r for r in cand["rows"]}
    lat = {n: round(c[n]["us"] / b[n]["us"] - 1, 4) for n in LAT_SIZES if n in b and n in c}
    out = {"lat_change": lat, "worst_small_lat_change": max(lat.values()) if lat else None}
    if PREFILL_AR in b and PREFILL_AR in c:
        out["prefill_ar_us"] = {"base": b[PREFILL_AR]["us"], "cand": c[PREFILL_AR]["us"]}
        out["prefill_speedup"] = round(b[PREFILL_AR]["us"] / c[PREFILL_AR]["us"], 3)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--rank", type=int, required=True)
    r.add_argument("--master", required=True)
    r.add_argument("--arm", default="")
    r.add_argument("--json")
    e = sub.add_parser("arm-env")
    e.add_argument("arm")
    sub.add_parser("arms")
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("cand")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args)
    if args.cmd == "arm-env":
        for k, v in arm_env(args.arm).items():
            print(f"{k}={v}")
        return 0
    if args.cmd == "arms":
        print(" ".join(ARMS))
        return 0
    res = compare(json.loads(Path(args.base).read_text()), json.loads(Path(args.cand).read_text()))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
