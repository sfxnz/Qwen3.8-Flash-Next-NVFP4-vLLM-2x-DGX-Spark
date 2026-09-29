#!/usr/bin/env python3
"""BF16 GEMM sweep and draft-head probe, one GPU, serve down (plan 0-9, S1.1).

gemm       torch F.linear (cuBLAS, what every BF16 linear runs on sm_121 at TP=2,
           F15) at M in MS for every per-rank (N, K) in SHAPES. Per cell: CUDA-graph
           replay cost hot and after a 64 MB L2 flush, achieved GB/s over the
           weight bytes, and the kernels one call launches (split-K shows as a
           second reduce kernel). The driver sets CUBLASLT_LOG_LEVEL=5 for a
           separate --algo-log pass so cuBLASLt heuristics land in a file.
drafthead  F.linear on a [V'/2, 2560] BF16 weight plus torch.argmax (C23): the
           L1b draft-head roofline without a K1 prototype.

  python3 -S gemm_sweep.py gemm --json OUT [--ms 1,4] [--algo-log]
  python3 -S gemm_sweep.py drafthead --json OUT
  python3 gemm_sweep.py plan          (prints shapes and M, no GPU)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# (N, K) per rank at TP=2; name -> where it runs (roofline §1, F15).
SHAPES = {
    "gdn_in_proj_qkvz_ba": (8240, 2560),   # L5 merge target: qkvz 8192 + ba 48
    "gdn_in_proj_qkvz": (8192, 2560),
    "gdn_out_proj|qsa_o": (2560, 3072),
    "qsa_qkv+index": (7296, 2560),         # L5 merge target
    "qsa_qkv": (6656, 2560),
    "shared_gate_up|indexer": (640, 2560),
    "gdn_in_proj_ba": (48, 2560),
    "router": (512, 2560),
    "ple_kv_proj": (12800, 2560),
    "lm_head_shard": (124160, 2560),
    # HC GEMMs (S1.1 row "cuBLAS choice for HC down and HC up at M=4")
    "hc_down_inject": (336, 10240),
    "hc_up": (10240, 320),
    "hc_final_down": (320, 10240),
}
MS = (1, 2, 4, 8, 16, 24, 32)
# V' candidates (0-10 picks the smallest with >=99% coverage) and the full vocab.
DRAFT_V = (32768, 47104, 65536, common.VOCAB)
DRAFT_MS = (1, 2, 4, 8)


def cell(timer, fn, nbytes: int, kernels: bool) -> dict:
    out = {}
    for mode in ("hot", "flush"):
        c = timer.cost(fn, mode)
        c["gbps"] = common.gbps(nbytes, c["median_us"])
        out[mode] = c
    if kernels:
        out["kernels"] = common.profile_kernels(fn, iters=5, pre=timer.flush)
    return out


def run_gemm(args) -> dict:
    import torch
    import torch.nn.functional as F

    torch.cuda.set_device(0)
    timer = common.Timer(calls=args.calls, replays=args.replays)
    ms = [int(x) for x in args.ms.split(",")] if args.ms else list(MS)
    names = args.shapes.split(",") if args.shapes else list(SHAPES)
    rows = []
    for name in names:
        n, k = SHAPES[name]
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
        for m in ms:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            fn = lambda x=x, w=w: F.linear(x, w)  # noqa: E731
            if args.algo_log:  # one eager call per cell; cuBLASLt logs the heuristic
                fn()
                torch.cuda.synchronize()
                print(f"ALGO_CELL {name} N={n} K={k} M={m}", flush=True)
                continue
            r = {"name": name, "n": n, "k": k, "m": m, "weight_bytes": n * k * common.BF16,
                 **cell(timer, fn, n * k * common.BF16, not args.no_kernels)}
            r["floor_us"] = round(common.floor_us(r["weight_bytes"]), 2)
            rows.append(r)
            print(f"{name:<24} N={n:<6} K={k:<5} M={m:<2} hot {r['hot']['median_us']:8.2f} us "
                  f"{r['hot']['gbps']} GB/s | flush {r['flush']['median_us']:8.2f} us {r['flush']['gbps']} GB/s "
                  f"| {len(r.get('kernels', []))} kernels", flush=True)
        del w
        common.cleanup()
    return {"bench": "gemm", "meta": common.meta(), "rows": rows}


def run_drafthead(args) -> dict:
    import torch
    import torch.nn.functional as F

    torch.cuda.set_device(0)
    timer = common.Timer(calls=args.calls, replays=args.replays)
    rows = []
    for v in DRAFT_V:
        rows_per_rank = v // common.TP
        w = torch.randn(rows_per_rank, common.HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.02
        nbytes = rows_per_rank * common.HIDDEN * common.BF16
        for m in DRAFT_MS:
            x = torch.randn(m, common.HIDDEN, dtype=torch.bfloat16, device="cuda")
            fn = lambda x=x, w=w: torch.argmax(F.linear(x, w), dim=-1)  # noqa: E731
            r = {"v_prime": v, "rows_per_rank": rows_per_rank, "m": m, "weight_bytes": nbytes,
                 **cell(timer, fn, nbytes, not args.no_kernels)}
            r["floor_us"] = round(common.floor_us(nbytes), 2)
            rows.append(r)
            print(f"draft V'={v:<6} rows={rows_per_rank:<6} M={m} hot {r['hot']['median_us']:8.2f} us "
                  f"{r['hot']['gbps']} GB/s | flush {r['flush']['median_us']:8.2f} us {r['flush']['gbps']} GB/s",
                  flush=True)
        del w
        common.cleanup()
    return {"bench": "drafthead", "meta": common.meta(), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gemm", "drafthead"):
        p = sub.add_parser(name)
        p.add_argument("--json")
        p.add_argument("--calls", type=int, default=20)
        p.add_argument("--replays", type=int, default=40)
        p.add_argument("--no-kernels", action="store_true", help="skip the per-cell profiler pass")
        if name == "gemm":
            p.add_argument("--ms", help="comma list, default " + ",".join(map(str, MS)))
            p.add_argument("--shapes", help="comma list of names, default all")
            p.add_argument("--algo-log", action="store_true",
                           help="one eager call per cell only (run with CUBLASLT_LOG_LEVEL=5)")
    sub.add_parser("plan")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        for name, (n, k) in SHAPES.items():
            print(f"gemm {name} N={n} K={k} M={','.join(map(str, MS))} weight {n * k * common.BF16} B")
        for v in DRAFT_V:
            print(f"drafthead V'={v} rows/rank={v // common.TP} M={','.join(map(str, DRAFT_MS))}")
        return 0
    res = run_gemm(args) if args.cmd == "gemm" else run_drafthead(args)
    if not getattr(args, "algo_log", False):
        common.write_json(args.json, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
