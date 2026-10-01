#!/usr/bin/env python3
"""HyperConnection module chain, one GPU, serve down (plan 0-9, S1.1, F14, L4 baseline).

A standalone replica of nvidia/hyperconnection.py GatedResidual.combine_and_mix
with random weights at the real shapes, calling the image's own Triton glue
(vllm.models.qwen4_exp.nvidia.ops.hc) and F.linear for the projections, the
same 5 kernels a serve module runs:

  hc_combine_norm(residual [M,10240], block_out [M,2560], injection [M,4], w [10240])
  F.linear(xn, W_down_inject [336,10240])   -> lora 320 | injection 4 | pad 12
  hc_silu(lora)
  F.linear(lora, W_up [10240,320])          -> gate [M,10240]
  hc_gate_mix(xn, gate)                     -> block_input [M,2560]

Modes: hot (one module, 13.43 MB of weights, L2-resident), stream (8 distinct
modules cycled, DRAM-bound with no flush artefacts, the serve case), flush
(one module after a 64 MB flush). The per-kernel split comes from one profiled
pass after a flush. Floor = 13.43 MB at MICRO_FLOOR_GBPS.

  python3 -S hc_bench.py run --json OUT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

HHS = common.HC_COUNT * common.HIDDEN  # 10240
DOWN_N = common.HC_LORA + common.HC_COUNT
PAD = (-DOWN_N) % 16
DOWN_ROWS = DOWN_N + PAD  # 336
MODULE_BYTES = (DOWN_ROWS * HHS + HHS * common.HC_LORA) * common.BF16  # 13,434,880
MS = (1, 4, 8, 16, 32)
STREAM_MODULES = 8
EPS = 1e-6


def build(m: int, modules: int):
    import torch
    import torch.nn.functional as F
    from vllm.models.qwen4_exp.nvidia.ops import hc

    dev = "cuda"
    residual = torch.randn(m, HHS, dtype=torch.bfloat16, device=dev)
    block_out = torch.randn(m, common.HIDDEN, dtype=torch.bfloat16, device=dev)
    inj = torch.randn(m, common.HC_COUNT, dtype=torch.bfloat16, device=dev)
    mods = []
    for _ in range(modules):
        mods.append((
            torch.randn(HHS, dtype=torch.bfloat16, device=dev) * 0.1,                      # norm weight
            torch.randn(DOWN_ROWS, HHS, dtype=torch.bfloat16, device=dev) * 0.01,         # down+inject
            torch.randn(HHS, common.HC_LORA, dtype=torch.bfloat16, device=dev) * 0.05,    # up
        ))
    split = [common.HC_LORA, common.HC_COUNT, PAD]

    def chain(p):
        nw, wd, wu = p
        _, xn = hc.hc_combine_norm(residual, block_out, inj, nw, EPS, common.HC_COUNT)
        lora, _inj, _ = F.linear(xn, wd).split(split, dim=-1)
        lora = hc.hc_silu(lora, common.HC_COUNT)
        gate = F.linear(lora, wu)
        return hc.hc_gate_mix(xn, gate, common.HC_COUNT)

    return [lambda p=p: chain(p) for p in mods]


def run(args) -> dict:
    import torch

    torch.cuda.set_device(0)
    timer = common.Timer(calls=args.calls, replays=args.replays)
    rows = []
    floor = common.floor_us(MODULE_BYTES)
    for m in MS:
        try:
            fns = build(m, STREAM_MODULES)
        except Exception as e:  # noqa: BLE001
            rows.append({"m": m, "skipped": f"{type(e).__name__}: {e}"[:300]})
            print(f"HC M={m} SKIP {rows[-1]['skipped']}", flush=True)
            continue
        r = {"m": m, "module_bytes": MODULE_BYTES, "floor_us": round(floor, 2),
             "hot": timer.cost(fns[0], "hot"),
             "stream": timer.cost(fns[0], "hot", fns=fns),
             "flush": timer.cost(fns[0], "flush")}
        for mode in ("hot", "stream", "flush"):
            r[mode]["gbps"] = common.gbps(MODULE_BYTES, r[mode]["median_us"])
            r[mode]["x_floor"] = round(r[mode]["median_us"] / floor, 3)
        if not args.no_kernels:
            r["kernels"] = common.profile_kernels(fns[0], iters=5, pre=timer.flush)
        rows.append(r)
        print(f"HC M={m:<2} hot {r['hot']['median_us']:7.2f} us | stream {r['stream']['median_us']:7.2f} us "
              f"({r['stream']['x_floor']}x floor) | flush {r['flush']['median_us']:7.2f} us", flush=True)
        del fns
        common.cleanup()
    return {"bench": "hc", "meta": common.meta(), "module_bytes": MODULE_BYTES, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--json")
    p.add_argument("--calls", type=int, default=24)
    p.add_argument("--replays", type=int, default=40)
    p.add_argument("--no-kernels", action="store_true")
    sub.add_parser("plan")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        print(f"hc chain M={','.join(map(str, MS))} module {MODULE_BYTES} B down [{DOWN_ROWS},{HHS}] "
              f"up [{HHS},{common.HC_LORA}] modes hot,stream({STREAM_MODULES}),flush")
        return 0
    common.write_json(args.json, run(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
