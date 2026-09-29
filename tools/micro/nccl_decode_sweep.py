#!/usr/bin/env python3
"""Two-node NCCL cost at this recipe's decode collective sizes, serve down (plan 0-9, S1.1).

Adapted from the DeepSeek-V4.1 sibling's tools/nccl_decode_sweep.py. The serve's
TP backend is vLLM's PyNcclCommunicator (F13: every collective is PYNCCL), so
this drives that class with the image's libnccl. Sizes (hidden 2560, vocab
248320, TP=2, bf16):

  all_reduce     M x 2560          5-160 KB: 107 per k=3 spec step (F13)
  all_gather     M x 2560 per rank the same payload gathered (L2d one-shot AR proxy)
  all_gather_lm  M x 124160        248 KB - 7.9 MB: target and draft logits
for M in (1, 4, 8, 16, 32).

Modes per size (sibling method, profiler off):
  eager    one launch per call, CUDA events, GPU spin first (device time)
  graph    G calls back to back in one CUDA graph
  gapped   G x [64 MB read flush ; call]: the serve's collective follows a
           bandwidth-bound kernel inside a graph; the slowed neighbour is in the cost
  prefetch (all_reduce at M=4 and 32 only) G x [flush ; AR ; HC-module GEMM
           pair on 13.4 MB]: serial, vs the same with the module read into L2 on
           a side stream while the AR runs. saving_us = serial - overlapped.
Graph modes replay a graph with and without the calls alternately; per slot
cost = period_with - median period_ref; steady = median over slots >= 5.

Arms (one process per arm: NCCL reads NCCL_GRAPH_MIXING_SUPPORT at init):
  keep  the serve env from run.sh (graph mixing on, NCCL default)
  mix0  keep + NCCL_GRAPH_MIXING_SUPPORT=0 (L2a)

  run:        python3 -S nccl_decode_sweep.py run --rank R --master HOST:PORT --arm NAME --json OUT
  arms / arm-env NAME / specs
  summarize:  python3 nccl_decode_sweep.py summarize DIR --out nccl-decode.json
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ROWS = (1, 4, 8, 16, 32)
PREFETCH_ROWS = (4, 32)
ARMS: dict[str, dict] = {
    "keep": {},
    "mix0": {"NCCL_GRAPH_MIXING_SUPPORT": "0"},
}
GRAPH_CALLS = 30
STEADY_FROM = 5
EAGER_ITERS = 300
GRAPH_REPLAYS = int(os.environ.get("NCCL_SWEEP_REPLAYS", "150"))
WARMUP = 20
EAGER_SPIN_CYCLES = 100_000_000

# Collectives per k=3 spec step (F13, verified in code): target 98 AR at the
# verify rows (c x 4) + 1 logits AG; per drafter step 3 AR + 3 AG, step 0 at
# c x 4 rows, steps 1-2 at c rows. Used for the L2 booking rule.
STEP_WEIGHTS = {
    "c1": {("all_reduce", 4): 98 + 3, ("all_reduce", 1): 6},
    "c8": {("all_reduce", 32): 98 + 3, ("all_reduce", 8): 6},
}


def arm_env(name: str) -> dict:
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; known: {', '.join(ARMS)}")
    env = dict(common.NCCL_BASE_ENV)
    env.update(ARMS[name])
    return env


def decode_specs() -> list[dict]:
    h, v = common.HIDDEN, common.VOCAB // common.TP
    out = [{"op": "all_reduce", "rows": m, "bytes": m * h * common.BF16} for m in ROWS]
    out += [{"op": "all_gather", "rows": m, "bytes": m * h * common.BF16} for m in ROWS]
    out += [{"op": "all_gather_lm", "rows": m, "bytes": m * v * common.BF16} for m in ROWS]
    return out


def graph_costs(with_slots: list[list[float]], ref_slots: list[list[float]]) -> dict:
    calls = len(with_slots[0])
    if any(len(r) != calls for r in with_slots + ref_slots) or calls <= STEADY_FROM:
        raise ValueError("every replay needs the same number of slots, more than STEADY_FROM")
    ref_med = [statistics.median(r[s] for r in ref_slots) for s in range(calls)]
    cost = [[r[s] - ref_med[s] for s in range(calls)] for r in with_slots]
    steady = [c[s] for c in cost for s in range(STEADY_FROM, calls)]
    st = statistics.median(steady)
    first = [statistics.median(c[s] for c in cost) - st for s in range(STEADY_FROM)]
    return {"steady": steady, "startup_us": round(sum(first), 2)}


def run(args) -> int:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    dist.init_process_group("gloo", init_method=f"tcp://{args.master}", rank=args.rank, world_size=2)
    comm = PyNcclCommunicator(group=dist.group.WORLD, device=dev)
    if comm.disabled:  # disabled comm makes every call a no-op: timings would be empty graphs
        raise RuntimeError("PyNcclCommunicator is disabled (libnccl not loaded or VLLM_DISABLE_PYNCCL)")
    flush = torch.ones(common.FLUSH_BYTES // 4, dtype=torch.float32, device=dev)
    ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731

    def sync():
        torch.cuda.synchronize()
        dist.barrier()

    def op_fn(spec: dict):
        n = spec["bytes"] // common.BF16
        x = torch.randn(n, dtype=torch.bfloat16, device=dev)
        if spec["op"] == "all_reduce":
            out = torch.empty_like(x)
            return lambda: comm.all_reduce(x, out)
        out = torch.empty(n * 2, dtype=torch.bfloat16, device=dev)
        return lambda: comm.all_gather(out, x)

    def capture(body):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        torch.cuda.synchronize()
        return g

    def replay_ab(graphs: dict, marks: dict) -> dict:
        for _ in range(WARMUP // 2):
            for g in graphs.values():
                g.replay()
        sync()
        out = {k: [] for k in graphs}
        for _ in range(GRAPH_REPLAYS):
            for k, g in graphs.items():
                g.replay()
                torch.cuda.synchronize()
                m = marks[k]
                out[k].append([m[i].elapsed_time(m[i + 1]) * 1e3 for i in range(len(m) - 1)])
        return out

    def new_marks():
        return [torch.cuda.Event(enable_timing=True, external=True) for _ in range(GRAPH_CALLS + 1)]

    def ab(body_with, body_ref) -> tuple[dict, float]:
        marks = {"with": new_marks(), "ref": new_marks()}
        gs = {"with": capture(body_with(marks["with"])), "ref": capture(body_ref(marks["ref"]))}
        t = replay_ab(gs, marks)
        c = graph_costs(t["with"], t["ref"])
        ref = statistics.median(x for r in t["ref"] for x in r)
        del gs
        sync()
        return c, ref

    def loop(marks, *steps):
        def b():
            for i in range(GRAPH_CALLS):
                marks[i].record()
                for s in steps:
                    s()
            marks[GRAPH_CALLS].record()
        return b

    rows = []
    for spec in decode_specs():
        f = op_fn(spec)
        for _ in range(WARMUP):
            f()
        sync()
        pairs = [(ev(), ev()) for _ in range(EAGER_ITERS)]
        torch.cuda._sleep(EAGER_SPIN_CYCLES)
        for a, b in pairs:
            a.record()
            f()
            b.record()
        torch.cuda.synchronize()
        rows.append({**spec, "mode": "eager", **common.stats([a.elapsed_time(b) * 1e3 for a, b in pairs])})
        sync()
        fl = lambda: flush.sum()  # noqa: E731
        for mode, w_steps, r_steps in (("graph", (f,), ()), ("gapped", (fl, f), (fl,))):
            c, ref = ab(lambda m, s=w_steps: loop(m, *s), lambda m, s=r_steps: loop(m, *s))
            rows.append({**spec, "mode": mode, "calls_per_graph": GRAPH_CALLS, "startup_us": c["startup_us"],
                         "ref_slot_us": round(ref, 3), **common.stats(c["steady"])})
        if args.rank == 0:
            for r in rows[-3:]:
                print(f"{r['op']:<13} m={r['rows']:<2} {r['bytes']:>8d} B {r['mode']:<6} "
                      f"med {r['median_us']:8.2f} p10 {r['p10_us']:8.2f} p90 {r['p90_us']:8.2f} us", flush=True)

    # L2 prefetch of one HC module (13.4 MB) under a real all-reduce (F16 revisit rule).
    hhs = common.HC_COUNT * common.HIDDEN
    wd = torch.randn(336, hhs, dtype=torch.bfloat16, device=dev) * 0.01
    wu = torch.randn(hhs, common.HC_LORA, dtype=torch.bfloat16, device=dev) * 0.05
    side = torch.cuda.Stream()
    for m in PREFETCH_ROWS:
        spec = {"op": "all_reduce", "rows": m, "bytes": m * common.HIDDEN * common.BF16}
        f = op_fn(spec)
        xn = torch.randn(m, hhs, dtype=torch.bfloat16, device=dev)
        chain = lambda xn=xn: F.linear(F.linear(xn, wd)[:, :common.HC_LORA], wu)  # noqa: E731
        fl = lambda: flush.sum()  # noqa: E731

        def overlapped(f=f):
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                wd.sum()
                wu.sum()
            f()
            torch.cuda.current_stream().wait_stream(side)

        # "with" = overlapped, "ref" = serial; cost = overlapped - serial (negative = saving)
        c, ref = ab(lambda mk: loop(mk, fl, overlapped, chain), lambda mk: loop(mk, fl, f, chain))
        st = common.stats(c["steady"])
        rows.append({**spec, "mode": "prefetch", "serial_slot_us": round(ref, 3),
                     "saving_us": round(-st["median_us"], 3), **st})
        if args.rank == 0:
            print(f"prefetch 13.4 MB under AR m={m}: saving {-st['median_us']:.2f} us/window "
                  f"(serial slot {ref:.1f} us)", flush=True)
        sync()

    res = {
        "arm": args.arm, "rank": args.rank, "host": socket.gethostname(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "nccl_version": comm.nccl.ncclGetVersion(), "torch": torch.__version__,
        "image": os.environ.get("MICRO_IMAGE", ""),
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("NCCL_")},
        "graph_replays": GRAPH_REPLAYS, "rows": rows,
    }
    comm.destroy()
    dist.destroy_process_group()
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1) + "\n")
    return 0


def key(r: dict) -> tuple:
    return (r["op"], r["rows"], r["mode"])


def summarize(root: Path) -> dict:
    """Median over reps of each (arm, op, rows, mode) cell from rank-0 JSONs,
    plus the modelled per-step collective ms (gapped steady x STEP_WEIGHTS)."""
    arms: dict[str, dict] = {}
    for f in sorted(root.glob("*/rep*.rank0.json")):
        d = json.loads(f.read_text())
        a = arms.setdefault(d["arm"], {"reps": 0, "cells": {}, "nccl_version": d.get("nccl_version")})
        a["reps"] += 1
        for r in d["rows"]:
            a["cells"].setdefault(key(r), []).append(r)
    out = {"arms": {}}
    for name, a in arms.items():
        cells = []
        for k, rs in sorted(a["cells"].items()):
            c = {"op": k[0], "rows": k[1], "mode": k[2], "bytes": rs[0]["bytes"], "reps": len(rs),
                 "median_us": round(statistics.median(r["median_us"] for r in rs), 2),
                 "rep_medians_us": [r["median_us"] for r in rs]}
            if "saving_us" in rs[0]:
                c["saving_us"] = round(statistics.median(r["saving_us"] for r in rs), 2)
            cells.append(c)
        gap = {(c["op"], c["rows"]): c["median_us"] for c in cells if c["mode"] == "gapped"}
        model = {w: round(sum(n * gap.get(k, float("nan")) for k, n in wt.items()) / 1e3, 3)
                 for w, wt in STEP_WEIGHTS.items()}
        out["arms"][name] = {"reps": a["reps"], "nccl_version": a["nccl_version"], "cells": cells,
                             "modeled_ms_per_step": model}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--rank", type=int, required=True)
    r.add_argument("--master", required=True, help="host:port reachable from both ranks")
    r.add_argument("--arm", required=True)
    r.add_argument("--json")
    e = sub.add_parser("arm-env")
    e.add_argument("arm")
    sub.add_parser("arms")
    sub.add_parser("specs")
    s = sub.add_parser("summarize")
    s.add_argument("dir")
    s.add_argument("--out")
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
    if args.cmd == "specs":
        for sp in decode_specs():
            print(f"{sp['op']} m={sp['rows']} {sp['bytes']} B")
        return 0
    common.write_json(args.out, summarize(Path(args.dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
