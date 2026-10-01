#!/usr/bin/env python3
"""Shared helpers for the S1.1 microbenches (plan 0-9).

Pure-python parts (stats, byte models, NCCL env) import without torch, so the
host-side go/no-go builder and the unit tests can use them. The GPU helpers
import torch lazily and only run inside the serve image, serve down.

Timing method (from the sibling's nccl_decode_sweep.py): CUDA events captured
into a CUDA graph mark every slot of every replay. A graph with the call and a
reference graph without it (same flush, same marks) replay alternately; the
per-call cost is slot_with - median(slot_ref). The profiler stays off while
timing.
"""
from __future__ import annotations

import json
import os
import socket
import statistics
import time
from pathlib import Path

# Model facts (config.json of nvidia/Qwen3.8-Flash-Next-NVFP4 @ fab0aec, TP=2).
HIDDEN = 2560
VOCAB = 248320
TP = 2
HC_COUNT = 4
HC_LORA = 320
BF16 = 2

# GB10 roofline figure used for every "x floor" ratio (plan §1.1 uses 230 GB/s).
FLOOR_GBPS = float(os.environ.get("MICRO_FLOOR_GBPS", "230"))
FLUSH_BYTES = 64 << 20  # plan §4: "hot and with a 64 MB flush"

# Serve NCCL env, copied from run.sh docker args (not tuning knobs).
IFACE = "enp1s0f1np1"
HCA = "rocep1s0f1"
HCA_TWIN = "roceP2p1s0f1"
HEAD_IP = "10.100.8.1"
WORKER_IP = "10.100.8.2"
NCCL_BASE_ENV = {
    "NCCL_IB_HCA": HCA,
    "NCCL_NET": "IB",
    "NCCL_IB_DISABLE": "0",
    "NCCL_CROSS_NIC": "1",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_CUMEM_ENABLE": "0",
}


def quantile(xs: list[float], q: float) -> float:
    """Linear-interpolated quantile (numpy 'linear'), stdlib only."""
    s = sorted(xs)
    if not s:
        raise ValueError("empty")
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def stats(us: list[float]) -> dict:
    return {
        "median_us": round(statistics.median(us), 3),
        "p10_us": round(quantile(us, 0.10), 3),
        "p90_us": round(quantile(us, 0.90), 3),
        "n": len(us),
    }


def gbps(nbytes: float, us: float) -> float | None:
    return round(nbytes / (us * 1e-6) / 1e9, 1) if us and us > 0 else None


def floor_us(nbytes: float, bw_gbps: float = FLOOR_GBPS) -> float:
    return nbytes / (bw_gbps * 1e9) * 1e6


def meta() -> dict:
    """Run metadata; torch/vllm versions only when importable."""
    out = {"host": socket.gethostname(), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "image": os.environ.get("MICRO_IMAGE", ""), "floor_gbps": FLOOR_GBPS}
    for mod in ("torch", "vllm", "flashinfer"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001 - record, never fail on metadata
            out[mod] = f"unavailable: {type(e).__name__}"
    return out


def write_json(path: str | None, obj: dict) -> None:
    text = json.dumps(obj, indent=1) + "\n"
    if path:
        Path(path).write_text(text)
    else:
        print(text)


# ---------------------------------------------------------------- GPU helpers
class Timer:
    """Per-call cost of fn inside a CUDA graph, with or without an L2 flush.

    mode "hot":   G calls back to back (weights L2-resident when they fit)
    mode "flush": G x [flush ; call]; the flush is a read-only sum over 64 MB
                  (clean lines, so the call pays no dirty write-back)
    cost per call = slot period with the call - median slot period without it.
    """

    def __init__(self, calls: int = 20, replays: int = 60, warmup: int = 5):
        import torch

        self.torch = torch
        self.calls = calls
        self.replays = replays
        self.warmup = warmup
        self.fbuf = torch.ones(FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")

    def flush(self) -> None:
        self.fbuf.sum()

    def _capture(self, body):
        torch = self.torch
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

    def cost(self, fn, mode: str = "flush", fns: list | None = None) -> dict:
        """fns: optional list of callables cycled per slot (distinct weights,
        a DRAM-streaming 'hot' without flush artefacts); default [fn]."""
        torch = self.torch
        fns = fns or [fn]
        gap = mode == "flush"
        n = self.calls
        marks = {k: [torch.cuda.Event(enable_timing=True, external=True) for _ in range(n + 1)]
                 for k in ("with", "ref")}

        def body(call: bool, m: list):
            def b():
                for i in range(n):
                    m[i].record()
                    if gap:
                        self.flush()
                    if call:
                        fns[i % len(fns)]()
                m[n].record()
            return b

        graphs = {k: self._capture(body(k == "with", marks[k])) for k in ("with", "ref")}
        for _ in range(self.warmup):
            for g in graphs.values():
                g.replay()
        torch.cuda.synchronize()
        slots = {"with": [], "ref": []}
        for _ in range(self.replays):
            for k, g in graphs.items():
                g.replay()
                torch.cuda.synchronize()
                m = marks[k]
                slots[k].append([m[i].elapsed_time(m[i + 1]) * 1e3 for i in range(n)])
        # drop slot 0 (graph-launch ramp)
        ref = statistics.median(x for r in slots["ref"] for x in r[1:])
        cost = [x - ref for r in slots["with"] for x in r[1:]]
        del graphs
        return {**stats(cost), "ref_slot_us": round(ref, 3), "mode": mode, "calls_per_graph": n,
                "distinct_fns": len(fns)}


def profile_kernels(fn, iters: int = 5, pre=None) -> list[dict]:
    """Kernels one call launches, in launch order, with median device us.

    pre (e.g. Timer.flush) runs before every call; its kernels are dropped by
    position (pre is profiled alone first to count them).
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    def kernels(body, n):
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                body()
            torch.cuda.synchronize()
        evs = [e for e in prof.events() if getattr(e, "device_type", None) is not None
               and str(e.device_type).endswith("CUDA")]
        evs.sort(key=lambda e: e.time_range.start)
        return [(e.name, e.time_range.elapsed_us()) for e in evs]

    try:
        npre = len(kernels(pre, 1)) if pre else 0
        fn()  # warm (autotune, lazy init)

        def one():
            if pre:
                pre()
            fn()

        seq = kernels(one, iters)
        if not seq or len(seq) % iters:
            return [{"name": f"irregular kernel count {len(seq)} over {iters} calls", "us": None}]
        k = len(seq) // iters
        out = []
        for j in range(npre, k):
            ds = [seq[i * k + j][1] for i in range(iters)]
            out.append({"name": seq[j][0][:160], "us": round(statistics.median(ds), 2)})
        return out
    except Exception as e:  # noqa: BLE001 - kernel names are auxiliary evidence
        return [{"name": f"profiler failed: {type(e).__name__}: {e}"[:200], "us": None}]


def cleanup() -> None:
    import gc

    import torch

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
