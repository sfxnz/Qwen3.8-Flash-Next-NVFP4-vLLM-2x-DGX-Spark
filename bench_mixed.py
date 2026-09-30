#!/usr/bin/env python3
"""Mixed-traffic ruler (plan 0-7; findings G20, G21, G22).

bench_decode never sends anything during steady-state decode, so it cannot see
prefill interference. Three cells only, each with fresh background decoders:

  base  --decoders (<=6) diverse ignore_eos decoders alone: the ITL reference.
  A     the same decoders plus short requests with Poisson arrivals at
        --rate (lambda <= 0.5/s), each with a unique prefix, max_tokens 32.
  B     the same decoders plus a scripted 32k-token prefill, then a 128k-token
        prefill. A ~500-token probe is sent 2 s after each long prefill starts
        (the HOL-blocking case of G21). Long prompts and probes are unique
        (salted), so the prefix cache cannot shorten them.

Reports, per cell: short/probe TTFT (p50/p99), decoder inter-chunk gap (ITL per
engine step) p50/p99/max inside the cell window, and for B the decoder ITL
while each long prefill is in flight. Rows: request, cell.
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from rulers import common as C


def start_decoders(args, salt: str, t_ref: float, stop: threading.Event, pool):
    items = C.diverse_items(("prose", "structured"))
    rng = random.Random(f"mixed-decoders:{args.seed}")
    chosen = rng.sample([i for i in items if not i.get("tools")], args.decoders)
    futs = []
    for it in chosen:
        body = C.chat_body(args.model, f"[{salt} {it['id']}] {it['prompt']}", args.decoder_max_tokens, ignore_eos=True)
        futs.append(pool.submit(C.stream_chat, args.url, body, timeout=args.timeout, stop=stop, t_ref=t_ref))
    return chosen, futs


def abs_times(r: dict) -> list[float]:
    return [r["t_send"] + t for t in r["times"]]


def run_cell(cell: str, args, em, prompts: dict) -> dict:
    salt = uuid.uuid4().hex[:8]
    stop = threading.Event()
    t_ref = time.perf_counter()
    now = lambda: time.perf_counter() - t_ref  # noqa: E731
    pool = ThreadPoolExecutor(max_workers=args.decoders + 32)
    foreign = C.busy(args.url)
    chosen, dec_futs = start_decoders(args, salt, t_ref, stop, pool)
    time.sleep(args.warmup)
    w0 = now()
    side: list[tuple[str, dict]] = []  # (role, future-result)
    windows: list[tuple[str, float, float]] = []

    def send(role: str, body: dict):
        return role, pool.submit(C.stream_chat, args.url, body, timeout=args.timeout, t_ref=t_ref)

    if cell == "base":
        time.sleep(args.duration)
    elif cell == "A":
        rng = random.Random(f"mixed-arrivals:{args.seed}")
        shorts = C.diverse_items(("prose",))
        pend = []
        t = w0
        k = 0
        while True:
            t += rng.expovariate(args.rate)
            if t - w0 > args.duration:
                break
            time.sleep(max(0.0, t - now()))
            it = shorts[k % len(shorts)]
            k += 1
            pend.append(send("short", C.chat_body(args.model, f"[{salt} s{k}] {it['prompt']}", 32)))
        side = [(role, f.result()) for role, f in pend]
        time.sleep(max(0.0, w0 + args.duration - now()))
    else:  # B
        for size in args.long_tokens:
            time.sleep(3.0)
            long_body = C.chat_body(args.model, prompts[size], 8)
            role, lf = send(f"long{size // 1024}k", long_body)
            t_long = now()
            time.sleep(2.0)
            pr = send("probe", C.chat_body(args.model, prompts[("probe", size)], 32))
            lr = lf.result()
            windows.append((role, t_long, now()))
            side += [(role, lr), (pr[0], pr[1].result())]
        time.sleep(3.0)
    w1 = now()
    stop.set()
    decs = [f.result() for f in dec_futs]
    pool.shutdown(wait=True)

    gaps = []
    for it, r in zip(chosen, decs):
        g = C.gaps_in_window(abs_times(r), w0, w1)
        gaps += g
        em.row("request", cell=cell, role="decoder", prompt_id=it["id"], window_chunks=len(g),
               itl_p50_ms=C.rnd(C.pct(g, 50)), itl_p99_ms=C.rnd(C.pct(g, 99)), error=r["error"])
    ttfts = {}
    for role, r in side:
        s = C.step_stats(r)
        em.row("request", cell=cell, role=role, t_send=C.rnd(r["t_send"]), **s)
        if s["ttft_s"] is not None:
            ttfts.setdefault(role, []).append(s["ttft_s"])
    out = {
        "cell": cell, "decoders": args.decoders, "window_s": C.rnd(w1 - w0), "foreign_requests": foreign,
        "decoder_itl_p50_ms": C.rnd(C.pct(gaps, 50)), "decoder_itl_p99_ms": C.rnd(C.pct(gaps, 99)),
        "decoder_itl_max_ms": C.rnd(max(gaps) if gaps else None),
        "errors": sum(1 for r in decs if r["error"]) + sum(1 for _, r in side if r["error"]),
    }
    for role, v in ttfts.items():
        out[f"{role}_n"] = len(v)
        out[f"{role}_ttft_p50_s"] = C.rnd(C.pct(v, 50))
        out[f"{role}_ttft_p99_s"] = C.rnd(C.pct(v, 99))
    for role, a, b in windows:
        g = []
        for r in decs:
            g += C.gaps_in_window(abs_times(r), a, b)
        out[f"decoder_itl_during_{role}_p50_ms"] = C.rnd(C.pct(g, 50))
        out[f"decoder_itl_during_{role}_p99_ms"] = C.rnd(C.pct(g, 99))
        out[f"decoder_itl_during_{role}_max_ms"] = C.rnd(max(g) if g else None)
    if cell == "A":
        out["rate_per_s"] = args.rate
    em.row("cell", **out)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(p)
    p.add_argument("--cells", nargs="+", default=["base", "A", "B"], choices=["base", "A", "B"])
    p.add_argument("--decoders", type=int, default=6, help="background decoders (<=6)")
    p.add_argument("--decoder-max-tokens", type=int, default=16384)
    p.add_argument("--rate", type=float, default=0.3, help="arm A Poisson arrivals per second (<=0.5)")
    p.add_argument("--duration", type=float, default=60.0, help="base / A window seconds")
    p.add_argument("--warmup", type=float, default=5.0, help="seconds of decode before the window opens")
    p.add_argument("--long-tokens", type=int, nargs="+", default=[32768, 131072], help="arm B scripted prefills, in order")
    p.add_argument("--probe-tokens", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=900)
    args = p.parse_args()
    if not 1 <= args.decoders <= 6:
        p.error("--decoders must be 1..6")
    if not 0 < args.rate <= 0.5:
        p.error("--rate must be in (0, 0.5]")
    C.require_health(args.url)
    em = C.Emitter(__file__, args.out, label=args.label)

    prompts: dict = {}
    if "B" in args.cells:
        run_salt = uuid.uuid4().hex[:8]
        for size in args.long_tokens:
            prompts[size], n = C.long_prompt(args.url, args.model, size, f"{run_salt}-{size}",
                                             question="Summarise the document above in one sentence.")
            prompts[("probe", size)], _ = C.long_prompt(args.url, args.model, args.probe_tokens,
                                                        f"{run_salt}-probe-{size}", question="Name one word used above.")
            print(f"built long prompt {size}: {n} tokens", flush=True)

    cells = [run_cell(c, args, em, prompts) for c in args.cells]
    for c in cells:
        print(" ".join(f"{k}={v}" for k, v in c.items()), flush=True)
    em.summary(cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
