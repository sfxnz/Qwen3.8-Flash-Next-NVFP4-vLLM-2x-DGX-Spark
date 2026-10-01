#!/usr/bin/env python3
"""Long-context ruler (plan 0-7; findings F05, F23, F24, F25). Owner-approved.

Cells, each a cold (salted, never prefix-cached) prompt with three needles at
depths 0.1 / 0.5 / 0.9:

  c=1 at 8k, 32k, 64k, 128k tokens (--with-250k adds a 250k cell)
  c=4 at 32k (four distinct prompts at once)

Every request asks for the three codes, then decodes exactly --itl-tokens (200)
tokens with ignore_eos, so each cell yields cold TTFT, a 200-token ITL / ms/step
figure at that context, and a needle score (codes found in the answer).
Per-request timeout 900 s.

While the ruler runs, a sampler records `free -h` and a vmstat line every 5 s on
the head and on --worker-host (read-only, ssh BatchMode) into
<out>.host.jsonl (default <recipe>/rulers-out/longctx-<ts>.host.jsonl). spark1 runs the
firmware on which long-prefill power-offs are reported (G17): watch the log.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rulers import common as C

DEPTHS = (0.1, 0.5, 0.9)
QUESTION = (
    "The document above hides three vault codes of the form VAULT-<word>-<number>. "
    "List all three codes, one per line, in the order they appear. Output only the codes."
)


def make(args, size: int, tag: str) -> tuple[dict, list[str], int | None]:
    rng = random.Random(tag)
    codes = [f"VAULT-{rng.choice(['amber', 'cobalt', 'ivory', 'jade', 'onyx', 'ruby'])}-{rng.randint(1000, 9999)}" for _ in DEPTHS]
    needles = [(d, f"Remember this: the vault code is {code}.") for d, code in zip(DEPTHS, codes)]
    text, n = C.long_prompt(args.url, args.model, size, tag, question=QUESTION, needles=needles)
    body = C.chat_body(args.model, text, args.itl_tokens, ignore_eos=True)
    return body, codes, n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(p)
    p.add_argument("--sizes", type=int, nargs="+", default=[8192, 32768, 65536, 131072], help="c=1 context sizes (tokens)")
    p.add_argument("--with-250k", action="store_true", help="also run a 250k c=1 cell (opt-in)")
    p.add_argument("--c4-size", type=int, default=32768, help="context size of the c=4 cell; 0 skips it")
    p.add_argument("--itl-tokens", type=int, default=200)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--worker-host", default="spark2", help="'' disables worker sampling")
    p.add_argument("--sample-period", type=float, default=5.0)
    args = p.parse_args()
    C.require_health(args.url)
    em = C.Emitter(__file__, args.out, label=args.label)
    log = Path(args.out).with_suffix(".host.jsonl") if args.out else C.ROOT / "rulers-out" / f"longctx-{int(time.time())}.host.jsonl"
    sampler = C.HostSampler(log, args.worker_host or None, args.sample_period)
    sampler.start()
    salt = uuid.uuid4().hex[:8]
    sizes = list(args.sizes) + ([250000] if args.with_250k else [])
    cells = []
    try:
        plan = [(s, 1) for s in sizes] + ([(args.c4_size, 4)] if args.c4_size else [])
        for size, c in plan:
            for rep in range(args.reps):
                reqs = [make(args, size, f"{salt}-{size}-c{c}-r{rep}-{i}") for i in range(c)]
                foreign = C.busy(args.url)
                with ThreadPoolExecutor(max_workers=c) as pool:
                    rows = list(pool.map(lambda q: C.stream_chat(args.url, q[0], timeout=args.timeout), reqs))
                found_all = []
                for (body, codes, n), r in zip(reqs, rows):
                    s = C.step_stats(r)
                    found = sum(1 for code in codes if code in r["content"])
                    found_all.append(found)
                    em.row("request", size=size, c=c, rep=rep, target_tokens=size, tokenized=n,
                           needles_found=found, needles=len(codes), answer_head=r["content"][:120], **s)
                ttft = [C.step_stats(r)["ttft_s"] for r in rows]
                mps = [C.step_stats(r).get("ms_per_step") for r in rows]
                cell = {
                    "size": size, "c": c, "rep": rep, "foreign_requests": foreign,
                    "errors": sum(1 for r in rows if r["error"]),
                    "median_ttft_s": C.rnd(C.med(t for t in ttft if t is not None)),
                    "median_ms_per_step": C.rnd(C.med(m for m in mps if m is not None)),
                    "needles_found": sum(found_all), "needles_total": len(DEPTHS) * c,
                }
                em.row("cell", **cell)
                print(" ".join(f"{k}={v}" for k, v in cell.items()), flush=True)
                cells.append(cell)
    finally:
        sampler.stop()
        print(f"host samples: {sampler.samples} -> {log}", flush=True)
    em.summary(cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
