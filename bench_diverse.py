#!/usr/bin/env python3
"""Diverse-prompt decode ruler (plan 0-7; findings F03, F19).

bench_decode.py sends one prompt to every stream at temperature 0, so c>=2
streams decode identical tokens and share routed experts (F03). This ruler
sends a DISTINCT prompt to every stream of a wave:

  prose       32 distinct prose prompts (low-acceptance regime)
  structured  32 code / JSON / CJK / table / tool-call prompts (high-acceptance)

For each category and c in --concurrency, the first --prompts prompts are sent
in waves of c distinct prompts (so every cell covers the same prompt set).
Greedy, thinking off, max_tokens 256, same request body as bench_decode.

Per request: ms/step = decode_s / (chunks - 1) from chunk timestamps (one chunk
is one engine step with stream_interval 1), TTFT, completion tokens.
Per wave and per cell: spec-decode acceptance from /metrics deltas (the counters
are server-global: run on an otherwise idle server; each row records how many
foreign requests were running when the wave started).

Rows: kind=request, kind=wave, kind=cell. Every row carries the RULER_SET version.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

from bench_decode import acceptance
from rulers import common as C


def run_wave(args, items: list[dict]) -> list[dict]:
    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        futs = [
            pool.submit(C.stream_chat, args.url, C.item_body(args.model, it, args.max_tokens), timeout=args.timeout)
            for it in items
        ]
        return [f.result() for f in futs]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(p)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--categories", nargs="+", default=["prose", "structured"], choices=["prose", "structured"])
    p.add_argument("--prompts", type=int, default=32, help="prompts per category (<=32)")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--timeout", type=float, default=600)
    args = p.parse_args()
    C.require_health(args.url)
    em = C.Emitter(__file__, args.out, label=args.label, max_tokens=args.max_tokens)

    cells = []
    for cat in args.categories:
        items = C.diverse_items((cat,), limit=args.prompts)
        for c in args.concurrency:
            cell_before = C.spec_from_scrape(C.scrape(args.url))
            steps, ttfts, tps = [], [], []
            for w in range(0, len(items), c):
                batch = items[w : w + c]
                foreign = C.busy(args.url)
                before = C.spec_from_scrape(C.scrape(args.url))
                rows = run_wave(args, batch)
                acc = acceptance(before, C.spec_from_scrape(C.scrape(args.url)))
                wave_steps = []
                for it, r in zip(batch, rows):
                    s = C.step_stats(r)
                    em.row(
                        "request", category=cat, c=c, wave=w // c, prompt_id=it["id"],
                        prompt_kind=it.get("kind", "prose"), text_sha=C.text_sha(r["content"] + str(r["tool_calls"])), **s
                    )
                    if s.get("ms_per_step") is not None:
                        wave_steps.append(s["ms_per_step"])
                        tps.append(s["tokens_per_step"] or 0)
                    if s["ttft_s"] is not None:
                        ttfts.append(s["ttft_s"])
                steps.extend(wave_steps)
                em.row(
                    "wave", category=cat, c=c, wave=w // c, n=len(batch), foreign_requests=foreign,
                    errors=sum(1 for r in rows if r["error"]),
                    median_ms_per_step=C.rnd(C.med(wave_steps)), **acc,
                )
            cell = {
                "category": cat,
                "c": c,
                "n": len(steps),
                "median_ms_per_step": C.rnd(C.med(steps)),
                "p90_ms_per_step": C.rnd(C.pct(steps, 90)),
                "median_tokens_per_step": C.rnd(C.med(tps)),
                "median_ttft_s": C.rnd(C.med(ttfts)),
                **acceptance(cell_before, C.spec_from_scrape(C.scrape(args.url))),
            }
            em.row("cell", **cell)
            print(
                f"category={cat} c={c} n={cell['n']} ms/step={cell['median_ms_per_step']} "
                f"tok/step={cell['median_tokens_per_step']} AL={cell.get('acceptance_len')}",
                flush=True,
            )
            cells.append(cell)
    em.summary(cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
