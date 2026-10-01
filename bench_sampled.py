#!/usr/bin/env python3
"""Sampled-decoding ruler (plan 0-7; findings G01, G06, F21).

Every frozen ruler is greedy, but real traffic samples. This ruler measures
decode speed and spec-decode acceptance under the sampling profiles clients use:

  A     temperature 1.0, top_p 0.95, top_k 20
  B     temperature 0.7, top_p 0.80, top_k 20
  T1    temperature 1.0, top_p 1.0, top_k 0 (disabled), on the two bench_decode
        ruler prompts (bench_decode.PHASES) only: the T=1 variant of the ruler

Profiles A and B run over the first --prompts prose and --prompts structured
prompts of rulers/prompts/diverse.json. Every request carries an explicit
seed = stable_seed(prompt_id, rep), so an arm pair samples the same streams;
text_sha lets two arms be compared for seeded reproducibility.

Waves hold c distinct (prompt, rep) requests. Rows: request, wave, cell.
Thinking off, max_tokens 256 (T1: 200, as bench_decode).
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

from bench_decode import PHASES, acceptance
from rulers import common as C

PROFILES = {
    "A": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    "B": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
    "T1": {"temperature": 1.0, "top_p": 1.0, "top_k": 0},
}


def requests_for(profile: str, args) -> list[dict]:
    if profile == "T1":
        items = [{"id": f"ruler-{k}", "prompt": v, "category": k} for k, v in PHASES.items()]
        max_tokens = 200
    else:
        items = C.diverse_items(("prose", "structured"), limit=args.prompts)
        max_tokens = args.max_tokens
    out = []
    for it in items:
        for rep in range(args.reps):
            seed = C.stable_seed(it["id"], rep)
            body = C.item_body(args.model, it, max_tokens, seed=seed, **PROFILES[profile])
            out.append({"item": it, "rep": rep, "seed": seed, "body": body})
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(p)
    p.add_argument("--profiles", nargs="+", default=["A", "B", "T1"], choices=list(PROFILES))
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    p.add_argument("--prompts", type=int, default=8, help="prompts per category for A/B (<=32)")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--timeout", type=float, default=600)
    args = p.parse_args()
    C.require_health(args.url)
    em = C.Emitter(__file__, args.out, label=args.label)

    cells = []
    for prof in args.profiles:
        reqs = requests_for(prof, args)
        for c in args.concurrency:
            cell_before = C.spec_from_scrape(C.scrape(args.url))
            by_cat: dict[str, list[float]] = {}
            for w in range(0, len(reqs), c):
                batch = reqs[w : w + c]
                foreign = C.busy(args.url)
                before = C.spec_from_scrape(C.scrape(args.url))
                with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                    rows = list(pool.map(lambda q: C.stream_chat(args.url, q["body"], timeout=args.timeout), batch))
                acc = acceptance(before, C.spec_from_scrape(C.scrape(args.url)))
                for q, r in zip(batch, rows):
                    s = C.step_stats(r)
                    cat = q["item"]["category"]
                    em.row(
                        "request", profile=prof, c=c, wave=w // c, prompt_id=q["item"]["id"], category=cat,
                        rep=q["rep"], seed=q["seed"], text_sha=C.text_sha(r["content"] + str(r["tool_calls"])), **s,
                    )
                    if s.get("ms_per_step") is not None:
                        by_cat.setdefault(cat, []).append(s["ms_per_step"])
                em.row("wave", profile=prof, c=c, wave=w // c, n=len(batch), foreign_requests=foreign,
                       errors=sum(1 for r in rows if r["error"]), **acc)
            cell = {
                "profile": prof,
                "c": c,
                **{f"median_ms_per_step_{k}": C.rnd(C.med(v)) for k, v in sorted(by_cat.items())},
                "n": sum(len(v) for v in by_cat.values()),
                **acceptance(cell_before, C.spec_from_scrape(C.scrape(args.url))),
            }
            em.row("cell", **cell)
            print(" ".join(f"{k}={v}" for k, v in cell.items()), flush=True)
            cells.append(cell)
    em.summary(cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
