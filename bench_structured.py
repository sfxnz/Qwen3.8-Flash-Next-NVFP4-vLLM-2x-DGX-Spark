#!/usr/bin/env python3
"""Structured-output (json_schema) ruler (plan 0-7, C22; finding G04).

Cells:
  catalog          --concurrency (8) concurrent response_format=json_schema
                   requests, thinking ON, sampled with fixed seeds
                   (T 0.6 / top_p 0.95 / top_k 20, seed 1000+i). Each asks for a
                   >=150-item catalogue, so outputs run long. Pin: --max-tokens
                   16384 (~11 min). U1 arm: --max-tokens 32768 (~22 min).
  large_maxlength  one request whose schema has a string with maxLength 100000
                   (xgrammar #852), plus a control request with the same prompt
                   and no maxLength. TTFT carries the grammar compile time.

Per request: per-step wall time vs output length. The stream is cut into bins
of --bin-steps chunks (one chunk = one engine step) and the median inter-chunk
gap of each bin is reported, with a least-squares slope in ms per 1000 steps.
A grammar/bitmask cost that grows with output length shows as a positive slope.
Also: finish_reason, completion tokens, json validity and item count.

Acceptance is NOT comparable with the greedy rulers: grammar-rejected drafts are
subtracted from the accepted count (scheduler.py:2728-2736). It is reported for
arm-vs-arm comparison only.

Optional (not run by this script): EngineCore CPU profile while a cell runs, e.g.
`py-spy dump --pid <EngineCore host pid>` on spark1 if py-spy is installed.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

from bench_decode import acceptance
from rulers import common as C

SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def rf(name: str, schema: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}


def binned(times: list[float], bin_steps: int) -> tuple[list[dict], float | None]:
    gaps = [(b - a) * 1000.0 for a, b in zip(times, times[1:])]
    bins = []
    for i in range(0, len(gaps), bin_steps):
        g = gaps[i : i + bin_steps]
        bins.append({"step0": i, "n": len(g), "median_gap_ms": C.rnd(C.med(g)), "p99_gap_ms": C.rnd(C.pct(g, 99))})
    # least-squares slope of gap vs step index, in ms per 1000 steps
    n = len(gaps)
    if n < 10:
        return bins, None
    mx = (n - 1) / 2.0
    my = sum(gaps) / n
    sxx = sum((i - mx) ** 2 for i in range(n))
    sxy = sum((i - mx) * (g - my) for i, g in enumerate(gaps))
    return bins, C.rnd(1000.0 * sxy / sxx) if sxx else None


def check_json(text: str) -> tuple[bool, int | None]:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False, None
    items = obj.get("items") if isinstance(obj, dict) else None
    return True, (len(items) if isinstance(items, list) else None)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(p)
    p.add_argument("--cells", nargs="+", default=["catalog", "large_maxlength"], choices=["catalog", "large_maxlength"])
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=16384, help="catalog cell; pin 16384, U1 32768")
    p.add_argument("--bin-steps", type=int, default=250)
    p.add_argument("--timeout", type=float, default=3600)
    args = p.parse_args()
    C.require_health(args.url)
    em = C.Emitter(__file__, args.out, label=args.label)
    spec = C.load_prompts("structured")
    summary = []

    if "catalog" in args.cells:
        prompts = spec["catalog_prompts"]
        n = args.concurrency
        bodies = [
            C.chat_body(
                args.model, prompts[i % len(prompts)]["prompt"], args.max_tokens, thinking=True,
                response_format=rf("catalog", spec["catalog_schema"]), seed=1000 + i, **SAMPLING,
            )
            for i in range(n)
        ]
        foreign = C.busy(args.url)
        before = C.spec_from_scrape(C.scrape(args.url))
        with ThreadPoolExecutor(max_workers=n) as pool:
            rows = list(pool.map(lambda b: C.stream_chat(args.url, b, timeout=args.timeout), bodies))
        acc = acceptance(before, C.spec_from_scrape(C.scrape(args.url)))
        slopes, steps = [], []
        for i, r in enumerate(rows):
            s = C.step_stats(r)
            bins, slope = binned(r["times"], args.bin_steps)
            ok, n_items = check_json(r["content"])
            em.row(
                "request", cell="catalog", idx=i, seed=1000 + i, reasoning_chunks=r["reasoning_chunks"],
                content_chunks=r["content_chunks"], json_valid=ok, n_items=n_items,
                slope_ms_per_kstep=slope, bins=bins, **s,
            )
            if slope is not None:
                slopes.append(slope)
            if s.get("ms_per_step") is not None:
                steps.append(s["ms_per_step"])
        cell = {
            "cell": "catalog", "c": n, "max_tokens": args.max_tokens, "foreign_requests": foreign,
            "errors": sum(1 for r in rows if r["error"]),
            "json_valid": sum(1 for r in rows if check_json(r["content"])[0]),
            "median_ms_per_step": C.rnd(C.med(steps)),
            "median_slope_ms_per_kstep": C.rnd(C.med(slopes)),
            "median_completion_tokens": C.med(int((r["usage"] or {}).get("completion_tokens") or 0) for r in rows),
            "acceptance_not_comparable": True, **acc,
        }
        em.row("cell", **cell)
        summary.append(cell)

    if "large_maxlength" in args.cells:
        prompt = spec["large_maxlength_prompt"]
        # xgrammar caches compiled grammars by schema text for the server's life, so
        # both schemas carry a per-run salt (an annotation; the grammar is unchanged).
        # Without it a second run, or a warm-up that reused the control schema, would
        # serve a cached grammar and hide the compile cost this cell measures.
        salt = uuid.uuid4().hex[:8]
        big = {**json.loads(json.dumps(spec["large_maxlength_schema"])), "description": f"run {salt}"}
        control = json.loads(json.dumps(big))
        control["properties"]["body"].pop("maxLength", None)
        # Discarded warm-up with a DIFFERENT schema, so the control does not carry the
        # first-grammar init cost but still compiles its own grammar cold.
        warm = {"type": "object", "properties": {"hi": {"type": "string"}}, "required": ["hi"],
                "description": f"warm {salt}"}
        C.stream_chat(args.url, C.chat_body(args.model, "Say hi as JSON.", 8, response_format=rf("warm", warm)))
        for name, schema in (("control", control), ("maxlength_100k", big)):
            body = C.chat_body(args.model, prompt, 1024, response_format=rf("essay", schema), seed=7)
            r = C.stream_chat(args.url, body, timeout=args.timeout)
            s = C.step_stats(r)
            ok, _ = check_json(r["content"])
            row = em.row("request", cell="large_maxlength", variant=name, json_valid=ok, **s)
            summary.append({k: row[k] for k in ("cell", "variant", "ttft_s", "ms_per_step", "error", "json_valid") if k in row})

    em.summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
