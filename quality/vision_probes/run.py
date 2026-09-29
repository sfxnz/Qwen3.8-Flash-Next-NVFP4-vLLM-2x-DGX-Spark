#!/usr/bin/env python3
"""V gate runner (plan 0-8, G09): sha-pinned synthetic vision probes with answer keys.

  python3 quality/vision_probes/run.py --out RUN_DIR [--only grid_wide,ocr] [--no-pair]
  python3 quality/vision_probes/run.py --write-keys     # after a deliberate probe change

Checks, all greedy with thinking off:
  1. every image regenerates to the pixel sha256 in keys.json (else exit 2, no verdict);
  2. every probe answers its key exactly (grid position W and H, OCR digits,
     chart, counting, quadrants, multi-image order, and the C29 chunk-straddle
     probe: >= 6k text tokens then a 4096-token image);
  3. a c=2 simultaneous image pair: both answers still right.
Reports per probe prompt_tokens; for the straddle probe the image token span
and which prefill chunk boundaries it crosses (MAX_NUM_BATCHED_TOKENS 8192 now,
8000/4800 in the plan).
Pass = all probes exact and the pair right. HTTP 400 "is not a multimodal
model" fails loudly.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import probes as P  # noqa: E402
from qlib import DEFAULT_URL, OFF, Client, HTTPFailure, append_jsonl, prepare_out, write_json, write_manifest  # noqa: E402

KEYS = HERE / "keys.json"
PAIR = ("grid_wide.red_col", "ocr.407218")
CHUNKS = (4800, 8000, 8192)
NOT_MM = "is not a multimodal model"


def expected_keys() -> dict:
    return {"images": P.image_shas(), "probes": {p["id"]: p["key"] for p in P.probes()},
            "straddle_paras": P.STRADDLE_PARAS}


def check_shas() -> list[str]:
    pinned = json.loads(KEYS.read_text())
    now = expected_keys()
    bad = [k for k in sorted(set(pinned["images"]) | set(now["images"]))
           if pinned["images"].get(k) != now["images"].get(k)]
    bad += [f"key:{k}" for k in sorted(set(pinned["probes"]) | set(now["probes"]))
            if pinned["probes"].get(k) != now["probes"].get(k)]
    return bad


def ask(c: Client, probe: dict) -> dict:
    try:
        r = c.chat(P.messages(probe), max_tokens=64, temperature=0, **OFF)
    except HTTPFailure as exc:
        if NOT_MM in exc.body:
            raise SystemExit(f"VISION FAIL: server says '{NOT_MM}'") from exc
        return {"id": probe["id"], "pass": False, "error": str(exc)[:300]}
    return {"id": probe["id"], "pass": P.judge(probe, r["content"]), "key": probe["key"],
            "content": r["content"].strip()[:120], "finish_reason": r["finish_reason"],
            "prompt_tokens": r["usage"].get("prompt_tokens"), "s": r["s"]}


def straddle_span(c: Client, probe: dict, row: dict) -> dict:
    """Image tokens start after the user header + prefix text (the template adds <|vision_start|>)."""
    start = c.tokenize_count("<|im_start|>user\n" + probe["prefix_text"] + "\n\n") + 1
    n_img = (2048 // P.PX_PER_TOKEN) ** 2
    span = [start, start + n_img]
    return {"text_tokens_before_image": start, "image_span": span,
            "crosses": {str(b): any(span[0] < k * b < span[1] for k in range(1, 4)) for b in CHUNKS},
            "prompt_tokens": row.get("prompt_tokens")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out")
    ap.add_argument("--only", default="", help="comma list of probe-id prefixes")
    ap.add_argument("--no-pair", action="store_true")
    ap.add_argument("--write-keys", action="store_true")
    a = ap.parse_args(argv)
    if a.write_keys:
        write_json(KEYS, expected_keys())
        print(f"wrote {KEYS}")
        return 0
    if not a.out:
        ap.error("--out is required")
    bad = check_shas()
    if bad:
        print(f"V INVALID: probe images or keys differ from keys.json: {bad}", file=sys.stderr)
        return 2
    c = Client(a.url, a.model, timeout=600)
    out = prepare_out(a.out)
    write_manifest(out, c, "vision_probes", vars(a), files=[KEYS])
    rows_path = out / "vision.jsonl"
    rows_path.write_text("")
    pre = tuple(x for x in a.only.split(",") if x)
    probes = [p for p in P.probes() if not pre or p["id"].startswith(pre)]
    rows = []
    for p in probes:
        row = ask(c, p)
        if p["id"].startswith("straddle") and "error" not in row:
            row["straddle"] = straddle_span(c, p, row)
        rows.append(row)
        append_jsonl(rows_path, row)
        print(f"  {'ok  ' if row['pass'] else 'FAIL'} {p['id']:<24} {row.get('content', row.get('error'))!r}",
              flush=True)
    pair = None
    if not a.no_pair:
        by_id = {p["id"]: p for p in P.probes()}
        with cf.ThreadPoolExecutor(max_workers=2) as ex:
            res = list(ex.map(lambda i: ask(c, by_id[i]), PAIR))
        c1 = {r["id"]: r.get("content") for r in rows}
        pair = {"pass": all(r["pass"] for r in res),
                "same_as_c1": {r["id"]: r.get("content") == c1.get(r["id"]) for r in res if r["id"] in c1},
                "rows": res}
        append_jsonl(rows_path, {"id": "pair.c2", **pair})
        print(f"  {'ok  ' if pair['pass'] else 'FAIL'} pair.c2 {[r.get('content') for r in res]}")
    passed = sum(r["pass"] for r in rows)
    ok = passed == len(rows) and (pair is None or pair["pass"])
    summary = {"pass": ok, "probes_passed": passed, "probes": len(rows),
               "failed": [r["id"] for r in rows if not r["pass"]], "pair": None if pair is None else pair["pass"],
               "straddle": next((r.get("straddle") for r in rows if "straddle" in r), None)}
    write_json(out / "vision.json", summary)
    print(f"V {'PASS' if ok else 'FAIL'} {passed}/{len(rows)} probes, pair={summary['pair']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
