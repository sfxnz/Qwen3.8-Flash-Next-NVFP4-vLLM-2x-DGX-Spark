#!/usr/bin/env python3
"""T1-D capture: decode-path greedy generations with logprobs (plan 1.3).

40 fixed prompts (data/prompts40.json) x 1024 greedy tokens, thinking off,
spec decoding as served, logprobs with top_logprobs=20 and
return_tokens_as_token_ids. --workers 8 runs the prompts 8 at a time: that is
the golden-vs-golden c=8 floor capture, meaningful only on a pin with F01
fixed (co-prefilled requests corrupt on the pinned image). The serve needs --max-logprobs >= 20
(vLLM default 20).

Output (--out DIR): manifest.json, index.jsonl (id, off, n, finish_reason),
t1d.npz (ids, lp, top_ids [n,20], top_lp [n,20]; prompts concatenated).

  python3 quality/collect_t1d.py --out RUN_DIR [--workers 1] [--limit 3] [--max-tokens 1024]

Score: `python3 quality/score.py t1d GOLDEN_DIR CAND_DIR --floor FLOOR.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qlib import DATA, DEFAULT_URL, OFF, Client, pmap, prepare_out, write_manifest  # noqa: E402

PROMPTS = DATA / "prompts40.json"
TOPK = 20


def generate(c: Client, prompt: dict, max_tokens: int, k: int) -> dict:
    r = c.chat(prompt["prompt"], max_tokens=max_tokens, temperature=0, logprobs=True, top_logprobs=k,
               return_tokens_as_token_ids=True, **OFF)
    rows = r["logprobs"] or []
    if not rows:
        raise RuntimeError(f"{prompt['id']}: no logprobs returned")
    top_ids = np.full((len(rows), k), -1, dtype=np.int32)
    top_lp = np.full((len(rows), k), -np.inf, dtype=np.float32)
    for j, row in enumerate(rows):
        top = row["top"][:k]
        top_ids[j, :len(top)] = [t[0] for t in top]
        top_lp[j, :len(top)] = [t[1] for t in top]
    return {"id": prompt["id"], "finish_reason": r["finish_reason"], "s": r["s"],
            "ids": np.asarray([x["id"] for x in rows], dtype=np.int32),
            "lp": np.asarray([x["lp"] for x in rows], dtype=np.float32), "top_ids": top_ids, "top_lp": top_lp}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--workers", type=int, default=1, help="concurrent prompts; 8 = the c=8 floor")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    prompts = json.loads(PROMPTS.read_text(encoding="utf-8"))
    prompts = prompts[:a.limit] if a.limit else prompts
    c = Client(a.url, a.model, timeout=1800)
    out = prepare_out(a.out)
    write_manifest(out, c, "collect_t1d", vars(a), files=[PROMPTS])
    t0 = time.time()
    res = pmap(lambda p: generate(c, p, a.max_tokens, a.topk), prompts, a.workers)
    off = 0
    with (out / "index.jsonl").open("w") as f:
        for r in res:
            f.write(json.dumps({"id": r["id"], "off": off, "n": len(r["ids"]), "finish_reason": r["finish_reason"],
                                "s": r["s"]}) + "\n")
            off += len(r["ids"])
    np.savez_compressed(out / "t1d.npz", **{k: np.concatenate([r[k] for r in res])
                                             for k in ("ids", "lp", "top_ids", "top_lp")})
    print(f"T1D captured {len(res)} prompts, {off} tokens, workers={a.workers}, {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
