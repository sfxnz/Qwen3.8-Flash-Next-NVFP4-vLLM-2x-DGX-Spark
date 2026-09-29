#!/usr/bin/env python3
"""T1-G capture: in-serve greedy agreement (plan 1.3; C2, C3).

40 fixed prompts (data/prompts40.json) x 5 repeats at c=1, greedy, thinking
off, max_tokens 256, return_token_ids. Each row records others_running
(vllm:num_requests_running before the request): run on an idle serve. Repeats go round-robin (all prompts,
then again), so a slow drift in the serve spreads over every prompt.

  python3 quality/collect_t1g.py --out RUN_DIR [--repeats 5] [--limit 3]

Score: `python3 quality/score.py t1g LKG_DIR CAND_DIR --aa AA.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qlib import DATA, DEFAULT_URL, OFF, Client, append_jsonl, prepare_out, running_requests, write_manifest  # noqa: E402

PROMPTS = DATA / "prompts40.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="first N prompts only (smoke runs)")
    a = ap.parse_args(argv)
    prompts = json.loads(PROMPTS.read_text(encoding="utf-8"))
    prompts = prompts[:a.limit] if a.limit else prompts
    c = Client(a.url, a.model, timeout=900)
    out = prepare_out(a.out)
    write_manifest(out, c, "collect_t1g", vars(a), files=[PROMPTS])
    path = out / "t1g.jsonl"
    path.write_text("")
    t0 = time.time()
    for rep in range(a.repeats):
        for p in prompts:
            busy = running_requests(c)
            r = c.chat(p["prompt"], max_tokens=a.max_tokens, temperature=0, return_token_ids=True, **OFF)
            if not r["token_ids"]:
                raise RuntimeError(f"{p['id']}: no token_ids (return_token_ids unsupported?)")
            append_jsonl(path, {"id": p["id"], "repeat": rep, "token_ids": r["token_ids"],
                                "finish_reason": r["finish_reason"], "s": r["s"],
                                "others_running": busy})
        print(f"  t1g repeat {rep + 1}/{a.repeats} done, {time.time() - t0:.0f} s", flush=True)
    print(f"T1G captured {a.repeats} x {len(prompts)} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
