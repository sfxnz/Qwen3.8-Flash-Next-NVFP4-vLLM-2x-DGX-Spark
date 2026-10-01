#!/usr/bin/env python3
"""T4 (plan 1.3, C24): full GSM8K test (1,319 items), thinking on, T=0.6, fixed seeds.

Sampling: temperature 0.6, top_p 0.95, top_k 20, seed = --seed + item index, so
two configs sample item by item from the same seed. Answers are read from the
final content (the reasoning parser strips <think>). Resumable: rerun the same
command and finished items are skipped, errored ones retried.

  python3 quality/t4_gsm8k_full.py --out RUN_DIR [--workers 1] [--limit 3] [--max-tokens 16384]

--workers >1 only on a pin with F01 fixed (co-prefilled requests corrupt; see t2.py).
Compare on the current golden base: `python3 quality/score.py t4 BASE_DIR CAND_DIR`
(paired McNemar not significant and drop <= 1.5 pp). RadixArk's 0.9727 is a
sanity check only.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qlib import DATA, DEFAULT_URL, ON, Client, HTTPFailure, append_jsonl, jsonl, pmap, prepare_out, write_json, write_manifest  # noqa: E402
from t2 import GSM8K_SUFFIX, dataset, gsm8k_correct, gsm8k_extract, gsm8k_gold  # noqa: E402

IDS = DATA / "t2_ids.json"
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent requests; >1 only on a pin with F01 fixed")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    rows = dataset("gsm8k", "test.jsonl", json.loads(IDS.read_text())["gsm8k"]["sha256"])
    items = list(range(len(rows)))[:a.limit or None]
    c = Client(a.url, a.model, timeout=7200)
    out = prepare_out(a.out)
    path = out / "t4.jsonl"
    done = {r["id"] for r in jsonl(path) if "error" not in r} if path.exists() else set()
    if path.exists():  # keep finished rows, drop errored ones so they are retried
        path.write_text("".join(json.dumps(r) + "\n" for r in jsonl(path) if r["id"] in done))
    write_manifest(out, c, "t4_gsm8k_full", vars(a), files=[IDS], extra={"sampling": SAMPLING})
    todo = [i for i in items if i not in done]
    lock, t0, count = threading.Lock(), time.time(), [0]

    def one(i: int) -> None:
        gold = gsm8k_gold(rows[i]["answer"])
        try:
            r = c.chat(rows[i]["question"] + GSM8K_SUFFIX, max_tokens=a.max_tokens, seed=a.seed + i, **SAMPLING, **ON)
            row = {"task": "gsm8k_full", "id": i, "seed": a.seed + i, "correct": int(gsm8k_correct(r["content"], gold)),
                   "got": gsm8k_extract(r["content"]), "gold": gold, "finish_reason": r["finish_reason"],
                   "completion_tokens": r["usage"].get("completion_tokens"), "reasoning_chars": len(r["reasoning"])}
        except (HTTPFailure, OSError) as exc:  # timeout / reset: recorded, retried on rerun
            row = {"task": "gsm8k_full", "id": i, "seed": a.seed + i, "correct": 0, "error": str(exc)[:300]}
        with lock:
            append_jsonl(path, row)
            count[0] += 1
            if count[0] % 50 == 0:
                print(f"  t4 {count[0]}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)

    pmap(one, todo, a.workers)
    res = [r for r in jsonl(path) if r["id"] in set(items)]
    ok = sum(r["correct"] for r in res)
    summary = {"n": len(res), "correct": ok, "acc": round(ok / max(1, len(res)), 4),
               "errors": sum("error" in r for r in res), "length_finishes": sum(r.get("finish_reason") == "length" for r in res),
               "complete": len(res) == len(items)}
    write_json(out / "t4.json", summary)
    print(f"T4 {json.dumps(summary)}")
    return 0 if summary["complete"] and not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
