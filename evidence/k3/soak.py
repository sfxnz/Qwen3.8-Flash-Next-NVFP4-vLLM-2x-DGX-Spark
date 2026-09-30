#!/usr/bin/env python3
"""Session 7 (K3) long-generation identity and soak driver.

  soak.py longid --out FILE.jsonl            fixed greedy prompts at c=1, ignore_eos, 4096 tokens, token ids
  soak.py cmp BASE.jsonl ARM.jsonl           per prompt: identical token ids? first differing position; exit 1 on any miss
  soak.py soak --minutes 20 --out F.jsonl --json SUMMARY.json
        8 long workers (ignore_eos, max_tokens 4096, greedy/T=0.7 alternating, distinct prompts) so every stream
        crosses the 1600- and 3200-token GDN block boundaries, 1 short-request worker, and a fresh ~32k-token
        prefill every 4 minutes. Error = HTTP/transport failure, or a long request that returns != max_tokens,
        or an empty short answer. Exit 1 on any error.

The 1600-token boundary is the mamba align block of this recipe (prefix caching on); K3's eager/lazy switch and
the prefix-cache checkpoint copy happen there (docker/v030/K3.md §2).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "quality"))
from qlib import OFF, Client, HTTPFailure  # noqa: E402

URL = "http://127.0.0.1:8000"
FILLER = ("The archive keeps ledgers from the harbour office, each page listing ships, cargo, tonnage and the name "
          "of the clerk on duty. ")
LONGID = [
    ("prose", "Write a long, detailed history of the invention of the printing press and its effects on Europe, "
              "chapter by chapter."),
    ("code", "Write a complete, well-commented Python implementation of a B-tree with insert, delete, search, "
             "range queries and a test suite."),
    # prompt of ~1550 tokens: the first 1600 crossing happens a few decode steps after prefill
    ("near1600", "Here are archive notes:\n" + FILLER * 55 + "\nSummarise the notes above, then continue with a "
                 "long invented chronicle of the harbour."),
]
TOPICS = ["the history of cartography", "a tutorial on writing a compiler in Rust", "the biology of coral reefs",
          "a JSON schema and example records for a library catalogue", "the economics of railways in the 1800s",
          "a SQL tutorial with many example queries", "a travel guide to the Silk Road cities",
          "a C++ implementation of a red-black tree with tests"]


def longid(a) -> int:
    c = Client(a.url, timeout=1800)
    out = Path(a.out)
    out.write_text("")
    for pid, prompt in LONGID:
        r = c.chat(prompt, max_tokens=a.max_tokens, temperature=0, ignore_eos=True, return_token_ids=True, **OFF)
        ids = r["token_ids"] or []
        row = {"id": pid, "n": len(ids), "prompt_tokens": r["usage"].get("prompt_tokens"), "s": r["s"],
               "token_ids": ids}
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{pid}: {len(ids)} tokens, prompt {row['prompt_tokens']}, {r['s']} s", flush=True)
    return 0


def cmp(a) -> int:
    load = lambda p: {r["id"]: r for r in map(json.loads, Path(p).read_text().splitlines())}  # noqa: E731
    b, c = load(a.base), load(a.arm)
    bad = 0
    for pid in b:
        x, y = b[pid]["token_ids"], c.get(pid, {}).get("token_ids") or []
        first = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), None if len(x) == len(y) else min(len(x), len(y)))
        ok = first is None
        bad += not ok
        print(json.dumps({"id": pid, "identical": ok, "n_base": len(x), "n_arm": len(y), "first_diff": first}))
    print(f"IDENTITY {'PASS' if not bad else 'FAIL'} {len(b) - bad}/{len(b)}")
    return 1 if bad else 0


def distinct4(ids) -> float:
    grams = [tuple(ids[i:i + 4]) for i in range(max(0, len(ids) - 3))]
    return round(len(set(grams)) / len(grams), 4) if grams else 0.0


def soak(a) -> int:
    c = Client(a.url, timeout=1800)
    deadline = time.time() + a.minutes * 60
    lock = threading.Lock()
    rows: list[dict] = []
    out = Path(a.out)
    out.write_text("")

    def rec(row):
        with lock:
            rows.append(row)
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")

    def call(kind, prompt, check, **body):
        t0 = time.time()
        try:
            r = c.chat(prompt, return_token_ids=True, **OFF, **body)
            err = check(r)
            ids = r["token_ids"] or []
            rec({"kind": kind, "t": round(t0 - start, 1), "s": r["s"], "prompt_tokens": r["usage"].get("prompt_tokens"),
                 "completion_tokens": r["usage"].get("completion_tokens"), "finish": r["finish_reason"],
                 "distinct4": distinct4(ids), "error": err})
        except (HTTPFailure, OSError, ValueError) as exc:
            rec({"kind": kind, "t": round(t0 - start, 1), "s": round(time.time() - t0, 1), "error": repr(exc)[:300]})

    def long_worker(i):
        n = 0
        while time.time() < deadline:
            temp = 0.0 if (i + n) % 2 == 0 else 0.7
            call("long", f"Write an extremely long and detailed piece about {TOPICS[i]} (part {n + 1}).",
                 lambda r: None if r["usage"].get("completion_tokens") == a.max_tokens else "short long-generation",
                 max_tokens=a.max_tokens, temperature=temp, ignore_eos=True, seed=1000 * i + n)
            n += 1

    def short_worker():
        rng = random.Random(7)
        while time.time() < deadline:
            x, y = rng.randint(10, 999), rng.randint(10, 999)
            call("short", f"What is {x} + {y}? Answer with the number only.",
                 lambda r: None if r["content"].strip() else "empty answer", max_tokens=32, temperature=0)
            time.sleep(2)

    def prefill_worker():
        n = 0
        while time.time() < deadline:
            body = "".join(f"[{n}-{j}] " + FILLER for j in range(a.prefill_lines))
            call("prefill32k", "Notes:\n" + body + f"\nHow many notes carry the prefix [{n}-? Answer briefly.",
                 lambda r: None if r["usage"].get("prompt_tokens", 0) > 30000 else "prefill shorter than 30k",
                 max_tokens=32, temperature=0)
            n += 1
            end = time.time() + 240
            while time.time() < min(end, deadline):
                time.sleep(2)

    start = time.time()
    threads = [threading.Thread(target=long_worker, args=(i,)) for i in range(8)]
    threads += [threading.Thread(target=short_worker), threading.Thread(target=prefill_worker)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    errs = [r for r in rows if r.get("error")]
    summ = {"minutes": a.minutes, "wall_s": round(time.time() - start, 1), "requests": len(rows),
            "by_kind": {k: sum(r["kind"] == k for r in rows) for k in ("long", "short", "prefill32k")},
            "long_tokens": sum(r.get("completion_tokens") or 0 for r in rows if r["kind"] == "long"),
            "long_distinct4_min": min((r["distinct4"] for r in rows if r["kind"] == "long" and "distinct4" in r), default=None),
            "errors": len(errs), "error_rows": errs[:20]}
    Path(a.json).write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    return 1 if errs else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("longid"); p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=4096); p.add_argument("--url", default=URL)
    p = sub.add_parser("cmp"); p.add_argument("base"); p.add_argument("arm")
    p = sub.add_parser("soak"); p.add_argument("--minutes", type=float, default=20)
    p.add_argument("--out", required=True); p.add_argument("--json", required=True)
    p.add_argument("--max-tokens", type=int, default=4096); p.add_argument("--url", default=URL)
    p.add_argument("--prefill-lines", type=int, default=930, help="~35 tokens each: ~32.5k tokens")
    a = ap.parse_args()
    return {"longid": longid, "cmp": cmp, "soak": soak}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
