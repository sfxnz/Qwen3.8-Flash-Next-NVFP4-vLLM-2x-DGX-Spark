#!/usr/bin/env python3
"""Per-prompt-kind acceptance (tokens/step) and tok/s at c=1 from bench_diverse request rows.
  domain_acc.py ARM [ARM ...]   (reads evidence/l1-l6-k5-l2/ARM/diverse.jsonl)"""
import json, statistics, sys
from collections import defaultdict
from pathlib import Path
EV = Path(__file__).resolve().parent
arms = sys.argv[1:]
data = {}
for a in arms:
    d = defaultdict(list)
    for l in (EV / a / "diverse.jsonl").read_text().splitlines():
        r = json.loads(l)
        if r.get("kind") == "request" and r["c"] == 1 and not r.get("error"):
            d[r["prompt_kind"]].append((r["tokens_per_step"], r["decode_tok_s"], r["text_sha"]))
    data[a] = d
kinds = sorted(set().union(*[set(d) for d in data.values()]))
print("| kind | n | " + " | ".join(f"{a} tok/step · tok/s" for a in arms) + " | text identical vs first |")
print("|---" * (len(arms) + 3) + "|")
for k in kinds:
    cells, n = [], len(data[arms[0]].get(k, []))
    for a in arms:
        v = data[a].get(k, [])
        cells.append(f"{statistics.mean(x[0] for x in v):.3f} · {statistics.mean(x[1] for x in v):.1f}" if v else "–")
    same = [sum(x[2] == y[2] for x, y in zip(data[arms[0]].get(k, []), data[a].get(k, []))) for a in arms[1:]]
    print(f"| {k} | {n} | " + " | ".join(cells) + f" | {'/'.join(map(str, same))} of {n} |")
