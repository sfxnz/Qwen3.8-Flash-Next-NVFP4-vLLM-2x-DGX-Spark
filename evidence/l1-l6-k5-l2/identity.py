#!/usr/bin/env python3
"""Greedy identity gate (session 4): per prompt, is the arm's T1-G output identical to the baseline's?

  identity.py BASE_T1G_DIR ARM_T1G_DIR [--json OUT]

Reads t1g.jsonl from both collect_t1g.py runs. For each prompt id, hashes the token ids of every
repeat (sha256 of the comma-joined ids). Baseline and arm must each be self-consistent (1 distinct
output per prompt on the determinism base); a prompt matches when the arm's hash set equals the
baseline's. Reports the first differing token position of repeat 0 on a mismatch.
Exit 0 all identical, 1 any mismatch.
"""
import argparse
import hashlib
import json
from pathlib import Path


def load(d):
    rows = {}
    for line in (Path(d) / "t1g.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows.setdefault(r["id"], []).append(r)
    return rows


def sha(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:16]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("arm")
    p.add_argument("--json")
    a = p.parse_args()
    base, arm = load(a.base), load(a.arm)
    out = {"base": a.base, "arm": a.arm, "prompts": [], "mismatches": 0}
    for pid in sorted(base):
        b = sorted(base[pid], key=lambda r: r["repeat"])
        c = sorted(arm.get(pid, []), key=lambda r: r["repeat"])
        hb, hc = {sha(r["token_ids"]) for r in b}, {sha(r["token_ids"]) for r in c}
        row = {"id": pid, "base_sha": sorted(hb), "arm_sha": sorted(hc), "base_distinct": len(hb),
               "arm_distinct": len(hc), "identical": bool(c) and hb == hc}
        if not row["identical"] and c:
            x, y = b[0]["token_ids"], c[0]["token_ids"]
            n = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
            row.update(first_diff=n, base_len=len(x), arm_len=len(y))
        out["prompts"].append(row)
        out["mismatches"] += not row["identical"]
    out["n"] = len(out["prompts"])
    out["verdict"] = "IDENTICAL" if out["mismatches"] == 0 else "MISMATCH"
    txt = json.dumps(out, indent=1)
    print(txt)
    if a.json:
        Path(a.json).write_text(txt + "\n")
    raise SystemExit(0 if out["mismatches"] == 0 else 1)


if __name__ == "__main__":
    main()
