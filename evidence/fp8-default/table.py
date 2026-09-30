#!/usr/bin/env python3
"""Session 8 (FP8 dense default) tables: ruler cells, frozen bench_decode, diverse per cell (file diverse.jsonl).

  python3 evidence/fp8-default/table.py ARM [ARM ...]

Diverse aggregate tok/s per cell = sum(completion_tokens) / sum over waves of the wave wall time,
where a wave's wall time is max(ttft_s + decode_s) over its requests (every request is sent at t=0).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

EV = Path(__file__).resolve().parent


def rows(p: Path):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def ruler(arm: str, name: str = "ruler-noanchor.json") -> dict:
    """Anchor-free ruler by default (session-8 comparisons); name="gate/ruler.json" or "ruler-anchor.json" for the anchored runs."""
    p = EV / arm / name
    if p.exists():
        j = json.loads(p.read_text())
        return {f"{c['phase']} c{c['concurrency']}": round(c["median_ms_per_step"], 2) for c in j.get("summary") or []}
    return {}


def diverse(arm: str) -> dict:
    waves = defaultdict(list)
    for r in rows(EV / arm / "diverse.jsonl"):
        if r.get("kind") == "request" and r.get("decode_s") is not None:
            waves[(r["category"], r["c"], r["wave"])].append(r)
    cells = defaultdict(lambda: [0, 0.0, 0])
    for (cat, c, _), rs in waves.items():
        cell = cells[(cat, c)]
        cell[0] += sum(r["completion_tokens"] for r in rs)
        cell[1] += max(r["ttft_s"] + r["decode_s"] for r in rs)
        cell[2] += len(rs)
    out = {}
    for r in rows(EV / arm / "diverse.jsonl"):
        if r.get("kind") == "cell":
            tok, wall, n = cells[(r["category"], r["c"])]
            out[f"{r['category']} c{r['c']}"] = {
                "ms_step": r.get("median_ms_per_step"), "acc": r.get("acceptance_len"),
                "ttft": r.get("median_ttft_s"), "agg_tok_s": round(tok / wall, 1) if wall else None, "n": n}
    return out


def frozen(arm: str) -> dict:
    p = EV / arm / "gate" / "bench-frozen.out"
    if not p.exists():
        return {}
    t = p.read_text()
    cells = json.loads(t[t.index("SUMMARY [") + 8:t.rindex("]") + 1])
    return {f"{c['phase']} c{c['concurrency']}": f"{c['median_decode_tok_s']:.1f} / {c['median_agg_tok_s']:.1f}" for c in cells}


def mixed(arm: str) -> list:
    return [r for r in rows(EV / arm / "mixed.jsonl") if r.get("kind") == "cell"]


def main() -> int:
    for arm in sys.argv[1:]:
        print(f"## {arm}")
        print("ruler (anchor-free):", json.dumps(ruler(arm)))
        for name in ("gate/ruler.json", "ruler-anchor.json"):
            if (EV / arm / name).exists():
                print(f"ruler ({name}):", json.dumps(ruler(arm, name)) or "void")
        print("frozen (per-stream / agg tok/s):", json.dumps(frozen(arm)))
        for k, v in diverse(arm).items():
            print(f"diverse {k}: {json.dumps(v)}")
        for c in mixed(arm):
            print("mixed", json.dumps({k: v for k, v in c.items() if k not in ("ruler", "ruler_set", "ruler_sha", "ts", "label", "kind")}))
        t4 = EV / arm / "t4.json"
        if t4.exists():
            print("t4", t4.read_text().strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
