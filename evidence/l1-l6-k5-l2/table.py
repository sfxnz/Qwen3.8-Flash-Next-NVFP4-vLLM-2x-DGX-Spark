#!/usr/bin/env python3
"""Session 4 per-arm table from evidence/l1-l6-k5-l2/<arm>/ files (stdlib only).

  table.py ARM [ARM ...] [--base u2-base]    -> markdown rows on stdout, one block per metric group
"""
import argparse
import json
import re
from pathlib import Path

EV = Path(__file__).resolve().parent
RAW = Path.home() / "projects/data/qwen38-evals/runs/session4"
CELLS = [("prose", 1), ("prose", 2), ("prose", 8), ("structured", 1), ("structured", 2), ("structured", 8)]


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return None


def ruler(arm):
    for name in ("gate/ruler.json", "ruler-noanchor.json", "ruler-anchor.json"):
        d = jload(EV / arm / name)
        if d and d.get("summary"):
            s = {(c["phase"], c["concurrency"]): c for c in d["summary"]}
            return name, d, s
    return None, None, {}


def frozen(arm):
    p = EV / arm / "gate/bench-frozen.out"
    if not p.exists():
        return {}
    t = p.read_text()
    starts = [m.start() for m in re.finditer(r'\[\s*\{\s*"phase"', t)]
    if not starts:
        return {}
    try:
        rows, _ = json.JSONDecoder().raw_decode(t[starts[-1]:])
    except ValueError:
        return {}
    return {(r["phase"], r["concurrency"]): r for r in rows}


def diverse(arm):
    p = EV / arm / "diverse.jsonl"
    out = {}
    if p.exists():
        for l in p.read_text().splitlines():
            r = json.loads(l)
            if r.get("kind") == "cell":
                out[(r["category"], r["c"])] = r
    return out


def fmt(x, n=2):
    return "–" if x is None else f"{x:.{n}f}"


def kv_tokens(arm):
    p = EV / arm / "run.log"
    for name in ("needles-rank0.txt",):
        q = EV / arm / name
        if q.exists():
            m = re.search(r"GPU KV cache size: ([\d,]+) tokens", q.read_text())
            if m:
                return m.group(1)
    return None


def free_avail(arm):
    p = EV / arm / "free-postboot.txt"
    if not p.exists():
        return None
    t = p.read_text()
    out = []
    for block in t.split("== ")[1:]:
        m = re.search(r"Mem:\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)", block)
        s = re.search(r"Swap:\s+\S+\s+(\S+)", block)
        if m:
            out.append(f"{m.group(1)}/{s.group(1) if s else '?'}")
    return " · ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--base", default="u2-base")
    a = ap.parse_args()
    print("| arm | ruler | " + " | ".join(f"{p[:5]} c{c} ms (acc)" for p, c in CELLS) + " | sentinel |")
    print("|---" * (len(CELLS) + 3) + "|")
    for arm in a.arms:
        name, d, s = ruler(arm)
        cells = []
        for k in CELLS:
            c = s.get(k)
            if not c:
                cells.append("–")
                continue
            acc = c.get("per_wave_acceptance_len") or [None]
            acc = sorted(x for x in acc if x is not None)
            cells.append(f"{fmt(c['median_ms_per_step'])} ({fmt(acc[len(acc)//2] if acc else None)})")
        sen = (d or {}).get("sentinel") or {}
        st = f"{fmt(sen.get('clean_median_ms'))} {'VOID' if sen.get('void') else ''} {sen.get('excursions')}/{sen.get('runs')}"
        print(f"| {arm} | {name} | " + " | ".join(cells) + f" | {st} |")
    print()
    print("| arm | frozen prose c1 | prose c2 ps/agg | struct c1 | struct c2 ps/agg | acc prose c1/c2 |")
    print("|---|---|---|---|---|---|")
    for arm in a.arms:
        f = frozen(arm)
        g = lambda k, key: f.get(k, {}).get(key)  # noqa: E731
        print(f"| {arm} | {fmt(g(('prose',1),'median_decode_tok_s'),1)} | {fmt(g(('prose',2),'median_decode_tok_s'),1)} / "
              f"{fmt(g(('prose',2),'median_agg_tok_s'),1)} | {fmt(g(('structured',1),'median_decode_tok_s'),1)} | "
              f"{fmt(g(('structured',2),'median_decode_tok_s'),1)} / {fmt(g(('structured',2),'median_agg_tok_s'),1)} | "
              f"{fmt(g(('prose',1),'acceptance_len'))} / {fmt(g(('prose',2),'acceptance_len'))} |")
    print()
    print("| arm | diverse prose c1 ms (acc) | prose c8 | struct c1 | struct c8 |")
    print("|---|---|---|---|---|")
    for arm in a.arms:
        dv = diverse(arm)
        cell = lambda k: (f"{fmt(dv[k]['median_ms_per_step'])} ({fmt(dv[k]['acceptance_len'])})" if k in dv else "–")  # noqa: E731
        print(f"| {arm} | {cell(('prose',1))} | {cell(('prose',8))} | {cell(('structured',1))} | {cell(('structured',8))} |")
    print()
    base_t1 = jload(EV / a.base / "t1-summary.json")
    print("| arm | T0 | smokes | identity | T1-G distinct | T1 mean NLL (Δ vs base) | T2 vs base | KV tokens | free avail/swap (s1 · s2) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm in a.arms:
        e = EV / arm
        gate = (e / "gate/gate.txt").read_text() if (e / "gate/gate.txt").exists() else ""
        smokes = sum(1 for n in ("count", "thinking", "tools", "vision")
                     if (e / f"gate/smoke-{n}.exit").exists() and (e / f"gate/smoke-{n}.exit").read_text().strip() == "0")
        probe = jload(e / "gate/probe.json") or {}
        t0 = (probe.get("totals") or {}).get("gate_flagged_bursts", "–")
        ident = jload(e / "identity.json") or {}
        idt = f"{ident.get('verdict','–')} {ident.get('n',0)-ident.get('mismatches',0)}/{ident.get('n','?')}" if ident else "–"
        t1g = (jload(e / "t1g-self.json") or {}).get("metrics", {}).get("distinct_mean")
        t1 = jload(e / "t1-summary.json")
        nll = "–"
        if t1:
            def mean(s):
                tot = sum(v["tokens"] for v in s["by_domain"].values())
                return sum(v["mean_nll"] * v["tokens"] for v in s["by_domain"].values()) / max(1, tot)
            try:
                m = mean(t1)
                nll = f"{m:.5f}" + (f" ({m - mean(base_t1):+.5f})" if base_t1 else "")
            except Exception:  # noqa: BLE001
                nll = "?"
        t2 = jload(e / "t2-vs-base.json")
        t2s = "–"
        if t2:
            t2s = t2.get("verdict", "?") + " " + ", ".join(
                f"{k} {v['cand_acc']*v['n']:.0f}/{v['n']} ({-v['drop_pp']:+.1f}pp, -{v['lost']}/+{v['gained']})"
                for k, v in sorted((t2.get("tasks") or {}).items()))
        print(f"| {arm} | {t0} | {smokes}/4 | {idt} | {fmt(t1g,1)} | {nll} | {t2s} | {kv_tokens(arm)} | {free_avail(arm)} |")


if __name__ == "__main__":
    main()
