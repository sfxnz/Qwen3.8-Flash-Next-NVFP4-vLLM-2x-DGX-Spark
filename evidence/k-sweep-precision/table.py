#!/usr/bin/env python3
"""Session 5 tables and the L7 keep rule from evidence/k-sweep-precision/<arm>/ (stdlib only).

  table.py ARM [ARM ...] [--base ref]

Speed is compared as per-stream decode tok/s (k changes acceptance, so ms/step alone is not the metric):
  ruler     6 cells (prose/structured x c1/c2/c8), median decode tok/s; spread = the base's per-wave range
  diverse   4 cells (prose/structured x c1/c8), median per-request tok/s; spread = 2 x the base's SE of the median
  sampled   profile A and T1 at c1/c8 per category, median per-request tok/s; spread as diverse
Keep rule (L7): geomean of arm/base over all cells >= +3%, and no cell below base - spread.
"""
import argparse
import json
import math
import re
import statistics
from pathlib import Path

EV = Path(__file__).resolve().parent
CELLS = [("prose", 1), ("prose", 2), ("prose", 8), ("structured", 1), ("structured", 2), ("structured", 8)]


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return None


def jsonl(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def fmt(x, n=2):
    return "–" if x is None else f"{x:.{n}f}"


def ruler_runs(arm):
    out = []
    for name in ("ruler-noanchor.json", "gate/ruler.json", "ruler-anchor.json"):
        d = jload(EV / arm / name)
        if d and d.get("summary"):
            out.append((name, d))
    return out


def ruler_cells(arm):
    """Per cell: pooled per-wave ms/step, tok/s and acceptance over every ruler run of the boot."""
    cells = {}
    for _, d in ruler_runs(arm):
        for c in d["summary"]:
            k = (c["phase"], c["concurrency"])
            e = cells.setdefault(k, {"ms": [], "tok": [], "acc": []})
            e["ms"] += c["per_wave_ms_per_step"]
            e["tok"] += c["per_wave_median_decode_tok_s"]
            e["acc"] += [a for a in c["per_wave_acceptance_len"] if a is not None]
    return cells


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


def req_cells(arm, fname, key):
    out = {}
    for r in jsonl(EV / arm / fname):
        if r.get("kind") == "request" and not r.get("error") and r.get("decode_tok_s"):
            out.setdefault(key(r), []).append(r)
    return out


def med_se(xs):
    m = statistics.median(xs)
    se = 1.2533 * statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return m, se


def speed_cells(arm):
    """{(ruler, cell): (tok/s, spread_abs, ms/step, acc)}"""
    out = {}
    for k, e in ruler_cells(arm).items():
        if e["tok"]:
            out[("ruler",) + k] = (statistics.median(e["tok"]), (max(e["tok"]) - min(e["tok"])) / 2,
                                   statistics.median(e["ms"]), statistics.median(e["acc"]) if e["acc"] else None)
    for k, rows in req_cells(arm, "diverse.jsonl", lambda r: (r["category"], r["c"])).items():
        m, se = med_se([r["decode_tok_s"] for r in rows])
        out[("diverse",) + k] = (m, 2 * se, statistics.median(r["ms_per_step"] for r in rows),
                                 statistics.median(r["tokens_per_step"] for r in rows))
    for k, rows in req_cells(arm, "sampled.jsonl", lambda r: (r["profile"], r["category"], r["c"])).items():
        m, se = med_se([r["decode_tok_s"] for r in rows])
        out[("sampled",) + k] = (m, 2 * se, statistics.median(r["ms_per_step"] for r in rows),
                                 statistics.median(r["tokens_per_step"] for r in rows))
    return out


def keep_rule(base, arm, need=0.03):
    b, a = speed_cells(base), speed_cells(arm)
    rows, logs, worse = [], [], []
    for k in sorted(set(b) & set(a)):
        r = a[k][0] / b[k][0]
        logs.append(math.log(r))
        bad = a[k][0] < b[k][0] - b[k][1]
        if bad:
            worse.append("/".join(map(str, k)))
        rows.append((k, b[k], a[k], r, bad))
    gm = math.exp(statistics.fmean(logs)) if logs else None
    return rows, gm, worse, bool(gm and gm >= 1 + need and not worse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--base", default="ref")
    a = ap.parse_args()
    arms = [a.base] + [x for x in a.arms if x != a.base]

    print("### Ruler (ms/step · acceptance · per-stream tok/s; pooled over every anchor-free ruler run of the boot)\n")
    print("| arm | " + " | ".join(f"{p[:6]} c{c}" for p, c in CELLS) + " |")
    print("|---" * (len(CELLS) + 1) + "|")
    for arm in arms:
        rc = ruler_cells(arm)
        cells = []
        for k in CELLS:
            e = rc.get(k)
            cells.append("–" if not e or not e["ms"] else
                         f"{statistics.median(e['ms']):.2f} · {fmt(statistics.median(e['acc']) if e['acc'] else None)} · "
                         f"{statistics.median(e['tok']):.1f}")
        print(f"| {arm} | " + " | ".join(cells) + " |")

    print("\n### Frozen bench_decode (tok/s; acceptance)\n")
    print("| arm | prose c1 | prose c2 ps/agg | struct c1 | struct c2 ps/agg | acc prose c1/c2 | TTFT p50 c1 |")
    print("|---|---|---|---|---|---|---|")
    for arm in arms:
        f = frozen(arm)
        g = lambda k, key: f.get(k, {}).get(key)  # noqa: E731
        print(f"| {arm} | {fmt(g(('prose',1),'median_decode_tok_s'),1)} | {fmt(g(('prose',2),'median_decode_tok_s'),1)} / "
              f"{fmt(g(('prose',2),'median_agg_tok_s'),1)} | {fmt(g(('structured',1),'median_decode_tok_s'),1)} | "
              f"{fmt(g(('structured',2),'median_decode_tok_s'),1)} / {fmt(g(('structured',2),'median_agg_tok_s'),1)} | "
              f"{fmt(g(('prose',1),'acceptance_len'))} / {fmt(g(('prose',2),'acceptance_len'))} | "
              f"{fmt(g(('prose',1),'ttft_p50_s'))} |")

    print("\n### Diverse and sampled (median ms/step · tokens/step · per-stream tok/s)\n")
    keys = sorted({k for arm in arms for k in speed_cells(arm) if k[0] != "ruler"})
    print("| arm | " + " | ".join("/".join(map(str, k)) for k in keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for arm in arms:
        s = speed_cells(arm)
        print(f"| {arm} | " + " | ".join(
            (f"{s[k][2]:.2f} · {fmt(s[k][3])} · {s[k][0]:.1f}" if k in s else "–") for k in keys) + " |")

    print("\n### Keep rule vs base (geomean of per-stream tok/s ratios; ≥ +3% and no cell below base − spread)\n")
    print("| arm | cells | geomean | worst cell | cells below spread | verdict |")
    print("|---|---|---|---|---|---|")
    for arm in arms[1:]:
        rows, gm, worse, ok = keep_rule(a.base, arm)
        if not rows:
            continue
        wk = min(rows, key=lambda r: r[3])
        print(f"| {arm} | {len(rows)} | {(gm - 1) * 100:+.2f}% | {'/'.join(map(str, wk[0]))} {(wk[3] - 1) * 100:+.1f}% | "
              f"{', '.join(worse) or 'none'} | {'KEEP' if ok else 'no'} |")
    for arm in arms[1:]:
        rows, gm, worse, ok = keep_rule(a.base, arm)
        if not rows:
            continue
        print(f"\n<details><summary>{arm} per cell</summary>\n\n| cell | base tok/s ± spread | arm tok/s | ratio |\n|---|---|---|---|")
        for k, b, x, r, bad in rows:
            print(f"| {'/'.join(map(str, k))} | {b[0]:.1f} ± {b[1]:.1f} | {x[0]:.1f} | {(r - 1) * 100:+.1f}%{' **below**' if bad else ''} |")
        print("\n</details>")


if __name__ == "__main__":
    main()
