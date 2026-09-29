#!/usr/bin/env python3
"""S1.1 go/no-go table from the microbench outputs (plan §3 S1.1, stdlib only).

Reads one run_s11.sh output dir (OUT/<image-tag>/...) and applies the S1.1
decision rules per image. Every row carries the measured value, the rule and a
verdict (GO / NO-GO / MARGINAL / INFO / MISSING). Writes gonogo.md and
gonogo.json into the output dir.

  python3 tools/micro/gonogo.py OUT
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nccl_decode_sweep as nds  # noqa: E402

# (name in gemm_sweep.SHAPES, label) for the cuBLAS GB/s row.
GEMM_RULE_SHAPES = (
    ("gdn_in_proj_qkvz_ba", "(8240,2560)"),
    ("gdn_out_proj|qsa_o", "(2560,3072)"),
    ("qsa_qkv+index", "(7296,2560)"),
    ("gdn_in_proj_ba", "(48,2560)"),
    ("router", "(512,2560)"),
)
RULE_M = 4  # c=1 verify rows at k=3


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def row(measurement: str, value: str, rule: str, verdict: str, note: str = "") -> dict:
    return {"measurement": measurement, "value": value, "rule": rule, "verdict": verdict, "note": note}


def missing(measurement: str, rule: str, why: str = "no data") -> dict:
    return row(measurement, "-", rule, "MISSING", why)


def _cell(summary: dict, arm: str, op: str, rows: int, mode: str):
    for c in summary.get("arms", {}).get(arm, {}).get("cells", []):
        if (c["op"], c["rows"], c["mode"]) == (op, rows, mode):
            return c
    return None


def rules_collectives(summary: dict | None) -> list[dict]:
    m1 = "In-graph AR 5-160 KB, mixing on vs off (L2a booking)"
    r1 = "book 0.5 x (107 x dAR + neighbour slowdown); build L2a if >= 0.75 ms"
    m2 = "AG vs AR at c=1 payload, mixing off (L2d)"
    r2 = "build L2d only if AG_off <= 0.7 x AR_off"
    m3 = "L2 prefetch of a 13.4 MB module during a real AR"
    r3 = ">= 15 us net per window -> revisit F16; otherwise 0"
    if not summary or not {"keep", "mix0"} <= set(summary.get("arms", {})):
        return [missing(m1, r1), missing(m2, r2), missing(m3, r3)]
    out = []
    deltas = {}
    for m in nds.ROWS:
        a, b = _cell(summary, "keep", "all_reduce", m, "gapped"), _cell(summary, "mix0", "all_reduce", m, "gapped")
        if a and b:
            deltas[m] = a["median_us"] - b["median_us"]
    booked = {}
    for w, wt in nds.STEP_WEIGHTS.items():
        if all(k[1] in deltas for k in wt):
            booked[w] = 0.5 * sum(n * deltas[k[1]] for k, n in wt.items()) / 1e3
    if "c1" in booked:
        v = f"dAR(M) us {', '.join(f'{m}:{d:.1f}' for m, d in deltas.items())}; booked c1 {booked['c1']:.2f} ms" + (
            f", c8 {booked['c8']:.2f} ms" if "c8" in booked else "")
        out.append(row(m1, v, r1, "GO" if booked["c1"] >= 0.75 else "NO-GO",
                       "gapped mode already contains the slowed neighbour"))
    else:
        out.append(missing(m1, r1, "gapped AR cells incomplete"))
    ag, ar = (_cell(summary, "mix0", "all_gather", RULE_M, "gapped"),
              _cell(summary, "mix0", "all_reduce", RULE_M, "gapped"))
    if ag and ar and ar["median_us"] > 0:
        ratio = ag["median_us"] / ar["median_us"]
        out.append(row(m2, f"AG {ag['median_us']:.1f} / AR {ar['median_us']:.1f} us = {ratio:.2f} at M={RULE_M}",
                       r2, "GO" if ratio <= 0.7 else "NO-GO"))
    else:
        out.append(missing(m2, r2))
    pf = _cell(summary, "keep", "all_reduce", RULE_M, "prefetch")
    if pf and "saving_us" in pf:
        out.append(row(m3, f"{pf['saving_us']:.1f} us/window at M={RULE_M}", r3,
                       "GO" if pf["saving_us"] >= 15 else "NO-GO"))
    else:
        out.append(missing(m3, r3))
    lm = [c for c in (_cell(summary, "keep", "all_gather_lm", m, "gapped") for m in nds.ROWS) if c]
    if lm:
        out.append(row("Logits all-gather 248 KB-8 MB (keep, gapped)",
                       ", ".join(f"{c['bytes'] // 1024} KB:{c['median_us']:.0f} us" for c in lm),
                       "input to F13 / L1 comm model", "INFO"))
    return out


def _gemm(gemm: dict | None, name: str, m: int):
    for r in (gemm or {}).get("rows", []):
        if r.get("name") == name and r.get("m") == m:
            return r
    return None


def _knames(r: dict) -> str:
    ks = [k["name"] for k in r.get("kernels", []) if k.get("us") is not None]
    return " + ".join(k[:60] for k in ks) or "kernels n/a"


def rules_hc_cublas(gemm: dict | None) -> list[dict]:
    meas = "cuBLAS choice for HC down (336,10240) and HC up at M=4"
    rule = "WMMA tile (>= 60 us per call) -> L4a first; split-K (~37 us) -> K5 directly, built W8A16-ready"
    d, u = _gemm(gemm, "hc_down_inject", RULE_M), _gemm(gemm, "hc_up", RULE_M)
    if not d:
        return [missing(meas, rule)]
    us = d["flush"]["median_us"]
    split = any("splitk" in k["name"].lower() or "reduce" in k["name"].lower() for k in d.get("kernels", []))
    if us >= 60:
        verdict, note = "GO", "WMMA case: L4a (algo pin / sm_121 skinny plan) first"
    elif us <= 45:
        verdict, note = "GO", "split-K case: skip L4a, build K5 directly (W8A16-ready)"
    else:
        verdict, note = "MARGINAL", "between the two cases; decide from the kernel names"
    v = f"down {us:.1f} us [{_knames(d)}]" + (f"; up {u['flush']['median_us']:.1f} us [{_knames(u)}]" if u else "")
    return [row(meas, v + (" (split-K kernel seen)" if split else ""), rule, verdict, note)]


def rules_gemm_gbps(gemm: dict | None) -> list[dict]:
    rule = "< 195 GB/s -> K4 BF16 pays; >= 215 -> build K4 only as the FP8 substrate"
    out = []
    for name, label in GEMM_RULE_SHAPES:
        meas = f"cuBLAS GB/s at {label} ({name})"
        r = _gemm(gemm, name, RULE_M)
        if not r:
            out.append(missing(meas, rule))
            continue
        g = r["flush"]["gbps"] or 0
        others = ", ".join(f"M={m}:{x['flush']['gbps']}" for m in (1, 32) if (x := _gemm(gemm, name, m)))
        verdict = "GO" if g < 195 else ("NO-GO" if g >= 215 else "MARGINAL")
        out.append(row(meas, f"{g} GB/s at M={RULE_M} cold ({others})", rule, verdict,
                       "GO = a K4 BF16 plan pays for this shape"))
    return out


def rules_fp8(fp8: dict | None) -> list[dict]:
    rule = ">= 180 GB/s -> stock kernels; < 150 -> K4 FP8 is a prerequisite for P4-1"
    out = []
    rows = (fp8 or {}).get("rows", [])
    for arm in ("w8a16_marlin_ch", "w8a16_marlin_b128", "w8a8_cutlass"):
        meas = f"{arm} GB/s at the P4-1 shapes"
        rs = [r for r in rows if r["arm"] == arm and r["m"] == RULE_M]
        ok = [r for r in rs if "flush" in r]
        if not ok:
            why = rs[0]["skipped"][:120] if rs and "skipped" in rs[0] else "no data"
            out.append(missing(meas, rule, why))
            continue
        worst = min(ok, key=lambda r: r["flush"]["gbps"] or 0)
        g = worst["flush"]["gbps"] or 0
        verdict = "GO" if g >= 180 else ("NO-GO" if g < 150 else "MARGINAL")
        per = ", ".join(f"{r['name']}:{r['flush']['gbps']}" for r in ok)
        out.append(row(meas, f"worst {g} GB/s ({worst['name']}) at M={RULE_M}; {per}", rule, verdict,
                       "GO = stock kernel; NO-GO = K4 FP8 first"))
    w8a8 = [r for r in rows if r["arm"] == "w8a8_cutlass" and r["m"] == RULE_M and "flush" in r]
    if w8a8:
        g = min(r["flush"]["gbps"] or 0 for r in w8a8)
        out.append(row("D8 arm order (W8A8 >= 200 GB/s?)", f"W8A8 worst {g} GB/s",
                       "W8A16 first unless prefill TTFT is the priority and W8A8 >= 200",
                       "INFO", "W8A8 qualifies" if g >= 200 else "W8A16 arm first"))
    return out


def _moe(moe: dict | None, config: str, m: int, routing: str = "uniform", inp: str = "fp4_in"):
    for r in (moe or {}).get("rows", []):
        if (r.get("config"), r.get("m"), r.get("routing"), r.get("input")) == (config, m, routing, inp):
            return r
    return None


def rules_moe(moe: dict | None) -> list[dict]:
    m1, r1 = "NVFP4 MoE TP2 routed vs floor", "TP2 routed <= 1.2x floor -> close F10"
    m2, r2 = "NVFP4 MoE TP2 FC2 vs floor", "TP2 FC2 >= 1.6x floor -> run the D2 `ep` A/B"
    tp = _moe(moe, "tp2", RULE_M)
    if not tp or "flush" not in tp:
        why = (tp or {}).get("skipped") or (moe or {}).get("skipped") or "harness not ready: S1.2 profile supplies TP2"
        return [missing(m1, r1, why[:160]), missing(m2, r2, why[:160])]
    ep = _moe(moe, "ep", RULE_M)
    epv = f"; EP {ep['x_floor']}x" if ep and ep.get("x_floor") else ""
    x = tp["x_floor"]
    out = [row(m1, f"{tp['flush']['median_us']:.1f} us vs floor {tp['floor_us']:.1f} us = {x}x "
               f"({tp['distinct_local_experts']} experts, M={RULE_M}, autotuned={tp.get('autotuned')}){epv}",
               r1, "GO" if x <= 1.2 else "NO-GO", "GO = close F10")]
    f2 = tp.get("fc2_x_floor")
    if f2 is None:
        out.append(missing(m2, r2, "FC2 kernel not identified in the profile"))
    else:
        out.append(row(m2, f"FC2 {tp['fc2_us']} us vs {tp['fc2_floor_us']} us = {f2}x", r2,
                       "GO" if f2 >= 1.6 else "NO-GO", "GO = run D2 ep A/B"))
    bf = _moe(moe, "tp2", RULE_M, inp="bf16_in")
    if bf and "flush" in bf:
        out.append(row("L6b BF16 activations into FlashInfer", f"{bf['flush']['median_us']:.1f} us vs "
                       f"fp4_in {tp['flush']['median_us']:.1f} us", "informs L6b (-0.1 ms expected)", "INFO"))
    return out


def rules_drafthead(dh: dict | None) -> list[dict]:
    meas, rule = "Draft-head probe F.linear [V'/2,2560] + argmax", ">= 200 GB/s -> L1b is on track"
    rs = [r for r in (dh or {}).get("rows", []) if r["m"] == 1]
    if not rs:
        return [missing(meas, rule)]
    best = [r for r in rs if r["v_prime"] == 32768] or rs
    g = best[0]["flush"]["gbps"] or 0
    v = ", ".join(f"V'={r['v_prime']}:{r['flush']['gbps']} GB/s {r['flush']['median_us']:.0f} us" for r in rs)
    return [row(meas, f"M=1 cold: {v}", rule, "GO" if g >= 200 else "NO-GO", "rule applied at V'=32k")]


def rules_prefill(single: dict | None, dual: dict | None) -> list[dict]:
    import nccl_allreduce_sweep as nas

    meas, rule = "42 MB all-reduce, single vs dual rail", "feeds longctx only; records the decode small-message penalty"
    if not single or not dual:
        return [missing(meas, rule, "dual arm failed or absent (check spark2 second HCA)" if single else "no data")]
    c = nas.compare(single, dual)
    if "prefill_speedup" not in c:
        return [missing(meas, rule)]
    w = c["worst_small_lat_change"]
    return [row(meas, f"single {c['prefill_ar_us']['base']:.0f} us, dual {c['prefill_ar_us']['cand']:.0f} us "
                f"({c['prefill_speedup']}x); worst small-msg latency {100 * w:+.1f}%", rule, "INFO")]


def rules_device(devs: list[dict]) -> list[dict]:
    if not devs:
        return [missing("deviceQuery facts", "48 SMs, 24 MB L2, ~99 KB opt-in smem")]
    out = []
    for d in devs:
        chk = d.get("check")
        if not chk:
            out.append(missing(f"deviceQuery facts ({d.get('meta', {}).get('host')})", "48 SMs, 24 MB L2, ~99 KB smem",
                               d.get("driver_error", "no data")))
            continue
        v = ", ".join(f"{k}={g}" for k, (_w, g, _ok) in chk.items())
        out.append(row(f"deviceQuery facts ({d['meta'].get('host')})", v, "48 SMs, 24 MB L2, ~99 KB opt-in smem",
                       "INFO" if all(ok for *_x, ok in chk.values()) else "MISMATCH",
                       "kernel track (§4) assumptions"))
    return out


def build(img_dir: Path) -> list[dict]:
    g = lambda n: _load(img_dir / n)  # noqa: E731
    summary = g("nccl-decode.json")
    if summary is None and (img_dir / "nccl-decode").is_dir():
        summary = nds.summarize(img_dir / "nccl-decode")
    rows = []
    rows += rules_collectives(summary)
    rows += rules_hc_cublas(g("gemm.json"))
    rows += rules_gemm_gbps(g("gemm.json"))
    rows += rules_fp8(g("fp8.json"))
    rows += rules_moe(g("moe.json"))
    rows += rules_drafthead(g("drafthead.json"))
    rows += rules_prefill(g("allreduce/single.json"), g("allreduce/dual.json"))
    # head writes device.json, a worker run device.<host>.json
    rows += rules_device([d for p in sorted(img_dir.glob("device*.json")) if (d := _load(p))])
    hc = g("hc.json")
    if hc:
        r4 = next((r for r in hc["rows"] if r.get("m") == RULE_M and "stream" in r), None)
        if r4:
            rows.append(row("HC module chain baseline (L4 gate: <= 0.6x this)",
                            f"M=4 stream {r4['stream']['median_us']:.1f} us ({r4['stream']['x_floor']}x floor), "
                            f"hot {r4['hot']['median_us']:.1f} us, flush {r4['flush']['median_us']:.1f} us",
                            "baseline for L4 / K5", "INFO"))
    return rows


def render(tables: dict[str, list[dict]]) -> str:
    lines = ["# S1.1 go/no-go", ""]
    for img, rows in tables.items():
        lines += [f"## {img}", "", "| Measurement | Value | Decision rule | Verdict | Note |", "|---|---|---|---|---|"]
        for r in rows:
            cells = [r[k].replace("|", "/") for k in ("measurement", "value", "rule", "verdict", "note")]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    out = Path(argv[0])
    imgs = sorted(p for p in out.iterdir() if p.is_dir()) if out.is_dir() else []
    tables = {p.name: build(p) for p in imgs}
    (out / "gonogo.json").write_text(json.dumps(tables, indent=1) + "\n")
    (out / "gonogo.md").write_text(render(tables))
    print(render(tables))
    return 0


if __name__ == "__main__":
    sys.exit(main())
