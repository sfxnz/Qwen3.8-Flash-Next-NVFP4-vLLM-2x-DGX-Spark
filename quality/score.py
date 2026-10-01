#!/usr/bin/env python3
"""Quality-gate scorer (plan 1.3): stdlib + numpy, no server access.

  score.py t1   REF_DIR CAND_DIR [--class A|B] [--aa AA_T1.json]   T1-A / T1-B
  score.py t1g  LKG_DIR CAND_DIR [--aa AA_T1G.json]                T1-G
  score.py t1d  GOLD_DIR CAND_DIR [--class A|B] [--floor FLOOR.json]  T1-D
  score.py t2   REF_DIR CAND_DIR [--aa AA_T2.json]                 T2 paired flips
  score.py t4   REF_DIR CAND_DIR                                   T4 paired flips
  score.py threshold --kind lower|agree --stated X --floor F       the C25 rule

Every subcommand prints one JSON object ({"verdict", "pass", "metrics", ...})
and writes it to --json when given. Exit 0 PASS or report-only, 1 FAIL,
2 INVALID (runs not comparable).

Floors (C25). An A/A comparison is the same subcommand run on two captures of
one config. Pass its JSON as --aa (t1, t1g, t2) or --floor (t1d):
  lower-is-better metrics (KL, flip rate): threshold = max(stated, 2 x floor)
  agreement metrics (top-1): threshold = min(stated, 1 - 2 x (1 - floor))
  a class-B top-1 threshold below 97% is not relaxed: verdict ESCALATE.
T1-G gates at A/A mean + 2 sigma, sigma the A/A standard error over prompts.
T1-D gates median first divergence >= 0.5 x the golden-vs-golden (c=8) floor.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qlib import jsonl, mcnemar, threshold_agree, threshold_lower  # noqa: E402

CLASS = {"A": {"top1": 0.995, "kl": 0.005}, "B": {"top1": 0.985, "kl": 0.02}}
ESCALATE_TOP1_B = 0.97
DOMAIN_NOISE = 0.1  # a domain may exceed 2x the mean while below 10% of the class threshold
EARLY = 32
T2_MAX_DROP_PP = 2.0
T4_MAX_DROP_PP = 1.5
P_SIG = 0.05
REP4_TOL = 0.005


# ------------------------------------------------------------------ KL

def kl_topk(p_ids, p_lp, q_ids, q_lp, q_tgt=None, q_tgt_lp=None) -> np.ndarray:
    """Per-position KL(p || q) over p's top-k plus one 'rest' bucket, vectorised.

    A p top-k token missing from q's list takes q's logprob for the target
    token when it is that token, else min(q's k-th prob, q's unreturned mass /
    (missing + 1)). Identical inputs give 0. Rows padded with id -1 are ignored.
    """
    p_ids, q_ids = np.asarray(p_ids), np.asarray(q_ids)
    p_lp, q_lp = np.asarray(p_lp, dtype=np.float64), np.asarray(q_lp, dtype=np.float64)
    pvalid = p_ids >= 0
    qprob = np.where(q_ids >= 0, np.exp(q_lp), 0.0)
    match = (p_ids[:, :, None] == q_ids[:, None, :]) & (q_ids[:, None, :] >= 0)
    has = match.any(2)
    q = (match * qprob[:, None, :]).sum(2)
    if q_tgt is not None:
        is_tgt = (p_ids == np.asarray(q_tgt)[:, None]) & ~has
        q = np.where(is_tgt, np.exp(np.asarray(q_tgt_lp, dtype=np.float64))[:, None], q)
        has = has | is_tgt
    missing = ~has & pvalid
    qmin = np.where(q_ids >= 0, qprob, np.inf).min(1)
    fill = np.minimum(qmin, np.maximum(0.0, 1.0 - qprob.sum(1)) / (missing.sum(1) + 1))
    q = np.where(missing, fill[:, None], q)
    q = np.maximum(q, 1e-12)
    p = np.where(pvalid, np.exp(p_lp), 0.0)
    kl = np.where(pvalid, p * (p_lp - np.log(q)), 0.0).sum(1)
    p_rest = np.maximum(0.0, 1.0 - p.sum(1))
    q_rest = np.maximum(1e-12, 1.0 - np.where(pvalid, q, 0.0).sum(1))
    kl += np.where(p_rest > 1e-12, p_rest * np.log(np.maximum(p_rest, 1e-300) / q_rest), 0.0)
    return np.maximum(kl, 0.0)


# ------------------------------------------------------------------ T1

def load_t1(d: Path) -> dict:
    idx = jsonl(d / "index.jsonl")
    shards, out = {}, {}
    for r in idx:
        s = r["shard"]
        if s not in shards:
            with np.load(d / f"t1-{s:03d}.npz") as z:
                shards[s] = {k: z[k] for k in z.files}
        sl = slice(r["off"], r["off"] + r["n"])
        out[r["id"]] = {"meta": r, **{k: v[sl] for k, v in shards[s].items()}}
    return out


def _t1_metrics(parts: list[dict]) -> dict:
    n = sum(len(p["agree"]) for p in parts)
    if n == 0:
        return {"tokens": 0, "top1_agree": 1.0, "kl": 0.0, "kl_p99": 0.0, "nll_ref": 0.0, "nll_delta": 0.0,
                "flips": {"lost": 0, "gained": 0, "flip_rate": 0.0, "p": 1.0}}
    agree = np.concatenate([p["agree"] for p in parts])
    kl = np.concatenate([p["kl"] for p in parts])
    rc = np.concatenate([p["ref_ok"] for p in parts])
    cc = np.concatenate([p["cand_ok"] for p in parts])
    nll_r = -np.concatenate([p["ref_lp"] for p in parts]).astype(np.float64)
    nll_c = -np.concatenate([p["cand_lp"] for p in parts]).astype(np.float64)
    lost, gained = int((rc & ~cc).sum()), int((cc & ~rc).sum())
    return {"tokens": n, "top1_agree": round(float(agree.mean()), 6), "kl": round(float(kl.mean()), 8),
            "kl_p99": round(float(np.quantile(kl, 0.99)), 6), "nll_ref": round(float(nll_r.mean()), 6),
            "nll_delta": round(float(nll_c.mean() - nll_r.mean()), 6),
            "flips": {"lost": lost, "gained": gained, "flip_rate": round((lost + gained) / max(1, n), 6),
                      "p": mcnemar_counts(lost, gained)}}


def mcnemar_counts(lost: int, gained: int) -> float:
    n, k = lost + gained, min(lost, gained)
    if n == 0:
        return 1.0
    if n > 2000:  # normal approximation with continuity correction
        z = (abs(lost - gained) - 1) / math.sqrt(n)
        return round(min(1.0, math.erfc(max(0.0, z) / math.sqrt(2))), 6)
    return round(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), 6)


def score_t1(ref_dir: Path, cand_dir: Path, klass: str, aa: dict | None) -> dict:
    ref, cand = load_t1(ref_dir), load_t1(cand_dir)
    missing = sorted(set(ref) ^ set(cand))
    mism = sorted(k for k in set(ref) & set(cand) if ref[k]["meta"]["prompt_ids_sha"] != cand[k]["meta"]["prompt_ids_sha"])
    by_dom: dict[str, list] = {}
    for k in sorted(set(ref) & set(cand)):
        if k in mism:
            continue
        r, c = ref[k], cand[k]
        by_dom.setdefault(r["meta"]["domain"], []).append({
            "agree": r["top_ids"][:, 0] == c["top_ids"][:, 0],
            "kl": kl_topk(r["top_ids"], r["top_lp"], c["top_ids"], c["top_lp"], c["tgt"], c["tgt_lp"]),
            "ref_ok": r["top_ids"][:, 0] == r["tgt"], "cand_ok": c["top_ids"][:, 0] == c["tgt"],
            "ref_lp": r["tgt_lp"], "cand_lp": c["tgt_lp"]})
    if not by_dom:
        return invalid("no comparable items", missing=missing, mismatched=mism)
    overall = _t1_metrics([p for v in by_dom.values() for p in v])
    domains = {d: _t1_metrics(v) for d, v in sorted(by_dom.items())}
    floor_top1 = aa["metrics"]["top1_agree"] if aa else None
    floor_kl = aa["metrics"]["kl"] if aa else None
    thr_top1 = threshold_agree(CLASS[klass]["top1"], floor_top1)
    thr_kl = threshold_lower(CLASS[klass]["kl"], floor_kl)
    dom_kl_lim = max(2 * overall["kl"], DOMAIN_NOISE * thr_kl)
    dom_dis_lim = max(2 * (1 - overall["top1_agree"]), DOMAIN_NOISE * (1 - thr_top1))
    bad_dom = [d for d, m in domains.items() if m["kl"] > dom_kl_lim or (1 - m["top1_agree"]) > dom_dis_lim]
    checks = {"top1": overall["top1_agree"] >= thr_top1, "kl": overall["kl"] <= thr_kl, "domains": not bad_dom}
    ok = all(checks.values())
    escalate = klass == "B" and thr_top1 < ESCALATE_TOP1_B
    verdict = "ESCALATE" if escalate else ("PASS" if ok else "FAIL")
    if missing or mism:
        verdict, ok = "INVALID", False
    return {"gate": f"T1-{klass}", "verdict": verdict, "pass": ok and not escalate, "checks": checks,
            "thresholds": {"top1": round(thr_top1, 6), "kl": thr_kl, "domain_kl": round(dom_kl_lim, 8),
                           "domain_disagree": round(dom_dis_lim, 6), "floor_top1": floor_top1, "floor_kl": floor_kl},
            "metrics": overall, "by_domain": domains, "bad_domains": bad_dom,
            "missing_items": missing, "mismatched_items": mism}


# ------------------------------------------------------------------ T1-G

def load_t1g(d: Path) -> dict[str, list[tuple]]:
    runs: dict[str, list[tuple]] = {}
    for r in sorted(jsonl(d / "t1g.jsonl"), key=lambda r: (r["id"], r["repeat"])):
        runs.setdefault(r["id"], []).append(tuple(r["token_ids"]))
    return runs


def modal(seqs: list[tuple]) -> tuple:
    counts = Counter(seqs)
    best = max(counts.values())
    return next(s for s in seqs if counts[s] == best)  # ties -> earliest repeat


def t1g_stats(run: dict, ref_modal: dict) -> dict:
    distinct = [len(set(v)) for v in run.values()]
    early = [sum(s[:EARLY] != ref_modal[k][:EARLY] for s in v) / len(v) for k, v in run.items()]

    def se(xs):
        return statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0

    return {"prompts": len(run), "repeats": max(len(v) for v in run.values()),
            "distinct_mean": round(statistics.fmean(distinct), 6), "distinct_se": round(se(distinct), 6),
            "early_div": round(statistics.fmean(early), 6), "early_div_se": round(se(early), 6),
            "prompts_with_divergence": sum(d > 1 for d in distinct)}


def score_t1g(lkg_dir: Path, cand_dir: Path, aa: dict | None) -> dict:
    lkg, cand = load_t1g(lkg_dir), load_t1g(cand_dir)
    if set(lkg) != set(cand):
        return invalid("prompt sets differ", missing=sorted(set(lkg) ^ set(cand)))
    ref_modal = {k: modal(v) for k, v in lkg.items()}
    m = t1g_stats(cand, ref_modal)
    out = {"gate": "T1-G", "metrics": m, "lkg_self": t1g_stats(lkg, ref_modal)}
    if not aa:
        return {**out, "verdict": "REPORT", "pass": None, "note": "no --aa floor: report only (C10/S1.5)"}
    f = aa["metrics"]
    lim_d = f["distinct_mean"] + 2 * f["distinct_se"]
    lim_e = f["early_div"] + 2 * f["early_div_se"]
    checks = {"distinct": m["distinct_mean"] <= lim_d, "early_div": m["early_div"] <= lim_e}
    ok = all(checks.values())
    return {**out, "verdict": "PASS" if ok else "FAIL", "pass": ok, "checks": checks,
            "thresholds": {"distinct_mean": round(lim_d, 6), "early_div": round(lim_e, 6)}}


# ------------------------------------------------------------------ T1-D

def load_t1d(d: Path) -> dict:
    idx = jsonl(d / "index.jsonl")
    with np.load(d / "t1d.npz") as z:
        arr = {k: z[k] for k in z.files}
    return {r["id"]: {"meta": r, **{k: v[r["off"]:r["off"] + r["n"]] for k, v in arr.items()}} for r in idx}


def first_div(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    neq = np.nonzero(a[:n] != b[:n])[0]
    return int(neq[0]) if len(neq) else n


def score_t1d(gold_dir: Path, cand_dir: Path, klass: str, floor: dict | None) -> dict:
    gold, cand = load_t1d(gold_dir), load_t1d(cand_dir)
    if set(gold) != set(cand):
        return invalid("prompt sets differ", missing=sorted(set(gold) ^ set(cand)))
    divs, kls, per = [], [], {}
    for k in sorted(gold):
        g, c = gold[k], cand[k]
        d = first_div(g["ids"], c["ids"])
        divs.append(d)
        if d:
            kl = kl_topk(g["top_ids"][:d], g["top_lp"][:d], c["top_ids"][:d], c["top_lp"][:d], c["ids"][:d], c["lp"][:d])
            kls.append(kl)
        per[k] = {"first_div": d, "len_gold": int(len(g["ids"])), "len_cand": int(len(c["ids"]))}
    allkl = np.concatenate(kls) if kls else np.zeros(1)
    m = {"prompts": len(divs), "median_first_div": float(statistics.median(divs)),
         "mean_first_div": round(statistics.fmean(divs), 2), "identical": sum(p["first_div"] == p["len_gold"] == p["len_cand"] for p in per.values()),
         "matched_prefix_tokens": int(sum(divs)), "matched_prefix_kl": round(float(allkl.mean()), 8)}
    out = {"gate": f"T1-D/{klass}", "metrics": m, "per_prompt": per}
    if not floor:
        return {**out, "verdict": "REPORT", "pass": None,
                "note": "no --floor (golden-vs-golden c=8 t1d JSON): report only"}
    fm = floor["metrics"]
    lim_div = 0.5 * fm["median_first_div"]
    lim_kl = threshold_lower(CLASS[klass]["kl"], fm["matched_prefix_kl"])
    checks = {"median_first_div": m["median_first_div"] >= lim_div, "matched_prefix_kl": m["matched_prefix_kl"] <= lim_kl}
    ok = all(checks.values())
    return {**out, "verdict": "PASS" if ok else "FAIL", "pass": ok, "checks": checks,
            "thresholds": {"median_first_div": lim_div, "matched_prefix_kl": lim_kl}}


# ------------------------------------------------------------------ T2 / T4

EXACT_TASKS = {"tools", "json", "effort"}  # candidate must be all-correct
VALUE_TASKS = {"rep4"}                    # lower is better, per-item "value"


def load_rows(d: Path, name: str) -> dict:
    rows = {}
    for r in jsonl(d / name):
        rows[(r["task"], str(r["id"]))] = r
    return rows


def score_paired(ref_dir: Path, cand_dir: Path, name: str, max_drop: float, aa: dict | None,
                 gate: str) -> dict:
    ref, cand = load_rows(ref_dir, name), load_rows(cand_dir, name)
    if set(ref) != set(cand):
        return invalid("item sets differ", missing=[list(k) for k in sorted(set(ref) ^ set(cand))][:20])
    tasks = sorted({t for t, _ in ref})
    res, checks = {}, {}
    for t in tasks:
        keys = sorted(k for k in ref if k[0] == t)
        if t in VALUE_TASKS:
            rv = statistics.fmean(ref[k]["value"] for k in keys)
            cv = statistics.fmean(cand[k]["value"] for k in keys)
            floor = aa["tasks"][t]["abs_diff"] if aa and t in aa.get("tasks", {}) else None
            tol = threshold_lower(REP4_TOL, floor)
            res[t] = {"n": len(keys), "ref": round(rv, 6), "cand": round(cv, 6), "abs_diff": round(abs(cv - rv), 6),
                      "tolerance": tol}
            checks[t] = cv <= rv + tol
            continue
        a = [bool(ref[k]["correct"]) for k in keys]
        b = [bool(cand[k]["correct"]) for k in keys]
        m = mcnemar(a, b)
        if t in EXACT_TASKS:
            m["rule"] = f"{sum(b)}/{len(b)} must be all correct"
            checks[t] = all(b)
        else:
            m["rule"] = f"McNemar p >= {P_SIG} and drop <= {max_drop} pp"
            checks[t] = m["p"] >= P_SIG and m["drop_pp"] <= max_drop
        res[t] = m
    ok = all(checks.values())
    return {"gate": gate, "verdict": "PASS" if ok else "FAIL", "pass": ok, "checks": checks, "tasks": res}


# ------------------------------------------------------------------ main

def invalid(reason: str, **kw) -> dict:
    return {"verdict": "INVALID", "pass": False, "reason": reason, **kw}


def _load(path: str | None) -> dict | None:
    return json.loads(Path(path).read_text()) if path else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("t1", "t1g", "t1d", "t2", "t4"):
        p = sub.add_parser(name)
        p.add_argument("ref", type=Path)
        p.add_argument("cand", type=Path)
        p.add_argument("--json", type=Path)
        if name in ("t1", "t1d"):
            p.add_argument("--class", dest="klass", choices=("A", "B"), default="A")
        if name in ("t1", "t1g", "t2"):
            p.add_argument("--aa", help="A/A JSON from the same subcommand (C25 floors)")
        if name == "t1d":
            p.add_argument("--floor", help="golden-vs-golden (c=8) t1d JSON")
    th = sub.add_parser("threshold")
    th.add_argument("--kind", choices=("lower", "agree"), required=True)
    th.add_argument("--stated", type=float, required=True)
    th.add_argument("--floor", type=float)
    a = ap.parse_args(argv)
    if a.cmd == "threshold":
        f = threshold_lower if a.kind == "lower" else threshold_agree
        print(json.dumps({"threshold": f(a.stated, a.floor)}))
        return 0
    if a.cmd == "t1":
        out = score_t1(a.ref, a.cand, a.klass, _load(a.aa))
    elif a.cmd == "t1g":
        out = score_t1g(a.ref, a.cand, _load(a.aa))
    elif a.cmd == "t1d":
        out = score_t1d(a.ref, a.cand, a.klass, _load(a.floor))
    elif a.cmd == "t2":
        out = score_paired(a.ref, a.cand, "t2.jsonl", T2_MAX_DROP_PP, _load(a.aa), "T2")
    else:
        out = score_paired(a.ref, a.cand, "t4.jsonl", T4_MAX_DROP_PP, None, "T4")
    out = {"ref": str(a.ref), "cand": str(a.cand), **out}
    text = json.dumps(out, indent=1)
    if a.json:
        a.json.write_text(text + "\n")
    print(text)
    return {"PASS": 0, "REPORT": 0, "FAIL": 1, "ESCALATE": 1}.get(out["verdict"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
