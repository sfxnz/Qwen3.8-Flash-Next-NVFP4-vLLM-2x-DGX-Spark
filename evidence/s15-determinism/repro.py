#!/usr/bin/env python3
"""S1.5 c=1 reproducibility harness (stdlib only). Talks to a running serve; never starts one.

Prompts are raw token ids cut from the English `long` domain of quality/data/t1_corpus.jsonl
(tokenized once through /tokenize), sent to /v1/completions, one request at a time.
Every request carries a fresh cache_salt unless --same-salt, so no prefix-cache hit.

  plp   teacher-forced prompt_logprobs=1 at each --lens, --n repeats. Per length: max |dlogprob| of the
        prompt token vs repeat 0, first differing position (> 0 and > 1e-3), top-1 agreement, number of
        distinct fingerprints, and the positions of the first diff per 512-token bucket.
  gen   greedy max_tokens --max-tokens, logprobs=1, at each --lens. Reports distinct outputs, first
        divergence index, and |d| of the first generated token's logprob (a prefill-only quantity).
  hist  history dependence: A alone x3 back to back, then F_i -> A for 3 different fillers F_i,
        then F_1 -> A again. If A only changes with its predecessor, per-request state leaks.

  python3 repro.py plp --lens 16,64,200,512,2048 --n 5 --out DIR
  python3 repro.py gen --lens 1,200,8192 --n 5 --out DIR
  python3 repro.py hist --len 200 --out DIR
Prompt logprobs cost about n x 248320 x 4 B transient per request (4 GB at 4096); --max-plp caps it.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "quality/data/t1_corpus.jsonl"
URL = "http://127.0.0.1:8000"


def post(url, path, body, timeout=1800):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def running(url):
    try:
        with urllib.request.urlopen(url + "/metrics", timeout=10) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("vllm:num_requests_running"):
                    return float(line.rsplit(" ", 1)[1])
    except OSError:
        return None
    return None


def model(url):
    with urllib.request.urlopen(url + "/v1/models", timeout=10) as r:
        return json.loads(r.read().decode())["data"][0]["id"]


def token_pool(url, m, need, offset=0):
    """Deterministic token ids from the corpus `long` domain (English novels), >= need + offset tokens."""
    text, ids = [], []
    for line in CORPUS.open():
        row = json.loads(line)
        if row["domain"] != "long":
            continue
        text.append(row["messages"][-1]["content"])
        if sum(len(t) for t in text) > (need + offset) * 5:
            break
    ids = post(url, "/tokenize", {"model": m, "prompt": "\n\n".join(text), "add_special_tokens": False})["tokens"]
    if len(ids) < need + offset:
        raise SystemExit(f"corpus gave {len(ids)} tokens < {need + offset}")
    return [int(t) for t in ids[offset:offset + need]]


def salt(args):
    return "s15-fixed" if args.same_salt else "s15-" + uuid.uuid4().hex


def plp_once(args, m, ids):
    body = {"model": m, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1,
            "cache_salt": salt(args)}
    r0 = running(args.url)
    t = time.time()
    out = post(args.url, "/v1/completions", body)
    plp = out["choices"][0]["prompt_logprobs"]
    lp, top = [], []
    for i in range(1, len(ids)):
        d = plp[i]
        lp.append(d[str(ids[i])]["logprob"])
        top.append(min(d.items(), key=lambda kv: kv[1]["rank"])[0])
    return {"lp": lp, "top": top, "running_before": r0, "s": round(time.time() - t, 3)}


def cmp_plp(runs, bucket=512):
    ref = runs[0]
    rows = []
    for k, r in enumerate(runs[1:], 1):
        d = [abs(a - b) for a, b in zip(ref["lp"], r["lp"])]
        first0 = next((i + 1 for i, x in enumerate(d) if x > 0), None)
        first3 = next((i + 1 for i, x in enumerate(d) if x > 1e-3), None)
        agree = sum(a == b for a, b in zip(ref["top"], r["top"])) / max(1, len(d))
        per_bucket = {}
        for i, x in enumerate(d):
            b = (i + 1) // bucket * bucket
            per_bucket[b] = max(per_bucket.get(b, 0.0), x)
        rows.append({"vs": k, "max_abs": round(max(d) if d else 0.0, 6), "mean_abs": round(sum(d) / max(1, len(d)), 6),
                     "first_pos_gt0": first0, "first_pos_gt1e-3": first3, "top1_agree": round(agree, 4),
                     "n_pos_gt1e-3": sum(x > 1e-3 for x in d), "frac_exact": round(sum(x == 0 for x in d) / max(1, len(d)), 5),
                     "bucket_max": {str(b): round(v, 4) for b, v in sorted(per_bucket.items())}})
    fp = {hashlib.sha256(json.dumps(r["lp"]).encode()).hexdigest()[:12] for r in runs}
    return {"distinct_fingerprints": len(fp), "pairs": rows,
            "max_abs": max((p["max_abs"] for p in rows), default=0.0),
            "frac_exact_min": min((p["frac_exact"] for p in rows), default=1.0),
            "first_pos_gt1e-3": min((p["first_pos_gt1e-3"] for p in rows if p["first_pos_gt1e-3"]), default=None),
            "running_before": [r["running_before"] for r in runs]}


def gen_once(args, m, ids):
    body = {"model": m, "prompt": ids, "max_tokens": args.max_tokens, "temperature": 0, "logprobs": 1,
            "cache_salt": salt(args), "return_token_ids": True, "ignore_eos": True}
    r0 = running(args.url)
    out = post(args.url, "/v1/completions", body)["choices"][0]
    return {"toks": out.get("token_ids") or out["logprobs"]["tokens"], "lp": out["logprobs"]["token_logprobs"],
            "running_before": r0}


def cmp_gen(runs):
    ref = runs[0]
    rows = []
    for k, r in enumerate(runs[1:], 1):
        div = next((i for i, (a, b) in enumerate(zip(ref["toks"], r["toks"])) if a != b), None)
        n = div if div is not None else len(ref["toks"])
        dlp = [abs(a - b) for a, b in zip(ref["lp"][:n], r["lp"][:n])]
        rows.append({"vs": k, "first_div": div, "first_tok_dlp": round(abs(ref["lp"][0] - r["lp"][0]), 6),
                     "max_dlp_before_div": round(max(dlp) if dlp else 0.0, 6)})
    return {"distinct": len({json.dumps(r["toks"]) for r in runs}), "pairs": rows,
            "first_div_min": min((p["first_div"] for p in rows if p["first_div"] is not None), default=None),
            "first_tok_dlp_max": max((p["first_tok_dlp"] for p in rows), default=0.0),
            "running_before": [r["running_before"] for r in runs]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["plp", "gen", "hist"])
    ap.add_argument("--url", default=URL)
    ap.add_argument("--lens", default="16,64,200,512,2048")
    ap.add_argument("--len", type=int, default=200)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-plp", type=int, default=4096)
    ap.add_argument("--same-salt", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = model(a.url)
    res = {"mode": a.mode, "argv": sys.argv[1:], "tag": a.tag, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "model": m, "results": {}}
    raw = {}
    if a.mode in ("plp", "gen"):
        lens = [int(x) for x in a.lens.split(",")]
        pool = token_pool(a.url, m, max(lens))
        for n in lens:
            if a.mode == "plp" and n > a.max_plp:
                print(f"skip plp len {n} > --max-plp {a.max_plp}", file=sys.stderr)
                continue
            ids = pool[:n]
            runs = [(plp_once if a.mode == "plp" else gen_once)(a, m, ids) for _ in range(a.n)]
            raw[str(n)] = runs
            r = (cmp_plp if a.mode == "plp" else cmp_gen)(runs)
            res["results"][str(n)] = r
            brief = {k: v for k, v in r.items() if k != "pairs"}
            print(json.dumps({"len": n, **brief}), flush=True)
    else:
        pool = token_pool(a.url, m, a.len * 5)
        A = pool[:a.len]
        F = [pool[a.len * (i + 1):a.len * (i + 2)] for i in range(3)]
        seq = [("A", A), ("A", A), ("A", A), ("F1", F[0]), ("A|F1", A), ("F2", F[1]), ("A|F2", A),
               ("F3", F[2]), ("A|F3", A), ("F1", F[0]), ("A|F1", A)]
        runs = []
        for name, ids in seq:
            r = plp_once(a, m, ids)
            r["name"] = name
            runs.append(r)
        raw["hist"] = runs
        a_runs = [r for r in runs if r["name"].startswith("A")]
        names = [r["name"] for r in a_runs]
        fps = [hashlib.sha256(json.dumps(r["lp"]).encode()).hexdigest()[:12] for r in a_runs]
        c = cmp_plp(a_runs)
        res["results"]["hist"] = {"names": names, "fingerprints": fps, **c}
        print(json.dumps({"names": names, "fingerprints": fps, "max_abs_vs_first": [p["max_abs"] for p in c["pairs"]]}))
    stem = f"{a.mode}{('-' + a.tag) if a.tag else ''}-{int(time.time())}"
    (out / f"{stem}.json").write_text(json.dumps(res, indent=1))
    (out / f"{stem}.raw.json").write_text(json.dumps(raw))
    print(f"wrote {out / stem}.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
