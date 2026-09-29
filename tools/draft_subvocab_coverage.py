#!/usr/bin/env python3
"""CPU-only vocabulary-coverage study for the L1b reduced-vocab MTP draft head (plan 0-10, F02).

Adapted from the DeepSeek-V4.1 sibling's tools/draft_subvocab_coverage.py.

The drafter proposes argmax(W' h) over a V' id subset, so a target token
outside V' can never be drafted and ends the accepted chain. Target tokens
are measured directly: greedy (T=0, thinking off) completions from the
served model, whose token ids ARE the target argmax sequence.

Subcommands (host Python: stdlib + numpy; no torch, no GPU):

  collect  sample ~300 greedy completions across domains from a live server
           (return_token_ids; light load: default concurrency 4, 256 tokens)
  corpus   tokenize local text corpora per domain (tokenizers lib if present,
           else the server's /tokenize endpoint) into id counts
  analyze  rank ids, score coverage per domain with 2-fold cross-validation
           over prompts, choose the smallest V' in --sizes with >= --target
           coverage in every domain, write draft_vocab_<V>.json + report

Special tokens (tokenizer added_tokens) and the 256 byte-level alphabet ids
are always in V'. Raw intermediates default to ~/projects/data/qwen38-draft-vocab
(AGENTS.md: large artifacts go to data/); only id lists and reports go to
tools/vocab/.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

VOCAB = 248320  # lm_head rows (tokenizer uses 248,077; the rest is padding)
MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"
SIZES = (32768, 47104, 65536)
ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "projects" / "data" / "qwen38-draft-vocab"
SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots"
    / "fab0aecb760cec45227f6656abcaafa11abca87a"
)
EVALS = Path.home() / "projects" / "data" / "glm53-evals"


# ----------------------------------------------------------------- math core
def rank_ids(score: np.ndarray, must: np.ndarray) -> np.ndarray:
    """must ids first (ascending), then by score desc, ties to the lower id."""
    order = np.lexsort((np.arange(score.size), -score))
    mask = np.zeros(score.size, dtype=bool)
    mask[must] = True
    return np.concatenate([np.sort(must), order[~mask[order]]])


def keep_set(ranked: np.ndarray, n: int) -> np.ndarray:
    return np.sort(ranked[:n])


def coverage(ids: np.ndarray, keep: np.ndarray, vocab: int = VOCAB) -> float:
    if ids.size == 0:
        return float("nan")
    mask = np.zeros(vocab, dtype=bool)
    mask[keep] = True
    return float(mask[ids].mean())


def accept_len(alpha: float, k: int, c: float = 1.0) -> float:
    """Tokens per verify step: bonus token + sum_i (alpha*c)^i over k drafts."""
    a = alpha * c
    return 1.0 + sum(a**i for i in range(1, k + 1))


def alpha_for(tau: float, k: int) -> float:
    """Per-position acceptance that gives tau tokens/step at coverage 1."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if accept_len(mid, k) < tau:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-level alphabet (the Qwen BPE base symbols)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def must_include(tokenizer_json: Path) -> tuple[np.ndarray, dict]:
    tok = json.loads(tokenizer_json.read_text())
    vocab = tok["model"]["vocab"]
    special = sorted(int(t["id"]) for t in tok.get("added_tokens", []))
    byte_ids = sorted(vocab[ch] for ch in bytes_to_unicode().values() if ch in vocab)
    info = {
        "special_ids": len(special),
        "byte_alphabet_ids": len(byte_ids),
        "tokenizer_json_sha256": hashlib.sha256(tokenizer_json.read_bytes()).hexdigest(),
        "tokenizer_vocab": len(vocab) + len(special),
    }
    return np.array(sorted(set(special) | set(byte_ids)), dtype=np.int64), info


# ---------------------------------------------------------------- server I/O
def _post(url: str, body: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_health(base: str, timeout: float) -> None:
    t0 = time.time()
    while True:
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        if time.time() - t0 > timeout:
            raise SystemExit(f"{base}/health not 200 after {timeout:.0f}s")
        time.sleep(10)


class Tokenizer:
    """tokenizers.Tokenizer when importable, else the server's /tokenize."""

    def __init__(self, base: str | None, model: str, tokenizer_json: Path):
        self.base, self.model, self.lib = base, model, None
        try:
            from tokenizers import Tokenizer as T

            self.lib = T.from_file(str(tokenizer_json))
        except ImportError:
            if not base:
                raise SystemExit("no `tokenizers` module: pass --base-url for /tokenize")

    def encode(self, text: str) -> list[int]:
        if self.lib is not None:
            return self.lib.encode(text, add_special_tokens=False).ids
        d = _post(
            self.base + "/tokenize",
            {"model": self.model, "prompt": text, "add_special_tokens": False},
        )
        return d["tokens"]


# ------------------------------------------------------------------- prompts
_TOPICS = [
    "a lighthouse keeper", "a bakery at dawn", "a lost umbrella", "a chess club",
    "a mountain village", "an old bicycle", "a rainy train station", "a beekeeper",
    "a night market", "a school science fair",
]
_CODE = [
    "Write a Python function that merges overlapping intervals, with a docstring and two doctests.",
    "Write a Go function that reverses the words in a string, with a table-driven test.",
    "Write a Rust function that parses a semantic version string into a struct, with error handling.",
    "Write a JavaScript debounce function with an example of its use.",
    "Write a SQL query that returns the top three customers by total order value per country.",
    "Write a bash script that finds the ten largest files under a directory.",
    "Implement an LRU cache class in Python using OrderedDict.",
    "Write a C function that computes a CRC32 of a byte buffer.",
    "Write a TypeScript interface and a function that validates a user profile object.",
    "Write a Python asyncio example that fetches three URLs concurrently with a timeout.",
    "Refactor this Python loop into a list comprehension and explain: result = []\nfor x in range(20):\n    if x % 3 == 0:\n        result.append(x * x)",
    "Write a Python class for a binary search tree with insert, search and in-order traversal.",
    "Write a Dockerfile for a small Flask app and explain each line briefly.",
    "Write a regular expression that matches ISO 8601 dates and show Python code that uses it.",
    "Write a Rust iterator adapter that yields every second element.",
]
_JSON = [
    "Return only a JSON object describing {t}, with keys name, location, year_founded, tags (array) and summary. No prose.",
    "Produce a JSON array of five fictional products related to {t}; each item has id, title, price and in_stock. Output JSON only.",
    "Output a JSON Schema (draft 2020-12) for an event record about {t}. Output only the schema.",
]
_LANG = {
    "zh": [
        "请用中文写一段大约两百字的短文，主题是{t}。",
        "用中文解释一下为什么天空是蓝色的，并举一个与{t}有关的例子。",
        "请用中文给出三条关于{t}的实用建议，并简要说明理由。",
    ],
    "es": [
        "Escribe en español un relato breve de unas doscientas palabras sobre {t}.",
        "Explica en español, de forma sencilla, cómo funciona la fotosíntesis, usando un ejemplo sobre {t}.",
        "Da en español tres consejos prácticos relacionados con {t} y justifica cada uno.",
    ],
    "de": [
        "Schreibe auf Deutsch eine kurze Geschichte von etwa zweihundert Wörtern über {t}.",
        "Erkläre auf Deutsch in einfachen Worten, wie ein Regenbogen entsteht, mit einem Beispiel über {t}.",
        "Nenne auf Deutsch drei praktische Tipps zum Thema {t} und begründe sie kurz.",
    ],
    "ja": [
        "{t}について、日本語で二百字程度の短い物語を書いてください。",
        "日本語で、空が青く見える理由を簡単に説明し、{t}に関する例を一つ挙げてください。",
        "{t}に関する実用的なアドバイスを日本語で三つ挙げ、それぞれ理由を説明してください。",
    ],
}
_LANG_TOPICS = {
    "zh": ["灯塔看守人", "清晨的面包店", "丢失的雨伞", "象棋俱乐部", "山村", "旧自行车", "雨中的火车站", "养蜂人", "夜市", "科学展览"],
    "es": ["un farero", "una panadería al amanecer", "un paraguas perdido", "un club de ajedrez", "un pueblo de montaña", "una bicicleta vieja", "una estación de tren bajo la lluvia", "un apicultor", "un mercado nocturno", "una feria de ciencias"],
    "de": ["einen Leuchtturmwärter", "eine Bäckerei im Morgengrauen", "einen verlorenen Regenschirm", "einen Schachverein", "ein Bergdorf", "ein altes Fahrrad", "einen verregneten Bahnhof", "einen Imker", "einen Nachtmarkt", "eine Wissenschaftsmesse"],
    "ja": ["灯台守", "夜明けのパン屋", "なくした傘", "チェスクラブ", "山の村", "古い自転車", "雨の駅", "養蜂家", "夜市", "科学フェア"],
}


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _tool(fn: dict) -> dict:
    def fix(x):
        if isinstance(x, dict):
            return {k: ("object" if k == "type" and v == "dict" else fix(v)) for k, v in x.items()}
        if isinstance(x, list):
            return [fix(v) for v in x]
        return x

    return {"type": "function", "function": fix(fn)}


def build_prompts(per: int, seed: int) -> list[dict]:
    """[{domain, pid, messages, tools?}]; deterministic for a given seed."""
    rng = random.Random(seed)
    out: list[dict] = []

    def add(domain: str, items: list[dict]) -> None:
        rng.shuffle(items)
        for i, it in enumerate(items[:per]):
            out.append({"domain": domain, "pid": f"{domain}-{i:03d}", **it})

    ds = EVALS / "datasets"
    add("prose", [{"messages": [{"role": "user", "content": d["prompt"]}]} for d in _jsonl(ds / "ifeval/items.jsonl")]
        or [{"messages": [{"role": "user", "content": f"Write a short essay about {t}."}]} for t in _TOPICS * 3])
    add("math", [{"messages": [{"role": "user", "content": d["question"] + "\nShow your working, then give the final answer."}]}
                 for d in _jsonl(ds / "gsm8k/items.jsonl")])
    add("code", [{"messages": [{"role": "user", "content": c}]} for c in _CODE * 2])
    add("json", [{"messages": [{"role": "user", "content": j.format(t=t)}]} for j in _JSON for t in _TOPICS])
    add("tools", [{"messages": d["messages"], "tools": [_tool(f) for f in d["functions"]]}
                  for d in _jsonl(ds / "bfcl/items.jsonl") if d.get("functions")])
    for lang, tmpls in _LANG.items():
        add(lang, [{"messages": [{"role": "user", "content": tm.format(t=t)}]}
                   for tm in tmpls for t in _LANG_TOPICS[lang]])
    return out


# ----------------------------------------------------------------- commands
def cmd_collect(a: argparse.Namespace) -> int:
    wait_health(a.base_url, a.wait)
    prompts = build_prompts(a.per_domain, a.seed)
    tok = None
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def one(p: dict) -> dict:
        body = {
            "model": a.model,
            "messages": p["messages"],
            "max_tokens": a.max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "return_token_ids": True,
        }
        if p.get("tools"):
            body["tools"] = p["tools"]
        t0 = time.time()
        d = _post(a.base_url + "/v1/chat/completions", body)
        ch = d["choices"][0]
        ids = ch.get("token_ids")
        src = "token_ids"
        if ids is None:  # older server: re-tokenize the text (approximate)
            nonlocal tok
            tok = tok or Tokenizer(a.base_url, a.model, a.tokenizer_json)
            ids, src = tok.encode(ch["message"].get("content") or ""), "retokenized"
        return {
            "domain": p["domain"],
            "pid": p["pid"],
            "finish_reason": ch.get("finish_reason"),
            "ids_source": src,
            "completion_tokens": d.get("usage", {}).get("completion_tokens"),
            "wall_s": round(time.time() - t0, 3),
            "ids": ids,
        }

    rows, fails = [], 0
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for fut in cf.as_completed([ex.submit(one, p) for p in prompts]):
            try:
                rows.append(fut.result())
            except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
                fails += 1
                print(f"collect: request failed: {e}", file=sys.stderr)
    rows.sort(key=lambda r: r["pid"])
    with a.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    by = {}
    for r in rows:
        by.setdefault(r["domain"], []).append(len(r["ids"]))
    for dom, lens in sorted(by.items()):
        print(f"collect: {dom:6} n={len(lens):3d} tokens={sum(lens):6d}")
    print(f"collect: wrote {len(rows)} completions ({fails} failed) to {a.out}")
    return 1 if fails > len(prompts) // 10 else 0


def _corpus_sources() -> dict[str, list[str]]:
    src = EVALS / "corpus-src"
    lang = {}
    for p in sorted(src.glob("*gutenberg*")):
        head = p.read_text(errors="replace")[:4000]
        for line in head.splitlines():
            if line.startswith("Language:"):
                lang.setdefault(line.split(":", 1)[1].strip().lower(), []).append(str(p))
                break
    code = [str(p) for p in sorted(src.iterdir()) if p.suffix in {".py", ".go", ".rs", ".c", ".js"}]
    math = [str(p) for p in sorted(src.glob("*hendrycks_math*"))] + [str(p) for p in sorted(src.glob("*grade_school_math*"))]
    out = {"code": code, "math": math}
    for k in ("english", "chinese", "spanish", "german", "japanese", "french", "italian", "portuguese", "dutch", "finnish"):
        if k in lang:
            out[k] = lang[k]
    return out


def _read_text(p: Path) -> str:
    """Plain text of a corpus file; JSON rows/lines reduced to their string values."""
    t = p.read_text(errors="replace")
    if "datasets-server" in p.name or p.name.endswith(".jsonl"):
        docs = []
        for line in t.splitlines():
            try:
                docs.append(json.loads(line))
            except ValueError:
                continue
        vals: list[str] = []

        def walk(x):
            if isinstance(x, str):
                vals.append(x)
            elif isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)

        for d in docs:
            walk(d.get("rows", d) if isinstance(d, dict) else d)
        return "\n\n".join(v for v in vals if len(v) > 20)
    return t


def _chunks(text: str, size: int) -> list[str]:
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        nl = text.rfind("\n", i, j)
        if j < len(text) and nl > i:
            j = nl + 1
        out.append(text[i:j])
        i = j
    return out


def cmd_corpus(a: argparse.Namespace) -> int:
    tok = Tokenizer(a.base_url, a.model, a.tokenizer_json)
    if tok.lib is None:
        wait_health(a.base_url, a.wait)
    result = {"budget_chars_per_domain": a.chars_per_domain, "domains": {}}
    for dom, files in _corpus_sources().items():
        per_file = max(1, a.chars_per_domain // max(1, len(files)))
        texts = []
        for f in files:
            t = _read_text(Path(f))
            if "*** START OF" in t:  # drop the Gutenberg licence header
                t = t.split("*** START OF", 1)[1].split("\n", 1)[-1]
            texts.extend(_chunks(t[:per_file], 6000))
        counts = np.zeros(VOCAB, dtype=np.int64)
        with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            for ids in ex.map(tok.encode, texts):
                np.add.at(counts, np.asarray(ids, dtype=np.int64), 1)
        nz = np.nonzero(counts)[0]
        result["domains"][dom] = {
            "files": len(files),
            "tokens": int(counts.sum()),
            "ids": nz.tolist(),
            "counts": counts[nz].tolist(),
        }
        print(f"corpus: {dom:10} files={len(files):2d} tokens={int(counts.sum()):8d} distinct={nz.size}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, separators=(",", ":")))
    print(f"corpus: wrote {a.out}")
    return 0


def _freq(ids_by_dom: dict[str, np.ndarray]) -> np.ndarray:
    """Sum of per-domain relative frequencies (each domain weighs the same)."""
    s = np.zeros(VOCAB, dtype=np.float64)
    for ids in ids_by_dom.values():
        if ids.size:
            s += np.bincount(ids, minlength=VOCAB)[:VOCAB] / ids.size
    return s


def _corpus_freq(corpus: dict, only: set[str] | None = None) -> np.ndarray:
    s = np.zeros(VOCAB, dtype=np.float64)
    for name, d in corpus.get("domains", {}).items():
        if only and name not in only:
            continue
        c = np.zeros(VOCAB, dtype=np.float64)
        c[np.asarray(d["ids"], dtype=np.int64)] = d["counts"]
        if c.sum():
            s += c / c.sum()
    return s


def cmd_analyze(a: argparse.Namespace) -> int:
    rows = _jsonl(a.completions)
    if not rows:
        raise SystemExit(f"analyze: no completions in {a.completions}")
    corpus = json.loads(a.corpus.read_text()) if a.corpus and a.corpus.exists() else {}
    must, tinfo = must_include(a.tokenizer_json)
    sizes = sorted(int(x) for x in a.sizes.split(","))
    doms = sorted({r["domain"] for r in rows})

    def ids_of(sel) -> dict[str, np.ndarray]:
        out = {}
        for dm in doms:
            parts = [np.asarray(r["ids"], dtype=np.int64) for r in rows if r["domain"] == dm and sel(r)]
            out[dm] = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        return out

    fold = lambda r: int(r["pid"].rsplit("-", 1)[1]) % 2  # noqa: E731
    all_ids = ids_of(lambda r: True)
    only = set(a.corpus_domains.split(",")) if a.corpus_domains else None
    cf_ = _corpus_freq(corpus, only) * a.corpus_weight
    ranked_all = rank_ids(_freq(all_ids) + cf_, must)
    folds = []
    for k in (0, 1):
        train = ids_of(lambda r, k=k: fold(r) != k)
        test = ids_of(lambda r, k=k: fold(r) == k)
        folds.append((rank_ids(_freq(train) + cf_, must), test))
    ranked_corpus_only = rank_ids(cf_, must) if corpus else None

    alpha = alpha_for(a.tau, a.k)
    rep = {
        "model": a.model,
        "vocab_rows": VOCAB,
        **tinfo,
        "must_include": int(must.size),
        "completions": {dm: {"requests": sum(r["domain"] == dm for r in rows), "tokens": int(all_ids[dm].size)} for dm in doms},
        "ids_source": sorted({r.get("ids_source", "?") for r in rows}),
        "corpus_domains": {k: v["tokens"] for k, v in corpus.get("domains", {}).items()},
        "corpus_weight": a.corpus_weight,
        "corpus_domains_used": sorted(only) if only else "all",
        "target": a.target,
        "tau": a.tau,
        "k": a.k,
        "alpha": alpha,
        "sizes": {},
    }
    lines = [
        f"draft-vocab coverage  target>={a.target:.3f} in every domain  (tau={a.tau}, k={a.k}, alpha={alpha:.4f})",
        f"must-include ids={must.size}  completion tokens={sum(v.size for v in all_ids.values())}  corpus domains={len(rep['corpus_domains'])}",
        "cov_cv = 2-fold CV over prompts (rank on other fold + corpus); cov_in = in-sample; cov_corpus = corpus-only ranking",
        f"{'V':>6} {'domain':7} {'tokens':>7} {'cov_cv':>8} {'cov_in':>8} {'cov_corpus':>10} {'tau_loss%':>9}",
    ]
    chosen = None
    for n in sizes:
        keep_all = keep_set(ranked_all, n)
        per = {}
        for dm in doms:
            hits = tot = 0
            for ranked, test in folds:
                t = test[dm]
                if t.size:
                    hits += coverage(t, keep_set(ranked, n)) * t.size
                    tot += t.size
            cv = hits / tot if tot else float("nan")
            cin = coverage(all_ids[dm], keep_all)
            cco = coverage(all_ids[dm], keep_set(ranked_corpus_only, n)) if ranked_corpus_only is not None else float("nan")
            loss = 100 * (1 - accept_len(alpha, a.k, cv) / accept_len(alpha, a.k))
            per[dm] = {"tokens": int(all_ids[dm].size), "cov_cv": cv, "cov_in": cin, "cov_corpus_only": cco, "tau_loss_pct": loss}
            lines.append(f"{n:6d} {dm:7} {all_ids[dm].size:7d} {cv:8.4f} {cin:8.4f} {cco:10.4f} {loss:9.2f}")
        ok = all(v["cov_cv"] >= a.target for v in per.values())
        half = -(-n // 2)
        rep["sizes"][str(n)] = {
            "passes": ok,
            "min_cov_cv": min(v["cov_cv"] for v in per.values()),
            "rows_per_rank_tp2": [half, n - half],
            "mib_per_rank_bf16": half * 2560 * 2 / 2**20,
            "domains": per,
        }
        if ok and chosen is None:
            chosen = n
        out = a.out_dir / f"draft_vocab_{n}.json"
        doc = {
            "model": a.model,
            "vocab_size": VOCAB,
            "size": n,
            "tokenizer_json_sha256": tinfo["tokenizer_json_sha256"],
            "must_include": int(must.size),
            "passes_target": ok,
            "generator": "tools/draft_subvocab_coverage.py analyze",
            "ids": keep_all.tolist(),
        }
        a.out_dir.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    rep["chosen"] = chosen
    lines.append(
        f"chosen V'={chosen}" if chosen else f"no size reaches {a.target} in every domain; use the largest ({sizes[-1]}) and gate on acceptance"
    )
    (a.out_dir / "coverage_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    (a.out_dir / "coverage_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tokenizer-json", type=Path, default=SNAPSHOT / "tokenizer.json")
    ap.add_argument("--wait", type=float, default=60.0, help="seconds to wait for /health")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="greedy completions from the live server")
    c.add_argument("--out", type=Path, default=DATA / "completions.jsonl")
    c.add_argument("--per-domain", type=int, default=30)
    c.add_argument("--max-tokens", type=int, default=256)
    c.add_argument("--concurrency", type=int, default=4)
    c.add_argument("--seed", type=int, default=20260929)

    k = sub.add_parser("corpus", help="tokenize local corpora into id counts")
    k.add_argument("--out", type=Path, default=DATA / "corpus_counts.json")
    k.add_argument("--chars-per-domain", type=int, default=200_000)
    k.add_argument("--concurrency", type=int, default=4)

    z = sub.add_parser("analyze", help="rank, cross-validate, write draft_vocab_<V>.json")
    z.add_argument("--completions", type=Path, default=DATA / "completions.jsonl")
    z.add_argument("--corpus", type=Path, default=DATA / "corpus_counts.json")
    z.add_argument("--corpus-weight", type=float, default=1.0)
    z.add_argument(
        "--corpus-domains",
        default="code,math,english,chinese,spanish,german,japanese",
        help="corpus domains that feed the ranking ('' = all); unrelated languages steal slots",
    )
    z.add_argument("--sizes", default=",".join(map(str, SIZES)))
    z.add_argument("--target", type=float, default=0.99)
    z.add_argument("--tau", type=float, default=2.455, help="current prose tokens/step (k=3)")
    z.add_argument("--k", type=int, default=3)
    z.add_argument("--out-dir", type=Path, default=ROOT / "tools" / "vocab")

    a = ap.parse_args(argv)
    a.base_url = a.base_url.rstrip("/")
    return {"collect": cmd_collect, "corpus": cmd_corpus, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
