#!/usr/bin/env python3
"""Multilingual paired quality check: capture on two builds, then a blind pairwise judge (stdlib only).

  python3 quality/ml_judge.py collect --out RUN_DIR [--limit N]
  python3 quality/ml_judge.py judge --ref REF_RUN --cand CAND_RUN --out JUDGE_DIR [--json SUMMARY.json]

collect: every prompt in data/ml_prompts.json (40 each in en, de, es, fr, ja, zh: explain, summarise a
provided paragraph, write an email, translate-and-explain, a reasoning question with a short answer) at
c=1, greedy, thinking off, max_tokens 512 -> RUN_DIR/ml.jsonl. Run it on an idle serve; at c=1 the
determinism base makes the capture repeatable.

judge: the served model is the judge (run it on the candidate serve). Each item is judged twice, once in
each A/B order (the first order is random per item, --seed); a verdict counts only when both orders agree
(A/B/TIE mapped back to ref/cand/tie). Byte-identical outputs are not judged (counted as identical).
Per language: win/tie/loss for the candidate, inconsistent pairs, and a sign test on wins vs losses.
Also per language, for both builds: language-ID failures (script and stopword check that the answer is in
the requested language; the 32 non-reasoning items per language, LaTeX stripped), length ratio (cand/ref chars), repeat-4gram rate, and reasoning-item accuracy.

Pass rule: no language where the candidate loses significantly (losses > wins and one-sided sign test
p < 0.05); overall wins >= losses - 10% of decided (wins + losses); language-ID failures of the candidate
not above the reference, overall and per language. Exit 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qlib import (DATA, DEFAULT_URL, OFF, Client, append_jsonl, jsonl, pmap, prepare_out,  # noqa: E402
                  running_requests, write_json, write_manifest)

PROMPTS = DATA / "ml_prompts.json"
LANGS = ["en", "de", "es", "fr", "ja", "zh"]
LANG_NAME = {"en": "English", "de": "German", "es": "Spanish", "fr": "French", "ja": "Japanese", "zh": "Chinese"}
# Language-exclusive function words (shared ones such as la/de/en/que/un/es/in are left out).
STOP = {
    "en": "the and is are of to that it you your for with this be on was were have has will would".split(),
    "de": "der die das und ist nicht ich sie mit den zu ein eine auf für dem sich auch wird sind oder bei".split(),
    "es": "el los las y una por para con se del al lo como más pero su sus está son también".split(),
    "fr": "le les des et est une du pour dans pas qui sur au avec ce vous il je sont aussi mais leur".split(),
}
JUDGE_TMPL = """Compare two AI assistant responses to the same user request. The request is written in {lang}; a good response must be written in {lang} (quoted source text in another language is fine where the task asks for a translation).

Judge with this rubric, in order of importance:
1. Correctness: facts, arithmetic and reasoning are right; nothing is invented.
2. Instruction following: the response does exactly what was asked (format, length, number of sentences, requested content).
3. Language and fluency: natural, grammatical {lang}, with no switching into another language.
Ignore length unless the instruction constrains it, and ignore the order in which the responses are shown. If neither response is clearly better, answer TIE.

[User request]
{prompt}

[Response A]
{a}

[Response B]
{b}

Reply with one line only: "Verdict: A", "Verdict: B" or "Verdict: TIE"."""
VERDICT = re.compile(r"verdict\s*[:：]\s*\**\s*(A|B|TIE)\b", re.I)


# ------------------------------------------------------------------ text metrics

def _counts(text: str) -> dict:
    han = len(re.findall(r"[㐀-䶿一-鿿]", text))
    kana = len(re.findall(r"[぀-ヿ]", text))
    latin = len(re.findall(r"[A-Za-zÀ-ɏ]", text))
    return {"han": han, "kana": kana, "latin": latin}


def lang_id_ok(text: str, lang: str) -> bool:
    """Rough check that `text` is in `lang`: script shares for ja/zh, script + function-word vote for the rest."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$|\\[a-z]+", " ", text, flags=re.S)  # LaTeX math and commands
    c = _counts(text)
    letters = c["han"] + c["kana"] + c["latin"]
    if letters == 0:
        return False
    cjk = c["han"] + c["kana"]
    if lang == "ja":
        return cjk / letters >= 0.5 and c["kana"] / cjk >= 0.15
    if lang == "zh":
        return cjk / letters >= 0.5 and c["kana"] / cjk < 0.05
    if c["latin"] / letters < 0.7:
        return False
    words = re.findall(r"[a-zà-ÿ]+", text.lower())
    votes = {k: sum(1 for w in words if w in set(v)) for k, v in STOP.items()}
    return max(votes, key=votes.get) == lang and votes[lang] > 0


def rep4(text: str, lang: str) -> float:
    """Share of repeated 4-grams: words for Latin-script languages, characters for ja/zh."""
    toks = ([ch for ch in text if not ch.isspace() and ch.isalnum()] if lang in ("ja", "zh")
            else re.findall(r"\w+", text.lower()))
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def answer_ok(text: str, answers: list[str]) -> bool:
    """Reasoning items: an accepted answer on the last non-empty line (numbers matched on digit boundaries)."""
    lines = [x for x in text.strip().splitlines() if x.strip()]
    if not lines:
        return False
    last = re.sub(r"[\s*$\\]", "", lines[-1].lower()).replace("{", "").replace("}", "")
    for a in answers:
        a2 = re.sub(r"\s", "", a.lower())
        if re.fullmatch(r"[\d.,]+", a2):
            if re.search(r"(?<![\d.,])" + re.escape(a2) + r"(?:[.,]0+)?(?![\d]|[.,]\d)", last):
                return True
        elif a2 in last:
            return True
    return False


# ------------------------------------------------------------------ stats

def sign_test(wins: int, losses: int) -> dict:
    n = wins + losses
    if n == 0:
        return {"n": 0, "p_two_sided": 1.0, "p_against_cand": 1.0}
    tail = lambda k: sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n  # P(X >= k)
    return {"n": n, "p_two_sided": round(min(1.0, 2 * tail(max(wins, losses))), 6),
            "p_against_cand": round(tail(losses), 6)}


# ------------------------------------------------------------------ collect

def collect(a) -> int:
    prompts = json.loads(PROMPTS.read_text(encoding="utf-8"))
    prompts = prompts[:a.limit] if a.limit else prompts
    c = Client(a.url, a.model, timeout=900)
    out = prepare_out(a.out)
    write_manifest(out, c, "ml_judge.collect", vars(a), files=[PROMPTS])
    path = out / "ml.jsonl"
    path.write_text("")
    t0 = time.time()

    def one(p):
        busy = running_requests(c) if a.workers == 1 else None
        r = c.chat(p["prompt"], max_tokens=a.max_tokens, temperature=0, return_token_ids=True, **OFF)
        ids = r["token_ids"] or []
        return {"id": p["id"], "lang": p["lang"], "task": p["task"], "content": r["content"],
                "reasoning_chars": len(r["reasoning"]), "finish_reason": r["finish_reason"],
                "completion_tokens": r["usage"].get("completion_tokens"),
                "ids_sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16], "s": r["s"],
                "others_running": busy}

    for i, row in enumerate(pmap(one, prompts, a.workers)):
        append_jsonl(path, row)
        if (i + 1) % 40 == 0:
            print(f"  ml {i + 1}/{len(prompts)}, {time.time() - t0:.0f} s", flush=True)
    print(f"ML captured {len(prompts)} -> {path}")
    return 0


# ------------------------------------------------------------------ judge

def judge(a) -> int:
    prompts = {p["id"]: p for p in json.loads(PROMPTS.read_text(encoding="utf-8"))}
    ref = {r["id"]: r for r in jsonl(Path(a.ref) / "ml.jsonl")}
    cand = {r["id"]: r for r in jsonl(Path(a.cand) / "ml.jsonl")}
    if set(ref) != set(cand):
        print(f"INVALID: item sets differ ({len(ref)} vs {len(cand)})")
        return 2
    ids = [i for i in prompts if i in ref]
    c = Client(a.url, a.model, timeout=600)
    out = prepare_out(a.out)
    write_manifest(out, c, "ml_judge.judge", vars(a), files=[PROMPTS, Path(a.ref) / "ml.jsonl", Path(a.cand) / "ml.jsonl"])
    rng = random.Random(a.seed)
    first = {i: rng.choice(("ref_first", "cand_first")) for i in ids}

    def ask(i: str, order: str) -> dict:
        p, rr, cc = prompts[i], ref[i]["content"], cand[i]["content"]
        x, y = (rr, cc) if order == "ref_first" else (cc, rr)
        msg = JUDGE_TMPL.format(lang=LANG_NAME[p["lang"]], prompt=p["prompt"], a=x, b=y)
        r = c.chat([{"role": "system", "content": "You are a strict, impartial evaluator."},
                    {"role": "user", "content": msg}], max_tokens=16, temperature=0, **OFF)
        m = VERDICT.search(r["content"])
        v = m.group(1).upper() if m else None
        side = {"A": "ref" if order == "ref_first" else "cand", "B": "cand" if order == "ref_first" else "ref",
                "TIE": "tie"}.get(v)
        return {"order": order, "raw": r["content"][:80], "verdict": v, "pick": side}

    jobs = [(i, o) for i in ids if ref[i]["content"] != cand[i]["content"]
            for o in ((first[i], "cand_first" if first[i] == "ref_first" else "ref_first"))]
    res = pmap(lambda j: (j[0], ask(*j)), jobs, a.workers)
    by = {}
    for i, r in res:
        by.setdefault(i, []).append(r)
    jpath = out / "judge.jsonl"
    jpath.write_text("")
    items = []
    for i in ids:
        p = prompts[i]
        if ref[i]["content"] == cand[i]["content"]:
            outcome, calls = "identical", []
        else:
            calls = by[i]
            picks = {x["pick"] for x in calls}
            outcome = ({"cand": "win", "ref": "loss", "tie": "tie"}[calls[0]["pick"]]
                       if len(picks) == 1 and None not in picks else "inconsistent")
        row = {"id": i, "lang": p["lang"], "task": p["task"], "outcome": outcome, "calls": calls,
               # reasoning answers are mostly numbers and math: not scored for language
               "langid_ref": p["task"] == "reason" or lang_id_ok(ref[i]["content"], p["lang"]),
               "langid_cand": p["task"] == "reason" or lang_id_ok(cand[i]["content"], p["lang"]),
               "len_ref": len(ref[i]["content"]), "len_cand": len(cand[i]["content"]),
               "rep4_ref": round(rep4(ref[i]["content"], p["lang"]), 4),
               "rep4_cand": round(rep4(cand[i]["content"], p["lang"]), 4),
               "trunc_ref": ref[i]["finish_reason"] == "length", "trunc_cand": cand[i]["finish_reason"] == "length"}
        if "answer" in p:
            row["correct_ref"] = answer_ok(ref[i]["content"], p["answer"])
            row["correct_cand"] = answer_ok(cand[i]["content"], p["answer"])
        append_jsonl(jpath, row)
        items.append(row)

    def agg(rows: list[dict]) -> dict:
        n = lambda k: sum(1 for r in rows if r["outcome"] == k)
        w, t, l = n("win"), n("tie"), n("loss")
        reason = [r for r in rows if "correct_ref" in r]
        pos_first = sum(1 for r in rows for x in r["calls"] if x["verdict"] == "A")
        calls = sum(len(r["calls"]) for r in rows)
        return {"items": len(rows), "win": w, "tie": t, "loss": l, "inconsistent": n("inconsistent"),
                "identical": n("identical"), "sign": sign_test(w, l),
                "langid_fail_ref": sum(not r["langid_ref"] for r in rows),
                "langid_fail_cand": sum(not r["langid_cand"] for r in rows),
                "len_ratio": round(sum(r["len_cand"] for r in rows) / max(1, sum(r["len_ref"] for r in rows)), 4),
                "rep4_ref": round(sum(r["rep4_ref"] for r in rows) / max(1, len(rows)), 4),
                "rep4_cand": round(sum(r["rep4_cand"] for r in rows) / max(1, len(rows)), 4),
                "trunc_ref": sum(r["trunc_ref"] for r in rows), "trunc_cand": sum(r["trunc_cand"] for r in rows),
                "reason_ref": f"{sum(r['correct_ref'] for r in reason)}/{len(reason)}",
                "reason_cand": f"{sum(r['correct_cand'] for r in reason)}/{len(reason)}",
                "judge_calls": calls, "judge_picked_A": pos_first}

    per_lang = {L: agg([r for r in items if r["lang"] == L]) for L in LANGS if any(r["lang"] == L for r in items)}
    per_task = {T: agg([r for r in items if r["task"] == T]) for T in sorted({r["task"] for r in items})}
    overall = agg(items)
    fails = []
    for L, s in per_lang.items():
        if s["loss"] > s["win"] and s["sign"]["p_against_cand"] < 0.05:
            fails.append(f"{L}: significant loss ({s['win']}W/{s['loss']}L, p {s['sign']['p_against_cand']})")
        if s["langid_fail_cand"] > s["langid_fail_ref"]:
            fails.append(f"{L}: language-ID failures {s['langid_fail_cand']} > ref {s['langid_fail_ref']}")
    decided = overall["win"] + overall["loss"]
    if overall["win"] < overall["loss"] - 0.1 * decided:
        fails.append(f"overall wins {overall['win']} < losses {overall['loss']} - 10% of {decided}")
    if overall["langid_fail_cand"] > overall["langid_fail_ref"]:
        fails.append("overall language-ID failures above ref")
    summary = {"gate": "ML-judge", "ref": str(a.ref), "cand": str(a.cand), "seed": a.seed,
               "overall": overall, "per_lang": per_lang, "per_task": per_task,
               "verdict": "PASS" if not fails else "FAIL", "fails": fails}
    write_json(out / "summary.json", summary)
    if a.json:
        write_json(Path(a.json), summary)
    print(f"{'lang':6} {'W':>3} {'T':>3} {'L':>3} {'inc':>3} {'idn':>3} {'p(two)':>7} {'p(vs)':>7} "
          f"{'LID r/c':>8} {'len':>6} {'rep4 r/c':>13} {'reason r c':>11}")
    for k, s in [*per_lang.items(), ("ALL", overall)]:
        print(f"{k:6} {s['win']:3} {s['tie']:3} {s['loss']:3} {s['inconsistent']:3} {s['identical']:3} "
              f"{s['sign']['p_two_sided']:7.3f} {s['sign']['p_against_cand']:7.3f} "
              f"{s['langid_fail_ref']:>3}/{s['langid_fail_cand']:<4} {s['len_ratio']:6.3f} "
              f"{s['rep4_ref']:.4f}/{s['rep4_cand']:.4f} {s['reason_ref']:>5} {s['reason_cand']:<5}")
    print("VERDICT", summary["verdict"], "; ".join(fails))
    return 0 if not fails else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("collect", "judge"):
        s = sub.add_parser(name)
        s.add_argument("--url", default=DEFAULT_URL)
        s.add_argument("--model", default=None)
        s.add_argument("--out", required=True)
    s = sub.choices["collect"]
    s.add_argument("--max-tokens", type=int, default=512)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--limit", type=int, default=0, help="first N prompts only (smoke runs)")
    s = sub.choices["judge"]
    s.add_argument("--ref", required=True)
    s.add_argument("--cand", required=True)
    s.add_argument("--json", default=None)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    return collect(a) if a.cmd == "collect" else judge(a)


if __name__ == "__main__":
    raise SystemExit(main())
