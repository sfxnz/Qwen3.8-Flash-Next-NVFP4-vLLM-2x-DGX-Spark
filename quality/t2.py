#!/usr/bin/env python3
"""T2 task gates (plan 1.3), paired item by item between two configs.

Tasks (all greedy, thinking off unless noted):
  gsm8k   GSM8K-250 (test ids pinned in data/t2_ids.json), "Answer: <number>"
  ifeval  IFEval-120 (strict prompt-level, quality/ifeval.py subset checker)
  tools   30 tool prompts (data/tools.json) x non-streamed + streamed = 60 calls;
          correct = exactly one call, right name, arguments exactly equal
  json    30 JSON-schema prompts (response_format json_schema, strict);
          correct = parses and validates against the schema
  rep4    64 open prompts, 512 tokens; value = repeated word-4-gram fraction
  effort  reasoning_effort unset / none / low / medium / xhigh: HTTP 200,
          non-empty content, finish stop; none gives empty reasoning

Datasets are read from $QWEN38_EVALS_DIR/datasets (default
~/projects/data/qwen38-evals/datasets): gsm8k/test.jsonl (openai/grade-school-math
@3101c7d) and ifeval/input_data.jsonl (google-research @26d8ccd); sha256 checked.

Default is c=1. On the pinned image, co-prefilled requests corrupt (F01):
live 2026-09-29, two rep4 prompts sent together gave one stream of
"Register Register ..." (repeat-4gram 0.996, 0.0 alone), and a json_schema
request at c=2 died with HTTP 500 "grammar rejected tokens". Use --workers >1
only on a pin with F01 fixed.

  python3 quality/t2.py --out RUN_DIR [--tasks gsm8k,tools] [--limit 3] [--workers 1]
Compare: `python3 quality/score.py t2 REF_DIR CAND_DIR`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ifeval  # noqa: E402
from qlib import (DATA, DEFAULT_URL, EVALS_DIR, OFF, Client, HTTPFailure, append_jsonl, jsonl,  # noqa: E402
                    pmap, prepare_out, write_json, write_manifest)

IDS = DATA / "t2_ids.json"
TOOLS = DATA / "tools.json"
TASKS = ("gsm8k", "ifeval", "tools", "json", "rep4", "effort")
NUM = r"-?\d[\d,]*(?:\.\d+)?"
GSM8K_SUFFIX = "\n\nSolve the problem step by step, then give the final answer on its own line as 'Answer: <number>'."


# ------------------------------------------------------------------ GSM8K

def gsm8k_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def gsm8k_extract(text: str) -> str | None:
    """Number after the last 'Answer:', else the last number in the text."""
    hits = re.findall(r"answer\s*(?:is)?\s*[:：]?\s*\**\s*\$?\s*(" + NUM + ")", text, re.I)
    hits = hits or re.findall(NUM, text)
    return hits[-1].replace(",", "").rstrip(".") if hits else None


def gsm8k_correct(text: str, gold: str) -> bool:
    got = gsm8k_extract(text)
    try:
        return got is not None and abs(float(got) - float(gold)) < 1e-6
    except ValueError:
        return False


def dataset(name: str, file: str, sha: str) -> list[dict]:
    path = EVALS_DIR / "datasets" / name / file
    if not path.exists():
        raise SystemExit(f"missing {path}; see the module docstring for the pinned source")
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != sha:
        raise SystemExit(f"{path}: sha256 {got} != pinned {sha}")
    return jsonl(path)


# ------------------------------------------------------------------ tools

def values_equal(got, want) -> bool:
    """Exact: same JSON type family, same keys, same values (1 == 1.0 for numbers)."""
    if isinstance(want, bool) or isinstance(got, bool):
        return got is want
    if isinstance(want, (int, float)):
        return isinstance(got, (int, float)) and float(got) == float(want)
    if isinstance(want, dict):
        return isinstance(got, dict) and set(got) == set(want) and all(values_equal(got[k], want[k]) for k in want)
    if isinstance(want, list):
        return isinstance(got, list) and len(got) == len(want) and all(map(values_equal, got, want))
    return got == want


def judge_tool(expect: dict, out: dict) -> dict:
    calls = out.get("tool_calls") or []
    try:
        args = json.loads(calls[0]["arguments"] or "{}") if calls else None
    except ValueError:
        args = None
    ok = len(calls) == 1 and calls[0]["name"] == expect["name"] and values_equal(args, expect["args"])
    return {"correct": int(ok), "calls": calls[:2], "content": (out.get("content") or "")[:120]}


# ------------------------------------------------------------------ JSON schema

def _obj(props: dict, required=None) -> dict:
    return {"type": "object", "properties": props, "required": list(required or props), "additionalProperties": False}


S, I, N, B = {"type": "string"}, {"type": "integer"}, {"type": "number"}, {"type": "boolean"}
SCHEMAS = {
    "person": _obj({"name": S, "age": I, "email": S, "languages": {"type": "array", "items": S}}),
    "order": _obj({"order_id": S, "items": {"type": "array", "items": _obj({"sku": S, "qty": I})}, "total": N, "paid": B}),
    "event": _obj({"title": S, "date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}, "attendees": I, "online": B}),
    "weather": _obj({"city": S, "temp_c": N, "conditions": {"type": "string", "enum": ["sunny", "cloudy", "rain", "snow"]}}),
    "book": _obj({"title": S, "author": S, "year": I, "genres": {"type": "array", "items": S}}),
    "sentiment": _obj({"label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                       "confidence": {"type": "number", "minimum": 0, "maximum": 1}}),
}
JSON_PROMPTS = [
    ("person", "Ana Silva is 34, her email is ana@example.com and she speaks Portuguese and English."),
    ("person", "Record: Kenji Mori, 58 years old, kenji.mori@example.jp, speaks Japanese."),
    ("person", "Liam O'Brien (27) can be reached at liam@example.ie. Languages: English, Irish, French."),
    ("person", "Our new hire Fatima Zahra, age 41, email fz@example.ma, speaks Arabic, French and Spanish."),
    ("person", "Extract the person: 'Name - Tomasz Nowak; Age - 19; Mail - tn@example.pl; Speaks - Polish'."),
    ("order", "Order A-1001: 2 of SKU PEN-01 and 1 of SKU INK-09, total 14.50, already paid."),
    ("order", "Order B-77 has 5 units of CUP-3; the total is 42 and it has not been paid."),
    ("order", "Invoice for order Z9: SKU BAG-2 x1, SKU TAG-8 x10. Total 88.2. Paid: yes."),
    ("order", "Order 2026-555 contains 3 x LAMP-7. Total due 129.99, unpaid."),
    ("order", "Build the order record: id Q-1, items KEY-1 qty 4 and KEY-2 qty 4, total 20, paid true."),
    ("event", "Team retro on 2026-11-03 with 8 attendees, held online."),
    ("event", "The annual gala is on 2027-02-14 in person, expecting 250 guests."),
    ("event", "Workshop 'Intro to Rust', 2026-10-21, 30 people, remote via video call."),
    ("event", "Board meeting: 2026-12-01, 9 attendees, on site."),
    ("event", "Hackathon kickoff 2027-01-09, 120 participants, in the main hall."),
    ("weather", "It is 21.5 degrees and sunny in Lisbon."),
    ("weather", "Oslo: minus 4 C with snow."),
    ("weather", "Weather report for Wellington: 13 degrees, cloudy."),
    ("weather", "Heavy rain in Mumbai today, 29 C."),
    ("weather", "Cairo is clear and sunny at 35 degrees Celsius."),
    ("book", "Dune by Frank Herbert, 1965, science fiction and adventure."),
    ("book", "Pride and Prejudice, written by Jane Austen in 1813; genres: romance, satire."),
    ("book", "The Hobbit (J. R. R. Tolkien, 1937) is a fantasy novel for children."),
    ("book", "Gabriel Garcia Marquez published One Hundred Years of Solitude in 1967: magical realism."),
    ("book", "Catalogue this: 'Neuromancer', William Gibson, 1984, cyberpunk, science fiction."),
    ("sentiment", "Classify: 'The update fixed every crash I had, brilliant work!'"),
    ("sentiment", "Classify: 'Delivery was late and the box was crushed.'"),
    ("sentiment", "Classify: 'The meeting is at 3 pm in room 4.'"),
    ("sentiment", "Classify: 'I expected more, but it is fine I guess.'"),
    ("sentiment", "Classify: 'Absolutely terrible customer service, never again.'"),
]


def validate(value, schema: dict, path: str = "$") -> list[str]:
    """Subset JSON-schema validator: type, properties, required, additionalProperties,
    items, enum, minimum, maximum, pattern."""
    t = schema.get("type")
    kinds = {"object": dict, "array": list, "string": str, "boolean": bool}
    errs = []
    if t == "integer":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif t == "number":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        ok = t is None or isinstance(value, kinds[t])
    if not ok:
        return [f"{path}: expected {t}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in enum")
    if "minimum" in schema and value < schema["minimum"]:
        errs.append(f"{path}: below minimum")
    if "maximum" in schema and value > schema["maximum"]:
        errs.append(f"{path}: above maximum")
    if "pattern" in schema and not re.search(schema["pattern"], value):
        errs.append(f"{path}: pattern mismatch")
    if t == "object":
        props = schema.get("properties", {})
        errs += [f"{path}.{k}: missing" for k in schema.get("required", []) if k not in value]
        if schema.get("additionalProperties") is False:
            errs += [f"{path}.{k}: not allowed" for k in value if k not in props]
        for k, v in value.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}")
    if t == "array" and "items" in schema:
        for i, v in enumerate(value):
            errs += validate(v, schema["items"], f"{path}[{i}]")
    return errs


# ------------------------------------------------------------------ rep4

REP_TOPICS = ["a lighthouse keeper", "a lost umbrella", "the first day of school", "a city at night", "a chess match",
              "a desert road trip", "a bakery at dawn", "an old library", "a mountain storm", "a robot gardener",
              "a river ferry", "a winter market", "a broken clock", "a jazz club", "a science fair", "a quiet harbour"]
REP_FORMS = ["Write a short story about {}.", "Describe {} in vivid detail.",
             "Write a diary entry from the point of view of {}.", "Write a news report about {}."]
REP_PROMPTS = [f.format(t) for t in REP_TOPICS for f in REP_FORMS]


def repeat_4gram(text: str) -> float:
    w = text.split()
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


# ------------------------------------------------------------------ effort

EFFORTS = [None, "none", "low", "medium", "xhigh"]
EFFORT_PROMPT = "What is 17 + 25? Reply with just the number."


# ------------------------------------------------------------------ jobs

def jobs(task: str, ids: dict, tools: dict) -> list[dict]:
    if task == "gsm8k":
        rows = dataset("gsm8k", "test.jsonl", ids["gsm8k"]["sha256"])
        return [{"task": task, "id": i, "prompt": rows[i]["question"] + GSM8K_SUFFIX,
                 "gold": gsm8k_gold(rows[i]["answer"])} for i in ids["gsm8k"]["ids"]]
    if task == "ifeval":
        rows = {r["key"]: r for r in dataset("ifeval", "input_data.jsonl", ids["ifeval"]["sha256"])}
        return [{"task": task, "id": k, "prompt": rows[k]["prompt"], "item": rows[k]} for k in ids["ifeval"]["keys"]]
    if task == "tools":
        items = {x["id"]: x for x in tools["items"]}
        return [{"task": task, "id": f"{i}.{mode}", "item": items[i], "stream": mode == "stream"}
                for i in ids["tools"]["ids"] for mode in ("nonstream", "stream")]
    if task == "json":
        return [{"task": task, "id": f"j{k:02d}.{s}", "schema": s, "prompt": p} for k, (s, p) in enumerate(JSON_PROMPTS)]
    if task == "rep4":
        return [{"task": task, "id": f"r{k:02d}", "prompt": p} for k, p in enumerate(REP_PROMPTS)]
    return [{"task": task, "id": str(e), "effort": e} for e in EFFORTS]


def run_job(c: Client, j: dict, tools: dict) -> dict:
    t = j["task"]
    try:
        if t == "gsm8k":
            r = c.chat(j["prompt"], max_tokens=1024, temperature=0, **OFF)
            return {"correct": int(gsm8k_correct(r["content"], j["gold"])), "got": gsm8k_extract(r["content"]),
                    "gold": j["gold"], "finish_reason": r["finish_reason"]}
        if t == "ifeval":
            r = c.chat(j["prompt"], max_tokens=2048, temperature=0, **OFF)
            s = ifeval.score(j["item"], r["content"])
            return {**s, "finish_reason": r["finish_reason"]}
        if t == "tools":
            it = j["item"]
            r = c.chat(it["prompt"], stream=j["stream"], tools=[tools["tools"][n] for n in it["tools"]],
                       tool_choice="auto", max_tokens=512, temperature=0, **OFF)
            return {**judge_tool(it["expect"], r), "finish_reason": r["finish_reason"]}
        if t == "json":
            schema = SCHEMAS[j["schema"]]
            r = c.chat(f"Extract the data as JSON. {j['prompt']}", max_tokens=512, temperature=0,
                       response_format={"type": "json_schema",
                                        "json_schema": {"name": j["schema"], "schema": schema, "strict": True}}, **OFF)
            try:
                errs = validate(json.loads(r["content"]), schema)
            except ValueError as exc:
                errs = [f"not JSON: {exc}"]
            return {"correct": int(not errs), "errors": errs[:5], "content": r["content"][:200]}
        if t == "rep4":
            r = c.chat(j["prompt"], max_tokens=512, temperature=0, **OFF)
            return {"correct": 1, "value": round(repeat_4gram(r["content"]), 6),
                    "completion_tokens": r["usage"].get("completion_tokens")}
        body = {"max_tokens": 4096, "temperature": 0}
        if j["effort"] is not None:
            body["reasoning_effort"] = j["effort"]
        r = c.chat(EFFORT_PROMPT, **body)
        ok = bool(r["content"].strip()) and r["finish_reason"] == "stop" and "42" in r["content"]
        if j["effort"] == "none":
            ok = ok and not r["reasoning"].strip()
        return {"correct": int(ok), "content": r["content"][:80], "reasoning_chars": len(r["reasoning"]),
                "finish_reason": r["finish_reason"]}
    except (HTTPFailure, OSError) as exc:  # OSError: timeout / reset; one item, not the run
        return {"correct": 0, "value": 1.0, "error": str(exc)[:300]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--limit", type=int, default=0, help="first N items per task (smoke runs)")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent requests; >1 only on a pin with F01 fixed")
    a = ap.parse_args(argv)
    ids, tools = json.loads(IDS.read_text()), json.loads(TOOLS.read_text())
    c = Client(a.url, a.model, timeout=1800)
    out = prepare_out(a.out)
    write_manifest(out, c, "t2", vars(a), files=[IDS, TOOLS])
    path = out / "t2.jsonl"
    path.write_text("")
    summary, t0 = {}, time.time()
    for task in [t for t in a.tasks.split(",") if t]:
        js = jobs(task, ids, tools)
        js = js[:a.limit] if a.limit else js
        res = pmap(lambda j: {"task": j["task"], "id": j["id"], **run_job(c, j, tools)}, js, a.workers)
        for r in res:
            append_jsonl(path, r)
        n_ok = sum(r["correct"] for r in res)
        summary[task] = {"n": len(res), "correct": n_ok, "acc": round(n_ok / max(1, len(res)), 4),
                         "errors": sum("error" in r for r in res)}
        if task == "rep4":
            summary[task]["mean_repeat_4gram"] = round(sum(r.get("value", 1.0) for r in res) / max(1, len(res)), 6)
        print(f"  {task}: {json.dumps(summary[task])} ({time.time() - t0:.0f} s)", flush=True)
    write_json(out / "t2.json", summary)
    print(f"T2 captured -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
