#!/usr/bin/env python3
"""Rebuild quality/data/t1_corpus.jsonl, the frozen T1 teacher-forced corpus.

You never need to run this to use T1: it records how the committed corpus was
cut. It needs a live serve only for /tokenize (CPU on the API server; no GPU
work), so pieces are cut with the served tokenizer:

  python3 quality/data/build_t1_corpus.py --url http://127.0.0.1:8000

Every row is one chat conversation whose final assistant message is the
teacher-forced text. Assistant pieces are at most PIECE_TOKENS tokens, so the
per-request prompt_logprobs transient stays near 1 GB (1024 x 248320 x 4 B).

Domains and sources (licenses in SOURCES.md):
  prose, zh, es, de, ja, multi, code, math   texts from the GLM-5.3 frozen corpus
                                             (Gutenberg, CPython/Go/SQLite/lodash/Rust,
                                             GSM8K/MATH train), re-cut to Qwen tokens
  long    8 English Gutenberg books x 16 consecutive pieces (a 16k-token document each);
          collect_t1.py --long-whole sends each as one 16k-token message
  json    16 seeded synthetic JSON documents
  tools   16 tool-call transcripts (call, tool result, answer) from data/tools.json
  image   16 image prompts: vision_probes images + a fixed description
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from qlib import DEFAULT_URL, Client  # noqa: E402

PIECE_TOKENS = 1000
MIN_TAIL = 128
GLM_CORPUS = Path("/home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark/.claude/"
                  "worktrees/wf_d8bf4264-311-4/quality/data/corpus.jsonl")
GUTENBERG_CACHE = Path.home() / "projects/data/glm53-evals/corpus-src"
LONG_BOOKS = [(1342, "Pride and Prejudice"), (2701, "Moby Dick"), (1661, "The Adventures of Sherlock Holmes"),
              (84, "Frankenstein"), (98, "A Tale of Two Cities"), (345, "Dracula"), (205, "Walden"),
              (145, "Middlemarch")]
LONG_PIECES = 16
USER = {"prose": "Write the next part of the book.", "multi": "Write the next part of the book.",
        "long": "Write the next part of the book.", "code": "Show me the source code.",
        "math": "Solve the problem and show your working.", "json": "Return the records as JSON."}


class Cutter:
    def __init__(self, c: Client):
        self.c = c

    def count(self, text: str) -> int:
        return self.c.tokenize_count(text)

    def pieces(self, text: str, limit: int = PIECE_TOKENS) -> list[tuple[str, int]]:
        """Split text into consecutive pieces of <= limit tokens at line/word boundaries."""
        out, rest = [], text.strip()
        ratio = max(1.0, len(rest[:20000]) / max(1, self.count(rest[:20000])))
        while rest:
            n = self.count(rest) if len(rest) < limit * ratio * 1.5 else limit + 1
            if n <= limit:
                out.append((rest, n))
                break
            size = int(limit * ratio * 0.97)
            while True:
                cut = rest[:size]
                pos = max(cut.rfind("\n"), cut.rfind(" "), cut.rfind("。"))
                cut = cut[:pos + 1] if pos > len(cut) // 2 else cut
                n = self.count(cut.rstrip())
                if n <= limit:
                    break
                size = int(size * 0.93)
            out.append((cut.rstrip(), n))
            rest = rest[len(cut):].lstrip()
        return out


def gutenberg_body(raw: str) -> str:
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", raw)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", raw)
    return raw[start.end() if start else 0:end.start() if end else len(raw)]


def unwrap(text: str) -> str:
    paras = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    return "\n\n".join(" ".join(ln.strip() for ln in p.split("\n") if ln.strip()) for p in paras if p.strip())


def chat_row(rid: str, domain: str, user, answer: str, **meta) -> dict:
    return {"id": rid, "domain": domain, **meta,
            "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]}


def glm_rows(cut: Cutter, path: Path) -> list[dict]:
    rows = []
    for d in (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()):
        if d["domain"] == "chat":  # GLM chat shape; not transferable
            continue
        dom = d["domain"] if d["domain"] != "multilingual" else (
            d["lang"] if d["lang"] in ("zh", "es", "de", "ja") else "multi")
        user = USER["multi" if dom in ("zh", "es", "de", "ja") else dom]
        for k, (text, n) in enumerate(cut.pieces(d["text"])):
            if n < MIN_TAIL and k:
                continue
            rows.append(chat_row(f"{d['id']}.p{k}", dom, user, text, source=d["source"],
                                 license=d["license"], tokens=n))
    return rows


def long_rows(cut: Cutter) -> list[dict]:
    rows = []
    for num, title in LONG_BOOKS:
        url = f"https://www.gutenberg.org/cache/epub/{num}/pg{num}.txt"
        raw = (GUTENBERG_CACHE / urllib.parse.quote(url, safe="")).read_text(encoding="utf-8-sig")
        body = unwrap(gutenberg_body(raw))
        start = body.find("\n\n", int(len(body) * 0.10)) + 2
        text = body[start:start + 90000]
        pieces = cut.pieces(text)[:LONG_PIECES]
        if len(pieces) < LONG_PIECES:
            raise SystemExit(f"book {num}: only {len(pieces)} pieces")
        for k, (t, n) in enumerate(pieces):
            rows.append(chat_row(f"long.{num}.p{k:02d}", "long", USER["long"], t, doc=f"long.{num}", piece=k,
                                 source=f"{url} ({title}), from 10% of the body", license="Public domain (US)",
                                 tokens=n))
    return rows


_WORDS = ("amber basalt cobalt delta ember fjord garnet harbor indigo juniper kelp lumen marble nectar onyx "
          "prism quartz raven sable tundra umber vesper willow xenon yarrow zephyr").split()


def json_rows(cut: Cutter) -> list[dict]:
    rows = []
    for k in range(16):
        rng = random.Random(9000 + k)
        if k % 2 == 0:
            obj = {"inventory": [
                {"id": rng.randint(10000, 99999), "sku": f"{rng.choice(_WORDS)[:3].upper()}-{rng.randint(100, 999)}",
                 "name": f"{rng.choice(_WORDS)} {rng.choice(_WORDS)}", "qty": rng.randint(0, 500),
                 "price": round(rng.uniform(1, 400), 2), "active": rng.random() > 0.3,
                 "tags": rng.sample(_WORDS, rng.randint(1, 3)),
                 "dims_cm": {"w": rng.randint(1, 90), "h": rng.randint(1, 90), "d": rng.randint(1, 90)},
                 "updated": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:"
                            f"{rng.randint(0, 59):02d}:00Z"} for _ in range(rng.randint(10, 14))]}
        else:
            obj = {"service": rng.choice(_WORDS), "version": f"{rng.randint(1, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 9)}",
                   "replicas": rng.randint(1, 12), "env": {w.upper(): str(rng.randint(0, 9999)) for w in rng.sample(_WORDS, 6)},
                   "routes": [{"path": f"/{rng.choice(_WORDS)}/{rng.choice(_WORDS)}", "method": rng.choice(["GET", "POST", "PUT", "DELETE"]),
                               "timeout_ms": rng.choice([250, 500, 1000, 3000]), "auth": rng.random() > 0.5}
                              for _ in range(rng.randint(8, 12))],
                   "limits": {"cpu": f"{rng.randint(1, 16)}", "memory_mib": rng.choice([512, 1024, 4096, 8192])}}
        text = json.dumps(obj, indent=2)
        (t, n), = cut.pieces(text)[:1]
        rows.append(chat_row(f"json.{k:02d}", "json", USER["json"], t, source="synthetic, seed 9000+k",
                             license="written for this repo", tokens=n))
    return rows


TOOL_RESULTS = {"get_weather": {"temp_c": 14, "conditions": "partly cloudy"}, "convert_currency": {"converted": 231.4},
                "create_event": {"event_id": "evt_1042", "status": "created"},
                "search_flights": {"flights": [{"flight": "NZ103", "depart": "07:05", "price": 312}]},
                "send_email": {"status": "sent"}, "set_thermostat": {"status": "ok"},
                "lookup_stock": {"price": 187.21, "currency": "USD"}, "add_todo": {"todo_id": 77},
                "run_sql": {"rows": [[1204]]}, "translate_text": {"translation": "(translated text)"},
                "get_route": {"distance_km": 142, "duration_min": 118},
                "book_restaurant": {"confirmation": "BK-5521"}, "compute_loan": {"monthly_payment": 2770.73},
                "resize_image": {"status": "ok", "bytes": 48213}}


def tool_rows(cut: Cutter, tools_path: Path) -> list[dict]:
    d = json.loads(tools_path.read_text())
    rows = []
    for it in [x for x in d["items"] if not x["ignore"]][:16]:
        name, args = it["expect"]["name"], it["expect"]["args"]
        result = TOOL_RESULTS[name]
        answer = (f"I called {name} with {', '.join(f'{k}={v}' for k, v in args.items())}. "
                  f"The tool returned {json.dumps(result)}, so the request is complete.")
        msgs = [{"role": "user", "content": it["prompt"]},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "call_0", "type": "function",
                                 "function": {"name": name, "arguments": json.dumps(args)}}]},
                {"role": "tool", "tool_call_id": "call_0", "content": json.dumps(result)},
                {"role": "assistant", "content": answer}]
        rows.append({"id": f"tools.{it['id']}", "domain": "tools", "messages": msgs,
                     "tools": [d["tools"][t] for t in it["tools"]], "source": "data/tools.json",
                     "license": "written for this repo"})
    return rows


IMAGE_ITEMS = [
    ("solid_red", "Describe the image.", "The image is a single solid red square with nothing else in it."),
    ("solid_green", "Describe the image.", "The image is a single solid green square with nothing else in it."),
    ("solid_blue", "Describe the image.", "The image is a single solid blue square with nothing else in it."),
    ("solid_yellow", "Describe the image.", "The image is a single solid yellow square with nothing else in it."),
    ("quadrants", "Describe the image.", "The image is split into four equal quadrants: red at the top left, "
     "green at the top right, blue at the bottom left and yellow at the bottom right."),
    ("grid_wide", "Describe the image.", "The image is a wide grid of 2 rows and 8 columns of white squares with "
     "grey lines. The square in row 1, column 6 is red, and the square in row 2, column 3 is blue."),
    ("grid_tall", "Describe the image.", "The image is a tall grid of 8 rows and 2 columns of white squares with "
     "grey lines. The square in row 7 of the left column is green."),
    ("digits_407218", "Describe the image.", "The image shows the number 407218 in black block digits on a white background."),
    ("digits_9315", "Describe the image.", "The image shows the number 9315 in large black block digits on a white background."),
    ("bars", "Describe the image.", "The image is a bar chart with five bars on a black axis. From left to right the "
     "bars are red, green, blue, yellow and black. The green bar is the tallest and the yellow bar is the shortest."),
    ("circles_3", "Describe the image.", "The image shows three black circles in a row on a white background."),
    ("circles_5", "Describe the image.", "The image shows five black circles in a row on a white background."),
    ("grid_wide", "Which column is the red square in?", "The red square is in column 6 of the top row."),
    ("bars", "Which bar is the tallest?", "The green bar, the second from the left, is the tallest."),
    ("digits_407218", "What number is written in the image?", "The number written in the image is 407218."),
    ("quadrants", "What colour is the bottom-left quadrant?", "The bottom-left quadrant is blue."),
]


def image_rows() -> list[dict]:
    return [chat_row(f"image.{k:02d}.{name}", "image",
                     [{"type": "image_ref", "name": name}, {"type": "text", "text": q}], a,
                     source="quality/vision_probes/probes.py", license="written for this repo")
            for k, (name, q, a) in enumerate(IMAGE_ITEMS)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--glm-corpus", type=Path, default=GLM_CORPUS)
    ap.add_argument("--out", type=Path, default=HERE / "t1_corpus.jsonl")
    a = ap.parse_args()
    cut = Cutter(Client(a.url))
    rows = (glm_rows(cut, a.glm_corpus) + long_rows(cut) + json_rows(cut)
            + tool_rows(cut, HERE / "tools.json") + image_rows())
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    by = {}
    for r in rows:
        by.setdefault(r["domain"], [0, 0])
        by[r["domain"]][0] += 1
        by[r["domain"]][1] += r.get("tokens", 0)
    print(json.dumps({"rows": len(rows), "by_domain": by}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
