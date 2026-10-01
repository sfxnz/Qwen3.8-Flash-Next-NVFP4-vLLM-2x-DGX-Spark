#!/usr/bin/env python3
"""T3 long-context needles (plan section 1.3): 3 depths x 2 needles per length.

Each (length, depth) cell is one synthetic document with two needles (two
names, two vault codes) in adjacent paragraphs at that depth. The two
questions go as two requests (two prefills: on this pin the second does not
hit the prefix cache, F20). Thinking off, greedy.

Default lengths 4k, 16k, 64k and 128k give 4 x 3 x 2 = 24 needles.
--with-250k adds 250,000 tokens (the served window is 262,144).
Pass: every needle at <= 128k found, and at most 2 misses overall.

  python3 quality/t3_needles.py --out RUN_DIR [--lengths 4096,16384] [--with-250k]

The filler is seeded prose with no per-run salt, so a miss is reproducible.
Watch `free -h` on both nodes during the 128k and 250k cells (AGENTS.md).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qlib import (DEFAULT_URL, OFF, Client, append_jsonl, prepare_out,  # noqa: E402
                    write_json, write_manifest)

DEPTHS = (0.1, 0.5, 0.9)
LENGTHS = (4096, 16384, 65536, 131072)
LONG = 250000
GATE_MAX = 131072

_ADJ = ("amber brittle copper dusky eager feral gilded hollow ivory jagged knotted languid mossy narrow "
        "ochre pallid quiet russet silent tawny umber velvet wary woven yellowed zealous ashen bleak").split()
_NOUN = ("lantern orchard ledger kettle bridge harbor quarry meadow chimney anvil cellar compass lighthouse "
         "granary loom mill parcel quill ridge saddle tannery thicket vault wagon well workshop barge").split()
_VERB = ("mended carried weighed painted counted guarded sealed traded repaired measured polished hid copied "
         "lifted buried sketched hauled borrowed wrapped tended").split()
_ADV = ("slowly", "quietly", "twice", "again", "carefully", "briskly", "at dawn", "before supper")
_SYL = "ka lo mir ven tas ob rune fel dra is quo zen hal por ith gam sel bri vo tur ney ash cor".split()


def _name(rng: random.Random) -> str:
    return " ".join("".join(rng.choice(_SYL) for _ in range(rng.randint(2, 3))).capitalize() for _ in range(2))


def _sentence(rng: random.Random) -> str:
    a, n, v = rng.choice(_ADJ), rng.choice(_NOUN), rng.choice(_VERB)
    t = rng.randrange(3)
    if t == 0:
        return f"{_name(rng)} {v} the {a} {n} {rng.choice(_ADV)}."
    if t == 1:
        return f"In {_name(rng).split()[0]}, the {a} {n} was {v} by {rng.randint(2, 97)} workers near the {rng.choice(_NOUN)}."
    return f"The ledger lists {rng.randint(3, 999)} {n}s, each {a}, and {_name(rng)} {v} them {rng.choice(_ADV)}."


def filler(seed: int, count: int) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(_sentence(rng) for _ in range(rng.randint(4, 7))) for _ in range(count)]


def needle_code(seed: int) -> tuple[str, str]:
    rng = random.Random(seed * 7919 + 1)
    letters = "BCDFGHJKLMNPQRSTVWXZ"
    code = (f"{''.join(rng.choice(letters) for _ in range(3))}-{rng.randint(1000, 9999)}-"
            f"{''.join(rng.choice(letters) for _ in range(2))}")
    return _name(rng), code


def needle_doc(paras: list[str], depth: float, needles: list[tuple[str, str]], seed: int) -> str:
    idx = max(1, min(len(paras) - 1, int(len(paras) * depth)))
    notes = [f"The vault code assigned to {n} is {c}. It was never written down again." for n, c in needles]
    body = paras[:idx] + notes[:1] + paras[idx:idx + 1] + notes[1:] + paras[idx + 1:]
    return "\n\n".join([f"Archive qwen38-t3-{seed}. Field notes follow."] + body)


def question(name: str) -> str:
    return f"\n\nWhat is the vault code assigned to {name}? Reply with only the code."


def build_cell(c: Client, length: int, depth: float, seed: int, ratio: float) -> tuple[str, list]:
    """Fit the document into [0.97, 1.0] x (length - 96) tokens via /tokenize."""
    needles = [needle_code(seed), needle_code(seed + 500)]
    budget = length - 96
    n = max(4, int(budget / ratio / 420))
    paras = filler(seed, int(n * 1.5))
    for _ in range(10):
        ntok = c.tokenize_count(needle_doc(paras[:n], depth, needles, seed))
        if 0.97 * budget <= ntok <= budget:
            break
        n = max(4, int(n * budget / ntok * (0.985 if ntok > budget else 1.0)))
        if n > len(paras):
            paras = filler(seed, int(n * 1.3))
    return needle_doc(paras[:n], depth, needles, seed), needles


def verdict(cells: list[dict]) -> dict:
    gated = [x for x in cells if x["length"] <= GATE_MAX]
    found = sum(x["found"] for x in cells)
    gated_found = sum(x["found"] for x in gated)
    ok = gated_found == len(gated) and found >= len(cells) - 2
    per_len = {}
    for x in cells:
        per_len.setdefault(str(x["length"]), [0, 0])
        per_len[str(x["length"])][0] += x["found"]
        per_len[str(x["length"])][1] += 1
    return {"pass": ok, "found": found, "total": len(cells), "gated_found": gated_found,
            "gated_total": len(gated), "per_length": {k: f"{a}/{b}" for k, (a, b) in per_len.items()},
            "rule": f"all needles at <= {GATE_MAX} found; overall misses <= 2"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    ap.add_argument("--with-250k", action="store_true")
    ap.add_argument("--depths", default=",".join(map(str, DEPTHS)))
    a = ap.parse_args(argv)
    lengths = [int(x) for x in a.lengths.split(",") if x] + ([LONG] if a.with_250k else [])
    depths = [float(x) for x in a.depths.split(",") if x]
    c = Client(a.url, a.model, timeout=1800)
    out = prepare_out(a.out)
    write_manifest(out, c, "t3_needles", vars(a))
    rows_path = out / "t3.jsonl"
    rows_path.write_text("")
    sample = "\n\n".join(filler(1, 40))
    ratio = c.tokenize_count(sample) / len(sample)
    cells = []
    for li, length in enumerate(lengths):
        for di, depth in enumerate(depths):
            seed = 1000 * (li + 1) + di
            doc, needles = build_cell(c, length, depth, seed, ratio)
            for ni, (name, code) in enumerate(needles):
                r = c.chat(doc + question(name), max_tokens=32, temperature=0, **OFF)
                row = {"length": length, "depth": depth, "needle": ni, "code": code,
                       "found": code in r["content"].upper(), "answer": r["content"].strip()[:80],
                       "prompt_tokens": r["usage"].get("prompt_tokens"), "s": r["s"]}
                cells.append(row)
                append_jsonl(rows_path, row)
                print(f"  {length:>7} @{depth} needle{ni}: {'found' if row['found'] else 'MISS'} "
                      f"({row['prompt_tokens']} tok, {row['s']} s)", flush=True)
    v = verdict(cells)
    write_json(out / "t3.json", v)
    print(f"T3 {'PASS' if v['pass'] else 'FAIL'} {v['found']}/{v['total']} per_length={v['per_length']}")
    return 0 if v["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
