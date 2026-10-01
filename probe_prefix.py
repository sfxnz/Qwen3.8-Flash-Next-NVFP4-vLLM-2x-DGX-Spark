#!/usr/bin/env python3
"""Prefix-cache reuse probe (plan 0-7; finding F20).

F20: with MTP on the pin, --enable-prefix-caching may buy zero (or one-turn-
lagged) reuse. This probe measures it directly, sequentially, one request at a
time, reading vllm:prefix_cache_hits / vllm:prefix_cache_queries deltas from
/metrics around each request:

  repeat   three identical sends of one salted --tokens (12k-20k) prompt
  chat     a 3-turn chat on a second salted document: each turn resends the
           whole history plus the model's previous answer and a new question

Per request: TTFT, prompt tokens, hits, queries, hit ratio, and
usage.prompt_tokens_details.cached_tokens when the server reports it.
The counters are server-global: run on an idle server. Each row records the
running+waiting request count seen just before the send.
"""

from __future__ import annotations

import argparse
import sys
import uuid

from rulers import common as C

QUESTIONS = (
    "In one sentence, what is the document about?",
    "Name three words that appear often in the document.",
    "Give the document a short title.",
)


def measured(args, em, mode: str, idx: int, messages: list[dict]) -> dict:
    foreign = C.busy(args.url)
    before = C.scrape(args.url)
    r = C.stream_chat(args.url, C.chat_body(args.model, messages, args.max_tokens), timeout=args.timeout)
    after = C.scrape(args.url)
    s = C.step_stats(r)
    hits = C.mdelta(before, after, "vllm:prefix_cache_hits")
    queries = C.mdelta(before, after, "vllm:prefix_cache_queries")
    row = em.row(
        "request", mode=mode, idx=idx, foreign_requests=foreign, prefix_hits=hits, prefix_queries=queries,
        hit_ratio=C.rnd(hits / queries) if hits is not None and queries else None,
        ttft_s=s["ttft_s"], prompt_tokens=s["prompt_tokens"], cached_tokens=s["cached_tokens"],
        completion_tokens=s["completion_tokens"], error=s["error"],
    )
    print(f"{mode} #{idx}: ttft={row['ttft_s']} prompt={row['prompt_tokens']} hits={hits} queries={queries}", flush=True)
    row["_content"] = r["content"]
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(p)
    p.add_argument("--tokens", type=int, default=16000, help="document size; the plan range is 12000-20000")
    p.add_argument("--modes", nargs="+", default=["repeat", "chat"], choices=["repeat", "chat"])
    p.add_argument("--sends", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--timeout", type=float, default=900)
    args = p.parse_args()
    C.require_health(args.url)
    em = C.Emitter(__file__, args.out, label=args.label, tokens=args.tokens)
    salt = uuid.uuid4().hex[:8]
    summary = []

    if "repeat" in args.modes:
        doc, n = C.long_prompt(args.url, args.model, args.tokens, f"{salt}-repeat", question=QUESTIONS[0])
        rows = [measured(args, em, "repeat", i, [{"role": "user", "content": doc}]) for i in range(args.sends)]
        summary.append({"mode": "repeat", "tokenized": n,
                        "ttft_s": [r["ttft_s"] for r in rows], "hits": [r["prefix_hits"] for r in rows]})

    if "chat" in args.modes:
        doc, n = C.long_prompt(args.url, args.model, args.tokens, f"{salt}-chat", question=QUESTIONS[0])
        msgs = [{"role": "user", "content": doc}]
        rows = []
        for turn in range(len(QUESTIONS)):
            row = measured(args, em, "chat", turn, msgs)
            rows.append(row)
            msgs = msgs + [{"role": "assistant", "content": row["_content"] or "(no answer)"}]
            if turn + 1 < len(QUESTIONS):
                msgs.append({"role": "user", "content": QUESTIONS[turn + 1]})
        summary.append({"mode": "chat", "tokenized": n,
                        "ttft_s": [r["ttft_s"] for r in rows], "hits": [r["prefix_hits"] for r in rows]})

    em.summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
