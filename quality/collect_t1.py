#!/usr/bin/env python3
"""T1-A / T1-B capture: teacher-forced prompt_logprobs over the frozen corpus.

Every corpus row is a chat whose last assistant message is forced text. It is
sent through the served chat template (/v1/chat/completions,
continue_final_message, thinking off) with prompt_logprobs=K, max_tokens=1.
Positions after the first </think> are kept: the forced assistant text, and
for tool transcripts the call, the tool result and the answer.

Capture on an idle serve. On the pinned image, an A/A pair captured next to 4-7
other agents' streams agreed on only 92.5% of top-1 tokens, with target
logprobs off by up to 3.8 nats: consistent with the co-prefill corruption in
F01. index.jsonl records
others_running (vllm:num_requests_running just before each request).

Pieces are <= 1000 tokens, so the per-request transient is about
1024 x 248320 x 4 B = 1 GB (vLLM computes prompt logprobs per scheduled
chunk). --long-whole sends each 16k-token `long` document as one message
instead: the transient is then MAX_NUM_BATCHED_TOKENS x 248320 x 4 B
(8.1 GB at 8192). Watch `free -h` on both nodes before using it.

Output (--out DIR):
  manifest.json        corpus sha, server /v1/models, git rev, args
  index.jsonl          one row per item: id, domain, shard, off, n, prompt_ids_sha
  t1-NNN.npz           tgt, tgt_lp, top_ids [n,K], top_lp [n,K]; rows concatenated
  summary.json         tokens and mean NLL per domain

  python3 quality/collect_t1.py --out RUN_DIR [--domains prose,code] [--limit 3]
Score two runs with `python3 quality/score.py t1 REF_DIR CAND_DIR`.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vision_probes"))
from qlib import (DATA, DEFAULT_URL, OFF, Client, jsonl, prepare_out, running_requests,  # noqa: E402
                    sha256_file, write_json, write_manifest)

CORPUS = DATA / "t1_corpus.jsonl"
TOPK = 5


def resolve_images(messages: list[dict]) -> list[dict]:
    import probes  # vision_probes/probes.py; imported lazily, text-only runs never draw
    out = copy.deepcopy(messages)
    for m in out:
        if isinstance(m.get("content"), list):
            m["content"] = [probes.image_part(p["name"]) if p.get("type") == "image_ref" else p
                            for p in m["content"]]
    return out


def long_whole(rows: list[dict]) -> list[dict]:
    """Merge each long document's consecutive pieces into one assistant message."""
    docs, out = {}, []
    for r in rows:
        if r["domain"] == "long":
            docs.setdefault(r["doc"], []).append(r)
        else:
            out.append(r)
    for doc, parts in docs.items():
        parts.sort(key=lambda r: r["piece"])
        text = "\n".join(p["messages"][-1]["content"] for p in parts)
        out.append({"id": f"{doc}.whole", "domain": "long_whole",
                    "messages": [parts[0]["messages"][0], {"role": "assistant", "content": text}]})
    return out


def select(rows: list[dict], domains: set, limit: int) -> list[dict]:
    seen, out = {}, []
    for r in rows:
        if domains and r["domain"] not in domains:
            continue
        seen[r["domain"]] = seen.get(r["domain"], 0) + 1
        if limit and seen[r["domain"]] > limit:
            continue
        out.append(r)
    return out


def parse_prompt_logprobs(ids: list[int], plp: list, start: int, k: int) -> dict:
    """prompt_logprobs[i] is {str(token_id): {logprob, rank, ...}} for prompt token i (None at 0)."""
    n = len(ids) - start
    tgt = np.asarray(ids[start:], dtype=np.int32)
    tgt_lp = np.empty(n, dtype=np.float32)
    top_ids = np.empty((n, k), dtype=np.int32)
    top_lp = np.empty((n, k), dtype=np.float32)
    for j, i in enumerate(range(start, len(ids))):
        entry = plp[i]
        tgt_lp[j] = entry[str(ids[i])]["logprob"]
        top = sorted((v["rank"], -v["logprob"], int(t)) for t, v in entry.items() if v["rank"] <= k)[:k]
        if len(top) < k:
            raise RuntimeError(f"position {i}: {len(top)} ranked logprobs < {k}")
        top_ids[j] = [t[2] for t in top]
        top_lp[j] = [-t[1] for t in top]
    return {"tgt": tgt, "tgt_lp": tgt_lp, "top_ids": top_ids, "top_lp": top_lp}


def capture(c: Client, row: dict, think_end: int, k: int) -> tuple[dict, dict]:
    body = {"messages": resolve_images(row["messages"]), "max_tokens": 1, "temperature": 0,
            "prompt_logprobs": k, "return_token_ids": True, "add_generation_prompt": False,
            "continue_final_message": True, **OFF}
    if row.get("tools"):
        body["tools"] = row["tools"]
    out = c.post("/v1/chat/completions", body)
    ids, plp = out.get("prompt_token_ids"), out.get("prompt_logprobs")
    if not ids or not plp:
        raise RuntimeError(f"{row['id']}: response has no prompt_token_ids/prompt_logprobs")
    if think_end not in ids:
        raise RuntimeError(f"{row['id']}: no </think> ({think_end}) in the prompt; template changed?")
    start = ids.index(think_end) + 1
    arrs = parse_prompt_logprobs(ids, plp, start, k)
    meta = {"id": row["id"], "domain": row["domain"], "n": len(ids) - start, "prompt_tokens": len(ids),
            "score_from": start, "prompt_ids_sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]}
    return meta, arrs


def flush(out: Path, shard: int, bufs: list[dict]) -> None:
    if not bufs:
        return
    np.savez_compressed(out / f"t1-{shard:03d}.npz",
                        **{key: np.concatenate([b[key] for b in bufs]) for key in bufs[0]})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--domains", default="", help="comma list; default all")
    ap.add_argument("--limit", type=int, default=0, help="max items per domain (smoke runs)")
    ap.add_argument("--long-whole", action="store_true", help="send long docs whole (16k tokens; see above)")
    ap.add_argument("--shard-size", type=int, default=64)
    a = ap.parse_args(argv)
    c = Client(a.url, a.model, timeout=1800)
    out = prepare_out(a.out)
    rows = jsonl(a.corpus)
    if a.long_whole:
        rows = long_whole(rows)
    rows = select(rows, {d for d in a.domains.split(",") if d}, a.limit)
    think = c.tokenize("</think>")
    if len(think) != 1:
        raise SystemExit(f"</think> is not one token: {think}")
    write_manifest(out, c, "collect_t1", vars(a), files=[a.corpus],
                   extra={"corpus_sha256": sha256_file(a.corpus), "items": len(rows), "think_end_id": think[0]})
    index = (out / "index.jsonl").open("w", encoding="utf-8")
    bufs, shard, off, t0 = [], 0, 0, time.time()
    dom, loaded = {}, 0
    for i, row in enumerate(rows):
        busy = running_requests(c)
        meta, arrs = capture(c, row, think[0], a.topk)
        meta.update(shard=shard, off=off, others_running=busy)
        loaded += bool(busy)
        index.write(json.dumps(meta) + "\n")
        bufs.append(arrs)
        off += meta["n"]
        d = dom.setdefault(row["domain"], {"items": 0, "tokens": 0, "nll_sum": 0.0})
        d["items"] += 1
        d["tokens"] += meta["n"]
        d["nll_sum"] -= float(arrs["tgt_lp"].astype(np.float64).sum())
        if len(bufs) == a.shard_size:
            flush(out, shard, bufs)
            bufs, shard, off = [], shard + 1, 0
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            tok = sum(x["tokens"] for x in dom.values())
            print(f"  t1 {i + 1}/{len(rows)} items, {tok} tokens, {time.time() - t0:.0f} s", flush=True)
    flush(out, shard, bufs)
    index.close()
    if loaded:
        print(f"WARNING: {loaded}/{len(rows)} items were captured while other requests were running. "
              "Co-prefilled captures are not comparable on this pin (F01); capture T1 on an idle serve.",
              file=sys.stderr)
    summary = {"items": len(rows), "items_with_other_traffic": loaded,
               "tokens": sum(x["tokens"] for x in dom.values()), "topk": a.topk,
               "s": round(time.time() - t0, 1),
               "by_domain": {k: {"items": v["items"], "tokens": v["tokens"],
                                 "mean_nll": round(v["nll_sum"] / max(1, v["tokens"]), 5)} for k, v in sorted(dom.items())}}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
