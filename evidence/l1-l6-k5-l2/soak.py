#!/usr/bin/env python3
"""L2a soak (session 4): mixed traffic for --minutes, 0 hangs allowed.

Loop until the deadline: a c=8 burst of distinct short prompts (streamed, greedy, 256 tokens), then a
c=1 request, then a c=2 pair; a ~32k-token prefill is sent at start and at the half-way mark while
a c=8 burst decodes next to it. A request that takes longer than --timeout (default 300 s) or errors
counts as a failure. /health is checked after every round. Writes one JSON line per round to --out.
"""
import argparse
import concurrent.futures as cf
import json
import time
import urllib.request

MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"
TOPICS = ["rivers", "volcanoes", "bread", "chess", "tides", "bees", "glass", "radio", "maps", "salt",
          "clocks", "whales", "ink", "bridges", "moss", "kites"]


def req(url, prompt, max_tokens, timeout):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True}
    t0 = time.time()
    r = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    n = 0
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            for line in resp:
                if line.startswith(b"data: ") and b"[DONE]" not in line:
                    n += 1
        return {"ok": True, "s": round(time.time() - t0, 2), "chunks": n}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "s": round(time.time() - t0, 2), "err": f"{type(e).__name__}: {e}"[:200]}


def health(base):
    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as r:
            return r.status
    except Exception:  # noqa: BLE001
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--long-words", type=int, default=24000, help="~32k tokens")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = a.url.split("/v1/")[0]
    long_prompt = "Summarise the following notes in one sentence.\n" + " ".join(
        f"note{i % 997} about {TOPICS[i % len(TOPICS)]}" for i in range(a.long_words // 4))
    end = time.time() + a.minutes * 60
    half = time.time() + a.minutes * 30
    long_sent = 0
    rnd = 0
    fails = 0
    with open(a.out, "w") as out, cf.ThreadPoolExecutor(16) as ex:
        while time.time() < end:
            rnd += 1
            futs = []
            if long_sent == 0 or (long_sent == 1 and time.time() > half):
                futs.append(("long", ex.submit(req, a.url, long_prompt, 16, a.timeout)))
                long_sent += 1
            for i in range(8):
                p = f"Write a short paragraph about {TOPICS[(rnd + i) % len(TOPICS)]} (variant {rnd}-{i})."
                futs.append(("c8", ex.submit(req, a.url, p, 256, a.timeout)))
            res = [(k, f.result()) for k, f in futs]
            res.append(("c1", req(a.url, f"Count from 1 to 40, round {rnd}.", 200, a.timeout)))
            pair = [ex.submit(req, a.url, f"List five facts about {t} (r{rnd}).", 160, a.timeout)
                    for t in TOPICS[rnd % 8: rnd % 8 + 2]]
            res += [("c2", f.result()) for f in pair]
            h = health(base)
            bad = [r for _, r in res if not r["ok"]]
            fails += len(bad) + (h != 200)
            row = {"round": rnd, "t": round(time.time(), 1), "health": h, "n": len(res), "fail": len(bad),
                   "kinds": {k: r["s"] for k, r in res if k == "long"}, "errors": [r.get("err") for r in bad]}
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps(row), flush=True)
            if h != 200:
                break
    summary = {"rounds": rnd, "failures": fails, "long_prefills": long_sent, "minutes": a.minutes}
    print("SUMMARY", json.dumps(summary))
    raise SystemExit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
