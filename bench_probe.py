#!/usr/bin/env python3
"""T0 concurrency-equivalence probe (plan 0-6; F01, F04, C19).

Gathers a c=1 greedy reference for every prompt first, then fires barrier
bursts of N in {2,3,5,8} x {identical, distinct} prompts (3 per shape),
staggered-arrival controls, two >8192-token prompts together and a c=2 image
pair. Each stream is compared with its c=1 reference.

Gate flags (exit 1): zero_acceptance (acceptance_len < 1.3 over >= 20 steps),
runaway (finish_reason=length where the reference stopped) and, in
identical-prompt bursts, slow (decode rate < 0.6x the burst median). Exit 4 when
nothing flagged but some bursts ran while foreign requests were running
(inconclusive: the prefill-only path was not exercised); exit 3 on a failed
--self-test.
Report-only: early_divergence (first 32 tokens differ from the reference)
and slow in distinct-prompt bursts.

Per-request acceptance comes from --per-request-spec-decode-metrics
(metrics.speculative_decoding on the final usage chunk) when the server has
it on; otherwise it is estimated from tokens per streamed chunk.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import math
import statistics
import struct
import sys
import threading
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

PROBE_VERSION = "1"

ZERO_ACCEPT_LEN = 1.3
ZERO_ACCEPT_MIN_STEPS = 20
SLOW_FRACTION = 0.6
PREFIX_TOKENS = 32
COPREFILL_TTFT_S = 0.020

# Short prompts that end on their own well under max_tokens, so a c=1
# reference that stops makes a burst stream that hits the cap a runaway.
DISTINCT_PROMPTS = [
    "Write a short paragraph about why sparse attention helps long-context "
    "language models. Keep it around eighty words. No bullet points.",
    "Explain in about sixty words how a bicycle gear system trades speed for torque.",
    "List the first twelve prime numbers, separated by commas, and nothing else.",
    "Describe the water cycle in four sentences for a ten-year-old.",
    "Write a Python function that returns the n-th Fibonacci number iteratively. Code only.",
    "Translate into French: 'The library opens at nine and closes at six on weekdays.'",
    "Give three practical tips for keeping houseplants alive in winter, one line each.",
    'Return a JSON object with keys "name", "year" and "tags" describing the Eiffel Tower. JSON only.',
]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def log(msg: str) -> None:
    print(f"{now_iso()} {msg}", flush=True)


def png_solid(rgb: tuple[int, int, int], size: int = 256) -> str:
    """Stdlib PNG of one colour, base64."""
    row = b"\x00" + bytes(rgb) * size
    raw = row * size

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


def image_messages(rgb: tuple[int, int, int]) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What single colour fills this image? Answer in one short sentence."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_solid(rgb)}"}},
            ],
        }
    ]


def long_text_prompt(seed: int, nonce: str, lines: int = 500) -> str:
    """>8192-token retrieval prompt (~20 tokens per line, ~10k tokens); short fixed answer."""
    body = [f"Session {nonce}. Read the ledger and answer the question at the end."]
    for i in range(lines):
        code = (i * 7919 + seed * 104729) % 100000
        body.append(f"Ledger line {i:04d}: bay {i % 53} holds crate code {code:05d}.")
    target = (lines * 3) // 4
    body.append(f"Question: what crate code is on ledger line {target:04d}? Reply with the code only.")
    return "\n".join(body)


def text_messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def stream_probe(url: str, model: str, messages: list[dict], max_tokens: int, barrier: threading.Barrier | None = None,
                 delay_s: float = 0.0) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    if barrier is not None:
        barrier.wait()
    if delay_s:
        time.sleep(delay_s)
    t0 = time.perf_counter()
    first = None
    chunks = 0
    parts: list[str] = []
    cont_usage = False
    prefix_chars = 0  # chars of the longest chunk-aligned prefix with <= PREFIX_TOKENS tokens
    usage: dict = {}
    finish = None
    spec = None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            m = ev.get("metrics") or {}
            if m.get("speculative_decoding"):
                spec = m["speculative_decoding"]
            choices = ev.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            delta = (ch.get("delta") or {}).get("content") or ""
            if delta:
                if first is None:
                    first = time.perf_counter()
                chunks += 1
                parts.append(delta)
                # continuous_usage_stats: usage on each chunk is cumulative.
                if ev.get("usage"):
                    cont_usage = True
                if int(usage.get("completion_tokens") or 0) <= PREFIX_TOKENS:
                    prefix_chars += len(delta)
    t1 = time.perf_counter()
    text = "".join(parts)
    if not cont_usage:
        prefix_chars = 4 * PREFIX_TOKENS  # no per-chunk usage: ~4 chars per token
    completion = int(usage.get("completion_tokens") or 0)
    decode_s = (t1 - first) if first is not None else 0.0
    steps = chunks - 1
    row = {
        "t0": t0,
        "ttft_s": (first - t0) if first is not None else None,
        "decode_s": decode_s,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": completion,
        "chunks": chunks,
        "finish_reason": finish,
        "decode_tok_s": (max(completion - 1, 0) / decode_s) if decode_s > 0 else 0.0,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "head": text[:80],
        "tail": text[-80:],
        "prefix_text": text[:prefix_chars],
        "text": text,
    }
    if spec is not None:
        row["spec_source"] = "server"
        row["steps"] = int(spec.get("num_spec_steps") or 0)
        row["accepted"] = int(spec.get("num_accepted_draft_tokens") or 0)
        row["acceptance_len"] = float(spec.get("mean_acceptance_length") or 0.0)
        row["acceptance_histogram"] = spec.get("acceptance_histogram")
    else:
        row["spec_source"] = "chunks"
        row["steps"] = max(steps, 0)
        row["accepted"] = max(completion - 1 - max(steps, 0), 0)
        row["acceptance_len"] = (max(completion - 1, 0) / steps) if steps > 0 else None
    return row


def flag_stream(row: dict, ref: dict | None, burst_median_rate: float,
                slow_gates: bool = False) -> tuple[list[str], list[str]]:
    gate: list[str] = []
    report: list[str] = []
    acc = row.get("acceptance_len")
    if row["steps"] >= ZERO_ACCEPT_MIN_STEPS and acc is not None and acc < ZERO_ACCEPT_LEN:
        gate.append("zero_acceptance")
    if ref is not None and row["finish_reason"] == "length" and ref["finish_reason"] == "stop":
        gate.append("runaway")
    if ref is not None and not row["text"].startswith(ref["prefix_text"]):
        report.append("early_divergence")
    if ref is not None and "text" in ref:
        a, b = row["text"], ref["text"]
        n = min(len(a), len(b))
        row["diverge_at_char"] = next((i for i in range(n) if a[i] != b[i]), None if len(a) == len(b) else n)
    if burst_median_rate > 0 and row["decode_tok_s"] < SLOW_FRACTION * burst_median_rate:
        # Identical prompts should decode at one rate, so slow gates there
        # (plan 0-6); distinct prompts differ in acceptance, so it only reports.
        (gate if slow_gates else report).append("slow")
    return gate, report


class Probe:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.refs: dict[str, dict] = {}
        self.bursts: list[dict] = []
        self.nonce = f"{time.time_ns():x}"
        self.long_calls = 0

    # Each request is (ref_key, messages).
    def requests_for(self, kind: str, n: int) -> list[tuple[str, list[dict]]]:
        if kind == "identical":
            return [("d0", text_messages(DISTINCT_PROMPTS[0]))] * n
        if kind == "distinct":
            return [(f"d{i}", text_messages(DISTINCT_PROMPTS[i % len(DISTINCT_PROMPTS)])) for i in range(n)]
        if kind == "long":
            # A fresh nonce per call keeps every long prefill cold (no prefix-cache
            # hit on the reference); the answer depends only on the seed.
            self.long_calls += 1
            return [
                (f"long{i}", text_messages(long_text_prompt(i + 1, f"{self.nonce}-{self.long_calls}-{i}")))
                for i in range(n)
            ]
        if kind == "image":
            cols = [("img_red", (220, 20, 60)), ("img_blue", (30, 60, 220))]
            return [(cols[i][0], image_messages(cols[i][1])) for i in range(n)]
        raise ValueError(kind)

    def reference(self) -> None:
        keys: dict[str, list[dict]] = {}
        for kind, n in (("distinct", len(DISTINCT_PROMPTS)), ("long", 2), ("image", 2)):
            if kind == "long" and self.args.skip_long:
                continue
            if kind == "image" and self.args.skip_image:
                continue
            for key, msgs in self.requests_for(kind, n):
                keys[key] = msgs
        for key, msgs in keys.items():
            r = stream_probe(self.args.url, self.args.model, msgs, self.args.max_tokens)
            self.refs[key] = r
            log(f"ref {key} finish={r['finish_reason']} completion={r['completion_tokens']} "
                f"acc={fmt(r['acceptance_len'])} ttft={fmt(r['ttft_s'], '.3f')} sha={r['text_sha256'][:12]} "
                f"head={r['head'][:40]!r}")
            time.sleep(self.args.idle_s / 3)

    def burst(self, name: str, kind: str, n: int, stagger_s: float = 0.0) -> dict:
        reqs = self.requests_for(kind, n)
        barrier = threading.Barrier(n)
        time.sleep(self.args.idle_s)
        # The bug path needs a prefill-only step: other running requests turn
        # the burst into a mixed batch, so wait for an idle engine (F01).
        deadline = time.time() + self.args.idle_wait_s
        running = running_requests(self.args.metrics_url)
        while running and time.time() < deadline:
            time.sleep(1.0)
            running = running_requests(self.args.metrics_url)
        t_start = time.time()
        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = [
                pool.submit(stream_probe, self.args.url, self.args.model, msgs, self.args.max_tokens, barrier,
                            i * stagger_s)
                for i, (_, msgs) in enumerate(reqs)
            ]
            rows = []
            errors = []
            for (key, _), f in zip(reqs, futs):
                try:
                    r = f.result()
                    r["ref_key"] = key
                    rows.append(r)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{key}: {exc}")
        ttfts = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
        # Absolute first-token times: equal within 20 ms means one prefill step.
        firsts = [r["t0"] + r["ttft_s"] for r in rows if r["ttft_s"] is not None]
        coprefill = bool(firsts) and (max(firsts) - min(firsts)) <= COPREFILL_TTFT_S and stagger_s == 0
        med = statistics.median(r["decode_tok_s"] for r in rows) if rows else 0.0
        for r in rows:
            gate, report = flag_stream(r, self.refs.get(r["ref_key"]), med, slow_gates=kind == "identical")
            r["gate_flags"] = gate
            r["report_flags"] = report
        b = {
            "name": name,
            "kind": kind,
            "n": n,
            "stagger_s": stagger_s,
            "t_start": t_start,
            "running_before": running,
            "idle": not running,
            "coprefill": coprefill,
            "first_token_spread_s": (max(firsts) - min(firsts)) if firsts else None,
            "median_decode_tok_s": med,
            "errors": errors,
            "streams": [{k: v for k, v in r.items() if k not in ("text", "prefix_text")} for r in rows],
            "gate_flags": sorted({f for r in rows for f in r["gate_flags"]}) + (["error"] if errors else []),
            "report_flags": sorted({f for r in rows for f in r["report_flags"]}),
        }
        self.bursts.append(b)
        self.print_burst(b)
        return b

    @staticmethod
    def print_burst(b: dict) -> None:
        log(
            f"burst={b['name']} kind={b['kind']} n={b['n']} stagger={b['stagger_s']:.3f} "
            f"coprefill={b['coprefill']} spread={fmt(b['first_token_spread_s'], '.3f')} "
            f"gate={b['gate_flags'] or '-'} report={b['report_flags'] or '-'}"
            + (f" running_before={b['running_before']:.0f}" if b["running_before"] else "")
        )
        for i, r in enumerate(b["streams"]):
            flags = "+".join(r["gate_flags"] + r["report_flags"]) or "ok"
            log(
                f"  [{i}] {r['ref_key']} finish={r['finish_reason']} completion={r['completion_tokens']} "
                f"chunks={r['chunks']} acc={fmt(r['acceptance_len'])}({r['spec_source']}) "
                f"rate={r['decode_tok_s']:.2f} ttft={fmt(r['ttft_s'], '.3f')} sha={r['text_sha256'][:12]} "
                f"{flags} div@{r.get('diverge_at_char', 'na')} head={r['head'][:32]!r} tail={r['tail'][-24:]!r}"
            )
        for e in b["errors"]:
            log(f"  ERROR {e}")


def running_requests(metrics_url: str) -> float | None:
    try:
        with urllib.request.urlopen(metrics_url, timeout=10) as resp:
            text = resp.read().decode("utf-8", "replace")
    except OSError:
        return None
    total = 0.0
    for line in text.splitlines():
        if line.split("{", 1)[0].split(" ", 1)[0] == "vllm:num_requests_running":
            try:
                total += float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return total


def fmt(v, spec: str = ".3f") -> str:
    return "na" if v is None else format(v, spec)


def statistical_self_test(bursts: list[dict], k: int) -> dict:
    """On an unfixed (B0) boot: row 0 is always healthy and >=30% of non-first
    co-prefilled bursts are flagged (plan 0-6)."""
    barrier = [b for b in bursts if b["stagger_s"] == 0 and b["kind"] in ("identical", "distinct")]
    non_first = barrier[1:]
    flagged = [b for b in non_first if b["gate_flags"]]
    row0_ok = all(
        sum(1 for r in b["streams"] if not r["gate_flags"]) >= 1 for b in barrier
    )
    frac = (len(flagged) / len(non_first)) if non_first else 0.0
    return {
        "k": k,
        "barrier_bursts": len(barrier),
        "non_first_flagged": len(flagged),
        "non_first_flagged_fraction": frac,
        "row0_always_healthy": row0_ok,
        "max_zero_streams_allowed": {str(b["n"]): b["n"] - math.ceil(b["n"] / (k + 1)) for b in barrier},
        "pass": row0_ok and frac >= 0.30,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="nvidia/Qwen3.8-Flash-Next-NVFP4")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 5, 8])
    p.add_argument("--kinds", nargs="+", default=["identical", "distinct"], choices=["identical", "distinct"])
    p.add_argument("--bursts-per-shape", type=int, default=3)
    p.add_argument("--idle-s", type=float, default=3.0, help="idle gap before each burst")
    p.add_argument("--stagger-s", type=float, default=0.25, help="arrival gap in staggered controls")
    p.add_argument("--idle-wait-s", type=float, default=60.0,
                   help="max wait for vllm:num_requests_running == 0 before each burst")
    p.add_argument("--allow-busy", action="store_true",
                   help="exit 0 without flags even if some bursts ran on a busy engine")
    p.add_argument("--skip-staggered", action="store_true")
    p.add_argument("--skip-long", action="store_true", help="skip the two >8192-token prompts")
    p.add_argument("--skip-image", action="store_true", help="skip the c=2 image pair")
    p.add_argument("--k", type=int, default=3, help="num_speculative_tokens (self-test signature)")
    p.add_argument("--self-test", action="store_true",
                   help="statistical self-test for an unfixed B0 boot: exit 0 iff the probe sees the bug")
    p.add_argument("--json", dest="json_path")
    args = p.parse_args()
    args.metrics_url = args.url.split("/v1/", 1)[0] + "/metrics"

    log(f"bench_probe v{PROBE_VERSION} url={args.url} sizes={args.sizes} kinds={args.kinds} "
        f"bursts_per_shape={args.bursts_per_shape} max_tokens={args.max_tokens}")
    probe = Probe(args)
    probe.reference()
    for n in args.sizes:
        for kind in args.kinds:
            for i in range(args.bursts_per_shape):
                probe.burst(f"{kind}-n{n}-b{i+1}", kind, n)
    if not args.skip_staggered:
        # 24 barrier + 4 staggered + long pair + image pair = 30 bursts.
        for n, kind in ((2, "identical"), (3, "distinct"), (5, "distinct"), (8, "identical")):
            if n <= max(args.sizes):
                probe.burst(f"staggered-{kind}-n{n}", kind, n, stagger_s=args.stagger_s)
    if not args.skip_long:
        probe.burst("long-pair", "long", 2)
    if not args.skip_image:
        probe.burst("image-pair", "image", 2)

    gate_bursts = [b for b in probe.bursts if b["gate_flags"]]
    streams = [r for b in probe.bursts for r in b["streams"]]
    server_spec = sum(1 for r in streams if r["spec_source"] == "server")
    div = sum(1 for r in streams if "early_divergence" in r["report_flags"])
    result = {
        "probe_version": PROBE_VERSION,
        "argv": sys.argv[1:],
        "finished_at": now_iso(),
        "refs": {k: {kk: vv for kk, vv in v.items() if kk not in ("text", "prefix_text")} for k, v in probe.refs.items()},
        "bursts": probe.bursts,
        "totals": {
            "bursts": len(probe.bursts),
            "streams": len(streams),
            "gate_flagged_bursts": len(gate_bursts),
            "gate_flagged_streams": sum(1 for r in streams if r["gate_flags"]),
            "zero_acceptance_streams": sum(1 for r in streams if "zero_acceptance" in r["gate_flags"]),
            "runaway_streams": sum(1 for r in streams if "runaway" in r["gate_flags"]),
            "early_divergence_streams": div,
            "first32_match_rate": (1 - div / len(streams)) if streams else None,
            "coprefill_bursts": sum(1 for b in probe.bursts if b["coprefill"]),
            "busy_bursts": sum(1 for b in probe.bursts if not b["idle"]),
            "long_prompt_tokens": [r["prompt_tokens"] for b in probe.bursts if b["kind"] == "long" for r in b["streams"]],
            "per_request_spec_metrics": server_spec == len(streams) and bool(streams),
        },
    }
    t = result["totals"]
    log(f"TOTALS bursts={t['bursts']} streams={t['streams']} gate_bursts={t['gate_flagged_bursts']} "
        f"zero_acc={t['zero_acceptance_streams']} runaway={t['runaway_streams']} "
        f"first32_match={fmt(t['first32_match_rate'])} coprefill_bursts={t['coprefill_bursts']} "
        f"per_request_spec_metrics={t['per_request_spec_metrics']}")
    if not t["per_request_spec_metrics"]:
        log("NOTE: server did not return metrics.speculative_decoding on every stream; acceptance "
            "estimated from tokens per chunk (serve with --per-request-spec-decode-metrics summary)")
    if args.self_test:
        st = statistical_self_test(probe.bursts, args.k)
        result["self_test"] = st
        log(f"SELFTEST pass={st['pass']} non_first_flagged={st['non_first_flagged']}/"
            f"{st['barrier_bursts'] - 1} ({st['non_first_flagged_fraction']:.0%}) "
            f"row0_always_healthy={st['row0_always_healthy']}")
        rc = 0 if st["pass"] else 3
    else:
        if gate_bursts:
            rc, verdict = 1, "FAIL"
        elif t["busy_bursts"] and not args.allow_busy:
            # No flags, but some bursts shared the engine with foreign requests,
            # so they never took the prefill-only path the gate must exercise.
            rc, verdict = 4, f"INCONCLUSIVE ({t['busy_bursts']} burst(s) on a busy engine)"
        else:
            rc, verdict = 0, "PASS"
        log(f"T0 {verdict}")
    if args.json_path:
        with open(args.json_path, "w") as fh:
            json.dump(result, fh, indent=1)
            fh.write("\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
