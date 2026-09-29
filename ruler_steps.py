#!/usr/bin/env python3
"""Per-stream step companion to the frozen bench_decode.py ruler (plan 0-5).

Imports stream_one, PHASES, spec_counters and wave from bench_decode unchanged
and runs the same wave loop. Adds what the frozen script computes but never
prints: per-stream completion_tokens, chunks, decode_s and TTFT, per-stream
ms/step = decode_s / (chunks - 1) (stream_interval 1: one chunk per engine
step), per-wave spec-counter deltas and their cross-check against the chunk
counts, a discarded warm-up (C26), structured c=1 sentinels at start, middle
and end (G13, G16), and the T0 stream flags (F01, F04). SUMMARY is refused
while any T0 flag fires or the boot is void.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from bench_decode import PHASES, spec_counters, stream_one, wave

RULER_STEPS_VERSION = "1"

# T0 per-stream thresholds.
ZERO_ACCEPT_TOKENS_PER_STEP = 1.05  # tokens/step at or under this is "no drafts accepted"
ZERO_ACCEPT_MIN_STEPS = 20
SLOW_FRACTION = 0.6  # per-stream rate below this x the wave median

# Historical self-test (plan 0-6, F04 validation): bench_decode default order
# on a fresh B0 boot must flag exactly these (phase, c, run) waves.
HISTORICAL_EXPECTED = (
    ("prose", 2, 2),
    ("prose", 8, 2),
    ("structured", 2, 1),
    ("structured", 2, 3),
    ("structured", 8, 2),
)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def log(msg: str) -> None:
    print(f"{now_iso()} {msg}", flush=True)


def fetch_metrics(metrics_url: str) -> dict[str, float] | None:
    """Sum selected vLLM request gauges/counters across label sets."""
    names = {
        "vllm:request_success_total": "request_success",
        "vllm:request_success": "request_success",
        "vllm:num_requests_running": "running",
        "vllm:num_requests_waiting": "waiting",
    }
    try:
        with urllib.request.urlopen(metrics_url, timeout=10) as resp:
            text = resp.read().decode("utf-8", "replace")
    except OSError:
        return None
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        key = names.get(name)
        if key is None:
            continue
        try:
            out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
        except ValueError:
            continue
    return out


def enrich_stream(r: dict) -> dict:
    """Add decode_s, steps, tokens/step and ms/step to a stream_one row."""
    decode_s = r["total_s"] - r["ttft_s"]
    steps = r["chunks"] - 1
    decode_tokens = max(r["completion_tokens"] - 1, 0)
    r = dict(r)
    r["decode_s"] = decode_s
    r["steps"] = steps
    r["tokens_per_step"] = (decode_tokens / steps) if steps > 0 else None
    r["ms_per_step"] = (1000.0 * decode_s / steps) if steps > 0 else None
    return r


def stream_flags(r: dict, wave_median_rate: float, runaway_ref: float | None, max_tokens: int) -> list[str]:
    flags = []
    tps = r.get("tokens_per_step")
    if tps is not None and r["steps"] >= ZERO_ACCEPT_MIN_STEPS and tps <= ZERO_ACCEPT_TOKENS_PER_STEP:
        flags.append("zero_acceptance")
    if wave_median_rate > 0 and r["decode_tok_s"] < SLOW_FRACTION * wave_median_rate:
        flags.append("slow")
    if runaway_ref is not None and runaway_ref < max_tokens and r["completion_tokens"] >= max_tokens:
        flags.append("runaway")
    return flags


# Flags that block SUMMARY. "slow" alone is an outlier signal (F04), but the
# plan's 0-5 rule refuses SUMMARY on any T0 flag, and slow is one of them.
T0_FLAGS = ("zero_acceptance", "runaway", "slow")
F01_FLAGS = {"zero_acceptance", "runaway"}


def analyse_wave(
    rows: list[dict],
    wall: float,
    agg: float,
    before: dict | None,
    after: dict | None,
    req_before: dict | None,
    req_after: dict | None,
    runaway_ref: float | None,
    max_tokens: int,
) -> dict:
    rows = [enrich_stream(r) for r in rows]
    rates = [r["decode_tok_s"] for r in rows]
    med_rate = statistics.median(rates)
    for r in rows:
        r["flags"] = stream_flags(r, med_rate, runaway_ref, max_tokens)
    ms = [r["ms_per_step"] for r in rows if r["ms_per_step"] is not None]
    out: dict = {
        "wall_s": wall,
        "agg_tok_s": agg,
        "streams": rows,
        "median_decode_tok_s": med_rate,
        "median_ms_per_step": statistics.median(ms) if ms else None,
        "sum_steps": sum(max(r["steps"], 0) for r in rows),
        "flags": sorted({f for r in rows for f in r["flags"]}),
    }
    if before is not None and after is not None:
        d = {k: after[k] - before[k] for k in after}
        out["d_num_drafts"] = d["num_drafts"]
        out["d_num_draft_tokens"] = d["num_draft_tokens"]
        out["d_num_accepted_tokens"] = d["num_accepted_tokens"]
        if d["num_drafts"] > 0:
            out["acceptance_len"] = 1.0 + d["num_accepted_tokens"] / d["num_drafts"]
            # num_drafts counts one per request per step, so it must match
            # the summed per-stream step counts (F06 cross-check).
            out["xcheck_steps_over_drafts"] = out["sum_steps"] / d["num_drafts"]
            if len(rows) == 1:
                out["xcheck_ms_per_draft"] = 1000.0 * rows[0]["decode_s"] / d["num_drafts"]
    if req_before is not None and req_after is not None and "request_success" in req_after:
        done = req_after["request_success"] - req_before.get("request_success", 0.0)
        out["d_request_success"] = done
        out["running_before"] = req_before.get("running")
        if done != len(rows) or (req_before.get("running") or 0) > 0:
            out["foreign_traffic"] = True
    return out


def run_wave(url: str, model: str, prompt: str, max_tokens: int, c: int, metrics_url: str,
             runaway_ref: float | None) -> dict:
    req_before = fetch_metrics(metrics_url)
    before = spec_counters(metrics_url)
    t_start = time.time()
    rows, wall, agg = wave(url, model, prompt, max_tokens, c)
    t_end = time.time()
    after = spec_counters(metrics_url)
    req_after = fetch_metrics(metrics_url)
    w = analyse_wave(rows, wall, agg, before, after, req_before, req_after, runaway_ref, max_tokens)
    w["t_start"] = t_start
    w["t_end"] = t_end
    return w


def fmt_opt(v: float | None, spec: str) -> str:
    return "na" if v is None else format(v, spec)


def print_wave(tag: str, w: dict) -> None:
    ms = ",".join(fmt_opt(r["ms_per_step"], ".1f") for r in w["streams"])
    comp = ",".join(str(r["completion_tokens"]) for r in w["streams"])
    chunks = ",".join(str(r["chunks"]) for r in w["streams"])
    dec = ",".join(f"{r['decode_s']:.3f}" for r in w["streams"])
    ttft = ",".join(f"{r['ttft_s']:.3f}" for r in w["streams"])
    rate = ",".join(f"{r['decode_tok_s']:.2f}" for r in w["streams"])
    flags = ",".join(
        f"{i}:{'+'.join(r['flags'])}" for i, r in enumerate(w["streams"]) if r["flags"]
    )
    log(
        f"{tag} wall={w['wall_s']:.2f}s agg={w['agg_tok_s']:.2f} "
        f"ms_step_med={fmt_opt(w['median_ms_per_step'], '.2f')} "
        f"acc={fmt_opt(w.get('acceptance_len'), '.3f')} "
        f"d_drafts={fmt_opt(w.get('d_num_drafts'), '.0f')} sum_steps={w['sum_steps']} "
        f"xcheck={fmt_opt(w.get('xcheck_steps_over_drafts'), '.3f')} "
        f"per_stream=[{rate}] ms_step=[{ms}] completion=[{comp}] chunks=[{chunks}] "
        f"decode_s=[{dec}] ttft=[{ttft}]"
        + (f" FLAGS=[{flags}]" if flags else "")
        + (" FOREIGN" if w.get("foreign_traffic") else "")
    )


# --- warm-up (C26) ---------------------------------------------------------


def long_prompt(min_tokens: int = 8193) -> str:
    """A prompt comfortably above one 8192-token prefill chunk."""
    nonce = f"{time.time_ns():x}"
    lines = [f"Warm-up notes {nonce}. Reply with the single word OK."]
    # ~22 tokens per line on this tokenizer (measured: 1000 lines = 22457
    # tokens), so this lands near 10k tokens, above one 8192-token chunk.
    n = max(450, min_tokens // 18)
    for i in range(n):
        lines.append(f"Note {i:05d}: the ledger entry for bay {i % 97} was checked and filed.")
    return "\n".join(lines)


def warmup(url: str, model: str) -> dict:
    prompt = PHASES["prose"]
    out: dict = {"t_start": time.time()}
    log("warmup: single request")
    out["single"] = stream_one(url, model, prompt, 32)
    log("warmup: 50 ms staggered pair")
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(stream_one, url, model, prompt, 32)
        time.sleep(0.05)
        f2 = pool.submit(stream_one, url, model, prompt, 32)
        out["staggered_pair"] = [f1.result(), f2.result()]
    log("warmup: simultaneous pair")
    barrier = threading.Barrier(2)

    def fire() -> dict:
        barrier.wait()
        return stream_one(url, model, prompt, 32)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(fire) for _ in range(2)]
        out["simultaneous_pair"] = [f.result() for f in futs]
    log("warmup: long prompt (>=8193 tokens)")
    lr = stream_one(url, model, long_prompt(), 8)
    out["long"] = lr
    if lr["prompt_tokens"] < 8193:
        log(f"warmup: WARNING long prompt was only {lr['prompt_tokens']} tokens")
    out["t_end"] = time.time()
    log(f"warmup: done long_prompt_tokens={lr['prompt_tokens']} long_ttft={lr['ttft_s']:.2f}s")
    return out


# --- sentinels (G13, G16) --------------------------------------------------


class Sentinel:
    def __init__(self, args: argparse.Namespace, metrics_url: str) -> None:
        self.args = args
        self.metrics_url = metrics_url
        self.boot_min: float | None = args.boot_min_ms
        self.runs: list[dict] = []
        self.blocks: list[dict] = []
        self.void = False

    def excursion(self, ms: float | None, foreign: bool = False) -> list[str]:
        if ms is None:
            return ["no_steps"]
        why = ["foreign_traffic"] if foreign else []
        lo, hi = self.args.anchor_ms
        if lo > 0 and not (lo <= ms <= hi):
            why.append(f"anchor[{lo:g},{hi:g}]")
        if self.boot_min is not None and ms > self.boot_min * (1 + self.args.boot_tol):
            why.append(f">boot_min{self.boot_min:.2f}+{self.args.boot_tol:.0%}")
        return why

    def block(self, where: str) -> bool:
        a = self.args
        for attempt in range(1, a.sentinel_retries + 1):
            vals = []
            for i in range(a.sentinel_runs):
                w = run_wave(a.url, a.model, PHASES["structured"], a.max_tokens, 1, self.metrics_url, None)
                ms = w["streams"][0]["ms_per_step"]
                vals.append((ms, w))
            # Update the boot minimum before judging, so the first block anchors itself.
            finite = [m for m, _ in vals if m is not None]
            if finite:
                m = min(finite)
                self.boot_min = m if self.boot_min is None else min(self.boot_min, m)
            bad = False
            for i, (ms, w) in enumerate(vals):
                why = self.excursion(ms, bool(w.get("foreign_traffic")))
                self.runs.append({"where": where, "attempt": attempt, "run": i + 1,
                                  "ms_per_step": ms, "excursion": why, "wave": w})
                print_wave(f"sentinel={where} attempt={attempt} run={i+1}"
                           + (f" EXCURSION={'+'.join(why)}" if why else ""), w)
                bad = bad or bool(why)
            self.blocks.append({"where": where, "attempt": attempt, "ok": not bad,
                                "ms_per_step": [m for m, _ in vals]})
            if not bad:
                return True
            if attempt < a.sentinel_retries:
                log(f"sentinel={where} excursion; waiting {a.excursion_wait_s:.0f}s before re-run")
                time.sleep(a.excursion_wait_s)
        log(f"sentinel={where} failed {a.sentinel_retries}x: BOOT VOID")
        self.void = True
        return False

    def report(self) -> dict:
        n = len(self.runs)
        exc = sum(1 for r in self.runs if r["excursion"])
        ms = [r["ms_per_step"] for r in self.runs if r["ms_per_step"] is not None and not r["excursion"]]
        return {
            "runs": n,
            "excursions": exc,
            "excursion_rate": (exc / n) if n else None,
            "excursion_rate_over_1_in_20": bool(n) and exc / n > 0.05,
            "boot_min_ms": self.boot_min,
            "clean_median_ms": statistics.median(ms) if ms else None,
            "anchor_ms": list(self.args.anchor_ms),
            "void": self.void,
            "blocks": self.blocks,
        }


# --- expected-flag checker (plan 0-6 deterministic self-test) ---------------


def parse_expected(spec: str) -> set[tuple[str, int, int]]:
    out = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        phase, c, run = item.split(":")
        out.add((phase, int(c), int(run)))
    return out


def check_expected(cells: list[dict], expected: set[tuple[str, int, int]], k: int) -> dict:
    """Compare flagged waves to the expected set, with the plan's fallback rule.

    Healthy-index signature: in a co-prefilled N-stream wave on the unfixed pin,
    exactly ceil(N/(k+1)) rows stay healthy (rows 0..ceil(N/(k+1))-1) and the
    other N-ceil(N/(k+1)) accept zero drafts.
    """
    flagged = set()
    signature_ok = True
    unflagged_zero = []
    slow_only = []
    details = []
    for cell in cells:
        for i, w in enumerate(cell["waves"], start=1):
            key = (cell["phase"], cell["concurrency"], i)
            n = cell["concurrency"]
            zero = sum(1 for r in w["streams"] if "zero_acceptance" in r["flags"])
            # The self-test targets F01, so only its signature flags count;
            # a slow-only straggler is reported, not matched against the set.
            f01 = {f for r in w["streams"] for f in r["flags"]} & F01_FLAGS
            is_flagged = bool(f01)
            if not is_flagged and w["flags"]:
                slow_only.append(list(key))
            if is_flagged:
                flagged.add(key)
                want_zero = n - math.ceil(n / (k + 1))
                sig = zero == want_zero
                signature_ok = signature_ok and sig
                details.append({"wave": list(key), "zero_streams": zero, "signature_zero": want_zero,
                                "signature_ok": sig})
            elif zero:
                unflagged_zero.append(list(key))
    exact = flagged == expected
    fallback = signature_ok and not unflagged_zero and bool(flagged)
    return {
        "expected": sorted(list(x) for x in expected),
        "flagged": sorted(list(x) for x in flagged),
        "exact_match": exact,
        "fallback_ok": fallback,
        "pass": exact or fallback,
        "slow_only_waves": slow_only,
        "details": details,
    }


# --- main ------------------------------------------------------------------


def summarise_cell(cell: dict) -> dict:
    waves = cell["waves"]
    per_wave_ms = [w["median_ms_per_step"] for w in waves]
    per_wave_rate = [w["median_decode_tok_s"] for w in waves]
    per_wave_agg = [w["agg_tok_s"] for w in waves]
    per_wave_acc = [w.get("acceptance_len") for w in waves]
    streams = [r for w in waves for r in w["streams"]]
    ms_ok = [x for x in per_wave_ms if x is not None]
    return {
        "phase": cell["phase"],
        "concurrency": cell["concurrency"],
        "n_waves": len(waves),
        "n_streams": len(streams),
        "median_ms_per_step": statistics.median(ms_ok) if ms_ok else None,
        "min_ms_per_step": min(ms_ok) if ms_ok else None,
        "per_wave_ms_per_step": per_wave_ms,
        "median_decode_tok_s": statistics.median(r["decode_tok_s"] for r in streams),
        "min_decode_tok_s": min(r["decode_tok_s"] for r in streams),
        "per_wave_median_decode_tok_s": per_wave_rate,
        "median_agg_tok_s": statistics.median(per_wave_agg),
        "min_agg_tok_s": min(per_wave_agg),
        "per_wave_agg_tok_s": per_wave_agg,
        "per_wave_acceptance_len": per_wave_acc,
        "median_ttft_s": statistics.median(r["ttft_s"] for r in streams),
        "median_completion_tokens": statistics.median(r["completion_tokens"] for r in streams),
        "per_wave_xcheck_steps_over_drafts": [w.get("xcheck_steps_over_drafts") for w in waves],
        "foreign_waves": sum(1 for w in waves if w.get("foreign_traffic")),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="nvidia/Qwen3.8-Flash-Next-NVFP4")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 8])
    p.add_argument("--phase", choices=[*PHASES, "both"], default="both")
    p.add_argument("--json", dest="json_path", help="write the machine-readable result here")
    p.add_argument("--no-warmup", action="store_true", help="skip the discarded warm-up block (C26)")
    p.add_argument("--no-sentinels", action="store_true", help="skip structured c=1 sentinel blocks")
    p.add_argument("--sentinel-runs", type=int, default=3)
    p.add_argument("--sentinel-retries", type=int, default=3,
                   help="attempts per sentinel block before the boot is void")
    p.add_argument("--anchor-ms", type=float, nargs=2, default=[59.0, 61.0], metavar=("LO", "HI"),
                   help="absolute structured c=1 ms/step window (k=3 default); 0 0 disables")
    p.add_argument("--boot-min-ms", type=float, default=None,
                   help="seed the boot minimum (else the first sentinel block sets it)")
    p.add_argument("--boot-tol", type=float, default=0.02, help="allowed excess over boot minimum")
    p.add_argument("--excursion-wait-s", type=float, default=35.0)
    p.add_argument("--allow-foreign", action="store_true",
                   help="write SUMMARY even when other requests ran during a wave")
    p.add_argument("--k", type=int, default=3, help="num_speculative_tokens, for the self-test signature")
    p.add_argument("--historical", action="store_true",
                   help="deterministic self-test: the c4 bench order (c 1 2 8, runs 3, both "
                        "phases, 200 tokens); implies --no-warmup --no-sentinels")
    p.add_argument("--expect-flagged", default=None,
                   help="phase:c:run,... expected flagged waves; exit 0 only if the checker passes. "
                        "Defaults to the F01 set under --historical")
    args = p.parse_args()

    if args.historical:
        # Plan 0-6: the historical order runs first on the fresh boot, with no
        # warm-up and no sentinels (either would pre-run waves and shift F01).
        args.no_warmup = True
        args.no_sentinels = True
        args.concurrency = [1, 2, 8]
        args.runs = 3
        args.phase = "both"
        args.max_tokens = 200
        if args.expect_flagged is None:
            args.expect_flagged = ",".join(f"{a}:{b}:{c}" for a, b, c in HISTORICAL_EXPECTED)

    phases = list(PHASES) if args.phase == "both" else [args.phase]
    metrics_url = args.url.split("/v1/", 1)[0] + "/metrics"
    log(
        f"ruler_steps v{RULER_STEPS_VERSION} url={args.url} model={args.model} "
        f"max_tokens={args.max_tokens} runs={args.runs} concurrency={args.concurrency} "
        f"phases={phases} warmup={not args.no_warmup} sentinels={not args.no_sentinels} "
        f"anchor_ms={args.anchor_ms}"
    )
    result: dict = {
        "ruler_steps_version": RULER_STEPS_VERSION,
        "argv": sys.argv[1:],
        "started_at": now_iso(),
        "cells": [],
    }
    if spec_counters(metrics_url) is None:
        log("WARNING: no vllm:spec_decode_* counters at /metrics; acceptance and cross-checks are n/a")

    if not args.no_warmup:
        result["warmup"] = warmup(args.url, args.model)

    sentinel = None if args.no_sentinels else Sentinel(args, metrics_url)
    cells_plan = [(ph, c) for ph in phases for c in args.concurrency]
    # Middle sentinel between phases, or halfway through the cells for one phase.
    mid = len(args.concurrency) if len(phases) > 1 else max(1, len(cells_plan) // 2)
    if sentinel:
        sentinel.block("start")

    c1_ref: dict[str, float] = {}
    for idx, (phase, c) in enumerate(cells_plan):
        if sentinel and not sentinel.void and idx == mid and len(cells_plan) > 1:
            sentinel.block("middle")
        if sentinel and sentinel.void:
            break
        cell = {"phase": phase, "concurrency": c, "waves": []}
        for i in range(args.runs):
            w = run_wave(args.url, args.model, PHASES[phase], args.max_tokens, c, metrics_url,
                         c1_ref.get(phase))
            cell["waves"].append(w)
            print_wave(f"phase={phase} c={c} run={i+1}", w)
        if c == 1:
            c1_ref[phase] = statistics.median(
                r["completion_tokens"] for w in cell["waves"] for r in w["streams"]
            )
        result["cells"].append(cell)
    if sentinel and not sentinel.void:
        sentinel.block("end")

    summary = [summarise_cell(c) for c in result["cells"]]
    t0 = [
        {"phase": c["phase"], "concurrency": c["concurrency"], "run": i + 1, "flags": w["flags"]}
        for c in result["cells"] for i, w in enumerate(c["waves"]) if set(w["flags"]) & set(T0_FLAGS)
    ]
    result["t0_flagged_waves"] = t0
    result["cell_stats"] = summary
    if sentinel:
        result["sentinel"] = sentinel.report()
    result["finished_at"] = now_iso()

    rc = 0
    if args.expect_flagged is not None:
        chk = check_expected(result["cells"], parse_expected(args.expect_flagged), args.k)
        result["self_test"] = chk
        log(f"SELFTEST pass={chk['pass']} exact={chk['exact_match']} fallback={chk['fallback_ok']} "
            f"flagged={chk['flagged']} expected={chk['expected']}")
        rc = 0 if chk["pass"] else 3
    else:
        void = bool(sentinel and sentinel.void)
        foreign = [
            f"{c['phase']} c{c['concurrency']} r{i + 1}"
            for c in result["cells"] for i, w in enumerate(c["waves"]) if w.get("foreign_traffic")
        ]
        if foreign and args.allow_foreign:
            foreign = []
        if t0 or void or foreign:
            reason = []
            if t0:
                reason.append(f"{len(t0)} wave(s) with T0 flags: "
                              + "; ".join(f"{x['phase']} c{x['concurrency']} r{x['run']} {x['flags']}" for x in t0))
            if void:
                reason.append("boot void (sentinel failed)")
            if foreign:
                reason.append(f"foreign traffic during {len(foreign)} wave(s): " + ", ".join(foreign))
            log("SUMMARY REFUSED: " + " | ".join(reason))
            result["summary_refused"] = reason
            rc = 2
        else:
            result["summary"] = summary
            print("SUMMARY", json.dumps(summary, indent=2), flush=True)
    if sentinel:
        s = result["sentinel"]
        log(f"SENTINEL runs={s['runs']} excursions={s['excursions']} rate={fmt_opt(s['excursion_rate'], '.3f')} "
            f"boot_min_ms={fmt_opt(s['boot_min_ms'], '.2f')} clean_median_ms={fmt_opt(s['clean_median_ms'], '.2f')} "
            f"void={s['void']}")
    if args.json_path:
        with open(args.json_path, "w") as fh:
            json.dump(result, fh, indent=1)
            fh.write("\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
