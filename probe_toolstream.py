#!/usr/bin/env python3
"""Streamed tool-call cost probe (plan 0-7; finding G05).

G05: streaming qwen3_xml tool-call arguments costs O(n^2) CPU in the single
API-server process. This probe streams one thinking-off write_file tool call
whose `content` argument is a large HTML page (8-16k tokens, max_tokens
--max-tokens), while one plain text stream (ignore_eos) decodes alongside it.

Reported:
  - tool stream: completion tokens, final argument length, inter-chunk gap
    median per quarter of the stream (growth = per-delta parse cost growing)
  - APIServer CPU: utime+stime of the container's PID 1 (the API server, which
    also runs the tool parser) sampled from /proc/<host pid>/stat every
    --cpu-period s, per quarter; plus one `docker top` snapshot. Read-only.
  - text stream: inter-chunk gap p50/p99/max before the tool call starts
    streaming arguments and during each quarter of it.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from rulers import common as C

WRITE_FILE = C.load_prompts("diverse")["tools"]["write_file"]
PROMPT = (
    "Build a complete, self-contained single-file HTML website for a fictional mountain railway museum: "
    "a large inline <style> block, a navigation bar, at least twelve content sections (history, timeline "
    "table with 40 rows, locomotive gallery with captions, FAQ with 20 questions, visitor information, "
    "opening-hours table, ticket price table, events calendar, staff directory, accessibility notes, "
    "gift shop catalogue with 30 items, contact form) and an inline <script> with form validation and a "
    "live clock. Write every section in full, with no placeholders. "
    "Save it by calling the write_file tool once with path index.html and the whole page as content."
)


class ProcCPU(threading.Thread):
    """Sample one process's CPU time (utime+stime, excluding children) from /proc."""

    def __init__(self, pid: int, period: float, t_ref: float) -> None:
        super().__init__(daemon=True)
        self.pid, self.period, self.t_ref = pid, period, t_ref
        self.samples: list[tuple[float, float]] = []  # (t, cpu seconds)
        self.halt = threading.Event()
        self.tck = os.sysconf("SC_CLK_TCK")

    def read(self) -> float | None:
        try:
            with open(f"/proc/{self.pid}/stat") as fh:
                fields = fh.read().rsplit(")", 1)[1].split()
            return (int(fields[11]) + int(fields[12])) / self.tck
        except (OSError, IndexError, ValueError):
            return None

    def run(self) -> None:
        while not self.halt.is_set():
            v = self.read()
            if v is not None:
                self.samples.append((time.perf_counter() - self.t_ref, v))
            self.halt.wait(self.period)

    def pct_between(self, a: float, b: float) -> float | None:
        pts = [s for s in self.samples if a <= s[0] <= b]
        if len(pts) < 2 or pts[-1][0] <= pts[0][0]:
            return None
        return 100.0 * (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])


def api_server_pid(container: str) -> tuple[int | None, str]:
    out = C.run_cmd(["docker", "inspect", "-f", "{{.State.Pid}}", container]).strip()
    try:
        pid = int(out)
    except ValueError:
        return None, f"docker inspect failed: {out[:200]}"
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            cmd = fh.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError as exc:
        cmd = f"unreadable: {exc}"
    return (pid if pid > 0 else None), cmd[:200]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(p)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--container", default="qwen38-flash-next-nvfp4")
    p.add_argument("--cpu-period", type=float, default=1.0)
    p.add_argument("--no-text-stream", action="store_true")
    p.add_argument("--timeout", type=float, default=1800)
    args = p.parse_args()
    C.require_health(args.url)
    em = C.Emitter(__file__, args.out, label=args.label)
    salt = uuid.uuid4().hex[:8]
    t_ref = time.perf_counter()

    pid, cmdline = api_server_pid(args.container)
    top = C.run_cmd(["docker", "top", args.container, "-eo", "pid,pcpu,rss,comm"])[:4000]
    cpu = ProcCPU(pid, args.cpu_period, t_ref) if pid else None
    if cpu:
        cpu.start()

    stop = threading.Event()
    pool = ThreadPoolExecutor(max_workers=2)
    text_f = None
    if not args.no_text_stream:
        tb = C.chat_body(args.model, f"[{salt}] Write a very long travel diary about a coastal railway.",
                         args.max_tokens, ignore_eos=True)
        text_f = pool.submit(C.stream_chat, args.url, tb, timeout=args.timeout, stop=stop, t_ref=t_ref)
        time.sleep(3.0)
    body = C.chat_body(args.model, f"[{salt}] {PROMPT}", args.max_tokens, tools=[WRITE_FILE], tool_choice="auto")
    tool = C.stream_chat(args.url, body, timeout=args.timeout, t_ref=t_ref)
    stop.set()
    text = text_f.result() if text_f else None
    pool.shutdown(wait=True)
    if cpu:
        cpu.halt.set()
        cpu.join()

    s = C.step_stats(tool)
    t_abs = [tool["t_send"] + t for t in tool["times"]]
    # arguments phase: first chunk that grew the argument string -> end of stream
    arg_t = [t for t, n in zip(t_abs[-len(tool["tool_arg_len"]):], tool["tool_arg_len"]) if n > 0] if tool["tool_arg_len"] else []
    quarters = []
    if len(arg_t) >= 8:
        a0, a1 = arg_t[0], arg_t[-1]
        edges = [a0 + (a1 - a0) * k / 4 for k in range(5)]
        for q in range(4):
            lo, hi = edges[q], edges[q + 1]
            tg = C.gaps_in_window(arg_t, lo, hi)
            row = {"q": q + 1, "tool_gap_p50_ms": C.rnd(C.pct(tg, 50)),
                   "api_cpu_pct": C.rnd(cpu.pct_between(lo, hi)) if cpu else None}
            if text:
                xg = C.gaps_in_window([text["t_send"] + t for t in text["times"]], lo, hi)
                row.update(text_gap_p50_ms=C.rnd(C.pct(xg, 50)), text_gap_p99_ms=C.rnd(C.pct(xg, 99)),
                           text_gap_max_ms=C.rnd(max(xg) if xg else None))
            quarters.append(row)
    pre = {}
    if text and arg_t:
        xg = C.gaps_in_window([text["t_send"] + t for t in text["times"]], 0.0, arg_t[0])
        pre = {"text_gap_before_args_p50_ms": C.rnd(C.pct(xg, 50)), "text_gap_before_args_p99_ms": C.rnd(C.pct(xg, 99))}
    calls = tool["tool_calls"]
    out = em.row(
        "probe", api_pid=pid, api_cmdline=cmdline, docker_top=top,
        tool_name=calls[0]["name"] if calls else None,
        arg_chars=len(calls[0]["arguments"]) if calls else 0,
        tool_chunks=tool["tool_chunks"], content_chunks=tool["content_chunks"],
        api_cpu_pct_whole=C.rnd(cpu.pct_between(arg_t[0], arg_t[-1])) if cpu and arg_t else None,
        text_error=text["error"] if text else None, quarters=quarters, **pre, **s,
    )
    em.summary({k: out.get(k) for k in ("tool_name", "arg_chars", "completion_tokens", "api_cpu_pct_whole",
                                        "quarters", "text_gap_before_args_p50_ms", "error")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
