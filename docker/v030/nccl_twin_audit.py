#!/usr/bin/env python3
"""L2a engagement audit: did the eager twin engage, and did NCCL really run the graph comm with mixing off?

Run on the saved `docker logs` of each rank after /v1/models is up AND after some traffic (so the
twin has routed eager calls). The boot must set NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ENV so NCCL prints
its own per-call line; with NCCL_DEBUG=WARN it prints nothing (the sibling's "0 of 434").

    python3 docker/v030/nccl_twin_audit.py --mode strict [--graphs N] [--expect-captured T] \
        head=<head.log> worker=<worker.log>

Per rank it checks:
  engaged      exactly one "qwen38: nccl twin engaged" line (the TP group), no DISARMED/REFUSED line
  graphs       one "graph" line per captured CUDA graph, total captured collectives > 0
               (--graphs N: exactly N graph lines = capture sizes x graph kinds, target and drafter;
               --expect-captured T: T captured PyNccl calls in total)
  eager        at least one "eager" line (the twin carried eager collectives)
  nccl-env     NCCL's "NCCL_GRAPH_MIXING_SUPPORT set by environment to 0" line: the env reached NCCL
               before its first communicator (else the write was ignored and mixing stayed on)
  nccl-lines   NCCL's "graphUsageMode is set to 0 but the user is capturing graphs" count equals the
               captured NCCL API calls the router counted (fewer: the graph comm is not in mode 0;
               more: something else captured a collective on a mode-0 communicator)
Across ranks: the per-graph captured counts are identical (a mismatch means the ranks captured
different collective sequences).
--mode warn prints WARNING lines and exits 0; --mode strict exits 1 on any problem.
Markers are read from nccl_twin.py with ast; nothing is imported.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parent / "nccl_twin.py"
NAMES = ("LOG_ENGAGED", "LOG_GRAPH", "LOG_EAGER", "LOG_DISARMED", "NCCL_ENV_LINE", "NCCL_CAPTURE_LINE")


def markers(path: Path = MODULE) -> dict:
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", "")
            if name in NAMES:
                out[name] = ast.literal_eval(node.value)
    missing = set(NAMES) - set(out)
    if missing:
        raise RuntimeError(f"nccl_twin.py lacks {sorted(missing)}")
    return out


COUNTS = r"on (?P<group>\S+) rank=(?P<rank>\d+) eager=(?P<eager>\d+) eager_nccl=(?P<eager_nccl>\d+) " \
         r"total_captured=(?P<total>\d+) total_captured_nccl=(?P<total_nccl>\d+) graphs=(?P<graphs>\d+)"


def parse(text: str, m: dict) -> dict:
    graph_re = re.compile(re.escape(m["LOG_GRAPH"]) + r" (?P<index>\d+) captured=\+(?P<delta>\d+) "
                          r"captured_nccl=\+(?P<delta_nccl>\d+) " + COUNTS)
    eager_re = re.compile(re.escape(m["LOG_EAGER"]) + " " + COUNTS)
    r = {"engaged": 0, "disarm": [], "graphs": [], "eager": 0, "total": 0, "total_nccl": 0,
         "nccl_env": 0, "nccl_capture": 0}
    for line in text.splitlines():
        if m["LOG_ENGAGED"] in line:
            r["engaged"] += 1
        if any(d in line for d in m["LOG_DISARMED"]):
            r["disarm"].append(line.strip()[:240])
        if m["NCCL_ENV_LINE"] in line:
            r["nccl_env"] += 1
        if m["NCCL_CAPTURE_LINE"] in line:
            r["nccl_capture"] += 1
        g = graph_re.search(line)
        e = None if g else eager_re.search(line)
        hit = g or e
        if hit:
            r["total"] = max(r["total"], int(hit["total"]))
            r["total_nccl"] = max(r["total_nccl"], int(hit["total_nccl"]))
        if g:
            r["graphs"].append((int(g["index"]), int(g["delta"]), int(g["delta_nccl"])))
        if e:
            r["eager"] = max(r["eager"], int(e["eager"]))
    return r


def audit(logs: dict, graphs: int | None = None, expect_captured: int | None = None,
          m: dict | None = None) -> tuple[list, dict]:
    """(problems, {rank: parsed}); no problems means every check passed on every rank."""
    m = m or markers()
    problems, parsed = [], {}
    for rank, text in logs.items():
        r = parsed[rank] = parse(text, m)
        if r["engaged"] != 1:
            problems.append(f"{rank}: engaged: {r['engaged']} '{m['LOG_ENGAGED']}' lines, want 1")
        problems += [f"{rank}: {line}" for line in r["disarm"]]
        if not r["graphs"] or r["total"] == 0:
            problems.append(f"{rank}: graphs: {len(r['graphs'])} graph lines, {r['total']} captured collectives")
        if graphs is not None and len(r["graphs"]) != graphs:
            problems.append(f"{rank}: graphs: {len(r['graphs'])} graph lines, want {graphs}")
        if expect_captured is not None and r["total"] != expect_captured:
            problems.append(f"{rank}: graphs: {r['total']} captured collectives, want {expect_captured}")
        if r["eager"] == 0:
            problems.append(f"{rank}: eager: no '{m['LOG_EAGER']}' line (no eager call reached the twin yet?)")
        if r["nccl_env"] == 0:
            problems.append(f"{rank}: nccl-env: no '{m['NCCL_ENV_LINE']}' line (boot without NCCL_DEBUG=INFO "
                            "NCCL_DEBUG_SUBSYS=ENV, or the env was written after NCCL read it)")
        if r["nccl_capture"] != r["total_nccl"]:
            problems.append(f"{rank}: nccl-lines: {r['nccl_capture']} NCCL capture lines vs "
                            f"{r['total_nccl']} captured NCCL calls counted by the router")
    seqs = {rank: [g[1:] for g in r["graphs"]] for rank, r in parsed.items()}
    if len({tuple(s) for s in seqs.values()}) > 1:
        problems.append(f"ranks: per-graph captured counts differ: {seqs}")
    return problems, parsed


def summary(rank: str, r: dict) -> str:
    hist: dict = {}
    for _, delta, _ in r["graphs"]:
        hist[delta] = hist.get(delta, 0) + 1
    per = ", ".join(f"{k} x{v}" for k, v in sorted(hist.items()))
    return (f"{rank}: engaged={r['engaged']} graphs={len(r['graphs'])} [collectives per graph: {per or '-'}] "
            f"captured={r['total']} captured_nccl={r['total_nccl']} nccl_lines={r['nccl_capture']} "
            f"nccl_env={r['nccl_env']} eager_on_twin={r['eager']}")


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("warn", "strict"), default="strict")
    ap.add_argument("--graphs", type=int, help="expected captured CUDA graphs per rank")
    ap.add_argument("--expect-captured", type=int, help="expected captured PyNccl calls per rank")
    ap.add_argument("logs", nargs="+", metavar="RANK=LOGFILE")
    args = ap.parse_args(argv)
    logs = {}
    for item in args.logs:
        rank, _, path = item.partition("=")
        try:
            logs[rank] = Path(path).read_text(errors="replace")
        except OSError as exc:
            logs[rank] = ""
            print(f"WARNING audit {rank}: cannot read {path}: {exc}", file=sys.stderr)
    problems, parsed = audit(logs, args.graphs, args.expect_captured)
    for rank, r in parsed.items():
        print(summary(rank, r))
    for p in problems:
        print(f"WARNING audit {p}", file=sys.stderr)
    if not problems:
        print(f"==> nccl twin audit ok on {', '.join(logs)}")
    return 1 if problems and args.mode == "strict" else 0


if __name__ == "__main__":
    sys.exit(main())
