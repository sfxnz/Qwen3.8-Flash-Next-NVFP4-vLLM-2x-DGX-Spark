#!/usr/bin/env python3
"""Offline tests for ruler_steps.py, bench_probe.py and the session tools."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import unittest
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bench_probe  # noqa: E402
import ruler_steps  # noqa: E402

BENCH_DECODE_SHA256 = "6a9c64bd821fa16574dd7b471ac7bba81ea95827a48a5b7a201daeb4f76f7c36"


def stream_row(rate: float, completion: int, chunks: int, ttft: float = 0.2) -> dict:
    """A bench_decode.stream_one row with consistent timing."""
    decode_s = (completion - 1) / rate
    return {
        "ttft_s": ttft,
        "total_s": ttft + decode_s,
        "prompt_tokens": 40,
        "completion_tokens": completion,
        "decode_tok_s": rate,
        "chunks": chunks,
    }


def healthy(phase: str, rate: float) -> dict:
    # structured: 200 tokens at acceptance 4 -> 50 steps; prose: ~93 tokens at ~2.4.
    return stream_row(rate, 200, 51) if phase == "structured" else stream_row(rate, 93, 39)


def corrupt(rate: float) -> dict:
    # Zero accepted drafts: one token per step, runs to max_tokens.
    return stream_row(rate, 200, 200)


# Per-wave per-stream rates from evidence/opt-c4-native-ctx-262144/bench.txt,
# with the F01 zero-acceptance streams marked "z".
C4 = {
    ("prose", 1): [[44.39], [43.16], [36.51]],
    ("prose", 2): [[39.94, 36.05], [38.79, "z18.84"], [36.39, 32.91]],
    ("prose", 8): [
        [27.82, 27.73, 27.51, 26.64, 23.69, 24.54, 24.94, 24.24],
        [24.84, 25.57] + ["z14.82"] * 6,
        [27.82, 27.15, 25.13, 24.77, 21.76, 22.66, 23.98, 21.99],
    ],
    ("structured", 1): [[67.16], [52.62], [55.47]],
    ("structured", 2): [[54.84, "z17.77"], [61.13, 62.02], [63.94, "z18.51"]],
    ("structured", 8): [
        [47.65] + [48.55] * 7,
        [50.17, 50.16] + ["z13.56"] * 6,
        [47.67] + [48.56] * 7,
    ],
}


def c4_cells() -> list[dict]:
    cells = []
    for (phase, c), waves in C4.items():
        cell = {"phase": phase, "concurrency": c, "waves": []}
        ref = None if c == 1 else 93 if phase == "prose" else 200
        for rates in waves:
            rows = [
                corrupt(float(r[1:])) if isinstance(r, str) else healthy(phase, r) for r in rates
            ]
            w = ruler_steps.analyse_wave(rows, 5.0, 100.0, None, None, None, None, ref, 200)
            cell["waves"].append(w)
        cells.append(cell)
    return cells


class FrozenRulerTests(unittest.TestCase):
    def test_bench_decode_is_byte_identical(self) -> None:
        got = hashlib.sha256((ROOT / "bench_decode.py").read_bytes()).hexdigest()
        self.assertEqual(got, BENCH_DECODE_SHA256)

    def test_ruler_steps_imports_frozen_functions(self) -> None:
        import bench_decode

        self.assertIs(ruler_steps.stream_one, bench_decode.stream_one)
        self.assertIs(ruler_steps.spec_counters, bench_decode.spec_counters)
        self.assertIs(ruler_steps.PHASES, bench_decode.PHASES)
        self.assertIs(ruler_steps.wave, bench_decode.wave)


class RulerStepsTests(unittest.TestCase):
    def test_ms_per_step_and_cross_check(self) -> None:
        row = stream_row(66.33, 200, 51)  # 199 decode tokens over 50 steps
        before = {"num_drafts": 10.0, "num_draft_tokens": 30.0, "num_accepted_tokens": 30.0}
        after = {"num_drafts": 60.0, "num_draft_tokens": 180.0, "num_accepted_tokens": 179.0}
        w = ruler_steps.analyse_wave([row], 3.2, 66.0, before, after, None, None, None, 200)
        s = w["streams"][0]
        self.assertAlmostEqual(s["decode_s"], 199 / 66.33, places=6)
        self.assertEqual(s["steps"], 50)
        self.assertAlmostEqual(s["ms_per_step"], 1000 * (199 / 66.33) / 50, places=6)
        self.assertAlmostEqual(w["acceptance_len"], 1 + 149 / 50)
        self.assertAlmostEqual(w["xcheck_steps_over_drafts"], 1.0)
        self.assertAlmostEqual(w["xcheck_ms_per_draft"], s["ms_per_step"])
        self.assertEqual(w["flags"], [])

    def test_zero_acceptance_slow_and_runaway_flags(self) -> None:
        rows = [healthy("prose", 38.8), corrupt(18.8)]
        w = ruler_steps.analyse_wave(rows, 10.7, 27.9, None, None, None, None, 93, 200)
        self.assertEqual(w["streams"][0]["flags"], [])
        # At N=2 the wave median sits between the two streams, so 18.8 is not
        # under 0.6x median (F04: the rate rule alone misses these); the
        # zero-acceptance and runaway flags catch it.
        self.assertEqual(set(w["streams"][1]["flags"]), {"zero_acceptance", "runaway"})
        rows8 = [healthy("prose", 25.0)] * 2 + [corrupt(14.8)] * 6
        w8 = ruler_steps.analyse_wave(rows8, 13.7, 101.7, None, None, None, None, 93, 200)
        self.assertNotIn("slow", w8["streams"][2]["flags"])  # median is a corrupt stream

    def test_short_streams_not_flagged_zero_acceptance(self) -> None:
        w = ruler_steps.analyse_wave([stream_row(40.0, 10, 10)], 1, 1, None, None, None, None, None, 200)
        self.assertNotIn("zero_acceptance", w["streams"][0]["flags"])

    def test_foreign_traffic(self) -> None:
        rows = [healthy("prose", 40.0)]
        w = ruler_steps.analyse_wave(rows, 2, 40, None, None, {"request_success": 5, "running": 0},
                                     {"request_success": 7, "running": 0}, None, 200)
        self.assertTrue(w["foreign_traffic"])

    def test_historical_c4_flags_exactly_the_f01_waves(self) -> None:
        chk = ruler_steps.check_expected(c4_cells(), set(ruler_steps.HISTORICAL_EXPECTED), k=3)
        self.assertTrue(chk["exact_match"], chk)
        self.assertTrue(chk["pass"])
        self.assertTrue(all(d["signature_ok"] for d in chk["details"]))

    def test_historical_fallback_accepts_shifted_wave_with_signature(self) -> None:
        cells = c4_cells()
        expected = set(ruler_steps.HISTORICAL_EXPECTED) - {("prose", 2, 2)} | {("prose", 2, 1)}
        chk = ruler_steps.check_expected(cells, expected, k=3)
        self.assertFalse(chk["exact_match"])
        self.assertTrue(chk["fallback_ok"])

    def test_historical_fails_without_signature(self) -> None:
        cells = c4_cells()
        # Only 3 of 6 corrupt streams in structured c8 r2: signature broken.
        w = cells[5]["waves"][1]
        for r in w["streams"][2:5]:
            r["flags"] = []
        chk = ruler_steps.check_expected(cells, {("x", 1, 1)}, k=3)
        self.assertFalse(chk["pass"])

    def test_historical_ignores_slow_only_straggler(self) -> None:
        cells = c4_cells()
        w = cells[2]["waves"][0]  # prose c8 r1, healthy
        w["streams"][4]["flags"] = ["slow"]
        w["flags"] = ["slow"]
        chk = ruler_steps.check_expected(cells, set(ruler_steps.HISTORICAL_EXPECTED), k=3)
        self.assertTrue(chk["exact_match"], chk)
        self.assertEqual(chk["slow_only_waves"], [["prose", 8, 1]])

    def test_sentinel_excursion_rules(self) -> None:
        args = argparse.Namespace(anchor_ms=[59.0, 61.0], boot_tol=0.02, boot_min_ms=59.5)
        s = ruler_steps.Sentinel(args, "http://x/metrics")
        self.assertEqual(s.excursion(60.0), [])
        self.assertTrue(s.excursion(76.0))  # c4 run2
        self.assertTrue(s.excursion(61.2))  # above absolute window and +2.9% of boot min
        args.anchor_ms = [0.0, 0.0]
        self.assertEqual(s.excursion(60.6), [])
        self.assertTrue(s.excursion(60.8))

    def test_summarise_cell_reports_median_min_and_every_wave(self) -> None:
        cell = c4_cells()[4]  # structured c=1
        s = ruler_steps.summarise_cell(cell)
        self.assertEqual(len(s["per_wave_ms_per_step"]), 3)
        self.assertEqual(s["min_ms_per_step"], min(s["per_wave_ms_per_step"]))
        self.assertIn("per_wave_agg_tok_s", s)


class FakeServer:
    """Minimal streaming /v1/chat/completions and /metrics."""

    def __init__(self, deltas: list[str], finish: str, spec: dict | None, tokens_per_delta: int = 1):
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # noqa: D401
                pass

            def do_GET(self):  # noqa: N802
                body = b"vllm:num_requests_running 0\n"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                outer.last_body = json.loads(self.rfile.read(n))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                done = 0
                for i, d in enumerate(deltas):
                    done += tokens_per_delta
                    ev = {"choices": [{"delta": {"content": d},
                                       "finish_reason": finish if i == len(deltas) - 1 else None}],
                          "usage": {"prompt_tokens": 12, "completion_tokens": done}}
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                final = {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": done}}
                if spec is not None:
                    final["metrics"] = {"speculative_decoding": spec}
                self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()


class BenchProbeTests(unittest.TestCase):
    def test_stream_probe_parses_text_finish_and_server_spec(self) -> None:
        spec = {"mean_acceptance_length": 3.9, "draft_acceptance_rate": 0.97, "acceptance_histogram": [0, 0, 1, 20],
                "num_spec_steps": 21, "num_accepted_draft_tokens": 61, "num_draft_tokens": 63, "num_spec_tokens": 3}
        srv = FakeServer([f"w{i} " for i in range(40)], "stop", spec)
        try:
            r = bench_probe.stream_probe(srv.url, "m", bench_probe.text_messages("hi"), 64)
        finally:
            srv.close()
        self.assertEqual(r["finish_reason"], "stop")
        self.assertEqual(r["chunks"], 40)
        self.assertEqual(r["completion_tokens"], 40)
        self.assertEqual(r["spec_source"], "server")
        self.assertEqual(r["steps"], 21)
        self.assertEqual(r["acceptance_len"], 3.9)
        self.assertEqual(r["text"], "".join(f"w{i} " for i in range(40)))
        self.assertEqual(r["prefix_text"], "".join(f"w{i} " for i in range(32)))
        self.assertTrue(srv.last_body["stream_options"]["continuous_usage_stats"])
        self.assertEqual(srv.last_body["temperature"], 0)

    def test_stream_probe_falls_back_to_chunk_acceptance(self) -> None:
        srv = FakeServer(["x"] * 30, "length", None, tokens_per_delta=1)
        try:
            r = bench_probe.stream_probe(srv.url, "m", bench_probe.text_messages("hi"), 30)
        finally:
            srv.close()
        self.assertEqual(r["spec_source"], "chunks")
        self.assertEqual(r["acceptance_len"], 1.0)
        gate, _ = bench_probe.flag_stream(r, None, r["decode_tok_s"])
        self.assertIn("zero_acceptance", gate)

    def test_flag_stream_runaway_and_divergence(self) -> None:
        ref = {"finish_reason": "stop", "prefix_text": "The answer is"}
        row = {"steps": 199, "acceptance_len": 1.0, "finish_reason": "length", "text": "Theductduct",
               "decode_tok_s": 18.0}
        gate, report = bench_probe.flag_stream(row, ref, 40.0)
        self.assertEqual(set(gate), {"zero_acceptance", "runaway"})
        self.assertEqual(set(report), {"early_divergence", "slow"})
        ok = {"steps": 40, "acceptance_len": 2.4, "finish_reason": "stop", "text": "The answer is 42.",
              "decode_tok_s": 39.0}
        self.assertEqual(bench_probe.flag_stream(ok, ref, 40.0), ([], []))

    def test_slow_gates_only_for_identical_bursts(self) -> None:
        row = {"steps": 40, "acceptance_len": 2.4, "finish_reason": "stop", "text": "x", "decode_tok_s": 10.0}
        self.assertEqual(bench_probe.flag_stream(dict(row), None, 40.0, slow_gates=True), (["slow"], []))
        self.assertEqual(bench_probe.flag_stream(dict(row), None, 40.0), ([], ["slow"]))

    def test_statistical_self_test(self) -> None:
        def b(n, flagged):
            s = [{"gate_flags": []}] + [{"gate_flags": ["zero_acceptance"] if flagged else []} for _ in range(n - 1)]
            return {"stagger_s": 0, "kind": "identical", "n": n, "streams": s,
                    "gate_flags": ["zero_acceptance"] if flagged else []}

        bursts = [b(2, False), b(2, True), b(8, True), b(5, False), b(3, False)]
        st = bench_probe.statistical_self_test(bursts, 3)
        self.assertTrue(st["pass"])  # 2 of 4 non-first flagged, row 0 healthy
        bursts = [b(2, False)] + [b(8, False)] * 4 + [b(8, True)]
        self.assertFalse(bench_probe.statistical_self_test(bursts, 3)["pass"])  # 1/5 = 20%

    def test_png_is_valid(self) -> None:
        import base64

        raw = base64.b64decode(bench_probe.png_solid((1, 2, 3), 4))
        self.assertTrue(raw.startswith(b"\x89PNG\r\n\x1a\n"))
        idat = raw[raw.index(b"IDAT") + 4: raw.index(b"IEND") - 8]
        self.assertEqual(zlib.decompress(idat), (b"\x00" + bytes((1, 2, 3)) * 4) * 4)

    def test_long_prompt_is_cold_and_answer_stable(self) -> None:
        a = bench_probe.long_text_prompt(1, "n1")
        b = bench_probe.long_text_prompt(1, "n2")
        self.assertNotEqual(a, b)
        self.assertEqual(a.splitlines()[1:], b.splitlines()[1:])
        # 500 lines measured at 11968 prompt tokens on the live serve.
        self.assertGreaterEqual(len(a.splitlines()), 500)


class ToolScriptTests(unittest.TestCase):
    def test_shell_syntax(self) -> None:
        for rel in ("tools/telemetry.sh", "tools/session_gate.sh"):
            subprocess.run(["bash", "-n", str(ROOT / rel)], check=True)

    def test_telemetry_never_queries_gpu_memory(self) -> None:
        src = (ROOT / "tools/telemetry.sh").read_text()
        fields = src[src.index("GPU_FIELDS = ["): src.index("]", src.index("GPU_FIELDS = ["))]
        self.assertNotIn("memory.", fields)
        self.assertIn("taskset -c 0 nice -n 10", src)
        self.assertIn("os.fsync", src)

    def test_cli_help(self) -> None:
        for rel in ("ruler_steps.py", "bench_probe.py"):
            out = subprocess.run([sys.executable, str(ROOT / rel), "--help"], capture_output=True, text=True,
                                 check=True).stdout
            self.assertIn("--json", out)


if __name__ == "__main__":
    unittest.main()
