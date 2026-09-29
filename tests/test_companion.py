#!/usr/bin/env python3
"""Offline tests for the companion rulers (plan 0-7). No live server needed:
streaming is exercised against an in-process fake SSE server."""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import threading
import time
import unittest
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bench_decode  # noqa: E402
from rulers import common as C  # noqa: E402

SCRIPTS = (
    "bench_diverse.py",
    "bench_sampled.py",
    "bench_structured.py",
    "bench_mixed.py",
    "bench_longctx.py",
    "bench_vision.py",
    "probe_prefix.py",
    "probe_toolstream.py",
)


def _sse(ev: dict) -> bytes:
    return f"data: {json.dumps(ev)}\n\n".encode()


class _Fake(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path == "/metrics":
            body = (
                "# HELP x\nvllm:prefix_cache_hits_total{engine=\"0\"} 1600.0\n"
                "vllm:prefix_cache_queries_total{engine=\"0\"} 3200.0\n"
                "vllm:spec_decode_num_drafts_total 10.0\nvllm:spec_decode_num_draft_tokens_total 30.0\n"
                "vllm:spec_decode_num_accepted_tokens_total 12.0\n"
            ).encode()
        else:
            body = b"ok"
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        if req.get("max_tokens") == 999:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "bad request"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        n = req.get("max_tokens", 5)
        if req.get("tools"):
            deltas = [{"tool_calls": [{"index": 0, "function": {"name": "write_file", "arguments": ""}}]}]
            deltas += [{"tool_calls": [{"index": 0, "function": {"arguments": p}}]} for p in ('{"path":', '"a"', "}")]
        elif req.get("chat_template_kwargs", {}).get("enable_thinking"):
            deltas = [{"reasoning": "hm"}, {"reasoning": "m"}] + [{"content": "x"}] * 3
        else:
            deltas = [{"content": f"t{i}"} for i in range(n)]
        try:
            for d in deltas:
                self.wfile.write(_sse({"choices": [{"index": 0, "delta": d}]}))
                self.wfile.flush()
                time.sleep(0.01)
            self.wfile.write(_sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}))
            self.wfile.write(_sse({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2 * len(deltas)}}))
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client aborted (stop test)


class CompanionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}/v1/chat/completions"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    # ---- freeze
    def test_ruler_set_matches_tree(self):
        self.assertEqual(C.drift(), [], "run `python3 rulers/common.py --write --version N` after an intended change")

    def test_ruler_set_lists_every_frozen_file(self):
        version, shas = C.read_manifest()
        self.assertTrue(version and version != "0")
        self.assertEqual(sorted(shas), sorted(C.FROZEN))
        for s in SCRIPTS:
            self.assertIn(s, C.FROZEN)

    def test_identity_carries_version(self):
        ident = C.ruler_identity(str(ROOT / "bench_diverse.py"))
        self.assertEqual(ident["ruler"], "bench_diverse")
        self.assertEqual(ident["ruler_set"], C.read_manifest()[0])

    def test_scripts_help(self):
        for s in SCRIPTS:
            r = subprocess.run([sys.executable, str(ROOT / s), "--help"], capture_output=True, text=True, cwd="/")
            self.assertEqual(r.returncode, 0, f"{s}: {r.stderr}")
            self.assertIn("--url", r.stdout)

    # ---- prompts
    def test_diverse_prompt_set(self):
        d = C.load_prompts("diverse")
        self.assertEqual(len(d["prose"]), 32)
        self.assertEqual(len(d["structured"]), 32)
        ids = [x["id"] for x in d["prose"] + d["structured"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len({x["prompt"] for x in d["prose"] + d["structured"]}), 64)
        kinds = {x["kind"] for x in d["structured"]}
        self.assertTrue({"code", "json", "cjk", "tool"} <= kinds)
        for x in d["structured"]:
            if x.get("tool"):
                self.assertIn(x["tool"], d["tools"])
        items = C.diverse_items(("structured",))
        self.assertTrue(any(i.get("tools") for i in items))

    def test_structured_schemas(self):
        s = C.load_prompts("structured")
        self.assertEqual(len(s["catalog_prompts"]), 8)
        self.assertEqual(s["large_maxlength_schema"]["properties"]["body"]["maxLength"], 100000)

    # ---- helpers
    def test_pct_and_seed(self):
        self.assertEqual(C.pct([1, 2, 3, 4], 50), 2.5)
        self.assertEqual(C.pct([5], 99), 5.0)
        self.assertIsNone(C.pct([], 50))
        self.assertEqual(C.stable_seed("p01", 0), C.stable_seed("p01", 0))
        self.assertNotEqual(C.stable_seed("p01", 0), C.stable_seed("p01", 1))

    def test_parse_prom_suffix(self):
        m = C.parse_prom('vllm:prefix_cache_hits_total{a="1"} 3\nvllm:prefix_cache_hits_total{a="2"} 4\nx_bucket{le="1"} 9\n')
        self.assertEqual(C.mget(m, "vllm:prefix_cache_hits"), 7.0)
        self.assertNotIn("x_bucket", m)

    def test_grid_png_is_valid(self):
        w, h = 37, 23
        png = C.grid_png(w, h, salt=1)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        pos, raw = 8, b""
        while pos < len(png):
            (n,) = struct.unpack(">I", png[pos : pos + 4])
            tag, data = png[pos + 4 : pos + 8], png[pos + 8 : pos + 8 + n]
            (crc,) = struct.unpack(">I", png[pos + 8 + n : pos + 12 + n])
            self.assertEqual(crc, zlib.crc32(tag + data) & 0xFFFFFFFF)
            if tag == b"IHDR":
                self.assertEqual(struct.unpack(">II", data[:8]), (w, h))
            if tag == b"IDAT":
                raw += data
            pos += 12 + n
        self.assertEqual(len(zlib.decompress(raw)), h * (1 + 3 * w))
        self.assertNotEqual(C.grid_png(w, h, salt=1), C.grid_png(w, h, salt=2))

    def test_filler_and_needles_offline(self):
        self.assertEqual(C.filler_text(50, "a"), C.filler_text(50, "a"))
        self.assertNotEqual(C.filler_text(50, "a"), C.filler_text(50, "b"))
        text, n = C.long_prompt(self.url, "m", 500, "s", question="Q?", needles=[(0.5, "NEEDLE-1.")])
        self.assertIsNone(n)  # fake server has no /tokenize: 1.3 tokens/word fallback
        self.assertIn("NEEDLE-1.", text)
        self.assertTrue(text.startswith("[doc s]") and text.endswith("Q?"))

    def test_expected_tokens(self):
        import bench_vision

        self.assertEqual(bench_vision.expected_tokens(4608, 3456), 15552)

    def test_binned_slope(self):
        import bench_structured

        times, t = [], 0.0
        for i in range(400):
            t += 0.05 + i * 1e-5
            times.append(t)
        bins, slope = bench_structured.binned(times, 100)
        self.assertEqual(len(bins), 4)
        self.assertAlmostEqual(slope, 10.0, places=3)  # 1e-5 s/step = 10 ms per 1000 steps

    # ---- streaming against the fake server
    def test_stream_parity_with_bench_decode(self):
        ref = bench_decode.stream_one(self.url, "m", "hi", 6)
        row = C.stream_chat(self.url, C.chat_body("m", "hi", 6))
        s = C.step_stats(row)
        self.assertEqual(s["chunks"], ref["chunks"])
        self.assertEqual(s["completion_tokens"], ref["completion_tokens"])
        self.assertEqual(s["prompt_tokens"], ref["prompt_tokens"])
        self.assertEqual(row["finish_reason"], "length")
        self.assertGreater(s["ms_per_step"], 5.0)
        self.assertAlmostEqual(s["tokens_per_step"], (12 - 1) / 5)

    def test_stream_tool_and_reasoning(self):
        row = C.stream_chat(self.url, {"model": "m", "messages": [], "tools": [{}], "max_tokens": 5})
        self.assertEqual(row["tool_calls"], [{"name": "write_file", "arguments": '{"path":"a"}'}])
        self.assertEqual(row["tool_arg_len"], [0, 8, 11, 12])
        row = C.stream_chat(self.url, C.chat_body("m", "hi", 5, thinking=True))
        self.assertEqual((row["reasoning_chunks"], row["content_chunks"]), (2, 3))
        self.assertEqual(len(row["times"]), 5)

    def test_stream_error_and_stop(self):
        row = C.stream_chat(self.url, C.chat_body("m", "hi", 999))
        self.assertTrue(row["error"].startswith("HTTP 400"))
        ev = threading.Event()
        ev.set()
        row = C.stream_chat(self.url, C.chat_body("m", "hi", 50), stop=ev)
        self.assertTrue(row["aborted"])

    def test_metrics_helpers(self):
        m = C.scrape(self.url)
        self.assertEqual(C.mget(m, "vllm:prefix_cache_hits"), 1600.0)
        self.assertEqual(C.spec_from_scrape(m), bench_decode.spec_counters(C.metrics_url(self.url)))
        self.assertTrue(C.health_ok(self.url))

    def test_emitter_writes_jsonl(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "x.jsonl"
            em = C.Emitter(str(ROOT / "probe_prefix.py"), str(out), label="t")
            em.row("request", a=1)
            rec = json.loads(out.read_text().splitlines()[0])
            self.assertEqual((rec["ruler"], rec["kind"], rec["label"], rec["a"]), ("probe_prefix", "request", "t", 1))
            self.assertIn("ruler_set", rec)


if __name__ == "__main__":
    unittest.main()
