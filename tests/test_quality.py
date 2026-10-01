#!/usr/bin/env python3
"""Offline tests for quality/ (plan 0-8 and the 1.3 gates). No live serve needed:
a fake OpenAI server drives the T1 capture end to end."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import struct
import sys
import tempfile
import threading
import unittest
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "quality"
sys.path.insert(0, str(Q))
sys.path.insert(0, str(Q / "vision_probes"))

import collect_t1  # noqa: E402
import probes  # noqa: E402
import qlib  # noqa: E402
import score  # noqa: E402
import t2  # noqa: E402
import t3_needles  # noqa: E402

THINK_END = 248069


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def kl_scalar(p_ids, p_lp, q_ids, q_lp, extra=None) -> float:
    """Reference scalar KL (the GLM-5.3 tier0 kl_topk), to pin the vectorised one."""
    qmap = dict(zip(q_ids, q_lp))
    for k, v in (extra or {}).items():
        qmap.setdefault(k, v)
    q_top = sum(math.exp(x) for x in q_lp)
    missing = [i for i in p_ids if i not in qmap]
    fill = min(math.exp(min(q_lp)), max(0.0, 1.0 - q_top) / (len(missing) + 1)) if missing else 0.0
    kl = ps = qs = 0.0
    for i, lp in zip(p_ids, p_lp):
        p = math.exp(lp)
        q = max(math.exp(qmap[i]) if i in qmap else fill, 1e-12)
        kl += p * (lp - math.log(q))
        ps += p
        qs += q
    pr, qr = max(0.0, 1 - ps), max(1e-12, 1 - qs)
    if pr > 1e-12:
        kl += pr * math.log(pr / qr)
    return max(0.0, kl)


class TestStats(unittest.TestCase):
    def test_kl_identical_is_zero(self):
        ids = np.array([[5, 7, 9, 11, 13]])
        lp = np.log(np.array([[0.5, 0.2, 0.1, 0.05, 0.05]]))
        self.assertAlmostEqual(float(score.kl_topk(ids, lp, ids, lp)[0]), 0.0, places=12)

    def test_kl_matches_scalar_reference(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            p_ids = rng.choice(20, 5, replace=False)
            q_ids = rng.choice(20, 5, replace=False)
            p_lp = np.sort(np.log(rng.dirichlet(np.ones(8))[:5]))[::-1]
            q_lp = np.sort(np.log(rng.dirichlet(np.ones(8))[:5]))[::-1]
            tgt, tgt_lp = int(p_ids[2]), float(np.log(0.01))
            want = kl_scalar(list(p_ids), list(p_lp), list(q_ids), list(q_lp), {tgt: tgt_lp})
            got = score.kl_topk(p_ids[None], p_lp[None], q_ids[None], q_lp[None], [tgt], [tgt_lp])[0]
            self.assertAlmostEqual(float(got), want, places=9)

    def test_mcnemar_exact(self):
        m = qlib.mcnemar([True] * 5 + [False] * 5, [False] * 5 + [False] * 5)
        self.assertEqual((m["lost"], m["gained"]), (5, 0))
        self.assertAlmostEqual(m["p"], 0.0625)
        self.assertAlmostEqual(m["drop_pp"], 50.0)
        self.assertEqual(qlib.mcnemar([True, False], [True, False])["p"], 1.0)
        self.assertEqual(score.mcnemar_counts(5, 0), 0.0625)

    def test_threshold_rule_c25(self):
        self.assertEqual(qlib.threshold_lower(0.005, None), 0.005)
        self.assertEqual(qlib.threshold_lower(0.005, 0.004), 0.008)
        self.assertEqual(qlib.threshold_lower(0.005, 0.001), 0.005)
        self.assertEqual(qlib.threshold_agree(0.995, 0.999), 0.995)
        self.assertAlmostEqual(qlib.threshold_agree(0.995, 0.996), 0.992)


def write_t1_run(d: Path, items: dict, k: int = 5) -> None:
    """items: id -> (domain, top_ids [n,k], top_lp [n,k], tgt [n], tgt_lp [n])."""
    d.mkdir(parents=True, exist_ok=True)
    arrs, off = {"tgt": [], "tgt_lp": [], "top_ids": [], "top_lp": []}, 0
    with (d / "index.jsonl").open("w") as f:
        for i, (dom, ti, tl, tg, tgl) in items.items():
            f.write(json.dumps({"id": i, "domain": dom, "shard": 0, "off": off, "n": len(tg), "prompt_ids_sha": "x"}) + "\n")
            off += len(tg)
            for key, v in zip(("top_ids", "top_lp", "tgt", "tgt_lp"), (ti, tl, tg, tgl)):
                arrs[key].append(v)
    np.savez_compressed(d / "t1-000.npz", **{key: np.concatenate(v) for key, v in arrs.items()})


def fake_items(seed: int, n: int = 200, flip: float = 0.0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for dom in ("prose", "code"):
        ti = np.tile(np.arange(5, dtype=np.int32), (n, 1)) + rng.integers(0, 1000, (n, 1)).astype(np.int32) * 5
        tl = np.log(np.tile(np.array([0.6, 0.2, 0.1, 0.05, 0.03], dtype=np.float32), (n, 1)))
        tg = ti[:, 0].copy()
        if flip:
            sw = rng.random(n) < flip
            ti[sw, 0], ti[sw, 1] = ti[sw, 1].copy(), ti[sw, 0].copy()
        out[f"{dom}.0"] = (dom, ti, tl, tg, tl[:, 0].copy())
    return out


class TestScoreT1(unittest.TestCase):
    def test_identical_passes_perturbed_fails(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            write_t1_run(t / "a", fake_items(1))
            write_t1_run(t / "b", fake_items(1))
            write_t1_run(t / "c", fake_items(1, flip=0.05))
            same = score.score_t1(t / "a", t / "b", "A", None)
            self.assertEqual(same["verdict"], "PASS", same)
            self.assertEqual(same["metrics"]["top1_agree"], 1.0)
            bad = score.score_t1(t / "a", t / "c", "A", None)
            self.assertEqual(bad["verdict"], "FAIL")
            self.assertFalse(bad["checks"]["top1"])
            self.assertGreater(bad["metrics"]["kl"], 0.0)

    def test_escalate_class_b(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            write_t1_run(t / "a", fake_items(1))
            write_t1_run(t / "b", fake_items(1))
            aa = {"metrics": {"top1_agree": 0.98, "kl": 0.001}}  # 1 - 2 x 0.02 = 0.96 < 0.97
            out = score.score_t1(t / "a", t / "b", "B", aa)
            self.assertEqual(out["verdict"], "ESCALATE")
            self.assertFalse(out["pass"])

    def test_missing_items_invalid(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            items = fake_items(1)
            write_t1_run(t / "a", items)
            write_t1_run(t / "b", {k: v for k, v in items.items() if k.startswith("prose")})
            self.assertEqual(score.score_t1(t / "a", t / "b", "A", None)["verdict"], "INVALID")


def write_t1g(d: Path, seqs: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    with (d / "t1g.jsonl").open("w") as f:
        for pid, reps in seqs.items():
            for r, s in enumerate(reps):
                f.write(json.dumps({"id": pid, "repeat": r, "token_ids": s}) + "\n")


class TestScoreT1GD(unittest.TestCase):
    def test_t1g(self):
        base = list(range(40))
        other = base[:10] + [999] + base[11:]
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            write_t1g(t / "lkg", {"a": [base] * 5, "b": [base] * 4 + [other]})
            write_t1g(t / "same", {"a": [base] * 5, "b": [base] * 5})
            write_t1g(t / "bad", {"a": [other] * 5, "b": [other] * 5})
            aa = score.score_t1g(t / "lkg", t / "same", None)
            self.assertEqual(aa["verdict"], "REPORT")
            self.assertEqual(aa["metrics"]["distinct_mean"], 1.0)
            self.assertEqual(aa["metrics"]["early_div"], 0.0)
            self.assertEqual(aa["lkg_self"]["distinct_mean"], 1.5)
            self.assertEqual(score.score_t1g(t / "lkg", t / "same", aa)["verdict"], "PASS")
            self.assertEqual(score.score_t1g(t / "lkg", t / "bad", aa)["verdict"], "FAIL")

    def test_t1d(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for name, div in (("gold", None), ("same", None), ("early", 3)):
                d = t / name
                d.mkdir()
                ids = np.arange(100, dtype=np.int32)
                if div is not None:
                    ids = ids.copy()
                    ids[div:] += 1000
                top_ids = np.stack([ids] + [ids + 5000 + j for j in range(19)], 1).astype(np.int32)
                top_lp = np.tile(np.log(np.linspace(0.5, 0.001, 20)), (100, 1)).astype(np.float32)
                np.savez_compressed(d / "t1d.npz", ids=ids, lp=top_lp[:, 0], top_ids=top_ids, top_lp=top_lp)
                (d / "index.jsonl").write_text(json.dumps({"id": "p0", "off": 0, "n": 100}) + "\n")
            floor = score.score_t1d(t / "gold", t / "same", "A", None)
            self.assertEqual(floor["metrics"]["median_first_div"], 100)
            self.assertEqual(floor["metrics"]["matched_prefix_kl"], 0.0)
            self.assertEqual(score.score_t1d(t / "gold", t / "same", "A", floor)["verdict"], "PASS")
            bad = score.score_t1d(t / "gold", t / "early", "A", floor)
            self.assertEqual(bad["metrics"]["median_first_div"], 3)
            self.assertEqual(bad["verdict"], "FAIL")


def write_rows(d: Path, name: str, rows: list[dict]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


class TestScorePaired(unittest.TestCase):
    def test_t2_rules(self):
        ref = ([{"task": "gsm8k", "id": i, "correct": 1} for i in range(50)]
               + [{"task": "tools", "id": i, "correct": 1} for i in range(4)]
               + [{"task": "rep4", "id": i, "correct": 1, "value": 0.01} for i in range(4)])
        worse = [dict(r, correct=0) if r["task"] == "gsm8k" and r["id"] < 10 else r for r in ref]
        tool_miss = [dict(r, correct=0) if r["task"] == "tools" and r["id"] == 0 else r for r in ref]
        loopy = [dict(r, value=0.5) if r["task"] == "rep4" else r for r in ref]
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for name, rows in (("ref", ref), ("same", ref), ("worse", worse), ("tool", tool_miss), ("loopy", loopy)):
                write_rows(t / name, "t2.jsonl", rows)
            self.assertEqual(score.score_paired(t / "ref", t / "same", "t2.jsonl", 2.0, None, "T2")["verdict"], "PASS")
            w = score.score_paired(t / "ref", t / "worse", "t2.jsonl", 2.0, None, "T2")
            self.assertFalse(w["checks"]["gsm8k"])
            self.assertLess(w["tasks"]["gsm8k"]["p"], 0.05)
            self.assertFalse(score.score_paired(t / "ref", t / "tool", "t2.jsonl", 2.0, None, "T2")["checks"]["tools"])
            self.assertFalse(score.score_paired(t / "ref", t / "loopy", "t2.jsonl", 2.0, None, "T2")["checks"]["rep4"])
            write_rows(t / "short", "t2.jsonl", ref[:-1])
            self.assertEqual(score.score_paired(t / "ref", t / "short", "t2.jsonl", 2.0, None, "T2")["verdict"], "INVALID")


class TestT2Helpers(unittest.TestCase):
    def test_gsm8k(self):
        self.assertEqual(t2.gsm8k_gold("blah #### 1,234"), "1234")
        self.assertTrue(t2.gsm8k_correct("so 3 + 4\nAnswer: 7", "7"))
        self.assertTrue(t2.gsm8k_correct("The answer is **$1,200.**", "1200"))
        self.assertFalse(t2.gsm8k_correct("Answer: 8", "7"))

    def test_tool_exact(self):
        exp = {"name": "get_weather", "args": {"city": "Tokyo", "unit": "fahrenheit"}}
        ok = {"tool_calls": [{"name": "get_weather", "arguments": '{"city": "Tokyo", "unit": "fahrenheit"}'}]}
        extra = {"tool_calls": [{"name": "get_weather", "arguments": '{"city": "Tokyo", "unit": "fahrenheit", "x": 1}'}]}
        case = {"tool_calls": [{"name": "get_weather", "arguments": '{"city": "tokyo", "unit": "fahrenheit"}'}]}
        self.assertEqual(t2.judge_tool(exp, ok)["correct"], 1)
        self.assertEqual(t2.judge_tool(exp, extra)["correct"], 0)
        self.assertEqual(t2.judge_tool(exp, case)["correct"], 0)
        self.assertEqual(t2.judge_tool(exp, {"tool_calls": []})["correct"], 0)
        self.assertTrue(t2.values_equal(250, 250.0))
        self.assertFalse(t2.values_equal(True, 1))

    def test_schema_validator(self):
        s = t2.SCHEMAS["order"]
        good = {"order_id": "A", "items": [{"sku": "X", "qty": 2}], "total": 1.5, "paid": True}
        self.assertEqual(t2.validate(good, s), [])
        self.assertTrue(t2.validate({**good, "extra": 1}, s))
        self.assertTrue(t2.validate({**good, "items": [{"sku": "X", "qty": 2.5}]}, s))
        self.assertTrue(t2.validate({k: v for k, v in good.items() if k != "paid"}, s))
        self.assertTrue(t2.validate({"label": "happy", "confidence": 0.5}, t2.SCHEMAS["sentiment"]))
        self.assertTrue(t2.validate({"title": "x", "date": "2026/1/1", "attendees": 1, "online": True},
                                    t2.SCHEMAS["event"]))
        self.assertEqual(len(t2.JSON_PROMPTS), 30)
        self.assertEqual(len(t2.REP_PROMPTS), 64)

    def test_repeat_4gram(self):
        self.assertEqual(t2.repeat_4gram("a b c d e f g"), 0.0)
        self.assertGreater(t2.repeat_4gram("Register " * 100), 0.9)

    def test_pinned_ids(self):
        ids = json.loads((Q / "data/t2_ids.json").read_text())
        tools = json.loads((Q / "data/tools.json").read_text())
        items = {x["id"]: x for x in tools["items"]}
        self.assertEqual(len(ids["gsm8k"]["ids"]), 250)
        self.assertEqual(len(set(ids["ifeval"]["keys"])), 120)
        self.assertEqual(len(set(ids["tools"]["ids"])), 30)
        for i in ids["tools"]["ids"]:
            self.assertEqual(items[i]["ignore"], [], i)  # exact-argument gate needs full expectations


class TestNeedles(unittest.TestCase):
    def test_doc_and_verdict(self):
        needles = [t3_needles.needle_code(1), t3_needles.needle_code(501)]
        doc = t3_needles.needle_doc(t3_needles.filler(1, 20), 0.5, needles, 1)
        for name, code in needles:
            self.assertEqual(doc.count(code), 1)
            self.assertIn(name, doc)
        self.assertLess(abs(doc.index(needles[0][1]) / len(doc) - 0.5), 0.1)
        self.assertEqual(t3_needles.filler(7, 3), t3_needles.filler(7, 3))
        cells = [{"length": L, "found": True} for L in (4096, 131072) for _ in range(6)]
        self.assertTrue(t3_needles.verdict(cells)["pass"])
        miss128 = [dict(c, found=False) if i == 7 else c for i, c in enumerate(cells)]
        self.assertFalse(t3_needles.verdict(miss128)["pass"])
        long_ok = cells + [{"length": 250000, "found": i < 4} for i in range(6)]
        self.assertTrue(t3_needles.verdict(long_ok)["pass"])
        long_bad = cells + [{"length": 250000, "found": i < 3} for i in range(6)]
        self.assertFalse(t3_needles.verdict(long_bad)["pass"])


def png_dims(png: bytes) -> tuple[int, int]:
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos = 8
    while pos < len(png):
        (n,) = struct.unpack(">I", png[pos:pos + 4])
        tag, data, crc = png[pos + 4:pos + 8], png[pos + 8:pos + 8 + n], png[pos + 8 + n:pos + 12 + n]
        assert struct.unpack(">I", crc)[0] == zlib.crc32(tag + data) & 0xFFFFFFFF
        if tag == b"IHDR":
            dims = struct.unpack(">II", data[:8])
        pos += 12 + n
    return dims


class TestVisionProbes(unittest.TestCase):
    def test_keys_pinned(self):
        run = _load("quality/vision_probes/run.py", "vision_probes_run")
        self.assertEqual(run.check_shas(), [], "probe images or keys changed: rerun run.py --write-keys deliberately")

    def test_geometry(self):
        self.assertEqual(png_dims(probes.grid_wide()), (2048, 512))
        self.assertEqual(png_dims(probes.grid_tall()), (512, 2048))
        self.assertEqual(png_dims(probes.straddle_image()), (2048, 2048))
        for name, fn in probes.IMAGES.items():
            w, h = png_dims(fn())
            self.assertEqual((w % 32, h % 32), (0, 0), name)  # no resize: 32 px per token

    def test_judge(self):
        by = {p["id"]: p for p in probes.probes()}
        self.assertTrue(probes.judge(by["grid_wide.red_col"], "6"))
        self.assertTrue(probes.judge(by["grid_wide.red_col"], "Column 6."))
        self.assertFalse(probes.judge(by["grid_wide.red_col"], "5"))
        self.assertTrue(probes.judge(by["chart.tallest"], "Green"))
        self.assertFalse(probes.judge(by["chart.tallest"], "red and green"))
        self.assertTrue(probes.judge(by["multi.blue_first"], "The first one."))
        self.assertTrue(probes.judge(by["multi.blue_second"], "The second image is blue."))
        self.assertTrue(probes.judge(by["grid_wide.blue_row"], "Bottom row (the blue square)."))
        self.assertFalse(probes.judge(by["multi.blue_first"], "first or second"))
        self.assertTrue(probes.judge(by["ocr.407218"], "407,218"))
        s = by["straddle.c29"]
        self.assertTrue(probes.judge(s, f"5831; {s['key'][1]}"))
        self.assertFalse(probes.judge(s, "5831; XXX-0000-XX"))
        self.assertEqual(len(by), 16)


class TestCorpus(unittest.TestCase):
    def test_corpus_shape(self):
        rows = qlib.jsonl(Q / "data/t1_corpus.jsonl")
        doms = {r["domain"] for r in rows}
        self.assertTrue({"prose", "code", "math", "json", "zh", "es", "de", "ja", "tools", "long", "image"} <= doms)
        self.assertGreaterEqual(sum(r.get("tokens", 0) for r in rows), 150000)
        self.assertEqual(sum(r["domain"] == "image" for r in rows), 16)
        self.assertEqual(len({r["doc"] for r in rows if r["domain"] == "long"}), 8)
        self.assertEqual(len({r["id"] for r in rows}), len(rows))
        for r in rows:
            self.assertEqual(r["messages"][-1]["role"], "assistant", r["id"])
            self.assertLessEqual(r.get("tokens", 0), 1000, r["id"])
        merged = collect_t1.long_whole(rows)
        whole = [r for r in merged if r["domain"] == "long_whole"]
        self.assertEqual(len(whole), 8)

    def test_parse_prompt_logprobs(self):
        ids = [1, 2, THINK_END, 10, 11]
        plp = [None] + [{str(ids[i]): {"logprob": -0.5, "rank": 2}, "99": {"logprob": -0.1, "rank": 1},
                         "98": {"logprob": -3.0, "rank": 3}} for i in range(1, 5)]
        a = collect_t1.parse_prompt_logprobs(ids, plp, 3, 3)
        self.assertEqual(a["tgt"].tolist(), [10, 11])
        self.assertEqual(a["top_ids"][0].tolist(), [99, 10, 98])
        self.assertAlmostEqual(float(a["tgt_lp"][1]), -0.5)
        with self.assertRaises(RuntimeError):
            collect_t1.parse_prompt_logprobs(ids, plp, 3, 5)


# ---------------------------------------------------------------- fake serve

def fake_tokens(text: str) -> list[int]:
    return [1000 + (ord(ch) % 500) for ch in text]


def fake_chat_ids(body: dict) -> list[int]:
    ids = []
    for m in body["messages"]:
        content = m.get("content") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if p.get("type") == "text")
        ids += [248045] + ([THINK_END] if m["role"] == "assistant" else []) + fake_tokens(content)
    return ids


class FakeServe(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, text=False):
        data = obj.encode() if text else json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/metrics":
            return self._send('vllm:num_requests_running{engine="0"} 0.0\n', text=True)
        return self._send({"data": [{"id": "fake/model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            toks = [THINK_END] if body["prompt"] == "</think>" else fake_tokens(body["prompt"])
            return self._send({"count": len(toks), "tokens": toks})
        ids = fake_chat_ids(body)
        k = body["prompt_logprobs"]
        plp = [None]
        for i in range(1, len(ids)):
            e = {str(7000 + j): {"logprob": -0.2 - j - (ids[i - 1] % 3) * 0.1, "rank": j + 1} for j in range(k)}
            e.setdefault(str(ids[i]), {"logprob": -9.0, "rank": 100})
            plp.append(e)
        self._send({"choices": [{"message": {"content": "x"}, "finish_reason": "length"}],
                    "prompt_token_ids": ids, "prompt_logprobs": plp, "usage": {"prompt_tokens": len(ids)}})


class TestCollectT1EndToEnd(unittest.TestCase):
    def test_capture_and_score(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeServe)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}"
            with tempfile.TemporaryDirectory() as t:
                t = Path(t)
                for name in ("a", "b"):
                    with contextlib.redirect_stdout(io.StringIO()):
                        rc = collect_t1.main(["--url", url, "--out", str(t / name), "--limit", "1",
                                              "--domains", "prose,code,tools,image", "--shard-size", "2"])
                    self.assertEqual(rc, 0)
                man = json.loads((t / "a/manifest.json").read_text())
                self.assertEqual(man["models"]["data"][0]["id"], "fake/model")
                self.assertEqual(man["corpus_sha256"], qlib.sha256_file(Q / "data/t1_corpus.jsonl"))
                self.assertEqual(len(list((t / "a").glob("t1-*.npz"))), 2)
                out = score.score_t1(t / "a", t / "b", "A", None)
                self.assertEqual(out["verdict"], "PASS", out)
                self.assertEqual(sorted(out["by_domain"]), ["code", "image", "prose", "tools"])
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
