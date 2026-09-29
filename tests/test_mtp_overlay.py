#!/usr/bin/env python3
"""L1a/L1b draft-head overlay (docker/v030/mtp.py) and draft-vocab outputs.

Host-only: stdlib (+ numpy for the tool test). The overlay's torch code is
exercised on a v0.30 boot; here we prove the file is exactly stock + our
hunks, compiles, carries a valid R11 header, and that the pure-python parts
of the L1b head (vocab validation, rank split, two-stage argmax) match a
full-vocab argmax restricted to V'.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import py_compile
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "docker" / "v030" / "mtp.py"
GEN = ROOT / "docker" / "v030" / "apply_mtp_overlay.py"
VOCAB_DIR = ROOT / "tools" / "vocab"
V030_DIGEST = "sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56"
PIN_MTP_SHA = "081315f06eb2be521c2f1f9eec99fcf806c68324426f212e0ac1920b91efbeee"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _overlay_funcs(*names: str) -> dict:
    """Exec selected top-level defs of the overlay without importing vllm."""
    src = OVERLAY.read_text()
    tree = ast.parse(src)
    ns: dict = {"json": json, "_DRAFT_VOCAB_MIN_IDS": 1024}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), str(OVERLAY), "exec"), ns)
    return ns


class TestMtpOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = _load(GEN, "apply_mtp_overlay")
        cls.text = OVERLAY.read_text()

    def test_r11_header(self):
        head = dict(
            line[2:].split(": ", 1)
            for line in self.text.splitlines()[1:6]
            if line.startswith("# ") and ": " in line
        )
        self.assertTrue(self.text.startswith("# R11-OVERLAY\n"))
        self.assertEqual(head["base_image_digest"], V030_DIGEST)
        self.assertEqual(head["upstream_file"], "vllm/models/qwen4_exp/nvidia/mtp.py")
        self.assertRegex(head["upstream_file_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotEqual(head["upstream_file_sha256"], PIN_MTP_SHA)
        self.assertIn("upstream_PR", head)
        self.assertTrue(self.gen.IN_IMAGE.endswith("/dist-packages/" + head["upstream_file"]))

    def test_diff_is_only_our_hunks(self):
        stock = self.gen.unoverlay(self.text)
        self.assertEqual(hashlib.sha256(stock.encode()).hexdigest(), self.gen.UPSTREAM_FILE_SHA256)
        self.assertEqual(self.gen.overlay(stock), self.text)

    def test_refuses_other_stock(self):
        stock = self.gen.unoverlay(self.text)
        with self.assertRaises(SystemExit):
            self.gen.overlay(stock + "\n")

    def test_compiles(self):
        with tempfile.TemporaryDirectory() as d:
            py_compile.compile(str(OVERLAY), cfile=os.path.join(d, "mtp.pyc"), doraise=True)

    def test_l1a_wiring(self):
        tree = ast.parse(self.text)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpMTP")
        self.assertEqual(ast.unparse(cls.bases[0]), "LocalArgmaxMixin")
        methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        self.assertTrue({"get_top_tokens", "compute_logits", "load_weights"} <= methods)

    def test_l1b_default_off(self):
        # Without the env the load path must not touch the reduced head.
        self.assertIn('draft_vocab = bool(os.environ.get(_DRAFT_VOCAB_ENV, "").strip())', self.text)
        self.assertIn(
            "if draft_vocab or get_tensor_model_parallel_world_size() > 1:\n"
            "            self._finish_draft_vocab(",
            self.text,
        )
        # A rank without the env joins the agreement but never installs/logs.
        self.assertIn("if not ok:\n            if not requested:\n                return", self.text)
        self.assertIn('_DRAFT_VOCAB_ENV = "VLLM_QWEN38_DRAFT_VOCAB"', self.text)
        self.assertIn("return super().get_top_tokens(hidden_states)", self.text)

    def test_stock_check_when_source_available(self):
        src = os.environ.get("V030_MTP_SRC")
        if not src:
            self.skipTest("set V030_MTP_SRC=<v0.30 stock mtp.py> to re-render")
        stock = Path(src).read_text()
        self.assertEqual(self.gen.overlay(stock), self.text)


class TestReducedHeadMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _overlay_funcs("_load_draft_vocab", "_draft_vocab_rank_slice")

    def test_rank_split_balanced_and_ordered(self):
        split = self.ns["_draft_vocab_rank_slice"]
        ids = list(range(0, 94208, 2))  # 47104 ids
        a, b = split(ids, 2, 0), split(ids, 2, 1)
        self.assertEqual((len(a), len(b)), (23552, 23552))
        self.assertEqual(a + b, ids)
        self.assertLess(a[-1], b[0])
        odd = list(range(1025))
        self.assertEqual(split(odd, 2, 0) + split(odd, 2, 1), odd)

    def test_load_validation(self):
        load = self.ns["_load_draft_vocab"]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v.json"
            p.write_text(json.dumps({"vocab_size": 248320, "ids": list(range(2048))}))
            self.assertEqual(len(load(str(p), 248320)), 2048)
            for bad in (
                {"vocab_size": 248320, "ids": list(range(10))},  # too few
                {"vocab_size": 248320, "ids": [5] + list(range(2048))},  # unsorted
                {"vocab_size": 248320, "ids": list(range(248000, 249100))},  # out of range
                {"vocab_size": 1000, "ids": list(range(2048))},  # wrong model
            ):
                p.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    load(str(p), 248320)

    def test_two_stage_argmax_equals_restricted_argmax(self):
        """Per-rank argmax + first-max over ranks == argmax over V' (lowest id on ties)."""
        split = self.ns["_draft_vocab_rank_slice"]
        rng = random.Random(0)
        vocab = 5000
        for trial in range(200):
            ids = sorted(rng.sample(range(vocab), 1500))
            # coarse values force many ties
            logits = [rng.randint(0, 40) for _ in range(vocab)]
            want = min(ids, key=lambda i: (-logits[i], i))
            pairs = []
            for r in range(2):
                part = split(ids, 2, r)
                vals = [logits[i] for i in part]
                j = vals.index(max(vals))  # torch.argmax: first max
                pairs.append((vals[j], part[j]))
            best = max(range(2), key=lambda r: (pairs[r][0], -r))  # first max rank
            self.assertEqual(pairs[best][1], want, trial)


class TestVocabOutputs(unittest.TestCase):
    def test_draft_vocab_files(self):
        files = sorted(VOCAB_DIR.glob("draft_vocab_*.json"))
        if not files:
            self.skipTest("tools/vocab not generated yet")
        ns = _overlay_funcs("_load_draft_vocab")
        tok = json.loads(
            (Path.home() / ".cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4"
             "/snapshots/fab0aecb760cec45227f6656abcaafa11abca87a/tokenizer.json").read_text()
        ) if (Path.home() / ".cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4").exists() else None
        for f in files:
            doc = json.loads(f.read_text())
            ids = ns["_load_draft_vocab"](str(f), 248320)
            self.assertEqual(len(ids), doc["size"], f.name)
            self.assertEqual(f.name, f"draft_vocab_{doc['size']}.json")
            if tok is not None:
                special = {t["id"] for t in tok["added_tokens"]}
                self.assertTrue(special <= set(ids), f.name)

    def test_tool_core(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        tool = _load(ROOT / "tools" / "draft_subvocab_coverage.py", "draft_subvocab_coverage")
        score = np.zeros(100)
        score[[7, 3, 50]] = [5.0, 5.0, 9.0]
        ranked = tool.rank_ids(score, np.array([99, 1]))
        self.assertEqual(ranked[:5].tolist(), [1, 99, 50, 3, 7])
        keep = tool.keep_set(ranked, 4)
        self.assertEqual(keep.tolist(), [1, 3, 50, 99])
        self.assertAlmostEqual(tool.coverage(np.array([1, 3, 4, 4]), keep, 100), 0.5)
        self.assertEqual(len(tool.bytes_to_unicode()), 256)
        self.assertAlmostEqual(tool.accept_len(tool.alpha_for(2.455, 3), 3), 2.455, places=6)


if __name__ == "__main__":
    unittest.main()
