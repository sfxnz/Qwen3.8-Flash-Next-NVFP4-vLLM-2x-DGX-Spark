#!/usr/bin/env python3
"""L1a/L1b/L1b' draft-head overlay (docker/v030/mtp.py) and draft-vocab outputs.

Host: stdlib (+ numpy for the tool test). The overlay's GPU kernels run on a
v0.30 boot; here we prove the file is exactly stock + our hunks, compiles,
carries a valid R11 header, and that the pure-python parts of the L1b head
(vocab validation, rank split, two-stage argmax) match a full-vocab argmax
restricted to V'. When torch is importable (CPU is enough) the L1b' FP8
quantizer and the install/fallback logic of _finish_draft_head,
get_top_tokens and compute_logits run against stubs; otherwise they skip.
Set V030_SRC=<v0.30 vllm package dir> to check the v0.30 APIs L1b' calls.
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
        # Without the envs the load path must not touch the draft head.
        self.assertIn('draft_vocab = bool(os.environ.get(_DRAFT_VOCAB_ENV, "").strip())', self.text)
        self.assertIn(
            "requested = draft_vocab or fp8_kernels or fp8_err\n"
            "        if requested or get_tensor_model_parallel_world_size() > 1:\n"
            "            self._finish_draft_head(",
            self.text,
        )
        self.assertIn('_DRAFT_VOCAB_ENV = "VLLM_QWEN38_DRAFT_VOCAB"', self.text)
        self.assertIn('_DRAFT_HEAD_FP8_ENV = "VLLM_QWEN38_DRAFT_HEAD_FP8"', self.text)
        self.assertIn("return super().get_top_tokens(hidden_states)", self.text)

    def test_l1b_prime_wiring(self):
        tree = ast.parse(self.text)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpMTP")
        fns = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        # Both TP agreements run unconditionally (no hang when one rank lacks an env).
        fin = ast.unparse(fns["_finish_draft_head"])
        self.assertEqual(fin.count("self._tp_agree("), 2)
        for stmt in fns["_finish_draft_head"].body:
            if "_tp_agree" in ast.unparse(stmt):
                self.assertIsInstance(stmt, ast.Assign)  # top level, not under an if
        # compute_logits only diverts for the full-vocab FP8 head.
        cl = ast.unparse(fns["compute_logits"])
        self.assertIn("self.draft_head_fp8 is not None and self.draft_head_full", cl)
        self.assertIn("return self.logits_processor(self.lm_head, hidden_states)", cl)
        # Kernel calls match the v0.30 API (see test_v030_api for the signatures).
        head = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_Fp8DraftHead")
        src = ast.unparse(head)
        self.assertIn("prepare_fp8_layer_for_marlin(layer, size_k_first=False)", src)
        self.assertIn("self.weight, self.scale, self.workspace, self.n, self.k, None", src)
        self.assertIn("scaled_fp8_quant(h, use_per_token_if_dynamic=True)", src)
        self.assertIn("cutlass_scaled_mm(hq, self.weight, hs, self.scale, self.dtype)", src)

    def test_v030_api(self):
        root = os.environ.get("V030_SRC")
        if not root:
            self.skipTest("set V030_SRC=<v0.30 vllm package dir>")
        root = Path(root)

        def sig(rel: str, fn: str) -> list[str]:
            tree = ast.parse((root / rel).read_text())
            node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn)
            return [a.arg for a in node.args.args + node.args.kwonlyargs]

        mu = "model_executor/layers/quantization/utils/marlin_utils_fp8.py"
        self.assertEqual(sig(mu, "apply_fp8_marlin_linear")[:7],
                         ["input", "weight", "weight_scale", "workspace", "size_n", "size_k", "bias"])
        self.assertEqual(sig(mu, "prepare_fp8_layer_for_marlin")[:2], ["layer", "size_k_first"])
        self.assertIn("use_per_token_if_dynamic", sig("_custom_ops.py", "scaled_fp8_quant"))
        self.assertEqual(sig("_custom_ops.py", "cutlass_scaled_mm")[:5],
                         ["a", "b", "scale_a", "scale_b", "out_dtype"])
        lp = (root / "model_executor/layers/logits_processor.py").read_text()
        self.assertIn("def _gather_logits(self, logits", lp)
        self.assertIn("self.org_vocab_size", lp)
        vp = (root / "model_executor/layers/vocab_parallel_embedding.py").read_text()
        for attr in ("num_org_vocab_padding", "num_added_elements_padded", "num_org_elements",
                     "org_vocab_start_index", "org_vocab_end_index"):
            self.assertIn(attr, vp)

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


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


class TestFp8Head(unittest.TestCase):
    def test_kernel_parse(self):
        ns = _overlay_funcs("_draft_head_fp8_kernels")
        ns["_DRAFT_HEAD_FP8_ENV"] = "VLLM_QWEN38_DRAFT_HEAD_FP8"
        parse = ns["_draft_head_fp8_kernels"]
        for v in ("", " ", "0", "off", "OFF", "false", "no"):
            self.assertEqual(parse(v), (), v)
        for v in ("1", "on", "true", "auto", " Auto "):
            self.assertEqual(parse(v), ("marlin", "w8a8"), v)
        self.assertEqual(parse("marlin"), ("marlin",))
        self.assertEqual(parse("w8a16"), ("marlin",))
        self.assertEqual(parse("w8a8"), ("w8a8",))
        with self.assertRaises(ValueError):
            parse("fp4")

    def test_quantize_rows(self):
        torch = _torch()
        if torch is None:
            self.skipTest("torch not installed")
        ns = _overlay_funcs("_quantize_rows_fp8")
        ns.update(torch=torch, _FP8_MAX=448.0)
        g = torch.Generator().manual_seed(0)
        w = (torch.randn(1000, 256, generator=g) * torch.rand(1000, 1, generator=g) * 0.1).to(torch.bfloat16)
        w[7] = 0  # all-zero row
        q, s = ns["_quantize_rows_fp8"](w, chunk=333)
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(s.shape), (1000,))
        self.assertTrue(torch.equal(s.to(torch.bfloat16).float(), s))  # Marlin == CUTLASS scale
        deq = q.float() * s[:, None]
        amax = w.float().abs().amax(dim=1)
        err = (deq - w.float()).abs().amax(dim=1)
        self.assertTrue(bool((err <= amax * 2**-4 + 1e-12).all()))  # E4M3: 3 mantissa bits
        self.assertEqual(float(deq[7].abs().max()), 0.0)
        # argmax over a clear-margin head survives FP8
        h = torch.randn(64, 256, generator=g)
        ref = h @ w.float().t()
        got = h @ deq.t()
        top2 = ref.topk(2, dim=-1).values
        clear = (top2[:, 0] - top2[:, 1]) > 0.05 * top2[:, 0].abs()
        self.assertTrue(bool((ref.argmax(-1) == got.argmax(-1))[clear].all()))


class _Stub:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _harness(torch, world=1):
    """Qwen4ExpMTP's draft-head methods on a stub model (CPU, TP=world stubs)."""
    tree = ast.parse(OVERLAY.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpMTP")
    names = {"get_top_tokens", "_draft_head_logits", "_check_draft_head_supported",
             "_full_head_rows", "_tp_agree", "_finish_draft_head", "compute_logits"}
    body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    harness = ast.ClassDef(name="Harness", bases=[ast.Name("Base", ast.Load())], keywords=[],
                           body=body, decorator_list=[], type_params=[])
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name in ("_quantize_rows_fp8", "_build_fp8_draft_head")]
    warnings: list[str] = []

    class FakeHead:
        """_Fp8DraftHead with a dequantized CPU matmul in place of the kernels."""

        def __init__(self, rows, kernel):
            if kernel == "broken":
                raise RuntimeError("kernel unavailable")
            q, s = ns["_quantize_rows_fp8"](rows)
            self.kernel, self.w = kernel, q.float() * s[:, None]
            self.nbytes = q.numel() + 4 * s.numel()

        def logits(self, h):
            return h.float() @ self.w.t()

        def check(self, rows):
            return 0.0

    class Base:
        def get_top_tokens(self, h):
            return "stock"

    import torch.nn.functional as F

    ns = {
        "torch": torch, "F": F, "_FP8_MAX": 448.0, "_Fp8DraftHead": FakeHead, "Base": Base,
        "_DRAFT_VOCAB_ENV": "VLLM_QWEN38_DRAFT_VOCAB",
        "_DRAFT_HEAD_FP8_ENV": "VLLM_QWEN38_DRAFT_HEAD_FP8",
        "get_tensor_model_parallel_world_size": lambda: world,
        "get_pp_group": lambda: _Stub(world_size=1),
        "logger": _Stub(warning=lambda *a: warnings.append(a[0] % a[1:]), info=lambda *a: None),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(helpers + [harness], [])), str(OVERLAY), "exec"), ns)
    return ns["Harness"], warnings


class TestFinishDraftHead(unittest.TestCase):
    V, K = 4096, 64

    def setUp(self):
        self.torch = _torch()
        if self.torch is None:
            self.skipTest("torch not installed")
        torch = self.torch
        self.H, self.warnings = _harness(torch)
        g = torch.Generator().manual_seed(1)
        self.w = torch.randn(self.V, self.K, generator=g).to(torch.bfloat16)
        self.h = torch.randn(16, self.K, generator=g).to(torch.bfloat16)
        self.ids = sorted(torch.randperm(self.V, generator=g)[:1500].tolist())

    def model(self):
        torch, V = self.torch, self.V
        m = self.H()
        m.config = _Stub(tie_word_embeddings=False, vocab_size=V)
        m.logits_processor = _Stub(
            scale=1.0, soft_cap=None, logits_as_input=False, head_dtype=None, org_vocab_size=V,
            _gather_logits=lambda x: x,
        )
        m.logits_processor.__class__ = type("LP", (_Stub,), {"__call__": lambda self, head, h: "stock"})
        m.lm_head = _Stub(weight=self.w, shard_indices=_Stub(
            num_org_vocab_padding=0, num_added_elements_padded=0, num_org_elements=V,
            org_vocab_start_index=0, org_vocab_end_index=V))
        m.draft_vocab_weight = m.draft_vocab_ids = m.draft_head_fp8 = None
        m.draft_head_full = False
        return m

    def finish(self, m, reduced, kernels=(), fp8_err=""):
        ids = self.ids if reduced else None
        rows = self.w[self.torch.tensor(self.ids)] if reduced else None
        m._finish_draft_head(ids, rows, "", reduced, kernels, fp8_err)
        return m

    def test_nothing_requested(self):
        m = self.finish(self.model(), False)
        self.assertIsNone(m.draft_vocab_ids)
        self.assertEqual(m.get_top_tokens(self.h), "stock")
        self.assertEqual(m.compute_logits(self.h), "stock")
        self.assertEqual(self.warnings, [])

    def test_reduced_bf16(self):
        m = self.finish(self.model(), True)
        self.assertIsNone(m.draft_head_fp8)
        self.assertEqual(tuple(m.draft_vocab_weight.shape), (len(self.ids), self.K))
        ref = (self.h @ self.w.t()).float()
        keep = self.torch.tensor(self.ids)
        want = keep[ref[:, keep].argmax(-1)]
        self.assertTrue(self.torch.equal(m.get_top_tokens(self.h), want))
        self.assertEqual(m.compute_logits(self.h), "stock")

    def test_fp8_full_vocab(self):
        m = self.finish(self.model(), False, ("marlin",))
        self.assertTrue(m.draft_head_full)
        self.assertIsNone(m.draft_vocab_weight)  # no BF16 copy kept
        self.assertTrue(self.torch.equal(m.draft_vocab_ids, self.torch.arange(self.V)))
        logits = m.compute_logits(self.h)
        self.assertEqual(tuple(logits.shape), (16, self.V))
        self.assertTrue(self.torch.equal(m.get_top_tokens(self.h), logits.argmax(-1)))
        agree = (logits.argmax(-1) == (self.h @ self.w.t()).float().argmax(-1)).float().mean()
        self.assertGreaterEqual(float(agree), 0.75)

    def test_fp8_on_reduced_rows(self):
        m = self.finish(self.model(), True, ("marlin", "w8a8"))
        self.assertFalse(m.draft_head_full)
        self.assertIsNone(m.draft_vocab_weight)
        self.assertEqual(m.draft_head_fp8.w.shape[0], len(self.ids))
        self.assertEqual(m.compute_logits(self.h), "stock")
        top = m.get_top_tokens(self.h)
        self.assertTrue(set(top.tolist()) <= set(self.ids))

    def test_fp8_kernel_fallback_order(self):
        m = self.finish(self.model(), False, ("broken", "w8a8"))
        self.assertEqual(m.draft_head_fp8.kernel, "w8a8")

    def test_fp8_failure_full_installs_nothing(self):
        m = self.finish(self.model(), False, ("broken",))
        self.assertIsNone(m.draft_vocab_ids)
        self.assertIsNone(m.draft_vocab_weight)
        self.assertEqual(m.get_top_tokens(self.h), "stock")
        self.assertTrue(any("FP8 draft head disabled" in w for w in self.warnings))

    def test_fp8_failure_reduced_keeps_bf16(self):
        m = self.finish(self.model(), True, ("broken",))
        self.assertIsNone(m.draft_head_fp8)
        self.assertEqual(tuple(m.draft_vocab_weight.shape), (len(self.ids), self.K))

    def test_bad_env_value_warns(self):
        m = self.finish(self.model(), False, (), "VLLM_QWEN38_DRAFT_HEAD_FP8='x': use ...")
        self.assertIsNone(m.draft_vocab_ids)
        self.assertTrue(any("FP8 draft head disabled" in w for w in self.warnings))

    def test_head_dtype_mismatch_refuses_fp8(self):
        m = self.model()
        m.logits_processor.head_dtype = self.torch.float32
        self.finish(m, False, ("marlin",))
        self.assertIsNone(m.draft_vocab_ids)


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
        # v2: per-domain frequencies (sources weigh equally) and the candidate rankings
        counts = {"zh/a": np.array([0, 4, 0, 0, 0.0]), "zh/b": np.array([0, 0, 2, 0, 0.0]),
                  "code/c": np.array([1, 0, 0, 3, 0.0])}
        fr = tool.domain_freqs(counts)
        self.assertEqual(fr["zh"].tolist(), [0, 0.5, 0.5, 0, 0])
        self.assertEqual(fr["code"].tolist(), [0.25, 0, 0, 0.75, 0])
        sc = tool.ranking_scores(fr, vocab=5)
        self.assertEqual(set(sc), {"bpe", "corpus_sum", "corpus_max", "blend"})
        none = np.array([], dtype=np.int64)
        self.assertEqual(tool.rank_ids(sc["bpe"], none).tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(tool.rank_ids(sc["corpus_sum"], none).tolist(), [3, 1, 2, 0, 4])
        self.assertEqual(tool.rank_ids(sc["corpus_max"], none).tolist(), [3, 1, 2, 0, 4])
        self.assertEqual(tool.rank_ids(sc["blend"], none).tolist(), [1, 0, 3, 2, 4])  # tie -> lower id


if __name__ == "__main__":
    unittest.main()
