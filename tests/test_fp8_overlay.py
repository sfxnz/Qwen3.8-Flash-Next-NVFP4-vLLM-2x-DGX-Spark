#!/usr/bin/env python3
"""docker/v030/modelopt.py is stock v0.30.0 plus only the P4-1 FP8-dense hunks."""
from __future__ import annotations

import ast
import importlib.util
import os
import re
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "docker" / "v030" / "modelopt.py"
_spec = importlib.util.spec_from_file_location(
    "apply_fp8_dense_overlay", ROOT / "docker" / "v030" / "apply_fp8_dense_overlay.py"
)
afo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(afo)
# Optional: the stock file extracted from the v0.30.0 image.
STOCK = os.environ.get("V030_MODELOPT")
ONLINE_FP8 = "vllm.model_executor.layers.quantization.online.fp8"


class _LinearBase:
    pass


class _ParallelLMHead:  # not a LinearBase, as in vLLM
    pass


class _PerBlock:
    pass


class _Ptpc:
    pass


def _load_helper(text: str) -> dict:
    """Exec only the overlay's helper with stub vLLM types (no torch needed)."""
    tree = ast.parse(text)
    keep = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name == "_qwen38_fp8_dense_method")
        or (
            isinstance(n, ast.Assign)
            and any(getattr(t, "id", "") == "_QWEN38_FP8_DENSE_DEFAULT_ALLOW"
                    for t in n.targets)
        )
    ]
    assert len(keep) == 2, keep
    for n in keep:
        if isinstance(n, ast.FunctionDef):
            for a in n.args.args:
                a.annotation = None
            n.returns = None
    logger = mock.Mock()
    # vLLM's info_once/debug_once go through functools.lru_cache: args must hash.
    logger.info_once.side_effect = lambda msg, *args, **kw: hash(args)
    logger.debug_once.side_effect = lambda msg, *args, **kw: hash(args)
    ns = {
        "LinearBase": _LinearBase,
        "logger": logger,
    }
    exec(compile(ast.Module(body=keep, type_ignores=[]), "overlay", "exec"), ns)
    return ns


class Fp8DenseOverlayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = OVERLAY.read_text()
        self.ns = _load_helper(self.text)
        stub = types.ModuleType(ONLINE_FP8)
        stub.Fp8PerBlockOnlineLinearMethod = _PerBlock
        stub.Fp8PtpcOnlineLinearMethod = _Ptpc
        patcher = mock.patch.dict(sys.modules, {ONLINE_FP8: stub})
        patcher.start()
        self.addCleanup(patcher.stop)

    def route(self, prefix: str, env: dict, layer: object | None = None):
        with mock.patch.dict(os.environ, env, clear=False):
            for k in ("VLLM_QWEN38_FP8_DENSE", "VLLM_QWEN38_FP8_DENSE_ALLOW"):
                if k not in env:
                    os.environ.pop(k, None)
            return self.ns["_qwen38_fp8_dense_method"](
                layer if layer is not None else _LinearBase(), prefix
            )

    def test_r11_header(self) -> None:
        head = dict(
            re.findall(r"^# (base_image_digest|upstream_file|upstream_file_sha256|upstream_PR): (.+)$",
                       self.text.split("# SPDX", 1)[0], re.M)
        )
        self.assertEqual(
            head["base_image_digest"],
            "sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56",
        )
        self.assertEqual(head["upstream_file"], "vllm/model_executor/layers/quantization/modelopt.py")
        self.assertEqual(head["upstream_file_sha256"], afo.UPSTREAM_FILE_SHA256)
        self.assertTrue(head["upstream_PR"].startswith("n/a"))

    def test_inverts_to_stock_and_rerenders(self) -> None:
        stock = afo.unoverlay(self.text)
        self.assertEqual(afo.sha256(stock), afo.UPSTREAM_FILE_SHA256)
        self.assertEqual(afo.overlay(stock), self.text)

    @unittest.skipUnless(STOCK and Path(STOCK).is_file(), "V030_MODELOPT not set")
    def test_generator_applies_to_v030_file(self) -> None:
        self.assertEqual(afo.overlay(Path(STOCK).read_text()), self.text)

    def test_py_compile(self) -> None:
        compile(self.text, str(OVERLAY), "exec")  # what py_compile checks, no .pyc

    def test_only_two_call_sites(self) -> None:
        self.assertEqual(self.text.count("_qwen38_fp8_dense_method(layer, prefix)"), 2)
        stock = afo.unoverlay(self.text)
        self.assertEqual(
            self.text.count("return UnquantizedLinearMethod()"),
            stock.count("return UnquantizedLinearMethod()"),
        )

    def test_default_off_is_stock(self) -> None:
        p = "language_model.model.layers.0.linear_attn.in_proj_qkvz"
        for env in ({}, {"VLLM_QWEN38_FP8_DENSE": ""}, {"VLLM_QWEN38_FP8_DENSE": "off"},
                    {"VLLM_QWEN38_FP8_DENSE": "0"},
                    {"VLLM_QWEN38_FP8_DENSE_ALLOW": r"in_proj_qkvz$"}):
            self.assertIsNone(self.route(p, env), env)

    def test_modes(self) -> None:
        p = "language_model.model.layers.3.self_attn.o_proj"
        self.assertIsInstance(self.route(p, {"VLLM_QWEN38_FP8_DENSE": "per_block"}), _PerBlock)
        self.assertIsInstance(self.route(p, {"VLLM_QWEN38_FP8_DENSE": "PTPC"}), _Ptpc)
        with self.assertRaises(ValueError):
            self.route(p, {"VLLM_QWEN38_FP8_DENSE": "fp8"})

    def test_non_linear_and_lm_head_untouched(self) -> None:
        env = {"VLLM_QWEN38_FP8_DENSE": "per_block", "VLLM_QWEN38_FP8_DENSE_ALLOW": ".*"}
        self.assertIsNone(self.route("lm_head", env, _ParallelLMHead()))
        self.assertIsInstance(self.route("lm_head", env), _PerBlock)  # allow-list only

    def test_custom_allow_list(self) -> None:
        env = {"VLLM_QWEN38_FP8_DENSE": "per_block",
               "VLLM_QWEN38_FP8_DENSE_ALLOW": r"\.out_proj$, \.o_proj$"}
        self.assertIsNotNone(self.route("language_model.model.layers.0.linear_attn.out_proj", env))
        self.assertIsNone(self.route("language_model.model.layers.0.linear_attn.in_proj_qkvz", env))

    @unittest.skipUnless((afo.SNAPSHOT / "model.safetensors.index.json").is_file(),
                         "checkpoint snapshot not present")
    def test_matcher_on_real_checkpoint(self) -> None:
        shapes = afo.read_shapes(afo.SNAPSHOT)
        prefixes = {afo.runtime_prefix(n) for n in shapes}
        env = {"VLLM_QWEN38_FP8_DENSE": "per_block"}
        picked = {p for p in prefixes if self.route(p, env) is not None}
        by_leaf: dict[str, int] = {}
        for p in picked:
            leaf = ("mtp." if p.startswith("mtp.") else "") + p.rsplit(".", 1)[-1]
            by_leaf[leaf] = by_leaf.get(leaf, 0) + 1
        self.assertEqual(
            by_leaf,
            {"in_proj_qkvz": 36, "out_proj": 36, "qkv_proj": 12, "o_proj": 12,
             "kv_proj": 1, "mtp.qkv_proj": 1, "mtp.o_proj": 1},
        )
        self.assertIn("mtp.layers.48.self_attn.qkv_proj", picked)
        self.assertIn("language_model.model.layers.1.ple.kv_proj", picked)
        for p in picked:  # the overlay's own list and the tool's agree
            self.assertTrue(afo.allow_match(p, afo._default_patterns(self.text)))
        # Kept BF16: in_proj_ba (C7), indexer, shared expert, router, HC, lm_head, MTP fc.
        for bad in ("in_proj_ba", "indexer", "shared_expert", "mlp.gate", "hyper_connection",
                    "lm_head", "fc_hidden", "fc_embedding", "embed_tokens", "visual"):
            self.assertFalse([p for p in picked if bad in p], bad)
        rows = afo.savings(afo.SNAPSHOT, afo._default_patterns(self.text), tp=2)
        bf16 = sum(r["bf16"] for r in rows.values())
        self.assertEqual(bf16, 2_789_212_160)
        self.assertLess(abs(bf16 - 2 * sum(r["per_block"] for r in rows.values())), 0.001 * bf16)


if __name__ == "__main__":
    unittest.main()
