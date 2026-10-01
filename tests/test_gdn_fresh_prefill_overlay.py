#!/usr/bin/env python3
"""docker/v030/gdn_attn.py is stock v0.30.0 plus only the fresh-prefill classification hunk (S1.5)."""
from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "docker" / "v030" / "gdn_attn.py"
_spec = importlib.util.spec_from_file_location(
    "apply_gdn_fresh_prefill_overlay", ROOT / "docker" / "v030" / "apply_gdn_fresh_prefill_overlay.py"
)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


class GdnFreshPrefillOverlayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = OVERLAY.read_text()

    def test_r11_header(self) -> None:
        head = self.text.split("# SPDX", 1)[0]
        self.assertTrue(head.startswith("# R11-OVERLAY\n"))
        self.assertIn(f"# base_image_digest: {gen.BASE_IMAGE_DIGEST}\n", head)
        self.assertIn(f"# upstream_file: {gen.UPSTREAM_FILE}\n", head)
        self.assertIn(f"# upstream_file_sha256: {gen.UPSTREAM_FILE_SHA256}\n", head)

    def test_inverts_to_stock(self) -> None:
        stock = gen.unoverlay(self.text)
        self.assertEqual(gen.sha256(stock), gen.UPSTREAM_FILE_SHA256)
        self.assertEqual(gen.overlay(stock), self.text)

    def test_parses(self) -> None:
        ast.parse(self.text)

    def test_keyword_on_the_one_call(self) -> None:
        tree = ast.parse(self.text)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", None) == "split_decodes_and_prefills"]
        self.assertEqual(len(calls), 1)
        kw = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
        self.assertEqual(kw, {"decode_threshold": "1", "treat_short_extends_as_decodes": "m.is_prefilling is None"})

    def test_refuses_other_stock(self) -> None:
        with self.assertRaises(SystemExit):
            gen.overlay(gen.unoverlay(self.text) + "\n")


if __name__ == "__main__":
    unittest.main()
