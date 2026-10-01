#!/usr/bin/env python3
"""docker/v030/serving.py is stock v0.30.0 plus only the vLLM #56067 hunks (G05)."""
from __future__ import annotations

import ast
import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "docker" / "v030" / "serving.py"
_spec = importlib.util.spec_from_file_location(
    "apply_api_overlays", ROOT / "docker" / "v030" / "apply_api_overlays.py"
)
aao = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aao)


class ApiOverlayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = OVERLAY.read_text()

    def test_r11_header(self) -> None:
        head = dict(
            re.findall(r"^# (base_image_digest|upstream_file_sha256|upstream_PR): (.+)$",
                       self.text.split("# SPDX", 1)[0], re.M)
        )
        self.assertEqual(
            head["base_image_digest"],
            "sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56",
        )
        self.assertRegex(head["upstream_file_sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("#56067", head["upstream_PR"])

    def test_inverts_to_pinned_stock(self) -> None:
        stock = aao.unoverlay(self.text)
        self.assertEqual(aao.sha256(stock), aao.UPSTREAM_FILE_SHA256)
        self.assertEqual(aao.overlay(stock), self.text)

    def test_parses(self) -> None:
        ast.parse(self.text)

    def test_count_path_deferred(self) -> None:
        body = self.text
        # Per-delta recount only under continuous usage.
        self.assertIn(
            "if include_continuous_usage:\n"
            "                            previous_reasoning_tokens[i] = (",
            body,
        )
        # One recount per choice before final usage.
        final = body.index("and not include_continuous_usage:")
        self.assertLess(final, body.index("if include_usage:\n                completion_tokens"))
        # Only two count_reasoning_tokens sites changed; stock had 1 in the stream.
        stock = aao.unoverlay(body)
        self.assertEqual(
            body.count("count_reasoning_tokens("),
            stock.count("count_reasoning_tokens(") + 1,
        )


if __name__ == "__main__":
    unittest.main()
