#!/usr/bin/env python3
"""docker/v030/qsa_indexer.py is stock v0.30.0 plus only the top-k order sort (S1.5)."""
from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "docker" / "v030" / "qsa_indexer.py"
_spec = importlib.util.spec_from_file_location(
    "apply_qsa_topk_order_overlay", ROOT / "docker" / "v030" / "apply_qsa_topk_order_overlay.py"
)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


class QsaTopkOrderOverlayTest(unittest.TestCase):
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

    def test_sort_follows_topk(self) -> None:
        body = self.text
        call = body.index("    topk_op(\n        logits,")
        sort = body.index("block_indices.sort(dim=1).values")
        self.assertLess(call, sort)
        self.assertLess(sort, body.index("def qsa_select_paged_decode("))
        self.assertIn("full = (visible_blocks > block_topk).unsqueeze(1)", body)

    def test_tie_repair_prefill_only(self) -> None:
        body = self.text
        prefill = body.index("def qsa_select_paged_prefill(")
        decode = body.index("def qsa_select_paged_decode(")
        calls = [i for i in range(len(body)) if body.startswith("        _deterministic_topk_ties(\n", i)]
        self.assertEqual(len(calls), 1)
        self.assertGreater(calls[0], prefill)
        self.assertLess(body.index("def _deterministic_topk_ties("), decode)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
    def test_tie_repair_matches_lowest_index_reference(self) -> None:
        import torch

        src = self.text
        ns: dict = {"torch": torch}
        exec(src[src.index("def _deterministic_topk_ties("):src.index("def qsa_select_paged_decode(")], ns)
        fn = ns["_deterministic_topk_ties"]
        g = torch.Generator().manual_seed(0)
        for _ in range(50):
            rows, width, k = 7, 1024, 512
            logits = (torch.randn(rows, width, generator=g) * 3).round() / 3
            vis = torch.randint(1, width + 1, (rows,), generator=g, dtype=torch.int32)
            bi = torch.full((rows, k), -1, dtype=torch.int32)
            exp = bi.clone()
            for r in range(rows):
                v = int(vis[r])
                if v <= k:
                    bi[r, :v] = exp[r, :v] = torch.arange(v, dtype=torch.int32)
                    continue
                perm = torch.randperm(v, generator=g)
                top = perm[torch.sort(logits[r, :v][perm], descending=True, stable=True).indices[:k]]
                bi[r] = top[torch.randperm(k, generator=g)].int()
                exp[r] = torch.sort(-logits[r, :v], stable=True).indices[:k].sort().values.int()
            fn(logits, vis, bi, k)
            self.assertTrue(torch.equal(bi, exp))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
    def test_tie_repair_ignores_bad_topk_picks(self) -> None:
        """Session 3 64k device assert: -1 pads or non-top picks from persistent_topk must not
        push more than k entries past the threshold (the scatter buffer is k + 1 wide)."""
        import torch

        src = self.text
        ns: dict = {"torch": torch}
        exec(src[src.index("def _deterministic_topk_ties("):src.index("def qsa_select_paged_decode(")], ns)
        fn = ns["_deterministic_topk_ties"]
        g = torch.Generator().manual_seed(1)
        rows, width, k = 5, 4096, 512
        logits = torch.randn(rows, width, generator=g)
        logits[1, 100:4000] = float("-inf")  # fewer finite visible logits than k
        logits[2, 0] = float("-inf")  # a clamped -1 pad would read this as the threshold
        logits[3, 5] = float("nan")
        vis = torch.tensor([width, width, width, 3000, 600], dtype=torch.int32)
        bi = torch.full((rows, k), -1, dtype=torch.int32)  # worst case: every pick is a pad
        exp = torch.empty_like(bi)
        for r in range(rows):
            v = int(vis[r])
            row = logits[r, :v].nan_to_num(nan=float("-inf"))
            exp[r] = torch.sort(-row, stable=True).indices[:k].sort().values.int()
        fn(logits, vis, bi, k)
        self.assertTrue(torch.equal(bi, exp))

    def test_refuses_other_stock(self) -> None:
        with self.assertRaises(SystemExit):
            gen.overlay(gen.unoverlay(self.text) + "\n")


if __name__ == "__main__":
    unittest.main()
