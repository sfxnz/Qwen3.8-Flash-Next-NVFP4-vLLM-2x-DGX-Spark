#!/usr/bin/env python3
"""Offline tests for quality/ml_judge.py (multilingual paired judge). No live serve needed."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "quality"))

import ml_judge  # noqa: E402


class TestPrompts(unittest.TestCase):
    def test_forty_per_language_five_tasks(self) -> None:
        rows = json.loads(ml_judge.PROMPTS.read_text(encoding="utf-8"))
        self.assertEqual(len({r["id"] for r in rows}), 240)
        for lang in ml_judge.LANGS:
            mine = [r for r in rows if r["lang"] == lang]
            self.assertEqual(len(mine), 40, lang)
            self.assertEqual({r["task"] for r in mine}, {"explain", "summarise", "email", "translate", "reason"})
            self.assertTrue(all(r["answer"] for r in mine if r["task"] == "reason"))
            # the long prompts are themselves in their language (one-line questions can lack function words)
            self.assertTrue(all(ml_judge.lang_id_ok(r["prompt"], lang) for r in mine
                                if r["task"] in ("summarise", "email")), lang)


class TestMetrics(unittest.TestCase):
    def test_lang_id(self) -> None:
        ok = ml_judge.lang_id_ok
        self.assertTrue(ok("Der Himmel ist blau, weil das Licht gestreut wird und die Luft es nicht absorbiert.", "de"))
        self.assertFalse(ok("The sky is blue because light is scattered by the air.", "de"))
        self.assertTrue(ok("El cielo es azul porque la luz se dispersa con las moléculas del aire.", "es"))
        self.assertTrue(ok("Le ciel est bleu parce que la lumière est diffusée dans l'air.", "fr"))
        self.assertTrue(ok("空が青いのは、光が空気によって散乱されるからです。", "ja"))
        self.assertFalse(ok("天空是蓝色的，因为光被空气散射。", "ja"))
        self.assertTrue(ok("天空是蓝色的，因为光被空气散射。", "zh"))
        self.assertFalse(ok("空が青いのは、光が空気によって散乱されるからです。", "zh"))
        self.assertFalse(ok("", "en"))

    def test_rep4(self) -> None:
        self.assertEqual(ml_judge.rep4("a b c d e f", "en"), 0.0)
        self.assertGreater(ml_judge.rep4("a b c d a b c d a b c d", "en"), 0.5)
        self.assertGreater(ml_judge.rep4("好好学习好好学习好好学习", "zh"), 0.5)

    def test_answer_ok(self) -> None:
        self.assertTrue(ml_judge.answer_ok("Work...\n\n**Final answer: 9 apples**", ["9"]))
        self.assertFalse(ml_judge.answer_ok("Final answer: 19", ["9"]))
        self.assertFalse(ml_judge.answer_ok("9 is wrong\nFinal answer: 10", ["9"]))
        self.assertTrue(ml_judge.answer_ok("Endpreis: 54,00 €", ["54"]))
        self.assertTrue(ml_judge.answer_ok("答え：土曜日", ["土曜"]))
        self.assertTrue(ml_judge.answer_ok("Il arrive à 17 h 25.", ["17h25"]))

    def test_sign_test(self) -> None:
        self.assertEqual(ml_judge.sign_test(0, 0)["p_two_sided"], 1.0)
        s = ml_judge.sign_test(1, 9)
        self.assertLess(s["p_against_cand"], 0.05)
        self.assertAlmostEqual(s["p_two_sided"], 0.021484, places=5)
        self.assertGreater(ml_judge.sign_test(9, 1)["p_against_cand"], 0.99)


class FakeClient:
    """Judge that always prefers the response containing 'GOOD' (whatever its position)."""
    url = "fake"

    def __init__(self, *a, **k):
        pass

    def get(self, path):
        raise OSError("offline")

    def chat(self, messages, **body):
        text = messages[-1]["content"]
        a = text.split("[Response A]")[1].split("[Response B]")[0]
        b = text.split("[Response B]")[1]
        v = "A" if "GOOD" in a and "GOOD" not in b else "B" if "GOOD" in b and "GOOD" not in a else "TIE"
        return {"content": f"Verdict: {v}", "reasoning": ""}


class TestJudge(unittest.TestCase):
    def test_both_orders_and_rules(self) -> None:
        prompts = json.loads(ml_judge.PROMPTS.read_text(encoding="utf-8"))
        de = [p for p in prompts if p["lang"] == "de"][:12]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for arm in ("ref", "cand"):
                (d / arm).mkdir()
                with open(d / arm / "ml.jsonl", "w", encoding="utf-8") as f:
                    for i, p in enumerate(de):
                        good = (arm == "ref") == (i < 10)  # ref wins 10, cand wins 2
                        txt = "Das ist die Antwort und sie ist gut." + (" GOOD" if good else "")
                        f.write(json.dumps({"id": p["id"], "content": txt, "finish_reason": "stop"}) + "\n")
            orig_prompts, orig_client = ml_judge.PROMPTS, ml_judge.Client
            (d / "p.json").write_text(json.dumps(de), encoding="utf-8")
            ml_judge.PROMPTS, ml_judge.Client = d / "p.json", FakeClient
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = ml_judge.main(["judge", "--ref", str(d / "ref"), "--cand", str(d / "cand"),
                                        "--out", str(d / "j"), "--workers", "2"])
            finally:
                ml_judge.PROMPTS, ml_judge.Client = orig_prompts, orig_client
            s = json.loads((d / "j" / "summary.json").read_text())
            self.assertEqual((s["overall"]["win"], s["overall"]["loss"]), (2, 10))
            self.assertEqual(s["overall"]["judge_calls"], 24)
            self.assertEqual(rc, 1)
            self.assertEqual(s["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()
