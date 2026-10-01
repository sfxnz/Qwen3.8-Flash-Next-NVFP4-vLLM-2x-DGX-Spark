#!/usr/bin/env python3
"""Offline checks for tools/micro (plan 0-9): no GPU, docker or ssh."""
from __future__ import annotations

import contextlib
import io
import json
import os
import py_compile
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MICRO = ROOT / "tools" / "micro"
sys.path.insert(0, str(MICRO))

import common  # noqa: E402
import gonogo  # noqa: E402
import hc_bench  # noqa: E402
import moe_nvfp4  # noqa: E402
import nccl_allreduce_sweep as nas  # noqa: E402
import nccl_decode_sweep as nds  # noqa: E402

PIN_DIGEST = "sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b"
V030_DIGEST = "sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56"


def _dry(out: Path, **extra: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in ("IMAGES", "NCCL_IMAGES", "STEPS", "IMAGE_PIN", "IMAGE_V030")}
    env.update(OUT=str(out), **extra)
    return subprocess.run([str(MICRO / "run_s11.sh"), "--dry-run"], capture_output=True, text=True, env=env,
                          cwd=str(ROOT), check=False)


class Static(unittest.TestCase):
    def test_py_compile(self):
        for p in sorted(MICRO.glob("*.py")):
            with tempfile.TemporaryDirectory() as d:
                py_compile.compile(str(p), cfile=str(Path(d) / "x.pyc"), doraise=True)

    def test_bash_syntax(self):
        for p in sorted(MICRO.glob("*.sh")):
            subprocess.run(["bash", "-n", str(p)], check=True)

    def test_nccl_env_matches_run_sh(self):
        run = (ROOT / "run.sh").read_text()
        self.assertIn(f'HCA="${{HCA:-{common.HCA}}}"', run)
        self.assertIn(f'IFACE="${{IFACE:-{common.IFACE}}}"', run)
        self.assertIn(f'HEAD_IP="${{HEAD_IP:-{common.HEAD_IP}}}"', run)
        for k, v in common.NCCL_BASE_ENV.items():
            if k != "NCCL_IB_HCA":
                self.assertIn(f'"{k}={v}"', run, k)


class DryRun(unittest.TestCase):
    def test_dry_run_prints_plan_and_touches_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "s11"
            r = _dry(out)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertFalse(out.exists(), "dry-run must not create the output dir")
        s = r.stdout
        self.assertIn(PIN_DIGEST, s)
        self.assertIn(V030_DIGEST, s)
        self.assertNotRegex(s, r"nightly-aarch64(?!@sha256)")
        for script in ("device_facts.py", "gemm_sweep.py", "hc_bench.py", "fp8_sweep.py", "moe_nvfp4.py",
                       "nccl_decode_sweep.py", "nccl_allreduce_sweep.py", "gonogo.py"):
            self.assertIn(script, s)
        self.assertIn("NCCL_GRAPH_MIXING_SUPPORT=0", s)
        self.assertIn("NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1", s)
        self.assertIn("CUBLASLT_LOG_LEVEL=5", s)
        self.assertIn("NCCL_CROSS_NIC=1", s)
        self.assertIn("NCCL_SOCKET_IFNAME=enp1s0f1np1", s)
        self.assertIn("--master 10.100.8.1:", s)
        # every GPU container is a printed command, never executed
        self.assertTrue(all(line.startswith(("+", "==")) for line in s.splitlines() if line))

    def test_dry_run_subset(self):
        with tempfile.TemporaryDirectory() as d:
            r = _dry(Path(d) / "s", IMAGES="pin", NCCL_IMAGES="pin", STEPS="hc")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("hc_bench.py", r.stdout)
        self.assertNotIn("/bench/gemm_sweep.py", r.stdout)
        self.assertNotIn("nccl_decode_sweep.py run", r.stdout)

    def test_refuses_unpinned_image(self):
        with tempfile.TemporaryDirectory() as d:
            r = _dry(Path(d) / "s", IMAGE_PIN="vllm/vllm-openai:nightly-aarch64")
        self.assertEqual(r.returncode, 1)
        self.assertIn("not digest-pinned", r.stderr)


class Models(unittest.TestCase):
    def test_collective_sizes(self):
        specs = {(s["op"], s["rows"]): s["bytes"] for s in nds.decode_specs()}
        self.assertEqual(specs[("all_reduce", 1)], 5120)
        self.assertEqual(specs[("all_reduce", 32)], 163840)
        self.assertEqual(specs[("all_gather_lm", 1)], 248320)
        self.assertEqual(specs[("all_gather_lm", 32)], 7946240)
        self.assertEqual(sum(nds.STEP_WEIGHTS["c1"].values()), 107)

    def test_arm_env(self):
        self.assertEqual(nds.arm_env("mix0")["NCCL_GRAPH_MIXING_SUPPORT"], "0")
        self.assertNotIn("NCCL_GRAPH_MIXING_SUPPORT", nds.arm_env("keep"))
        dual = nas.arm_env("dual")
        self.assertEqual((dual["NCCL_CROSS_NIC"], dual["NCCL_IB_MERGE_NICS"]), ("0", "0"))
        self.assertIn(nas.PREFILL_AR, nas.sizes())
        self.assertEqual(nas.PREFILL_AR, 41943040)

    def test_byte_models(self):
        self.assertEqual(moe_nvfp4.expert_bytes(320)["total"], 1382400)
        self.assertEqual(moe_nvfp4.expert_bytes(320)["fc2"], 409600 + 51200)
        self.assertEqual(hc_bench.MODULE_BYTES, 13434880)
        self.assertEqual(hc_bench.DOWN_ROWS, 336)
        self.assertEqual(moe_nvfp4.distinct_local([[1, 300], [1, 2]], e_local=256), 2)

    def test_graph_costs(self):
        ref = [[10.0] * 8 for _ in range(4)]
        w = [[30.0, 25.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0] for _ in range(4)]
        c = nds.graph_costs(w, ref)
        self.assertEqual(len(c["steady"]), 12)
        self.assertAlmostEqual(c["startup_us"], 15.0)


def _cells(ar4: float, ar1: float, ag4: float, pf: float) -> list[dict]:
    rows = []
    for m in nds.ROWS:
        ar = ar4 if m == 4 else ar1
        rows += [{"op": "all_reduce", "rows": m, "mode": "gapped", "bytes": m * 5120, "median_us": ar},
                 {"op": "all_gather", "rows": m, "mode": "gapped", "bytes": m * 5120, "median_us": ag4},
                 {"op": "all_gather_lm", "rows": m, "mode": "gapped", "bytes": m * 248320, "median_us": 100.0}]
    rows.append({"op": "all_reduce", "rows": 4, "mode": "prefetch", "bytes": 20480, "median_us": -pf,
                 "saving_us": pf})
    return rows


def _fixture(root: Path) -> Path:
    img = root / "pin"
    for arm, (ar4, ar1) in {"keep": (60.0, 60.0), "mix0": (25.0, 25.0)}.items():
        d = img / "nccl-decode" / arm
        d.mkdir(parents=True)
        (d / "rep1.rank0.json").write_text(json.dumps({"arm": arm, "rows": _cells(ar4, ar1, 15.0, 10.0)}))
    gemm_rows = []
    for name, (n, k) in {"hc_down_inject": (336, 10240), "hc_up": (10240, 320), "gdn_in_proj_qkvz_ba": (8240, 2560),
                         "gdn_out_proj|qsa_o": (2560, 3072), "qsa_qkv+index": (7296, 2560),
                         "gdn_in_proj_ba": (48, 2560), "router": (512, 2560)}.items():
        us = 37.0 if name == "hc_down_inject" else 20.0
        gb = 220.0 if name == "gdn_in_proj_qkvz_ba" else 180.0
        for m in (1, 4, 32):
            gemm_rows.append({"name": name, "n": n, "k": k, "m": m, "flush": {"median_us": us, "gbps": gb},
                              "kernels": [{"name": "sm80_xmma_gemm_splitK", "us": 30.0},
                                          {"name": "splitKreduce_kernel", "us": 5.0}]})
    (img / "gemm.json").write_text(json.dumps({"rows": gemm_rows}))
    fp8 = [{"name": "a", "m": 4, "arm": "w8a16_marlin_ch", "flush": {"gbps": 190.0}},
           {"name": "b", "m": 4, "arm": "w8a16_marlin_ch", "flush": {"gbps": 185.0}},
           {"name": "a", "m": 4, "arm": "w8a8_cutlass", "flush": {"gbps": 140.0}},
           {"name": "a", "m": 4, "arm": "w8a16_marlin_b128", "skipped": "RuntimeError: nope"}]
    (img / "fp8.json").write_text(json.dumps({"rows": fp8}))
    (img / "drafthead.json").write_text(json.dumps({"rows": [
        {"v_prime": 32768, "m": 1, "flush": {"gbps": 205.0, "median_us": 410.0}}]}))
    dev = {"meta": {"host": "spark1"}, "check": {"sm_count": [48, 48, True], "l2_bytes": [25165824, 25165824, True]}}
    (img / "device.json").write_text(json.dumps(dev))  # head name, as run_s11.sh writes it
    (img / "device.spark2.json").write_text(json.dumps({**dev, "meta": {"host": "spark2"}}))
    (img / "allreduce").mkdir()
    for arm, us in (("single", 4000.0), ("dual", 2500.0)):
        (img / "allreduce" / f"{arm}.json").write_text(json.dumps({"rows": [
            {"bytes": nas.PREFILL_AR, "us": us}, {"bytes": 8, "us": 20.0 if arm == "single" else 21.0}]}))
    return root


class GoNoGo(unittest.TestCase):
    def test_table_from_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            root = _fixture(Path(d))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(gonogo.main([str(root)]), 0)
            table = json.loads((root / "gonogo.json").read_text())["pin"]
            md = (root / "gonogo.md").read_text()
        v = {r["measurement"]: r for r in table}
        l2a = v["In-graph AR 5-160 KB, mixing on vs off (L2a booking)"]
        self.assertEqual(l2a["verdict"], "GO")
        self.assertIn("booked c1 1.87 ms", l2a["value"])  # 0.5 x 107 x 35 us
        self.assertEqual(v["AG vs AR at c=1 payload, mixing off (L2d)"]["verdict"], "GO")  # 15/25 = 0.6
        self.assertEqual(v["L2 prefetch of a 13.4 MB module during a real AR"]["verdict"], "NO-GO")
        hc = v["cuBLAS choice for HC down (336,10240) and HC up at M=4"]
        self.assertEqual(hc["verdict"], "GO")
        self.assertIn("split-K", hc["note"])
        self.assertEqual(v["cuBLAS GB/s at (8240,2560) (gdn_in_proj_qkvz_ba)"]["verdict"], "NO-GO")  # 220 >= 215
        self.assertEqual(v["cuBLAS GB/s at (48,2560) (gdn_in_proj_ba)"]["verdict"], "GO")  # 180 < 195
        self.assertEqual(v["w8a16_marlin_ch GB/s at the P4-1 shapes"]["verdict"], "GO")  # worst 185
        self.assertEqual(v["w8a8_cutlass GB/s at the P4-1 shapes"]["verdict"], "NO-GO")  # 140 < 150
        self.assertEqual(v["w8a16_marlin_b128 GB/s at the P4-1 shapes"]["verdict"], "MISSING")
        self.assertEqual(v["NVFP4 MoE TP2 routed vs floor"]["verdict"], "MISSING")
        self.assertEqual(v["Draft-head probe F.linear [V'/2,2560] + argmax"]["verdict"], "GO")
        self.assertIn("1.6x", v["42 MB all-reduce, single vs dual rail"]["value"])
        self.assertIn("| Measurement | Value |", md)
        self.assertEqual(v["deviceQuery facts (spark1)"]["verdict"], "INFO")
        self.assertEqual(v["deviceQuery facts (spark2)"]["verdict"], "INFO")

    def test_moe_rules(self):
        moe = {"rows": [{"config": "tp2", "m": 4, "routing": "uniform", "input": "fp4_in",
                         "flush": {"median_us": 300.0}, "floor_us": 200.0, "x_floor": 1.5,
                         "distinct_local_experts": 39, "fc2_us": 150.0, "fc2_floor_us": 80.0, "fc2_x_floor": 1.875}]}
        rows = {r["measurement"]: r["verdict"] for r in gonogo.rules_moe(moe)}
        self.assertEqual(rows["NVFP4 MoE TP2 routed vs floor"], "NO-GO")
        self.assertEqual(rows["NVFP4 MoE TP2 FC2 vs floor"], "GO")

    def test_empty_dir_is_all_missing(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "v030").mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                gonogo.main([d])
            table = json.loads((Path(d) / "gonogo.json").read_text())["v030"]
        self.assertTrue(table)
        self.assertTrue(all(r["verdict"] == "MISSING" for r in table))


if __name__ == "__main__":
    unittest.main()
