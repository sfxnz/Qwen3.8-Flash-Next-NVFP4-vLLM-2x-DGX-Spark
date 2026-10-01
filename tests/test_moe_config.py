#!/usr/bin/env python3
"""L6a: docker/v030/moe_configs holds the GB10 drafter MoE config v0.30.0 looks up.

Host (no torch): the checker accepts the seed and rejects broken files; the synthetic
tuner config matches the checkpoint. In the v0.30.0 image (CPU only, no --gpus), the
vLLM tests below also run the real filename and parsing code:

  docker run --rm --network none --memory 4g -e PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint python3 -v "$PWD:/w" -w /w vllm/vllm-openai:v0.30.0-aarch64 \
    -m unittest discover -s tests -p test_moe_config.py -v
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kernels" / "tune_draft_moe.sh"
CONFIG_DIR = ROOT / "docker" / "v030" / "moe_configs"
CFG_NAME = "E=512,N=320,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[64,64].json"
SEED_SRC = "E=512,N=320,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[64,64].json"
BENCH_DIR = "/vllm-workspace/benchmarks/kernels"
HF_CONFIG = Path.home() / (
    ".cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/"
    "fab0aecb760cec45227f6656abcaafa11abca87a/config.json"
)
HAS_VLLM = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("vllm") is not None


def _check(d: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), "check", str(d)], capture_output=True, text=True)


def _seed() -> dict:
    return json.loads((CONFIG_DIR / CFG_NAME).read_text())


class HostCheck(unittest.TestCase):
    def test_dir_holds_exactly_the_lookup_name(self):
        self.assertEqual(sorted(p.name for p in CONFIG_DIR.iterdir()), [CFG_NAME])

    def test_checker_accepts_seed(self):
        r = _check(CONFIG_DIR)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("CHECK OK", r.stdout)

    def _broken(self, mutate, name=CFG_NAME) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as t:
            cfg = _seed()
            text = mutate(cfg)
            (Path(t) / name).write_text(text if isinstance(text, str) else json.dumps(cfg))
            return _check(Path(t))

    def test_checker_rejects(self):
        cases = {
            "wrong name": (lambda c: None, CFG_NAME.replace("GB10", "B200")),
            "not json": (lambda c: "{", CFG_NAME),
            "provenance key": (lambda c: c.update(note="x"), CFG_NAME),
            "missing num_stages": (lambda c: c["1"].pop("num_stages"), CFG_NAME),
            "BLOCK_N 96": (lambda c: c["8"].update(BLOCK_SIZE_N=96), CFG_NAME),
            "BLOCK_N 48 unaligned": (lambda c: c["8"].update(BLOCK_SIZE_N=48), CFG_NAME),
            "SPLIT_K 2": (lambda c: c["8"].update(SPLIT_K=2), CFG_NAME),
            "smem": (lambda c: c["8192"].update(BLOCK_SIZE_M=256, num_stages=8), CFG_NAME),
            "no M=1": (lambda c: c.pop("1"), CFG_NAME),
        }
        for label, (mutate, name) in cases.items():
            with self.subTest(label):
                r = self._broken(mutate, name)
                self.assertEqual(r.returncode, 1, r.stdout)
                self.assertIn("CHECK FAIL", r.stdout)

    def test_extra_file_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            shutil.copy(CONFIG_DIR / CFG_NAME, t)
            (Path(t) / "other.json").write_text("{}")
            self.assertEqual(_check(Path(t)).returncode, 1)

    def test_seed_covers_decode_and_prefill(self):
        keys = {int(k) for k in _seed() if k != "triton_version"}
        self.assertTrue({1, 2, 4, 8, 16, 32, 64}.issubset(keys), keys)
        self.assertEqual(max(keys), 8192)  # MAX_NUM_BATCHED_TOKENS default

    def test_bench_dry_run_mounts_absolute_paths(self):
        # docker -v treats a relative host path as a volume name and refuses it; the
        # runbook passes CFG=evidence/... relative. Triton cache must stay off the mount
        # (root-owned files there cannot be removed by the host user).
        with tempfile.TemporaryDirectory(dir=ROOT) as t:
            env = {**os.environ, "CFG": "docker/v030/moe_configs",
                   "OUT": os.path.relpath(t, ROOT)}
            r = subprocess.run(["bash", str(SCRIPT), "--dry-run", "bench"], cwd=ROOT, env=env,
                               capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        runs = [l for l in r.stdout.splitlines() if l.startswith("docker run")]
        self.assertEqual(len(runs), 4)  # ABAB
        self.assertEqual(sum("VLLM_TUNED_CONFIG_FOLDER=/cfg" in l for l in runs), 2)
        for line in runs:
            toks = line.split()
            mounts = [toks[i + 1] for i, x in enumerate(toks) if x == "-v"]
            self.assertTrue(mounts and all(m.startswith("/") for m in mounts), mounts)
            self.assertIn("TRITON_CACHE_DIR=/tmp/", line)

    @unittest.skipUnless(HF_CONFIG.exists(), "checkpoint config.json not on this host")
    def test_synth_matches_checkpoint(self):
        with tempfile.TemporaryDirectory() as t:
            subprocess.run(["bash", str(SCRIPT), "synth", t], check=True)
            synth = json.loads((Path(t) / "config.json").read_text())
        ck = json.loads(HF_CONFIG.read_text())
        ck = ck.get("text_config", ck)
        for k in ("num_experts", "num_experts_per_tok", "moe_intermediate_size", "hidden_size"):
            self.assertEqual(synth[k], ck[k], k)
        # moe_intermediate_size / TP=2 = the N in the U1 missing-config warning.
        self.assertEqual(synth["moe_intermediate_size"] // 2, 320)


@unittest.skipUnless(HAS_VLLM, "needs torch + vllm (run inside the v0.30.0 image, CPU only)")
class Vllm030(unittest.TestCase):
    def setUp(self):
        from vllm.model_executor.layers.fused_moe import fused_moe as fm
        from vllm.platforms import current_platform

        self.fm = fm
        fm.get_moe_configs.cache_clear()
        self.addCleanup(fm.get_moe_configs.cache_clear)
        # Raw name torch reports on GB10 (evidence/s11-micro device.json: "NVIDIA GB10").
        p = mock.patch.object(type(current_platform), "get_device_name",
                              classmethod(lambda cls, device_id=0: "NVIDIA GB10"))
        p.start()
        self.addCleanup(p.stop)

    def test_version(self):
        import vllm

        self.assertEqual(vllm.__version__, "0.30.0")

    def test_filename_matches_v030(self):
        # Same call try_get_optimal_moe_config makes: w2 (E, K, N), block [64,64].
        self.assertEqual(self.fm.get_config_file_name(512, 320, "fp8_w8a8", [64, 64]), CFG_NAME)

    def test_loads_from_tuned_folder(self):
        with mock.patch.dict(os.environ, {"VLLM_TUNED_CONFIG_FOLDER": str(CONFIG_DIR)}):
            got = self.fm.get_moe_configs(512, 320, "fp8_w8a8", 64, 64)
        want = {int(k): v for k, v in _seed().items() if k != "triton_version"}
        self.assertEqual(got, want)

    def test_optimal_config_lookup(self):
        seed = _seed()
        with mock.patch.dict(os.environ, {"VLLM_TUNED_CONFIG_FOLDER": str(CONFIG_DIR)}):
            for m, key in ((1, "1"), (3, "2"), (8, "8"), (32, "32"), (8192, "8192")):
                with self.subTest(M=m):
                    cfg = self.fm.try_get_optimal_moe_config(
                        (512, 640, 2560), (512, 2560, 320), 10, "fp8_w8a8", m, [64, 64])
                    self.assertEqual(cfg, seed[key])

    def test_unset_folder_is_stock_default(self):
        env = {k: v for k, v in os.environ.items() if k != "VLLM_TUNED_CONFIG_FOLDER"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsNone(self.fm.get_moe_configs(512, 320, "fp8_w8a8", 64, 64))

    @unittest.skipUnless(os.path.isdir(BENCH_DIR), "benchmark_moe.py not in this image")
    def test_tuner_wrapper_targets_same_name(self):
        """Exec the script's own wrapper (ray stub + search space) in "plan" mode."""
        import torch

        wrapper = subprocess.run(["bash", str(SCRIPT), "wrapper"], check=True,
                                 capture_output=True, text=True).stdout
        saved = {k: sys.modules.get(k) for k in ("ray", "ray.experimental", "ray.experimental.tqdm_ray")}
        self.addCleanup(lambda: [sys.modules.pop(k, None) if v is None else sys.modules.__setitem__(k, v)
                                 for k, v in saved.items()])
        with tempfile.TemporaryDirectory() as t:
            subprocess.run(["bash", str(SCRIPT), "synth", f"{t}/synth"], check=True)
            ns: dict = {"__name__": "wrapper"}
            with mock.patch.object(sys, "argv", ["wrapper", "plan"]), \
                 mock.patch.dict(os.environ, {"SYNTH": f"{t}/synth", "BENCH_DIR": BENCH_DIR}):
                exec(compile(wrapper, "tune_draft_moe.sh:TUNE_PY", "exec"), ns)
            plan = ns["PLAN"]
            self.assertEqual(plan["file"], CFG_NAME)
            self.assertEqual((plan["E"], plan["topk"], plan["N"], plan["hidden"]), (512, 10, 320, 2560))
            self.assertEqual(plan["block"], [64, 64])
            # stock 5x4x3x4x2x4 = 1920; K=64 only -> 640; + num_warps 2 twins -> 960
            self.assertEqual(plan["space"], 960)
            bm = sys.modules["benchmark_moe"]
            space = bm.get_configs_compute_bound(False, [64, 64])
            self.assertEqual({c["BLOCK_SIZE_K"] for c in space}, {64})
            self.assertEqual({c["num_warps"] for c in space}, {2, 4, 8})
            # The tuner's writer (save_configs) lands on the same name.
            bm.save_configs({1: _seed()["1"]}, 512, 640, 2560, 10, torch.bfloat16,
                            True, False, False, [64, 64], f"{t}/out")
            self.assertEqual(os.listdir(f"{t}/out"), [CFG_NAME])

    def test_seed_is_image_b200_file(self):
        stock = Path(self.fm.__file__).parent / "configs" / SEED_SRC
        self.assertEqual(stock.read_bytes(), (CONFIG_DIR / CFG_NAME).read_bytes())
        self.assertFalse((stock.parent / CFG_NAME).exists(), "image already ships a GB10 file")


if __name__ == "__main__":
    unittest.main()
