#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import socket
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED_IMAGE = (
    "vllm/vllm-openai:qwen38-flash-next@"
    "sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e"
)


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


def _env(**extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("FORCE_UNSAFE_CTX", None)
    env.pop("FORCE_UNSAFE_MOE", None)
    env.update(extra)
    return env


def _run_sh(**extra: str) -> subprocess.CompletedProcess[str]:
    env = _env(**extra)
    env["VALIDATE_ONLY"] = "1"
    return subprocess.run(
        [str(ROOT / "run.sh")],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )


def _func_body(src: str, name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{(.*?)^\}}", src, re.M | re.S)
    if m is None:
        raise AssertionError(f"missing function {name}")
    return m.group(1)


class RecipeOpsTests(unittest.TestCase):
    def test_stop_sh_ssh_probe_has_else_exit_1(self) -> None:
        stop = _read("stop.sh")
        self.assertRegex(
            stop,
            r"ssh -o BatchMode=yes.*\n(?:.*\n)*?\s+else\n(?:.*\n)*?\s+exit 1",
        )
        self.assertIn(">&2", stop)

    def test_stop_sh_reads_worker_host_state_then_default(self) -> None:
        stop = _read("stop.sh")
        self.assertIn(".run-state/worker_host", stop)
        self.assertLess(stop.find(".run-state/worker_host"), stop.find("WORKER_HOST:-spark2"))

    def test_stop_sh_orchestrate_zero_is_local_only(self) -> None:
        stop = _read("stop.sh")
        self.assertRegex(stop, re.compile(r'ORCHESTRATE" == "0".*exit 0', re.S))
        proc = subprocess.run(
            [str(ROOT / "stop.sh")],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env=_env(
                ORCHESTRATE="0",
                CONTAINER_NAME="qwen38-ops-test-no-such-container",
            ),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_stop_sh_ssh_probe_fails_loud_when_not_spark2(self) -> None:
        host = socket.gethostname().split(".", 1)[0].lower()
        if host.startswith("spark2"):
            self.skipTest("this host is spark2")
        proc = subprocess.run(
            [str(ROOT / "stop.sh")],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env=_env(
                ORCHESTRATE="auto",
                WORKER_HOST="no-such-host-xyz",
                CONTAINER_NAME="qwen38-ops-test-no-such-container",
            ),
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue(proc.stderr.strip())

    def test_run_sh_worker_ssh_forwards_snapshot_and_overlay(self) -> None:
        run = _read("run.sh")
        ssh_idx = run.find('ssh "$WORKER_HOST"')
        self.assertGreater(ssh_idx, 0)
        ssh_block = run[ssh_idx : ssh_idx + 3500]
        self.assertIn("SNAPSHOT_SHA='$SNAPSHOT_SHA'", ssh_block)
        self.assertIn("HF_CACHE='$HF_CACHE'", ssh_block)
        self.assertIn("MODEL='$MODEL'", ssh_block)
        self.assertIn("PLE_OVERLAY='/tmp/qwen38-ple_layer.py'", ssh_block)
        self.assertIn("VLLM_PLE_FP8_CHECKPOINT='$VLLM_PLE_FP8_CHECKPOINT'", ssh_block)
        self.assertIn("VLLM_ALLOW_LONG_MAX_MODEL_LEN='$VLLM_ALLOW_LONG_MAX_MODEL_LEN'", ssh_block)
        self.assertIn('--revision "$SNAPSHOT_SHA"', run)
        self.assertIn(".run-state/worker_host", run)
        self.assertNotIn("starting local rank only", run)
        self.assertIn('scp -q "$PLE_OVERLAY"', run)

    def test_resolve_model_does_not_fall_back_to_hub_id(self) -> None:
        body = _func_body(_read("run.sh"), "resolve_model")
        self.assertNotIn("$MODEL", body)
        self.assertIn("$SNAPSHOT_IN_CONTAINER", body)

    def test_image_is_digest_pinned(self) -> None:
        self.assertIn(PINNED_IMAGE, _read("run.sh"))
        self.assertIn(PINNED_IMAGE, _read("recipe.yaml"))

    def test_ple_overlay_has_fp8_gate(self) -> None:
        overlay = _read("docker/ple_layer.py")
        self.assertIn('os.environ.get("VLLM_PLE_FP8_CHECKPOINT") == "1"', overlay)
        self.assertIn("Qwen3_8FlashNextPLEFp8EmbeddingMethod", overlay)
        gate = overlay.find("VLLM_PLE_FP8_CHECKPOINT")
        fp8 = overlay.find("isinstance(quant_config, Fp8Config)")
        self.assertGreater(gate, 0)
        self.assertGreater(fp8, gate)

    def test_head_preflight_before_worker_scp(self) -> None:
        run = _read("run.sh")
        idx = run.find('ORCHESTRATE" == "auto" && "$ROLE" == "head"')
        self.assertGreater(idx, 0)
        block = run[idx:]
        scp = block.find("scp ")
        self.assertGreater(scp, 0)
        self.assertGreater(block.find("refuse_foreign_serve"), -1)
        self.assertGreater(block.find("refuse_busy_port"), -1)
        self.assertLess(block.find("refuse_foreign_serve"), scp)
        self.assertLess(block.find("refuse_busy_port"), scp)
        wait = _func_body(run, "wait_ready")
        self.assertIn("$SERVED_NAME", wait)
        self.assertIn("/health", wait)

    def test_validate_only_defaults_pass(self) -> None:
        proc = _run_sh()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("validate-only", proc.stdout)

    def test_validate_only_refuses_max_num_seqs(self) -> None:
        proc = _run_sh(MAX_NUM_SEQS="8")
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue(proc.stderr.strip())

    def test_validate_only_refuses_window_above_1m(self) -> None:
        proc = _run_sh(MAX_MODEL_LEN="2097152")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("above 1048576", proc.stderr)

    def test_validate_only_refuses_marlin_moe(self) -> None:
        proc = _run_sh(MOE_BACKEND="marlin")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("unquantized MTP", proc.stderr)

    def test_validate_only_refuses_flashinfer_cutlass_moe(self) -> None:
        proc = _run_sh(MOE_BACKEND="flashinfer_cutlass")
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue(proc.stderr.strip())

    def test_validate_only_refuses_ple_gate_off(self) -> None:
        proc = _run_sh(VLLM_PLE_FP8_CHECKPOINT="0")
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue(proc.stderr.strip())

    def test_validate_only_refuses_1m_without_allow_long(self) -> None:
        proc = _run_sh(VLLM_ALLOW_LONG_MAX_MODEL_LEN="0")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("VLLM_ALLOW_LONG_MAX_MODEL_LEN", proc.stderr)

    def test_gitignore_run_state(self) -> None:
        self.assertIn(".run-state/", _read(".gitignore"))

    def test_mtp_spec_uses_triton_for_unquantized_experts(self) -> None:
        run = _read("run.sh")
        self.assertIn('"moe_backend":"triton"', run)
        self.assertIn("--moe-backend", run)

    def test_parsers_are_from_the_card_shape(self) -> None:
        run = _read("run.sh")
        self.assertIn('--tool-call-parser "$TOOL_CALL_PARSER"', run)
        self.assertIn('--reasoning-parser "$REASONING_PARSER"', run)
        self.assertIn("qwen3_xml", run)
        self.assertIn("enable_thinking", run)


if __name__ == "__main__":
    unittest.main()
