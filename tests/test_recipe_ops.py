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
    "vllm/vllm-openai:nightly-aarch64@"
    "sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b"
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
        self.assertIn("MTP_OVERLAY='/tmp/qwen38-modelopt.py'", ssh_block)
        self.assertIn("VLLM_PLE_FP8_CHECKPOINT='$VLLM_PLE_FP8_CHECKPOINT'", ssh_block)
        self.assertIn("VLLM_ALLOW_LONG_MAX_MODEL_LEN='$VLLM_ALLOW_LONG_MAX_MODEL_LEN'", ssh_block)
        self.assertIn('--revision "$SNAPSHOT_SHA"', run)
        self.assertIn(".run-state/worker_host", run)
        self.assertNotIn("starting local rank only", run)
        self.assertIn('scp -q "$PLE_OVERLAY"', run)
        self.assertIn('scp -q "$MTP_OVERLAY"', run)

    def test_resolve_model_does_not_fall_back_to_hub_id(self) -> None:
        body = _func_body(_read("run.sh"), "resolve_model")
        self.assertNotIn("$MODEL", body)
        self.assertIn("$SNAPSHOT_IN_CONTAINER", body)

    def test_snapshot_hub_dir_follows_model_id(self) -> None:
        run = _read("run.sh")
        m = re.search(r'^MODEL="\$\{MODEL:-([^}]+)\}"$', run, re.M)
        self.assertIsNotNone(m)
        hub = "models--" + m.group(1).replace("/", "--")
        self.assertIn(f'SNAPSHOT="${{HF_CACHE}}/hub/{hub}/snapshots/${{SNAPSHOT_SHA}}"', run)
        self.assertIn(
            f'SNAPSHOT_IN_CONTAINER="${{HF_HOME_IN_CONTAINER}}/hub/{hub}/snapshots/${{SNAPSHOT_SHA}}"',
            run,
        )

    def test_image_is_digest_pinned(self) -> None:
        self.assertIn(PINNED_IMAGE, _read("run.sh"))
        self.assertIn(PINNED_IMAGE, _read("recipe.yaml"))

    def test_ple_overlay_has_fp8_gate(self) -> None:
        overlay = _read("docker/ple_layer.py")
        env_gate = 'os.environ.get("VLLM_PLE_FP8_CHECKPOINT") == "1"' in overlay
        mixed = (
            "Qwen4ExpPLEFp8EmbeddingMethod" in overlay
            and "ModelOptMixedPrecisionConfig" in overlay
        )
        self.assertTrue(env_gate or mixed, "PLE overlay needs an FP8 PLE selector")

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

    def test_validate_only_accepts_occupancy_pin(self) -> None:
        proc = _run_sh(MAX_NUM_SEQS="8")
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_validate_only_refuses_max_num_seqs(self) -> None:
        proc = _run_sh(MAX_NUM_SEQS="16")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("exceeds 8", proc.stderr)

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

    def test_validate_only_refuses_b12x_moe(self) -> None:
        proc = _run_sh(MOE_BACKEND="b12x")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("b12x", proc.stderr)

    def test_validate_only_refuses_ple_gate_off(self) -> None:
        proc = _run_sh(VLLM_PLE_FP8_CHECKPOINT="0")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("validate-only", proc.stdout)
        run = _read("run.sh")
        self.assertNotIn("VLLM_PLE_FP8_CHECKPOINT=$VLLM_PLE_FP8_CHECKPOINT. Mixed ModelOpt", run)
        self.assertNotIn('-e "VLLM_PLE_FP8_CHECKPOINT=$VLLM_PLE_FP8_CHECKPOINT"', run)

    def test_validate_only_refuses_1m_without_allow_long(self) -> None:
        proc = _run_sh(MAX_MODEL_LEN="1048576", VLLM_ALLOW_LONG_MAX_MODEL_LEN="0")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("VLLM_ALLOW_LONG_MAX_MODEL_LEN", proc.stderr)

    def test_validate_only_accepts_native_window_without_allow(self) -> None:
        proc = _run_sh(MAX_MODEL_LEN="262144", VLLM_ALLOW_LONG_MAX_MODEL_LEN="0")
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_gitignore_run_state(self) -> None:
        self.assertIn(".run-state/", _read(".gitignore"))

    def test_mtp_spec_uses_triton_for_unquantized_experts(self) -> None:
        run = _read("run.sh")
        self.assertIn('"moe_backend":"triton"', run)
        self.assertIn("--moe-backend", run)

    def test_mtp_overlay_dispatches_fp8_block_scales(self) -> None:
        overlay = _read("docker/modelopt.py")
        self.assertIn('marker = "mtp.layers."', overlay)
        self.assertIn("_BLOCK_FP8_MOE_ALGOS", overlay)
        self.assertIn("FP8_PB_WO", overlay)
        self.assertIn("FP8_BLOCK_SCALES", overlay)
        self.assertIn("fp8_block_config", overlay)
        self.assertIn("Fp8MoEMethod", overlay)
        self.assertIn("weight_block_size", overlay)
        mixed = overlay.split("class ModelOptMixedPrecisionConfig", 1)[1]
        blocked = mixed.split("def has_blocked_weights", 1)[1]
        self.assertIn("_BLOCK_FP8_MOE_ALGOS", blocked[:400])
        run = _read("run.sh")
        self.assertIn("${MTP_OVERLAY}:${MTP_IN_CONTAINER}", run)

    def test_mtp_overlay_apply_is_idempotent_on_current_file(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "apply_mtp_fp8_overlay",
            ROOT / "docker" / "apply_mtp_fp8_overlay.py",
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        text = _read("docker/modelopt.py")
        self.assertEqual(mod.overlay(text), text)

    def test_serve_argv_matches_official_flash_next_flags(self) -> None:
        body = _func_body(_read("run.sh"), "start_local")
        self.assertIn("--no-enable-flashinfer-autotune", body)
        self.assertIn("--mamba-cache-mode align", body)
        self.assertIn("--quantization modelopt", body)
        self.assertIn("--enable-prefix-caching", body)
        self.assertIn("--enable-chunked-prefill", body)
        self.assertNotIn("--enable-expert-parallel", body)
        self.assertNotIn("--language-model-only", body)
        self.assertNotIn("qwen3_coder", body)
        self.assertNotRegex(body, r"--kv-cache-dtype fp8\b")

    def test_measured_decode_rows_are_prose_only(self) -> None:
        measured = _read("recipe.yaml").split("measured:", 1)[1]
        self.assertIn("Structured and code cells are not a decode score", measured)
        self.assertIn("phase: prose", measured)
        self.assertNotIn("phase: structured", measured)
        self.assertNotIn("phase: code", measured)

    def test_bench_decode_defaults_to_prose_score(self) -> None:
        bench = _read("bench_decode.py")
        self.assertIn('default="prose"', bench)
        self.assertIn("decode_score=prose", bench)
        self.assertIn("Do not keep or revert a pin from a structured or code cell", bench)

    def test_parsers_are_from_the_card_shape(self) -> None:
        run = _read("run.sh")
        self.assertIn('--tool-call-parser "$TOOL_CALL_PARSER"', run)
        self.assertIn('--reasoning-parser "$REASONING_PARSER"', run)
        self.assertIn("qwen3_xml", run)
        self.assertIn("enable_thinking", run)


if __name__ == "__main__":
    unittest.main()
