#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shlex
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED_IMAGE = (
    "vllm/vllm-openai:nightly-aarch64@"
    "sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b"
)
PIN_DIGEST = PINNED_IMAGE.rsplit("@", 1)[1]
V030_IMAGE = (
    "vllm/vllm-openai:v0.30.0-aarch64@"
    "sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56"
)
PIN_OVERLAYS = {
    "docker/ple_layer.py": "vllm/models/qwen4_exp/nvidia/ple_layer.py",
    "docker/modelopt.py": "vllm/model_executor/layers/quantization/modelopt.py",
    "docker/ple_ops.py": "vllm/models/qwen4_exp/nvidia/ops/ple.py",
}
V030_OVERLAYS = {
    "docker/v030/flashinfer_cutlass_moe.py": "vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py",
    "docker/v030/gdn_attn.py": "vllm/v1/attention/backends/gdn_attn.py",
    "docker/v030/qsa_indexer.py": "vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py",
    "docker/v030/serving.py": "vllm/entrypoints/openai/chat_completion/serving.py",
    "docker/v030/mtp.py": "vllm/models/qwen4_exp/nvidia/mtp.py",
}
HEADER_KEYS = ("base_image_digest", "upstream_file", "upstream_file_sha256", "upstream_PR", "generator")


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


def _env(**extra: str) -> dict[str, str]:
    env = os.environ.copy()
    for name in ("FORCE_UNSAFE_CTX", "FORCE_UNSAFE_MOE", "DIAGNOSTIC", "BENCH_ONLY", "OVERLAYS", "IMAGE",
                 "SPEC", "SPEC_CONFIG", "COMPILATION_CONFIG", "EXTRA_ENV", "EXTRA_ARGS", "MOE_BACKEND"):
        env.pop(name, None)
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


def _load(rel: str):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _header(text: str) -> dict[str, str]:
    lines = text.split("\n")
    assert lines[0] == "# R11-OVERLAY", lines[0]
    head = {}
    for line in lines[1:]:
        m = re.match(r"^# ([A-Za-z0-9_]+): (.*)$", line)
        if not m or m.group(1) not in HEADER_KEYS:
            break
        head[m.group(1)] = m.group(2)
    return head


def _body(text: str) -> str:
    lines = text.split("\n")
    return "\n".join(lines[1 + len(_header(text)) :])


def _func_src(src: str, name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{.*?^\}}\n", src, re.M | re.S)
    if m is None:
        raise AssertionError(f"missing function {name}")
    return m.group(0)


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

    def test_run_sh_worker_forwards_every_serve_setting(self) -> None:
        run = _read("run.sh")
        m = re.search(r"^FORWARD_VARS=\((.*?)\)", run, re.M | re.S)
        self.assertIsNotNone(m)
        forwarded = set(m.group(1).split())
        for name in ("SNAPSHOT_SHA", "HF_CACHE", "MODEL", "IMAGE", "SPEC_CONFIG", "COMPILATION_CONFIG",
                     "EXTRA_ARGS", "EXTRA_ENV", "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "VLLM_PLE_FP8_CHECKPOINT",
                     "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "DIAGNOSTIC", "BENCH_ONLY", "MOE_BACKEND",
                     "HF_HUB_DISABLE_XET", "FORCE_UNSAFE_CTX", "FORCE_UNSAFE_MOE"):
            self.assertIn(name, forwarded)
        self.assertIn('--revision "$SNAPSHOT_SHA"', run)
        self.assertIn('ssh "$WORKER_HOST" "$(worker_env "$remote_overlays") bash /tmp/qwen38-run.sh"', run)
        self.assertIn('scp -q "$f" "${WORKER_HOST}:$remote"', run)
        self.assertNotIn("starting local rank only", run)

    def test_worker_env_round_trips_quoted_values(self) -> None:
        run = _read("run.sh")
        forward = re.search(r"^FORWARD_VARS=\(.*?\)\n", run, re.M | re.S).group(0)
        names = forward.split("(", 1)[1].rsplit(")", 1)[0].split()
        values = {n: f"v-{n}" for n in names}
        values["SPEC_CONFIG"] = '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
        values["EXTRA_ARGS"] = "--override-generation-config '{\"temperature\": 0.7}' --x \"y z\""
        values["EXTRA_ENV"] = "NCCL_DEBUG=INFO A=$HOME"
        script = forward + _func_src(run, "worker_env")
        script += "".join(f"{n}={shlex.quote(v)}\n" for n, v in values.items())
        script += 'line="$(worker_env "/tmp/qwen38-overlays/a.py /tmp/qwen38-overlays/b.py ")"\n'
        script += 'env -i bash -c "$line env -0"\n'
        out = subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True).stdout
        got = dict(item.split("=", 1) for item in out.split("\0") if "=" in item)
        for n, v in values.items():
            self.assertEqual(got[n], v, n)
        self.assertEqual(got["ROLE"], "worker")
        self.assertEqual(got["ORCHESTRATE"], "0")
        self.assertEqual(got["OVERLAYS"], "/tmp/qwen38-overlays/a.py /tmp/qwen38-overlays/b.py ")

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
        self.assertIn(f'IMAGE="${{IMAGE:-{V030_IMAGE}}}"', _read("run.sh"))
        self.assertIn(V030_IMAGE, _read("recipe.yaml"))
        # The pin stays the documented rollback.
        self.assertIn(PINNED_IMAGE, _read("recipe.yaml"))
        proc = _run_sh(IMAGE=PINNED_IMAGE)
        self.assertEqual(proc.returncode, 0, proc.stderr)

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

    def test_serve_is_preferred_oom_victim(self) -> None:
        text = (ROOT / "run.sh").read_text()
        self.assertIn('--oom-score-adj "$OOM_SCORE_ADJ"', text)
        self.assertIn('OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-1000}"', text)
        proc = _run_sh(OOM_SCORE_ADJ="2000")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("OOM_SCORE_ADJ=2000", proc.stderr)

    def test_crashing_worker_writes_no_core(self) -> None:
        # RLIMIT_CORE=1 is the kernel's "no piped core dump" value; 0 still feeds apport.
        self.assertIn("--ulimit core=1 \\\n", _func_body(_read("run.sh"), "start_local"))

    def test_memguard_starts_with_each_rank_and_is_forwarded(self) -> None:
        text = (ROOT / "run.sh").read_text()
        self.assertIn('MEMGUARD="${MEMGUARD:-1}"', text)
        self.assertIn("    $EXTRA_ARGS\n  start_memguard\n}", text)
        for name in ("OOM_SCORE_ADJ", "MEMGUARD", "MEMGUARD_MIN_AVAIL_MB", "MEMGUARD_MIN_SWAP_FREE_MB"):
            self.assertRegex(text, rf"FORWARD_VARS=\([^)]*\b{name}\b")
        proc = _run_sh(MEMGUARD="2")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("MEMGUARD=2 must be 0 or 1", proc.stderr)

    def test_memguard_loop_kills_when_ram_and_swap_are_low(self) -> None:
        text = (ROOT / "run.sh").read_text()
        body = text[text.index("memguard_loop() {"):]
        body = body[: body.index("\n}\n") + 3]
        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp)
            (stub / "docker").write_text('#!/bin/sh\necho "$@" >> "$STUB_LOG"\necho true\n')
            (stub / "logger").write_text("#!/bin/sh\nexit 0\n")
            (stub / "sleep").write_text("#!/bin/sh\nexit 0\n")
            for f in stub.iterdir():
                f.chmod(0o755)
            log = stub / "calls"
            env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}", STUB_LOG=str(log))
            # Thresholds far above any real host: every sample is "low", so it must kill on sample 3.
            proc = subprocess.run(["bash", "-c", body + "\nmemguard_loop qwen-test 99999999 99999999"],
                                  capture_output=True, text=True, env=env, timeout=30, check=False)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("kill qwen-test", log.read_text())
            self.assertIn("docker kill qwen-test", proc.stdout)

    def test_validate_only_accepts_occupancy_pin(self) -> None:
        proc = _run_sh(MAX_NUM_SEQS="8")
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_validate_only_refuses_max_num_seqs(self) -> None:
        proc = _run_sh(MAX_NUM_SEQS="16")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("exceeds 8", proc.stderr)
        self.assertIn("THROUGHPUT_PROFILE=1", proc.stderr)

    def test_throughput_profile_serves_16_seqs(self) -> None:
        proc = _run_sh(THROUGHPUT_PROFILE="1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("seqs=16", proc.stdout)

    def test_throughput_profile_refuses_above_16(self) -> None:
        proc = _run_sh(THROUGHPUT_PROFILE="1", MAX_NUM_SEQS="24")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("exceeds 16", proc.stderr)

    def test_validate_only_refuses_window_above_1m(self) -> None:
        proc = _run_sh(MAX_MODEL_LEN="2097152")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("above 1048576", proc.stderr)

    def test_validate_only_refuses_marlin_moe(self) -> None:
        proc = _run_sh(MOE_BACKEND="marlin", IMAGE=PINNED_IMAGE)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("drafter inherits --moe-backend", proc.stderr)
        self.assertNotIn("unquantized", _read("run.sh"))

    def test_validate_only_refuses_flashinfer_cutlass_moe(self) -> None:
        proc = _run_sh(MOE_BACKEND="flashinfer_cutlass", IMAGE=PINNED_IMAGE)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("only Triton accepts", proc.stderr)
        self.assertNotIn("GLM", _read("run.sh"))

    def test_validate_only_accepts_ple_gate_off(self) -> None:
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
        self.assertIn('quant_algo == "FP8_BLOCK_SCALES"', overlay)
        self.assertIn("Fp8MoEMethod", overlay)
        self.assertIn("weight_block_size", overlay)
        self.assertIn("docker/modelopt.py", _read("recipe.yaml"))

    def test_parsers_are_from_the_card_shape(self) -> None:
        run = _read("run.sh")
        self.assertIn('--tool-call-parser "$TOOL_CALL_PARSER"', run)
        self.assertIn('--reasoning-parser "$REASONING_PARSER"', run)
        self.assertIn("qwen3_xml", run)
        self.assertIn("enable_thinking", run)


class OverlayTests(unittest.TestCase):
    """R11 headers, generators and the data-driven mount list (plan 0-1, R11, C1)."""

    def test_pin_overlays_carry_r11_header(self) -> None:
        for rel, target in PIN_OVERLAYS.items():
            head = _header(_read(rel))
            self.assertEqual(head["base_image_digest"], PIN_DIGEST, rel)
            self.assertEqual(head["upstream_file"], target, rel)
            self.assertRegex(head["upstream_file_sha256"], r"^[0-9a-f]{64}$", rel)
            self.assertTrue(head["upstream_PR"], rel)
            self.assertTrue(head["generator"].startswith("docker/apply_"), rel)

    def test_default_overlay_list_is_the_pin_set(self) -> None:
        run = _read("run.sh")
        m = re.search(r'^OVERLAYS_PIN="\$\{OVERLAYS_PIN:-(.*)\}"$', run, re.M)
        self.assertEqual(m.group(1).split(), list(PIN_OVERLAYS))
        proc = _run_sh(IMAGE=PINNED_IMAGE)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for rel, target in PIN_OVERLAYS.items():
            self.assertIn(f"{ROOT / rel} -> /usr/local/lib/python3.12/dist-packages/{target}", proc.stdout)

    def test_legacy_sha_list_matches_pin_overlay_bodies(self) -> None:
        run = _read("run.sh")
        listed = re.search(r'^LEGACY_PIN_OVERLAY_SHA256="([0-9a-f ]+)"$', run, re.M).group(1).split()
        bodies = [hashlib.sha256(_body(_read(rel)).encode()).hexdigest() for rel in PIN_OVERLAYS]
        self.assertEqual(sorted(listed), sorted(bodies))
        self.assertIn(f'NIGHTLY_PIN_DIGEST="{PIN_DIGEST}"', run)
        self.assertIn(f'V030_DIGEST="{V030_IMAGE.rsplit("@", 1)[1]}"', run)

    def test_ple_layer_body_is_stock(self) -> None:
        text = _read("docker/ple_layer.py")
        body = _body(text).encode()
        self.assertEqual(hashlib.sha256(body).hexdigest(), _header(text)["upstream_file_sha256"])

    def test_ple_ops_is_stock_plus_only_the_55375_stride_hunks(self) -> None:
        gen = _load("docker/apply_ple_stride_overlay.py")
        text = _read("docker/ple_ops.py")
        head, body = _header(text), _body(text)
        self.assertEqual(head["upstream_file_sha256"], gen.STOCK_SHA256)
        self.assertIn("55375", head["upstream_PR"])
        self.assertEqual(body.count("state_idx_ptr + r * state_idx_stride"), 2)
        self.assertNotIn("state_idx_ptr + r)", body)
        self.assertEqual(body.count("state_idx_stride = state_indices.stride(0)"), 1)
        self.assertNotIn("outer_residual", body)
        stock = body
        for old, new, _count in reversed(gen.HUNKS):
            stock = stock.replace(new, old)
        self.assertEqual(hashlib.sha256(stock.encode()).hexdigest(), gen.STOCK_SHA256)

    def test_generators_reproduce_committed_overlays_from_src(self) -> None:
        mtp = _load("docker/apply_mtp_fp8_overlay.py")
        stride = _load("docker/apply_ple_stride_overlay.py")
        cases = {
            "docker/apply_ple_overlay.py": ("docker/ple_layer.py", lambda b: b),
            "docker/apply_mtp_fp8_overlay.py": (
                "docker/modelopt.py",
                lambda b: b.replace(mtp.NEW_PREFIX, mtp.OLD_PREFIX).replace(mtp.NEW_MOE, mtp.OLD_MOE),
            ),
            "docker/apply_ple_stride_overlay.py": (
                "docker/ple_ops.py",
                lambda b: _unapply(b, stride.HUNKS),
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for gen, (rel, unpatch) in cases.items():
                committed = _read(rel)
                stock = Path(tmp) / "stock.py"
                stock.write_text(unpatch(_body(committed)))
                self.assertEqual(
                    hashlib.sha256(stock.read_bytes()).hexdigest(), _header(committed)["upstream_file_sha256"], rel
                )
                out = Path(tmp) / "out.py"
                subprocess.run(
                    ["python3", str(ROOT / gen), "--src", str(stock), "--out", str(out)],
                    check=True, capture_output=True, text=True,
                )
                self.assertEqual(out.read_text(), committed, f"{gen} does not reproduce {rel}")

    def test_stride_generator_refuses_a_foreign_stock_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stock = Path(tmp) / "ple.py"
            stock.write_text("x = 1\n")
            proc = subprocess.run(
                ["python3", str(ROOT / "docker/apply_ple_stride_overlay.py"), "--src", str(stock),
                 "--out", str(Path(tmp) / "out.py")],
                check=False, capture_output=True, text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("sha256", proc.stderr)

    def test_refuses_missing_overlay(self) -> None:
        proc = _run_sh(OVERLAYS="docker/ple_layer.py docker/modelopt.py /nonexistent/ple_ops.py", IMAGE=PINNED_IMAGE)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Overlay missing", proc.stderr)

    def test_refuses_pin_without_stride_overlay(self) -> None:
        proc = _run_sh(OVERLAYS="docker/ple_layer.py docker/modelopt.py", IMAGE=PINNED_IMAGE)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("vllm/models/qwen4_exp/nvidia/ops/ple.py", proc.stderr)
        proc = _run_sh(OVERLAYS="docker/ple_layer.py docker/modelopt.py", DIAGNOSTIC="1", IMAGE=PINNED_IMAGE)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_refuses_overlay_without_header(self) -> None:
        proc = _run_sh(OVERLAYS="docker/ple_layer.py docker/modelopt.py docker/ple_ops.py docker/apply_ple_overlay.py",
                       IMAGE=PINNED_IMAGE)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("no R11 header", proc.stderr)

    def test_refuses_digest_mismatch(self) -> None:
        proc = _run_sh(IMAGE=V030_IMAGE, OVERLAYS="docker/modelopt.py")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("was generated for", proc.stderr)

    def test_refuses_pin_overlay_body_on_v030_even_with_edited_digest(self) -> None:
        v030 = V030_IMAGE.rsplit("@", 1)[1]
        with tempfile.TemporaryDirectory() as tmp:
            for rel in PIN_OVERLAYS:
                forged = Path(tmp) / Path(rel).name
                forged.write_text(_read(rel).replace(PIN_DIGEST, v030, 1))
                proc = _run_sh(IMAGE=V030_IMAGE, OVERLAYS=str(forged))
                self.assertNotEqual(proc.returncode, 0, rel)
                self.assertIn("pin-only overlay", proc.stderr, rel)

    def test_refuses_duplicate_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "ple_ops_copy.py"
            copy.write_text(_read("docker/ple_ops.py"))
            proc = _run_sh(OVERLAYS=f"docker/ple_layer.py docker/modelopt.py docker/ple_ops.py {copy}", IMAGE=PINNED_IMAGE)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Two overlays target", proc.stderr)

    def test_v030_auto_uses_its_own_list(self) -> None:
        run = _read("run.sh")
        m = re.search(r'^OVERLAYS_V030="\$\{OVERLAYS_V030:-(.*)\}"$', run, re.M)
        self.assertEqual(m.group(1).split(), list(V030_OVERLAYS))
        proc = _run_sh(GDN_LAZY="0")  # K3 (GDN_LAZY=1, the default) swaps gdn_attn.py: tests/test_gdn_lazy.py
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for rel, target in V030_OVERLAYS.items():
            self.assertEqual(_header(_read(rel))["base_image_digest"], V030_IMAGE.rsplit("@", 1)[1], rel)
            self.assertIn(f"{ROOT / rel} -> /usr/local/lib/python3.12/dist-packages/{target}", proc.stdout)
        proc = _run_sh(IMAGE=V030_IMAGE, OVERLAYS="none")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("overlay none", proc.stdout)
        proc = _run_sh(OVERLAYS="none", DIAGNOSTIC="1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("overlay none", proc.stdout)

    def test_overlays_mount_on_both_ranks(self) -> None:
        run = _read("run.sh")
        start = _func_body(run, "start_local")
        self.assertIn('vol_args+=(-v "${OVERLAY_FILES[$i]}:${SITE_PACKAGES}/${OVERLAY_TARGETS[$i]}:ro")', start)
        self.assertIn("/tmp/qwen38-overlays/", run)

    def test_draft_moe_config_mounts_on_v030_only(self) -> None:
        run = _read("run.sh")
        start = _func_body(run, "start_local")
        self.assertIn('vol_args+=(-v "${DRAFT_MOE_CONFIG_DIR}:${DRAFT_MOE_CONFIG_MOUNT}:ro")', start)
        self.assertIn('env_args+=(-e "VLLM_TUNED_CONFIG_FOLDER=$DRAFT_MOE_CONFIG_MOUNT")', start)
        self.assertIn("DRAFT_MOE_CONFIG_DIR=/tmp/qwen38-moe-configs", _func_body(run, "worker_env"))
        forwarded = re.search(r"^FORWARD_VARS=\((.*?)\)", run, re.M | re.S).group(1).split()
        self.assertIn("DRAFT_MOE_CONFIG", forwarded)
        proc = _run_sh(IMAGE=V030_IMAGE, DRAFT_MOE_CONFIG="1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("draft-moe-config", proc.stdout)
        proc = _run_sh(IMAGE=V030_IMAGE, DRAFT_MOE_CONFIG="1", DRAFT_MOE_CONFIG_DIR="/nonexistent")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("draft-moe-config", proc.stdout)  # failing check falls closed to stock
        self.assertIn("WARNING: DRAFT_MOE_CONFIG=1", proc.stderr)
        proc = _run_sh(IMAGE=PINNED_IMAGE, DRAFT_MOE_CONFIG="1")  # the pin rollback ignores it
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("draft-moe-config", proc.stdout)
        self.assertNotEqual(_run_sh(DRAFT_MOE_CONFIG="2").returncode, 0)

    def test_fp8_dense_is_default(self) -> None:
        # Default since session 8 (evidence/fp8-default); FP8_DENSE=none is the BF16 rollback.
        self.assertIn('FP8_DENSE="${FP8_DENSE:-per_block}"', _read("run.sh"))
        self.assertIn("FP8_DENSE: per_block", _read("recipe.yaml"))
        for extra in ({}, {"FP8_DENSE": "per_block"}):
            proc = _run_sh(IMAGE=V030_IMAGE, **extra)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("docker/v030/modelopt.py -> ", proc.stdout)
            self.assertIn("VLLM_QWEN38_FP8_DENSE=per_block", proc.stdout)
            self.assertIn('--kernel-config {"linear_backend_per_quant":{"fp8_block_w8a8":"marlin"}}', proc.stdout)
        proc = _run_sh(IMAGE=V030_IMAGE, FP8_DENSE="none")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("quantization/modelopt.py", proc.stdout)
        self.assertNotIn("VLLM_QWEN38_FP8_DENSE", proc.stdout)
        self.assertNotIn("--kernel-config", proc.stdout)
        # The worker gets the head's resolved overlay list and must not add the overlay a second time.
        proc = _run_sh(IMAGE=V030_IMAGE, FP8_DENSE="per_block", ROLE="worker",
                       OVERLAYS="docker/v030/modelopt.py")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("Two overlays target", proc.stderr)
        # The pin rollback and OVERLAYS=none ignore it (the overlay is generated against v0.30).
        for extra in ({"IMAGE": PINNED_IMAGE}, {"IMAGE": V030_IMAGE, "OVERLAYS": "none"}):
            proc = _run_sh(**extra)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("FP8_DENSE=per_block ignored", proc.stderr)
            self.assertNotIn("VLLM_QWEN38_FP8_DENSE", proc.stdout)
            self.assertNotIn("v030/modelopt.py", proc.stdout)
        self.assertNotEqual(_run_sh(IMAGE=V030_IMAGE, FP8_DENSE="ptpc").returncode, 0)
        self.assertRegex(_read("run.sh"), r"FORWARD_VARS=\([^)]*\bFP8_DENSE\b")

    def test_draft_head_knobs(self) -> None:
        mtp = " ".join(V030_OVERLAYS)
        base = " ".join(k for k in V030_OVERLAYS if not k.endswith("mtp.py"))
        with tempfile.TemporaryDirectory() as hf:
            os.makedirs(f"{hf}/qwen38-draft-vocab")
            Path(f"{hf}/qwen38-draft-vocab/draft_vocab_163840.json").write_text("[]")
            on = dict(IMAGE=V030_IMAGE, OVERLAYS_V030=mtp, HF_CACHE=hf, DRAFT_LOCAL_ARGMAX="1",
                      DRAFT_HEAD_FP8="1", DRAFT_VOCAB="draft_vocab_163840.json")
            proc = _run_sh(**on)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn('"use_local_argmax_reduction":true}', proc.stdout)
            self.assertIn("VLLM_QWEN38_DRAFT_HEAD_FP8=1", proc.stdout)
            self.assertIn("VLLM_QWEN38_DRAFT_VOCAB=/cache/huggingface/qwen38-draft-vocab/draft_vocab_163840.json",
                          proc.stdout)
            # the knobs are read only by the mtp.py overlay: ignored (stock drafter) without it
            for extra in ({"OVERLAYS_V030": base}, {"OVERLAYS": "none"}):
                proc = _run_sh(**{**on, **extra})
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn("ignored", proc.stderr)
                self.assertNotIn("use_local_argmax_reduction", proc.stdout)
                self.assertNotIn("VLLM_QWEN38_DRAFT", proc.stdout)
            # an explicit SPEC_CONFIG wins over DRAFT_LOCAL_ARGMAX
            proc = _run_sh(**{**on, "DRAFT_VOCAB": "none",
                              "SPEC_CONFIG": '{"method":"mtp","num_speculative_tokens":3}'})
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertNotIn("use_local_argmax_reduction", proc.stdout)
            # a reduced vocab needs local argmax
            proc = _run_sh(**{**on, "DRAFT_LOCAL_ARGMAX": "0"})
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("use_local_argmax_reduction", proc.stderr)
            proc = _run_sh(**{**on, "DRAFT_VOCAB": "draft_vocab_131072.json"})
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("missing", proc.stderr)
            self.assertNotEqual(_run_sh(**{**on, "DRAFT_VOCAB": "../x.json"}).returncode, 0)
            # the pin rollback ignores all three
            proc = _run_sh(**{**on, "IMAGE": PINNED_IMAGE})
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertNotIn("use_local_argmax_reduction", proc.stdout)
            self.assertNotIn("VLLM_QWEN38_DRAFT", proc.stdout)


def _unapply(body: str, hunks) -> str:
    for old, new, _count in reversed(hunks):
        body = body.replace(new, old)
    return body


class GuardTests(unittest.TestCase):
    """Plan 0-2 / 0-3 guards: each refuses with its true reason, before VALIDATE_ONLY exits."""

    def refused(self, needle: str, **env: str) -> None:
        proc = _run_sh(**env)
        self.assertNotEqual(proc.returncode, 0, f"{env} was accepted")
        self.assertIn(needle, proc.stderr, env)

    def accepted(self, **env: str) -> subprocess.CompletedProcess[str]:
        proc = _run_sh(**env)
        self.assertEqual(proc.returncode, 0, f"{env}: {proc.stderr}")
        return proc

    def test_integers_are_decimal(self) -> None:
        self.refused("not a positive decimal integer", MAX_NUM_SEQS="010")
        self.refused("not a positive decimal integer", MAX_NUM_SEQS="8x")
        self.refused("not a positive decimal integer", MAX_MODEL_LEN="2097152x")
        self.refused("not a positive decimal integer", NUM_SPECULATIVE_TOKENS="abc")
        self.refused("not empty or a positive decimal integer", MAX_NUM_BATCHED_TOKENS="08192")
        self.refused("must be 0 or 1", FORCE_UNSAFE_CTX="yes")
        self.refused("must be a decimal", UTIL="1.5")

    def test_json_configs_are_parsed(self) -> None:
        self.refused("SPEC_CONFIG is not valid JSON", SPEC_CONFIG="{'method':'mtp'}")
        self.refused("COMPILATION_CONFIG must be a JSON object", COMPILATION_CONFIG="[0]")

    def test_image_must_be_digest_pinned(self) -> None:
        self.refused("not digest-pinned", IMAGE="vllm/vllm-openai:nightly-aarch64")

    def test_spec_none_only_with_diagnostic(self) -> None:
        self.refused("DIAGNOSTIC=1", SPEC="none")
        proc = self.accepted(SPEC="none", DIAGNOSTIC="1")
        self.assertIn("diagnostic=1", proc.stdout)

    def test_moe_backend_must_be_auto_on_pin(self) -> None:
        for backend in ("marlin", "flashinfer_cutlass", "triton", "cutlass", "humming"):
            self.refused("only Triton accepts", MOE_BACKEND=backend, IMAGE=PINNED_IMAGE)
        self.accepted(MOE_BACKEND="marlin", FORCE_UNSAFE_MOE="1", IMAGE=PINNED_IMAGE)

    def test_b12x_refused_everywhere(self) -> None:
        for backend in ("b12x", "flashinfer_b12x"):
            self.refused("#57946", MOE_BACKEND=backend)
            self.refused("#57946", MOE_BACKEND=backend, IMAGE=V030_IMAGE, OVERLAYS="none")

    def test_other_backends_allowed_off_pin(self) -> None:
        self.accepted(MOE_BACKEND="marlin", IMAGE=V030_IMAGE, OVERLAYS="none")

    def test_compile_mode_zero_on_pin(self) -> None:
        pin = {"IMAGE": PINNED_IMAGE}
        self.refused("#55272", COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}', **pin)
        self.refused("#55272", COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY"}', **pin)
        self.refused("#55272", COMPILATION_CONFIG='{"mode":null}', **pin)
        self.accepted(COMPILATION_CONFIG='{"mode":"NONE"}', **pin)
        self.accepted(COMPILATION_CONFIG='{"mode":3}', ENFORCE_EAGER="1", **pin)
        self.accepted(COMPILATION_CONFIG='{"mode":3}', IMAGE=V030_IMAGE, OVERLAYS="none")

    def test_local_argmax_needs_get_top_tokens_overlay(self) -> None:
        spec = '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}'
        no_mtp = " ".join(k for k in V030_OVERLAYS if not k.endswith("mtp.py"))
        self.refused("get_top_tokens", SPEC_CONFIG=spec, OVERLAYS=no_mtp)
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "mtp.py"
            fake.write_text(
                f"# R11-OVERLAY\n# base_image_digest: {PIN_DIGEST}\n"
                "# upstream_file: vllm/models/qwen4_exp/nvidia/mtp.py\n\n"
                "def get_top_tokens(self, h):\n    pass\n"
            )
            overlays = " ".join(PIN_OVERLAYS) + f" {fake}"
            self.accepted(SPEC_CONFIG=spec, OVERLAYS=overlays, IMAGE=PINNED_IMAGE)
            probabilistic = spec[:-1] + ',"draft_sample_method":"probabilistic"}'
            self.refused("probabilistic", SPEC_CONFIG=probabilistic, OVERLAYS=overlays, IMAGE=PINNED_IMAGE)

    def test_extra_args_cannot_reset_guarded_flags(self) -> None:
        for args in ("--compilation-config {\"mode\":3}", "-O3", "-cc.mode=3", "--moe-backend marlin",
                     "--speculative-config.num_speculative_tokens=5", "--max-num-batched-tokens=4096",
                     "--max-num-seqs 16", "--max-model-len 2097152", "--enforce-eager",
                     "--long-prefill-token-threshold 2048"):
            self.refused("which run.sh passes and guards itself", EXTRA_ARGS=args)
        self.accepted(EXTRA_ARGS="--enable-log-requests")

    def test_adaptive_verification_refused(self) -> None:
        self.refused("UNIFORM_BATCH", SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":3,"enable_adaptive_verification":true}')

    def test_batch_sharded_sampling_needs_overlay(self) -> None:
        self.refused("compute_logits_local", EXTRA_ARGS="--enable-batch-sharded-sampling")

    def test_batch_invariant_refused(self) -> None:
        self.refused("supports_batch_invariance", EXTRA_ENV="VLLM_BATCH_INVARIANT=1")

    def test_synthetic_only_bench_only_on_loopback(self) -> None:
        spec = '{"method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"synthetic","synthetic_acceptance_length":2.46}'
        self.refused("BENCH_ONLY=1", SPEC_CONFIG=spec)
        proc = self.accepted(SPEC_CONFIG=spec, BENCH_ONLY="1")
        self.assertIn("host=127.0.0.1", proc.stdout)
        self.assertIn("host=0.0.0.0", self.accepted().stdout)
        self.assertIn('--host "$API_HOST"', _read("run.sh"))

    def test_indexer_knobs_need_logits_cap(self) -> None:
        off = {"LONG_PREFILL_TOKEN_THRESHOLD": "none"}
        self.refused("#56457", VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="none", MAX_NUM_BATCHED_TOKENS="4096", **off)
        self.refused("#56457", VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="none")  # the default threshold 4800
        self.refused("#56457", VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="none",
                     EXTRA_ARGS='--attention-config {"indexer_kv_dtype":"fp8"}', **off)
        self.accepted(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="none", **off)
        self.refused("LONG_PREFILL_TOKEN_THRESHOLD=04800", LONG_PREFILL_TOKEN_THRESHOLD="04800")
        self.assertIn("long-prefill-token-threshold=4800", self.accepted().stdout)
        self.assertIn('--long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD"', _func_body(_read("run.sh"), "start_local"))
        self.accepted(MAX_NUM_BATCHED_TOKENS="4096")

    def test_logits_cap_default_reaches_both_ranks(self) -> None:
        run = _read("run.sh")
        self.assertIn('VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-64}"', run)
        self.assertIn('-e "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=$VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"', _func_body(run, "start_local"))

    def test_extra_env_passthrough(self) -> None:
        self.refused("is not KEY=VALUE", EXTRA_ENV="NCCL_DEBUG")
        proc = self.accepted(EXTRA_ENV="NCCL_DEBUG=INFO VLLM_COMPUTE_NANS_IN_LOGITS=1")
        self.assertIn("extra-env NCCL_DEBUG=INFO", proc.stdout)
        start = _func_body(_read("run.sh"), "start_local")
        self.assertLess(start.find('-e "NCCL_DEBUG=WARN"'), start.find('env_args+=("${EXTRA_ENV_ARGS[@]}")'))

    def test_v030_base_env_on_both_ranks_not_via_extra_env(self) -> None:
        proc = self.accepted()
        self.assertIn("base-env VLLM_PLE_CPU_OFFLOAD=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0", proc.stdout)
        self.assertNotIn("base-env", self.accepted(IMAGE=PINNED_IMAGE).stdout)
        run = _read("run.sh")
        self.assertIn('[[ "$IMAGE_DIGEST" == "$V030_DIGEST" ]] && BASE_ENV=(VLLM_PLE_CPU_OFFLOAD=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0)', run)
        start = _func_body(run, "start_local")
        # Before EXTRA_ENV, so a diagnostic EXTRA_ENV can still override (later -e wins).
        self.assertLess(start.find('env_args+=(-e "$kv")'), start.find('env_args+=("${EXTRA_ENV_ARGS[@]}")'))
        self.assertNotIn("per-request-spec-decode-metrics", run)

    def test_serve_flags_async_and_vision_caps(self) -> None:
        proc = self.accepted()
        self.assertIn('mm-processor-kwargs={"images_kwargs":{"min_pixels":65536,"max_pixels":4194304},'
                      '"videos_kwargs":{"cap_pixels_per_frame":true}}', proc.stdout)
        self.assertIn('limit-mm-per-prompt={"image":8,"video":1}', proc.stdout)
        self.assertIn("async-scheduling=1", proc.stdout)
        self.assertIn("mm-processor-cache-gb=1", proc.stdout)
        start = _func_body(_read("run.sh"), "start_local")
        for flag in ('"${async_args[@]}"', '--mm-processor-kwargs "$MM_PROCESSOR_KWARGS"',
                     '--limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT"', '--mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB"'):
            self.assertIn(flag, start)
        self.refused("exceeds MM_MAX_PIXELS", MM_MIN_PIXELS="5000000")
        self.refused("MM_LIMIT_IMAGE=08", MM_LIMIT_IMAGE="08")
        self.refused("ASYNC_SCHEDULING=2", ASYNC_SCHEDULING="2")
        self.refused("MM_PROCESSOR_CACHE_GB=x", MM_PROCESSOR_CACHE_GB="x")
        for args in ("--async-scheduling", "--no-async-scheduling", "--limit-mm-per-prompt {}",
                     "--mm-processor-kwargs {}", "--mm-processor-cache-gb 4", "--chat-template x.jinja"):
            self.refused("which run.sh passes itself", EXTRA_ARGS=args)

    def test_chat_template_head_only(self) -> None:
        proc = self.accepted()
        self.assertIn(f"chat-template {ROOT / 'chat_template_alias.jinja'}", proc.stdout)
        self.assertIn("chat-template none", self.accepted(CHAT_TEMPLATE="none").stdout)
        self.refused("not found", CHAT_TEMPLATE="no-such-template.jinja")
        run = _read("run.sh")
        start = _func_body(run, "start_local")
        head = start[start.find('if [[ "$rank" == "0" ]]; then'):start.find("--headless")]
        self.assertIn('--chat-template "$CHAT_TEMPLATE_IN_CONTAINER"', head)
        self.assertIn(':${CHAT_TEMPLATE_IN_CONTAINER}:ro"', head)
        self.assertIn("CHAT_TEMPLATE=none", _func_body(run, "worker_env"))

    def test_validate_only_exits_after_every_guard(self) -> None:
        run = _read("run.sh")
        exit_at = run.find('if [[ "${VALIDATE_ONLY:-0}" == "1" ]]')
        for needle in ("Overlay missing", "no R11 header", "was generated for", "pin-only overlay",
                       "#55272", "#57946", "get_top_tokens", "BENCH_ONLY=1", "#56457"):
            self.assertLess(run.find(needle), exit_at, needle)


class RunOpsTests(unittest.TestCase):
    """Plan 0-2 plumbing that runs only on a real boot, checked statically or in isolation."""

    def snapshot_complete(self, snap: Path) -> subprocess.CompletedProcess[str]:
        script = _func_src(_read("run.sh"), "snapshot_complete") + f"SNAPSHOT={shlex.quote(str(snap))}\nsnapshot_complete\n"
        return subprocess.run(["bash", "-c", script], check=False, capture_output=True, text=True)

    def test_snapshot_complete_checks_index_shards_and_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp)
            for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "a.safetensors"):
                (snap / name).write_text("x")
            (snap / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"w1": "a.safetensors", "w2": "b.safetensors"}})
            )
            proc = self.snapshot_complete(snap)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("b.safetensors", proc.stderr)
            (snap / "b.safetensors").write_text("")
            self.assertNotEqual(self.snapshot_complete(snap).returncode, 0)
            (snap / "b.safetensors").write_text("x")
            self.assertEqual(self.snapshot_complete(snap).returncode, 0)
            (snap / "tokenizer.json").unlink()
            proc = self.snapshot_complete(snap)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("tokenizer.json", proc.stderr)

    def test_no_hf_token_in_container_and_offline(self) -> None:
        run = _read("run.sh")
        self.assertNotIn("HF_TOKEN", run)
        self.assertIn('-e "HF_HUB_OFFLINE=1"', _func_body(run, "start_local"))
        self.assertIn('HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"', run)

    def test_drop_caches_after_stop_and_logged(self) -> None:
        run = _read("run.sh")
        start = _func_body(run, "start_local")
        self.assertLess(start.find("stop_local"), start.find("maybe_drop_caches"))
        drop = _func_body(run, "maybe_drop_caches")
        self.assertIn("drop_caches: ran", drop)
        self.assertIn("drop_caches: skipped", drop)

    def test_run_state_under_script_dir(self) -> None:
        for rel in ("run.sh", "stop.sh"):
            text = _read(rel)
            self.assertNotIn("${PWD}/.run-state", text, rel)
            self.assertIn('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"', text, rel)
        self.assertIn('STATE_DIR="$SCRIPT_DIR/.run-state"', _read("run.sh"))

    def test_wait_ready_watches_worker(self) -> None:
        run = _read("run.sh")
        wait = _func_body(run, "wait_ready")
        self.assertIn("worker_state", wait)
        self.assertIn("abort_worker_dead", wait)
        self.assertIn("i % 2 == 0", wait)  # every ~10 s + 5 s ssh timeout, well inside 60 s
        self.assertIn('wait_ready "$watch_worker"', run)
        self.assertIn("ConnectTimeout=5", _func_body(run, "worker_state"))

    def test_worker_state_reads_a_missing_container(self) -> None:
        # Real docker prints an empty line before failing on a missing container (seen on spark2).
        fn = _func_src(_read("run.sh"), "worker_state")
        for ssh_out, rc, want in (("\\nmissing\\n", 0, "missing"), ("false\\n", 0, "false"), ("", 255, "")):
            with tempfile.TemporaryDirectory() as tmp:
                fake = Path(tmp) / "ssh"
                fake.write_text(f"#!/bin/sh\nprintf '{ssh_out}'\nexit {rc}\n")
                fake.chmod(0o755)
                script = f"set -euo pipefail\n{fn}WORKER_HOST=w CONTAINER_NAME=c\nprintf '[%s]' \"$(worker_state)\"\n"
                env = dict(os.environ, PATH=f"{tmp}:{os.environ['PATH']}")
                out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, check=True)
                self.assertEqual(out.stdout, f"[{want}]")


class DocsTests(unittest.TestCase):
    """Plan 0-11 ledger and doc corrections."""

    def test_withdrawn_and_stale_claims_are_gone(self) -> None:
        for rel in ("README.md", "AGENTS.md", "run.sh", "recipe.yaml"):
            text = _read(rel)
            self.assertNotIn("held eight short streams with no drop", text, rel)
            self.assertNotIn("walks flashinfer then marlin", text, rel)
        self.assertIn("max_tokens 200 (prose ≈93 at EOS)", _read("README.md"))
        # Structured c=2 is back only because the table is re-measured on a base with F01 fixed.
        self.assertIn("v0.30.0 + determinism overlays", _read("README.md"))
        self.assertNotIn("measured before the F01 PLE stride fix", _read("README.md"))
        self.assertIn("Pre-resize large images", _read("README.md"))

    def test_decision_ledger_marks_c1_rationale_as_f01(self) -> None:
        rows = [line.split("\t") for line in _read("evidence/decision.tsv").splitlines()]
        width = len(rows[0])
        self.assertTrue(all(len(r) == width for r in rows))
        note = [r for r in rows if r[0] == "note-f01-ledger"]
        self.assertEqual(len(note), 1)
        self.assertIn("F01 artefact", note[0][-1])
        self.assertIn("c5 revert stands", note[0][-1])

    def test_overlays_doc_exists(self) -> None:
        doc = _read("docker/OVERLAYS.md")
        for key in HEADER_KEYS[:-1]:
            self.assertIn(key, doc)


if __name__ == "__main__":
    unittest.main()
