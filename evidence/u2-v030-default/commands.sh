#!/usr/bin/env bash
# Session 3 (2026-09-29): U2 = v0.30.0 + determinism overlays + serving.py as the recipe default, then P2-5.
# Run from the recipe worktree root (opt/plan-exec). boot.sh stops the serve, runs the memcheck
# (spark1 MemAvailable >= 6 GiB, swap used <= 10 GiB) and boots ./run.sh with the rendered defaults.
# Raw data: ~/projects/data/qwen38-evals/runs/session3/<subdir>/.

# --- pre: baselines on the live U1 + 3 determinism overlays serve (no serving.py, no vision caps)
python3 probe_toolstream.py --out evidence/u2-v030-default/pre/toolstream-u1det.jsonl --label u1det-no-serving
python3 bench_vision.py --out evidence/u2-v030-default/pre/vision-u1det.jsonl --label u1det-nocap

# --- Part A checks (no GPU)
python3 -m unittest discover -s tests -q
python3 kit/render.py --check
VALIDATE_ONLY=1 ./run.sh
IMAGE=vllm/vllm-openai:nightly-aarch64@sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b VALIDATE_ONLY=1 ./run.sh
# flags exist in v0.30 (CPU container, source grep; `vllm serve --help` needs a device)
docker run --rm --entrypoint bash vllm/vllm-openai:v0.30.0-aarch64@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56 \
  -c 'grep -nE "async_scheduling|mm_processor_cache_gb|mm_processor_kwargs|limit_mm_per_prompt|long_prefill_token_threshold" /usr/local/lib/python3.12/dist-packages/vllm/engine/arg_utils.py'

# --- Boot A: new default (first QSA overlay)
evidence/u2-v030-default/boot.sh a-default
evidence/u2-v030-default/steps.sh a-default alias toolstream
evidence/u2-v030-default/steps.sh a-default gate                      # smokes, T0, ruler --anchor-ms 0 0, frozen bench_decode
ANCHOR="61.4 63.4" evidence/u2-v030-default/steps.sh a-default ruler-anchor diverse sampled t1g t2 vision vbench prefix t3 mixed receipts
#   -> t3-65536: device assert (QSA overlay scatter OOB) + apport core dump -> memguard docker kill at 19:42:52 BST; mixed skipped
python3 quality/score.py t2 ~/projects/data/qwen38-evals/runs/session1/u1-v030/t2 ~/projects/data/qwen38-evals/runs/session3/a-default/t2 \
  --json evidence/u2-v030-default/a-default/t2-vs-u1.json

# --- Fix (no GPU serve): QSA tie-repair threshold from exact torch.topk + slot clamp; --ulimit core=1
python3 docker/v030/apply_qsa_topk_order_overlay.py
docker run --rm --ulimit core=1 -v "$PWD":/r:ro -w /r --entrypoint python3 \
  vllm/vllm-openai:v0.30.0-aarch64@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56 \
  -m unittest discover -s tests -p test_qsa_topk_order_overlay.py -v

# --- Boot A2: default with the fixed overlay
evidence/u2-v030-default/boot.sh a2-default
evidence/u2-v030-default/steps.sh a2-default t3                       # 4k..128k, one length at a time, memcheck before each
ANCHOR="61.4 63.4" evidence/u2-v030-default/steps.sh a2-default gate ruler-anchor t1g vision vbench mixed receipts
#   -> ruler-anchor VOID at [61.4, 63.4] (sentinels 60.9-61.4); kept as ruler-anchor-61.4-63.4-void.*
ANCHOR="60.7 62.7" evidence/u2-v030-default/steps.sh a2-default ruler-anchor

# --- Boot B: P2-5 arm
evidence/u2-v030-default/boot.sh b-lpt4800 'EXTRA_ARGS=--long-prefill-token-threshold 4800'
ANCHOR="60.7 62.7" evidence/u2-v030-default/steps.sh b-lpt4800 mixed ruler-anchor receipts

# --- keep P2-5: recipe.yaml LONG_PREFILL_TOKEN_THRESHOLD 4800; render; tests
python3 kit/render.py && python3 kit/render.py --check && python3 -m unittest discover -s tests -q

# --- Boot C: final rendered default, left serving
evidence/u2-v030-default/boot.sh c-final
evidence/u2-v030-default/steps.sh c-final gate alias receipts
