#!/usr/bin/env bash
# U1: vLLM v0.30.0 by digest, OVERLAYS=auto (OVERLAYS_V030 empty -> none), run.sh defaults at 6aefbfc (UTIL 0.76, memguard)
# Both envs verified in v0.30 vllm/envs.py (VLLM_PLE_CPU_OFFLOAD default 1, VLLM_USE_BREAKABLE_CUDAGRAPH default 0).
ssh spark2 docker pull vllm/vllm-openai@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56   # RepoDigest (done at S1.1)
IMAGE=vllm/vllm-openai:v0.30.0-aarch64@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56 \
  EXTRA_ENV="VLLM_PLE_CPU_OFFLOAD=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0" \
  EXTRA_ARGS='--per-request-spec-decode-metrics summary' ./run.sh > evidence/u1-v030/run.log 2>&1 &
evidence/u1-v030/steps.sh u1-v030 u1 gate diverse sampled qaa t2 t3 prefix receipts
python3 quality/score.py t1 /home/sfxnz/projects/data/qwen38-evals/runs/session1/b1-pin-stride/t1-a /home/sfxnz/projects/data/qwen38-evals/runs/session1/u1-v030/t1-a --aa evidence/b1-pin-stride/t1-aa.json --json evidence/u1-v030/t1a-u1-vs-b1.json
python3 quality/score.py t1g /home/sfxnz/projects/data/qwen38-evals/runs/session1/b1-pin-stride/t1g /home/sfxnz/projects/data/qwen38-evals/runs/session1/u1-v030/t1g --aa evidence/b1-pin-stride/t1g-self.json --json evidence/u1-v030/t1g-u1-vs-b1.json
python3 quality/score.py t2 /home/sfxnz/projects/data/qwen38-evals/runs/session1/b1-pin-stride/t2 /home/sfxnz/projects/data/qwen38-evals/runs/session1/u1-v030/t2 --json evidence/u1-v030/t2-u1-vs-b1.json
python3 ruler_steps.py --anchor-ms 0 0 --json evidence/u1-v030/ruler-noanchor.json
