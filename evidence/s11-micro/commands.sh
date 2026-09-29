#!/usr/bin/env bash
# session1 S1.1 commands (2026-09-29T11:53:17Z)
./stop.sh
tools/micro/run_s11.sh --dry-run
OUT=$HOME/projects/data/qwen38-s11/session1 tools/micro/run_s11.sh
ssh spark2 docker pull vllm/vllm-openai@sha256:4864d466...  # RepoDigest for v0.30 on spark2
IMAGES=pin STEPS=moe OUT=$HOME/projects/data/qwen38-s11/session1 tools/micro/run_s11.sh   # rerun after moe_nvfp4.py scale-shape fix
