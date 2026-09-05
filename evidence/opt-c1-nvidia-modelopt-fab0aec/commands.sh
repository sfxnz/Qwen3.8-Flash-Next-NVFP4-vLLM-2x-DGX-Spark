#!/usr/bin/env bash
# c1 NVIDIA ModelOpt pin: day-0 boot, then nightly if MTP load fails.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/evidence/opt-c1-nvidia-modelopt-fab0aec"
mkdir -p "$OUT"
MODEL_ID="nvidia/Qwen3.8-Flash-Next-NVFP4"
SNAPSHOT_SHA="fab0aecb760cec45227f6656abcaafa11abca87a"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
SNAP="${HF_CACHE}/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/${SNAPSHOT_SHA}"
NIGHTLY="vllm/vllm-openai:nightly-aarch64@sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b"

python3 kit/render.py
python3 -m unittest discover -s tests -q
python3 kit/render.py --check
VALIDATE_ONLY=1 ./run.sh

test -f "$SNAP/config.json"
ssh -o BatchMode=yes -o ConnectTimeout=5 spark2 "test -f '$SNAP/config.json'"

./stop.sh
./run.sh
