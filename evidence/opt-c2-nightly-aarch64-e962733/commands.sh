#!/usr/bin/env bash
# c2 official nightly-aarch64 e962733: drop VLLM_PLE_FP8_CHECKPOINT, reboot, freeze bench.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/evidence/opt-c2-nightly-aarch64-e962733"
mkdir -p "$OUT"
IMAGE='vllm/vllm-openai:nightly-aarch64@sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b'

python3 kit/render.py
python3 -m unittest discover -s tests -q
python3 kit/render.py --check
VALIDATE_ONLY=1 ./run.sh
VALIDATE_ONLY=1 VLLM_PLE_FP8_CHECKPOINT=0 ./run.sh

docker image inspect "$IMAGE" >/dev/null
ssh -o BatchMode=yes -o ConnectTimeout=5 spark2 "docker image inspect '$IMAGE' >/dev/null"

./stop.sh
./run.sh
