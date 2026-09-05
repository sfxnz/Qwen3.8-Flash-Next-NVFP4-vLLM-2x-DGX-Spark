#!/usr/bin/env bash
# c4 native 262144: drop ALLOW_LONG, reboot, freeze bench.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/evidence/opt-c4-native-ctx-262144"
mkdir -p "$OUT"

python3 kit/render.py
python3 -m unittest discover -s tests -q
python3 kit/render.py --check
VALIDATE_ONLY=1 ./run.sh
set +e
VALIDATE_ONLY=1 MAX_MODEL_LEN=1048576 VLLM_ALLOW_LONG_MAX_MODEL_LEN=0 ./run.sh >"$OUT/validate-1m-without-allow.txt" 2>&1
rc=$?
set -e
if [[ "$rc" -eq 0 ]]; then
  echo "expected 1M without ALLOW to fail" >&2
  exit 1
fi
grep -q VLLM_ALLOW_LONG_MAX_MODEL_LEN "$OUT/validate-1m-without-allow.txt"

./stop.sh
./run.sh
