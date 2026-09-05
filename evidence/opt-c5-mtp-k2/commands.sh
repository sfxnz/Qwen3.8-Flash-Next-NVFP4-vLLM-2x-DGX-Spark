#!/usr/bin/env bash
# c5 MTP k=2: one knob, render, reboot, freeze bench.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/evidence/opt-c5-mtp-k2"
mkdir -p "$OUT"

python3 kit/render.py
python3 -m unittest discover -s tests -q
python3 kit/render.py --check
VALIDATE_ONLY=1 ./run.sh | tee "$OUT/validate-only.txt"
grep -q 'spec_tokens=2' "$OUT/validate-only.txt"

./stop.sh | tee "$OUT/stop-before-reboot.txt"
./run.sh
