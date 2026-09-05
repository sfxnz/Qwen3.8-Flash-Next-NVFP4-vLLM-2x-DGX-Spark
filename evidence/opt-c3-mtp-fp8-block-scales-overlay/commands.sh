#!/usr/bin/env bash
# c3 recapture: overlay already live from c1/c2. Do not reboot a health-200 pin.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/evidence/opt-c3-mtp-fp8-block-scales-overlay"
mkdir -p "$OUT"

python3 -m unittest discover -s tests -q
python3 kit/render.py --check
VALIDATE_ONLY=1 ./run.sh
