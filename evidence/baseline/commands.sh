#!/usr/bin/env bash
# Frozen correctness + decode baseline against the live seqs=2 serve.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/evidence/baseline"
python3 smoke_thinking.py >"$OUT/thinking-off.json"
python3 smoke_tools.py >"$OUT/tools.json"
python3 smoke_vision.py >"$OUT/vision.json"
python3 smoke_count.py >"$OUT/count.json"
python3 bench_decode.py --runs 3 --concurrency 1 2 --max-tokens 200 | tee "$OUT/bench.txt"
free -h >"$OUT/free-after.txt"
ssh spark2 'free -h' >"$OUT/free-after-spark2.txt"
