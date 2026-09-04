#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
./stop.sh
FORCE_UNSAFE_CTX=1 MAX_NUM_SEQS=8 ./run.sh
python3 smoke_thinking.py
python3 smoke_tools.py
python3 smoke_vision.py
python3 smoke_count.py
python3 bench_decode.py --runs 3 --concurrency 1 2 4 8 --max-tokens 200
