#!/usr/bin/env bash
# Commands used to boot this pin. Rerun from the recipe root.
set -euo pipefail
python3 kit/render.py --check
python3 -m unittest discover -s tests -q
VALIDATE_ONLY=1 ./run.sh
./run.sh
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models
