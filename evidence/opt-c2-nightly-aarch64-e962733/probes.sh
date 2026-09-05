#!/usr/bin/env bash
# Post-boot probes for c2. Run only after /health is 200.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/evidence/opt-c2-nightly-aarch64-e962733"
mkdir -p "$OUT"

curl -sS --max-time 5 -o "$OUT/health.txt" -w '%{http_code}\n' http://127.0.0.1:8000/health | tee "$OUT/health.code"
curl -sS --max-time 5 http://127.0.0.1:8000/v1/models > "$OUT/models.json"
docker inspect qwen38-flash-next-nvfp4 --format '{{.Config.Image}} {{.Created}}' > "$OUT/inspect-image.txt"
docker inspect qwen38-flash-next-nvfp4 --format '{{json .Config.Env}}' > "$OUT/inspect-env.json"
docker inspect qwen38-flash-next-nvfp4 --format '{{json .HostConfig.Binds}}' > "$OUT/inspect-mounts.json"
docker inspect qwen38-flash-next-nvfp4 --format '{{json .Args}}' > "$OUT/inspect-args.json"
docker logs qwen38-flash-next-nvfp4 2>&1 | grep -E 'Unknown vLLM|quantization=|FP8 MoE|KV cache size|Graph capturing finished|Qwen4ExpPLE|w2_weight_scale|v0.28.1|AttributeError' | head -80 > "$OUT/engine-needles.txt" || true
python3 - <<'PY'
import json
from pathlib import Path
env = json.loads(Path("evidence/opt-c2-nightly-aarch64-e962733/inspect-env.json").read_text())
needles = Path("evidence/opt-c2-nightly-aarch64-e962733/engine-needles.txt").read_text()
has_env = any(x.startswith("VLLM_PLE_FP8_CHECKPOINT=") for x in env)
unknown = "Unknown vLLM environment variable detected: VLLM_PLE_FP8_CHECKPOINT" in needles
mixed = "quantization=modelopt_mixed" in needles
attr = "w2_weight_scale_inv" in needles and "AttributeError" in needles
Path("evidence/opt-c2-nightly-aarch64-e962733/ple-gate-check.txt").write_text(
    f"has_VLLM_PLE_FP8_CHECKPOINT={has_env}\nunknown_env_warning={unknown}\nquantization_modelopt_mixed={mixed}\nmtp_attr_error={attr}\n"
)
print(f"has_env={has_env} unknown={unknown} mixed={mixed} attr={attr}")
PY
free -h > "$OUT/free-after-boot.txt"
ssh -o BatchMode=yes -o ConnectTimeout=8 spark2 'free -h' > "$OUT/free-after-boot-spark2.txt"

set +e
python3 smoke_thinking.py > "$OUT/thinking-off.json" 2>"$OUT/thinking-off.err"
echo $? > "$OUT/thinking-off.exit"
python3 smoke_tools.py > "$OUT/tools.json" 2>"$OUT/tools.err"
echo $? > "$OUT/tools.exit"
python3 smoke_vision.py > "$OUT/vision.json" 2>"$OUT/vision.err"
echo $? > "$OUT/vision.exit"
python3 smoke_count.py > "$OUT/count.json" 2>"$OUT/count.err"
echo $? > "$OUT/count.exit"
python3 bench_decode.py --runs 3 --concurrency 1 2 8 --max-tokens 200 > "$OUT/bench.txt" 2>"$OUT/bench.err"
echo $? > "$OUT/bench.exit"
set -e
python3 - <<'PY'
import re
from pathlib import Path
text = Path("evidence/opt-c2-nightly-aarch64-e962733/bench.txt").read_text()
m = re.search(r"SUMMARY\s+(\[.*\])", text, re.S)
if not m:
    raise SystemExit("no SUMMARY in bench.txt")
Path("evidence/opt-c2-nightly-aarch64-e962733/bench.txt.summary.json").write_text(m.group(1) + "\n")
print(m.group(1)[:400])
PY
curl -sS --max-time 5 -o "$OUT/health-after.txt" -w '%{http_code}\n' http://127.0.0.1:8000/health > "$OUT/health-after.code"
free -h > "$OUT/free-after-bench.txt"
ssh -o BatchMode=yes -o ConnectTimeout=8 spark2 'free -h' > "$OUT/free-after-bench-spark2.txt"
echo probes-done
