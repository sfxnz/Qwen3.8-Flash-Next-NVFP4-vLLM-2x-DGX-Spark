#!/usr/bin/env bash
# Post-boot probes for c4. Run only after /health is 200.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/evidence/opt-c4-native-ctx-262144"
mkdir -p "$OUT"

curl -sS --max-time 5 -o "$OUT/health.txt" -w '%{http_code}\n' http://127.0.0.1:8000/health | tee "$OUT/health.code"
curl -sS --max-time 5 http://127.0.0.1:8000/v1/models > "$OUT/models.json"
docker inspect qwen38-flash-next-nvfp4 --format '{{.Config.Image}} {{.Created}}' > "$OUT/inspect-image.txt"
docker inspect qwen38-flash-next-nvfp4 --format '{{json .Config.Env}}' > "$OUT/inspect-env.json"
docker inspect qwen38-flash-next-nvfp4 --format '{{json .HostConfig.Binds}}' > "$OUT/inspect-mounts.json"
docker inspect qwen38-flash-next-nvfp4 --format '{{json .Args}}' > "$OUT/inspect-args.json"
docker logs qwen38-flash-next-nvfp4 2>&1 | grep -E 'Unknown vLLM|quantization=|FP8 MoE|KV cache size|Graph capturing finished|Qwen4ExpPLE|w2_weight_scale|v0.28.1|AttributeError|max_seq_len|max_model_len|ALLOW_LONG' | head -80 > "$OUT/engine-needles.txt" || true
python3 - <<'PY'
import json
from pathlib import Path
out = Path("evidence/opt-c4-native-ctx-262144")
env = json.loads((out / "inspect-env.json").read_text())
redact = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}
env = [
    f"{x.split('=', 1)[0]}=REDACTED" if x.split('=', 1)[0] in redact else x
    for x in env
]
(out / "inspect-env.json").write_text(json.dumps(env) + "\n")
args = json.loads((out / "inspect-args.json").read_text())
models = json.loads((out / "models.json").read_text())
allow = [x for x in env if x.startswith("VLLM_ALLOW_LONG_MAX_MODEL_LEN=")]
idx = args.index("--max-model-len") + 1 if "--max-model-len" in args else None
max_len = args[idx] if idx is not None else None
served = models["data"][0]["max_model_len"]
(out / "ctx-gate-check.txt").write_text(
    f"allow={allow}\narg_max_model_len={max_len}\nserved_max_model_len={served}\n"
)
print(f"allow={allow} arg={max_len} served={served}")
if allow != ["VLLM_ALLOW_LONG_MAX_MODEL_LEN=0"]:
    raise SystemExit("ALLOW_LONG is not 0")
if str(max_len) != "262144" or int(served) != 262144:
    raise SystemExit("served window is not 262144")
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
text = Path("evidence/opt-c4-native-ctx-262144/bench.txt").read_text()
m = re.search(r"SUMMARY\s+(\[.*\])", text, re.S)
if not m:
    raise SystemExit("no SUMMARY in bench.txt")
Path("evidence/opt-c4-native-ctx-262144/bench.txt.summary.json").write_text(m.group(1) + "\n")
print(m.group(1)[:400])
PY
curl -sS --max-time 5 -o "$OUT/health-after.txt" -w '%{http_code}\n' http://127.0.0.1:8000/health > "$OUT/health-after.code"
free -h > "$OUT/free-after-bench.txt"
ssh -o BatchMode=yes -o ConnectTimeout=8 spark2 'free -h' > "$OUT/free-after-bench-spark2.txt"
echo probes-done
