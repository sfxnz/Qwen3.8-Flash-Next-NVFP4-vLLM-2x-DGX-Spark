#!/usr/bin/env bash
# Post-boot probes for c5. Run only after /health is 200.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/evidence/opt-c5-mtp-k2"
mkdir -p "$OUT"

curl -sS --max-time 5 -o "$OUT/health.txt" -w '%{http_code}\n' http://127.0.0.1:8000/health | tee "$OUT/health.code"
curl -sS --max-time 5 http://127.0.0.1:8000/v1/models > "$OUT/models.json"
docker inspect qwen38-flash-next-nvfp4 --format '{{.Config.Image}} {{.Created}}' > "$OUT/inspect-image.txt"
docker inspect qwen38-flash-next-nvfp4 --format '{{json .Config.Env}}' > "$OUT/inspect-env.json"
python3 - <<'PY'
import json
from pathlib import Path
p = Path("evidence/opt-c5-mtp-k2/inspect-env.json")
redact = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}
env = json.loads(p.read_text())
env = [
    f"{x.split('=', 1)[0]}=REDACTED" if x.split('=', 1)[0] in redact else x
    for x in env
]
p.write_text(json.dumps(env) + "\n")
PY
docker inspect qwen38-flash-next-nvfp4 --format '{{json .HostConfig.Binds}}' > "$OUT/inspect-mounts.json"
docker inspect qwen38-flash-next-nvfp4 --format '{{json .Args}}' > "$OUT/inspect-args.json"
docker inspect qwen38-flash-next-nvfp4 --format '{{.Image}}' > "$OUT/inspect-image-id.txt"
docker logs qwen38-flash-next-nvfp4 2>&1 | grep -E 'Unknown vLLM|quantization=|FP8 MoE|KV cache size|Graph capturing finished|Qwen4ExpPLE|w2_weight_scale|v0.28.1|AttributeError|max_seq_len|max_model_len|ALLOW_LONG|speculative|num_speculative' | head -80 > "$OUT/engine-needles.txt" || true
python3 - <<'PY'
import json
from pathlib import Path
out = Path("evidence/opt-c5-mtp-k2")
args = json.loads((out / "inspect-args.json").read_text())
models = json.loads((out / "models.json").read_text())
spec = args[args.index("--speculative-config") + 1]
served = models["data"][0]
(out / "spec-gate-check.txt").write_text(
    f"spec={spec}\nserved_id={served['id']}\nserved_max_model_len={served['max_model_len']}\n"
)
print(f"spec={spec} served={served['id']} max_model_len={served['max_model_len']}")
if '"num_speculative_tokens":2' not in spec.replace(" ", ""):
    raise SystemExit("speculative tokens is not 2")
if served["id"] != "nvidia/Qwen3.8-Flash-Next-NVFP4":
    raise SystemExit("served id is not nvidia pin")
if int(served["max_model_len"]) != 262144:
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
text = Path("evidence/opt-c5-mtp-k2/bench.txt").read_text()
m = re.search(r"SUMMARY\s+(\[.*\])", text, re.S)
if not m:
    raise SystemExit("no SUMMARY in bench.txt")
Path("evidence/opt-c5-mtp-k2/bench.txt.summary.json").write_text(m.group(1) + "\n")
print(m.group(1)[:400])
PY
curl -sS --max-time 5 -o "$OUT/health-after.txt" -w '%{http_code}\n' http://127.0.0.1:8000/health > "$OUT/health-after.code"
free -h > "$OUT/free-after-bench.txt"
ssh -o BatchMode=yes -o ConnectTimeout=8 spark2 'free -h' > "$OUT/free-after-bench-spark2.txt"
echo probes-done
