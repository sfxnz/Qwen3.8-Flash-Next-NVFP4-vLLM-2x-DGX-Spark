#!/usr/bin/env bash
# Official NVIDIA ModelOpt pin: stop the live RadixArk serve, boot fab0aec, gate, bench.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/evidence/opt-h0-nvidia-modelopt-pin"
mkdir -p "$OUT"
MODEL_ID="nvidia/Qwen3.8-Flash-Next-NVFP4"
SNAPSHOT_SHA="fab0aecb760cec45227f6656abcaafa11abca87a"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
SNAP="${HF_CACHE}/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/${SNAPSHOT_SHA}"

run_smoke() {
  local name="$1" script="$2"
  set +e
  python3 "$script" --model "$MODEL_ID" >"$OUT/${name}.json" 2>"$OUT/${name}.err"
  local rc=$?
  set -e
  printf '%s\n' "$rc" >"$OUT/${name}.exit"
  return "$rc"
}

date -u +%Y-%m-%dT%H:%M:%SZ >"$OUT/started-at.txt"
free -h >"$OUT/free-before.txt"
ssh -o BatchMode=yes -o ConnectTimeout=5 spark2 'free -h' >"$OUT/free-before-spark2.txt"

if [[ ! -f "$SNAP/config.json" ]]; then
  echo "pinned snapshot missing: $SNAP" >&2
  exit 1
fi
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 spark2 "test -f '$SNAP/config.json'"; then
  echo "spark2 missing $SNAP" >&2
  exit 1
fi

./stop.sh
./run.sh

curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health >"$OUT/health.txt"
curl -sf http://127.0.0.1:8000/v1/models >"$OUT/models.json"
docker inspect qwen38-flash-next-nvfp4 --format '{{json .Config.Cmd}}' >"$OUT/inspect-args.json"
docker inspect qwen38-flash-next-nvfp4 --format '{{json .Config.Env}}' >"$OUT/inspect-env.json"
docker inspect qwen38-flash-next-nvfp4 --format '{{.Config.Image}}' >"$OUT/inspect-image.txt"
sha256sum bench_decode.py smoke_thinking.py smoke_tools.py smoke_vision.py smoke_count.py >"$OUT/harness.sha256"
free -h >"$OUT/free-after-boot.txt"
ssh -o BatchMode=yes -o ConnectTimeout=5 spark2 'free -h' >"$OUT/free-after-boot-spark2.txt"
docker logs qwen38-flash-next-nvfp4 2>&1 | grep -E 'KV cache size|Available KV cache|Graph capturing finished|Maximum concurrency|CUDA graphs|non-default args|Using V2 Model Runner|gpu_memory_utilization|quantization|MIXED_PRECISION|modelopt|PLE' | tail -80 >"$OUT/engine-needles.txt"

python3 - <<PY >"$OUT/warm.json"
import json, urllib.request
body = json.dumps({
    "model": "$MODEL_ID",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 8,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=180) as resp:
    payload = json.load(resp)
msg = (payload.get("choices") or [{}])[0].get("message") or {}
print(json.dumps({
    "content": msg.get("content"),
    "finish": (payload.get("choices") or [{}])[0].get("finish_reason"),
}))
PY

gate=0
run_smoke thinking-off smoke_thinking.py || gate=1
run_smoke tools smoke_tools.py || gate=1
run_smoke vision smoke_vision.py || gate=1
run_smoke count smoke_count.py || gate=1

set +e
python3 bench_decode.py --model "$MODEL_ID" --runs 3 --concurrency 1 2 8 --max-tokens 200 >"$OUT/bench.txt" 2>"$OUT/bench.err"
bench_rc=$?
set -e
printf '%s\n' "$bench_rc" >"$OUT/bench.exit"

python3 - <<'PY'
import json, pathlib, sys
p = pathlib.Path("evidence/opt-h0-nvidia-modelopt-pin/bench.txt")
text = p.read_text()
idx = text.find("SUMMARY")
if idx < 0:
    sys.exit("no SUMMARY in bench.txt")
summary = json.loads(text[idx + len("SUMMARY"):].strip())
pathlib.Path("evidence/opt-h0-nvidia-modelopt-pin/bench.txt.summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
PY

sha256sum "$OUT/bench.txt" "$OUT/bench.txt.summary.json" >>"$OUT/harness.sha256"
free -h >"$OUT/free-after-bench.txt"
ssh -o BatchMode=yes -o ConnectTimeout=5 spark2 'free -h' >"$OUT/free-after-bench-spark2.txt"
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health >"$OUT/health-after.txt"

if [[ "$gate" != 0 || "$bench_rc" != 0 ]]; then
  echo "opt-h0-nvidia-modelopt-pin gate failed smoke=$gate bench=$bench_rc" >&2
  exit 1
fi
printf 'opt-h0-nvidia-modelopt-pin capture ok\n'
