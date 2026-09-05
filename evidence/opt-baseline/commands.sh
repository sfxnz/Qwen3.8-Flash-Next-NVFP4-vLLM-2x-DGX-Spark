#!/usr/bin/env bash
# Frozen opt-baseline: correctness gate plus bench_decode.py on the live seqs=8 pin.
# Does not restart the serve. Does not change recipe.yaml or run.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/evidence/opt-baseline"
mkdir -p "$OUT"

run_smoke() {
  local name="$1" script="$2"
  set +e
  python3 "$script" >"$OUT/${name}.json" 2>"$OUT/${name}.err"
  local rc=$?
  set -e
  printf '%s\n' "$rc" >"$OUT/${name}.exit"
  return "$rc"
}

curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health >"$OUT/health.txt"
curl -sf http://127.0.0.1:8000/v1/models >"$OUT/models.json"
docker inspect qwen38-flash-next-nvfp4 --format '{{json .Config.Cmd}}' >"$OUT/inspect-args.json"
docker inspect qwen38-flash-next-nvfp4 --format '{{json .Config.Env}}' >"$OUT/inspect-env.json"
python3 - "$OUT/inspect-env.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
env = json.loads(p.read_text())
keys = ("HF_TOKEN=", "HUGGING_FACE_HUB_TOKEN=")
out = []
for item in env:
    hit = next((k for k in keys if item.startswith(k)), None)
    out.append((hit + "REDACTED") if hit else item)
p.write_text(json.dumps(out) + "\n")
PY
docker inspect qwen38-flash-next-nvfp4 --format '{{.Config.Image}}' >"$OUT/inspect-image.txt"
sha256sum bench_decode.py smoke_thinking.py smoke_tools.py smoke_vision.py smoke_count.py >"$OUT/harness.sha256"
free -h >"$OUT/free-before.txt"
ssh -o BatchMode=yes -o ConnectTimeout=5 spark2 'free -h' >"$OUT/free-before-spark2.txt"
docker logs qwen38-flash-next-nvfp4 2>&1 | grep -E 'KV cache size|Available KV cache|Graph capturing finished|Maximum concurrency|CUDA graphs|non-default args|Using V2 Model Runner|gpu_memory_utilization' | tail -40 >"$OUT/engine-needles.txt"

python3 - <<'PY' >"$OUT/warm.json"
import json, urllib.request
body = json.dumps({
    "model": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
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
python3 bench_decode.py --runs 3 --concurrency 1 2 8 --max-tokens 200 >"$OUT/bench.txt" 2>"$OUT/bench.err"
bench_rc=$?
set -e
printf '%s\n' "$bench_rc" >"$OUT/bench.exit"

python3 - <<'PY'
import json, pathlib, sys
p = pathlib.Path("evidence/opt-baseline/bench.txt")
text = p.read_text()
idx = text.find("SUMMARY")
if idx < 0:
    sys.exit("no SUMMARY in bench.txt")
summary = json.loads(text[idx + len("SUMMARY"):].strip())
pathlib.Path("evidence/opt-baseline/bench.txt.summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
PY

sha256sum "$OUT/bench.txt" "$OUT/bench.txt.summary.json" >>"$OUT/harness.sha256"
free -h >"$OUT/free-after.txt"
ssh -o BatchMode=yes -o ConnectTimeout=5 spark2 'free -h' >"$OUT/free-after-spark2.txt"
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health >"$OUT/health-after.txt"

if [[ "$gate" != 0 || "$bench_rc" != 0 ]]; then
  echo "opt-baseline gate failed smoke=$gate bench=$bench_rc" >&2
  exit 1
fi
printf 'opt-baseline capture ok\n'
