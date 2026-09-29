#!/usr/bin/env bash
# Torch-profiler window on a live serve (S1.2 step 5), adapted from the
# DeepSeek-V4.1 sibling's tools/profile_window.sh. Boot the serve with
#   EXTRA_ARGS='--profiler-config {"profiler":"torch","torch_profiler_dir":"/tmp/qwen38-traces"}'
# (run.sh forwards EXTRA_ARGS to the worker, so both ranks write traces).
#
# Protocol: warmup (profiler off) -> start_profile -> CONCURRENCY streaming
# prose requests (thinking off, the recipe default) -> stop_profile -> flush
# wait -> list traces on BOTH ranks -> docker cp on each host BEFORE ./stop.sh
# (docker rm destroys them). Parse each rank on its own idle host with
# tools/micro/extract_kernels.py (RLIMIT_AS, streaming ijson, never json.load).
# Watch MemAvail (free -h): profiled steps grow worker host memory.
#
# CONCURRENCY=N (default 1) sends N identical streams inside the window (8 for
# the c=8 cell). MAX_TOKENS (default 128) keeps the window short. stop_profile
# blocks until the export ends, so STOP_TIMEOUT_S defaults to 600.
# This script never stops the serve. Run ./stop.sh yourself afterwards.
set -euo pipefail
PORT="${PORT:-8000}"
API="${API:-http://127.0.0.1:$PORT}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next-nvfp4}"
MODEL="${MODEL:-nvidia/Qwen3.8-Flash-Next-NVFP4}"
WORKER_HOST="${WORKER_HOST:-spark2}"
TRACE_DIR="${TRACE_DIR:-/tmp/qwen38-traces}"
OUT_DIR="${OUT_DIR:-$HOME/projects/data/qwen38-traces/$(date +%Y%m%d-%H%M%S)}"
FLUSH_S="${FLUSH_S:-15}"
STOP_TIMEOUT_S="${STOP_TIMEOUT_S:-600}"
CONCURRENCY="${CONCURRENCY:-1}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PROMPT='Write a short paragraph about why sparse attention helps long-context language models. Keep it around eighty words. No bullet points.'
body() {
  printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%d,"min_tokens":%d,"ignore_eos":true,"temperature":0,%s,"chat_template_kwargs":{"enable_thinking":false}}' \
    "$MODEL" "$PROMPT" "$MAX_TOKENS" "$MAX_TOKENS" "$1"
}

echo "== warmup (profiler off) =="
curl -s --max-time 120 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "$(body '"stream":false')" >"$TMP/warmup.json"
python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print('warmup completion tokens:', r['usage']['completion_tokens'])" "$TMP/warmup.json"

echo "== start profile =="
curl -s --max-time 30 -X POST "$API/start_profile"
echo
echo "== profiled request(s) (stream, $MAX_TOKENS tok, concurrency $CONCURRENCY) =="
S=$(date +%s.%N)
pids=()
for ((i = 1; i <= CONCURRENCY; i++)); do
  curl -sN --max-time 180 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(body '"stream":true,"stream_options":{"include_usage":true}')" >"$TMP/profiled.$i.sse" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
E=$(date +%s.%N)
python3 - "$S" "$E" "$TMP"/profiled.*.sse <<'PY'
import json, sys
for path in sys.argv[3:]:
    usage, chunks = None, 0
    for line in open(path):
        if line.startswith("data: ") and "[DONE]" not in line:
            try:
                d = json.loads(line[6:])
            except ValueError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices"):
                chunks += 1
    print("profiled req usage:", usage, "content chunks (~steps):", chunks)
print("wall s:", round(float(sys.argv[2]) - float(sys.argv[1]), 2))
PY
echo "== stop profile =="
stop_rc=0
curl -s --max-time "$STOP_TIMEOUT_S" -X POST "$API/stop_profile" || stop_rc=$?
echo
[[ "$stop_rc" == 0 ]] || echo "WARN: stop_profile curl rc=$stop_rc after ${STOP_TIMEOUT_S}s; copying both ranks anyway (traces may be partial)"
date
sleep "$FLUSH_S"

echo "== trace files: rank 0 (head) =="
docker exec "$CONTAINER_NAME" sh -c "ls -la $TRACE_DIR/" \
  || echo "WARN: no rank-0 trace dir; still copying rank 1"
echo "== trace files: rank 1 ($WORKER_HOST) =="
ssh "$WORKER_HOST" docker exec "$CONTAINER_NAME" sh -c "'ls -la $TRACE_DIR/'" \
  || echo "WARN: no rank-1 trace dir on $WORKER_HOST; still copying rank 0"

echo "== MemAvail before cp =="
free -h | sed -n 1,2p
ssh "$WORKER_HOST" free -h | sed -n 1,2p

echo "== docker cp BEFORE stop: rank 0 -> $(hostname):$OUT_DIR/rank0 =="
mkdir -p "$OUT_DIR"
docker cp "$CONTAINER_NAME:$TRACE_DIR" "$OUT_DIR/rank0" && ls -la "$OUT_DIR/rank0" \
  || echo "WARN: rank-0 copy failed; still copying rank 1"
echo "== docker cp BEFORE stop: rank 1 -> $WORKER_HOST:$OUT_DIR/rank1 =="
ssh "$WORKER_HOST" "mkdir -p '$OUT_DIR' && docker cp '$CONTAINER_NAME:$TRACE_DIR' '$OUT_DIR/rank1' && ls -la '$OUT_DIR/rank1'" \
  || echo "WARN: rank-1 copy failed on $WORKER_HOST"

echo "== MemAvail after cp =="
free -h | sed -n 1,2p
ssh "$WORKER_HOST" free -h | sed -n 1,2p
echo "Traces saved. Now ./stop.sh, then parse each rank on its own idle host."
[[ "$stop_rc" == 0 ]] || { echo "WARN: stop_profile failed (rc=$stop_rc); check the traces are complete" >&2; exit 1; }
