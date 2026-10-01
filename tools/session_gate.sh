#!/usr/bin/env bash
# Gate pack for one Spark session on a live serve (plan 0-4, 0-5, 0-6).
#
#   tools/session_gate.sh EVDIR
#
# Order: receipts and free -h (before) -> telemetry start -> smokes ->
# bench_probe (T0) -> ruler_steps (warm-up + sentinels + cells) -> frozen
# bench_decode.py once (R2 regression guard) -> telemetry stop -> receipts
# (free -h both nodes, docker logs tail both ranks, engine needles, /metrics
# snapshot, harness sha256) -> gate.txt. Exits nonzero if any step fails.
#
# Env:
#   SELF_TEST=1      B0 self-test boot: no warm-up, no sentinels; ruler_steps
#                    --historical checks the expected flagged-wave set and
#                    bench_probe --self-test expects the bug (plan 0-6).
#   RUN_FROZEN=0     skip the frozen bench_decode.py run.
#   TELEMETRY=0      skip tools/telemetry.sh.
#   PROBE_ARGS / RULER_ARGS   extra args for bench_probe.py / ruler_steps.py.
#   URL, MODEL, WORKER_HOST, CONTAINER_NAME, PORT as in run.sh.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -eq 1 ]] || { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
EV="$(mkdir -p "$1" && cd "$1" && pwd)"
PORT="${PORT:-8000}"
BASE="http://127.0.0.1:${PORT}"
URL="${URL:-${BASE}/v1/chat/completions}"
MODEL="${MODEL:-nvidia/Qwen3.8-Flash-Next-NVFP4}"
WORKER_HOST="${WORKER_HOST:-spark2}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next-nvfp4}"
SELF_TEST="${SELF_TEST:-0}"
RUN_FROZEN="${RUN_FROZEN:-1}"
TELEMETRY="${TELEMETRY:-1}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8)
export WORKER_HOST CONTAINER_NAME PORT
cd "$ROOT" || exit 2

declare -a RESULTS=()
FAILED=0
stamp() { date -u +%FT%T.%3NZ; }
log() { echo "$(stamp) session_gate: $*" | tee -a "$EV/gate.log"; }

# step NAME CMD... : run with stdout/stderr/exit captured under EVDIR.
step() {
  local name="$1"; shift
  log "start $name"
  "$@" >"$EV/$name.out" 2>"$EV/$name.err"
  local rc=$?
  echo "$rc" >"$EV/$name.exit"
  RESULTS+=("$name=$rc")
  [[ $rc -eq 0 ]] || FAILED=1
  log "end $name rc=$rc"
  return 0
}

free_both() {
  free -h >"$EV/free-$1.txt" 2>&1
  "${SSH[@]}" "$WORKER_HOST" 'free -h' >"$EV/free-$1-${WORKER_HOST}.txt" 2>&1 || echo "ssh failed" >>"$EV/free-$1-${WORKER_HOST}.txt"
}

metrics_snapshot() {
  curl -sS --max-time 10 "$BASE/metrics" >"$EV/metrics-$1.txt" 2>&1 || true
}

needles() {
  docker logs "$CONTAINER_NAME" 2>&1 | grep -E 'Unknown vLLM|quantization=|FP8 MoE|KV cache size|Graph capturing finished|Qwen4ExpPLE|w2_weight_scale|v0\.[0-9]+\.[0-9]|AttributeError|max_seq_len|max_model_len|ALLOW_LONG|per_request_spec_decode_metrics|Traceback|ERROR' \
    | head -120 >"$EV/engine-needles.txt" || true
}

logs_tail() {
  docker logs --tail 400 "$CONTAINER_NAME" >"$EV/docker-tail-head.log" 2>&1 || true
  "${SSH[@]}" "$WORKER_HOST" "docker logs --tail 400 '$CONTAINER_NAME' 2>&1" >"$EV/docker-tail-${WORKER_HOST}.log" 2>&1 || true
}

# --- preflight ---------------------------------------------------------------
log "evidence dir $EV self_test=$SELF_TEST"
code="$(curl -s -o "$EV/health.txt" -w '%{http_code}' --max-time 5 "$BASE/health" || true)"
echo "$code" >"$EV/health.code"
if [[ "$code" != 200 ]]; then
  log "GET /health = ${code:-none}; refusing to run the gate on a server that is not up"
  exit 1
fi
curl -sS --max-time 5 "$BASE/v1/models" >"$EV/models.json" || true
if ! grep -q "\"$MODEL\"" "$EV/models.json"; then
  log "/v1/models does not list $MODEL"
  exit 1
fi
sha256sum bench_decode.py ruler_steps.py bench_probe.py smoke_*.py tools/telemetry.sh tools/session_gate.sh >"$EV/harness.sha256"
git rev-parse HEAD >"$EV/git-head.txt" 2>/dev/null || true
free_both before
metrics_snapshot before
[[ "$TELEMETRY" == 1 ]] && { tools/telemetry.sh receipt "$EV" >>"$EV/gate.log" 2>&1; tools/telemetry.sh start "$EV" >>"$EV/gate.log" 2>&1; }
trap '[[ "$TELEMETRY" == 1 ]] && tools/telemetry.sh stop "$EV" >>"$EV/gate.log" 2>&1' EXIT

# --- smokes ------------------------------------------------------------------
step smoke-thinking python3 smoke_thinking.py --url "$URL" --model "$MODEL"
step smoke-tools python3 smoke_tools.py --url "$URL" --model "$MODEL"
step smoke-vision python3 smoke_vision.py --url "$URL" --model "$MODEL"
step smoke-count python3 smoke_count.py --url "$URL" --model "$MODEL"

# --- T0, warm-up, rulers ------------------------------------------------------
# shellcheck disable=SC2086
if [[ "$SELF_TEST" == 1 ]]; then
  # Fresh B0 boot: the historical order must run first, with no warm-up.
  step ruler-historical python3 ruler_steps.py --url "$URL" --model "$MODEL" --historical \
    --no-warmup --no-sentinels --json "$EV/ruler-historical.json" ${RULER_ARGS:-}
  step probe-selftest python3 bench_probe.py --url "$URL" --model "$MODEL" --self-test \
    --json "$EV/probe-selftest.json" ${PROBE_ARGS:-}
else
  step probe python3 bench_probe.py --url "$URL" --model "$MODEL" --json "$EV/probe.json" ${PROBE_ARGS:-}
  step ruler python3 ruler_steps.py --url "$URL" --model "$MODEL" --json "$EV/ruler.json" ${RULER_ARGS:-}
fi
if [[ "$RUN_FROZEN" == 1 ]]; then
  step bench-frozen python3 bench_decode.py --url "$URL" --model "$MODEL" --runs 3 --concurrency 1 2 --max-tokens 200
fi

# --- receipts ----------------------------------------------------------------
metrics_snapshot after
free_both after
logs_tail
needles
{
  echo "finished_at=$(stamp)"
  echo "self_test=$SELF_TEST"
  printf '%s\n' "${RESULTS[@]}"
  if [[ $FAILED -eq 0 ]]; then echo "GATE=PASS"; else echo "GATE=FAIL"; fi
} >"$EV/gate.txt"
cat "$EV/gate.txt" | tee -a "$EV/gate.log"
exit "$FAILED"
