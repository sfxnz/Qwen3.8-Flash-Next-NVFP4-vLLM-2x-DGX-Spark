#!/usr/bin/env bash
# Session 3 per-boot step runner (U2: v0.30 + determinism default). Run from the recipe worktree root.
#   evidence/u2-v030-default/steps.sh SUBDIR STAGE...
# Evidence goes to evidence/u2-v030-default/SUBDIR, raw data to ~/projects/data/qwen38-evals/runs/session3/SUBDIR.
# Stages: gate ruler-anchor diverse sampled t1g t2 alias t3 vision vbench prefix toolstream mixed receipts
# Env: ANCHOR="LO HI" for ruler-anchor; T3_LENGTHS (default 4096 16384 32768 65536 131072); MIXED_CELLS (default base A B).
set -uo pipefail
SUB="$1"; shift
ROOT="$(pwd)"
EV="$ROOT/evidence/u2-v030-default/$SUB"
R="$HOME/projects/data/qwen38-evals/runs/session3/$SUB"
URL=http://127.0.0.1:8000/v1/chat/completions
mkdir -p "$EV" "$R"
stamp() { date -u +%FT%T.%3NZ; }
log() { echo "$(stamp) [$SUB] $*" | tee -a "$EV/steps.log"; }
run() {  # run NAME CMD... ; stdout/err to EV/NAME.out, rc to EV/NAME.exit
  local n="$1"; shift
  log "start $n: $*"
  "$@" >"$EV/$n.out" 2>&1; local rc=$?
  echo "$rc" >"$EV/$n.exit"; log "end $n rc=$rc"
}
freeboth() {
  { echo "== spark1 $(stamp)"; free -h; echo "== spark2"; ssh spark2 free -h; } >>"$EV/free-$1.txt" 2>&1
}
# Safety rule: before a heavy step, skip if spark1 MemAvailable < 6 GiB or swap used > 10 GiB (either node).
memok() {
  local a1 s1 s2
  a1=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  s1=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)
  s2=$(ssh spark2 "awk '/SwapTotal/{t=\$2}/SwapFree/{f=\$2}END{print int((t-f)/1024)}' /proc/meminfo")
  freeboth "memcheck-$1"
  if (( a1 < 6144 || s1 > 10240 || ${s2:-0} > 10240 )); then
    log "SKIP $1: memcheck spark1 avail=${a1}MB swap_used spark1=${s1}MB spark2=${s2}MB"; return 1
  fi
  log "memcheck ok before $1: spark1 avail=${a1}MB swap_used spark1=${s1}MB spark2=${s2}MB"
}
alive() {  # memguard trip or crash: the serve is gone
  [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8000/health)" == 200 ]] && return 0
  log "SERVE DOWN after $1 (memguard log: $(tail -1 "$ROOT/.run-state/memguard.log" 2>/dev/null))"; return 1
}

for st in "$@"; do
  case "$st" in
    gate)  # smokes, T0 probe, anchor-free ruler, frozen bench_decode
      freeboth pre-gate
      RULER_ARGS="--anchor-ms 0 0" tools/session_gate.sh "$EV/gate" >"$EV/gate.out" 2>&1
      echo $? >"$EV/gate.exit"; log "gate rc=$(cat "$EV/gate.exit")" ;;
    ruler-anchor)
      # shellcheck disable=SC2086
      run ruler-anchor python3 ruler_steps.py --url "$URL" --anchor-ms ${ANCHOR:?set ANCHOR="LO HI"} --json "$EV/ruler-anchor.json" ;;
    ruler-noanchor)
      run ruler-noanchor python3 ruler_steps.py --url "$URL" --anchor-ms 0 0 --json "$EV/ruler-noanchor.json" ;;
    diverse)
      run diverse python3 bench_diverse.py --url "$URL" --concurrency 1 8 --out "$EV/diverse.jsonl" --label "$SUB" ;;
    sampled)
      run sampled python3 bench_sampled.py --url "$URL" --profiles A T1 --concurrency 1 8 --prompts 8 --reps 1 \
        --out "$EV/sampled.jsonl" --label "$SUB" ;;
    t1g)
      run t1g python3 quality/collect_t1g.py --url "$URL" --out "$R/t1g" --limit 10 --repeats 5
      run t1g-self python3 quality/score.py t1g "$R/t1g" "$R/t1g" --json "$EV/t1g-self.json" ;;
    t2)
      run t2 python3 quality/t2.py --url "$URL" --out "$R/t2" --tasks tools,json,rep4,effort,ifeval,gsm8k --workers 4
      for f in "$R"/t2/*.json; do [[ -f "$f" ]] && cp "$f" "$EV/t2-$(basename "$f")"; done ;;
    alias)  # P2-3: reasoning_effort high / max / minimal must return 200 through the alias template
      run alias python3 - <<'PY'
import json, urllib.request
bad = 0
for e in ("high", "max", "minimal", "xhigh", "low"):
    body = {"model": "nvidia/Qwen3.8-Flash-Next-NVFP4", "messages": [{"role": "user", "content": "What is 17*3?"}],
            "max_tokens": 256, "temperature": 0, "reasoning_effort": e}
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            j = json.load(r); code = r.status
    except urllib.error.HTTPError as ex:
        code, j = ex.code, {"error": ex.read().decode()[:300]}
    m = (j.get("choices") or [{}])[0].get("message", {})
    print(json.dumps({"effort": e, "http": code, "reasoning_chars": len(m.get("reasoning") or m.get("reasoning_content") or ""),
                      "content": (m.get("content") or "")[:80], "error": j.get("error")}))
    bad += code != 200
raise SystemExit(bad)
PY
      ;;
    t3)  # one length at a time, memory check before each
      for L in ${T3_LENGTHS:-4096 16384 32768 65536 131072}; do
        memok "t3-$L" || break
        run "t3-$L" python3 quality/t3_needles.py --url "$URL" --out "$R/t3-$L" --lengths "$L"
        for f in "$R/t3-$L"/*.json; do [[ -f "$f" ]] && cp "$f" "$EV/t3-$L-$(basename "$f")"; done
        freeboth "post-t3-$L"
        alive "t3-$L" || break
      done ;;
    vision)
      run vision python3 quality/vision_probes/run.py --url "$URL" --out "$R/vision"
      for f in "$R"/vision/*.json; do [[ -f "$f" ]] && cp "$f" "$EV/vision-$(basename "$f")"; done ;;
    vbench)
      memok vbench || continue
      run vbench python3 bench_vision.py --url "$URL" --out "$EV/vbench.jsonl" --label "$SUB" ;;
    prefix)
      run prefix python3 probe_prefix.py --url "$URL" --out "$EV/prefix.jsonl" --label "$SUB" ;;
    toolstream)
      run toolstream python3 probe_toolstream.py --url "$URL" --out "$EV/toolstream.jsonl" --label "$SUB" ;;
    mixed)  # arm B carries a 32k then a 128k prefill
      memok mixed || continue
      # shellcheck disable=SC2086
      run mixed python3 bench_mixed.py --url "$URL" --cells ${MIXED_CELLS:-base A B} --out "$EV/mixed.jsonl" --label "$SUB"
      freeboth post-mixed
      alive mixed ;;
    receipts)
      freeboth end
      docker logs qwen38-flash-next-nvfp4 >"$R/docker-rank0.log" 2>&1
      ssh spark2 "docker logs qwen38-flash-next-nvfp4 2>&1" >"$R/docker-rank1.log" 2>&1
      tail -n 300 "$R/docker-rank0.log" >"$EV/docker-rank0-tail.log"
      tail -n 300 "$R/docker-rank1.log" >"$EV/docker-rank1-tail.log"
      for rk in 0 1; do
        grep -nE 'Unknown vLLM|quantization=|MoE backend|Fp8 MoE|FP8 MoE|KV cache|GPU KV|Graph capturing finished|Qwen4ExpPLE|PLE embedding|pinned=|GDN|gdn|FlashInfer|fused_finalize|DETERMINISTIC|w2_weight_scale|AttributeError|max_model_len|Traceback|ERROR|CUDA error|NCCL WARN|Maximum concurrency|Available KV|model weights took|init engine|Application startup|async_scheduling|Asynchronous scheduling|chat template|chat_template|mm_processor|limit_mm|long_prefill' \
          "$R/docker-rank$rk.log" | head -300 >"$EV/needles-rank$rk.txt"
      done
      cp "$ROOT/.run-state/memguard.log" "$EV/memguard.log" 2>/dev/null
      curl -sS --max-time 10 http://127.0.0.1:8000/metrics >"$EV/metrics-end.txt" 2>&1 ;;
    *) log "unknown stage $st" ;;
  esac
done
