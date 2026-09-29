#!/usr/bin/env bash
# Session 1 per-boot step runner (operator script, copied into evidence/<id>/steps.sh).
#   steps.sh ID MODE STAGE...    MODE = b0 | b1 | u1 ; run from the recipe worktree root
# Stages: gate ruler diverse sampled qaa t2 t3 prefix profile receipts
set -uo pipefail
ID="$1"; MODE="$2"; shift 2
ROOT="$(pwd)"
EV="$ROOT/evidence/$ID"
R="$HOME/projects/data/qwen38-evals/runs/session1/$ID"
URL=http://127.0.0.1:8000/v1/chat/completions
mkdir -p "$EV" "$R"
stamp() { date -u +%FT%T.%3NZ; }
log() { echo "$(stamp) [$ID] $*" | tee -a "$EV/steps.log"; }
run() {  # run NAME CMD... ; stdout/err to EV/NAME.out, rc to EV/NAME.exit
  local n="$1"; shift
  log "start $n: $*"
  "$@" >"$EV/$n.out" 2>&1; local rc=$?
  echo "$rc" >"$EV/$n.exit"; log "end $n rc=$rc"
}
freeboth() {
  { echo "== spark1 $(stamp)"; free -h; echo "== spark2"; ssh spark2 free -h; } >>"$EV/free-$1.txt" 2>&1
}

for st in "$@"; do
  case "$st" in
    gate)
      freeboth pre-gate
      if [[ "$MODE" == b0 ]]; then SELF_TEST=1 tools/session_gate.sh "$EV/gate" >"$EV/gate.out" 2>&1
      else tools/session_gate.sh "$EV/gate" >"$EV/gate.out" 2>&1; fi
      echo $? >"$EV/gate.exit"; log "gate rc=$(cat "$EV/gate.exit")" ;;
    ruler)  # B0 only: the post-self-test ruler with warm-up and sentinels
      run ruler python3 ruler_steps.py --url "$URL" --json "$EV/ruler.json" ;;
    diverse)
      run diverse python3 bench_diverse.py --url "$URL" --concurrency 1 8 --out "$EV/diverse.jsonl" --label "$ID" ;;
    sampled)
      run sampled python3 bench_sampled.py --url "$URL" --profiles A T1 --concurrency 1 8 --prompts 8 --reps 1 \
        --out "$EV/sampled.jsonl" --label "$ID" ;;
    qaa)
      run t1-a python3 quality/collect_t1.py --url "$URL" --out "$R/t1-a" --limit 2
      run t1-b python3 quality/collect_t1.py --url "$URL" --out "$R/t1-b" --limit 2
      run t1-aa python3 quality/score.py t1 "$R/t1-a" "$R/t1-b" --json "$EV/t1-aa.json"
      run t1g python3 quality/collect_t1g.py --url "$URL" --out "$R/t1g" --limit 10 --repeats 5
      run t1g-self python3 quality/score.py t1g "$R/t1g" "$R/t1g" --json "$EV/t1g-self.json"
      cp "$R/t1-a/summary.json" "$EV/t1-a-summary.json" 2>/dev/null; cp "$R/t1-b/summary.json" "$EV/t1-b-summary.json" 2>/dev/null ;;
    t2)
      W="${T2_WORKERS:-4}"
      run t2 python3 quality/t2.py --url "$URL" --out "$R/t2" --tasks tools,json,rep4,effort,ifeval,gsm8k --workers "$W"
      for f in "$R"/t2/*.json; do [[ -f "$f" ]] && cp "$f" "$EV/t2-$(basename "$f")"; done ;;
    t3)
      freeboth pre-t3
      run t3 python3 quality/t3_needles.py --url "$URL" --out "$R/t3" --lengths 4096,16384,32768
      for f in "$R"/t3/*.json; do [[ -f "$f" ]] && cp "$f" "$EV/t3-$(basename "$f")"; done
      freeboth post-t3 ;;
    prefix)
      run prefix python3 probe_prefix.py --url "$URL" --out "$EV/prefix.jsonl" --label "$ID" ;;
    profile)
      OUT_DIR="$HOME/projects/data/qwen38-traces/session1-$ID" run profile tools/micro/profile_window.sh ;;
    receipts)
      freeboth end
      docker logs qwen38-flash-next-nvfp4 >"$R/docker-rank0.log" 2>&1
      ssh spark2 "docker logs qwen38-flash-next-nvfp4 2>&1" >"$R/docker-rank1.log" 2>&1
      tail -n 300 "$R/docker-rank0.log" >"$EV/docker-rank0-tail.log"
      tail -n 300 "$R/docker-rank1.log" >"$EV/docker-rank1-tail.log"
      for rk in 0 1; do
        grep -nE 'Unknown vLLM|quantization=|MoE backend|Fp8 MoE|FP8 MoE|KV cache|GPU KV|Graph capturing finished|Qwen4ExpPLE|PLE embedding|pinned=|GDN|gdn|FlashInfer|w2_weight_scale|AttributeError|max_model_len|Traceback|ERROR|CUDA error|NCCL WARN|Maximum concurrency|Available KV|model weights took|init engine|Application startup' \
          "$R/docker-rank$rk.log" | head -250 >"$EV/needles-rank$rk.txt"
      done
      curl -sS --max-time 10 http://127.0.0.1:8000/metrics >"$EV/metrics-end.txt" 2>&1 ;;
    *) log "unknown stage $st" ;;
  esac
done
