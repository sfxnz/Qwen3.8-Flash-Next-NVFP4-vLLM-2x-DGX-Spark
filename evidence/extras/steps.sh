#!/usr/bin/env bash
# Session 6 per-boot step runner (X10 throughput profile, P2-5 step 2, T4 for lab-fp8-dense). Run from the recipe worktree root.
#   evidence/extras/steps.sh SUBDIR STAGE...
# Evidence goes to evidence/extras/SUBDIR, raw data to ~/projects/data/qwen38-evals/runs/session6/SUBDIR.
# Stages: free gate ruler-noanchor diverse16 mixed t4 receipts
# Env: ANCHOR (gate, default 53.8 55.8; 0 0 = anchor-free); PROBE_ARGS (gate: extra bench_probe args, e.g. --sizes 2 3 5 8 16);
#      DIVERSE_C (diverse16, default "1 8 16"); MIXED_CELLS (default base A B); T4_LIMIT (t4, 0 = full 1,319); T4_WORKERS (default 8).
set -uo pipefail
SUB="$1"; shift
ROOT="$(pwd)"
EV="$ROOT/evidence/extras/$SUB"
R="$HOME/projects/data/qwen38-evals/runs/session6/$SUB"
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
preempt() { curl -s --max-time 10 http://127.0.0.1:8000/metrics | awk '/^vllm:num_preemptions_total/{s+=$2}END{print s+0}'; }
# memsampler NAME: background MemAvailable/swap sampler for both nodes every 5 s -> EV/mem-NAME.tsv; echo its pid
memsampler() {
  ( echo -e "ts\tspark1_avail_mb\tspark1_swap_used_mb\tspark2_avail_mb\tspark2_swap_used_mb"
    while :; do
      a=$(awk '/MemAvailable/{a=int($2/1024)}/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print a"\t"int((t-f)/1024)}' /proc/meminfo)
      b=$(ssh -o BatchMode=yes spark2 "awk '/MemAvailable/{a=int(\$2/1024)}/SwapTotal/{t=\$2}/SwapFree/{f=\$2}END{print a\"\t\"int((t-f)/1024)}' /proc/meminfo")
      echo -e "$(date -u +%T)\t$a\t$b"; sleep 5
    done ) >"$EV/mem-$1.tsv" 2>/dev/null &
  echo $!
}
memsummary() {  # min available / max swap used over the sampler file
  awk -F'\t' 'NR>1 && $2!=""{if(m1==""||$2<m1)m1=$2; if($3>s1)s1=$3; if(m2==""||$4<m2)m2=$4; if($5>s2)s2=$5}
    END{printf "spark1 min_avail_mb=%s max_swap_used_mb=%s | spark2 min_avail_mb=%s max_swap_used_mb=%s\n",m1,s1,m2,s2}' "$EV/mem-$1.tsv"
}

for st in "$@"; do
  case "$st" in
    gate)  # smokes, T0 probe (PROBE_ARGS), anchored ruler, frozen bench_decode
      freeboth pre-gate
      p0=$(preempt)
      PROBE_ARGS="${PROBE_ARGS:-}" RULER_ARGS="--anchor-ms ${ANCHOR:-53.8 55.8} --k 3" tools/session_gate.sh "$EV/gate" >"$EV/gate.out" 2>&1
      echo $? >"$EV/gate.exit"; log "gate rc=$(cat "$EV/gate.exit") preemptions_delta=$(( $(preempt) - p0 ))" ;;
    ruler-noanchor)
      run ruler-noanchor python3 ruler_steps.py --url "$URL" --anchor-ms 0 0 --k 3 --json "$EV/ruler-noanchor.json" ;;
    diverse16)  # 32 distinct prompts per category in waves of c; preemptions and host memory under load
      memok diverse16 || continue
      pid=$(memsampler diverse16); p0=$(preempt)
      # shellcheck disable=SC2086
      run diverse16 python3 bench_diverse.py --url "$URL" --concurrency ${DIVERSE_C:-1 8 16} --out "$EV/diverse16.jsonl" --label "$SUB"
      kill "$pid" 2>/dev/null
      log "diverse16 preemptions_delta=$(( $(preempt) - p0 )) $(memsummary diverse16)"
      alive diverse16 ;;
    mixed)  # arm B carries a 32k then a 128k prefill
      memok mixed || continue
      pid=$(memsampler mixed)
      # shellcheck disable=SC2086
      run mixed python3 bench_mixed.py --url "$URL" --cells ${MIXED_CELLS:-base A B} --out "$EV/mixed.jsonl" --label "$SUB"
      kill "$pid" 2>/dev/null
      log "mixed $(memsummary mixed)"
      alive mixed ;;
    t4)  # full GSM8K, thinking on, T=0.6, seed + item index; identical settings on every arm
      memok t4 || continue
      pid=$(memsampler t4)
      run t4 python3 quality/t4_gsm8k_full.py --url "$URL" --out "$R/t4" --workers "${T4_WORKERS:-8}" --limit "${T4_LIMIT:-0}"
      kill "$pid" 2>/dev/null
      cp "$R/t4/t4.json" "$EV/t4.json" 2>/dev/null
      log "t4 $(memsummary t4)"
      alive t4 ;;
    free)
      freeboth snapshot ;;
    receipts)
      freeboth end
      docker logs qwen38-flash-next-nvfp4 >"$R/docker-rank0.log" 2>&1
      ssh spark2 "docker logs qwen38-flash-next-nvfp4 2>&1" >"$R/docker-rank1.log" 2>&1
      tail -n 300 "$R/docker-rank0.log" >"$EV/docker-rank0-tail.log"
      tail -n 300 "$R/docker-rank1.log" >"$EV/docker-rank1-tail.log"
      for rk in 0 1; do
        grep -nE 'Unknown vLLM|quantization=|MoE backend|Fp8 MoE|FP8 MoE|KV cache|GPU KV|Graph capturing finished|pinned=|FlashInfer GDN|max_model_len|max_num_seqs|max_num_batched_tokens|long_prefill|Traceback|ERROR|CUDA error|NCCL WARN|Maximum concurrency|Available KV|model weights took|Application startup|Draft head|FP8_DENSE|fp8_block|preempt' \
          "$R/docker-rank$rk.log" | head -300 >"$EV/needles-rank$rk.txt"
      done
      cp "$ROOT/.run-state/memguard.log" "$EV/memguard.log" 2>/dev/null
      curl -sS --max-time 10 http://127.0.0.1:8000/metrics >"$EV/metrics-end.txt" 2>&1 ;;
    *) log "unknown stage $st" ;;
  esac
done
