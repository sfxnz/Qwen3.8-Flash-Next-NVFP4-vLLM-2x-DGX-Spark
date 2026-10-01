#!/usr/bin/env bash
# Session 7 per-boot step runner (K3 GDN lazy state commit A/B vs the lossless default). Run from the recipe worktree root.
#   evidence/k3/steps.sh SUBDIR STAGE...
# Evidence goes to evidence/k3/SUBDIR, raw data to ~/projects/data/qwen38-evals/runs/session7/SUBDIR.
# Stages: longid soak gate ruler-anchor ruler-noanchor diverse sampled t1g identity t1full t1d t1d-c8 t2full t1nll t2q t2 alias t3 vision vbench
#         prefix toolstream mixed receipts
# Env: BASE=SUBDIR of the reference arm for identity / t1 / t1d / t2 comparisons (default ref).
# Env: K=num_speculative_tokens for ruler_steps (default 3); ANCHOR (gate and ruler-anchor, default 53.8 55.8; 0 0 = anchor-free).
# Env: ANCHOR="LO HI" for ruler-anchor; T3_LENGTHS (default 4096 16384 32768 65536); MIXED_CELLS (default base A B).
set -uo pipefail
SUB="$1"; shift
ROOT="$(pwd)"
EV="$ROOT/evidence/k3/$SUB"
R="$HOME/projects/data/qwen38-evals/runs/session7/$SUB"
URL=http://127.0.0.1:8000/v1/chat/completions
REFR="$HOME/projects/data/qwen38-evals/runs/session7/${BASE:-ref}"
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
memsummary() {
  awk -F'\t' 'NR>1 && $2!=""{if(m1==""||$2<m1)m1=$2; if($3>s1)s1=$3; if(m2==""||$4<m2)m2=$4; if($5>s2)s2=$5}
    END{printf "spark1 min_avail_mb=%s max_swap_used_mb=%s | spark2 min_avail_mb=%s max_swap_used_mb=%s\n",m1,s1,m2,s2}' "$EV/mem-$1.tsv"
}
alive() {  # memguard trip or crash: the serve is gone
  [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8000/health)" == 200 ]] && return 0
  log "SERVE DOWN after $1 (memguard log: $(tail -1 "$ROOT/.run-state/memguard.log" 2>/dev/null))"; return 1
}

for st in "$@"; do
  case "$st" in
    gate)  # smokes, T0 probe, anchored ruler (ANCHOR, default 60.7 62.7), frozen bench_decode
      freeboth pre-gate
      p0=$(preempt)
      PROBE_ARGS="${PROBE_ARGS:---sizes 2 3 5 8}" RULER_ARGS="--anchor-ms ${ANCHOR:-53.8 55.8} --k ${K:-3}" tools/session_gate.sh "$EV/gate" >"$EV/gate.out" 2>&1
      echo $? >"$EV/gate.exit"; log "gate rc=$(cat "$EV/gate.exit") preemptions_delta=$(( $(preempt) - p0 ))" ;;
    ruler-anchor)
      # shellcheck disable=SC2086
      run ruler-anchor python3 ruler_steps.py --url "$URL" --anchor-ms ${ANCHOR:?set ANCHOR="LO HI"} --json "$EV/ruler-anchor.json" ;;
    ruler-noanchor)
      run ruler-noanchor python3 ruler_steps.py --url "$URL" --anchor-ms 0 0 --k ${K:-3} --json "$EV/ruler-noanchor.json" ;;
    diverse)
      run diverse python3 bench_diverse.py --url "$URL" --concurrency 1 8 --out "$EV/diverse.jsonl" --label "$SUB" ;;
    sampled)
      run sampled python3 bench_sampled.py --url "$URL" --profiles A T1 --concurrency 1 8 --prompts 8 --reps 1 \
        --out "$EV/sampled.jsonl" --label "$SUB" ;;
    t1g)
      run t1g python3 quality/collect_t1g.py --url "$URL" --out "$R/t1g" --limit 10 --repeats 5
      run t1g-self python3 quality/score.py t1g "$R/t1g" "$R/t1g" --json "$EV/t1g-self.json" ;;
    t1full)  # T1-B: teacher-forced prompt_logprobs over the full frozen corpus (pieces <= 1000 tokens), class B vs BASE
      memok t1full || continue
      run t1full python3 quality/collect_t1.py --url "$URL" --out "$R/t1"
      cp "$R/t1/summary.json" "$EV/t1-summary.json" 2>/dev/null
      [[ "$SUB" == "${BASE:-ref}" ]] || run t1full-score python3 quality/score.py t1 "$REFR/t1" "$R/t1" --class B --json "$EV/t1-vs-ref.json"
      alive t1full ;;
    t1long)  # the 8 x 16k `long` docs sent whole: multi-chunk prefill, the only T1 rows where carried GDN state
             # matters (<=1000-token pieces prefill in one chunk). Transient ~4.8 GB at a 4800-token chunk.
      memok t1long || continue
      run t1long python3 quality/collect_t1.py --url "$URL" --out "$R/t1-long" --domains long_whole --long-whole
      freeboth post-t1long
      if [[ -d "$REFR/t1-long" && "$SUB" != "${BASE:-ref}" ]]; then
        run t1long-score python3 quality/score.py t1 "$REFR/t1-long" "$R/t1-long" --class B --json "$EV/t1long-vs-ref.json"
      fi
      alive t1long ;;
    t1d)  # T1-D: 40 x 1024 greedy with top-20 logprobs at c=1; class B vs BASE with the c=8 floor
      run t1d python3 quality/collect_t1d.py --url "$URL" --out "$R/t1d"
      [[ "$SUB" == "${BASE:-ref}" ]] || run t1d-score python3 quality/score.py t1d "$REFR/t1d" "$R/t1d" --class B \
        --floor "$ROOT/evidence/k-sweep-precision/${BASE:-ref}/t1d-floor.json" --json "$EV/t1d-vs-ref.json" ;;
    t1d-c8)  # golden-vs-golden c=8 floor for T1-D (reference arm only)
      run t1d-c8 python3 quality/collect_t1d.py --url "$URL" --out "$R/t1d-c8" --workers 8
      run t1d-floor python3 quality/score.py t1d "$R/t1d" "$R/t1d-c8" --json "$EV/t1d-floor.json" ;;
    t2full)  # T2 at c=1 (bit-exact on this base, so every discordant pair is the lever): paired vs BASE
      run t2full python3 quality/t2.py --url "$URL" --out "$R/t2" --tasks gsm8k,ifeval,tools,json,rep4,effort --workers 1
      for f in "$R"/t2/*.json; do [[ -f "$f" ]] && cp "$f" "$EV/t2-$(basename "$f")"; done
      [[ "$SUB" == "${BASE:-ref}" ]] || run t2full-score python3 quality/score.py t2 "$REFR/t2" "$R/t2" --json "$EV/t2-vs-ref.json" ;;
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
      for L in ${T3_LENGTHS:-4096 16384 32768 65536}; do
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
    identity)  # greedy T1-G output identity vs the baseline arm, per prompt (text sha of token ids)
      run identity python3 "$ROOT/evidence/k-sweep-precision/identity.py" "$HOME/projects/data/qwen38-evals/runs/session7/${BASE:-ref}/t1g" "$R/t1g" \
        --json "$EV/identity.json" ;;
    t1nll)  # T1 teacher-forced NLL, 2 items per domain; delta vs the baseline arm
      memok t1nll || continue
      run t1nll python3 quality/collect_t1.py --url "$URL" --out "$R/t1" --limit 2
      cp "$R/t1/summary.json" "$EV/t1-summary.json" 2>/dev/null
      run t1nll-score python3 quality/score.py t1 "$HOME/projects/data/qwen38-evals/runs/session7/${BASE:-ref}/t1" "$R/t1" --json "$EV/t1-vs-base.json" ;;
    t2q)  # T2 gsm8k-250 + tools + json, paired vs the baseline arm
      run t2q python3 quality/t2.py --url "$URL" --out "$R/t2" --tasks tools,json,gsm8k --workers 4
      for f in "$R"/t2/*.json; do [[ -f "$f" ]] && cp "$f" "$EV/t2-$(basename "$f")"; done
      run t2q-score python3 quality/score.py t2 "$HOME/projects/data/qwen38-evals/runs/session7/${BASE:-ref}/t2" "$R/t2" --json "$EV/t2-vs-base.json" ;;
    longid)  # fixed long greedy prompts (c=1, ignore_eos, 4096 tokens: crosses the 1600/3200 GDN blocks); LONGID_TAG names the file
      memok longid || continue
      run "longid-${LONGID_TAG:-pre}" python3 "$ROOT/evidence/k3/soak.py" longid --out "$R/longid-${LONGID_TAG:-pre}.jsonl"
      cp "$R/longid-${LONGID_TAG:-pre}.jsonl" "$EV/" 2>/dev/null
      if [[ -f "$REFR/longid-pre.jsonl" ]]; then
        run "longid-${LONGID_TAG:-pre}-cmp" python3 "$ROOT/evidence/k3/soak.py" cmp "$REFR/longid-pre.jsonl" "$R/longid-${LONGID_TAG:-pre}.jsonl"
      fi
      alive longid ;;
    soak)  # SOAK_MIN minutes of mixed traffic: 8 long ignore_eos streams (4096 tokens), short requests, a 32k prefill every 4 min
      memok soak || continue
      pid=$(memsampler soak); p0=$(preempt)
      run soak python3 "$ROOT/evidence/k3/soak.py" soak --minutes "${SOAK_MIN:-20}" --out "$R/soak.jsonl" --json "$EV/soak.json"
      kill "$pid" 2>/dev/null
      log "soak preemptions_delta=$(( $(preempt) - p0 )) $(memsummary soak)"
      alive soak ;;
    free)
      freeboth snapshot ;;
    receipts)
      freeboth end
      docker logs qwen38-flash-next-nvfp4 >"$R/docker-rank0.log" 2>&1
      ssh spark2 "docker logs qwen38-flash-next-nvfp4 2>&1" >"$R/docker-rank1.log" 2>&1
      tail -n 300 "$R/docker-rank0.log" >"$EV/docker-rank0-tail.log"
      tail -n 300 "$R/docker-rank1.log" >"$EV/docker-rank1-tail.log"
      for rk in 0 1; do
        grep -nE 'Unknown vLLM|quantization=|MoE backend|Fp8 MoE|FP8 MoE|KV cache|GPU KV|Graph capturing finished|Qwen4ExpPLE|PLE embedding|pinned=|GDN|gdn|FlashInfer|fused_finalize|DETERMINISTIC|w2_weight_scale|AttributeError|max_model_len|Traceback|ERROR|CUDA error|NCCL WARN|Maximum concurrency|Available KV|model weights took|init engine|Application startup|async_scheduling|Asynchronous scheduling|chat template|chat_template|mm_processor|limit_mm|long_prefill|Draft head|L1b|qwen38|hc_fused|K5|nccl twin|NCCL_GRAPH_MIXING|Using configuration from|Using default MoE config|TUNED' \
          "$R/docker-rank$rk.log" | head -300 >"$EV/needles-rank$rk.txt"
      done
      cp "$ROOT/.run-state/memguard.log" "$EV/memguard.log" 2>/dev/null
      curl -sS --max-time 10 http://127.0.0.1:8000/metrics >"$EV/metrics-end.txt" 2>&1 ;;
    *) log "unknown stage $st" ;;
  esac
done
