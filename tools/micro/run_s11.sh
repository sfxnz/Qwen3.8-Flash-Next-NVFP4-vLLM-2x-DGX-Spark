#!/usr/bin/env bash
# S1.1 microbench session (plan 0-9): vLLM DOWN, exclusive GPUs on both Sparks.
# Runs every S1.1 measurement inside the base image(s), then writes the go/no-go
# table (tools/micro/gonogo.py) into the output dir.
#
# Per image tag in IMAGES (pin = the recipe pin, v030 = vllm/vllm-openai v0.30.0):
#   device     device_facts.py on head (and worker)            deviceQuery facts
#   gemm       gemm_sweep.py gemm, plus a CUBLASLT_LOG_LEVEL=5 algo pass
#   drafthead  gemm_sweep.py drafthead                          L1b probe
#   hc         hc_bench.py                                      HC chain cold/hot
#   fp8        fp8_sweep.py                                     W8A8 CUTLASS vs Marlin W8A16
#   moe        moe_nvfp4.py                                     NVFP4 MoE TP2 vs EP (best effort)
#   nccl       nccl_decode_sweep.py, 2 nodes, arms keep/mix0 x REPS (ABAB order)
#   allreduce  nccl_allreduce_sweep.py, 2 nodes, arms single/dual
# Single-GPU steps run on SINGLE_HOSTS (default: head only).
#
# Usage: tools/micro/run_s11.sh [--dry-run]
#        IMAGES=pin STEPS="gemm hc" OUT=dir tools/micro/run_s11.sh
#        --dry-run prints every command and touches no GPU, docker or ssh.
# Then:  python3 tools/micro/gonogo.py $OUT   (run automatically at the end)
set -euo pipefail

MICRO="$(cd "$(dirname "$0")" && pwd)"
DRY=0
case "${1:-}" in
  --dry-run) DRY=1 ;;
  -h | --help) sed -n '2,20p' "$0"; exit 0 ;;
  "") ;;
  *) echo "unknown argument: $1 (use --dry-run or --help)" >&2; exit 2 ;;
esac

IMAGE_PIN="${IMAGE_PIN:-vllm/vllm-openai:nightly-aarch64@sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b}"
IMAGE_V030="${IMAGE_V030:-vllm/vllm-openai:v0.30.0-aarch64@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56}"
IMAGES="${IMAGES:-pin v030}"
NCCL_IMAGES="${NCCL_IMAGES:-$IMAGES}"
STEPS="${STEPS:-device gemm drafthead hc fp8 moe nccl allreduce}"
SINGLE_HOSTS="${SINGLE_HOSTS:-head}"  # "head worker" also runs the single-GPU steps on the worker, after the head
WORKER_HOST="${WORKER_HOST:-spark2}"
HEAD_IP="${HEAD_IP:-10.100.8.1}"
IFACE="${IFACE:-enp1s0f1np1}"
PORT0="${MICRO_PORT:-29741}"
REPS="${REPS:-2}"
TIMEOUT_S="${TIMEOUT_S:-1800}"
SMI_MS="${SMI_MS:-1000}"  # clocks/power sampling during every step; 0 = off
OUT="${OUT:-$HOME/projects/data/qwen38-s11/$(date +%Y%m%d-%H%M)}"
CACHE="${CACHE:-$HOME/projects/data/qwen38-s11/cache}"
RDIR=/tmp/qwen38-s11
PY=(-S)  # python3 -S; PYTHONPATH carries dist-packages (sibling convention)

image_ref() {
  case "$1" in
    pin) printf '%s\n' "$IMAGE_PIN" ;;
    v030) printf '%s\n' "$IMAGE_V030" ;;
    *) echo "unknown image tag $1 (pin|v030)" >&2; return 1 ;;
  esac
}
for tag in $IMAGES $NCCL_IMAGES; do
  ref="$(image_ref "$tag")"
  [[ "$ref" == *@sha256:* ]] || { echo "refusing: $tag image $ref is not digest-pinned" >&2; exit 1; }
done

# x CMD...: run, or print it under --dry-run.
x() {
  if ((DRY)); then printf '+'; printf ' %q' "$@"; printf '\n'; else "$@"; fi
}
# xlog LOG CMD...: run with stdout+stderr into LOG; returns the command's rc.
xlog() {
  local lf="$1"
  shift
  if ((DRY)); then printf '+'; printf ' %q' "$@"; printf ' > %q 2>&1\n' "$lf"; else "$@" >"$lf" 2>&1; fi
}
log() { printf '== %s\n' "$*"; }
has_step() { [[ " $STEPS " == *" $1 "* ]]; }
on_worker() { [[ " $SINGLE_HOSTS " == *" worker "* ]]; }

# Exclusive GPUs: the run.sh rule (any container holding GPUs or InfiniBand).
gpu_holders() {  # $1 = "" (local) or ssh host
  local pre=() name devices
  [[ -n "$1" ]] && pre=(ssh -o BatchMode=yes "$1")
  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    devices="$("${pre[@]}" docker inspect -f '{{json .HostConfig.DeviceRequests}} {{json .HostConfig.Devices}}' "$name" 2>/dev/null || true)"
    if printf '%s' "$devices" | grep -Eqi 'gpu|nvidia|infiniband'; then printf '%s ' "$name"; fi
  done < <("${pre[@]}" docker ps --format '{{.Names}}')
}

# docker run args for one bench container into DARGS.
# $1 name, $2 bench dir, $3 out dir, $4 cache dir, $5 image tag; rest KEY=VAL env.
DARGS=()
dargs() {
  local name="$1" bench="$2" out="$3" cache="$4" tag="$5"
  shift 5
  DARGS=(--rm --init --name "$name" --gpus all --network host --ipc host --shm-size 16g
    --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1
    --user "$(id -u):$(id -g)" --entrypoint python3
    -e HOME=/tmp -e XDG_CACHE_HOME=/cache -e TRITON_CACHE_DIR=/cache/triton
    -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages
    -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" -e NCCL_DEBUG=WARN
    -e MICRO_IMAGE="$(image_ref "$tag")"
    -v "$bench:/bench:ro" -v "$out:/out" -v "$cache:/cache")
  local kv
  for kv in "$@"; do DARGS+=(-e "$kv"); done
}

SMI_Q="nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader -lms $SMI_MS"
smi_start() {  # $1 csv path on the host that runs it, $2 "" (head) or ssh host
  ((SMI_MS > 0)) || return 0
  if ((DRY)); then printf '+ [%s] %s > %s &\n' "${2:-head}" "$SMI_Q" "$1"; return 0; fi
  if [[ -z "$2" ]]; then
    $SMI_Q >"$1" 2>/dev/null &
    echo $! >"$OUT/.smi.pid"
  else
    ssh -o BatchMode=yes "$2" "mkdir -p \$(dirname $1) && { nohup $SMI_Q > $1 2>/dev/null </dev/null & echo \$! > $RDIR/smi.pid; }" || true
  fi
}
smi_stop() {  # both hosts; harmless when one never started
  if ((DRY)) || ((SMI_MS == 0)); then return 0; fi
  kill "$(cat "$OUT/.smi.pid" 2>/dev/null)" 2>/dev/null || true
  ssh -o BatchMode=yes "$WORKER_HOST" "kill \$(cat $RDIR/smi.pid 2>/dev/null) 2>/dev/null; true" || true
}

# One single-GPU bench. $1 tag, $2 head|worker, $3 output stem, $4 script; rest script args.
# Extra container env: BENCH_ENV array. Head writes $OUT/$tag/STEM.json, the worker STEM.$WORKER_HOST.json.
BENCH_ENV=()
single() {
  local tag="$1" where="$2" stem="$3" script="$4"
  shift 4
  local ref d cmd
  ref="$(image_ref "$tag")"
  d="$OUT/$tag"
  log "$tag $where: $script $*"
  if [[ "$where" == head ]]; then
    dargs "s11-$stem" "$MICRO" "$d" "$CACHE/$tag" "$tag" "${BENCH_ENV[@]}"
    smi_start "$d/$stem.smi.csv" ""
    xlog "$d/$stem.log" timeout -k 10 "$TIMEOUT_S" docker run "${DARGS[@]}" "$ref" "${PY[@]}" "/bench/$script" "$@" \
      --json "/out/$stem.json" || echo "  FAILED ($script, see $d/$stem.log)"
  else
    dargs "s11-$stem" "$RDIR/micro" "$RDIR/out/$tag" "$RDIR/cache/$tag" "$tag" "${BENCH_ENV[@]}"
    smi_start "$RDIR/out/$tag/$stem.smi.csv" "$WORKER_HOST"
    cmd="$(printf '%q ' docker run "${DARGS[@]}" "$ref" "${PY[@]}" "/bench/$script" "$@" --json "/out/$stem.json")"
    xlog "$d/$stem.$WORKER_HOST.log" timeout -k 10 "$TIMEOUT_S" ssh -o BatchMode=yes "$WORKER_HOST" \
      "mkdir -p $RDIR/out/$tag $RDIR/cache/$tag && $cmd" || echo "  FAILED on $WORKER_HOST ($script)"
    x scp -q "$WORKER_HOST:$RDIR/out/$tag/$stem.json" "$d/$stem.$WORKER_HOST.json" || true
    smi_stop
    ((DRY)) || scp -q "$WORKER_HOST:$RDIR/out/$tag/$stem.smi.csv" "$d/$stem.smi.$WORKER_HOST.csv" 2>/dev/null || true
  fi
  smi_stop
  return 0
}

# One two-node run: rank 1 on WORKER_HOST in the background, rank 0 here.
# $1 tag, $2 script (also the arm-env source), $3 arm, $4 out subdir under $OUT/$tag, $5 file stem; rest args.
N=0
pair() {
  local tag="$1" script="$2" arm="$3" sub="$4" stem="$5"
  shift 5
  local ref d port env_kv=() kv wcmd rc0=0 rc1=0 wpid=""
  ref="$(image_ref "$tag")"
  d="$OUT/$tag/$sub"
  N=$((N + 1))
  port=$((PORT0 + N % 50))
  while IFS= read -r kv; do [[ -n "$kv" ]] && env_kv+=("$kv"); done < <(python3 "$MICRO/$script" arm-env "$arm")
  log "$tag $script arm=$arm port=$port env: ${env_kv[*]}"
  ((DRY)) || mkdir -p "$d"
  dargs "s11-r1" "$RDIR/micro" "$RDIR/out/$tag" "$RDIR/cache/$tag" "$tag" "${env_kv[@]}"
  wcmd="$(printf '%q ' docker run "${DARGS[@]}" "$ref" "${PY[@]}" "/bench/$script" run --rank 1 \
    --master "$HEAD_IP:$port" --arm "$arm" "$@" --json "/out/$stem.rank1.json")"
  smi_start "$d/$stem.smi.spark1.csv" ""
  smi_start "$RDIR/out/$tag/smi.csv" "$WORKER_HOST"
  if ((DRY)); then
    xlog "$d/$stem.rank1.log" ssh -o BatchMode=yes "$WORKER_HOST" "mkdir -p $RDIR/out/$tag $RDIR/cache/$tag && $wcmd"
  else
    timeout -k 10 "$TIMEOUT_S" ssh -o BatchMode=yes "$WORKER_HOST" "mkdir -p $RDIR/out/$tag $RDIR/cache/$tag && $wcmd" \
      >"$d/$stem.rank1.log" 2>&1 &
    wpid=$!
  fi
  dargs "s11-r0" "$MICRO" "$d" "$CACHE/$tag" "$tag" "${env_kv[@]}"
  xlog "$d/$stem.rank0.log" timeout -k 10 "$TIMEOUT_S" docker run "${DARGS[@]}" "$ref" "${PY[@]}" "/bench/$script" run \
    --rank 0 --master "$HEAD_IP:$port" --arm "$arm" "$@" --json "/out/$stem.rank0.json" || rc0=$?
  [[ -z "$wpid" ]] || wait "$wpid" || rc1=$?
  smi_stop
  ((DRY)) && return 0
  scp -q "$WORKER_HOST:$RDIR/out/$tag/smi.csv" "$d/$stem.smi.spark2.csv" 2>/dev/null || true
  scp -q "$WORKER_HOST:$RDIR/out/$tag/$stem.rank1.json" "$d/$stem.rank1.json" 2>/dev/null || true
  if [[ $rc0 != 0 || $rc1 != 0 ]]; then
    echo "  FAILED rc0=$rc0 rc1=$rc1 (see $d/$stem.rank*.log)"
    docker rm -f s11-r0 >/dev/null 2>&1 || true
    ssh -o BatchMode=yes "$WORKER_HOST" "docker rm -f s11-r1 >/dev/null 2>&1; true"
  fi
  return 0
}

# ------------------------------------------------------------------ preflight
log "S1.1 microbench: OUT=$OUT IMAGES=[$IMAGES] NCCL_IMAGES=[$NCCL_IMAGES] STEPS=[$STEPS] REPS=$REPS"
if ((DRY)); then
  log "dry-run: would refuse if any GPU/InfiniBand container runs on this host or $WORKER_HOST"
else
  held="$(gpu_holders "")$(gpu_holders "$WORKER_HOST")"
  if [[ -n "$held" ]]; then
    echo "refusing: GPU holders up ($held). Stop the serve (./stop.sh) first; S1.1 needs exclusive GPUs." >&2
    exit 1
  fi
  for tag in $IMAGES $NCCL_IMAGES; do
    ref="$(image_ref "$tag")"
    docker image inspect "$ref" >/dev/null 2>&1 || { echo "missing image on head: $ref" >&2; exit 1; }
  done
  for tag in $NCCL_IMAGES; do
    ref="$(image_ref "$tag")"
    ssh -o BatchMode=yes "$WORKER_HOST" "docker image inspect '$ref' >/dev/null 2>&1" \
      || { echo "missing image on $WORKER_HOST: $ref" >&2; exit 1; }
  done
  mkdir -p "$OUT"
  for tag in $IMAGES $NCCL_IMAGES; do mkdir -p "$OUT/$tag" "$CACHE/$tag"; done
  {
    echo "date=$(date -u +%FT%TZ)"
    echo "git=$(git -C "$MICRO" rev-parse HEAD 2>/dev/null || echo none)"
    echo "image_pin=$IMAGE_PIN"
    echo "image_v030=$IMAGE_V030"
    echo "steps=$STEPS reps=$REPS single_hosts=$SINGLE_HOSTS worker=$WORKER_HOST head_ip=$HEAD_IP iface=$IFACE"
    echo "--- free -h head"; free -h
    echo "--- free -h $WORKER_HOST"; ssh -o BatchMode=yes "$WORKER_HOST" free -h || true
    echo "--- ibdev2netdev head"; ibdev2netdev 2>/dev/null || true
    echo "--- ibdev2netdev $WORKER_HOST"; ssh -o BatchMode=yes "$WORKER_HOST" "ibdev2netdev 2>/dev/null" || true
  } >"$OUT/session.txt"
fi
x ssh -o BatchMode=yes "$WORKER_HOST" "rm -rf $RDIR/micro && mkdir -p $RDIR/micro $RDIR/out"
x scp -q "$MICRO"/*.py "$WORKER_HOST:$RDIR/micro/"

# ------------------------------------------------------------ single-GPU steps
for tag in $IMAGES; do
  hosts=(head)
  on_worker && hosts+=(worker)
  for where in "${hosts[@]}"; do
    has_step device && single "$tag" "$where" device device_facts.py
    if has_step gemm; then
      BENCH_ENV=(CUBLASLT_LOG_LEVEL=5 CUBLASLT_LOG_FILE=/out/cublaslt.%i.log)
      single "$tag" "$where" gemm-algo gemm_sweep.py gemm --algo-log
      BENCH_ENV=()
      single "$tag" "$where" gemm gemm_sweep.py gemm
    fi
    has_step drafthead && single "$tag" "$where" drafthead gemm_sweep.py drafthead
    has_step hc && single "$tag" "$where" hc hc_bench.py run
    has_step fp8 && single "$tag" "$where" fp8 fp8_sweep.py run
    has_step moe && single "$tag" "$where" moe moe_nvfp4.py run
  done
done

# ----------------------------------------------------------- two-node steps
for tag in $NCCL_IMAGES; do
  if has_step nccl; then
    for ((rep = 1; rep <= REPS; rep++)); do  # reps outside arms: drift spreads over both arms
      for arm in $(python3 "$MICRO/nccl_decode_sweep.py" arms); do
        pair "$tag" nccl_decode_sweep.py "$arm" "nccl-decode/$arm" "rep$rep"
      done
    done
    ((DRY)) || python3 "$MICRO/nccl_decode_sweep.py" summarize "$OUT/$tag/nccl-decode" --out "$OUT/$tag/nccl-decode.json" \
      || echo "  nccl summarize failed"
  fi
  if has_step allreduce; then
    for arm in $(python3 "$MICRO/nccl_allreduce_sweep.py" arms); do
      pair "$tag" nccl_allreduce_sweep.py "$arm" allreduce "$arm"
      ((DRY)) || { [[ -f "$OUT/$tag/allreduce/$arm.rank0.json" ]] && cp "$OUT/$tag/allreduce/$arm.rank0.json" "$OUT/$tag/allreduce/$arm.json"; } || true
    done
  fi
done

# ------------------------------------------------------------------ go/no-go
if ((DRY)); then
  x python3 "$MICRO/gonogo.py" "$OUT"
else
  python3 "$MICRO/gonogo.py" "$OUT" >/dev/null && log "go/no-go table: $OUT/gonogo.md"
fi
log "results in $OUT"
