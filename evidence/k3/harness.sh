#!/usr/bin/env bash
# Session 7 step 1: K3 standalone GPU harness on spark1 (docker/v030/K3.md §6), serve DOWN, one container at a time.
#   evidence/k3/harness.sh STAGE...    stages: selftest exact exact-red sim64 sim1600 bench ncu
# Output: evidence/k3/harness/<stage>.out + .exit. GB10 has no dram__ counters (n/a): ncu also reads L2 (lts) sectors and L1 global ld/st bytes. Env: HARNESS_EXTRA_ENV="-e KEY=VAL ..." (tunable trials).
set -uo pipefail
cd "$(dirname "$0")/../.."
EV=evidence/k3/harness${HARNESS_TAG:+-$HARNESS_TAG}
mkdir -p "$EV"
IMG=vllm/vllm-openai:v0.30.0-aarch64@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$EV/harness.log"; }
guard() {  # serve down, no other GPU container, memory headroom
  if docker ps --format '{{.Image}} {{.Names}}' | grep -v conduit | grep -q .; then log "REFUSE: containers running"; docker ps >>"$EV/harness.log"; exit 3; fi
  local a s
  a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  s=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)
  log "memcheck avail=${a}MB swap_used=${s}MB"
  if (( a < 6144 || s > 10240 )); then log "REFUSE: memcheck"; exit 3; fi
}
R() {  # R NAME ARGS... : one harness container
  local n="$1"; shift
  guard
  log "start $n: $*"
  # shellcheck disable=SC2086
  docker run --rm --gpus all --ipc host --network none --ulimit core=1 --memory 24g ${HARNESS_EXTRA_ENV:-} \
    -v "$PWD":/w -w /w --entrypoint python3 "$IMG" tools/kernels/k3_harness.py "$@" >"$EV/$n.out" 2>&1
  local rc=$?; echo "$rc" >"$EV/$n.exit"; log "end $n rc=$rc"
}
for st in "$@"; do
  case "$st" in
    selftest) R selftest selftest ;;
    exact) R exact exact --conc 1 8 --steps 256 ;;
    exact-red) HARNESS_EXTRA_ENV="${HARNESS_EXTRA_ENV:-} -e VLLM_QWEN38_GDN_LAZY_RED=1" R exact-red exact --conc 1 --steps 32 ;;
    sim64) R sim64 sim --block 64 --nreq 8 --steps 400 ;;
    sim1600) R sim1600 sim --block 1600 --nreq 8 --steps 1500 ;;
    bench) R bench bench --conc 1 8 --abab 4 --reps 50 ;;
    ncu)
      guard; log "start ncu"
      # shellcheck disable=SC2086
      docker run --rm --gpus all --ipc host --network none --ulimit core=1 --memory 24g --cap-add SYS_ADMIN ${HARNESS_EXTRA_ENV:-} \
        -v /opt/nvidia/nsight-compute:/opt/nvidia/nsight-compute:ro -v "$PWD":/w -w /w \
        --entrypoint /opt/nvidia/nsight-compute/2025.3.1/ncu "$IMG" \
        --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,gpu__time_duration.sum \
        -k 'regex:gdn_decode_post_conv_mtp|_gdl_decode_kernel' python3 tools/kernels/k3_harness.py ncu --conc 1 8 \
        >"$EV/ncu.out" 2>&1
      rc=$?; echo "$rc" >"$EV/ncu.exit"; log "end ncu rc=$rc" ;;
    *) log "unknown stage $st" ;;
  esac
done
