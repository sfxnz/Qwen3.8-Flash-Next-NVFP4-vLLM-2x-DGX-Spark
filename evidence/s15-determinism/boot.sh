#!/usr/bin/env bash
# S1.5 bisect boot: stop the serve, then run.sh with U1 settings plus the arm's overrides.
#   evidence/s15-determinism/boot.sh <arm> [VAR=value ...]
# U1 = v0.30.0 by digest, no overlays, UTIL 0.76, memguard on, EXTRA_ENV below, per-request metrics.
# Refuses to boot if spark1 MemAvailable < 6 GiB or swap used > 10 GiB after the stop.
set -euo pipefail
cd "$(dirname "$0")/../.."
arm="$1"; shift
log=~/projects/data/qwen38-evals/runs/s15/"$arm"
mkdir -p "$log"
export IMAGE="vllm/vllm-openai:v0.30.0-aarch64@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56"
export OVERLAYS=auto
export EXTRA_ENV="VLLM_PLE_CPU_OFFLOAD=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0"
export EXTRA_ARGS="--per-request-spec-decode-metrics summary"
for kv in "$@"; do export "${kv?}"; done
env | grep -E '^(IMAGE|OVERLAYS|EXTRA_ENV|EXTRA_ARGS|SPEC|DIAGNOSTIC|ENFORCE_EAGER)=' | sort | tee "$log/boot-env.txt"
./stop.sh
avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
swap=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)
echo "spark1 after stop: avail ${avail} MiB, swap used ${swap} MiB" | tee -a "$log/boot-env.txt"
if (( avail < 6144 || swap > 10240 )); then echo "memcheck FAIL; not booting" >&2; exit 3; fi
start=$(date -u +%s)
./run.sh > "$log/run.log" 2>&1
echo "boot_s=$(( $(date -u +%s) - start ))" | tee -a "$log/boot-env.txt"
docker logs qwen38-flash-next-nvfp4 > "$log/rank0.log" 2>&1 || true
