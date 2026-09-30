#!/usr/bin/env bash
# Session 4 boot: stop the serve, memcheck, then ./run.sh with the recipe defaults plus any VAR=value overrides.
#   evidence/l1-l6-k5-l2/boot.sh SUBDIR [VAR=value ...]
# Refuses to boot if spark1 MemAvailable < 6 GiB or swap used > 10 GiB after the stop.
set -euo pipefail
cd "$(dirname "$0")/../.."
sub="$1"; shift
ev="evidence/l1-l6-k5-l2/$sub"
mkdir -p "$ev"
for kv in "$@"; do export "${kv?}"; done
{ echo "overrides: $*"; git rev-parse HEAD; VALIDATE_ONLY=1 ./run.sh; } | tee "$ev/boot-env.txt"
./stop.sh || true
{ echo "== spark1 after stop"; free -h; echo "== spark2"; ssh spark2 free -h; } > "$ev/free-preboot.txt" 2>&1
avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
swap=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)
echo "spark1 after stop: avail ${avail} MiB, swap used ${swap} MiB" | tee -a "$ev/boot-env.txt"
if (( avail < 6144 || swap > 10240 )); then echo "memcheck FAIL; not booting" >&2; exit 3; fi
start=$(date -u +%s)
date -u +%FT%TZ > "$ev/boot-start.txt"
rc=0; ./run.sh > "$ev/run.log" 2>&1 || rc=$?
echo "run_rc=$rc boot_s=$(( $(date -u +%s) - start ))" | tee -a "$ev/boot-env.txt"
{ echo "== spark1 postboot"; free -h; echo "== spark2"; ssh spark2 free -h; } > "$ev/free-postboot.txt" 2>&1
exit "$rc"
