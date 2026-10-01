#!/usr/bin/env bash
# Per-node 1 Hz telemetry sampler and per-boot receipts (plan 0-4; G15, G16, G17, F26, F27).
#
#   tools/telemetry.sh start   EVDIR   start samplers on this node and WORKER_HOST
#   tools/telemetry.sh stop    EVDIR   stop them
#   tools/telemetry.sh receipt EVDIR   per-boot static receipts from both nodes
#   tools/telemetry.sh sample  [OUT|-] run one sampler in the foreground (used by start)
#
# Samplers run under `taskset -c 0 nice -n 10`. Output is long-format CSV
# (ts_unix,ts_iso,node,key,value), flushed and fsync'd every second. The
# worker's sampler streams over ssh into a file on this node, so the last
# seconds before a spark2 power cut survive. Nothing is written on spark2.
#
# GPU: nvidia-smi --query-gpu with clock, power, temperature, utilization.gpu
# and clocks_event_reasons fields only. Never memory.* (AGENTS.md: read unified
# memory with free -h).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
WORKER_HOST="${WORKER_HOST:-spark2}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next-nvfp4}"
PORT="${PORT:-8000}"
METRICS_URL="${METRICS_URL:-http://127.0.0.1:${PORT}/metrics}"
NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f1}"
INTERVAL_S="${INTERVAL_S:-1}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8)

usage() {
  sed -n '2,9p' "$SELF" | sed 's/^# \{0,1\}//'
  exit 2
}

sampler_py() {
  cat <<'PY'
import os, subprocess, sys, threading, time, datetime, urllib.request, socket, glob

out_path, node, metrics_url, hca, interval = sys.argv[1:6]
interval = float(interval)
fh = sys.stdout if out_path == "-" else open(out_path, "a", buffering=1)
lock = threading.Lock()
children = []

def shutdown(*_):
    # Children (nvidia-smi, journalctl) must not outlive the sampler, above all
    # on the worker, where killing the local ssh only closes our stdout.
    for c in children:
        try:
            c.terminate()
        except OSError:
            pass
    os._exit(0)

import signal
for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
    signal.signal(sig, shutdown)

def iso(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat(timespec="milliseconds")

def emit(rows, t=None):
    t = time.time() if t is None else t
    s = iso(t)
    with lock:
        try:
            for k, v in rows:
                fh.write(f"{t:.3f},{s},{node},{k},{v}\n")
            fh.flush()
        except (BrokenPipeError, ValueError, OSError):
            shutdown()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass  # stdout pipe: the receiving side fsyncs

def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None

GPU_FIELDS = [
    "pstate", "clocks.sm", "clocks.gr", "clocks.mem", "clocks.max.sm", "power.draw",
    "temperature.gpu", "utilization.gpu",
    "clocks_event_reasons.active", "clocks_event_reasons.gpu_idle",
    "clocks_event_reasons.applications_clocks_setting", "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.hw_slowdown", "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown", "clocks_event_reasons.hw_power_brake_slowdown",
    "clocks_event_reasons.sync_boost",
]

def gpu_thread():
    cmd = ["nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits",
           "-lms", str(int(interval * 1000))]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        children.append(p)
    except OSError as e:
        emit([("gpu.error", str(e).replace(",", ";"))])
        return
    for line in p.stdout:
        vals = [v.strip() for v in line.split(",")]
        if len(vals) != len(GPU_FIELDS):
            continue
        emit([("gpu." + f, v.replace(" ", "")) for f, v in zip(GPU_FIELDS, vals)])

def journal_thread():
    try:
        p = subprocess.Popen(["journalctl", "-f", "-n", "0", "-o", "short-iso-precise"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        children.append(p)
    except OSError:
        return
    for line in p.stdout:
        emit([("journal", '"' + line.rstrip("\n").replace('"', "'") + '"')])

def vmstat():
    out = {}
    for line in (read("/proc/vmstat") or "").splitlines():
        k, _, v = line.partition(" ")
        if k in ("pswpin", "pswpout", "pgmajfault") or k.startswith("allocstall"):
            out[k] = int(v)
    return out

def meminfo():
    out = {}
    for line in (read("/proc/meminfo") or "").splitlines():
        k, _, v = line.partition(":")
        if k in ("MemAvailable", "SwapFree", "SwapTotal", "MemFree"):
            out[k] = int(v.split()[0])
    return out

def psi():
    out = {}
    for res in ("cpu", "memory", "io"):
        for line in (read(f"/proc/pressure/{res}") or "").splitlines():
            parts = line.split()
            kind = parts[0]
            kv = dict(x.split("=") for x in parts[1:])
            out[f"psi.{res}.{kind}.avg10"] = kv.get("avg10")
            out[f"psi.{res}.{kind}.total_us"] = int(kv.get("total", 0))
    return out

def cpustat():
    out = {}
    for line in (read("/proc/stat") or "").splitlines():
        if line.startswith("cpu") and line[3:4].isdigit():
            f = line.split()
            vals = list(map(int, f[1:]))
            out[f[0]] = (sum(vals), vals[3] + vals[4])  # total, idle+iowait
    return out

THREAD_WORDS = ("VLLM", "EngineCor", "Worker", "APIServer", "NCCL", "python")
vllm_pids = []
last_scan = 0.0

def scan_pids():
    pids = []
    for d in glob.glob("/proc/[0-9]*"):
        comm = read(d + "/comm") or ""
        if comm.startswith("VLLM") or comm in ("vllm",):
            pids.append(int(d.rsplit("/", 1)[1]))
            continue
        if comm.startswith("python"):
            cmd = (read(d + "/cmdline") or "").replace("\0", " ")
            if "vllm" in cmd and "serve" in cmd:
                pids.append(int(d.rsplit("/", 1)[1]))
    return pids

def threads():
    out = {}
    for pid in vllm_pids:
        for t in glob.glob(f"/proc/{pid}/task/[0-9]*"):
            st = read(t + "/stat")
            if not st:
                continue
            comm = st[st.index("(") + 1: st.rindex(")")]
            rest = st[st.rindex(")") + 2:].split()
            # rest[0] is field 3; utime=14, stime=15, processor=39.
            cpu = int(rest[11]) + int(rest[12])
            proc = rest[36]
            tid = t.rsplit("/", 1)[1]
            out[f"{pid}/{tid}/{comm.replace(',', ';').replace(' ', '_')}"] = (cpu, proc)
    return out

def roce():
    out = {}
    base = f"/sys/class/infiniband/{hca}/ports/1"
    for f in glob.glob(base + "/hw_counters/*") + [base + "/counters/port_xmit_data", base + "/counters/port_rcv_data"]:
        v = read(f)
        if v is not None and v.lstrip("-").isdigit():
            out["roce." + os.path.basename(f)] = int(v)
    return out

def metrics():
    if metrics_url in ("", "none"):
        return {}
    try:
        with urllib.request.urlopen(metrics_url, timeout=2) as r:
            text = r.read().decode("utf-8", "replace")
    except OSError:
        return {"metrics.up": 0}
    out = {"metrics.up": 1}
    keep = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:request_success",
            "vllm:spec_decode_", "vllm:kv_cache_usage_perc", "vllm:prefix_cache_", "vllm:num_preemptions")
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(keep):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            out["metrics." + name] = out.get("metrics." + name, 0.0) + float(line.rsplit(" ", 1)[1])
        except ValueError:
            pass
    return out

def static_temps():
    rows = []
    for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        t = read(z + "/temp")
        if t is not None:
            rows.append((f"thermal.{os.path.basename(z)}.{read(z + '/type')}", int(t) / 1000))
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name = read(h + "/name") or "?"
        for ti in sorted(glob.glob(h + "/temp*_input")):
            t = read(ti)
            if t is not None and t.lstrip("-").isdigit():
                rows.append((f"hwmon.{os.path.basename(h)}.{name}.{os.path.basename(ti)}", int(t) / 1000))
    return rows

def cpufreq():
    rows = []
    for pol in sorted(glob.glob("/sys/devices/system/cpu/cpufreq/policy*")):
        f = read(pol + "/scaling_cur_freq")
        if f is not None:
            rows.append((f"cpufreq.{os.path.basename(pol)}", f))
    return rows

emit([("sampler.start", socket.gethostname())])
threading.Thread(target=gpu_thread, daemon=True).start()
threading.Thread(target=journal_thread, daemon=True).start()
prev = {"vm": vmstat(), "psi": psi(), "cpu": cpustat(), "thr": {}, "roce": roce()}
hz = os.sysconf("SC_CLK_TCK")
next_t = time.time()
while True:
    next_t += interval
    now = time.time()
    if now - last_scan > 10:
        vllm_pids = scan_pids()
        last_scan = now
    rows = []
    vm = vmstat()
    rows += [(f"vmstat.{k}.delta", v - prev["vm"].get(k, v)) for k, v in vm.items()]
    rows += [(f"mem.{k}_kB", v) for k, v in meminfo().items()]
    ps = psi()
    for k, v in ps.items():
        if k.endswith("total_us"):
            rows.append((k.replace("total_us", "delta_us"), v - prev["psi"].get(k, v)))
        else:
            rows.append((k, v))
    cs = cpustat()
    for c, (tot, idle) in cs.items():
        pt, pi = prev["cpu"].get(c, (tot, idle))
        if tot > pt:
            rows.append((f"cpu.{c}.busy_pct", round(100 * (1 - (idle - pi) / (tot - pt)), 1)))
    rows += cpufreq()
    rows += static_temps()
    th = threads()
    for k, (cpu, proc) in th.items():
        d = cpu - prev["thr"].get(k, (cpu, proc))[0]
        if d > 0 or any(w in k for w in ("EngineCor", "Worker_TP", "NCCL")):
            rows.append((f"thread.{k}.cpu_pct", round(100 * d / hz / interval, 1)))
            rows.append((f"thread.{k}.last_cpu", proc))
    rc = roce()
    rows += [(f"{k}.delta", v - prev["roce"].get(k, v)) for k, v in rc.items() if v != prev["roce"].get(k, v)]
    rows += list(metrics().items())
    emit(rows, now)
    prev = {"vm": vm, "psi": ps, "cpu": cs, "thr": th, "roce": rc}
    time.sleep(max(0.0, next_t - time.time()))
PY
}

# Receiving side of the worker stream: append each line and fsync once per batch.
fsync_writer_py() {
  cat <<'PY'
import os, sys
fh = open(sys.argv[1], "a", buffering=1)
for line in sys.stdin:
    fh.write(line)
    fh.flush()
    os.fsync(fh.fileno())
PY
}

cmd_sample() {
  local out="${1:--}" node="${NODE:-$(hostname)}"
  exec taskset -c 0 nice -n 10 python3 -u -c "$(sampler_py)" "$out" "$node" "$METRICS_URL" "$NCCL_IB_HCA" "$INTERVAL_S"
}

# Worker sampler over ssh; the worker has no API server, so no /metrics there.
cmd__remote() {
  "${SSH[@]}" "$WORKER_HOST" \
    "taskset -c 0 nice -n 10 python3 -u - - '$WORKER_HOST' none '$NCCL_IB_HCA' '$INTERVAL_S'" \
    < <(sampler_py) | python3 -u -c "$(fsync_writer_py)" "$1"
}

cmd_start() {
  local ev="$1" dir="$1/telemetry" head
  head="$(hostname)"
  mkdir -p "$dir"
  if [[ -s "$dir/pids" ]]; then
    echo "telemetry already running for $ev (see $dir/pids)" >&2
    exit 1
  fi
  : >"$dir/pids"
  NODE="$head" setsid "$SELF" sample "$dir/${head}.csv" </dev/null >"$dir/${head}.sampler.log" 2>&1 &
  echo "$! local ${head}" >>"$dir/pids"
  if [[ "${NO_WORKER:-0}" != 1 ]]; then
    if "${SSH[@]}" "$WORKER_HOST" true >/dev/null 2>&1; then
      setsid "$SELF" _remote "$dir/${WORKER_HOST}.csv" </dev/null >"$dir/${WORKER_HOST}.sampler.log" 2>&1 &
      echo "$! remote ${WORKER_HOST}" >>"$dir/pids"
    else
      echo "WARNING: cannot ssh to $WORKER_HOST; worker telemetry not started" | tee -a "$dir/${head}.sampler.log" >&2
    fi
  fi
  date -u +%FT%T.%3NZ >"$dir/started-at.txt"
  echo "telemetry started → $dir"
}

cmd_stop() {
  local dir="$1/telemetry"
  [[ -f "$dir/pids" ]] || { echo "no telemetry pids in $dir" >&2; exit 1; }
  while read -r pid _ _; do
    # Each sampler leads its own session; kill the whole group (nvidia-smi, journalctl, ssh).
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done <"$dir/pids"
  : >"$dir/pids"
  date -u +%FT%T.%3NZ >"$dir/stopped-at.txt"
  echo "telemetry stopped"
}

# Static per-boot receipt for one node, printed as a sectioned text blob.
node_receipt_sh() {
  cat <<'SH'
section() { printf '\n=== %s ===\n' "$1"; }
section date; date -u +%FT%T.%3NZ
section hostname; hostname
section uname; uname -a
section nvidia-driver; cat /proc/driver/nvidia/version 2>&1
section nvidia-smi-q; nvidia-smi -q -d CLOCK,PERFORMANCE,POWER,TEMPERATURE 2>&1
section bios; for f in bios_vendor bios_version bios_date product_name board_name; do printf '%s=%s\n' "$f" "$(cat /sys/class/dmi/id/$f 2>/dev/null)"; done
section fwupdmgr; timeout 30 fwupdmgr get-devices --no-unreported-check 2>&1 | grep -E 'Device|Current version|Name|GUID' | head -80 || true
section governors; cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor 2>/dev/null | sort | uniq -c
section cpufreq-driver; cat /sys/devices/system/cpu/cpufreq/policy0/scaling_driver 2>/dev/null
section free; free -h
section swap; swapon --show 2>&1
section docker-ps; docker ps --format '{{.Names}} {{.Image}} {{.Status}}' 2>&1
section docker-image; docker inspect "$CN" --format '{{.Config.Image}} {{.Image}} {{.Created}}' 2>&1
section docker-mounts; docker inspect "$CN" --format '{{json .HostConfig.Binds}}' 2>&1
section docker-env-redacted; docker inspect "$CN" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>&1 | sed -E 's/^((HF_TOKEN|HUGGING_FACE_HUB_TOKEN)=).*/\1REDACTED/'
section caches
for v in VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR TRITON_CACHE_DIR VLLM_CACHE_ROOT; do
  cdir="$(docker inspect "$CN" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | sed -n "s/^$v=//p")"
  [ -z "$cdir" ] && { echo "$v=unset"; continue; }
  host="$(docker inspect "$CN" --format '{{range .Mounts}}{{.Destination}} {{.Source}}{{println}}{{end}}' 2>/dev/null | awk -v d="$cdir" 'index(d,$1)==1 && length($1)>length(best){best=$1; src=$2 substr(d,length($1)+1)} END{print src}')"
  echo "$v=$cdir host=${host:-not-mounted}"
  if [ -n "$host" ] && [ -d "$host" ]; then
    echo "files=$(find "$host" -type f | wc -l) bytes=$(du -sb "$host" | cut -f1)"
    echo "tree_sha256=$(cd "$host" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
    find "$host" -type f -name '*.json' -newer /proc/1 2>/dev/null | head -20
  fi
done
SH
}

cmd_receipt() {
  local ev="$1" dir="$1/receipts" head
  head="$(hostname)"
  mkdir -p "$dir"
  CN="$CONTAINER_NAME" bash -c "$(node_receipt_sh)" >"$dir/${head}-boot.txt" 2>&1 || true
  docker logs "$CONTAINER_NAME" >"$dir/${head}-docker.log" 2>&1 || true
  if "${SSH[@]}" "$WORKER_HOST" true >/dev/null 2>&1; then
    "${SSH[@]}" "$WORKER_HOST" "CN='$CONTAINER_NAME' bash -s" < <(node_receipt_sh) >"$dir/${WORKER_HOST}-boot.txt" 2>&1 || true
    "${SSH[@]}" "$WORKER_HOST" "docker logs '$CONTAINER_NAME' 2>&1" >"$dir/${WORKER_HOST}-docker.log" 2>&1 || true
  else
    echo "cannot ssh to $WORKER_HOST" >"$dir/${WORKER_HOST}-boot.txt"
  fi
  echo "receipts → $dir"
}

main() {
  local sub="${1:-}"
  shift || true
  case "$sub" in
    sample) cmd_sample "$@" ;;
    _remote) cmd__remote "$1" ;;
    start|stop|receipt)
      [[ $# -ge 1 ]] || usage
      "cmd_$sub" "$1"
      ;;
    *) usage ;;
  esac
}

main "$@"
