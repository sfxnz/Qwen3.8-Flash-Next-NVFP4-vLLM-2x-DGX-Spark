#!/usr/bin/env bash
# Qwen3.8-Flash-Next-NVFP4 on 2x DGX Spark (GB10) — vLLM TP=2
set -euo pipefail

# BEGIN generated from recipe.yaml — edit recipe.yaml and run kit/render.py
MODEL="${MODEL:-nvidia/Qwen3.8-Flash-Next-NVFP4}"
SERVED_NAME="${SERVED_NAME:-nvidia/Qwen3.8-Flash-Next-NVFP4}"
IMAGE="${IMAGE:-vllm/vllm-openai:nightly-aarch64@sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next-nvfp4}"
PORT="${PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29523}"
HEAD_IP="${HEAD_IP:-10.100.8.1}"
WORKER_HOST="${WORKER_HOST:-spark2}"
IFACE="${IFACE:-enp1s0f1np1}"
HCA="${HCA:-rocep1s0f1}"
TP="${TP:-2}"
NNODES="${NNODES:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
UTIL="${UTIL:-0.80}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
SPEC="${SPEC:-mtp}"
MOE_BACKEND="${MOE_BACKEND:-auto}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
FORCE_UNSAFE_CTX="${FORCE_UNSAFE_CTX:-0}"
FORCE_UNSAFE_MOE="${FORCE_UNSAFE_MOE:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
VLLM_PLE_FP8_CHECKPOINT="${VLLM_PLE_FP8_CHECKPOINT:-0}"
VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-0}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
HF_HOME_IN_CONTAINER="/cache/huggingface"
SNAPSHOT_SHA="${SNAPSHOT_SHA:-fab0aecb760cec45227f6656abcaafa11abca87a}"
SNAPSHOT="${HF_CACHE}/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/${SNAPSHOT_SHA}"
SNAPSHOT_IN_CONTAINER="${HF_HOME_IN_CONTAINER}/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/${SNAPSHOT_SHA}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
ORCHESTRATE="${ORCHESTRATE:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# END generated
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLE_OVERLAY="${PLE_OVERLAY:-$SCRIPT_DIR/docker/ple_layer.py}"
PLE_IN_CONTAINER="/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ple_layer.py"
MTP_OVERLAY="${MTP_OVERLAY:-$SCRIPT_DIR/docker/modelopt.py}"
MTP_IN_CONTAINER="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py"

if [[ -z "${SPEC_CONFIG:-}" ]]; then
  case "$SPEC" in
    mtp)
      # MTP experts are FP8_BLOCK_SCALES. Global --moe-backend marlin is for
      # NVFP4 routed experts only. marlin on unquantized MoE raises
      # moe_backend='marlin' is not supported for unquantized MoE.
      SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":'"$NUM_SPECULATIVE_TOKENS"',"moe_backend":"triton"}'
      ;;
    *)
      echo "Unknown SPEC=$SPEC (want mtp)" >&2
      exit 1
      ;;
  esac
fi

if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'
fi

if [[ "$MAX_MODEL_LEN" -gt 1048576 && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  echo "--max-model-len $MAX_MODEL_LEN is above 1048576. Native max_position_embeddings is 262144. 1M is a lab ceiling. Community 1M YaRN on GB10 hangs on long prefills (vLLM #54629). FORCE_UNSAFE_CTX=1 overrides." >&2
  exit 1
fi
if [[ "$MAX_NUM_SEQS" -gt 8 && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  echo "MAX_NUM_SEQS=$MAX_NUM_SEQS exceeds 8 on this occupancy pin. seqs=8 held eight short streams with no drop. seqs above 8 is unmeasured. FORCE_UNSAFE_CTX=1 overrides." >&2
  exit 1
fi
if [[ "$MOE_BACKEND" == marlin && "$FORCE_UNSAFE_MOE" != 1 ]]; then
  echo "MOE_BACKEND=marlin is not supported for unquantized MTP MoE on this image (ValueError: moe_backend='marlin' is not supported for unquantized MoE). In-band MTP stays BF16. Use auto. FORCE_UNSAFE_MOE=1 overrides." >&2
  exit 1
fi
if [[ "$MOE_BACKEND" == flashinfer_cutlass && "$FORCE_UNSAFE_MOE" != 1 ]]; then
  echo "MOE_BACKEND=flashinfer_cutlass OOM'd spark2 on GLM after 90.67 GiB weights. FORCE_UNSAFE_MOE=1 overrides." >&2
  exit 1
fi
if [[ "$MOE_BACKEND" == b12x && "$FORCE_UNSAFE_MOE" != 1 ]]; then
  echo "MOE_BACKEND=b12x is unmeasured on this SHA and has Xid 31 reports on sm_121. FORCE_UNSAFE_MOE=1 overrides." >&2
  exit 1
fi
if [[ "$MAX_MODEL_LEN" -gt 262144 && "$VLLM_ALLOW_LONG_MAX_MODEL_LEN" != 1 ]]; then
  echo "MAX_MODEL_LEN=$MAX_MODEL_LEN exceeds native 262144. vLLM refuses that unless VLLM_ALLOW_LONG_MAX_MODEL_LEN=1. This does not enable YaRN. FORCE_UNSAFE_CTX=1 does not replace that env." >&2
  exit 1
fi

if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  printf '==> validate-only spec=%s seqs=%s spec_tokens=%s eager=%s compilation=%s image=%s moe=%s ple=%s\n' \
    "$SPEC" "$MAX_NUM_SEQS" "$NUM_SPECULATIVE_TOKENS" "$ENFORCE_EAGER" "$COMPILATION_CONFIG" \
    "$IMAGE" "$MOE_BACKEND" "$VLLM_PLE_FP8_CHECKPOINT"
  exit 0
fi

log() { printf '==> %s\n' "$*"; }

host_short() { hostname -s | tr '[:upper:]' '[:lower:]'; }

detect_role() {
  if [[ -n "${ROLE:-}" ]]; then
    printf '%s\n' "$ROLE"
    return
  fi
  case "$(host_short)" in
    spark2*) printf 'worker\n' ;;
    *) printf 'head\n' ;;
  esac
}

hf_bin() {
  if command -v hf >/dev/null 2>&1; then
    echo hf
  elif command -v huggingface-cli >/dev/null 2>&1; then
    echo huggingface-cli
  else
    return 1
  fi
}

token_env() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s' "$HF_TOKEN"
    return
  fi
  if [[ -f "$HOME/.cache/huggingface/token" ]]; then
    tr -d '[:space:]' <"$HOME/.cache/huggingface/token"
  fi
}

resolve_model() {
  printf '%s\n' "$SNAPSHOT_IN_CONTAINER"
}

maybe_drop_caches() {
  if sudo -n true >/dev/null 2>&1; then
    sync
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  fi
}

ensure_ple_overlay() {
  if [[ ! -f "$PLE_OVERLAY" ]]; then
    echo "PLE overlay missing: $PLE_OVERLAY. Run python3 docker/apply_ple_overlay.py from the recipe root." >&2
    exit 1
  fi
  if grep -q 'VLLM_PLE_FP8_CHECKPOINT' "$PLE_OVERLAY"; then
    :
  elif grep -q 'ModelOptMixedPrecisionConfig' "$PLE_OVERLAY" && grep -q 'Qwen4ExpPLEFp8EmbeddingMethod' "$PLE_OVERLAY"; then
    :
  else
    echo "PLE overlay $PLE_OVERLAY is missing an FP8 PLE selector." >&2
    exit 1
  fi
}

ensure_mtp_overlay() {
  if [[ ! -f "$MTP_OVERLAY" ]]; then
    echo "MTP overlay missing: $MTP_OVERLAY. Run python3 docker/apply_mtp_fp8_overlay.py from the recipe root." >&2
    exit 1
  fi
  if ! grep -q 'FP8_BLOCK_SCALES' "$MTP_OVERLAY"; then
    echo "MTP overlay $MTP_OVERLAY is missing FP8_BLOCK_SCALES dispatch." >&2
    exit 1
  fi
}

ensure_image() {
  log "Ensuring image $IMAGE"
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    log "Pulling $IMAGE"
    docker pull "$IMAGE"
  fi
}

ensure_weights() {
  if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
    local HF=""
    HF="$(hf_bin || true)"
    if [[ -d "$SNAPSHOT" ]]; then
      log "Using pinned snapshot $SNAPSHOT"
    elif [[ -n "$HF" ]]; then
      export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
      log "Downloading $MODEL revision $SNAPSHOT_SHA (resumes under $HF_CACHE)"
      "$HF" download "$MODEL" --revision "$SNAPSHOT_SHA"
    else
      echo "No hf CLI on PATH and snapshot $SNAPSHOT is missing" >&2
      exit 1
    fi
  fi
  if [[ ! -d "$SNAPSHOT" ]]; then
    echo "Pinned snapshot missing: $SNAPSHOT" >&2
    exit 1
  fi
}

refuse_foreign_serve() {
  local name devices
  while IFS= read -r name; do
    [[ -z "$name" || "$name" == "$CONTAINER_NAME" ]] && continue
    devices="$(docker inspect -f '{{json .HostConfig.DeviceRequests}} {{json .HostConfig.Devices}} {{json .Config.Env}}' "$name" 2>/dev/null || true)"
    if printf '%s' "$devices" | grep -Eqi 'gpu|nvidia|infiniband'; then
      echo "$name is using GPUs or InfiniBand. This recipe needs exclusive GPUs on both Sparks. Do not start. Do not docker rm that container from this script." >&2
      exit 1
    fi
  done < <(docker ps --format '{{.Names}}')
}

refuse_busy_port() {
  if (echo >/dev/tcp/127.0.0.1/"$PORT") >/dev/null 2>&1; then
    echo "Port $PORT is already in use" >&2
    exit 1
  fi
}

stop_local() {
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    log "Removing existing container $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null
  fi
}

start_local() {
  local rank="$1"
  mkdir -p "$HF_CACHE"
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found" >&2
    exit 1
  fi
  refuse_foreign_serve
  maybe_drop_caches
  stop_local
  ensure_image
  ensure_weights
  ensure_ple_overlay
  ensure_mtp_overlay

  local serve_model
  serve_model="$(resolve_model)"

  local tok
  tok="$(token_env || true)"
  local env_args=(
    -e "HF_HOME=$HF_HOME_IN_CONTAINER"
    -e "TORCH_CUDA_ARCH_LIST=12.1a"
    -e "FLASHINFER_CUDA_ARCH_LIST=12.1a"
    -e "FLASHINFER_DISABLE_VERSION_CHECK=1"
    -e "VLLM_ENGINE_READY_TIMEOUT_S=3600"
    -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    -e "VLLM_ALLOW_LONG_MAX_MODEL_LEN=$VLLM_ALLOW_LONG_MAX_MODEL_LEN"
    -e "NCCL_SOCKET_IFNAME=$IFACE"
    -e "GLOO_SOCKET_IFNAME=$IFACE"
    -e "TP_SOCKET_IFNAME=$IFACE"
    -e "NCCL_IB_HCA=$HCA"
    -e "NCCL_NET=IB"
    -e "NCCL_IB_DISABLE=0"
    -e "NCCL_CROSS_NIC=1"
    -e "NCCL_NVLS_ENABLE=0"
    -e "NCCL_CUMEM_ENABLE=0"
    -e "NCCL_DEBUG=WARN"
  )
  local host_ip="$HEAD_IP"
  if [[ "$rank" != "0" ]]; then
    host_ip="$(ip -4 -o addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
    host_ip="${host_ip:-10.100.8.2}"
  fi
  env_args+=(-e "VLLM_HOST_IP=$host_ip")
  if [[ -n "$tok" ]]; then
    env_args+=(-e "HF_TOKEN=$tok" -e "HUGGING_FACE_HUB_TOKEN=$tok")
  fi

  local rank_args=()
  if [[ "$rank" == "0" ]]; then
    rank_args+=(--host 0.0.0.0 --port "$PORT")
  else
    rank_args+=(--headless)
  fi

  local eager_args=()
  if [[ "$ENFORCE_EAGER" == "1" ]]; then
    eager_args+=(--enforce-eager)
  else
    eager_args+=(--compilation-config "$COMPILATION_CONFIG")
  fi

  local vol_args=(
    -v "${HF_CACHE}:${HF_HOME_IN_CONTAINER}"
    -v "${PLE_OVERLAY}:${PLE_IN_CONTAINER}:ro"
    -v "${MTP_OVERLAY}:${MTP_IN_CONTAINER}:ro"
  )
  local batched_args=()
  if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
    batched_args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
  fi

  log "Starting $CONTAINER_NAME rank=$rank model=$serve_model ctx=$MAX_MODEL_LEN kv=$KV_CACHE_DTYPE spec=$SPEC moe=$MOE_BACKEND"
  docker run -d \
    --name "$CONTAINER_NAME" \
    --restart no \
    --gpus all \
    --network host \
    --ipc host \
    --shm-size 32g \
    --device /dev/infiniband \
    --cap-add IPC_LOCK \
    --ulimit memlock=-1:-1 \
    "${vol_args[@]}" \
    "${env_args[@]}" \
    "$IMAGE" \
    "$serve_model" \
    --tensor-parallel-size "$TP" \
    --nnodes "$NNODES" \
    --node-rank "$rank" \
    --distributed-executor-backend mp \
    --master-addr "$HEAD_IP" \
    --master-port "$MASTER_PORT" \
    "${rank_args[@]}" \
    --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --gpu-memory-utilization "$UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    "${batched_args[@]}" \
    "${eager_args[@]}" \
    --moe-backend "$MOE_BACKEND" \
    --speculative-config "$SPEC_CONFIG" \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --tool-call-parser "$TOOL_CALL_PARSER" \
    --enable-auto-tool-choice \
    --reasoning-parser "$REASONING_PARSER" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --served-model-name "$SERVED_NAME" \
    --trust-remote-code \
    $EXTRA_ARGS
}

wait_ready() {
  log "Waiting for http://127.0.0.1:${PORT}/health and /v1/models"
  local i health body
  for i in $(seq 1 720); do
    health="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" || true)"
    body="$(curl -sf "http://127.0.0.1:${PORT}/v1/models" || true)"
    if [[ "$health" == "200" && -n "$body" && "$body" == *"$SERVED_NAME"* ]]; then
      log "Ready → http://127.0.0.1:${PORT}/v1  (context=$MAX_MODEL_LEN health=$health)"
      printf '%s\n' "$body"
      echo
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
      echo "Container exited early. Logs:" >&2
      docker logs "$CONTAINER_NAME" 2>&1 | tail -120 >&2
      exit 1
    fi
    sleep 5
    if (( i % 12 == 0 )); then
      log "still loading… (${i}×5s) health=${health:-none} — docker logs -f $CONTAINER_NAME"
    fi
  done
  echo "Timed out waiting for API. Recent logs:" >&2
  docker logs "$CONTAINER_NAME" 2>&1 | tail -120 >&2
  exit 1
}

ROLE="$(detect_role)"
log "role=$ROLE host=$(host_short)"

if [[ "$ORCHESTRATE" == "auto" && "$ROLE" == "head" ]]; then
  refuse_foreign_serve
  refuse_busy_port
  if [[ "$NNODES" -gt 1 ]]; then
    if ! command -v ssh >/dev/null 2>&1 || ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" true >/dev/null 2>&1; then
      echo "Cannot SSH to $WORKER_HOST. Refusing to start a TP=$TP head rank alone (NNODES=$NNODES)." >&2
      exit 1
    fi
    log "Starting worker on $WORKER_HOST first"
    mkdir -p "${PWD}/.run-state"
    printf '%s\n' "$WORKER_HOST" >"${PWD}/.run-state/worker_host"
    scp -q "$0" "${WORKER_HOST}:/tmp/qwen38-run.sh"
    scp -q "$PLE_OVERLAY" "${WORKER_HOST}:/tmp/qwen38-ple_layer.py"
    scp -q "$MTP_OVERLAY" "${WORKER_HOST}:/tmp/qwen38-modelopt.py"
    ssh "$WORKER_HOST" \
      "ROLE=worker ORCHESTRATE=0 IMAGE='$IMAGE' CONTAINER_NAME='$CONTAINER_NAME' PORT='$PORT' MASTER_PORT='$MASTER_PORT' HEAD_IP='$HEAD_IP' IFACE='$IFACE' HCA='$HCA' MAX_MODEL_LEN='$MAX_MODEL_LEN' MAX_NUM_SEQS='$MAX_NUM_SEQS' UTIL='$UTIL' KV_CACHE_DTYPE='$KV_CACHE_DTYPE' TP='$TP' NNODES='$NNODES' SERVED_NAME='$SERVED_NAME' SKIP_DOWNLOAD='$SKIP_DOWNLOAD' SPEC='$SPEC' SPEC_CONFIG='$SPEC_CONFIG' NUM_SPECULATIVE_TOKENS='$NUM_SPECULATIVE_TOKENS' ENFORCE_EAGER='$ENFORCE_EAGER' COMPILATION_CONFIG='$COMPILATION_CONFIG' MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS' FORCE_UNSAFE_CTX='$FORCE_UNSAFE_CTX' FORCE_UNSAFE_MOE='$FORCE_UNSAFE_MOE' VLLM_PLE_FP8_CHECKPOINT='$VLLM_PLE_FP8_CHECKPOINT' VLLM_ALLOW_LONG_MAX_MODEL_LEN='$VLLM_ALLOW_LONG_MAX_MODEL_LEN' TOOL_CALL_PARSER='$TOOL_CALL_PARSER' REASONING_PARSER='$REASONING_PARSER' MOE_BACKEND='$MOE_BACKEND' SNAPSHOT_SHA='$SNAPSHOT_SHA' HF_CACHE='$HF_CACHE' MODEL='$MODEL' PLE_OVERLAY='/tmp/qwen38-ple_layer.py' MTP_OVERLAY='/tmp/qwen38-modelopt.py' EXTRA_ARGS='$EXTRA_ARGS' bash /tmp/qwen38-run.sh"
    log "Worker container started. Waiting 25s for NCCL listen, then starting head"
    sleep 25
  fi
  start_local 0
  wait_ready
  log "Stop with: ./stop.sh"
elif [[ "$ROLE" == "worker" ]]; then
  start_local 1
  log "Worker rank 1 is up. Head should start next."
else
  start_local 0
  wait_ready
  log "Stop with: ./stop.sh"
fi
