#!/usr/bin/env bash
# Idempotent acquire of the NVFP4 snapshot and the day-0 vLLM pin.
# Does not start a serve. Does not occupy a GPU.
set -euo pipefail

HF_ID="${HF_ID:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
SNAPSHOT="${SNAPSHOT:-7b719225242aacd3dbd3f9407468c2ee9a9d2594}"
IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
SNAP_DIR="${HF_CACHE}/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/${SNAPSHOT}"
HERE="$(cd "$(dirname "$0")" && pwd)"
EXPECTED="${HERE}/hub-dry-run.json"
CHECK_ONLY="${CHECK_ONLY:-0}"

log() { printf '==> %s\n' "$*"; }

need_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    return
  fi
  if [[ -f "${HOME}/.cache/huggingface/token" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' <"${HOME}/.cache/huggingface/token")"
    export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
  fi
}

hf_bin() {
  if command -v hf >/dev/null 2>&1; then
    printf 'hf\n'
  elif command -v huggingface-cli >/dev/null 2>&1; then
    printf 'huggingface-cli\n'
  else
    return 1
  fi
}

snapshot_complete() {
  python3 - "$SNAP_DIR" "$EXPECTED" <<'PY'
import json, os, sys
snap, expected_path = sys.argv[1], sys.argv[2]
if not os.path.isdir(snap):
    sys.exit(1)
with open(expected_path) as f:
    expected = json.load(f)
missing = []
for row in expected:
    path = os.path.join(snap, row["file"])
    if not os.path.isfile(path):
        missing.append(row["file"])
if missing:
    sys.stderr.write(f"snapshot missing {len(missing)} files, first={missing[0]}\n")
    sys.exit(1)
print(f"snapshot ok files={len(expected)} dir={snap}")
PY
}

image_present() {
  docker image inspect "$IMAGE" >/dev/null 2>&1
}

ensure_snapshot() {
  if snapshot_complete; then
    log "snapshot already present $SNAP_DIR"
    return
  fi
  if [[ "$CHECK_ONLY" == "1" ]]; then
    echo "snapshot incomplete: $SNAP_DIR" >&2
    exit 1
  fi
  local HF attempt=0
  HF="$(hf_bin)"
  need_token
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
  export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
  while ! snapshot_complete; do
    attempt=$((attempt + 1))
    if (( attempt > 30 )); then
      echo "snapshot still incomplete after $attempt download attempts" >&2
      exit 1
    fi
    log "downloading $HF_ID revision $SNAPSHOT (attempt $attempt, resumes under $HF_CACHE)"
    "$HF" download "$HF_ID" --revision "$SNAPSHOT" --max-workers 2 || true
    if snapshot_complete; then
      return
    fi
    sleep 5
  done
}

ensure_image() {
  if image_present; then
    log "image already present $IMAGE"
    return
  fi
  if [[ "$CHECK_ONLY" == "1" ]]; then
    echo "image missing: $IMAGE" >&2
    exit 1
  fi
  log "pulling $IMAGE"
  docker pull "$IMAGE"
  if ! image_present; then
    echo "docker image inspect failed for $IMAGE after pull" >&2
    exit 1
  fi
}

ensure_image
ensure_snapshot
log "acquire ok"
