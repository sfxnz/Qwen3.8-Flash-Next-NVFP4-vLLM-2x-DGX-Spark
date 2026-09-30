#!/usr/bin/env bash
# Qwen3.8-Flash-Next-NVFP4 on 2x DGX Spark (GB10) — vLLM TP=2
set -euo pipefail

# BEGIN generated from recipe.yaml — edit recipe.yaml and run kit/render.py
MODEL="${MODEL:-nvidia/Qwen3.8-Flash-Next-NVFP4}"
SERVED_NAME="${SERVED_NAME:-nvidia/Qwen3.8-Flash-Next-NVFP4}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.30.0-aarch64@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56}"
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
THROUGHPUT_PROFILE="${THROUGHPUT_PROFILE:-0}"
UTIL="${UTIL:-0.76}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
SPEC="${SPEC:-mtp}"
MOE_BACKEND="${MOE_BACKEND:-auto}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-4800}"
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-64}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-1000}"
MEMGUARD="${MEMGUARD:-1}"
MEMGUARD_MIN_AVAIL_MB="${MEMGUARD_MIN_AVAIL_MB:-2048}"
MEMGUARD_MIN_SWAP_FREE_MB="${MEMGUARD_MIN_SWAP_FREE_MB:-2048}"
FORCE_UNSAFE_CTX="${FORCE_UNSAFE_CTX:-0}"
FORCE_UNSAFE_MOE="${FORCE_UNSAFE_MOE:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
DIAGNOSTIC="${DIAGNOSTIC:-0}"
BENCH_ONLY="${BENCH_ONLY:-0}"
VLLM_PLE_FP8_CHECKPOINT="${VLLM_PLE_FP8_CHECKPOINT:-0}"
VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-0}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-chat_template_alias.jinja}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-1}"
MM_MIN_PIXELS="${MM_MIN_PIXELS:-65536}"
MM_MAX_PIXELS="${MM_MAX_PIXELS:-4194304}"
MM_LIMIT_IMAGE="${MM_LIMIT_IMAGE:-8}"
MM_LIMIT_VIDEO="${MM_LIMIT_VIDEO:-1}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
HF_HOME_IN_CONTAINER="/cache/huggingface"
SNAPSHOT_SHA="${SNAPSHOT_SHA:-fab0aecb760cec45227f6656abcaafa11abca87a}"
SNAPSHOT="${HF_CACHE}/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/${SNAPSHOT_SHA}"
SNAPSHOT_IN_CONTAINER="${HF_HOME_IN_CONTAINER}/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/${SNAPSHOT_SHA}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"
ORCHESTRATE="${ORCHESTRATE:-auto}"
OVERLAYS="${OVERLAYS:-auto}"
OVERLAYS_PIN="${OVERLAYS_PIN:-docker/ple_layer.py docker/modelopt.py docker/ple_ops.py}"
OVERLAYS_V030="${OVERLAYS_V030:-docker/v030/flashinfer_cutlass_moe.py docker/v030/gdn_attn.py docker/v030/qsa_indexer.py docker/v030/serving.py docker/v030/mtp.py}"
GDN_LAZY="${GDN_LAZY:-1}"
DRAFT_MOE_CONFIG="${DRAFT_MOE_CONFIG:-1}"
DRAFT_LOCAL_ARGMAX="${DRAFT_LOCAL_ARGMAX:-1}"
DRAFT_HEAD_FP8="${DRAFT_HEAD_FP8:-1}"
DRAFT_VOCAB="${DRAFT_VOCAB:-none}"
FP8_DENSE="${FP8_DENSE:-none}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_ENV="${EXTRA_ENV:-}"
# END generated
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$SCRIPT_DIR/.run-state"
SITE_PACKAGES="/usr/local/lib/python3.12/dist-packages"
# The pinned nightly (0.28.1rc1.dev437+ge962733e0). Several guards below hold only on this digest.
NIGHTLY_PIN_DIGEST="sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b"
# vllm/vllm-openai:v0.30.0-aarch64
V030_DIGEST="sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56"
# Bodies (R11 header stripped) of the three pin overlays. They must never mount on another digest:
# v0.30 model.py calls ple.start_prefetch(), which the pinned ple_layer.py lacks; v0.30 ple_conv
# takes a required outer_residual; the pinned modelopt.py would shadow v0.30's own (F11).
LEGACY_PIN_OVERLAY_SHA256="859ae689a7a74b8e4d8ea8c62b3479dbb214314f4fb168ebc2cb963ab3e4a664 becd555ca949ffe68c27dc2350f0ab2e8c09c3b3c34a4c967e732d9339392943 cc8994b653dd4b59d6ba35b6ec10af34017e9944482ec67cad6a32a02c5dd00a"
# Upstream files the pin cannot serve correctly without: MTP FP8_BLOCK_SCALES load, F01 stride fix.
PIN_REQUIRED_OVERLAYS="vllm/model_executor/layers/quantization/modelopt.py vllm/models/qwen4_exp/nvidia/ops/ple.py"
# MAX_NUM_BATCHED_TOKENS the indexer-memory analysis (F05) was done at.
MEASURED_BATCHED_TOKENS=8192

die() {
  echo "$*" >&2
  exit 1
}

# --- integers and flags. Leading zeros are refused: bash reads 010 as octal 8, vLLM as 10.
for name in MAX_MODEL_LEN MAX_NUM_SEQS NUM_SPECULATIVE_TOKENS PORT MASTER_PORT TP NNODES MEMGUARD_MIN_AVAIL_MB MEMGUARD_MIN_SWAP_FREE_MB MM_MIN_PIXELS MM_MAX_PIXELS MM_LIMIT_IMAGE MM_LIMIT_VIDEO; do
  [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "$name=${!name} is not a positive decimal integer."
done
[[ -z "$MAX_NUM_BATCHED_TOKENS" || "$MAX_NUM_BATCHED_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS is not empty or a positive decimal integer."
[[ "$LONG_PREFILL_TOKEN_THRESHOLD" == none || "$LONG_PREFILL_TOKEN_THRESHOLD" =~ ^[1-9][0-9]*$ ]] || die "LONG_PREFILL_TOKEN_THRESHOLD=$LONG_PREFILL_TOKEN_THRESHOLD is not none or a positive decimal integer."
# none = do not pass the env (the image default is 512 MiB).
[[ "$VLLM_SPARSE_INDEXER_MAX_LOGITS_MB" == none || "$VLLM_SPARSE_INDEXER_MAX_LOGITS_MB" =~ ^[1-9][0-9]*$ ]] || die "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=$VLLM_SPARSE_INDEXER_MAX_LOGITS_MB is not none or a positive decimal integer."
for name in THROUGHPUT_PROFILE FORCE_UNSAFE_CTX FORCE_UNSAFE_MOE ENFORCE_EAGER DIAGNOSTIC BENCH_ONLY VLLM_ALLOW_LONG_MAX_MODEL_LEN SKIP_DOWNLOAD HF_HUB_DISABLE_XET MEMGUARD ASYNC_SCHEDULING DRAFT_MOE_CONFIG DRAFT_LOCAL_ARGMAX DRAFT_HEAD_FP8 GDN_LAZY; do
  [[ "${!name}" =~ ^[01]$ ]] || die "$name=${!name} must be 0 or 1."
done
[[ "$OOM_SCORE_ADJ" =~ ^(-?[1-9][0-9]*|0)$ ]] && (( OOM_SCORE_ADJ >= -1000 && OOM_SCORE_ADJ <= 1000 )) || die "OOM_SCORE_ADJ=$OOM_SCORE_ADJ must be an integer in [-1000, 1000]."
[[ "$FP8_DENSE" == none || "$FP8_DENSE" == per_block ]] || die "FP8_DENSE=$FP8_DENSE must be none or per_block (lab-fp8-dense profile, docker/v030/FP8_DENSE.md)."
[[ "$DRAFT_VOCAB" == none || "$DRAFT_VOCAB" =~ ^[A-Za-z0-9_.-]+\.json$ ]] || die "DRAFT_VOCAB=$DRAFT_VOCAB must be none or a .json file name under \$HF_CACHE/qwen38-draft-vocab/."
[[ "$UTIL" =~ ^0\.[0-9]+$ ]] || die "UTIL=$UTIL must be a decimal in (0, 1), e.g. 0.80."
[[ "$MM_PROCESSOR_CACHE_GB" =~ ^(0|[1-9][0-9]*)(\.[0-9]+)?$ ]] || die "MM_PROCESSOR_CACHE_GB=$MM_PROCESSOR_CACHE_GB is not a non-negative decimal."
(( MM_MIN_PIXELS <= MM_MAX_PIXELS )) || die "MM_MIN_PIXELS=$MM_MIN_PIXELS exceeds MM_MAX_PIXELS=$MM_MAX_PIXELS."

# --- image: digest-pinned only.
[[ "$IMAGE" == *@sha256:* ]] || die "IMAGE=$IMAGE is not digest-pinned. Use <repo>:<tag>@sha256:<digest>."
IMAGE_DIGEST="${IMAGE##*@}"
ON_PIN=0
[[ "$IMAGE_DIGEST" == "$NIGHTLY_PIN_DIGEST" ]] && ON_PIN=1

# --- speculative and compilation config.
SPEC_DEFAULTED=0
if [[ -z "${SPEC_CONFIG:-}" ]]; then
  case "$SPEC" in
    mtp)
      # moe_backend here is a no-op on the pin: its V2 runner builds the drafter from the target's
      # kernel config (v1/worker/gpu/spec_decode/eagle/utils.py:72-100), so the drafter inherits
      # --moe-backend auto, which picks Triton for the 64x64-refined FP8 blocks. v0.30 honours it.
      SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":'"$NUM_SPECULATIVE_TOKENS"',"moe_backend":"triton"}'
      SPEC_DEFAULTED=1
      ;;
    none)
      [[ "$DIAGNOSTIC" == 1 ]] || die "SPEC=none boots without MTP. That is a diagnostic arm only. Set DIAGNOSTIC=1."
      SPEC_CONFIG=""
      ;;
    *)
      die "Unknown SPEC=$SPEC (want mtp, or none with DIAGNOSTIC=1)"
      ;;
  esac
elif [[ "$SPEC" == none ]]; then
  die "SPEC=none conflicts with an explicit SPEC_CONFIG."
fi

if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'
fi

# Prints: compile_mode local_argmax draft_sample_method adaptive_verification rejection_sample_method
CONFIG_FIELDS="$(SPEC_CONFIG="$SPEC_CONFIG" COMPILATION_CONFIG="$COMPILATION_CONFIG" python3 -c '
import json, os, sys
cfg = {}
for name in ("SPEC_CONFIG", "COMPILATION_CONFIG"):
    raw = os.environ[name]
    if not raw:
        cfg[name] = {}
        continue
    try:
        cfg[name] = json.loads(raw)
    except ValueError as exc:
        sys.exit(f"{name} is not valid JSON: {exc}: {raw}")
    if not isinstance(cfg[name], dict):
        sys.exit(f"{name} must be a JSON object: {raw}")
spec, comp = cfg["SPEC_CONFIG"], cfg["COMPILATION_CONFIG"]
word = lambda v: str(v).lower().replace(" ", "_") or "unset"
# JSON null mode is unset too: vLLM then picks VLLM_COMPILE (config/vllm.py mode-is-None branch).
mode = comp.get("mode")
print("unset" if mode is None else word(mode), word(spec.get("use_local_argmax_reduction", False)),
      word(spec.get("draft_sample_method", "greedy")), word(spec.get("enable_adaptive_verification", False)),
      word(spec.get("rejection_sample_method", "standard")))
')" || exit 1
read -r COMPILE_MODE LOCAL_ARGMAX DRAFT_SAMPLE_METHOD ADAPTIVE_VERIFICATION REJECTION_SAMPLE_METHOD <<<"$CONFIG_FIELDS"

# --- EXTRA_ENV: KEY=VALUE words, no spaces inside a value.
EXTRA_ENV_ARGS=()
for kv in $EXTRA_ENV; do
  [[ "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || die "EXTRA_ENV entry '$kv' is not KEY=VALUE."
  EXTRA_ENV_ARGS+=(-e "$kv")
done

# --- EXTRA_ARGS must not re-set a flag run.sh builds and guards; argparse keeps the last value, so a
# duplicate would bypass the guard on its variable.
for w in $EXTRA_ARGS; do
  case "${w%%=*}" in
    --compilation-config* | -cc* | -O* | --speculative-config* | --moe-backend | --max-num-batched-tokens | --max-num-seqs | --max-model-len | --enforce-eager | --long-prefill-token-threshold)
      die "EXTRA_ARGS sets $w, which run.sh passes and guards itself. Use COMPILATION_CONFIG, SPEC_CONFIG, MOE_BACKEND, MAX_NUM_BATCHED_TOKENS, MAX_NUM_SEQS, MAX_MODEL_LEN, ENFORCE_EAGER or LONG_PREFILL_TOKEN_THRESHOLD instead."
      ;;
    --async-scheduling | --no-async-scheduling | --mm-processor-kwargs | --limit-mm-per-prompt | --mm-processor-cache-gb | --chat-template)
      die "EXTRA_ARGS sets $w, which run.sh passes itself. Use ASYNC_SCHEDULING, MM_MIN_PIXELS / MM_MAX_PIXELS, MM_LIMIT_IMAGE / MM_LIMIT_VIDEO, MM_PROCESSOR_CACHE_GB or CHAT_TEMPLATE instead."
      ;;
  esac
done

# --- overlays: resolve the list, then check every R11 header against IMAGE.
overlay_field() {
  # Value of `# <key>: ` inside the leading R11 header block, or empty.
  awk -v key="$2" '
    NR == 1 && $0 != "# R11-OVERLAY" { exit }
    NR > 1 && !/^# [A-Za-z0-9_]+: / { exit }
    index($0, "# " key ": ") == 1 { print substr($0, length(key) + 5); exit }
  ' "$1"
}

overlay_body_sha256() {
  awk 'body || !/^# (R11-OVERLAY$|(base_image_digest|upstream_file|upstream_file_sha256|upstream_PR|generator): )/ { body = 1; print }' "$1" |
    sha256sum | cut -d' ' -f1
}

case "$OVERLAYS" in
  auto)
    if [[ "$ON_PIN" == 1 ]]; then
      OVERLAYS="$OVERLAYS_PIN"
    elif [[ "$IMAGE_DIGEST" == "$V030_DIGEST" ]]; then
      OVERLAYS="$OVERLAYS_V030"
    else
      OVERLAYS=""
    fi
    ;;
  none) OVERLAYS="" ;;
esac

# --- lab-fp8-dense profile (plan P4-1, R8; opt-in, not the default: fails T1-B top-1, passes T1-D/T2/T3/V).
FP8_DENSE_OVERLAY=docker/v030/modelopt.py
if [[ "$FP8_DENSE" != none ]]; then
  [[ "$IMAGE_DIGEST" == "$V030_DIGEST" ]] || die "FP8_DENSE=$FP8_DENSE needs the v0.30 digest ($FP8_DENSE_OVERLAY is generated against it)."
  # The head adds the overlay; the worker receives the head's resolved list.
  if [[ "${ROLE:-}" != worker && " $OVERLAYS " != *" $FP8_DENSE_OVERLAY "* ]]; then
    OVERLAYS="$OVERLAYS $FP8_DENSE_OVERLAY"
  fi
fi

# --- K3 GDN MTP decode with lazy state commit (docker/v030/K3.md, evidence/k3). v0.30 digest only: the head swaps
# docker/v030/gdn_attn.py for docker/v030/gdn_lazy_attn.py (the same S1.5 hunk plus K3 metadata) and adds
# docker/v030/gdn_lazy_linear_attn.py; the worker receives the head's resolved list. Both ranks then get
# VLLM_QWEN38_GDN_LAZY=1 (BASE_ENV below). Bit-exact vs the stock kernel (load-time self-test, fail closed).
GDN_LAZY_ATTN=docker/v030/gdn_lazy_attn.py
GDN_LAZY_LINEAR=docker/v030/gdn_lazy_linear_attn.py
if [[ "$GDN_LAZY" == 1 && "$IMAGE_DIGEST" == "$V030_DIGEST" && -n "$OVERLAYS" && "${ROLE:-}" != worker ]]; then
  OVERLAYS=" $OVERLAYS "
  OVERLAYS="${OVERLAYS// docker\/v030\/gdn_attn.py / $GDN_LAZY_ATTN }"
  if [[ "$OVERLAYS" == *" $GDN_LAZY_ATTN "* ]]; then
    [[ "$OVERLAYS" == *" $GDN_LAZY_LINEAR "* ]] || OVERLAYS="$OVERLAYS$GDN_LAZY_LINEAR"
  else
    echo "note: GDN_LAZY=1 ignored (the overlay list has neither docker/v030/gdn_attn.py nor $GDN_LAZY_ATTN)." >&2
  fi
  OVERLAYS="$(echo $OVERLAYS)"
elif [[ "$GDN_LAZY" == 1 && "$IMAGE_DIGEST" != "$V030_DIGEST" ]]; then
  echo "note: GDN_LAZY=1 ignored (K3 overlays are generated against the v0.30 digest)." >&2
fi

OVERLAY_FILES=()
OVERLAY_TARGETS=()
for f in $OVERLAYS; do
  [[ "$f" == /* ]] || f="$SCRIPT_DIR/$f"
  [[ -f "$f" ]] || die "Overlay missing: $f. Regenerate it with its docker/apply_*.py generator (docker/OVERLAYS.md)."
  digest="$(overlay_field "$f" base_image_digest)"
  upstream="$(overlay_field "$f" upstream_file)"
  [[ -n "$digest" && -n "$upstream" ]] || die "Overlay $f has no R11 header (base_image_digest, upstream_file). See docker/OVERLAYS.md."
  [[ "$digest" == "$IMAGE_DIGEST" ]] || die "Overlay $f was generated for $digest, not IMAGE $IMAGE_DIGEST. Regenerate it against this image or drop it from OVERLAYS."
  [[ "$upstream" =~ ^vllm/[A-Za-z0-9_/.]+\.py$ && "$upstream" != *..* ]] || die "Overlay $f upstream_file=$upstream must be a vllm/... .py path."
  if [[ "$ON_PIN" != 1 ]] && [[ " $LEGACY_PIN_OVERLAY_SHA256 " == *" $(overlay_body_sha256 "$f") "* ]]; then
    die "Overlay $f is a pin-only overlay by content and IMAGE is not the pinned nightly. v0.30 model.py calls ple.start_prefetch() (absent from the pinned ple_layer.py), v0.30 ple_conv needs outer_residual, and the pinned modelopt.py shadows v0.30's own (F11)."
  fi
  for t in "${OVERLAY_TARGETS[@]}"; do
    [[ "$t" != "$upstream" ]] || die "Two overlays target $upstream."
  done
  OVERLAY_FILES+=("$f")
  OVERLAY_TARGETS+=("$upstream")
done

overlay_mounted() {
  local t
  for t in "${OVERLAY_TARGETS[@]}"; do
    [[ "$t" == "$1" ]] && return 0
  done
  return 1
}

overlays_define() {
  # Any mounted overlay defines `def $1`.
  [[ ${#OVERLAY_FILES[@]} -gt 0 ]] && grep -q "def $1(" "${OVERLAY_FILES[@]}"
}

if [[ "$ON_PIN" == 1 && "$DIAGNOSTIC" != 1 ]]; then
  for t in $PIN_REQUIRED_OVERLAYS; do
    overlay_mounted "$t" || die "The pinned nightly needs an overlay for $t (docker/modelopt.py: MTP FP8_BLOCK_SCALES load; docker/ple_ops.py: F01 PLE short-conv state-index stride, vLLM #55375). DIAGNOSTIC=1 overrides."
  done
fi

# --- chat template: a recipe-root (or absolute) file, mounted read-only on the head; none = checkpoint template.
CHAT_TEMPLATE_FILE=""
if [[ "$CHAT_TEMPLATE" != none ]]; then
  CHAT_TEMPLATE_FILE="$CHAT_TEMPLATE"
  [[ "$CHAT_TEMPLATE_FILE" == /* ]] || CHAT_TEMPLATE_FILE="$SCRIPT_DIR/$CHAT_TEMPLATE_FILE"
  [[ -f "$CHAT_TEMPLATE_FILE" ]] || die "CHAT_TEMPLATE=$CHAT_TEMPLATE not found. Regenerate it with tools/make_alias_template.py, or set CHAT_TEMPLATE=none."
fi
CHAT_TEMPLATE_IN_CONTAINER="/recipe/$(basename "${CHAT_TEMPLATE_FILE:-none}")"

# --- vision: the processor pixel window and per-prompt item limits (plan P2-3).
MM_PROCESSOR_KWARGS='{"images_kwargs":{"min_pixels":'"$MM_MIN_PIXELS"',"max_pixels":'"$MM_MAX_PIXELS"'},"videos_kwargs":{"cap_pixels_per_frame":true}}'
LIMIT_MM_PER_PROMPT='{"image":'"$MM_LIMIT_IMAGE"',"video":'"$MM_LIMIT_VIDEO"'}'

# --- v0.30 envs, always on that digest (not via EXTRA_ENV): VLLM_PLE_CPU_OFFLOAD defaults to 1 there and
# pins 32 GiB of host memory per node for the PLE table (F33); breakable CUDA graphs stay off (D4).
BASE_ENV=()
[[ "$IMAGE_DIGEST" == "$V030_DIGEST" ]] && BASE_ENV=(VLLM_PLE_CPU_OFFLOAD=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0)
# K3: the env is inert without the overlay; set it only where the K3 dispatch is mounted.
if [[ "$GDN_LAZY" == 1 && "$IMAGE_DIGEST" == "$V030_DIGEST" ]] && overlays_define gdl_decode; then
  BASE_ENV+=(VLLM_QWEN38_GDN_LAZY=1)
fi
FP8_DENSE_ARGS=()
if [[ "$FP8_DENSE" == per_block ]]; then
  BASE_ENV+=(VLLM_QWEN38_FP8_DENSE=per_block)
  FP8_DENSE_ARGS=(--kernel-config '{"linear_backend_per_quant":{"fp8_block_w8a8":"marlin"}}')
fi

# --- L1a / L1b' / L1b draft head (docker/v030/MTP_HEAD.md). They take effect only on the v0.30 digest with
# the docker/v030/mtp.py overlay mounted (it defines get_top_tokens and reads the two envs); otherwise
# (the pin rollback, OVERLAYS=none) the drafter is stock and the knobs are ignored.
DRAFT_HEAD_ON=0
if [[ "$IMAGE_DIGEST" == "$V030_DIGEST" ]] && overlays_define get_top_tokens; then
  DRAFT_HEAD_ON=1
elif [[ "$DRAFT_LOCAL_ARGMAX" == 1 || "$DRAFT_HEAD_FP8" == 1 || "$DRAFT_VOCAB" != none ]]; then
  echo "note: DRAFT_LOCAL_ARGMAX / DRAFT_HEAD_FP8 / DRAFT_VOCAB ignored (needs the v0.30 digest and the docker/v030/mtp.py overlay)." >&2
fi
if [[ "$DRAFT_HEAD_ON" == 1 ]]; then
  # L1a: drafts pick a local argmax per TP shard and all-gather [M, 2]. An explicit SPEC_CONFIG wins.
  if [[ "$DRAFT_LOCAL_ARGMAX" == 1 && "$SPEC_DEFAULTED" == 1 ]]; then
    SPEC_CONFIG="${SPEC_CONFIG%\}}"',"use_local_argmax_reduction":true}'
    LOCAL_ARGMAX=true
  fi
  [[ "$DRAFT_HEAD_FP8" == 1 ]] && BASE_ENV+=(VLLM_QWEN38_DRAFT_HEAD_FP8=1)
  [[ "$DRAFT_VOCAB" != none ]] && BASE_ENV+=("VLLM_QWEN38_DRAFT_VOCAB=$HF_HOME_IN_CONTAINER/qwen38-draft-vocab/$DRAFT_VOCAB")
fi
if [[ "$DRAFT_VOCAB" != none && "$DRAFT_HEAD_ON" == 1 ]]; then
  [[ "$LOCAL_ARGMAX" == true ]] || die "DRAFT_VOCAB=$DRAFT_VOCAB needs use_local_argmax_reduction (DRAFT_LOCAL_ARGMAX=1 or SPEC_CONFIG); the reduced head only serves get_top_tokens (docker/v030/MTP_HEAD.md)."
  [[ "${ROLE:-}" == worker || -f "$HF_CACHE/qwen38-draft-vocab/$DRAFT_VOCAB" ]] || die "DRAFT_VOCAB=$DRAFT_VOCAB: $HF_CACHE/qwen38-draft-vocab/$DRAFT_VOCAB is missing. Copy tools/vocab/$DRAFT_VOCAB there on BOTH nodes."
fi

# --- L6a drafter Triton MoE config (tools/kernels/README.md): v0.30 digest only. The head validates the
# folder; a failing check falls closed to the stock default config. The worker gets a copy (like overlays).
DRAFT_MOE_CONFIG_DIR="${DRAFT_MOE_CONFIG_DIR:-$SCRIPT_DIR/docker/v030/moe_configs}"
DRAFT_MOE_CONFIG_MOUNT=/opt/qwen38-moe-configs
# Other digests (the pin rollback) ignore it: the config is tuned against v0.30.0's Triton fused_moe.
[[ "$IMAGE_DIGEST" == "$V030_DIGEST" ]] || DRAFT_MOE_CONFIG=0
if [[ "$DRAFT_MOE_CONFIG" == 1 ]]; then
  if [[ "${ROLE:-}" != worker ]] && ! "$SCRIPT_DIR/tools/kernels/tune_draft_moe.sh" check "$DRAFT_MOE_CONFIG_DIR" >&2; then
    echo "WARNING: DRAFT_MOE_CONFIG=1 but $DRAFT_MOE_CONFIG_DIR fails tune_draft_moe.sh check; the drafter keeps the stock default MoE config." >&2
    DRAFT_MOE_CONFIG=0
  fi
fi

# --- context and occupancy.
if [[ "$MAX_MODEL_LEN" -gt 1048576 && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  die "--max-model-len $MAX_MODEL_LEN is above 1048576. Native max_position_embeddings is 262144. 1M is a lab ceiling. Community 1M YaRN on GB10 hangs on long prefills (vLLM #54629). FORCE_UNSAFE_CTX=1 overrides."
fi
# throughput profile (X10): 16 sequences, the only occupancy row above 8 (evidence/extras/SUMMARY.md).
[[ "$THROUGHPUT_PROFILE" == 1 && "$MAX_NUM_SEQS" == 8 ]] && MAX_NUM_SEQS=16
if [[ "$MAX_NUM_SEQS" -gt 16 && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  die "MAX_NUM_SEQS=$MAX_NUM_SEQS exceeds 16. The occupancy rows are seqs=8 (default) and seqs=16 (THROUGHPUT_PROFILE=1, evidence/extras/SUMMARY.md); above 16 has no row. FORCE_UNSAFE_CTX=1 overrides."
fi
if [[ "$MAX_NUM_SEQS" -gt 8 && "$THROUGHPUT_PROFILE" != 1 && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  die "MAX_NUM_SEQS=$MAX_NUM_SEQS exceeds 8. seqs=8 is the default occupancy pin; 9..16 is the opt-in throughput profile: set THROUGHPUT_PROFILE=1 (occupancy row at 16: evidence/extras/SUMMARY.md). FORCE_UNSAFE_CTX=1 overrides."
fi
if [[ "$MAX_MODEL_LEN" -gt 262144 && "$VLLM_ALLOW_LONG_MAX_MODEL_LEN" != 1 ]]; then
  die "MAX_MODEL_LEN=$MAX_MODEL_LEN exceeds native 262144. vLLM refuses that unless VLLM_ALLOW_LONG_MAX_MODEL_LEN=1. This does not enable YaRN. FORCE_UNSAFE_CTX=1 does not replace that env."
fi
if [[ "$VLLM_SPARSE_INDEXER_MAX_LOGITS_MB" == none && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  if [[ "$MAX_NUM_BATCHED_TOKENS" != "$MEASURED_BATCHED_TOKENS" || "$LONG_PREFILL_TOKEN_THRESHOLD" != none || "$EXTRA_ARGS $EXTRA_ENV" == *indexer_kv_dtype* ]]; then
    die "MAX_NUM_BATCHED_TOKENS, LONG_PREFILL_TOKEN_THRESHOLD or indexer_kv_dtype changed with VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=none. The QSA indexer prefill logits buffer grows with chunk rows; budget 4096 reproduces the GB10 long-prefill hang (vLLM #56457, F05). Set VLLM_SPARSE_INDEXER_MAX_LOGITS_MB (default 64). FORCE_UNSAFE_CTX=1 overrides."
  fi
fi

# --- MoE backend.
case "$MOE_BACKEND" in
  b12x | flashinfer_b12x)
    [[ "$FORCE_UNSAFE_MOE" == 1 ]] || die "MOE_BACKEND=$MOE_BACKEND has open crash reports on this model's MoE shape on GB10 (vLLM #57946: -1 sentinel topk ids give Xid 31; FlashInfer #5446: illegal memory access at E=512 top-10). FORCE_UNSAFE_MOE=1 overrides."
    ;;
esac
if [[ "$ON_PIN" == 1 && "$MOE_BACKEND" != auto && "$FORCE_UNSAFE_MOE" != 1 ]]; then
  die "MOE_BACKEND=$MOE_BACKEND on the pinned nightly. Its V2 runner ignores SPEC_CONFIG moe_backend, so the MTP drafter inherits --moe-backend. The drafter's FP8_BLOCK_SCALES experts are refined to 64x64 blocks at TP=2 and only Triton accepts that block; every other backend fails at drafter construction or at the NVFP4 target. auto is the only value that serves both on sm_121 (F17). FORCE_UNSAFE_MOE=1 overrides."
fi

# --- compilation.
if [[ "$ON_PIN" == 1 && "$ENFORCE_EAGER" != 1 && "$COMPILE_MODE" != 0 && "$COMPILE_MODE" != none && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  die "COMPILATION_CONFIG mode=$COMPILE_MODE on the pinned nightly. Qwen4Exp carries @support_torch_compile there, and Inductor's embedding autotune adds a second copy of the ~23.8 GiB/rank PLE n-gram table on a UMA node with ~11 GiB free (vLLM #55272 removed compile for this reason after GB10 hosts hung; F31). Keep mode 0. FORCE_UNSAFE_CTX=1 overrides."
fi

# --- speculative decoding and sampling. These fail at boot or never ship.
if [[ "$DIAGNOSTIC" != 1 ]]; then
  if [[ "$LOCAL_ARGMAX" == true ]] && ! overlays_define get_top_tokens; then
    die "SPEC_CONFIG use_local_argmax_reduction=true needs Qwen4ExpMTP.get_top_tokens, which neither the pin nor v0.30 has; the drafter raises ValueError at load (G02). Mount a get_top_tokens overlay. DIAGNOSTIC=1 overrides."
  fi
  if [[ "$LOCAL_ARGMAX" == true && "$DRAFT_SAMPLE_METHOD" == probabilistic ]]; then
    die "SPEC_CONFIG use_local_argmax_reduction=true cannot combine with draft_sample_method=probabilistic; vLLM's validator refuses it (speculator.py:343-353, G02). DIAGNOSTIC=1 overrides."
  fi
  if [[ "$ADAPTIVE_VERIFICATION" == true ]]; then
    die "SPEC_CONFIG enable_adaptive_verification=true: the pin allows it only for DSpark (config/speculative.py:1519-1520); v0.30 needs ALWAYS CUDA-graph support, and the QSA/GDN/short-conv builders report UNIFORM_BATCH (G02). DIAGNOSTIC=1 overrides."
  fi
  if [[ "$EXTRA_ARGS" == *--enable-batch-sharded-sampling* ]] && ! overlays_define compute_logits_local; then
    die "--enable-batch-sharded-sampling without a compute_logits_local overlay: Qwen4ExpForCausalLM lacks it, so the engine boots and then raises AttributeError on the first sample (G02). DIAGNOSTIC=1 overrides."
  fi
  for kv in $EXTRA_ENV; do
    if [[ "$kv" == VLLM_BATCH_INVARIANT=1 ]]; then
      die "VLLM_BATCH_INVARIANT=1 cannot boot this model: the attention selector raises for any Mamba backend without supports_batch_invariance(), and no GDN or short-conv backend has it (pin selector.py:229-233; v0.30 selector.py:235-239). DIAGNOSTIC=1 overrides."
    fi
  done
fi
if [[ "$REJECTION_SAMPLE_METHOD" == synthetic && "$BENCH_ONLY" != 1 ]]; then
  die "SPEC_CONFIG rejection_sample_method=synthetic accepts drafts regardless of the target. It is a bench-only engine ruler and must never serve. Set BENCH_ONLY=1 (binds 127.0.0.1)."
fi

API_HOST=0.0.0.0
[[ "$BENCH_ONLY" == 1 ]] && API_HOST=127.0.0.1

if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  printf '==> validate-only spec=%s seqs=%s spec_tokens=%s eager=%s compilation=%s image=%s moe=%s ple=%s host=%s diagnostic=%s\n' \
    "$SPEC" "$MAX_NUM_SEQS" "$NUM_SPECULATIVE_TOKENS" "$ENFORCE_EAGER" "$COMPILATION_CONFIG" \
    "$IMAGE" "$MOE_BACKEND" "$VLLM_PLE_FP8_CHECKPOINT" "$API_HOST" "$DIAGNOSTIC"
  for i in "${!OVERLAY_FILES[@]}"; do
    printf '==> overlay %s -> %s/%s\n' "${OVERLAY_FILES[$i]}" "$SITE_PACKAGES" "${OVERLAY_TARGETS[$i]}"
  done
  [[ ${#OVERLAY_FILES[@]} -gt 0 ]] || printf '==> overlay none for %s\n' "$IMAGE_DIGEST"
  [[ ${#BASE_ENV[@]} -eq 0 ]] || printf '==> base-env %s\n' "${BASE_ENV[*]}"
  printf '==> chat-template %s\n' "${CHAT_TEMPLATE_FILE:-none}"
  printf '==> long-prefill-token-threshold=%s\n' "$LONG_PREFILL_TOKEN_THRESHOLD"
  printf '==> fp8-dense=%s %s\n' "$FP8_DENSE" "${FP8_DENSE_ARGS[*]}"
  printf '==> async-scheduling=%s mm-processor-kwargs=%s limit-mm-per-prompt=%s mm-processor-cache-gb=%s\n' \
    "$ASYNC_SCHEDULING" "$MM_PROCESSOR_KWARGS" "$LIMIT_MM_PER_PROMPT" "$MM_PROCESSOR_CACHE_GB"
  printf '==> spec-config %s\n' "${SPEC_CONFIG:-none}"
  [[ "$DRAFT_MOE_CONFIG" != 1 ]] || printf '==> draft-moe-config %s -> %s (VLLM_TUNED_CONFIG_FOLDER)\n' "$DRAFT_MOE_CONFIG_DIR" "$DRAFT_MOE_CONFIG_MOUNT"
  [[ -z "$EXTRA_ENV" ]] || printf '==> extra-env %s\n' "$EXTRA_ENV"
  exit 0
fi

[[ "$DIAGNOSTIC" == 1 ]] && printf '==> DIAGNOSTIC=1: this boot is never pinned and no tok/s from it is kept.\n'

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

resolve_model() {
  printf '%s\n' "$SNAPSHOT_IN_CONTAINER"
}

maybe_drop_caches() {
  if sudo -n true >/dev/null 2>&1; then
    sync
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
    log "drop_caches: ran"
  else
    log "drop_caches: skipped (no passwordless sudo)"
  fi
}

ensure_image() {
  log "Ensuring image $IMAGE"
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    log "Pulling $IMAGE"
    docker pull "$IMAGE"
  fi
}

snapshot_complete() {
  # Every *.safetensors.index.json shard, config.json and the tokenizer exist and are non-empty.
  python3 - "$SNAPSHOT" <<'PY'
import json
import sys
from pathlib import Path

snap = Path(sys.argv[1])
missing = []


def need(path):
    if not path.is_file() or path.stat().st_size == 0:
        missing.append(path.name)


for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
    need(snap / name)
indexes = sorted(snap.glob("*.safetensors.index.json"))
if not indexes:
    missing.append("*.safetensors.index.json")
for index in indexes:
    try:
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    except (OSError, ValueError, KeyError) as exc:
        missing.append(f"{index.name} ({exc})")
        continue
    for shard in shards:
        need(snap / shard)
if missing:
    more = " ..." if len(missing) > 8 else ""
    sys.exit(f"snapshot {snap} incomplete: {', '.join(missing[:8])}{more}")
PY
}

ensure_weights() {
  if snapshot_complete 2>/dev/null; then
    log "Using pinned snapshot $SNAPSHOT (index shards and tokenizer present)"
    return
  fi
  if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    snapshot_complete || true
    die "SKIP_DOWNLOAD=1 and the pinned snapshot is incomplete."
  fi
  local HF=""
  HF="$(hf_bin || true)"
  [[ -n "$HF" ]] || die "No hf CLI on PATH and snapshot $SNAPSHOT is incomplete."
  export HF_HUB_DISABLE_XET
  log "Downloading $MODEL revision $SNAPSHOT_SHA (resumes under $HF_CACHE; HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET)"
  "$HF" download "$MODEL" --revision "$SNAPSHOT_SHA"
  snapshot_complete || die "Snapshot still incomplete after download."
}

refuse_foreign_serve() {
  local name devices
  while IFS= read -r name; do
    [[ -z "$name" || "$name" == "$CONTAINER_NAME" ]] && continue
    devices="$(docker inspect -f '{{json .HostConfig.DeviceRequests}} {{json .HostConfig.Devices}} {{json .Config.Env}}' "$name" 2>/dev/null || true)"
    if printf '%s' "$devices" | grep -Eqi 'gpu|nvidia|infiniband'; then
      die "$name is using GPUs or InfiniBand. This recipe needs exclusive GPUs on both Sparks. Do not start. Do not docker rm that container from this script."
    fi
  done < <(docker ps --format '{{.Names}}')
}

refuse_busy_port() {
  if (echo >/dev/tcp/127.0.0.1/"$PORT") >/dev/null 2>&1; then
    die "Port $PORT is already in use"
  fi
}

stop_local() {
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    log "Removing existing container $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null
  fi
}

# Host memory watchdog. GB10 is unified memory: when the serve plus host load exhausts RAM and swap,
# the host thrashes for minutes (sshd and LAIL unreachable) before the kernel OOM killer acts, and it
# then picks small services first. Kill the serve instead once available RAM and free swap are both low
# for 3 samples in a row (6 s). Exits when the container is gone.
memguard_loop() {
  local name="$1" min_avail_kb=$(( $2 * 1024 )) min_swap_kb=$(( $3 * 1024 )) hits=0 n=0 avail swapfree
  while :; do
    avail="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    swapfree="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
    if (( avail < min_avail_kb && swapfree < min_swap_kb )); then
      hits=$(( hits + 1 ))
    else
      hits=0
    fi
    if (( hits >= 3 )); then
      logger -t qwen38-memguard "MemAvailable=${avail}kB SwapFree=${swapfree}kB: docker kill $name"
      echo "$(date -Is) MemAvailable=${avail}kB SwapFree=${swapfree}kB: docker kill $name"
      docker kill "$name" >/dev/null 2>&1 || true
      return 0
    fi
    n=$(( n + 1 ))
    if (( n % 8 == 0 )) && [[ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]]; then
      return 0
    fi
    sleep 2
  done
}

start_memguard() {
  [[ "$MEMGUARD" == 1 ]] || { log "memguard off (MEMGUARD=$MEMGUARD)"; return 0; }
  local logf="$STATE_DIR/memguard.log"
  mkdir -p "$STATE_DIR"
  setsid bash -c "$(declare -f memguard_loop); memguard_loop \"\$@\"" memguard \
    "$CONTAINER_NAME" "$MEMGUARD_MIN_AVAIL_MB" "$MEMGUARD_MIN_SWAP_FREE_MB" >>"$logf" 2>&1 </dev/null &
  disown || true
  log "memguard pid=$! min_avail=${MEMGUARD_MIN_AVAIL_MB}MB min_swap_free=${MEMGUARD_MIN_SWAP_FREE_MB}MB log=$logf"
}

start_local() {
  local rank="$1"
  mkdir -p "$HF_CACHE"
  command -v docker >/dev/null 2>&1 || die "docker not found"
  refuse_foreign_serve
  stop_local
  maybe_drop_caches
  ensure_image
  ensure_weights

  local serve_model
  serve_model="$(resolve_model)"

  # The snapshot is complete (ensure_weights), so the container never needs the Hub or a token.
  local env_args=(
    -e "HF_HOME=$HF_HOME_IN_CONTAINER"
    -e "HF_HUB_OFFLINE=1"
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
  if [[ "$VLLM_SPARSE_INDEXER_MAX_LOGITS_MB" != none ]]; then
    env_args+=(-e "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=$VLLM_SPARSE_INDEXER_MAX_LOGITS_MB")
  fi
  local host_ip="$HEAD_IP"
  if [[ "$rank" != "0" ]]; then
    host_ip="$(ip -4 -o addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
    host_ip="${host_ip:-10.100.8.2}"
  fi
  env_args+=(-e "VLLM_HOST_IP=$host_ip")
  local kv
  for kv in "${BASE_ENV[@]}"; do
    env_args+=(-e "$kv")
  done
  [[ "$DRAFT_MOE_CONFIG" != 1 ]] || env_args+=(-e "VLLM_TUNED_CONFIG_FOLDER=$DRAFT_MOE_CONFIG_MOUNT")
  # Later -e wins, so EXTRA_ENV can override the defaults above (e.g. NCCL_DEBUG=INFO).
  env_args+=("${EXTRA_ENV_ARGS[@]}")

  local rank_args=()
  local vol_args=(-v "${HF_CACHE}:${HF_HOME_IN_CONTAINER}")
  if [[ "$rank" == "0" ]]; then
    rank_args+=(--host "$API_HOST" --port "$PORT")
    if [[ -n "$CHAT_TEMPLATE_FILE" ]]; then
      vol_args+=(-v "${CHAT_TEMPLATE_FILE}:${CHAT_TEMPLATE_IN_CONTAINER}:ro")
      rank_args+=(--chat-template "$CHAT_TEMPLATE_IN_CONTAINER")
    fi
  else
    rank_args+=(--headless)
  fi

  local eager_args=()
  if [[ "$ENFORCE_EAGER" == "1" ]]; then
    eager_args+=(--enforce-eager)
  else
    eager_args+=(--compilation-config "$COMPILATION_CONFIG")
  fi

  local i
  for i in "${!OVERLAY_FILES[@]}"; do
    vol_args+=(-v "${OVERLAY_FILES[$i]}:${SITE_PACKAGES}/${OVERLAY_TARGETS[$i]}:ro")
  done
  [[ "$DRAFT_MOE_CONFIG" != 1 ]] || vol_args+=(-v "${DRAFT_MOE_CONFIG_DIR}:${DRAFT_MOE_CONFIG_MOUNT}:ro")
  local batched_args=()
  if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
    batched_args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
  fi
  batched_args+=("${FP8_DENSE_ARGS[@]}")
  if [[ "$LONG_PREFILL_TOKEN_THRESHOLD" != none ]]; then
    batched_args+=(--long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD")
  fi
  local spec_args=()
  if [[ -n "$SPEC_CONFIG" ]]; then
    spec_args+=(--speculative-config "$SPEC_CONFIG")
  fi
  local async_args=(--async-scheduling)
  [[ "$ASYNC_SCHEDULING" == 1 ]] || async_args=(--no-async-scheduling)

  # --ulimit core=1: the kernel skips a piped core dump when RLIMIT_CORE is exactly 1. Otherwise Ubuntu
  # apport buffers the aborting worker's ~11 GiB core in host RAM and swap on both nodes
  # (evidence/u2-v030-default: a 64k-prefill device assert took spark1 to memguard this way).
  log "Starting $CONTAINER_NAME rank=$rank model=$serve_model ctx=$MAX_MODEL_LEN kv=$KV_CACHE_DTYPE spec=$SPEC moe=$MOE_BACKEND overlays=${#OVERLAY_FILES[@]}"
  # shellcheck disable=SC2086 # EXTRA_ARGS is word-split on purpose
  docker run -d \
    --name "$CONTAINER_NAME" \
    --restart no \
    --oom-score-adj "$OOM_SCORE_ADJ" \
    --ulimit core=1 \
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
    "${spec_args[@]}" \
    "${async_args[@]}" \
    --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" \
    --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" \
    --mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB" \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --tool-call-parser "$TOOL_CALL_PARSER" \
    --enable-auto-tool-choice \
    --reasoning-parser "$REASONING_PARSER" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --served-model-name "$SERVED_NAME" \
    --trust-remote-code \
    $EXTRA_ARGS
  start_memguard
}

# Variables the worker rank needs, forwarded shell-quoted over ssh.
FORWARD_VARS=(
  IMAGE CONTAINER_NAME PORT MASTER_PORT HEAD_IP IFACE HCA MAX_MODEL_LEN MAX_NUM_SEQS UTIL
  KV_CACHE_DTYPE TP NNODES SERVED_NAME SKIP_DOWNLOAD HF_HUB_DISABLE_XET SPEC SPEC_CONFIG
  NUM_SPECULATIVE_TOKENS ENFORCE_EAGER COMPILATION_CONFIG MAX_NUM_BATCHED_TOKENS
  VLLM_SPARSE_INDEXER_MAX_LOGITS_MB FORCE_UNSAFE_CTX FORCE_UNSAFE_MOE DIAGNOSTIC BENCH_ONLY
  VLLM_PLE_FP8_CHECKPOINT VLLM_ALLOW_LONG_MAX_MODEL_LEN TOOL_CALL_PARSER REASONING_PARSER
  MOE_BACKEND SNAPSHOT_SHA HF_CACHE MODEL EXTRA_ARGS EXTRA_ENV
  OOM_SCORE_ADJ MEMGUARD MEMGUARD_MIN_AVAIL_MB MEMGUARD_MIN_SWAP_FREE_MB
  LONG_PREFILL_TOKEN_THRESHOLD ASYNC_SCHEDULING MM_PROCESSOR_CACHE_GB MM_MIN_PIXELS MM_MAX_PIXELS MM_LIMIT_IMAGE MM_LIMIT_VIDEO
  DRAFT_MOE_CONFIG DRAFT_LOCAL_ARGMAX DRAFT_HEAD_FP8 DRAFT_VOCAB FP8_DENSE THROUGHPUT_PROFILE GDN_LAZY
)

worker_env() {
  # $1: space-separated overlay paths on the worker.
  local v out
  # The chat template is an API-server (head) setting; the worker copy of run.sh has no template file.
  out="ROLE=worker ORCHESTRATE=0 CHAT_TEMPLATE=none OVERLAYS=$(printf '%q' "${1:-none}") DRAFT_MOE_CONFIG_DIR=/tmp/qwen38-moe-configs"
  for v in "${FORWARD_VARS[@]}"; do
    out+=" $v=$(printf '%q' "${!v}")"
  done
  printf '%s\n' "$out"
}

worker_state() {
  # Prints true, false or missing; prints nothing when ssh itself fails. docker inspect prints an
  # empty line before failing on a missing container, so strip whitespace.
  { ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" \
    "docker inspect -f '{{.State.Running}}' '$CONTAINER_NAME' 2>/dev/null || echo missing" 2>/dev/null || true; } |
    tr -d '[:space:]'
}

abort_worker_dead() {
  echo "Worker $CONTAINER_NAME on $WORKER_HOST is not running ($1). Worker logs:" >&2
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" "docker logs --tail 120 '$CONTAINER_NAME'" >&2 2>&1 || true
  echo "Stop the head with ./stop.sh" >&2
  exit 1
}

wait_ready() {
  local watch_worker="${1:-0}"
  log "Waiting for http://127.0.0.1:${PORT}/health and /v1/models"
  local i health body state
  for i in $(seq 1 720); do
    health="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" || true)"
    body="$(curl -sf "http://127.0.0.1:${PORT}/v1/models" || true)"
    if [[ "$health" == "200" && -n "$body" && "$body" == *"$SERVED_NAME"* ]]; then
      log "Ready → http://${API_HOST}:${PORT}/v1  (context=$MAX_MODEL_LEN health=$health)"
      printf '%s\n' "$body"
      echo
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
      echo "Container exited early. Logs:" >&2
      docker logs "$CONTAINER_NAME" 2>&1 | tail -120 >&2
      exit 1
    fi
    # Every ~10 s (plus up to 5 s ssh timeout), so a dead worker aborts the head within 60 s.
    if [[ "$watch_worker" == 1 ]] && (( i % 2 == 0 )); then
      state="$(worker_state)"
      [[ "$state" == false || "$state" == missing ]] && abort_worker_dead "$state"
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
  watch_worker=0
  if [[ "$NNODES" -gt 1 ]]; then
    if ! command -v ssh >/dev/null 2>&1 || ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" true >/dev/null 2>&1; then
      die "Cannot SSH to $WORKER_HOST. Refusing to start a TP=$TP head rank alone (NNODES=$NNODES)."
    fi
    log "Starting worker on $WORKER_HOST first"
    mkdir -p "$STATE_DIR"
    printf '%s\n' "$WORKER_HOST" >"$STATE_DIR/worker_host"
    scp -q "$0" "${WORKER_HOST}:/tmp/qwen38-run.sh"
    remote_overlays=""
    if [[ ${#OVERLAY_FILES[@]} -gt 0 ]]; then
      ssh "$WORKER_HOST" "rm -rf /tmp/qwen38-overlays && mkdir -p /tmp/qwen38-overlays"
      for f in "${OVERLAY_FILES[@]}"; do
        # Upstream paths are unique, so flatten them into unique remote names.
        remote="/tmp/qwen38-overlays/$(overlay_field "$f" upstream_file | tr '/' '_')"
        scp -q "$f" "${WORKER_HOST}:$remote"
        remote_overlays+="$remote "
      done
    fi
    if [[ "$DRAFT_MOE_CONFIG" == 1 ]]; then
      ssh "$WORKER_HOST" "rm -rf /tmp/qwen38-moe-configs && mkdir -p /tmp/qwen38-moe-configs"
      scp -q "$DRAFT_MOE_CONFIG_DIR"/*.json "${WORKER_HOST}:/tmp/qwen38-moe-configs/"
    fi
    ssh "$WORKER_HOST" "$(worker_env "$remote_overlays") bash /tmp/qwen38-run.sh"
    log "Worker container started. Waiting 25s for NCCL listen, then starting head"
    sleep 25
    state="$(worker_state)"
    [[ "$state" == false || "$state" == missing ]] && abort_worker_dead "$state"
    watch_worker=1
  fi
  start_local 0
  wait_ready "$watch_worker"
  log "Stop with: ./stop.sh"
elif [[ "$ROLE" == "worker" ]]; then
  start_local 1
  log "Worker rank 1 is up. Head should start next."
else
  start_local 0
  wait_ready 0
  log "Stop with: ./stop.sh"
fi
