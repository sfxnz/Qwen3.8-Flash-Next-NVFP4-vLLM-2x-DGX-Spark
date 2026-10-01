#!/usr/bin/env bash
# L6a: Triton fused-MoE config for the MTP drafter's FP8 block-scaled experts on GB10.
# Shape (U1 boot log, both ranks): E=512, N=320 per rank at TP=2, dtype=fp8_w8a8,
# block [64,64] after the 64x64 refine. vLLM v0.30.0 looks the config up as
#   E=512,N=320,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[64,64].json
# first in $VLLM_TUNED_CONFIG_FOLDER, then in fused_moe/configs/ (fused_moe.py get_moe_configs).
#
# Subcommands (only tune and bench touch a GPU; they refuse while any GPU container is up):
#   seed           rewrite the seed from the image's B200 file for the same shape (CPU, no GPU)
#   check [DIR]    validate every JSON in DIR (default docker/v030/moe_configs); host python3, no torch
#   synth DIR      write the synthetic HF config.json the tuner reads (Qwen3Moe stand-in)
#   tune           run benchmark_moe.py --tune on THIS Spark's one GPU; output under OUT
#   bench          benchmark_moe.py without --tune: stock default config vs CFG dir, ABAB, per batch size
#   wrapper        print the in-container Python wrapper (tests exec it on CPU in "plan" mode)
#   --dry-run as the first argument prints the docker command instead of running it.
#
# Env: IMAGE (digest-pinned v0.30.0), OUT (default evidence/l6a-tune/<UTC stamp>),
#      BATCH_SIZES (default: vLLM tuner list up to 4096, plus 8192 = MAX_NUM_BATCHED_TOKENS).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY=1; shift; fi
CMD="${1:-}"; shift || true

IMAGE="${IMAGE:-vllm/vllm-openai:v0.30.0-aarch64@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56}"
CONFIG_DIR="$ROOT/docker/v030/moe_configs"
CFG_NAME='E=512,N=320,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[64,64].json'
SEED_SRC='E=512,N=320,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[64,64].json'
STOCK_CONFIGS=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 24 32 48 64 96 128 256 512 1024 1536 2048 3072 4096 8192}"

die() { echo "tune_draft_moe: $*" >&2; exit 1; }
run() { if (( DRY )); then printf '%q ' "$@"; echo; else "$@"; fi; }

[[ "$IMAGE" == *@sha256:* ]] || die "IMAGE=$IMAGE is not digest-pinned."

gpu_guard() {
  (( DRY )) && return 0
  local id
  for id in $(docker ps -q); do
    # --gpus (DeviceRequests) or --runtime nvidia both hold the GPU.
    if docker inspect --format '{{.HostConfig.Runtime}} {{json .HostConfig.DeviceRequests}}' "$id" \
        | grep -qE 'nvidia|gpu'; then
      die "container $(docker inspect --format '{{.Name}}' "$id") holds the GPU. GPUs are exclusive; stop it first."
    fi
  done
}

cmd_check() {
  local dir="${1:-$CONFIG_DIR}"
  CFG_NAME="$CFG_NAME" python3 - "$dir" <<'PY'
import json, os, sys
d = sys.argv[1]
want = os.environ["CFG_NAME"]
SMEM_OPTIN = 101376  # GB10 smem_per_block_optin (evidence/s11-micro device facts)
KEYS = {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "num_warps", "num_stages"}
errs = []
names = sorted(f for f in os.listdir(d) if f.endswith(".json")) if os.path.isdir(d) else []
if names != [want]:
    errs.append(f"{d}: expected exactly [{want}], found {names}")
for name in names:
    try:
        cfg = json.load(open(os.path.join(d, name)))
    except Exception as e:  # noqa: BLE001
        errs.append(f"{name}: not JSON: {e}")
        continue
    if not isinstance(cfg, dict):
        errs.append(f"{name}: top level is not an object")
        continue
    # get_moe_configs pops triton_version, then int()s every other key.
    tv = cfg.pop("triton_version", None)
    if tv is not None and not isinstance(tv, str):
        errs.append(f"{name}: triton_version is not a string")
    bad = [k for k in cfg if not k.isdigit()]
    if bad:
        errs.append(f"{name}: non-integer batch keys {bad}")
        continue
    if 1 not in {int(k) for k in cfg}:
        errs.append(f"{name}: no entry for M=1")
    for k, c in cfg.items():
        extra = set(c) - KEYS - {"SPLIT_K"}
        if set(c) & KEYS != KEYS or extra:
            errs.append(f"{name}[{k}]: keys {sorted(c)}")
            continue
        if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in c.values()):
            errs.append(f"{name}[{k}]: non-positive or non-int value")
            continue
        if c.get("SPLIT_K", 1) != 1:
            errs.append(f"{name}[{k}]: SPLIT_K must be 1 (the kernel forces 1)")
        bm, bn, bk = c["BLOCK_SIZE_M"], c["BLOCK_SIZE_N"], c["BLOCK_SIZE_K"]
        for tag, v in (("BLOCK_SIZE_M", bm), ("BLOCK_SIZE_N", bn), ("BLOCK_SIZE_K", bk),
                       ("GROUP_SIZE_M", c["GROUP_SIZE_M"]), ("num_warps", c["num_warps"])):
            if v & (v - 1):
                errs.append(f"{name}[{k}]: {tag}={v} is not a power of two")
        if bm < 16 or bn < 16 or bk < 16:
            errs.append(f"{name}[{k}]: tile below 16 (tl.dot minimum)")
        # Tile must align to the [64,64] scale grid (benchmark_moe.py filter).
        if not (bn % 64 == 0 or 64 % bn == 0) or not (bk % 64 == 0 or 64 % bk == 0):
            errs.append(f"{name}[{k}]: tile ({bn},{bk}) does not align to block [64,64]")
        if c["num_warps"] > 8 or not 1 <= c["num_stages"] <= 8:
            errs.append(f"{name}[{k}]: num_warps/num_stages out of range")
        # Conservative fp8 A+B pipeline estimate; the runtime clamps BLOCK_SIZE_K to 64.
        smem = c["num_stages"] * (bm * min(bk, 64) + min(bk, 64) * bn)
        if smem > SMEM_OPTIN:
            errs.append(f"{name}[{k}]: ~{smem} B shared memory > GB10 opt-in {SMEM_OPTIN}")
if errs:
    print("CHECK FAIL\n  " + "\n  ".join(errs))
    sys.exit(1)
print(f"CHECK OK {d}/{want}")
PY
}

cmd_seed() {
  # Seed = the image's B200 file for the identical shape, byte for byte, renamed for GB10.
  # All 14 batch keys (1..8192) are kept: nearest-key lookup would otherwise hand
  # 8192-token prefill chunks the M=64 tile. Every entry is replaced by `tune`.
  mkdir -p "$CONFIG_DIR"
  if (( DRY )); then
    run docker run --rm --network none --entrypoint cat "$IMAGE" "$STOCK_CONFIGS/$SEED_SRC"
    return 0
  fi
  docker run --rm --network none --entrypoint cat "$IMAGE" "$STOCK_CONFIGS/$SEED_SRC" \
    > "$CONFIG_DIR/$CFG_NAME.tmp" || { rm -f "$CONFIG_DIR/$CFG_NAME.tmp"; die "seed extract failed"; }
  mv "$CONFIG_DIR/$CFG_NAME.tmp" "$CONFIG_DIR/$CFG_NAME"
  cmd_check "$CONFIG_DIR"
}

cmd_synth() {
  # Stand-in HF config for benchmark_moe.get_model_params: Qwen4ExpForConditionalGeneration
  # is not in its table. Values from nvidia/Qwen3.8-Flash-Next-NVFP4@fab0aec config.json:
  # num_experts 512, num_experts_per_tok 10, moe_intermediate_size 640, hidden_size 2560, bf16.
  # weight_block_size [64,64] is the MTP experts' block AFTER the refine (checkpoint group 128).
  local d="${1:?synth DIR}"
  mkdir -p "$d"
  cat > "$d/config.json" <<'JSON'
{
  "architectures": ["Qwen3MoeForCausalLM"],
  "model_type": "qwen3_moe",
  "num_experts": 512,
  "num_experts_per_tok": 10,
  "moe_intermediate_size": 640,
  "hidden_size": 2560,
  "dtype": "bfloat16",
  "quantization_config": {"quant_method": "fp8", "weight_block_size": [64, 64]}
}
JSON
}

# Tuner wrapper: stock benchmark_moe.py from the image, run in-process on one GPU.
#  - The v0.30.0 image has no `ray`. A stub runs BenchmarkWorker directly and in
#    sequence (one Spark = one GPU); everything else is the stock tuner/benchmark code.
#  - BLOCK_SIZE_K fixed at 64: invoke_fused_moe_triton_kernel clamps it to min(block_shape)=64,
#    so 128/256 compile the same kernel three times.
#  - num_warps 2 added: the seed (B200) chose it for M<=8, the draft-step batch sizes.
# Modes: tune | bench (GPU), plan (CPU: print search-space size and output file name).
TUNE_PY='
import argparse, os, sys, types
import tqdm as _tqdm

class _Handle:
    def __init__(self, obj): self._obj = obj
    def __getattr__(self, name): return types.SimpleNamespace(remote=getattr(self._obj, name))

class _Actor:
    def __init__(self, cls): self._cls = cls
    def remote(self, *a, **k): return _Handle(self._cls(*a, **k))

ray = types.ModuleType("ray")
ray.remote = lambda *a, **k: _Actor
ray.init = lambda *a, **k: None
ray.get_gpu_ids = lambda: [0]
ray.available_resources = lambda: {"GPU": 1}
ray.get = lambda x: x
tq = types.ModuleType("ray.experimental.tqdm_ray"); tq.tqdm = _tqdm.tqdm
ray.experimental = types.ModuleType("ray.experimental"); ray.experimental.tqdm_ray = tq
sys.modules.update({"ray": ray, "ray.experimental": ray.experimental, "ray.experimental.tqdm_ray": tq})

sys.path.insert(0, os.environ.get("BENCH_DIR", "/vllm-workspace/benchmarks/kernels"))
import benchmark_moe as bm

_stock = bm.get_configs_compute_bound
def _space(use_fp16, block):
    out = []
    for c in _stock(use_fp16, block):
        if c["BLOCK_SIZE_K"] != 64:
            continue
        out.append(c)
        if c["num_warps"] == 4:
            out.append({**c, "num_warps": 2})
    return out
bm.get_configs_compute_bound = _space

mode = sys.argv[1]
synth = os.environ.get("SYNTH", "/synth")
if mode == "plan":
    cfg = bm.get_config(model=synth, trust_remote_code=False)
    E, topk, inter, hidden = bm.get_model_params(cfg)
    block = bm.get_weight_block_size_safety(cfg)
    PLAN = {"E": E, "topk": topk, "N": 2 * inter // 2 // 2, "hidden": hidden, "block": block,
            "space": len(_space(False, block)),
            "file": bm.get_config_file_name(E, 2 * inter // 2 // 2, "fp8_w8a8", block)}
    print("PLAN", PLAN)
else:
    bm.main(argparse.Namespace(
        model=synth, tp_size=2, enable_expert_parallel=False, dtype="fp8_w8a8",
        use_deep_gemm=False, save_dir=os.environ.get("SAVE_DIR", "/out"), seed=0,
        batch_size=[int(x) for x in os.environ["BATCH_SIZES"].split()],
        tune=(mode == "tune"), trust_remote_code=False, model_prefix=None))
'

gpu_docker() {  # gpu_docker OUT [extra docker args...] -- python args
  # Triton cache stays inside the --rm container: the container runs as root, so a cache
  # under the /out bind mount is root-owned and the host user cannot rm it (set -e abort).
  local out="$1"; shift
  run docker run --rm --gpus all --ipc host --network none \
    -e BATCH_SIZES="$BATCH_SIZES" -e PYTHONPATH=/vllm-workspace/benchmarks/kernels \
    -e TRITON_CACHE_DIR=/tmp/triton-cache \
    -v "$out/synth:/synth:ro" -v "$out:/out" "$@"
}

cmd_tune() {
  gpu_guard
  local out="${OUT:-$ROOT/evidence/l6a-tune/$(date -u +%Y%m%dT%H%M%SZ)}"
  mkdir -p "$out"; out="$(cd "$out" && pwd)"  # docker -v needs absolute host paths
  cmd_synth "$out/synth"
  { free -h; docker image inspect "$IMAGE" --format '{{.RepoDigests}}' 2>/dev/null || true; } > "$out/pre.txt"
  # Self-test on the real device first: the wrapper must target exactly CFG_NAME.
  gpu_docker "$out" --entrypoint python3 "$IMAGE" -c "$TUNE_PY" plan 2>&1 | tee "$out/plan.log"
  (( DRY )) || grep -qF "'file': '$CFG_NAME'" "$out/plan.log" \
    || die "plan does not target $CFG_NAME (see $out/plan.log)"
  gpu_docker "$out" --entrypoint python3 "$IMAGE" -c "$TUNE_PY" tune 2>&1 | tee "$out/tune.log"
  (( DRY )) && return 0
  [[ -f "$out/$CFG_NAME" ]] || die "tuner wrote no $CFG_NAME in $out (see tune.log)"
  local stage="$out/install"; mkdir -p "$stage"; cp "$out/$CFG_NAME" "$stage/"
  cmd_check "$stage"
  echo "Next: bench it, then install with"
  echo "  CFG=\"$stage\" $0 bench"
  echo "  cp '$stage/$CFG_NAME' '$CONFIG_DIR/' && $0 check"
}

cmd_bench() {
  # Stock arm: no VLLM_TUNED_CONFIG_FOLDER (image has no GB10 file -> get_default_config).
  # Config arm: VLLM_TUNED_CONFIG_FOLDER=CFG (default docker/v030/moe_configs).
  gpu_guard
  local cfg="${CFG:-$CONFIG_DIR}"
  local out="${OUT:-$ROOT/evidence/l6a-tune/bench-$(date -u +%Y%m%dT%H%M%SZ)}"
  [[ -d "$cfg" ]] || die "CFG=$cfg is not a directory"
  cfg="$(cd "$cfg" && pwd)"  # docker -v needs absolute host paths
  (( DRY )) || cmd_check "$cfg"
  mkdir -p "$out"; out="$(cd "$out" && pwd)"; cmd_synth "$out/synth"
  local arm
  for arm in stock config stock config; do  # ABAB
    local extra=()
    [[ "$arm" == config ]] && extra=(-v "$cfg:/cfg:ro" -e VLLM_TUNED_CONFIG_FOLDER=/cfg)
    echo "== arm $arm" | tee -a "$out/bench.log"
    gpu_docker "$out" "${extra[@]}" --entrypoint python3 "$IMAGE" -c "$TUNE_PY" bench \
      2>&1 | tee -a "$out/bench.log"
  done
}

case "$CMD" in
  seed) cmd_seed ;;
  check) cmd_check "$@" ;;
  synth) cmd_synth "$@" ;;
  tune) cmd_tune ;;
  bench) cmd_bench ;;
  wrapper) printf '%s' "$TUNE_PY" ;;
  -h | --help | "") sed -n '2,20p' "$0" ;;
  *) die "unknown subcommand $CMD (seed|check|synth|tune|bench|wrapper)" ;;
esac
