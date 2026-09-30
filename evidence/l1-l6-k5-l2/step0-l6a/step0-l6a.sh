#!/usr/bin/env bash
# Step 0b: L6a drafter MoE tune on ONE GB10 (spark2, serve DOWN), time-boxed to DEADLINE_MIN (45).
# tune_draft_moe.sh tune writes the file only after all its batch sizes finish, so the batch list is
# run one size at a time in draft-step priority order (M<=32 first) and every finished size is kept.
# Run on spark2 from a copy of tools/kernels/tune_draft_moe.sh at $DIR/tools/kernels/.
set -uo pipefail
DIR="${DIR:-$HOME/projects/ai-lab/recipes/qwen38-l6a-tune}"
DEADLINE_MIN="${DEADLINE_MIN:-45}"
SIZES="${SIZES:-1 2 4 8 16 24 32 48 64 96 128 256 512 1024 1536 2048 3072 4096 8192}"
end=$(( $(date +%s) + DEADLINE_MIN * 60 ))
cd "$DIR"
mkdir -p "$DIR/out"
for bs in $SIZES; do
  now=$(date +%s)
  (( now < end )) || { echo "deadline: stop before bs=$bs"; break; }
  t0=$now
  BATCH_SIZES="$bs" OUT="$DIR/out/bs$bs" timeout $(( end - now + 60 )) tools/kernels/tune_draft_moe.sh tune > "$DIR/out/bs$bs.log" 2>&1
  rc=$?
  if (( rc == 124 )); then  # the docker CLI died; the --rm container may still hold the GPU
    docker ps -q --filter ancestor=vllm/vllm-openai:v0.30.0-aarch64 | xargs -r docker kill
  fi
  echo "bs=$bs rc=$rc s=$(( $(date +%s) - t0 ))"
done
