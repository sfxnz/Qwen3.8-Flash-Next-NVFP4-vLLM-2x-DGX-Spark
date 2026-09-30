#!/usr/bin/env bash
# Step 0a follow-up: K5 tile sweep at M=4,32 (main variant), serve DOWN, one GPU.
set -uo pipefail
cd "$(dirname "$0")/../.."
V030=vllm/vllm-openai:v0.30.0-aarch64@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56
OUT=evidence/l1-l6-k5-l2/step0-k5/sweep; mkdir -p "$OUT"
D=(docker run --rm --gpus all --ipc host --network none --ulimit core=1 -v "$PWD":/w -w /w
   -e TRITON_CACHE_DIR=/tmp/triton-cache --entrypoint python3 "$V030")
"${D[@]}" -m unittest discover -s tests -p test_hc_fused.py -v > evidence/l1-l6-k5-l2/step0-k5/test_hc_fused-gpu.log 2>&1
echo "test rc=$?" | tee evidence/l1-l6-k5-l2/step0-k5/test.exit
for cfg in block_j=8 block_j=32 block_j=64 block_k_up=32 block_k_up=128 split_k=4 split_k=16 block_n=32 block_k=256 "split_k=16,block_j=8"; do
  tag=$(echo "$cfg" | tr ',=' '_-')
  "${D[@]}" tools/kernels/hc_bench.py run --variant main --ms 4,32 --no-kernels --cfg "$cfg" --json "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1
  echo "$cfg: $(grep HC-K5 "$OUT/$tag.log" | sed 's/.*stream stock/stock/' | tr '\n' ' ')"
done | tee "$OUT/summary.txt"
