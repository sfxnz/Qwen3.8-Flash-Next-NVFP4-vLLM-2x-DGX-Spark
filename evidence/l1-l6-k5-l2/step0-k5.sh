#!/usr/bin/env bash
# Step 0a: K5 GPU numerics + chain bench, one GB10, serve DOWN (no second CUDA process next to a serve).
set -uo pipefail
cd "$(dirname "$0")/../.."
V030=vllm/vllm-openai:v0.30.0-aarch64@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56
OUT=evidence/l1-l6-k5-l2/step0-k5
mkdir -p "$OUT"
if docker ps --format '{{.Names}}' | grep -q qwen38; then echo "serve is up; refusing" >&2; exit 3; fi
free -h > "$OUT/free-pre.txt"
D=(docker run --rm --gpus all --ipc host --network none --ulimit core=1 -v "$PWD":/w -w /w
   -e TRITON_CACHE_DIR=/tmp/triton-cache --entrypoint python3 "$V030")
"${D[@]}" -m unittest tests.test_hc_fused -v > "$OUT/test_hc_fused-gpu.log" 2>&1; echo "test rc=$?" | tee "$OUT/test.exit"
for v in main final; do
  "${D[@]}" tools/kernels/hc_bench.py run --variant "$v" --json "$OUT/hc_bench-$v.json" > "$OUT/hc_bench-$v.log" 2>&1
  echo "bench $v rc=$?" | tee -a "$OUT/bench.exit"
done
free -h > "$OUT/free-post.txt"
