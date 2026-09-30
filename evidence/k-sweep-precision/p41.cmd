OVL="docker/v030/flashinfer_cutlass_moe.py docker/v030/gdn_attn.py docker/v030/qsa_indexer.py docker/v030/serving.py docker/v030/mtp.py docker/v030/modelopt.py"
evidence/k-sweep-precision/boot.sh p41-fp8-blk-marlin "OVERLAYS_V030=$OVL" "EXTRA_ENV=VLLM_QWEN38_FP8_DENSE=per_block" \
  'EXTRA_ARGS=--kernel-config={"linear_backend_per_quant":{"fp8_block_w8a8":"marlin"}}' || exit $?
ANCHOR="0 0" evidence/k-sweep-precision/steps.sh p41-fp8-blk-marlin free receipts gate ruler-noanchor diverse t1g identity t1full t1long t1d t2full t3 vision
