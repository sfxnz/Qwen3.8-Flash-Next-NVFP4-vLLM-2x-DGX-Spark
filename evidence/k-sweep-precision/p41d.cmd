# Calibration: FP8 on ONE layer (PLE kv_proj, replicated, 65 MB). Does T1 top-1 scale with the perturbation?
OVL="docker/v030/flashinfer_cutlass_moe.py docker/v030/gdn_attn.py docker/v030/qsa_indexer.py docker/v030/serving.py docker/v030/mtp.py docker/v030/modelopt.py"
evidence/k-sweep-precision/boot.sh p41d-fp8-blk-plekv-only "OVERLAYS_V030=$OVL" 'EXTRA_ENV=VLLM_QWEN38_FP8_DENSE=per_block VLLM_QWEN38_FP8_DENSE_ALLOW=ple\.kv_proj$' \
  'EXTRA_ARGS=--kernel-config={"linear_backend_per_quant":{"fp8_block_w8a8":"marlin"}}' || exit $?
ANCHOR="0 0" evidence/k-sweep-precision/steps.sh p41d-fp8-blk-plekv-only free receipts t1full t1long t1d
