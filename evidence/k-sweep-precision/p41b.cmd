OVL="docker/v030/flashinfer_cutlass_moe.py docker/v030/gdn_attn.py docker/v030/qsa_indexer.py docker/v030/serving.py docker/v030/mtp.py docker/v030/modelopt.py"
evidence/k-sweep-precision/boot.sh p41b-fp8-ptpc "OVERLAYS_V030=$OVL" "EXTRA_ENV=VLLM_QWEN38_FP8_DENSE=ptpc" || exit $?
ANCHOR="0 0" evidence/k-sweep-precision/steps.sh p41b-fp8-ptpc free receipts t1full t1long gate ruler-noanchor t1d
