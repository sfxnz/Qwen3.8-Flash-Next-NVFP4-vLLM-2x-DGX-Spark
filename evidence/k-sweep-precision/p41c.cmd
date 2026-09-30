# Ladder step 3: GDN in_proj_qkvz (the q/k/v/z rows; one merged parameter, so all four) back to BF16.
OVL="docker/v030/flashinfer_cutlass_moe.py docker/v030/gdn_attn.py docker/v030/qsa_indexer.py docker/v030/serving.py docker/v030/mtp.py docker/v030/modelopt.py"
ALLOW='linear_attn\.out_proj$,self_attn\.qkv_proj$,self_attn\.o_proj$,ple\.kv_proj$'
evidence/k-sweep-precision/boot.sh p41c-fp8-blk-noqkvz "OVERLAYS_V030=$OVL" "EXTRA_ENV=VLLM_QWEN38_FP8_DENSE=per_block VLLM_QWEN38_FP8_DENSE_ALLOW=$ALLOW" \
  'EXTRA_ARGS=--kernel-config={"linear_backend_per_quant":{"fp8_block_w8a8":"marlin"}}' || exit $?
ANCHOR="0 0" evidence/k-sweep-precision/steps.sh p41c-fp8-blk-noqkvz free receipts t1full t1long gate ruler-noanchor t1d
