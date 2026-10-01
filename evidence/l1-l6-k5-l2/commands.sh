#!/usr/bin/env bash
# Session 4 (2026-09-29): lossless decode levers L6a, L1a, L1b', L1b, K5, L2a on the U2 default.
# Run from the recipe worktree root (opt/plan-levers). boot.sh = stop, memcheck, ./run.sh with overrides;
# steps.sh = per-boot gate stages. Raw data: ~/projects/data/qwen38-evals/runs/session4/<arm>/.
# Every arm: gate (smokes, T0, ruler --anchor-ms 60.7 62.7, frozen bench_decode), diverse c1/c8,
# T1-G 10x5 self + identity vs u2-base; class-A arms add t1nll + t2q (gsm8k-250, tools, json).
V030=vllm/vllm-openai:v0.30.0-aarch64@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56
BASE_OVL="docker/v030/flashinfer_cutlass_moe.py docker/v030/gdn_attn.py docker/v030/qsa_indexer.py docker/v030/serving.py"

# --- u2-base: the live c-final default (booted by session 3), measured once for this session
evidence/l1-l6-k5-l2/steps.sh u2-base free gate diverse t1g identity t1nll t2q receipts

# --- step 0 (serve DOWN; ./stop.sh). 0a K5 on spark1, 0b L6a tune on spark2, in parallel (one GPU each)
./stop.sh
evidence/l1-l6-k5-l2/step0-k5.sh                 # GPU unit tests + hc_bench main/final (first test call used a
                                                 # non-package module path; rerun inside step0-k5-sweep.sh)
evidence/l1-l6-k5-l2/step0-k5-sweep.sh           # tests/test_hc_fused.py on CUDA (13 OK) + 10-cfg tile sweep at M=4,32
#   plus 4 combined tiles (block_j/block_k_up/split_k/block_n), same docker run, results in step0-k5/sweep/summary.txt
#   -> NO-GO: best 0.941x at M=4, 0.972x at M=32 (gate <= 0.95x both). Arm D not booted.
# spark2: copy tools/kernels/tune_draft_moe.sh + step0-l6a.sh to ~/projects/ai-lab/recipes/qwen38-l6a-tune/
ssh spark2 'cd ~/projects/ai-lab/recipes/qwen38-l6a-tune && ./step0-l6a.sh'   # M=1 search alone took 50.5 min;
#   the size loop was stopped after M=1 and its timeout wrapper killed so the finished search was kept
python3 evidence/l1-l6-k5-l2/merge_l6a.py docker/v030/moe_configs/<CFG> evidence/l1-l6-k5-l2/step0-l6a/install/<CFG> evidence/l1-l6-k5-l2/step0-l6a/bs1
CFG=evidence/l1-l6-k5-l2/step0-l6a/install OUT=evidence/l1-l6-k5-l2/step0-l6a/bench BATCH_SIZES="1 2 4 8 32" tools/kernels/tune_draft_moe.sh bench
cp evidence/l1-l6-k5-l2/step0-l6a/install/<CFG> docker/v030/moe_configs/ && tools/kernels/tune_draft_moe.sh check
# run.sh: DRAFT_MOE_CONFIG (mount docker/v030/moe_configs as VLLM_TUNED_CONFIG_FOLDER, v0.30 only) + tests

# --- arm A: L6a config + L1a (mtp.py overlay + use_local_argmax_reduction)
SPEC_A='{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton","use_local_argmax_reduction":true}'
evidence/l1-l6-k5-l2/boot.sh a-l6a-l1a DRAFT_MOE_CONFIG=1 "OVERLAYS_V030=$BASE_OVL docker/v030/mtp.py" "SPEC_CONFIG=$SPEC_A"
evidence/l1-l6-k5-l2/steps.sh a-l6a-l1a free gate diverse t1g identity receipts

# --- arm B: A + L1b' FP8 draft head (anchored ruler voids: sentinel 54.7 ms < 60.7 -> anchor-free rerun)
evidence/l1-l6-k5-l2/boot.sh b-fp8head DRAFT_MOE_CONFIG=1 "OVERLAYS_V030=$BASE_OVL docker/v030/mtp.py" "SPEC_CONFIG=$SPEC_A" \
  "EXTRA_ENV=VLLM_QWEN38_DRAFT_HEAD_FP8=1"
evidence/l1-l6-k5-l2/steps.sh b-fp8head free gate diverse t1g identity receipts
evidence/l1-l6-k5-l2/steps.sh b-fp8head ruler-noanchor

# --- arm C: B + L1b 131072; arm C2: B + L1b 163840 (C lost 8% CJK acceptance)
evidence/l1-l6-k5-l2/boot.sh c-vocab131k-fp8 DRAFT_MOE_CONFIG=1 "OVERLAYS_V030=$BASE_OVL docker/v030/mtp.py" "SPEC_CONFIG=$SPEC_A" \
  "EXTRA_ENV=VLLM_QWEN38_DRAFT_HEAD_FP8=1 VLLM_QWEN38_DRAFT_VOCAB=/cache/huggingface/qwen38-draft-vocab/draft_vocab_131072.json"
evidence/l1-l6-k5-l2/steps.sh c-vocab131k-fp8 free gate ruler-noanchor diverse t1g identity receipts
evidence/l1-l6-k5-l2/boot.sh c2-vocab164k-fp8 DRAFT_MOE_CONFIG=1 "OVERLAYS_V030=$BASE_OVL docker/v030/mtp.py" "SPEC_CONFIG=$SPEC_A" \
  "EXTRA_ENV=VLLM_QWEN38_DRAFT_HEAD_FP8=1 VLLM_QWEN38_DRAFT_VOCAB=/cache/huggingface/qwen38-draft-vocab/draft_vocab_163840.json"
evidence/l1-l6-k5-l2/steps.sh c2-vocab164k-fp8 free gate ruler-noanchor diverse t1g identity receipts
python3 evidence/l1-l6-k5-l2/domain_acc.py u2-base b-fp8head c-vocab131k-fp8 c2-vocab164k-fp8
# -> B chosen (no CJK loss). run.sh: DRAFT_LOCAL_ARGMAX / DRAFT_HEAD_FP8 / DRAFT_VOCAB knobs + tests

# --- arm E: B + L2a. e0 = engagement audit (NCCL INFO), e1 = timing (NCCL WARN) + 20-min soak
OVL_E="$BASE_OVL docker/v030/mtp.py docker/v030/nccl_twin_cuda_communicator.py"
evidence/l1-l6-k5-l2/boot.sh e0-l2a-audit DRAFT_MOE_CONFIG=1 DRAFT_LOCAL_ARGMAX=1 DRAFT_HEAD_FP8=1 "OVERLAYS_V030=$OVL_E" \
  "EXTRA_ENV=VLLM_QWEN38_NCCL_TWIN=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ENV"
python3 bench_decode.py > evidence/l1-l6-k5-l2/e0-l2a-audit/bench-frozen.out
R=~/projects/data/qwen38-evals/runs/session4/e0-l2a-audit
docker logs qwen38-flash-next-nvfp4 > $R/docker-rank0.log 2>&1; ssh spark2 'docker logs qwen38-flash-next-nvfp4 2>&1' > $R/docker-rank1.log
python3 docker/v030/nccl_twin_audit.py --mode warn head=$R/docker-rank0.log worker=$R/docker-rank1.log
python3 docker/v030/nccl_twin_audit.py --mode strict --graphs 25 --expect-captured 787 head=$R/docker-rank0.log worker=$R/docker-rank1.log
evidence/l1-l6-k5-l2/boot.sh e1-l2a DRAFT_MOE_CONFIG=1 DRAFT_LOCAL_ARGMAX=1 DRAFT_HEAD_FP8=1 "OVERLAYS_V030=$OVL_E" "EXTRA_ENV=VLLM_QWEN38_NCCL_TWIN=1"
evidence/l1-l6-k5-l2/steps.sh e1-l2a free gate ruler-noanchor diverse t1g identity receipts
python3 evidence/l1-l6-k5-l2/soak.py --minutes 20 --out evidence/l1-l6-k5-l2/e1-l2a/soak.jsonl
# -> L2a not kept (c1 within boot noise); soak 93 rounds, 0 failures, 0 NCCL WARN

# --- keep L6a + L1a + L1b': recipe.yaml OVERLAYS_V030 += mtp.py, DRAFT_MOE_CONFIG/DRAFT_LOCAL_ARGMAX/DRAFT_HEAD_FP8=1
python3 kit/render.py && python3 kit/render.py --check && python3 -m unittest discover -s tests -q
VALIDATE_ONLY=1 ./run.sh
IMAGE=vllm/vllm-openai:nightly-aarch64@sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b VALIDATE_ONLY=1 ./run.sh

# --- f-final: the rendered default, full gate pack, left serving
evidence/l1-l6-k5-l2/boot.sh f-final
T3_LENGTHS='4096 16384 32768' evidence/l1-l6-k5-l2/steps.sh f-final free gate ruler-noanchor diverse t1g identity t1nll t2q t3 vision alias receipts
python3 evidence/l1-l6-k5-l2/table.py u2-base a-l6a-l1a b-fp8head c-vocab131k-fp8 c2-vocab164k-fp8 e1-l2a f-final
