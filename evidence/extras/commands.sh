#!/usr/bin/env bash
# Session 6 (2026-09-30): X10 throughput profile, P2-5 step 2 (LPT 3200 / MNBT 4832), T4 for lab-fp8-dense.
# Run from the recipe worktree root (opt/plan-extras). boot.sh = stop, memcheck, ./run.sh with overrides;
# steps.sh = per-boot stages; table.py = tables. Raw data: ~/projects/data/qwen38-evals/runs/session6/<arm>/.
# Not a script to run top to bottom: each block was run by hand, one boot at a time, in this order.

# --- ref: the live lossless default (session-5 final boot), reference for every arm
evidence/extras/steps.sh ref free gate diverse16 mixed receipts     # diverse at c=1/8/16 (16 queues behind seqs=8); mixed at LPT 4800
evidence/extras/steps.sh ref t4                                    # T4 base arm: 1,319 items, thinking on, T=0.6, seed+index, --workers 8

# --- X10: throughput profile (THROUGHPUT_PROFILE=1 -> --max-num-seqs 16)
evidence/extras/boot.sh x10-seqs16 THROUGHPUT_PROFILE=1
PROBE_ARGS="--sizes 2 3 5 8 16" evidence/extras/steps.sh x10-seqs16 free gate diverse16 receipts

# --- P2-5 step 2: long-prefill threshold 3200 with 4832 batched tokens
evidence/extras/boot.sh p25-lpt3200 LONG_PREFILL_TOKEN_THRESHOLD=3200 MAX_NUM_BATCHED_TOKENS=4832
evidence/extras/steps.sh p25-lpt3200 free gate mixed receipts

# --- T4 candidate arm: lab-fp8-dense (identical T4 settings)
evidence/extras/boot.sh t4-fp8-dense FP8_DENSE=per_block
evidence/extras/steps.sh t4-fp8-dense free t4 receipts
python3 quality/score.py t4 ~/projects/data/qwen38-evals/runs/session6/ref/t4 ~/projects/data/qwen38-evals/runs/session6/t4-fp8-dense/t4 \
  --json evidence/extras/t4-fp8-dense/t4-vs-ref.json

# --- final: the rendered default, confirmation gate pack, left serving; T4 again as the default A/A floor
evidence/extras/boot.sh final
evidence/extras/steps.sh final free gate t4 receipts
python3 quality/score.py t4 ~/projects/data/qwen38-evals/runs/session6/ref/t4 ~/projects/data/qwen38-evals/runs/session6/final/t4 \
  --json evidence/extras/final/t4-aa.json
python3 evidence/extras/table.py ref x10-seqs16 p25-lpt3200 t4-fp8-dense final
