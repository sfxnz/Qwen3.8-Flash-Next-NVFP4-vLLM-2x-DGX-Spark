#!/usr/bin/env bash
# Session 5 (2026-09-30): L7 k sweep, P4-0 BF16 GDN SSM state, P4-1 FP8 dense, on the session-4 default (f-final).
# Run from the recipe worktree root (opt/plan-precision). boot.sh = stop, memcheck, ./run.sh with overrides;
# steps.sh = per-boot stages. Raw data: ~/projects/data/qwen38-evals/runs/session5/<arm>/.
# Not a script to run top to bottom: each block was run by hand, one boot at a time, in this order.
R=~/projects/data/qwen38-evals/runs/session5

# --- ref: the live f-final default (k=3, booted by session 4), the reference for every comparison
BASE=ref evidence/k-sweep-precision/steps.sh ref free receipts gate ruler-noanchor diverse sampled t1g t1full t1d t1d-c8 t2full t3 vision alias
python3 quality/collect_t1.py --out $R/ref/t1-aa                              # T1 A/A: 312,326 tokens, top-1 1.0, KL 0
python3 quality/score.py t1 $R/ref/t1 $R/ref/t1-aa --class B --json evidence/k-sweep-precision/ref/t1-aa.json

# --- L7 k sweep: one boot each; anchor-free (step time changes with k)
evidence/k-sweep-precision/boot.sh k2 NUM_SPECULATIVE_TOKENS=2
ANCHOR="0 0" K=2 evidence/k-sweep-precision/steps.sh k2 free receipts gate ruler-noanchor diverse sampled t1g identity
evidence/k-sweep-precision/boot.sh k4 NUM_SPECULATIVE_TOKENS=4
ANCHOR="0 0" K=4 evidence/k-sweep-precision/steps.sh k4 free receipts gate ruler-noanchor diverse sampled t1g identity

# --- P4-0: BF16 GDN SSM state (v0.30 --mamba-ssm-cache-dtype, config/cache.py:174; the HF config says float32)
evidence/k-sweep-precision/boot.sh p40-ssm-bf16 "EXTRA_ARGS=--mamba-ssm-cache-dtype bfloat16"
ANCHOR="0 0" evidence/k-sweep-precision/steps.sh p40-ssm-bf16 free receipts gate ruler-noanchor diverse t1g identity t1full t1d t2full t3 vision
evidence/k-sweep-precision/steps.sh p40-ssm-bf16 t1long receipts   # 8 x 16k docs sent whole (multi-chunk prefill)

# --- ref2: a second default boot, for the whole-doc T1 reference and a boot-to-boot speed A/A
evidence/k-sweep-precision/boot.sh ref2
BASE=ref ANCHOR="0 0" evidence/k-sweep-precision/steps.sh ref2 free receipts t1long ruler-noanchor diverse
ln -sfn $R/ref2/t1-long $R/ref/t1-long
python3 quality/score.py t1 $R/ref/t1-long $R/p40-ssm-bf16/t1-long --class B --json evidence/k-sweep-precision/p40-ssm-bf16/t1long-vs-ref.json

# --- P4-1: FP8 dense allow-list overlay (docker/v030/FP8_DENSE.md), then the fallback ladder
bash evidence/k-sweep-precision/p41.cmd    # (i) per_block, Marlin W8A16 via --kernel-config linear_backend_per_quant
bash evidence/k-sweep-precision/p41b.cmd   # ladder: ptpc (CUTLASS W8A8, per-token activations)
bash evidence/k-sweep-precision/p41c.cmd   # ladder: per_block without GDN in_proj_qkvz (q/k/v/z rows BF16)
bash evidence/k-sweep-precision/p41d.cmd   # calibration: per_block on PLE kv_proj only (one 65 MB layer)
# -> every FP8 arm fails T1-B (top-1 ~94%, KL ~0.03) whatever the layer count; P4-0 fails T1-B on whole docs.
#    Neither lever passes on its own, so the combined boot (step 4) was not run. Recipe unchanged.

# --- final: the rendered default (unchanged recipe), full gate pack, left serving
evidence/k-sweep-precision/boot.sh final
T3_LENGTHS="4096 16384 32768" evidence/k-sweep-precision/steps.sh final free receipts gate ruler-noanchor diverse sampled t1g identity \
  t1full t1long t1d t2full t3 vision alias receipts
python3 evidence/k-sweep-precision/table.py ref ref2 k2 k4 p40-ssm-bf16 p41-fp8-blk-marlin p41b-fp8-ptpc p41c-fp8-blk-noqkvz final
