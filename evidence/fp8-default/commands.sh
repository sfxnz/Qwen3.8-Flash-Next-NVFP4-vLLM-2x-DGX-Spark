# Session 8 (2026-09-30): FP8_DENSE=per_block (lab-fp8-dense) on top of the K3 default, then the default flip.
# Run from the recipe worktree root (opt/plan-fp8-default). boot.sh = stop, memcheck, ./run.sh with overrides;
# steps.sh = per-boot stages (soak/longid reuse evidence/k3/soak.py); table.py = ruler / frozen / diverse tables;
# quality/ml_judge.py = multilingual capture + blind pairwise judge. Raw data: ~/projects/data/qwen38-evals/runs/session8/<arm>/.
# Not a script to run top to bottom: each block was run by hand, in this order.

# --- step 1: references on the live session-7 default (K3 on, FP8 off), before it was stopped
evidence/fp8-default/steps.sh ref free t1g t1d t1d-s5 gate ruler-noanchor diverse longid
python3 evidence/k-sweep-precision/identity.py ~/projects/data/qwen38-evals/runs/session5/ref/t1g \
  ~/projects/data/qwen38-evals/runs/session8/ref/t1g --json evidence/fp8-default/ref/identity-vs-session5-ref.json
python3 evidence/k-sweep-precision/identity.py ~/projects/data/qwen38-evals/runs/session7/final/t1g \
  ~/projects/data/qwen38-evals/runs/session8/ref/t1g --json evidence/fp8-default/ref/identity-vs-session7-final.json
ANCHOR="52.8 54.8" evidence/fp8-default/steps.sh ref ml ruler-anchor receipts   # 53.8-55.8 voided the gate ruler

# --- step 2 + 3: FP8_DENSE=per_block on the K3 default
evidence/fp8-default/boot.sh fp8 FP8_DENSE=per_block
ANCHOR="0 0" T3_LENGTHS="4096 16384 32768 65536" SOAK_MIN=15 evidence/fp8-default/steps.sh fp8 \
  free t1g identity gate ruler-noanchor t1d longid ml ml-judge t2full t3 vision diverse soak
LONGID_TAG=post evidence/fp8-default/steps.sh fp8 longid receipts

# post-soak long identity failed (near1600): repeat on the idle FP8 serve, then attribute (FP8 without K3) and control (BF16 default)
for t in rep1 rep2; do LONGID_TAG=$t evidence/fp8-default/steps.sh fp8 longid; done
S=evidence/fp8-default   # near1600_reps.py: repeat one longid prompt N times at c=1, report distinct outputs
python3 $S/near1600_reps.py near1600 5 ~/projects/data/qwen38-evals/runs/session8/fp8/near1600-reps.jsonl
evidence/fp8-default/steps.sh fp8 receipts

# --- STOP: the default is NOT flipped. Drafted flip saved as flip-draft-not-applied.patch and reverted.
evidence/fp8-default/boot.sh diag-fp8-nok3 FP8_DENSE=per_block GDN_LAZY=0
python3 $S/near1600_reps.py near1600 5 ~/projects/data/qwen38-evals/runs/session8/diag-fp8-nok3/near1600-reps.jsonl
evidence/fp8-default/boot.sh final                       # the unchanged default (BF16 + K3), left serving
python3 $S/near1600_reps.py near1600 5 ~/projects/data/qwen38-evals/runs/session8/final/near1600-reps.jsonl   # run twice (10 reps)
ANCHOR="52.8 54.8" BASE=ref evidence/fp8-default/steps.sh final free t1g identity gate receipts
python3 evidence/fp8-default/table.py ref fp8 final
