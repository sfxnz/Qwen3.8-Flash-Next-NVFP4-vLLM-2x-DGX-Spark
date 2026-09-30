#!/usr/bin/env bash
# Session 7 (2026-09-30): K3 GDN MTP decode with lazy state commit (docker/v030/K3.md). Run from the recipe worktree
# root (opt/plan-extras). boot.sh = stop, memcheck, ./run.sh with overrides; steps.sh = per-boot stages;
# harness.sh = standalone single-GPU kernel harness (serve DOWN); soak.py = long-generation identity + soak driver;
# table.py = ruler / frozen / diverse tables. Raw data: ~/projects/data/qwen38-evals/runs/session7/<arm>/.
# Not a script to run top to bottom: each block was run by hand, in this order.

# --- ref: references captured on the live session-6 default BEFORE it was stopped
evidence/k3/steps.sh ref free t1g t1nll t1long longid gate diverse receipts

# --- step 1: harness on spark1, serve down (./stop.sh), one container at a time
./stop.sh
evidence/k3/harness.sh selftest                 # FAILED as built: 1 output element (evidence/k3/harness-pre-fix)
# debug: evidence/k3/debug/dbg.py (mounted as /s; dumps TTGIR) -> tl.reshape(delta) #linear layout; fix = tl.expand_dims (docker/v030/gdn_lazy.py)
evidence/k3/harness.sh selftest exact exact-red sim64 sim1600 bench     # BV=32/4 warps: bitwise, c1 +0.072 ms (gate 4 miss)
#   -> evidence/k3/harness-bv32-nw4 (+ tunables.txt: BV x num_warps sweep through evidence/k3/debug/variant.py)
evidence/k3/harness.sh selftest exact exact-red sim64 sim1600 bench ncu # BV=64/8 warps: exact c8 + sims FAIL (epilogue WAR race)
#   -> evidence/k3/harness-bv64-nw8-prebarrier; fix = debug_barrier before the epilogue store (+ materialize t=0 store)
bash evidence/k3/harness/stress.sh              # exact c1/2/5/8 x 1024 steps and sim64 x 600, seeds 1-3
evidence/k3/harness.sh selftest exact exact-red sim64 sim1600 bench ncu # final: evidence/k3/harness
docker run --rm --network none --memory 8g --ulimit core=1 --entrypoint python3 -e TRITON_INTERPRET=1 -e CUDA_VISIBLE_DEVICES= \
  -v "$PWD":/w -w /w vllm/vllm-openai:v0.30.0-aarch64@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56 \
  -m unittest discover -s tests -p test_gdn_lazy.py   # CPU interpreter tests, 17 OK (evidence/k3/harness/cpu-tests.out)

# --- step 2: serve A/B, K3 via overrides on the session-6 default
evidence/k3/boot.sh k3 "OVERLAYS_V030=docker/v030/flashinfer_cutlass_moe.py docker/v030/gdn_lazy_attn.py docker/v030/qsa_indexer.py docker/v030/serving.py docker/v030/mtp.py docker/v030/gdn_lazy_linear_attn.py" \
  "EXTRA_ENV=VLLM_QWEN38_GDN_LAZY=1"
BASE=ref T3_LENGTHS="4096 16384 32768" evidence/k3/steps.sh k3 free t1g identity t1nll t1long longid gate diverse t3 vision soak
LONGID_TAG=post BASE=ref evidence/k3/steps.sh k3 longid receipts

# --- step 3: GDN_LAZY=1 rendered as the default (recipe.yaml -> kit/render.py), tests, VALIDATE_ONLY, confirmation boot
python3 kit/render.py && python3 kit/render.py --check && python3 -m unittest discover -s tests -q
VALIDATE_ONLY=1 ./run.sh > evidence/k3/validate-only-default.txt
evidence/k3/boot.sh final
BASE=ref T3_LENGTHS="4096 16384 32768" evidence/k3/steps.sh final free t1g identity longid gate diverse t3 vision receipts
python3 evidence/k3/table.py ref k3 final
