# Step 0a: K5 fused HC on one GB10 (serve down), 2026-09-29

- GPU numerics: `tests/test_hc_fused.py` on CUDA, 13/13 OK (`test_hc_fused-gpu.log`): residual bitwise,
  lora / injection / block_input class A, self-test passes and fails closed, dispatch gates.
  No defects found. (The first `-m unittest tests.test_hc_fused` call failed only because `tests/` is
  not a package; use `-m unittest discover -s tests -p test_hc_fused.py`.)
- Every bench M passed its correctness check (residual bitwise, block_input/injection within 2x stock error).
- Gate (restated): fused stream <= 0.95x stock stream at M=4 and M=32 (target ~0.9x).

| cfg | M=4 stream fused/stock | M=32 |
|---|---|---|
| default (split_k 8, block_n 16, block_k 128, block_j 16, block_k_up 64) | 0.993 | 1.037 |
| default, `final` variant (320 rows) | 1.015 | 1.042 |
| block_j 32 | 0.977 | 0.994 |
| block_j 64 | 0.966 | 0.987 |
| block_k_up 128 | 0.959 | 0.993 |
| **block_j 32 + block_k_up 128 (best)** | **0.941** | **0.972** |
| block_j 64 + block_k_up 128 | 0.956 | 1.087 |
| others (block_j 8, block_k_up 32, split_k 4/16, block_n 32, block_k 256, combos) | 0.955–1.212 | 0.986–1.219 |

Per-kernel (default cfg, M=4): stock = combine 2.9 + down GEMM 32.9 + splitK reduce 1.6 + silu 1.1 +
up GEMM 30.4 + gate_mix 1.2 us; fused = H1 2.7 + H2 32.3 + H3 35.2 us. The stock chain is already
~1.21x the DRAM floor (13.4 MB/module); K5 removes ~6 us of glue but H3 (up + gate-mix) is slower than
cuBLAS's up GEMM, so the chain ends up at parity.

**Verdict: NO-GO.** The best tile misses the gate at M=32 (0.972 > 0.95) and only clears it at M=4
(0.941, ≈ 4 us x 106 modules ≈ 0.4 ms/step at c=1, below the ruler's boot-to-boot spread). Arm D
(K5 in serve) was not booted. Next step if revisited: an H3 with a wider J tile per CTA and a
split over streams (the up GEMM is the loss), or the W8A16 form (halves the bytes, ~-31 us/module).
