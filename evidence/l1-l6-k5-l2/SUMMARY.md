# Session 4: lossless decode levers L6a, L1a, L1b′, L1b, K5, L2a (2026-09-29/30)

**Result:** the new default is U2 + L1a (local-argmax drafts) + L1b′ (full-vocab FP8 draft head, Marlin W8A16) + L6a (GB10 drafter MoE config). It is serving (`f-final`).
- c=1 ms/step: prose 60.40 → 53.17, structured 61.98 → 54.60.
- Frozen tok/s: prose c1 39.3 → 44.0 (+12%), structured c1 64.1 → 73.3 (+14%).
- Greedy c=1 output is bit-identical to U2, and T1 NLL is exactly equal (Δ 0, top-1 100%, KL 0).

Not kept:
- **L1b** (reduced vocab): CJK acceptance −8.3% at 131072 and −5.2% at 163840, which fails the ≥ −5% gate. 163840 stays available as an opt-in for non-CJK traffic.
- **K5**: missed the microbench gate at M=32, so it was never booted.
- **L2a**: engaged and soaked clean, but its c=1 gain is inside boot-to-boot noise.

Runners are in this directory: `commands.sh`, `boot.sh`, `steps.sh`, `identity.py`, `table.py`, `domain_acc.py`, `soak.py`, `merge_l6a.py`. Raw data: `~/projects/data/qwen38-evals/runs/session4/<arm>/`.

## Arms (one boot each, stacked)

| Arm | Config | Boot | KV tokens |
|---|---|---|---|
| u2-base | live U2 c-final, re-measured | – | 1,655,049 |
| a-l6a-l1a | + DRAFT_MOE_CONFIG (L6a) + mtp.py + use_local_argmax_reduction (L1a) | 11.4 min | 1,585,616 |
| b-fp8head | A + VLLM_QWEN38_DRAFT_HEAD_FP8=1 (Marlin W8A16, 303.6 MiB/rank) | 11.3 | 1,606,871 |
| c-vocab131k-fp8 | B + L1b V′=131072 (160 MiB) | 11.1 | 1,632,377 |
| c2-vocab164k-fp8 | B + L1b V′=163840 (200 MiB) | 12.0 | 1,606,871 |
| e0-l2a-audit | B + L2a, NCCL_DEBUG=INFO SUBSYS=ENV | 11.4 | – |
| e1-l2a | B + L2a, NCCL WARN, + 20-min soak | 11.8 | 1,625,292 |
| **f-final** | rendered default (= B), full gate pack, left serving | 12.0 | 1,588,450 |

## Ruler (ms/step, acceptance in brackets)

From arm B on, the sentinel runs about 7 ms under the U2 window, so those rows are anchor-free re-runs. **New anchor on this base: `--anchor-ms 53.8 55.8`** (f-final: 0/9 excursions, clean median 54.81).

| Arm | prose c1 | prose c2 | prose c8 | struct c1 | struct c2 | struct c8 |
|---|---|---|---|---|---|---|
| u2-base | 60.40 (2.37) | 66.07 (2.15) | 86.56 (2.56) | 61.98 | 62.85 | 81.98 |
| a-l6a-l1a | 60.43 | 66.09 | 86.28 | 61.04 | 62.78 | 79.09 |
| b-fp8head | 53.09 | 60.78 | 88.20* | 54.78 | 58.78 | 75.62 |
| c-vocab131k | 50.99 | 58.49 | 83.05* | 52.43 | 57.00 | 72.70 |
| c2-vocab164k | 51.21 | 60.64 | 79.50* | 52.88 | 57.45 | 73.11 |
| e1-l2a | 52.65 | 60.25 | 79.88* | 54.73 | 57.87 | 73.91 |
| **f-final** | **53.17 (2.37)** | **61.59 (2.15)** | 81.00* | **54.60** | **58.71** | **74.63** |

Structured acceptance is 4.00 in every cell.

\* The prose c8 cell is one prompt × 8 streams, and c=8 greedy is not batch-invariant: a draft change shifts the batch composition, so this cell's text and acceptance drift. The 32-prompt diverse c8 cells show no acceptance change, and c=1 text is identical in every arm.

## Frozen bench_decode (tok/s)

| Arm | prose c1 | prose c2 per-stream / agg | struct c1 | struct c2 per-stream / agg |
|---|---|---|---|---|
| u2-base | 39.3 | 32.4 / 63.9 | 64.1 | 63.2 / 126.4 |
| a | 38.9 | 32.0 / 63.1 | 64.6 | 62.7 / 125.3 |
| b | 44.2 | 34.8 / 68.8 | 72.9 | 68.1 / 136.2 |
| c | 45.9 | 35.2 / 68.0 | 76.7 | 70.5 / 140.9 |
| c2 | 45.6 | 35.3 / 69.8 | 74.9 | 69.5 / 139.0 |
| e1 | 44.5 | 35.2 / 69.7 | 73.1 | 69.0 / 138.0 |
| **f-final** | **44.0** | **34.8 / 69.0** | **73.3** | **67.4 / 134.8** |

## Diverse (ms/step, acceptance)

| Arm | prose c1 | prose c8 | struct c1 | struct c8 |
|---|---|---|---|---|
| u2-base | 60.24 (2.26) | 107.81 (2.27) | 60.53 (3.15) | 113.55 (3.10) |
| b | 53.30 (2.26) | 102.60 (2.27) | 53.73 (3.15) | 107.94 (3.11) |
| c | 50.97 (2.26) | 100.86 (2.26) | 51.43 (2.97) | 103.42 (2.89) |
| **f-final** | 53.58 (2.26) | 103.48 (2.29) | 53.60 (3.15) | 107.67 (3.10) |

## Per-domain c=1 (bench_diverse; tokens/step · tok/s)

| kind (n) | U2 | final (B) | C 131072 | C2 163840 |
|---|---|---|---|---|
| prose (32) | 2.259 · 37.5 | 2.263 · 42.5 | 2.261 · 44.4 | 2.262 · 43.9 |
| code (8) | 3.581 · 59.3 | 3.601 · 66.9 | 3.567 · 69.6 | 3.601 · 69.6 |
| json (8) | 3.753 · 62.1 | 3.752 · 69.6 | 3.752 · 73.0 | 3.752 · 71.9 |
| tool (4) | 5.453 · 62.8 | 5.453 · 70.3 | 5.464 · 72.7 | 5.453 · 73.6 |
| **cjk (8)** | 2.384 · 39.7 | 2.382 · 45.0 | 2.185 · 42.9 | 2.259 · 43.9 |

All 64 completions are text-identical to U2 in every arm.

## Gates

| Arm | T0 | smokes | identity (10) | T1-G | T1 NLL Δ | T2 vs u2-base | verdict |
|---|---|---|---|---|---|---|---|
| a | 0 | 4/4 | identical | 1.0 | – | – | keep |
| b | 0 | 4/4 | identical | 1.0 | – | – | keep |
| c / c2 | 0 | 4/4 | identical | 1.0 | – | – | revert (CJK acceptance) |
| e1 | 0 | 4/4 | identical | 1.0 | – | – | revert (noise) |
| **f-final** | 0 | 4/4 | identical | 1.0 | 0.00000 (top-1 100%, KL 0) | gsm8k 241/250 vs 242 (−2/+1, p 1.0); json 30/30; tools 59/60 (same ambiguous item) | **default** |

On f-final, T3 needles pass 18/18 to 32k, vision 16/16 plus the c=2 pair, and every `reasoning_effort` value returns 200. At the end, spark1 had 14 GiB available and 7.1 GiB swap used; spark2 had 17 GiB and 4.1 GiB.

## Step 0: standalone GPU work (serve down)

- **K5** (`step0-k5/RESULT.md`): GPU unit tests pass 13/13 (residual bitwise). Fused/stock chain at M=4 / M=32 is 0.993 / 1.037 with default tiles, and 0.941 / 0.972 with the best of 14 configs. That is **NO-GO** against the ≤ 0.95× gate. The stock chain is already at 1.21× the DRAM floor, and H3 is slower than cuBLAS's up GEMM.
- **L6a** (`step0-l6a/RESULT.md`): only the M=1 search finished (50.5 min); the other keys are the B200 seed. The kernel is 1.6–2.9% faster at M=1..32, about 12 µs/step at c=1. It rides in arm A, and the log shows it engaged on both ranks.

## Engagement

- **L1a:** "Using local argmax reduction" appears from arm A on.
- **L1b′:** logs `Draft head (L1b'): 124160 rows … FP8 marlin, 303.6 MiB`, with no disable warning, and the drafter's CUDA graphs were captured.
- **L2a audit:** engaged on both ranks; 25 graphs with equal per-graph counts on both ranks. NCCL printed 763 capture lines against 787 counted captured calls, identically on both ranks. This is unexplained and left open.
- **L2a soak:** 20 min, 93 rounds of mixed traffic plus two ~32k prefills. 0 failures, 0 hangs, 0 NCCL WARN.

## Open

- L6a has no standalone A/B.
- The L2a 24-line gap, plus an ABAB 2+2 if L2a is revisited.
- K5: try a wider-J H3 or the W8A16 form.
- A CJK-aware L1b ranking.
